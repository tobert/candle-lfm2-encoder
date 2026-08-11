//! Single-vector vs late-interaction, same corpus, same queries.
//!
//! ```text
//! cargo run --release --example compare_retrievers -- \
//!     .models/LFM2.5-Embedding-350M .models/LFM2.5-ColBERT-350M
//! ```
//!
//! Reports retrieval quality AND what each approach costs to store, since
//! that trade is the whole decision: ColBERT keeps one 128-dim vector per
//! token instead of one 1024-dim vector per document.

use std::time::Instant;

use lfm2_encoder::{cosine_similarity, ColbertModel, Lfm2Embedding, MultiVector, TextKind};
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    documents: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Document {
    id: String,
    text: String,
}

#[derive(Deserialize)]
struct Query {
    text: String,
    relevant: Vec<String>,
    hard_negatives: Vec<String>,
}

struct Scores {
    r1: f64,
    r3: f64,
    r5: f64,
    hard_neg: f64,
    index_bytes: usize,
    vectors: usize,
    index_time: std::time::Duration,
    query_time: std::time::Duration,
}

fn report(name: &str, s: &Scores, docs: usize) {
    println!("\n── {name}");
    println!(
        "  recall@1 {:.1}%   recall@3 {:.1}%   recall@5 {:.1}%   hard-neg wins {:.1}%",
        s.r1 * 100.0,
        s.r3 * 100.0,
        s.r5 * 100.0,
        s.hard_neg * 100.0
    );
    println!(
        "  index: {} vectors, {:.2} MiB ({:.0} B/doc)   built in {:.2?} ({:.1?}/doc)",
        s.vectors,
        s.index_bytes as f64 / (1024.0 * 1024.0),
        s.index_bytes as f64 / docs as f64,
        s.index_time,
        s.index_time / docs as u32,
    );
    println!("  mean query latency (encode + score all docs): {:.2?}", s.query_time);
}

/// Rank-based metrics shared by both retrievers.
fn tally(
    corpus: &Corpus,
    ids: &[String],
    mut rank_docs: impl FnMut(&str) -> Vec<usize>,
) -> (f64, f64, f64, f64, std::time::Duration) {
    let (mut h1, mut h3, mut h5) = (0usize, 0usize, 0usize);
    let (mut neg_wins, mut neg_seen) = (0usize, 0usize);
    let t = Instant::now();

    for q in &corpus.queries {
        let order = rank_docs(&q.text);
        let rank_of = |id: &str| -> Option<usize> {
            let d = ids.iter().position(|x| x == id)?;
            order.iter().position(|i| *i == d)
        };
        let Some(best) = q.relevant.iter().filter_map(|id| rank_of(id)).min() else {
            continue;
        };
        h1 += usize::from(best == 0);
        h3 += usize::from(best < 3);
        h5 += usize::from(best < 5);
        for neg in &q.hard_negatives {
            if let Some(nr) = rank_of(neg) {
                neg_seen += 1;
                neg_wins += usize::from(nr < best);
            }
        }
    }
    let elapsed = t.elapsed() / corpus.queries.len() as u32;
    let n = corpus.queries.len() as f64;
    (
        h1 as f64 / n,
        h3 as f64 / n,
        h5 as f64 / n,
        neg_wins as f64 / neg_seen.max(1) as f64,
        elapsed,
    )
}

fn main() -> lfm2_encoder::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: compare_retrievers <embedding-dir> <colbert-dir>");
        std::process::exit(2);
    }

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/semantic_search_eval.json");
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let ids: Vec<String> = corpus.documents.iter().map(|d| d.id.clone()).collect();
    println!("{} documents, {} queries", corpus.documents.len(), corpus.queries.len());

    // ── single-vector
    let emb = Lfm2Embedding::from_dir(&args[0])?;
    // Warm before timing: the first model touched pays for faulting in
    // ~1.4 GiB of mmap'd weights, which otherwise lands entirely on
    // whichever retriever happens to run first and makes it look 2x slower
    // than it is.
    let _ = emb.embed("warm", TextKind::Document)?;
    let t = Instant::now();
    let doc_vecs: Vec<Vec<f32>> = corpus
        .documents
        .iter()
        .map(|d| emb.embed_normalized(&d.text, TextKind::Document))
        .collect::<lfm2_encoder::Result<_>>()?;
    let emb_index_time = t.elapsed();

    let (r1, r3, r5, hn, qt) = tally(&corpus, &ids, |q| {
        let qv = emb.embed_normalized(q, TextKind::Query).unwrap();
        let mut scored: Vec<(f32, usize)> = doc_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (cosine_similarity(&qv, v), i))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().map(|(_, i)| i).collect()
    });
    let emb_scores = Scores {
        r1,
        r3,
        r5,
        hard_neg: hn,
        vectors: doc_vecs.len(),
        index_bytes: doc_vecs.iter().map(|v| v.len() * 4).sum(),
        index_time: emb_index_time,
        query_time: qt,
    };

    // ── late interaction
    let colbert = ColbertModel::from_dir(&args[1])?;
    let _ = colbert.encode_document("warm")?;
    let t = Instant::now();
    let doc_mvs: Vec<MultiVector> = corpus
        .documents
        .iter()
        .map(|d| colbert.encode_document(&d.text))
        .collect::<lfm2_encoder::Result<_>>()?;
    let cb_index_time = t.elapsed();

    let (r1, r3, r5, hn, qt) = tally(&corpus, &ids, |q| {
        let qv = colbert.encode_query(q).unwrap();
        let mut scored: Vec<(f32, usize)> = doc_mvs
            .iter()
            .enumerate()
            .map(|(i, d)| (qv.max_sim(d), i))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().map(|(_, i)| i).collect()
    });
    let cb_scores = Scores {
        r1,
        r3,
        r5,
        hard_neg: hn,
        vectors: doc_mvs.iter().map(|m| m.len()).sum(),
        index_bytes: doc_mvs
            .iter()
            .map(|m| m.len() * colbert.dim() * 4)
            .sum(),
        index_time: cb_index_time,
        query_time: qt,
    };

    let n = corpus.documents.len();
    report("single-vector (Embedding-350M, CLS + cosine)", &emb_scores, n);
    report("late interaction (ColBERT-350M, MaxSim)", &cb_scores, n);

    println!(
        "\nColBERT stores {:.1}x the bytes and scores {:.1}x slower per query.",
        cb_scores.index_bytes as f64 / emb_scores.index_bytes as f64,
        cb_scores.query_time.as_secs_f64() / emb_scores.query_time.as_secs_f64(),
    );
    Ok(())
}
