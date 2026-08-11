//! Embed text and rank documents against a query.
//!
//! ```text
//! cargo run --release --example embed -- .models/LFM2.5-Embedding-350M
//! ```
//!
//! Doubles as the latency check behind the crate's CPU-first claim.

use std::time::Instant;

use lfm2_encoder::{cosine_similarity, Lfm2Embedding, TextKind};

const DOCS: [&str; 5] = [
    "The borrow checker enforces Rust's ownership and lifetime rules at compile time.",
    "Preheat the oven to 200C and butter a cake tin before adding the batter.",
    "LFM2 is a hybrid architecture interleaving short convolutions with full attention.",
    "Kubernetes schedules containers across a cluster of worker nodes.",
    "Lifetimes let the compiler prove that a reference never outlives its referent.",
];

const QUERY: &str = "how does Rust stop me from using a freed value?";

fn main() -> lfm2_encoder::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: embed <checkpoint-dir>");
        std::process::exit(2);
    });

    let t = Instant::now();
    let model = Lfm2Embedding::from_dir(&dir)?;
    println!("loaded {} ({}-dim) in {:.2?}", dir, model.dim(), t.elapsed());

    // Queries and documents take DIFFERENT prefixes — this model is
    // asymmetric, and using one prefix for both quietly hurts retrieval.
    let t = Instant::now();
    let q = model.embed_normalized(QUERY, TextKind::Query)?;
    let query_ms = t.elapsed();

    let t = Instant::now();
    let embedded: Vec<Vec<f32>> = DOCS
        .iter()
        .map(|d| model.embed_normalized(d, TextKind::Document))
        .collect::<lfm2_encoder::Result<_>>()?;
    let per_doc = t.elapsed() / DOCS.len() as u32;

    let mut ranked: Vec<(f32, &str)> = embedded
        .iter()
        .zip(DOCS)
        .map(|(v, d)| (cosine_similarity(&q, v), d))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!("\nquery: {QUERY}\n");
    for (score, doc) in &ranked {
        println!("  {score:.4}  {doc}");
    }
    println!("\nquery embed: {query_ms:.2?}   mean per document: {per_doc:.2?}");
    Ok(())
}
