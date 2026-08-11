//! Does halving memory cost you retrieval quality?
//!
//! Loads one checkpoint at f32 and again at f16, embeds the same texts, and
//! reports how far apart the vectors land. The number that matters for a
//! retrieval consumer is the cosine between the two — absolute drift is
//! meaningless once you normalize, but a rank flip is not.
//!
//! ```text
//! cargo run --release --example dtype_drift -- .models/LFM2.5-Embedding-350M
//! ```

use candle_core::{DType, Device};
use lfm2_encoder::{cosine_similarity, Lfm2Embedding, TextKind};

const TEXTS: [&str; 8] = [
    "how does the borrow checker prevent use-after-free in Rust?",
    "The borrow checker enforces ownership and lifetime rules at compile time.",
    "Lifetimes let the compiler prove a reference never outlives its referent.",
    "Preheat the oven to 200C and butter a cake tin.",
    "Kubernetes schedules containers across a cluster of worker nodes.",
    "LFM2 interleaves short convolutions with full attention.",
    "日本語（にほんご）を勉強中です。",
    "vector search 🔍 with a 350M encoder",
];

fn main() -> lfm2_encoder::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dtype_drift <checkpoint-dir>");
        std::process::exit(2);
    });

    let f32_model = Lfm2Embedding::from_dir_with(&dir, DType::F32, &Device::Cpu)?;
    let f16_model = Lfm2Embedding::from_dir_with(&dir, DType::F16, &Device::Cpu)?;

    println!("{:<62} {:>10} {:>10}", "text", "cos(f32,f16)", "max|Δ|");
    let mut worst_cos = 1.0f32;
    let mut f32_vecs = Vec::new();
    let mut f16_vecs = Vec::new();

    for text in TEXTS {
        let a = f32_model.embed_normalized(text, TextKind::Document)?;
        let b = f16_model.embed_normalized(text, TextKind::Document)?;
        let cos = cosine_similarity(&a, &b);
        let max_d = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let shown: String = text.chars().take(58).collect();
        println!("{shown:<62} {cos:>12.6} {max_d:>10.2e}");
        worst_cos = worst_cos.min(cos);
        f32_vecs.push(a);
        f16_vecs.push(b);
    }

    // The consumer-facing question: does f16 reorder results? Rank the
    // documents against the query under each dtype and compare orderings.
    let query_f32 = &f32_vecs[0];
    let query_f16 = &f16_vecs[0];
    let mut rank_f32: Vec<usize> = (1..TEXTS.len()).collect();
    let mut rank_f16 = rank_f32.clone();
    rank_f32.sort_by(|&a, &b| {
        cosine_similarity(query_f32, &f32_vecs[b])
            .total_cmp(&cosine_similarity(query_f32, &f32_vecs[a]))
    });
    rank_f16.sort_by(|&a, &b| {
        cosine_similarity(query_f16, &f16_vecs[b])
            .total_cmp(&cosine_similarity(query_f16, &f16_vecs[a]))
    });

    println!("\nworst cos(f32, f16) = {worst_cos:.6}");
    println!("ranking f32: {rank_f32:?}");
    println!("ranking f16: {rank_f16:?}");
    println!(
        "rankings {}",
        if rank_f32 == rank_f16 {
            "IDENTICAL — f16 is a safe memory/latency trade for retrieval"
        } else {
            "DIFFER — f16 reorders results; check before trusting it"
        }
    );
    Ok(())
}
