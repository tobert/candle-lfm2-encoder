//! What does `from_trunk` actually save, in RSS?
//!
//! Loads the SAME sequence-classification checkpoint twice two different
//! ways: once as two independent `from_dir` calls (today's only option
//! before this feature), once as one `Lfm2Trunk::load_shared` plus two
//! `from_trunk` calls — and reports the resident-memory delta between them.
//! Companion to `examples/bench_memory.rs`, which answers the same question
//! for independently-loaded embedding models.
//!
//! ```text
//! cargo run --release --example bench_shared_trunk -- <sequence-classifier-checkpoint-dir>
//! ```

use std::sync::Arc;
use std::time::Instant;

use candle_lfm2_encoder::{Lfm2SequenceClassifier, Lfm2Trunk};

/// Resident set size in MiB, from /proc/self/status.
fn rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn main() -> candle_lfm2_encoder::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: bench_shared_trunk <sequence-classifier-checkpoint-dir>");
        std::process::exit(2);
    });

    let baseline = rss_mib();
    println!("baseline RSS {baseline:.0} MiB\n");

    // Two independent trunk loads — what every head's from_dir does today.
    let t = Instant::now();
    let a = Lfm2SequenceClassifier::from_dir(&dir)?;
    let b = Lfm2SequenceClassifier::from_dir(&dir)?;
    let independent_time = t.elapsed();
    let independent_rss = rss_mib() - baseline;
    println!(
        "two independent from_dir() loads:  {independent_rss:>8.0} MiB   ({independent_time:.2?})"
    );
    drop(a);
    drop(b);

    // Force the two dropped classifiers' pages back out before measuring
    // the shared path, so the two numbers aren't polluted by the
    // allocator holding freed-but-not-returned pages from the first half.
    let after_drop = rss_mib();
    println!("  (after drop: {:.0} MiB over baseline)\n", after_drop - baseline);

    let t = Instant::now();
    let trunk = Lfm2Trunk::load_shared(&dir)?;
    let sa = Lfm2SequenceClassifier::from_trunk(Arc::clone(&trunk), &dir)?;
    let sb = Lfm2SequenceClassifier::from_trunk(Arc::clone(&trunk), &dir)?;
    let shared_time = t.elapsed();
    let shared_rss = rss_mib() - after_drop;
    println!("load_shared() + two from_trunk():  {shared_rss:>8.0} MiB   ({shared_time:.2?})");

    println!(
        "\nsharing saved {:.0} MiB ({:.1}x) for a second head over the same trunk",
        independent_rss - shared_rss,
        independent_rss / shared_rss.max(1.0)
    );

    std::hint::black_box((&sa, &sb));
    Ok(())
}
