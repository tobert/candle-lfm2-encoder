//! What does it actually cost to keep these models resident?
//!
//! Answers the question a consumer has to answer before embedding this in a
//! daemon: how much RAM per model, at which dtype, and what does the
//! cheaper dtype cost in latency.
//!
//! ```text
//! cargo run --release --example bench_memory -- .models/*
//! cargo run --release --example bench_memory -- --dtype bf16 .models/*
//! ```

use std::time::Instant;

use candle_core::{DType, Device};
use candle_lfm2_encoder::{Lfm2Embedding, TextKind};

/// Resident set size in MiB, from /proc/self/status.
fn rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

const PROBE: &str = "how does the borrow checker prevent use-after-free in Rust?";

fn main() -> candle_lfm2_encoder::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let mut dtype = DType::F32;
    if let Some(i) = args.iter().position(|a| a == "--dtype") {
        let name = args.get(i + 1).cloned().unwrap_or_default();
        dtype = match name.as_str() {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            "f16" => DType::F16,
            other => {
                eprintln!("unknown dtype {other:?}; use f32 | bf16 | f16");
                std::process::exit(2);
            }
        };
        args.drain(i..=i + 1);
    }

    let dirs: Vec<String> = args
        .into_iter()
        .filter(|d| std::path::Path::new(d).join("model.safetensors").is_file())
        .collect();
    if dirs.is_empty() {
        eprintln!("usage: bench_memory [--dtype f32|bf16|f16] <checkpoint-dir>...");
        std::process::exit(2);
    }

    let baseline = rss_mib();
    println!("dtype {dtype:?}, baseline RSS {baseline:.0} MiB\n");
    println!(
        "{:<38} {:>10} {:>10} {:>12}",
        "checkpoint", "ΔRSS MiB", "load", "embed"
    );

    // Held so nothing is dropped — we want the cost of keeping them ALL
    // resident at once, which is the actual deployment question.
    let mut resident = Vec::new();
    let mut prev = baseline;

    for dir in &dirs {
        let name = std::path::Path::new(dir)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.clone());

        let t = Instant::now();
        let model = Lfm2Embedding::from_dir_with(dir, dtype, &Device::Cpu)?;
        let load = t.elapsed();

        // Warm once (first pass touches mmap pages), then time.
        let _ = model.embed(PROBE, TextKind::Query)?;
        let t = Instant::now();
        let _ = model.embed(PROBE, TextKind::Query)?;
        let embed = t.elapsed();

        let now = rss_mib();
        println!(
            "{:<38} {:>10.0} {:>10.2?} {:>12.2?}",
            name,
            now - prev,
            load,
            embed
        );
        prev = now;
        resident.push(model);
    }

    println!(
        "\n{} models resident: {:.2} GiB total RSS ({:.0} MiB over baseline)",
        resident.len(),
        rss_mib() / 1024.0,
        rss_mib() - baseline
    );
    Ok(())
}
