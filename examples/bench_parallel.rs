//! Can two of these share a daemon?
//!
//! Runs each checkpoint solo, then all of them concurrently on the same
//! box, and reports what concurrency costs. Cheap by design: wallclock plus
//! CPU-seconds from /proc/self/stat, no profiler.
//!
//! ```text
//! cargo run --release --example bench_parallel -- .models/*
//! cargo run --release --example bench_parallel -- --dtype f16 .models/*
//! ```
//!
//! Note: every checkpoint is driven through the embedding path, because the
//! task heads aren't built yet. That is fine for a CONTENTION measurement —
//! the trunk is ~99% of the compute and identical across the family — but
//! the PII checkpoint's pooled output is meaningless here. This measures
//! throughput, not quality.

use std::time::{Duration, Instant};

use candle_core::{DType, Device};
use candle_lfm2_encoder::{Lfm2Embedding, TextKind};

const WORKLOAD: [&str; 8] = [
    "how does the borrow checker prevent use-after-free",
    "Arc<Mutex<T>> shares mutable state across threads safely.",
    "connection string with an embedded password in a log line",
    "kubernetes pod stuck in CrashLoopBackOff after a config change",
    "日本語のテキストをベクトルに変換する",
    "git rebase interactive squash the last three commits",
    "the encoder interleaves short convolutions with full attention",
    "rotate the API key and invalidate the old one",
];

/// CPU seconds this process has burned (user + system), from /proc.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the (possibly space-containing) comm field.
    let Some(rest) = stat.rsplit_once(would_be_comm_end()).map(|(_, r)| r) else {
        return 0.0;
    };
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = std::env::var("CLK_TCK")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(100.0);
    // utime is field 14 overall = index 11 after comm; stime the next.
    let utime: f64 = f.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = f.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (utime + stime) / ticks
}

fn would_be_comm_end() -> &'static str {
    ") "
}

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|r| r.split_whitespace().next()?.parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// One pass over the workload; returns mean per-embed latency.
fn run_workload(model: &Lfm2Embedding, passes: usize) -> Duration {
    let t = Instant::now();
    for _ in 0..passes {
        for text in WORKLOAD {
            let _ = model.embed(text, TextKind::Query).expect("embed");
        }
    }
    t.elapsed() / (passes * WORKLOAD.len()) as u32
}

fn main() -> candle_lfm2_encoder::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut dtype = DType::F32;
    if let Some(i) = args.iter().position(|a| a == "--dtype") {
        if args.get(i + 1).map(String::as_str) == Some("f16") {
            dtype = DType::F16;
        }
        args.drain(i..=i + 1);
    }
    let dirs: Vec<String> = args
        .into_iter()
        .filter(|d| std::path::Path::new(d).join("model.safetensors").is_file())
        .collect();
    if dirs.is_empty() {
        eprintln!("usage: bench_parallel [--dtype f16] <checkpoint-dir>...");
        std::process::exit(2);
    }

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("{cores} logical cores, dtype {dtype:?}\n");

    let models: Vec<(String, Lfm2Embedding)> = dirs
        .iter()
        .map(|d| {
            let name = std::path::Path::new(d)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| d.clone());
            Ok((name, Lfm2Embedding::from_dir_with(d, dtype, &Device::Cpu)?))
        })
        .collect::<candle_lfm2_encoder::Result<_>>()?;

    println!("{} models resident: {:.2} GiB RSS\n", models.len(), rss_mib() / 1024.0);

    // Warm every model so page faults don't land in the measurement.
    for (_, m) in &models {
        let _ = m.embed(WORKLOAD[0], TextKind::Query)?;
    }

    const PASSES: usize = 3;
    println!("{:<38} {:>12} {:>12} {:>10}", "checkpoint", "solo", "concurrent", "slowdown");

    let mut solo = Vec::new();
    for (name, model) in &models {
        solo.push((name.clone(), run_workload(model, PASSES)));
    }

    let wall = Instant::now();
    let cpu_before = cpu_seconds();
    let concurrent: Vec<Duration> = std::thread::scope(|scope| {
        let handles: Vec<_> = models
            .iter()
            .map(|(_, model)| scope.spawn(move || run_workload(model, PASSES)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = wall.elapsed();
    let cpu = cpu_seconds() - cpu_before;

    for ((name, s), c) in solo.iter().zip(&concurrent) {
        let ratio = c.as_secs_f64() / s.as_secs_f64();
        println!("{name:<38} {s:>12.2?} {c:>12.2?} {ratio:>9.2}x");
    }

    let embeds = models.len() * PASSES * WORKLOAD.len();
    println!(
        "\nconcurrent phase: {:.2?} wallclock, {cpu:.2}s CPU → {:.1} cores busy",
        wall,
        cpu / wall.as_secs_f64()
    );
    println!(
        "throughput: {:.1} embeds/s across {} models ({:.1}/s per model)",
        embeds as f64 / wall.as_secs_f64(),
        models.len(),
        embeds as f64 / wall.as_secs_f64() / models.len() as f64
    );
    Ok(())
}
