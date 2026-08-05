//! Classify command risk with a fine-tuned sequence-classification head.
//!
//! ```text
//! cargo run --release --example classify -- <checkpoint-dir> [text ...]
//! echo "kubectl delete ns staging" | cargo run --release --example classify -- <checkpoint-dir>
//! ```
//!
//! With no text arguments, reads one input per line from stdin — the same
//! whole-statement shape a kaish ledger entry has. Doubles as the smoke
//! test that the Python trainer's export contract loads in candle.

use std::io::BufRead;
use std::time::Instant;

use candle_lfm2_encoder::Lfm2SequenceClassifier;

const DEMO: [&str; 8] = [
    "kubectl get pods -n payments -o wide",
    "kubectl rollout restart deployment/checkout -n prod",
    "kubectl delete namespace staging --wait=false",
    "kubectl delete pod api-7f9d --dry-run=client",
    "echo 'never run: kubectl delete namespace kube-system'",
    "kubectl exec -it redis-0 -- redis-cli FLUSHALL",
    "can you bounce the payments deployment in staging?",
    "clean out everything in the loadtest namespace after standup",
];

fn main() -> candle_lfm2_encoder::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| {
        eprintln!("usage: classify <checkpoint-dir> [text ...]");
        std::process::exit(2);
    });
    let inputs: Vec<String> = args.collect();

    let t = Instant::now();
    let model = Lfm2SequenceClassifier::from_dir(&dir)?;
    println!(
        "loaded {} ({} labels: {}) in {:.2?}",
        dir,
        model.num_labels(),
        model.labels().join(", "),
        t.elapsed()
    );

    let texts: Vec<String> = if !inputs.is_empty() {
        inputs
    } else if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        DEMO.iter().map(|s| s.to_string()).collect()
    } else {
        std::io::stdin()
            .lock()
            .lines()
            .collect::<std::io::Result<Vec<_>>>()
            .unwrap_or_else(|e| {
                eprintln!("reading stdin: {e}");
                std::process::exit(1);
            })
    };

    let mut total = std::time::Duration::ZERO;
    for text in &texts {
        let t = Instant::now();
        let probs = model.predict(text)?;
        total += t.elapsed();

        let mut ranked: Vec<(&str, f32)> = model
            .labels()
            .iter()
            .map(String::as_str)
            .zip(probs)
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let dist = ranked
            .iter()
            .map(|(label, p)| format!("{label} {p:.3}"))
            .collect::<Vec<_>>()
            .join("  ");
        println!("\n  {:>12}  {text}", ranked[0].0);
        println!("  {:>12}  [{dist}]", "");
    }
    println!(
        "\n{} inputs, mean {:.2?} per classification",
        texts.len(),
        total / texts.len().max(1) as u32
    );
    Ok(())
}
