//! Zero-shot prompt routing with `LFM2.5-Encoder-350M-Prompt-Router`.
//!
//! ```text
//! cargo run --release --example route -- <ckpt-dir> \
//!     --route "Coding" --route "Sales" --route "Creative writing" \
//!     "Can you help me debug a failing Python unit test?"
//!
//! # routes from a JSON array file instead of repeated --route
//! cargo run --release --example route -- <ckpt-dir> \
//!     --routes-file routes.json "some prompt"
//!
//! # batch mode: one line per prompt, an accuracy summary if "expect" is set
//! cargo run --release --example route -- <ckpt-dir> \
//!     --routes-file routes.json --prompts-file prompts.json
//!
//! # raw pre-softmax scores instead of probabilities
//! cargo run --release --example route -- <ckpt-dir> --scores \
//!     --routes-file routes.json --prompts-file prompts.json
//! ```
//!
//! `routes.json` is a JSON array of strings. `prompts.json` is a JSON
//! array of `{"text": "...", "expect": "..."}`, where `expect` (a route
//! name) is optional — prompts without it are scored but excluded from the
//! accuracy summary.
//!
//! # Read this before trusting the `prob` column
//!
//! This checkpoint's softmax is measured to be SATURATED (see
//! `candle_lfm2_encoder::routing`'s module docs): for a clear-cut match,
//! the winning probability is close to a pure function of how many routes
//! you passed in, not of how confident the match actually is. This example
//! always prints `score` (the pre-softmax scaled logit) alongside `prob`
//! for exactly that reason — if you're building a threshold/confidence
//! decision on top of this, use `score` (or run with `--scores`), not
//! `prob`.

use std::time::Instant;

use candle_lfm2_encoder::Lfm2SequenceRouter;
use serde::Deserialize;

#[derive(Deserialize)]
struct PromptCase {
    text: String,
    expect: Option<String>,
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: route <checkpoint-dir> [--route TEXT]... [--routes-file routes.json] \
         [--prompts-file prompts.json] [--threshold F] [--scores] [prompt text]"
    );
    std::process::exit(2);
}

fn main() -> candle_lfm2_encoder::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage_and_exit();
    }
    let dir = args[0].clone();

    let mut routes: Vec<String> = Vec::new();
    let mut routes_file: Option<String> = None;
    let mut prompts_file: Option<String> = None;
    let mut threshold: Option<f32> = None;
    let mut show_scores = false;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--route" => {
                let v = args.get(i + 1).unwrap_or_else(|| {
                    eprintln!("--route needs a value");
                    std::process::exit(2);
                });
                routes.push(v.clone());
                i += 2;
            }
            "--routes-file" => {
                routes_file = Some(args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("--routes-file needs a path");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--prompts-file" => {
                prompts_file = Some(args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("--prompts-file needs a path");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--threshold" => {
                let v = args.get(i + 1).unwrap_or_else(|| {
                    eprintln!("--threshold needs a value");
                    std::process::exit(2);
                });
                threshold = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("--threshold value must be a float, got {v:?}");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--scores" => {
                show_scores = true;
                i += 1;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    if let Some(path) = &routes_file {
        let bytes = std::fs::read(path).unwrap_or_else(|e| {
            eprintln!("reading {path}: {e}");
            std::process::exit(1);
        });
        let extra: Vec<String> = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            eprintln!("parsing {path} as a JSON array of strings: {e}");
            std::process::exit(1);
        });
        routes.extend(extra);
    }
    if routes.is_empty() {
        eprintln!("no routes given — pass --route (repeatable) and/or --routes-file");
        std::process::exit(2);
    }

    let t = Instant::now();
    let model = Lfm2SequenceRouter::from_dir(&dir)?;
    println!(
        "loaded {dir} ({} routes, proj_dim={}) in {:.2?}",
        routes.len(),
        model.proj_dim(),
        t.elapsed()
    );
    println!("routes: {}", routes.join(" | "));

    match prompts_file {
        Some(path) => run_batch(&model, &path, &routes, show_scores),
        None => {
            if positional.is_empty() {
                eprintln!("no prompt text given (and no --prompts-file)");
                std::process::exit(2);
            }
            let text = positional.join(" ");
            run_single(&model, &text, &routes, show_scores, threshold)?;
        }
    }
    Ok(())
}

fn print_ranked(model: &Lfm2SequenceRouter, text: &str, routes: &[String], show_scores: bool) -> candle_lfm2_encoder::Result<()> {
    let scores = model.route_scores(text, routes)?;
    let probs = model.route_probs(text, routes)?;
    let mut ranked: Vec<(usize, f32, f32)> = (0..routes.len())
        .map(|i| (i, scores[i], probs[i]))
        .collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));

    if show_scores {
        println!("  {:>10}  route", "score");
        for (i, score, _) in &ranked {
            println!("  {:>10.4}  {}", score, routes[*i]);
        }
    } else {
        println!("  {:>7}  {:>10}  route", "prob", "score");
        for (i, score, prob) in &ranked {
            println!("  {:>7.4}  {:>10.4}  {}", prob, score, routes[*i]);
        }
    }
    Ok(())
}

fn run_single(
    model: &Lfm2SequenceRouter,
    text: &str,
    routes: &[String],
    show_scores: bool,
    threshold: Option<f32>,
) -> candle_lfm2_encoder::Result<()> {
    println!("\nprompt: {text}\n");
    print_ranked(model, text, routes, show_scores)?;

    if let Some(t) = threshold {
        let matches = model.route(text, routes, Some(t))?;
        println!(
            "\n  routes with score >= {t}: {}",
            matches
                .iter()
                .map(|m| format!("{} ({:.4})", m.route, m.score))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

fn run_batch(model: &Lfm2SequenceRouter, path: &str, routes: &[String], show_scores: bool) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("reading {path}: {e}");
        std::process::exit(1);
    });
    let cases: Vec<PromptCase> = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        eprintln!("parsing {path} as a JSON array of {{text, expect?}}: {e}");
        std::process::exit(1);
    });

    let mut correct = 0usize;
    let mut checked = 0usize;
    // (expected, predicted) -> count, for mismatches only.
    let mut confusion: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();

    println!("\n{} prompts:\n", cases.len());
    let t = Instant::now();
    for case in &cases {
        let probs = match model.route_probs(&case.text, routes) {
            Ok(p) => p,
            Err(e) => {
                println!("  ERROR  {:?}: {e}", case.text);
                continue;
            }
        };
        let scores = if show_scores {
            model.route_scores(&case.text, routes).ok()
        } else {
            None
        };

        let mut ranked: Vec<(usize, f32)> = (0..routes.len()).map(|i| (i, probs[i])).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let (top_idx, top_prob) = ranked[0];
        let runner_up = ranked.get(1).map(|&(i, p)| (routes[i].as_str(), p));

        let margin_str = match runner_up {
            Some((route, p)) => format!("(runner-up: {route} {p:.4}, margin {:.4})", top_prob - p),
            None => String::new(),
        };
        let score_str = match &scores {
            Some(s) => format!(" score={:.4}", s[top_idx]),
            None => String::new(),
        };
        let text_preview: String = case.text.chars().take(72).collect();

        let mark = match &case.expect {
            Some(expect) => {
                checked += 1;
                let ok = expect == &routes[top_idx];
                if ok {
                    correct += 1;
                } else {
                    *confusion
                        .entry((expect.clone(), routes[top_idx].clone()))
                        .or_insert(0) += 1;
                }
                if ok { "OK  " } else { "MISS" }
            }
            None => "    ",
        };

        println!(
            "  {mark}  {:>7.4}{score_str}  {:<24} {margin_str}  {text_preview:?}",
            top_prob, routes[top_idx]
        );
    }
    let elapsed = t.elapsed();

    println!(
        "\n{} prompts in {:.2?} ({:.1?}/prompt)",
        cases.len(),
        elapsed,
        elapsed / cases.len().max(1) as u32
    );

    if checked > 0 {
        println!(
            "\naccuracy: {correct}/{checked} ({:.1}%)",
            100.0 * correct as f64 / checked as f64
        );
        if !confusion.is_empty() {
            println!("\nconfusion (expected -> predicted, count):");
            let mut rows: Vec<(&(String, String), &usize)> = confusion.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1));
            for ((expect, got), n) in rows {
                println!("  {expect} -> {got}: {n}");
            }
        }
    }
}
