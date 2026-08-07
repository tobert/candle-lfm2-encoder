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

/// One decomposed statement: the whole thing, and the clauses a real parser
/// says it contains. `clauses` comes from kaish `Plan.commands[]` — this
/// example does not split text itself, and neither does the crate (see
/// `Lfm2SequenceRouter::route_clauses`).
#[derive(Deserialize)]
struct ClauseCase {
    statement: String,
    expect: Option<String>,
    clauses: Vec<String>,
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: route <checkpoint-dir> [--route TEXT]... [--routes-file routes.json] \
         [--prompts-file prompts.json] [--clauses-file clauses.json] [--threshold F] \
         [--scores] [--all-scores] [prompt text]"
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
    let mut clauses_file: Option<String> = None;
    let mut threshold: Option<f32> = None;
    let mut show_scores = false;
    let mut all_scores = false;
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
            "--clauses-file" => {
                clauses_file = Some(args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("--clauses-file needs a path");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--scores" => {
                show_scores = true;
                i += 1;
            }
            "--all-scores" => {
                all_scores = true;
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

    match (clauses_file, prompts_file) {
        (Some(path), _) => run_clauses(&model, &path, &routes, all_scores, threshold),
        (None, Some(path)) => run_batch(&model, &path, &routes, show_scores, all_scores),
        (None, None) => {
            if positional.is_empty() {
                eprintln!("no prompt text given (and no --prompts-file/--clauses-file)");
                std::process::exit(2);
            }
            let text = positional.join(" ");
            run_single(&model, &text, &routes, show_scores, threshold)?;
        }
    }
    Ok(())
}

/// Route each statement two ways — whole, and clause by clause — and print
/// both, so the comparison the fan-out analysis needs is one run.
///
/// Cosines throughout, never probabilities: this head's softmax is a pure
/// function of route count (see the crate's `routing` module docs), so it
/// carries no confidence at all, and across clauses it is worse than
/// meaningless — every clause renormalizes over the same lanes, so a clause
/// with no intent still hands its best lane a large share.
fn run_clauses(
    model: &Lfm2SequenceRouter,
    path: &str,
    routes: &[String],
    all_scores: bool,
    threshold: Option<f32>,
) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("reading {path}: {e}");
        std::process::exit(1);
    });
    let cases: Vec<ClauseCase> = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        eprintln!("parsing {path} as a JSON array of {{statement, expect?, clauses}}: {e}");
        std::process::exit(1);
    });

    let mut joint_correct = 0usize;
    let mut union_correct = 0usize;
    let mut checked = 0usize;
    // Statements where decomposition changed the answer, either way.
    let mut fixed: Vec<&str> = Vec::new();
    let mut broken: Vec<&str> = Vec::new();
    // Top-1 is the wrong question for a statement with two real intents —
    // one of them must lose an argmax no matter how well both were
    // detected. What a fan-out cascade actually needs is RECALL: did the
    // expected lane clear a threshold at all? Collected as (joint, union)
    // cosines on the expected lane so any threshold can be swept after.
    let mut expected_cosines: Vec<(f32, f32)> = Vec::new();

    println!("\n{} statements, clause-decomposed:\n", cases.len());
    let t = Instant::now();

    for case in &cases {
        let joint = match model.route_cosines(&case.statement, routes) {
            Ok(c) => c,
            Err(e) => {
                println!("  ERROR (joint) {:?}: {e}", case.statement);
                continue;
            }
        };
        let split = match model.route_clauses(&case.clauses, routes) {
            Ok(s) => s,
            Err(e) => {
                println!("  ERROR (clauses) {:?}: {e}", case.statement);
                continue;
            }
        };

        let joint_top = argmax(&joint);
        let union_top = argmax(&split.union_cosines);

        let mark = match &case.expect {
            Some(expect) => {
                checked += 1;
                if let Some(lane) = routes.iter().position(|r| r == expect) {
                    expected_cosines.push((joint[lane], split.union_cosines[lane]));
                } else {
                    eprintln!(
                        "expect {expect:?} is not one of the routes — recall cannot be \
                         measured for {:?}",
                        case.statement
                    );
                    std::process::exit(2);
                }
                let j = expect == &routes[joint_top];
                let u = expect == &routes[union_top];
                if j {
                    joint_correct += 1;
                }
                if u {
                    union_correct += 1;
                }
                match (j, u) {
                    (false, true) => {
                        fixed.push(&case.statement);
                        "FIXED "
                    }
                    (true, false) => {
                        broken.push(&case.statement);
                        "BROKE "
                    }
                    (true, true) => "both  ",
                    (false, false) => "neither",
                }
            }
            None => "      ",
        };

        println!("  {mark} {:?}", case.statement);
        if let (Some(&(j, u)), Some(expect)) = (expected_cosines.last(), case.expect.as_deref()) {
            println!("         expected lane: joint {j:>7.4} → union {u:>7.4}   ({expect})");
        }
        println!(
            "         joint  {:>7.4}  {}",
            joint[joint_top], routes[joint_top]
        );
        let fired = split.firing_clause(union_top).unwrap_or(0);
        println!(
            "         union  {:>7.4}  {}   (from clause {fired}: {:?})",
            split.union_cosines[union_top], routes[union_top], case.clauses[fired]
        );
        if let Some(t) = threshold {
            let lanes = split.lanes_above(t);
            println!(
                "         lanes >= {t}: {}",
                lanes
                    .iter()
                    .map(|&l| format!("{} ({:.4})", routes[l], split.union_cosines[l]))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        for (ci, clause) in case.clauses.iter().enumerate() {
            let row = &split.clause_cosines[ci];
            let top = argmax(row);
            if all_scores {
                println!("           [{ci}] {clause:?}");
                for (r, route) in routes.iter().enumerate() {
                    println!("               {:>7.4}  {route}", row[r]);
                }
            } else {
                println!(
                    "           [{ci}] {:>7.4}  {:<24} {clause:?}",
                    row[top], routes[top]
                );
            }
        }
        println!();
    }

    let elapsed = t.elapsed();
    println!(
        "{} statements in {:.2?} ({:.1?}/statement)",
        cases.len(),
        elapsed,
        elapsed / cases.len().max(1) as u32
    );

    if checked > 0 {
        let pct = |n: usize| 100.0 * n as f64 / checked as f64;
        println!(
            "\ntop-1 accuracy over {checked} statements:\n  \
             whole statement: {joint_correct}/{checked} ({:.1}%)\n  \
             clause union:    {union_correct}/{checked} ({:.1}%)",
            pct(joint_correct),
            pct(union_correct)
        );
        if !fixed.is_empty() {
            println!("\nfixed by decomposition ({}):", fixed.len());
            for s in &fixed {
                println!("  + {s:?}");
            }
        }
        if !broken.is_empty() {
            println!("\nbroken by decomposition ({}):", broken.len());
            for s in &broken {
                println!("  - {s:?}");
            }
        }

        println!("\nrecall of the expected lane (does it clear the bar at all?):");
        println!("  {:>10}  {:>12}  {:>12}", "threshold", "whole stmt", "clause union");
        for t in [0.9_f32, 0.5, 0.25, 0.0, -0.25] {
            let j = expected_cosines.iter().filter(|(c, _)| *c >= t).count();
            let u = expected_cosines.iter().filter(|(_, c)| *c >= t).count();
            println!("  {t:>10.2}  {j:>7}/{checked}  {u:>9}/{checked}");
        }
        let gained = expected_cosines.iter().filter(|(j, u)| u > j).count();
        let lost = expected_cosines.iter().filter(|(j, u)| u < j).count();
        println!(
            "  expected-lane cosine rose on {gained}/{checked}, fell on {lost}/{checked}"
        );
    }
}

fn argmax(cosines: &[f32]) -> usize {
    (0..cosines.len())
        .max_by(|&a, &b| cosines[a].total_cmp(&cosines[b]))
        .unwrap_or(0)
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

fn run_batch(
    model: &Lfm2SequenceRouter,
    path: &str,
    routes: &[String],
    show_scores: bool,
    all_scores: bool,
) {
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

        // Every lane's raw cosine, for fan-out work: the top-1 line above
        // says nothing about whether a second lane also fired, and the
        // softmax cannot be asked (route-count arithmetic, see the header).
        if all_scores {
            match model.route_cosines(&case.text, routes) {
                Ok(cosines) => {
                    for (r, route) in routes.iter().enumerate() {
                        println!("           cos {:>7.4}  {route}", cosines[r]);
                    }
                }
                Err(e) => println!("           cosines unavailable: {e}"),
            }
        }
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
