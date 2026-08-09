//! The rank-then-route composite (`src/cascade.rs`), as a CLI — rank a
//! statement's clauses by severity with `kube_ordinal_v6`, route the
//! winner with the Prompt-Router. See
//! `training/router/findings_clause_decomposition.md` for what this
//! reproduces (5/5, against 3/5 whole-statement top-1 and 0/5 max-cosine
//! union) and why the two heads are composed rather than either used alone.
//!
//! ```text
//! cargo run --release --example cascade -- \
//!     <classifier-dir> <router-dir> \
//!     --routes-file routes.json \
//!     --clauses-file clauses.json \
//!     [--severe LABEL]...  (repeatable; default: mutating, destructive)
//!     [--threshold F]      (union lanes at/above this cosine, per statement)
//! ```
//!
//! `--routes-file` is a JSON array of route strings — same format
//! `examples/route.rs --routes-file` reads (NOT `safety_specialists.json`
//! directly, which is a richer per-lane object; extract one of its
//! `variant_*` fields into a plain string array first, the same way
//! `tests/clause_routing.rs`'s `LANES` const does by hand).
//!
//! `--clauses-file` is a JSON array of `{"statement": "...", "expect":
//! "...", "clauses": ["...", ...]}` — the exact shape
//! `examples/route.rs --clauses-file` reads, so existing decomposed-probe
//! files work unchanged. `expect` is optional and, here, purely
//! informational (printed alongside the winner; this example does not
//! score accuracy against it — `examples/route.rs --clauses-file` already
//! does that for the router alone).
//!
//! No splitter here either: clauses must already be split by a real
//! parser (kaish `Plan.commands[]`) before they reach `--clauses-file`.

use std::time::Instant;

use candle_lfm2_encoder::{Cascade, Lfm2SequenceClassifier, Lfm2SequenceRouter};
use serde::Deserialize;

/// One decomposed statement — identical shape to `examples/route.rs`'s
/// `ClauseCase`, so files written for that example work here unchanged.
#[derive(Deserialize)]
struct ClauseCase {
    statement: String,
    expect: Option<String>,
    clauses: Vec<String>,
}

const DEFAULT_SEVERE: [&str; 2] = ["mutating", "destructive"];

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: cascade <classifier-dir> <router-dir> --routes-file routes.json \
         --clauses-file clauses.json [--severe LABEL]... [--threshold F]"
    );
    std::process::exit(2);
}

fn main() -> candle_lfm2_encoder::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        usage_and_exit();
    }
    let classifier_dir = args[0].clone();
    let router_dir = args[1].clone();

    let mut routes_file: Option<String> = None;
    let mut clauses_file: Option<String> = None;
    let mut severe: Vec<String> = Vec::new();
    let mut threshold: Option<f32> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--routes-file" => {
                routes_file = Some(args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("--routes-file needs a path");
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
            "--severe" => {
                severe.push(args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("--severe needs a label name");
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
            other => {
                eprintln!("unrecognized argument {other:?}");
                usage_and_exit();
            }
        }
    }

    let routes_path = routes_file.unwrap_or_else(|| {
        eprintln!("--routes-file is required");
        usage_and_exit();
    });
    let clauses_path = clauses_file.unwrap_or_else(|| {
        eprintln!("--clauses-file is required");
        usage_and_exit();
    });

    let routes: Vec<String> = {
        let bytes = std::fs::read(&routes_path).unwrap_or_else(|e| {
            eprintln!("reading {routes_path}: {e}");
            std::process::exit(1);
        });
        serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            eprintln!("parsing {routes_path} as a JSON array of strings: {e}");
            std::process::exit(1);
        })
    };
    if routes.is_empty() {
        eprintln!("{routes_path} has no routes");
        std::process::exit(2);
    }

    let cases: Vec<ClauseCase> = {
        let bytes = std::fs::read(&clauses_path).unwrap_or_else(|e| {
            eprintln!("reading {clauses_path}: {e}");
            std::process::exit(1);
        });
        serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            eprintln!("parsing {clauses_path} as a JSON array of {{statement, expect?, clauses}}: {e}");
            std::process::exit(1);
        })
    };

    let severe = if severe.is_empty() {
        let d: Vec<String> = DEFAULT_SEVERE.iter().map(|s| s.to_string()).collect();
        println!(
            "no --severe given; defaulting to {} (kube_ordinal_v6's convention — pass \
             --severe to override for a different checkpoint's label set)",
            d.join(", ")
        );
        d
    } else {
        severe
    };

    let t = Instant::now();
    let classifier = Lfm2SequenceClassifier::from_dir(&classifier_dir)?;
    let router = Lfm2SequenceRouter::from_dir(&router_dir)?;
    println!(
        "loaded classifier {classifier_dir} ({} labels: {}) and router {router_dir} \
         ({} routes) in {:.2?}",
        classifier.num_labels(),
        classifier.labels().join(", "),
        routes.len(),
        t.elapsed()
    );
    println!("severe set: {}", severe.join(", "));
    println!("routes: {}\n", routes.join(" | "));

    let cascade = Cascade::new(&classifier, &router);

    let t = Instant::now();
    for case in &cases {
        println!("statement: {:?}", case.statement);
        if let Some(expect) = &case.expect {
            println!("  (router --clauses-file 'expect' field: {expect})");
        }

        let verdict = match cascade.run(&case.clauses, &routes, &severe) {
            Ok(v) => v,
            Err(e) => {
                println!("  ERROR: {e}\n");
                continue;
            }
        };

        // Per-clause table: severity score, top lane, that lane's cosine.
        println!(
            "  {:>3}  {:>8}  {:<28} {:>8}  clause",
            "#", "severity", "top lane", "cosine"
        );
        for (i, c) in verdict.clauses.iter().enumerate() {
            let top = (0..c.lane_cosines.len())
                .max_by(|&a, &b| c.lane_cosines[a].total_cmp(&c.lane_cosines[b]))
                .expect("routes is non-empty, checked above");
            let marker = if i == verdict.winner { "*" } else { " " };
            println!(
                "  {marker}{i:>2}  {:>8.4}  {:<28} {:>8.4}  {:?}",
                c.severity_score, routes[top], c.lane_cosines[top], c.clause
            );
        }

        // Ranked order (severity descending; ties keep input order — see
        // src/cascade.rs's `aggregate` docs).
        println!(
            "  ranked by severity: {}",
            verdict
                .ranked
                .iter()
                .map(|&i| format!("[{i}] {:.4}", verdict.clauses[i].severity_score))
                .collect::<Vec<_>>()
                .join(" > ")
        );

        // Winner + its lane — the cascade's single answer.
        let winner = &verdict.clauses[verdict.winner];
        println!(
            "  WINNER: clause [{}] {:?} — severity {:.4} — routes to {} (cosine {:.4})",
            verdict.winner,
            winner.clause,
            winner.severity_score,
            routes[verdict.winner_lane],
            winner.lane_cosines[verdict.winner_lane],
        );

        // Union lanes above --threshold — the fan-out/recall channel,
        // independent of which clause won the severity ranking.
        if let Some(t) = threshold {
            let mut lanes: Vec<usize> = (0..routes.len())
                .filter(|&l| verdict.union_cosines[l] >= t)
                .collect();
            lanes.sort_by(|&a, &b| verdict.union_cosines[b].total_cmp(&verdict.union_cosines[a]));
            if lanes.is_empty() {
                println!("  union lanes >= {t}: (none)");
            } else {
                println!("  union lanes >= {t}:");
                for lane in lanes {
                    let fired = verdict.firing_clause(lane).expect("non-empty clause list");
                    println!(
                        "    {:<28} {:>8.4}  (from clause [{fired}] {:?})",
                        routes[lane], verdict.union_cosines[lane], verdict.clauses[fired].clause
                    );
                }
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

    Ok(())
}
