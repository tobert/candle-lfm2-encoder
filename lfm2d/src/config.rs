//! CLI configuration — flags with env-var fallbacks (`clap`'s `env`
//! feature). Startup fails loudly on anything incoherent; see
//! [`Cli::validate`].

use std::path::PathBuf;

use clap::Parser;

/// `lfm2d` — HTTP sidecar serving LFM2.5 encoder heads. Loads every
/// configured checkpoint once at startup (no lazy loading); serves the same
/// API over a Unix socket and/or TCP.
#[derive(Parser, Debug, Clone)]
#[command(name = "lfm2d", about, long_about = None)]
pub struct Cli {
    /// Directory holding an `Lfm2Embedding`-shaped checkpoint
    /// (`config.json`, `tokenizer.json`, `model.safetensors`). Backs
    /// `/embed`.
    #[arg(long, env = "LFM2D_EMBEDDER_DIR")]
    pub embedder_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2SequenceClassifier`-shaped checkpoint.
    /// Backs `/predict`, `/v1/classify`, and (with `--router-dir`)
    /// `/v1/cascade`.
    #[arg(long, env = "LFM2D_CLASSIFIER_DIR")]
    pub classifier_dir: Option<PathBuf>,

    /// Directory holding an `Lfm2SequenceRouter`-shaped checkpoint (the
    /// Prompt-Router). Backs `/v1/route` and (with `--classifier-dir`)
    /// `/v1/cascade`.
    #[arg(long, env = "LFM2D_ROUTER_DIR")]
    pub router_dir: Option<PathBuf>,

    /// A cascade candidate route (repeatable), or a comma-separated list
    /// via `LFM2D_CASCADE_ROUTES`. `/v1/cascade` needs at least one; the
    /// request body carries only `clauses` (see the crate root docs on why
    /// routes are server-side config, not a per-request field).
    #[arg(long = "cascade-route", env = "LFM2D_CASCADE_ROUTES", value_delimiter = ',')]
    pub cascade_routes: Vec<String>,

    /// Which of the classifier's own labels count toward `/v1/cascade`'s
    /// severity ranking sum — repeatable, or comma-separated via
    /// `LFM2D_CASCADE_SEVERE_LABELS`. Defaults to `kube_ordinal_v6`'s
    /// convention (`examples/cascade.rs`'s `DEFAULT_SEVERE`).
    #[arg(
        long = "cascade-severe-label",
        env = "LFM2D_CASCADE_SEVERE_LABELS",
        value_delimiter = ',',
        default_value = "mutating,destructive"
    )]
    pub cascade_severe_labels: Vec<String>,

    /// Unix domain socket path to serve on. At least one of this or
    /// `--bind-addr` is required.
    #[arg(long, env = "LFM2D_SOCKET_PATH")]
    pub socket_path: Option<PathBuf>,

    /// TCP address to serve on, e.g. `127.0.0.1:8080`. At least one of this
    /// or `--socket-path` is required.
    #[arg(long, env = "LFM2D_BIND_ADDR")]
    pub bind_addr: Option<String>,

    /// Size of the rayon global thread pool that candle's matmul runs on
    /// (`rayon::ThreadPoolBuilder::num_threads`), set BEFORE any model is
    /// loaded — rayon's global pool can only be built once, so this must
    /// win the race against candle's own lazy default. Defaults to
    /// `std::thread::available_parallelism()` when unset, which under a
    /// k8s CPU *limit* (a cgroup quota, not just a request) usually
    /// over-reports — see `lfm2d/deploy/k8s.yaml`'s CPU-limits comment for
    /// why this daemon's deploy guidance is "requests without limits."
    #[arg(long, env = "LFM2D_THREADS")]
    pub threads: Option<usize>,
}

impl Cli {
    /// Cross-field checks `clap` itself can't express. Pure and
    /// unit-testable without ever invoking the CLI parser — see the tests
    /// below.
    ///
    /// # Errors
    /// - neither `--socket-path` nor `--bind-addr` given: nothing to serve
    ///   on.
    /// - none of `--embedder-dir`/`--classifier-dir`/`--router-dir` given:
    ///   nothing to load, so this process would serve `/healthz` and
    ///   nothing else — almost certainly a misconfiguration, not a
    ///   deliberate deployment.
    pub fn validate(&self) -> Result<(), String> {
        if self.socket_path.is_none() && self.bind_addr.is_none() {
            return Err(
                "no transport configured: pass --socket-path and/or --bind-addr \
                 (env LFM2D_SOCKET_PATH / LFM2D_BIND_ADDR) — at least one is required"
                    .to_string(),
            );
        }
        if self.embedder_dir.is_none() && self.classifier_dir.is_none() && self.router_dir.is_none() {
            return Err(
                "no models configured: pass at least one of --embedder-dir/--classifier-dir/\
                 --router-dir (env LFM2D_EMBEDDER_DIR/LFM2D_CLASSIFIER_DIR/LFM2D_ROUTER_DIR)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Cli {
        Cli {
            embedder_dir: None,
            classifier_dir: None,
            router_dir: None,
            cascade_routes: Vec::new(),
            cascade_severe_labels: vec!["mutating".into(), "destructive".into()],
            socket_path: None,
            bind_addr: None,
            threads: None,
        }
    }

    #[test]
    fn rejects_no_transport_configured() {
        let mut cli = base();
        cli.embedder_dir = Some("/tmp/x".into());
        let err = cli.validate().expect_err("no socket_path/bind_addr must be refused");
        assert!(err.contains("transport"), "{err}");
    }

    #[test]
    fn rejects_no_models_configured() {
        let mut cli = base();
        cli.bind_addr = Some("127.0.0.1:0".into());
        let err = cli.validate().expect_err("no models configured must be refused");
        assert!(err.contains("models"), "{err}");
    }

    #[test]
    fn accepts_one_model_and_socket_path_only() {
        let mut cli = base();
        cli.router_dir = Some("/tmp/router".into());
        cli.socket_path = Some("/tmp/lfm2d.sock".into());
        cli.validate().expect("one model + one transport is valid");
    }

    #[test]
    fn accepts_both_transports_and_all_three_models() {
        let cli = Cli {
            embedder_dir: Some("/tmp/e".into()),
            classifier_dir: Some("/tmp/c".into()),
            router_dir: Some("/tmp/r".into()),
            cascade_routes: vec!["shell".into()],
            cascade_severe_labels: vec!["mutating".into()],
            socket_path: Some("/tmp/lfm2d.sock".into()),
            bind_addr: Some("0.0.0.0:8080".into()),
            threads: Some(4),
        };
        cli.validate().expect("fully specified config is valid");
    }
}
