//! Hand-rolled container-runtime probe for startup observability — NOT a
//! new dependency, just the standard filesystem/env markers every runtime
//! leaves behind:
//!
//! - `/.dockerenv` present → docker
//! - `/run/.containerenv` present → podman
//! - `KUBERNETES_SERVICE_HOST` env set → kubernetes (checked before the
//!   two file markers below so a k8s-scheduled pod is reported as
//!   "kubernetes," the operationally relevant fact, rather than whichever
//!   underlying container runtime the kubelet happens to use)
//! - `container` env set (podman/systemd-nspawn convention) → its value
//! - none of the above → bare metal / a VM, reported as `none`
//!
//! Also reads `/sys/fs/cgroup/cpu.max` when present — the cgroup v2 CPU
//! quota, directly relevant to whether `--threads`/`available_parallelism`
//! is about to oversubscribe a throttled container.
//!
//! Everything takes an explicit root directory and environment snapshot
//! (rather than reading `/` and `std::env::vars()` directly) so this is
//! unit-testable without mutating real process state — see the tests below
//! for the injection pattern; [`detect`] is the real-environment call
//! `main.rs` uses.

use std::collections::HashMap;
use std::path::Path;

/// What [`detect`] found. `Debug`/`Display`-friendly on purpose — this is
/// logged as a single structured tracing field at startup, never branched
/// on by any other code path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerProbe {
    /// `"docker"`, `"podman"`, `"kubernetes"`, the raw `container` env var
    /// value, or `"none"` if nothing matched.
    pub runtime: String,
    /// Raw contents of `cpu.max` (e.g. `"400000 100000"` for a 4-core
    /// quota, or `"max 100000"` for no quota) when the cgroup v2 file is
    /// present and readable; `None` otherwise (cgroup v1, no cgroup at
    /// all, or unreadable).
    pub cgroup_cpu_max: Option<String>,
}

/// Probe `root` (real use: `/`) and `env` (real use: a snapshot of
/// `std::env::vars()`) for container-runtime markers. Pure and
/// side-effect-free given its inputs — never panics, never touches real
/// process state itself.
pub fn detect(root: &Path, env: &HashMap<String, String>) -> ContainerProbe {
    let runtime = if env.contains_key("KUBERNETES_SERVICE_HOST") {
        "kubernetes".to_string()
    } else if root.join(".dockerenv").is_file() {
        "docker".to_string()
    } else if root.join("run/.containerenv").is_file() {
        "podman".to_string()
    } else if let Some(v) = env.get("container") {
        v.clone()
    } else {
        "none".to_string()
    };

    let cgroup_cpu_max = std::fs::read_to_string(root.join("sys/fs/cgroup/cpu.max"))
        .ok()
        .map(|s| s.trim().to_string());

    ContainerProbe { runtime, cgroup_cpu_max }
}

/// Probe the REAL environment — what `main.rs` calls. Thin wrapper around
/// [`detect`] so production code doesn't need to build its own env snapshot
/// by hand.
pub fn detect_real() -> ContainerProbe {
    let env: HashMap<String, String> = std::env::vars().collect();
    detect(Path::new("/"), &env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn empty_env() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn kubernetes_env_wins_even_with_docker_markers_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(".dockerenv"), "").expect("write");
        let mut env = empty_env();
        env.insert("KUBERNETES_SERVICE_HOST".to_string(), "10.0.0.1".to_string());
        let probe = detect(dir.path(), &env);
        assert_eq!(probe.runtime, "kubernetes");
    }

    #[test]
    fn dockerenv_marker_is_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(".dockerenv"), "").expect("write");
        let probe = detect(dir.path(), &empty_env());
        assert_eq!(probe.runtime, "docker");
    }

    #[test]
    fn containerenv_marker_is_detected_as_podman() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir(dir.path().join("run")).expect("mkdir run");
        fs::write(dir.path().join("run/.containerenv"), "").expect("write");
        let probe = detect(dir.path(), &empty_env());
        assert_eq!(probe.runtime, "podman");
    }

    #[test]
    fn container_env_var_is_reported_verbatim_when_no_file_markers_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut env = empty_env();
        env.insert("container".to_string(), "systemd-nspawn".to_string());
        let probe = detect(dir.path(), &env);
        assert_eq!(probe.runtime, "systemd-nspawn");
    }

    #[test]
    fn nothing_present_is_reported_as_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probe = detect(dir.path(), &empty_env());
        assert_eq!(probe.runtime, "none");
        assert_eq!(probe.cgroup_cpu_max, None);
    }

    #[test]
    fn dockerenv_beats_containerenv_when_both_somehow_present() {
        // Order matters when both markers exist (shouldn't happen for a
        // real runtime, but the probe must still be deterministic): docker
        // is checked first.
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(".dockerenv"), "").expect("write");
        fs::create_dir(dir.path().join("run")).expect("mkdir run");
        fs::write(dir.path().join("run/.containerenv"), "").expect("write");
        let probe = detect(dir.path(), &empty_env());
        assert_eq!(probe.runtime, "docker");
    }

    #[test]
    fn cgroup_cpu_max_is_read_verbatim_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(dir.path().join("sys/fs/cgroup")).expect("mkdir cgroup");
        fs::write(dir.path().join("sys/fs/cgroup/cpu.max"), "400000 100000\n").expect("write");
        let probe = detect(dir.path(), &empty_env());
        assert_eq!(probe.cgroup_cpu_max.as_deref(), Some("400000 100000"));
    }

    #[test]
    fn cgroup_cpu_max_absent_is_none_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probe = detect(dir.path(), &empty_env());
        assert_eq!(probe.cgroup_cpu_max, None);
    }

    #[test]
    fn detect_real_does_not_panic() {
        // Smoke test only — the real environment's actual runtime varies
        // by where this test happens to run (bare cargo test, inside a
        // container, in CI), so nothing about the result is asserted.
        let _ = detect_real();
    }
}
