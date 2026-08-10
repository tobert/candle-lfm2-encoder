//! Proves `server::serve` actually binds and answers requests on BOTH a
//! Unix domain socket and TCP, off the SAME router — not just that the
//! code type-checks. Speaks raw HTTP/1.1 over each transport by hand
//! (no HTTP client dependency needed for one GET) rather than trusting
//! that `axum::serve` accepting a `UnixListener` compiles into something
//! that actually serves.

use std::io::ErrorKind;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UnixStream};

use lfm2d::engine_stub::StubEngine;
use lfm2d::server::{build_router, serve, AppState};
use lfm2d::worker::WorkerHandle;

async fn raw_get_healthz(mut stream: impl AsyncReadExt + AsyncWriteExt + Unpin) -> String {
    stream
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write request");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.expect("read response");
    String::from_utf8(buf).expect("response is valid utf8")
}

/// Poll until `path` exists (the socket file, once `UnixListener::bind` has
/// run) or the timeout elapses — avoids a fixed `sleep` racing against
/// however long the spawned server task takes to reach `serve()`.
async fn wait_for_path(path: &std::path::Path, timeout: Duration) {
    let start = std::time::Instant::now();
    while !path.exists() {
        if start.elapsed() > timeout {
            panic!("socket at {} never appeared within {timeout:?}", path.display());
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn serve_answers_healthz_on_both_a_unix_socket_and_tcp_concurrently() {
    let worker = WorkerHandle::spawn(StubEngine::fully_configured());
    let router = build_router(AppState { worker, ready: Arc::new(AtomicBool::new(true)) });

    let dir = tempfile::tempdir().expect("tempdir");
    let socket_path = dir.path().join("lfm2d.sock");

    // Bind TCP to an ephemeral port ourselves first so we know which port
    // `serve` will end up using — `serve` binds it again internally from
    // the string, so grab a free port and immediately release it.
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("find a free port");
    let port = probe.local_addr().unwrap().port();
    drop(probe);
    let bind_addr = format!("127.0.0.1:{port}");

    let (_shutdown_handle, shutdown_signal) = lfm2d::shutdown::channel();
    let serve_socket_path = socket_path.clone();
    let serve_bind_addr = bind_addr.clone();
    let handle = tokio::spawn(async move {
        serve(router, Some(serve_socket_path), Some(serve_bind_addr), shutdown_signal, Duration::from_secs(30)).await
    });

    wait_for_path(&socket_path, Duration::from_secs(5)).await;

    // TCP: retry the connect briefly — the listener's `bind` inside
    // `serve` happens before the socket-file wait above can observe it, so
    // by the time we've seen the socket file the TCP listener is also up,
    // but connect a couple of times just in case of scheduler jitter.
    let mut tcp_response = None;
    for _ in 0..20 {
        match TcpStream::connect(&bind_addr).await {
            Ok(stream) => {
                tcp_response = Some(raw_get_healthz(stream).await);
                break;
            }
            Err(e) if e.kind() == ErrorKind::ConnectionRefused => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(e) => panic!("unexpected TCP connect error: {e}"),
        }
    }
    let tcp_response = tcp_response.expect("TCP listener never came up");
    assert!(tcp_response.starts_with("HTTP/1.1 200"), "{tcp_response}");
    assert!(tcp_response.ends_with("ok"), "{tcp_response}");

    let uds_stream = UnixStream::connect(&socket_path).await.expect("connect over the unix socket");
    let uds_response = raw_get_healthz(uds_stream).await;
    assert!(uds_response.starts_with("HTTP/1.1 200"), "{uds_response}");
    assert!(uds_response.ends_with("ok"), "{uds_response}");

    handle.abort();
}

#[tokio::test]
async fn serve_over_a_stale_socket_file_removes_it_and_binds_anyway() {
    let worker = WorkerHandle::spawn(StubEngine::fully_configured());
    let router = build_router(AppState { worker, ready: Arc::new(AtomicBool::new(true)) });

    let dir = tempfile::tempdir().expect("tempdir");
    let socket_path = dir.path().join("stale.sock");
    // A stale, non-socket regular file at the target path — simulates a
    // leftover from a previous run that crashed without cleaning up.
    std::fs::write(&socket_path, b"not a real socket").expect("create stale file");

    let (_shutdown_handle, shutdown_signal) = lfm2d::shutdown::channel();
    let serve_socket_path = socket_path.clone();
    let handle = tokio::spawn(async move {
        serve(router, Some(serve_socket_path), None, shutdown_signal, Duration::from_secs(30)).await
    });

    // Can't wait-for-the-path-to-exist here (unlike the fresh-socket test
    // above): a stale REGULAR FILE already exists at this path before
    // `serve` ever runs, so the path "exists" from t=0 — the thing worth
    // waiting for is `serve` removing it and successfully binding a real
    // socket in its place, which only a successful connect proves.
    let mut stream = None;
    for _ in 0..50 {
        match UnixStream::connect(&socket_path).await {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
    let stream = stream.expect("serve() never replaced the stale file with a working socket");
    let response = raw_get_healthz(stream).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    handle.abort();
}
