//! Process-exit contract (residency contract §1.6): a port clash is a
//! structured exit, not a panic or a signal, so a supervisor can tell
//! "another engine holds the port" from "the engine crashed".
//!
//! The bind must happen before any model/CUDA work is started. With the bind
//! after engine init, `std::process::exit(98)` runs atexit handlers while the
//! loader threads are live and the process dies with SIGSEGV (observed exit
//! code 139) instead of 98.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Occupy a port, start `mettled` against it, and assert it exits 98.
#[test]
fn port_clash_exits_98() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe port");
    let port = listener.local_addr().unwrap().port();

    let dir = std::env::temp_dir().join(format!("mettle-exit98-{port}"));
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.toml");
    std::fs::write(
        &config,
        format!("[server]\nhost = \"127.0.0.1\"\nport = {port}\n"),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_mettled"))
        .env("METTLE_CONFIG", &config)
        // Hermetic: never read the operator's real collections dir. The
        // `info` level is load-bearing - the ordering assertion below reads
        // a startup log line, and `error` would suppress it.
        .env("XDG_DATA_HOME", &dir)
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mettled");

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        match child.try_wait().expect("wait mettled") {
            Some(s) => break s,
            None if Instant::now() > deadline => {
                let _ = child.kill();
                panic!("mettled did not exit while the port was held");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    };

    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("piped stderr")
        .read_to_string(&mut stderr)
        .unwrap();
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .expect("piped stdout")
        .read_to_string(&mut stdout)
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        status.code(),
        Some(98),
        "expected exit 98, got {status:?}; stderr: {stderr}"
    );
    assert!(
        stderr.contains("already in use"),
        "stderr should name the clash, got: {stderr}"
    );
    // Ordering, not just the code: `load_all` logs this unconditionally at
    // startup, so its presence proves post-bind work ran before the bind
    // (the ordering that segfaulted on a CUDA host behind `exit(98)`).
    // Exit code and message alone stay green with the bind moved back.
    assert!(
        !stdout.contains("collections from disk"),
        "startup work ran before the bind; stdout: {stdout}"
    );
}
