//! An entry that names a quant file must resolve to THAT file.
//!
//! `resolve_gguf_path` used to fall back to "whatever GGUF this repo has in
//! the cache" even when the entry named a file. The gemma4 MTP drafter shares
//! its target's repo (`ggml-org/gemma-4-E2B-it-GGUF`) and names a file
//! (`mtp-gemma-4-E2B-it-Q4_0.gguf`) that neither `/api/pull` nor a first-run
//! download fetches, so the fallback handed the MTP lane the target's own
//! weights - a slot wired to draft for itself, where the pre-fallback
//! behaviour was "no drafter, load without speculation".
//!
//! `HF_HOME` is process-global, so this file must stay a single-test binary
//! (cargo gives each integration-test binary its own process).

use mettle_core::model_registry::ModelRegistry;
use mettle_server::llama_bridge::resolve_gguf_path;

const REPO_DIR: &str = "models--ggml-org--gemma-4-E2B-it-GGUF";
const TARGET_FILE: &str = "gemma-4-E2B-it-Q4_0.gguf";
const MTP_FILE: &str = "mtp-gemma-4-E2B-it-Q4_0.gguf";

/// A minimal HF cache layout as `Cache::get` and `weights::cached_gguf` read
/// it: `hub/<repo>/refs/main` -> commit, `hub/<repo>/snapshots/<commit>/...`.
fn cache_with(files: &[&str]) -> tempfile::TempDir {
    let home = tempfile::tempdir().expect("temp HF_HOME");
    let hub = home.path().join("hub").join(REPO_DIR);
    std::fs::create_dir_all(hub.join("refs")).unwrap();
    std::fs::create_dir_all(hub.join("snapshots").join("rev1")).unwrap();
    std::fs::write(hub.join("refs").join("main"), "rev1").unwrap();
    for file in files {
        std::fs::write(hub.join("snapshots").join("rev1").join(file), b"gguf").unwrap();
    }
    home
}

fn file_of(path: std::path::PathBuf) -> String {
    path.file_name().unwrap().to_string_lossy().to_string()
}

#[test]
fn mtp_drafter_never_falls_back_to_the_targets_weights() {
    let registry = ModelRegistry::new();
    let drafter = registry.resolve("gemma4:e2b-mtp").expect("drafter entry");
    let target = registry.resolve("gemma4:e2b").expect("target entry");

    // The state after a normal `/api/pull gemma4:e2b`: the target's file is
    // cached, the `-mtp` file is not.
    let home = cache_with(&[TARGET_FILE]);
    std::env::set_var("HF_HOME", home.path());

    assert_eq!(
        file_of(resolve_gguf_path(target).unwrap()),
        TARGET_FILE,
        "the target resolves to its own file"
    );
    let err = resolve_gguf_path(drafter)
        .expect_err("the drafter must refuse, not borrow the target's GGUF");
    assert!(err.contains(MTP_FILE), "error should name the file: {err}");

    // Entries that name no quant file keep the intended sibling fallback.
    let mut unnamed = drafter.clone();
    unnamed.hf_quant_filename = None;
    assert_eq!(file_of(resolve_gguf_path(&unnamed).unwrap()), TARGET_FILE);

    // And a drafter file actually present in the cache still resolves.
    let both = cache_with(&[TARGET_FILE, MTP_FILE]);
    std::env::set_var("HF_HOME", both.path());
    assert_eq!(file_of(resolve_gguf_path(drafter).unwrap()), MTP_FILE);
}
