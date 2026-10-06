//! First-run weight acquisition (residency contract §1.1–§1.4 of
//! `agent-docs/ENGINE_FIRST_RUN_SURFACE_CONTRACT.md`).
//!
//! mettle owns its weights: a boot whose active profile pins a model that is
//! neither at its configured path nor in the Hugging Face cache downloads it
//! itself, reporting progress into the profile `Phase` while it does. Route
//! code (`/api/pull`) shares the same fetch, so there is exactly one
//! downloader in the process.
//!
//! Resume comes from `hf-hub` itself: the tokio API writes `<blob>.sync.part`
//! and continues it with a `Range` request after a kill, verifying the sha
//! before the file appears under its real name.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use mettle_core::model_registry::RegistryEntry;

/// One download progress event, normalized away from any transport.
#[derive(Debug, Clone)]
pub struct WeightProgress {
    pub filename: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

/// Sink for [`WeightProgress`] events. `None` means "no reporting".
pub type ProgressSink = Arc<dyn Fn(WeightProgress) + Send + Sync>;

/// hf-hub reporter that forwards into the sink. hf-hub clones the reporter
/// per chunk, so the running totals live behind a shared lock.
#[derive(Clone)]
struct SinkProgress {
    sink: ProgressSink,
    filename: Arc<std::sync::Mutex<String>>,
    state: Arc<std::sync::Mutex<(u64, u64)>>,
}

impl hf_hub::api::tokio::Progress for SinkProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        if let Ok(mut f) = self.filename.lock() {
            *f = filename.to_string();
        }
        if let Ok(mut s) = self.state.lock() {
            s.0 = 0;
            s.1 = size as u64;
        }
        self.report();
    }

    async fn update(&mut self, size: usize) {
        if let Ok(mut s) = self.state.lock() {
            s.0 += size as u64;
        }
        self.report();
    }

    async fn finish(&mut self) {
        if let Ok(mut s) = self.state.lock() {
            s.0 = s.1;
        }
        self.report();
    }
}

impl SinkProgress {
    fn new(sink: ProgressSink, filename: String) -> Self {
        Self {
            sink,
            filename: Arc::new(std::sync::Mutex::new(filename)),
            state: Arc::new(std::sync::Mutex::new((0, 0))),
        }
    }

    fn report(&self) {
        let filename = self.filename.lock().map(|f| f.clone()).unwrap_or_default();
        let (bytes_done, bytes_total) = self.state.lock().map(|s| *s).unwrap_or((0, 0));
        (self.sink)(WeightProgress {
            filename,
            bytes_done,
            bytes_total,
        });
    }
}

/// Every file of an entry that lives in the HF cache: the GGUF (or the full
/// safetensors set) plus a vision projector when the entry declares one.
fn hf_filenames(entry: &RegistryEntry) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(quant) = &entry.hf_quant_filename {
        names.push(quant.clone());
    }
    if let mettle_core::model_registry::LoaderKind::GgufVision {
        mmproj_filename, ..
    } = &entry.loader_kind
    {
        if entry.local_path.is_none() {
            names.push(mmproj_filename.clone());
        }
    }
    names
}

/// Pick the GGUF file for an entry out of a repo listing when the entry does
/// not name one: the quantisation the entry advertises, else the first.
pub fn pick_gguf<'a>(files: &'a [String], quantization: &str) -> Option<&'a String> {
    let quants: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    let want = quantization.to_ascii_uppercase();
    if !want.is_empty() {
        if let Some(hit) = quants
            .iter()
            .find(|f| f.to_ascii_uppercase().contains(&want))
        {
            return Some(*hit);
        }
    }
    quants.first().copied()
}

/// Find an already-downloaded GGUF for a repo in the HF cache without knowing
/// its filename. Last resort for entries that name no quant file.
pub fn cached_gguf(entry: &RegistryEntry) -> Option<std::path::PathBuf> {
    let snapshots = hf_hub::Cache::from_env()
        .path()
        .join(format!("models--{}", entry.hf_repo_id.replace('/', "--")))
        .join("snapshots");
    let mut files: Vec<String> = Vec::new();
    for snapshot in std::fs::read_dir(&snapshots).ok()?.flatten() {
        for file in std::fs::read_dir(snapshot.path()).ok()?.flatten() {
            let name = file.file_name().to_string_lossy().to_string();
            if name.ends_with(".gguf") {
                files.push(name);
            }
        }
    }
    files.sort();
    let picked = pick_gguf(&files, &entry.metadata.quantization)?;
    for snapshot in std::fs::read_dir(&snapshots).ok()?.flatten() {
        let candidate = snapshot.path().join(picked);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Process-wide cap on concurrent mettle-originated downloads.
///
/// A pull storm (or a boot pass racing a pull) must not launch unbounded
/// parallel multi-GB fetches; two keeps a link busy without trampling the
/// cache disk.
const MAX_CONCURRENT_DOWNLOADS: usize = 2;

static DOWNLOAD_PERMITS: LazyLock<tokio::sync::Semaphore> =
    LazyLock::new(|| tokio::sync::Semaphore::new(MAX_CONCURRENT_DOWNLOADS));

/// In-process per-file download locks, keyed by `(repo, filename)`.
///
/// hf-hub's own per-blob `flock` is non-blocking (five one-second retries,
/// then `LockAcquisition`), so two mettle downloads of the same file race and
/// the loser errors out after ~5s. Waiting here serialises them; the waiter
/// then finds the file cached and skips the transfer.
type FileLocks = Mutex<HashMap<(String, String), Arc<tokio::sync::Mutex<()>>>>;

static FILE_LOCKS: LazyLock<FileLocks> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn file_lock(repo_id: &str, filename: &str) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = FILE_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    Arc::clone(
        locks
            .entry((repo_id.to_string(), filename.to_string()))
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
    )
}

/// Is `name` a plain relative path (no absolute root, no `..`/`.` component)?
///
/// Repo listings are remote input, and hf-hub's `symlink_or_rename` does not
/// reject traversal, so anything else is dropped before it can reach the cache.
fn is_safe_relative_path(name: &str) -> bool {
    use std::path::Component;
    let path = std::path::Path::new(name);
    !name.is_empty()
        && !path.is_absolute()
        && path.components().all(|c| matches!(c, Component::Normal(_)))
}

/// Resolve `filename` in the local Hugging Face cache without any network I/O.
///
/// hf-hub's streaming path (`download_with_progress`) never consults the cache
/// — it always fetches metadata and rewrites the blob — so the pull path
/// checks here first and skips the transfer for files already on disk.
fn cached_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    hf_hub::Cache::from_env()
        .model(repo_id.to_string())
        .get(filename)
}

/// Report an already-cached file to the sink as a finished transfer, keeping
/// the progress event sequence valid for NDJSON clients.
fn report_cached(sink: &ProgressSink, filename: &str, path: &std::path::Path) {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    sink(WeightProgress {
        filename: filename.to_string(),
        bytes_done: bytes,
        bytes_total: bytes,
    });
}

/// Download an entry's files into the standard Hugging Face cache, where the
/// engine loader finds them. GGUF entries fetch the quantised file; entries
/// without one fetch the full safetensors set.
pub async fn fetch_model_files(
    entry: &RegistryEntry,
    sink: Option<ProgressSink>,
) -> Result<(), String> {
    // Bound concurrent downloads process-wide. Held for the whole entry, so
    // the files of one entry occupy one download slot.
    let _download_slot = DOWNLOAD_PERMITS
        .acquire()
        .await
        .map_err(|e| format!("download limiter closed: {}", e))?;

    let api = hf_hub::api::tokio::ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| format!("hf api: {}", e))?;
    let repo = api.model(entry.hf_repo_id.clone());

    let mut filenames = hf_filenames(entry);
    if filenames.is_empty() {
        let info = repo
            .info()
            .await
            .map_err(|e| format!("failed to fetch repo info for {}: {}", entry.hf_repo_id, e))?;
        // Repo listings are remote input: drop anything that is not a plain
        // relative path before it can reach the cache filesystem.
        let listing: Vec<String> = info
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|f| is_safe_relative_path(f))
            .collect();
        // An entry that names no quant file (a user-defined GGUF entry, or a
        // repo with a single quant) still needs its weight file.
        match pick_gguf(&listing, &entry.metadata.quantization) {
            Some(gguf) => filenames = vec![gguf.clone()],
            None => {
                filenames = listing
                    .into_iter()
                    .filter(|f| {
                        f.ends_with(".safetensors")
                            || f.ends_with(".json")
                            || f.ends_with(".jinja")
                            || f.ends_with(".model")
                    })
                    .collect()
            }
        }
    }
    if filenames.is_empty() {
        return Err(format!("no downloadable files in {}", entry.hf_repo_id));
    }

    for filename in &filenames {
        // Cache-first: hf-hub's streaming path re-downloads even a fully
        // cached file, so resolve it here and skip the network entirely.
        if let Some(path) = cached_file(&entry.hf_repo_id, filename) {
            if let Some(sink) = &sink {
                report_cached(sink, filename, &path);
            }
            continue;
        }

        // Serialise with any other in-process download of this exact file,
        // then re-check: the download we queued behind may have cached it.
        let lock = file_lock(&entry.hf_repo_id, filename);
        let _file_guard = lock.lock().await;
        if let Some(path) = cached_file(&entry.hf_repo_id, filename) {
            if let Some(sink) = &sink {
                report_cached(sink, filename, &path);
            }
            continue;
        }

        match &sink {
            Some(sink) => {
                let reporter = SinkProgress::new(Arc::clone(sink), filename.clone());
                repo.download_with_progress(filename, reporter)
                    .await
                    .map_err(|e| format!("download of {} failed: {}", filename, e))?;
            }
            None => {
                repo.get(filename)
                    .await
                    .map_err(|e| format!("download of {} failed: {}", filename, e))?;
            }
        }
    }
    Ok(())
}

/// Can this entry be fetched from Hugging Face at all?
///
/// An entry with a configured local path is operator-provided: if it is
/// missing, that is a config error, never something to download around. And an
/// entry must name a concrete file in a known repo — a `[[model]]` id that
/// matches no built-in alias gets `hf_repo_id = <the alias itself>`, which is
/// not a repo to fetch from.
pub fn is_downloadable(entry: &RegistryEntry) -> bool {
    entry.local_path.is_none() && entry.hf_quant_filename.is_some()
}

/// Is every file this entry needs already resolvable on disk?
///
/// Non-vision entries need only the GGUF, so this is the old GGUF-only test
/// (`resolve_gguf_path`, strict named-file rule included) for them. A
/// [`LoaderKind::GgufVision`](mettle_core::model_registry::LoaderKind::GgufVision)
/// entry also needs its projector: a download interrupted between the two
/// files would otherwise leave the model text-only forever, because a
/// GGUF-only presence test reports it as complete and nothing else re-derives
/// presence from the mmproj.
pub fn model_files_present(entry: &RegistryEntry) -> bool {
    crate::llama_bridge::resolve_gguf_path(entry).is_ok() && mmproj_present(entry)
}

/// Non-logging twin of the lookup `llama_bridge::resolve_mmproj_path`
/// performs for a vision entry: the configured `mmproj_path` first, then the
/// HF cache. `true` for entries that declare no projector at all.
///
/// Presence is probed for every pinned entry on every boot, so unlike the
/// loader's resolver this must not warn about a configured-but-missing
/// `mmproj_path` — an incomplete entry is exactly what the probe is for.
fn mmproj_present(entry: &RegistryEntry) -> bool {
    let mettle_core::model_registry::LoaderKind::GgufVision {
        mmproj_filename, ..
    } = &entry.loader_kind
    else {
        return true;
    };
    if let Some(p) = &entry.mmproj_path {
        if p.exists() {
            return true;
        }
    }
    cached_file(&entry.hf_repo_id, mmproj_filename).is_some()
}

/// Ensure an entry's weights are on disk and return the GGUF path.
///
/// Order: configured local path → HF cache → download (when the registry
/// knows a Hugging Face source). Offline mode never downloads.
pub async fn ensure_weights(
    entry: &RegistryEntry,
    offline: bool,
    sink: Option<ProgressSink>,
) -> Result<PathBuf, String> {
    if model_files_present(entry) {
        return crate::llama_bridge::resolve_gguf_path(entry);
    }
    // Not complete, but the GGUF itself resolves: the only file
    // `model_files_present` can be missing here is a vision entry's mmproj
    // (it is in `hf_filenames`, so `fetch_model_files` re-checks it and
    // downloads just that file). When there is no source to fetch from — a
    // configured local path, no HF source, or offline mode — return the GGUF
    // so the entry still loads text-only, exactly as the GGUF-only test did.
    if let Ok(path) = crate::llama_bridge::resolve_gguf_path(entry) {
        if is_downloadable(entry) && !offline {
            tracing::info!(
                "model '{}' is partially on disk; fetching its missing files from {}",
                entry.alias,
                entry.hf_repo_id
            );
            fetch_model_files(entry, sink).await?;
        }
        return Ok(path);
    }
    if let Some(p) = &entry.local_path {
        return Err(format!(
            "model '{}' configured local path '{}' does not exist",
            entry.alias,
            p.display()
        ));
    }
    if !is_downloadable(entry) {
        return Err(format!(
            "model '{}' has no Hugging Face source to download from",
            entry.alias
        ));
    }
    if offline {
        return Err(format!(
            "model '{}' is not on disk and offline mode forbids downloading it",
            entry.alias
        ));
    }
    tracing::info!(
        "model '{}' is not on disk; downloading {} from {}",
        entry.alias,
        entry.hf_quant_filename.as_deref().unwrap_or("(full repo)"),
        entry.hf_repo_id
    );
    fetch_model_files(entry, sink).await?;
    crate::llama_bridge::resolve_gguf_path(entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mettle_core::model_registry::{LoaderKind, ModelRegistry};

    fn entry(id: &str) -> RegistryEntry {
        let r = ModelRegistry::new();
        r.resolve(id).unwrap().clone()
    }

    #[test]
    fn only_entries_with_a_known_file_are_downloadable() {
        // A built-in alias: repo + quant file known.
        assert!(is_downloadable(&entry("qwen3-embedding-0.6b:latest")));
        // A configured local path is never downloaded around.
        let mut local = entry("gemma4:e2b");
        local.local_path = Some(PathBuf::from("/nonexistent/model.gguf"));
        assert!(!is_downloadable(&local));
        // A user `[[model]]` id that matches no alias has the alias itself as
        // its "repo": there is nothing to download from.
        let mut unknown = entry("gemma4:e2b");
        unknown.local_path = None;
        unknown.hf_quant_filename = None;
        unknown.hf_repo_id = "m1".into();
        assert!(!is_downloadable(&unknown));
    }

    #[tokio::test]
    async fn ensure_weights_short_circuits_on_a_present_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        let mut e = entry("gemma4:e2b");
        e.local_path = Some(path.clone());
        // Offline proves no network path was taken.
        let got = ensure_weights(&e, true, None).await.unwrap();
        assert_eq!(got, path);
    }

    #[tokio::test]
    async fn vision_entry_is_incomplete_without_its_mmproj() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        let mmproj = dir.path().join("mmproj.gguf");
        std::fs::write(&mmproj, b"MMPROJ").unwrap();

        let mut e = entry("unlimited-ocr:latest");
        assert!(matches!(e.loader_kind, LoaderKind::GgufVision { .. }));
        // A repo that cannot exist in the local HF cache, so presence depends
        // only on the two paths above.
        e.hf_repo_id = "mettle-tests/nonexistent-vision-repo".into();
        e.loader_kind = LoaderKind::GgufVision {
            quant_filename: "model.gguf".into(),
            mmproj_filename: "mmproj-not-in-cache.gguf".into(),
        };
        e.local_path = Some(gguf.clone());
        e.mmproj_path = Some(mmproj.clone());
        assert!(model_files_present(&e));

        // The GGUF alone is not enough: a missing projector is detected even
        // though the weight file resolves.
        std::fs::remove_file(&mmproj).unwrap();
        assert!(!model_files_present(&e));

        // A local-path entry has no source to fetch the mmproj from, so it
        // still degrades to the text-only GGUF instead of failing the boot.
        assert_eq!(ensure_weights(&e, true, None).await.unwrap(), gguf);
    }

    #[tokio::test]
    async fn missing_local_path_is_a_config_error_not_a_download() {
        let mut e = entry("gemma4:e2b");
        e.local_path = Some(PathBuf::from("/nonexistent/model.gguf"));
        let err = ensure_weights(&e, false, None).await.unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
    }

    #[tokio::test]
    async fn ensure_weights_refuses_an_entry_with_no_hf_source() {
        let mut e = entry("gemma4:e2b");
        e.local_path = None;
        e.hf_quant_filename = None;
        let err = ensure_weights(&e, false, None).await.unwrap_err();
        assert!(err.contains("no Hugging Face source"), "{err}");
    }

    #[tokio::test]
    async fn offline_refuses_to_download_an_absent_model() {
        let mut e = entry("gemma4:e2b");
        e.local_path = None;
        e.hf_quant_filename = Some("something-not-cached.gguf".into());
        let err = ensure_weights(&e, true, None).await.unwrap_err();
        assert!(err.contains("offline mode"), "{err}");
    }

    #[test]
    fn gguf_selection_prefers_the_advertised_quant() {
        let files = vec![
            "README.md".to_string(),
            "model-Q4_K_M.gguf".to_string(),
            "model-Q8_0.gguf".to_string(),
        ];
        assert_eq!(pick_gguf(&files, "Q8_0").unwrap(), "model-Q8_0.gguf",);
        // No advertisesd quant match -> first GGUF, never a non-GGUF file.
        assert_eq!(pick_gguf(&files, "F16").unwrap(), "model-Q4_K_M.gguf",);
        assert!(pick_gguf(&["README.md".to_string()], "Q8_0").is_none());
    }

    #[test]
    fn gguf_entries_carry_their_quant_filename() {
        let e = entry("qwen3-embedding-0.6b:latest");
        assert!(matches!(e.loader_kind, LoaderKind::Embedding { .. }));
        assert!(is_downloadable(&e), "{:?}", e.local_path);
    }
}
