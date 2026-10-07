//! First-run weight acquisition (residency contract §1.1–§1.4 of
//! `agent-docs/ENGINE_FIRST_RUN_SURFACE_CONTRACT.md`).
//!
//! mettle owns its weights: a boot whose active profile pins a model that is
//! neither at its configured path nor in the Hugging Face cache downloads it
//! itself, reporting progress into the profile `Phase` while it does. Route
//! code (`/api/pull`) shares the same fetch, so there is exactly one
//! downloader in the process.
//!
//! Every decision about where weights come from matches on
//! [`RegistryEntry::weight_source`]; nothing here re-derives it from
//! `local_path` / `hf_quant_filename`.
//!
//! Resume comes from `hf-hub` itself: the tokio API writes `<blob>.sync.part`
//! and continues it with a `Range` request after a kill, verifying the sha
//! before the file appears under its real name.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use mettle_core::model_registry::{RegistryEntry, WeightSource};

/// One download progress event, normalized away from any transport.
#[derive(Debug, Clone)]
pub struct WeightProgress {
    pub filename: Arc<str>,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

/// What an acquisition reports to its [`ProgressSink`].
#[derive(Debug, Clone)]
pub enum WeightEvent {
    /// An entry passed the download gate and its fetch begins (it may still
    /// queue behind the process-wide download cap before bytes flow).
    Fetching,
    /// Byte progress for one file. Throttled to about one event per percent
    /// of the file, always including the first and the final one.
    Progress(WeightProgress),
}

/// Sink for [`WeightEvent`]s. `None` means "no reporting".
pub type ProgressSink = Arc<dyn Fn(WeightEvent) + Send + Sync>;

/// Why an entry's weights could not be put on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    /// A configured `[[model]] path` that does not exist: a config error,
    /// never something to download around.
    LocalPathMissing { alias: String, path: PathBuf },
    /// The entry has no Hugging Face file to fetch: it is configured from a
    /// local path, or it names no quantised GGUF.
    NotDownloadable { alias: String, local_path: bool },
    /// Not on disk, and `[server] offline` forbids fetching it.
    Offline { alias: String },
    /// The transfer itself failed.
    Download(String),
}

impl WeightError {
    /// A configuration problem rather than an acquisition failure: retrying
    /// cannot fix it, only the operator can.
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            WeightError::LocalPathMissing { .. } | WeightError::NotDownloadable { .. }
        )
    }
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::LocalPathMissing { alias, path } => write!(
                f,
                "model '{alias}' configured local path '{}' does not exist",
                path.display()
            ),
            WeightError::NotDownloadable { alias, local_path } => write!(
                f,
                "model '{alias}' has no Hugging Face source to download from: {}",
                if *local_path {
                    "it is configured from a local path"
                } else {
                    "it names no quantised GGUF file in its Hugging Face repo"
                }
            ),
            WeightError::Offline { alias } => write!(
                f,
                "model '{alias}' is not on disk and offline mode forbids downloading it"
            ),
            WeightError::Download(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for WeightError {}

/// Emit about this many progress events per file (plus the first and last).
const PROGRESS_STEPS: u64 = 100;

/// hf-hub reporter that forwards throttled progress into the sink. hf-hub
/// clones the reporter for every parallel chunk, so the running totals are
/// shared atomics.
#[derive(Clone)]
struct SinkProgress {
    sink: ProgressSink,
    transfer: Arc<Transfer>,
}

struct Transfer {
    filename: Arc<str>,
    done: AtomicU64,
    total: AtomicU64,
    /// `done` at the last emitted event.
    reported: AtomicU64,
}

impl hf_hub::api::tokio::Progress for SinkProgress {
    async fn init(&mut self, size: usize, _filename: &str) {
        let t = &self.transfer;
        t.total.store(size as u64, Ordering::Relaxed);
        t.done.store(0, Ordering::Relaxed);
        t.reported.store(0, Ordering::Relaxed);
        self.emit(0);
    }

    async fn update(&mut self, size: usize) {
        let t = &self.transfer;
        let done = t.done.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
        let step = (t.total.load(Ordering::Relaxed) / PROGRESS_STEPS).max(1);
        let last = t.reported.load(Ordering::Relaxed);
        // One winner per step even with parallel chunks reporting at once.
        if done >= last.saturating_add(step)
            && t.reported
                .compare_exchange(last, done, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.emit(done);
        }
    }

    async fn finish(&mut self) {
        let t = &self.transfer;
        let total = t.total.load(Ordering::Relaxed);
        t.done.store(total, Ordering::Relaxed);
        t.reported.store(total, Ordering::Relaxed);
        self.emit(total);
    }
}

impl SinkProgress {
    fn new(sink: ProgressSink, filename: &str) -> Self {
        Self {
            sink,
            transfer: Arc::new(Transfer {
                filename: Arc::from(filename),
                done: AtomicU64::new(0),
                total: AtomicU64::new(0),
                reported: AtomicU64::new(0),
            }),
        }
    }

    fn emit(&self, bytes_done: u64) {
        (self.sink)(WeightEvent::Progress(WeightProgress {
            filename: Arc::clone(&self.transfer.filename),
            bytes_done,
            bytes_total: self.transfer.total.load(Ordering::Relaxed),
        }));
    }
}

/// Pick the GGUF file for an entry out of a file listing when the entry does
/// not name one: the quantisation the entry advertises, else the first.
fn pick_gguf<'a, S: AsRef<str>>(files: &'a [S], quantization: &str) -> Option<&'a S> {
    let mut ggufs = files.iter().filter(|f| f.as_ref().ends_with(".gguf"));
    let want = quantization.to_ascii_uppercase();
    if !want.is_empty() {
        if let Some(hit) = ggufs
            .clone()
            .find(|f| f.as_ref().to_ascii_uppercase().contains(&want))
        {
            return Some(hit);
        }
    }
    ggufs.next()
}

/// Resolve `filename` of `repo_id` in the local Hugging Face cache without
/// any network I/O. The one cache lookup every resolver shares.
///
/// hf-hub's streaming path (`download_with_progress`) never consults the cache
/// — it always fetches metadata and rewrites the blob — so the download path
/// checks here first and skips the transfer for files already on disk.
pub fn cached_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    cached_in(&hf_hub::Cache::from_env(), repo_id, filename)
}

fn cached_in(cache: &hf_hub::Cache, repo_id: &str, filename: &str) -> Option<PathBuf> {
    cache.model(repo_id.to_string()).get(filename)
}

/// Find an already-downloaded GGUF for a repo in the HF cache without knowing
/// its filename ([`WeightSource::HfAnyGguf`]): the advertised quantisation,
/// else the first by name.
pub fn cached_gguf(repo_id: &str, quantization: &str) -> Option<PathBuf> {
    let snapshots = hf_hub::Cache::from_env()
        .path()
        .join(hf_hub::Repo::model(repo_id.to_string()).folder_name())
        .join("snapshots");
    // One listing pass: (name, path) per GGUF across every snapshot. The
    // stable sort keeps snapshot order among equal names.
    let mut ggufs: Vec<(String, PathBuf)> = Vec::new();
    for snapshot in std::fs::read_dir(&snapshots).ok()?.flatten() {
        for file in std::fs::read_dir(snapshot.path()).ok()?.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            if name.ends_with(".gguf") {
                ggufs.push((name, file.path()));
            }
        }
    }
    ggufs.sort_by(|a, b| a.0.cmp(&b.0));
    let names: Vec<&str> = ggufs.iter().map(|(name, _)| name.as_str()).collect();
    let picked = *pick_gguf(&names, quantization)?;
    ggufs
        .iter()
        .find(|(name, path)| name == picked && path.exists())
        .map(|(_, path)| path.clone())
}

/// Process-wide cap on concurrent mettle-originated downloads.
///
/// A pull storm (or a boot pass racing a pull) must not launch unbounded
/// parallel multi-GB fetches; two keeps a link busy without trampling the
/// cache disk. Only an actual transfer holds a permit: a cache hit never
/// queues behind a running download.
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

/// If `filename` is already cached, report it to the sink as a finished
/// transfer (keeping the progress sequence valid for NDJSON clients) and
/// return `true`.
fn reuse_cached(
    cache: &hf_hub::Cache,
    repo_id: &str,
    filename: &str,
    sink: Option<&ProgressSink>,
) -> bool {
    let Some(path) = cached_in(cache, repo_id, filename) else {
        return false;
    };
    if let Some(sink) = sink {
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        sink(WeightEvent::Progress(WeightProgress {
            filename: Arc::from(filename),
            bytes_done: bytes,
            bytes_total: bytes,
        }));
    }
    true
}

/// The Hugging Face repo and files an entry downloads, or why it has none.
/// The GGUF plus the vision projector when the entry declares one.
fn download_set(entry: &RegistryEntry) -> Result<(&str, Vec<&str>), WeightError> {
    match entry.weight_source() {
        WeightSource::HfFile { repo, file } => Ok((
            repo,
            std::iter::once(file)
                .chain(entry.mmproj_filename())
                .collect(),
        )),
        source => Err(WeightError::NotDownloadable {
            alias: entry.alias.clone(),
            local_path: matches!(source, WeightSource::Local(_)),
        }),
    }
}

/// Download an entry's files into the standard Hugging Face cache, where the
/// engine loader finds them. Files already cached are reported, not fetched.
///
/// An entry with no Hugging Face file ([`WeightSource::Local`] /
/// [`WeightSource::HfAnyGguf`]) is refused before any I/O.
pub async fn fetch_model_files(
    entry: &RegistryEntry,
    sink: Option<ProgressSink>,
) -> Result<(), WeightError> {
    let (repo_id, files) = download_set(entry)?;
    if let Some(sink) = &sink {
        sink(WeightEvent::Fetching);
    }
    fetch_files(
        &hf_hub::Cache::from_env(),
        &DOWNLOAD_PERMITS,
        repo_id,
        &files,
        sink.as_ref(),
    )
    .await
}

/// Fetch `files` of `repo_id` into `cache`: cache-first, serialised per file,
/// and holding a `permits` slot only for an actual transfer.
///
/// Transfers use hf-hub's environment-configured API, whose cache is
/// `Cache::from_env()` — the cache production passes in.
async fn fetch_files(
    cache: &hf_hub::Cache,
    permits: &tokio::sync::Semaphore,
    repo_id: &str,
    files: &[&str],
    sink: Option<&ProgressSink>,
) -> Result<(), WeightError> {
    let mut api_repo: Option<hf_hub::api::tokio::ApiRepo> = None;
    for &filename in files {
        // Cache-first: hf-hub's streaming path re-downloads even a fully
        // cached file, so resolve it here and skip the network entirely.
        if reuse_cached(cache, repo_id, filename, sink) {
            continue;
        }

        // Serialise with any other in-process download of this exact file,
        // then re-check: the download we queued behind may have cached it.
        let lock = file_lock(repo_id, filename);
        let _file_guard = lock.lock().await;
        if reuse_cached(cache, repo_id, filename, sink) {
            continue;
        }

        // Bound concurrent transfers process-wide. Taken after the file lock
        // (never while waiting on one), so the two cannot deadlock.
        let _download_slot = permits
            .acquire()
            .await
            .map_err(|e| WeightError::Download(format!("download limiter closed: {e}")))?;
        let repo = match &api_repo {
            Some(repo) => repo,
            None => {
                let api = hf_hub::api::tokio::ApiBuilder::from_env()
                    .with_progress(false)
                    .build()
                    .map_err(|e| WeightError::Download(format!("hf api: {e}")))?;
                api_repo.insert(api.model(repo_id.to_string()))
            }
        };
        let fetched = match sink {
            Some(sink) => {
                let reporter = SinkProgress::new(Arc::clone(sink), filename);
                repo.download_with_progress(filename, reporter).await
            }
            None => repo.get(filename).await,
        };
        fetched
            .map_err(|e| WeightError::Download(format!("download of {filename} failed: {e}")))?;
    }
    Ok(())
}

/// Non-logging twin of the lookup `llama_bridge::resolve_mmproj_path`
/// performs for a vision entry: the configured `mmproj_path` first, then the
/// HF cache. `true` for entries that declare no projector at all.
///
/// Presence is probed for every pinned entry on every boot, so unlike the
/// loader's resolver this must not warn about a configured-but-missing
/// `mmproj_path` — an incomplete entry is exactly what the probe is for.
fn mmproj_present(entry: &RegistryEntry) -> bool {
    let Some(mmproj_filename) = entry.mmproj_filename() else {
        return true;
    };
    if entry.mmproj_path.as_deref().is_some_and(Path::exists) {
        return true;
    }
    cached_file(&entry.hf_repo_id, mmproj_filename).is_some()
}

/// Outcome of [`ensure_weights`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Acquired {
    /// The resolved GGUF.
    pub gguf: PathBuf,
    /// A fetch ran (some file was missing and the entry is downloadable).
    pub downloaded: bool,
}

/// Ensure an entry's weights are on disk and return the GGUF path.
///
/// Order: configured local path → HF cache → download (only for
/// [`WeightSource::HfFile`]). Offline mode never downloads.
pub async fn ensure_weights(
    entry: &RegistryEntry,
    offline: bool,
    sink: Option<ProgressSink>,
) -> Result<Acquired, WeightError> {
    let source = entry.weight_source();
    let fetchable = source.is_downloadable() && !offline;
    if let Ok(gguf) = crate::llama_bridge::resolve_gguf_path(entry) {
        // The GGUF resolves, so the only file that can be missing is a vision
        // entry's mmproj. Fetch it when possible (`fetch_model_files` reports
        // the cached GGUF and downloads just the projector); otherwise the
        // entry still loads text-only, exactly as a GGUF-only test would.
        if mmproj_present(entry) || !fetchable {
            return Ok(Acquired {
                gguf,
                downloaded: false,
            });
        }
        tracing::info!(
            "model '{}' is partially on disk; fetching its missing files from {}",
            entry.alias,
            entry.hf_repo_id
        );
        fetch_model_files(entry, sink).await?;
        return Ok(Acquired {
            gguf,
            downloaded: true,
        });
    }
    match source {
        WeightSource::Local(path) => {
            return Err(WeightError::LocalPathMissing {
                alias: entry.alias.clone(),
                path: path.to_path_buf(),
            })
        }
        WeightSource::HfAnyGguf { .. } => {
            return Err(WeightError::NotDownloadable {
                alias: entry.alias.clone(),
                local_path: false,
            })
        }
        WeightSource::HfFile { .. } if offline => {
            return Err(WeightError::Offline {
                alias: entry.alias.clone(),
            })
        }
        WeightSource::HfFile { repo, file } => {
            tracing::info!(
                "model '{}' is not on disk; downloading {file} from {repo}",
                entry.alias
            );
        }
    }
    fetch_model_files(entry, sink).await?;
    let gguf = crate::llama_bridge::resolve_gguf_path(entry).map_err(WeightError::Download)?;
    Ok(Acquired {
        gguf,
        downloaded: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::tokio::Progress;
    use mettle_core::model_registry::{LoaderKind, ModelRegistry};
    use std::time::Duration;

    fn entry(id: &str) -> RegistryEntry {
        let r = ModelRegistry::new();
        r.resolve(id).unwrap().clone()
    }

    /// A sink that records every event.
    fn recording_sink() -> (ProgressSink, Arc<Mutex<Vec<WeightEvent>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&events);
        let sink: ProgressSink = Arc::new(move |e| seen.lock().unwrap().push(e));
        (sink, events)
    }

    /// A minimal HF cache (`<repo>/refs/main` -> commit,
    /// `<repo>/snapshots/<commit>/<file>`) holding `files` for `repo`.
    fn cache_with(repo: &str, files: &[&str]) -> (tempfile::TempDir, hf_hub::Cache) {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir
            .path()
            .join(hf_hub::Repo::model(repo.to_string()).folder_name());
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::create_dir_all(root.join("snapshots").join("rev1")).unwrap();
        std::fs::write(root.join("refs").join("main"), "rev1").unwrap();
        for file in files {
            std::fs::write(root.join("snapshots").join("rev1").join(file), b"GGUF").unwrap();
        }
        let cache = hf_hub::Cache::new(dir.path().to_path_buf());
        (dir, cache)
    }

    #[test]
    fn only_entries_with_a_known_file_are_downloadable() {
        // A built-in alias: repo + quant file known.
        let builtin = entry("qwen3-embedding-0.6b:latest");
        let (repo, files) = download_set(&builtin).unwrap();
        assert_eq!(repo, "Qwen/Qwen3-Embedding-0.6B-GGUF");
        assert_eq!(files, vec!["Qwen3-Embedding-0.6B-Q8_0.gguf"]);
        // A vision entry fetches its projector too.
        let ocr = entry("unlimited-ocr:latest");
        assert_eq!(
            download_set(&ocr).unwrap().1,
            vec!["Unlimited-OCR-Q4_K_M.gguf", "mmproj-Unlimited-OCR-F16.gguf"]
        );
        // A configured local path is never downloaded around.
        let mut local = entry("gemma4:e2b");
        local.local_path = Some(PathBuf::from("/nonexistent/model.gguf"));
        assert_eq!(
            download_set(&local).unwrap_err(),
            WeightError::NotDownloadable {
                alias: "gemma4:e2b".into(),
                local_path: true
            }
        );
        // No named file: nothing to download.
        let mut unknown = entry("gemma4:e2b");
        unknown.hf_quant_filename = None;
        unknown.hf_repo_id = "m1".into();
        let err = download_set(&unknown).unwrap_err();
        assert!(err.is_config_error());
        assert!(err.to_string().contains("no Hugging Face source"), "{err}");
    }

    #[tokio::test]
    async fn pull_refuses_an_undownloadable_entry_before_any_io() {
        let mut local = entry("gemma4:e2b");
        local.local_path = Some(PathBuf::from("/nonexistent/model.gguf"));
        let (sink, events) = recording_sink();
        let err = fetch_model_files(&local, Some(sink)).await.unwrap_err();
        assert!(err.to_string().contains("local path"), "{err}");
        // Refused at the gate: not even the `Fetching` announcement.
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_cached_file_never_waits_for_a_download_permit() {
        let (_dir, cache) = cache_with("org/repo-GGUF", &["model.gguf"]);
        // Every permit taken: a transfer would block forever.
        let permits = tokio::sync::Semaphore::new(0);
        let (sink, events) = recording_sink();
        tokio::time::timeout(
            Duration::from_secs(5),
            fetch_files(
                &cache,
                &permits,
                "org/repo-GGUF",
                &["model.gguf"],
                Some(&sink),
            ),
        )
        .await
        .expect("cache hit must not queue behind the download cap")
        .unwrap();
        let events = events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [WeightEvent::Progress(p)] if &*p.filename == "model.gguf"
                && p.bytes_done == 4 && p.bytes_total == 4
        ));
    }

    #[tokio::test]
    async fn a_cache_miss_takes_a_permit_before_any_transfer() {
        let (_dir, cache) = cache_with("org/repo-GGUF", &["model.gguf"]);
        let permits = tokio::sync::Semaphore::new(0);
        // The cached file is reported, then the miss blocks on the permit
        // (never reaching the network).
        let (sink, events) = recording_sink();
        let blocked = tokio::time::timeout(
            Duration::from_millis(200),
            fetch_files(
                &cache,
                &permits,
                "org/repo-GGUF",
                &["model.gguf", "missing.gguf"],
                Some(&sink),
            ),
        )
        .await;
        assert!(blocked.is_err(), "a miss must wait for a download slot");
        assert_eq!(events.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn progress_is_throttled_but_keeps_first_and_last() {
        let (sink, events) = recording_sink();
        let mut reporter = SinkProgress::new(sink, "big.gguf");
        let total = 1_000_000usize;
        reporter.init(total, "big.gguf").await;
        for _ in 0..total / 100 {
            reporter.update(100).await;
        }
        reporter.finish().await;
        let events = events.lock().unwrap();
        let progress: Vec<&WeightProgress> = events
            .iter()
            .map(|e| match e {
                WeightEvent::Progress(p) => p,
                WeightEvent::Fetching => panic!("unexpected Fetching"),
            })
            .collect();
        // 10_000 hf-hub updates -> ~one event per percent.
        assert!(
            progress.len() <= PROGRESS_STEPS as usize + 2,
            "{}",
            progress.len()
        );
        assert_eq!(progress.first().unwrap().bytes_done, 0);
        let last = progress.last().unwrap();
        assert_eq!((last.bytes_done, last.bytes_total), (1_000_000, 1_000_000));
        assert!(progress
            .windows(2)
            .all(|w| w[0].bytes_done <= w[1].bytes_done));
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
        assert_eq!(
            got,
            Acquired {
                gguf: path,
                downloaded: false
            }
        );
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
        assert!(mmproj_present(&e));

        // The GGUF alone is not enough: a missing projector is detected even
        // though the weight file resolves.
        std::fs::remove_file(&mmproj).unwrap();
        assert!(!mmproj_present(&e));

        // A local-path entry has no source to fetch the mmproj from, so it
        // still degrades to the text-only GGUF instead of failing the boot.
        assert_eq!(ensure_weights(&e, true, None).await.unwrap().gguf, gguf);
    }

    #[tokio::test]
    async fn missing_local_path_is_a_config_error_not_a_download() {
        let mut e = entry("gemma4:e2b");
        e.local_path = Some(PathBuf::from("/nonexistent/model.gguf"));
        let err = ensure_weights(&e, false, None).await.unwrap_err();
        assert!(matches!(err, WeightError::LocalPathMissing { .. }));
        assert!(err.is_config_error());
        assert!(err.to_string().contains("does not exist"), "{err}");
    }

    #[tokio::test]
    async fn ensure_weights_refuses_an_entry_with_no_hf_source() {
        let mut e = entry("gemma4:e2b");
        e.local_path = None;
        e.hf_quant_filename = None;
        e.hf_repo_id = "mettle-tests/nonexistent-repo".into();
        let err = ensure_weights(&e, false, None).await.unwrap_err();
        assert!(err.to_string().contains("no Hugging Face source"), "{err}");
    }

    #[tokio::test]
    async fn offline_refuses_to_download_an_absent_model() {
        let mut e = entry("gemma4:e2b");
        e.local_path = None;
        e.hf_quant_filename = Some("something-not-cached.gguf".into());
        let err = ensure_weights(&e, true, None).await.unwrap_err();
        assert!(!err.is_config_error());
        assert!(err.to_string().contains("offline mode"), "{err}");
    }

    #[test]
    fn gguf_selection_prefers_the_advertised_quant() {
        let files = ["README.md", "model-Q4_K_M.gguf", "model-Q8_0.gguf"];
        assert_eq!(*pick_gguf(&files, "Q8_0").unwrap(), "model-Q8_0.gguf");
        // No advertised quant match -> first GGUF, never a non-GGUF file.
        assert_eq!(*pick_gguf(&files, "F16").unwrap(), "model-Q4_K_M.gguf");
        assert!(pick_gguf(&["README.md"], "Q8_0").is_none());
    }
}
