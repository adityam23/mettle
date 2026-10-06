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

use std::path::PathBuf;
use std::sync::Arc;

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

/// Download an entry's files into the standard Hugging Face cache, where the
/// engine loader finds them. GGUF entries fetch the quantised file; entries
/// without one fetch the full safetensors set.
pub async fn fetch_model_files(
    entry: &RegistryEntry,
    sink: Option<ProgressSink>,
) -> Result<(), String> {
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
        let listing: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
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

/// Ensure an entry's weights are on disk and return the GGUF path.
///
/// Order: configured local path → HF cache → download (when the registry
/// knows a Hugging Face source). Offline mode never downloads.
pub async fn ensure_weights(
    entry: &RegistryEntry,
    offline: bool,
    sink: Option<ProgressSink>,
) -> Result<PathBuf, String> {
    if let Ok(path) = crate::llama_bridge::resolve_gguf_path(entry) {
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
