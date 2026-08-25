use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use super::error::api_error;
use crate::state::AppState;
use mettle_core::model_registry::RegistryEntry;
use mettle_core::ollama_types::{
    CopyRequest, CreateRequest, DeleteRequest, PullProgress, PushRequest, VersionResponse,
};

// ---------------------------------------------------------------------------
// GET / — health check
// ---------------------------------------------------------------------------

pub async fn health() -> &'static str {
    "engine is running"
}

// ---------------------------------------------------------------------------
// GET /api/version
// ---------------------------------------------------------------------------

pub async fn version(State(_state): State<Arc<AppState>>) -> Json<VersionResponse> {
    Json(VersionResponse {
        version: format!("engine {}", env!("CARGO_PKG_VERSION")),
    })
}

// ---------------------------------------------------------------------------
// GET /metrics — Prometheus text exposition (spec §9)
// ---------------------------------------------------------------------------

pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use std::fmt::Write;
    use std::sync::atomic::Ordering;

    let m = &state.metrics;
    let mut out = String::new();

    let _ = writeln!(out, "# TYPE engine_requests_total counter");
    let _ = writeln!(
        out,
        "engine_requests_total {}",
        m.requests_total.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# TYPE engine_search_requests_total counter");
    let _ = writeln!(
        out,
        "engine_search_requests_total {}",
        m.search_requests_total.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        out,
        "# TYPE engine_search_latency_microseconds_total counter"
    );
    let _ = writeln!(
        out,
        "engine_search_latency_microseconds_total {}",
        m.search_latency_us_total.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# TYPE engine_ingest_documents_total counter");
    let _ = writeln!(
        out,
        "engine_ingest_documents_total {}",
        m.ingest_documents_total.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# TYPE engine_ingest_chunks_total counter");
    let _ = writeln!(
        out,
        "engine_ingest_chunks_total {}",
        m.ingest_chunks_total.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# TYPE engine_uptime_seconds gauge");
    let _ = writeln!(
        out,
        "engine_uptime_seconds {}",
        state.startup_time.elapsed().as_secs()
    );

    // Per-collection index sizes.
    if let Ok(store) = state.collections.lock() {
        let _ = writeln!(out, "# TYPE engine_collection_chunks gauge");
        for name in store.list() {
            if let Some(handle) = store.get(&name) {
                if let Ok(coll) = handle.lock() {
                    let _ = writeln!(
                        out,
                        "engine_collection_chunks{{collection=\"{}\"}} {}",
                        name,
                        coll.len()
                    );
                }
            }
        }
    }

    let _ = writeln!(out, "# TYPE engine_model_loaded gauge");
    let _ = writeln!(
        out,
        "engine_model_loaded {}",
        u8::from(state.engine().is_some())
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        out,
    )
}

// ---------------------------------------------------------------------------
// POST /api/pull — download a model into the HF cache and load it.
// ---------------------------------------------------------------------------

/// A status-only progress event (no byte counts).
fn progress(status: impl Into<String>) -> PullProgress {
    PullProgress {
        status: status.into(),
        digest: None,
        total: None,
        completed: None,
    }
}

/// A progress event with byte counts for an in-flight download.
fn progress_bytes(status: String, total: u64, completed: u64) -> PullProgress {
    PullProgress {
        status,
        digest: None,
        total: Some(total),
        completed: Some(completed),
    }
}

pub async fn pull_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<mettle_core::ollama_types::PullRequest>,
) -> axum::response::Response {
    if state.config.server.offline {
        return api_error(
            StatusCode::FORBIDDEN,
            "offline mode: network egress (including model pulls) is disabled",
        )
        .into_response();
    }

    let entry = match state.resolve_model(&req.model) {
        Ok(e) => e,
        Err(e) => return api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };

    let is_streaming = req.stream.unwrap_or(true);

    if is_streaming {
        let stream = pull_stream(state, entry);
        crate::response_stream::NdJsonStream::new(stream).into_response()
    } else {
        match download_model_files(&entry, None).await {
            Ok(()) => match ensure_loaded(&state, &entry.alias).await {
                Ok(()) => Json(progress("success")).into_response(),
                Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
            },
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        }
    }
}

fn pull_stream(
    state: Arc<AppState>,
    entry: RegistryEntry,
) -> impl futures::Stream<Item = Result<PullProgress, String>> {
    async_stream::stream! {
        yield Ok(progress(format!("pulling manifest for {}", entry.alias)));

        // Run downloads on a task, forwarding progress events as they arrive.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PullProgress>();
        let dl_entry = entry.clone();
        let mut download =
            tokio::spawn(async move { download_model_files(&dl_entry, Some(tx)).await });

        loop {
            tokio::select! {
                event = rx.recv() => match event {
                    Some(p) => yield Ok(p),
                    None => break, // downloads finished; join below
                },
                result = &mut download => {
                    // Drain remaining progress, then handle the result.
                    while let Ok(p) = rx.try_recv() {
                        yield Ok(p);
                    }
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => { yield Err(e); return; }
                        Err(e) => { yield Err(format!("download task failed: {}", e)); return; }
                    }
                    yield Ok(progress("loading model"));
                    if let Err(e) = ensure_loaded(&state, &entry.alias).await {
                        yield Err(e);
                        return;
                    }
                    yield Ok(progress("success"));
                    return;
                }
            }
        }

        // Channel closed before the task completed (all progress drained).
        match download.await {
            Ok(Ok(())) => {
                yield Ok(progress("loading model"));
                if let Err(e) = ensure_loaded(&state, &entry.alias).await {
                    yield Err(e);
                    return;
                }
                yield Ok(progress("success"));
            }
            Ok(Err(e)) => yield Err(e),
            Err(e) => yield Err(format!("download task failed: {}", e)),
        }
    }
}

/// NDJSON progress reporter for hf-hub downloads.
#[derive(Clone)]
struct NdjsonProgress {
    tx: tokio::sync::mpsc::UnboundedSender<PullProgress>,
    filename: String,
    total: u64,
    completed: u64,
}

impl hf_hub::api::tokio::Progress for NdjsonProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.filename = filename.to_string();
        self.total = size as u64;
        self.completed = 0;
        let _ = self.tx.send(progress_bytes(
            format!("downloading {}", self.filename),
            self.total,
            0,
        ));
    }

    async fn update(&mut self, size: usize) {
        self.completed += size as u64;
        let _ = self.tx.send(progress_bytes(
            format!("downloading {}", self.filename),
            self.total,
            self.completed,
        ));
    }

    async fn finish(&mut self) {
        let _ = self.tx.send(progress_bytes(
            format!("verifying {}", self.filename),
            self.total,
            self.total,
        ));
    }
}

/// Download a model's files into the standard Hugging Face cache, where the
/// engine loader finds them. GGUF entries fetch the quantised file; other
/// entries fetch the full safetensors set.
async fn download_model_files(
    entry: &RegistryEntry,
    progress: Option<tokio::sync::mpsc::UnboundedSender<PullProgress>>,
) -> Result<(), String> {
    let api = hf_hub::api::tokio::ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| format!("hf api: {}", e))?;
    let repo = api.model(entry.hf_repo_id.clone());

    let filenames: Vec<String> = if let Some(quant) = &entry.hf_quant_filename {
        vec![quant.clone()]
    } else {
        let info = repo
            .info()
            .await
            .map_err(|e| format!("failed to fetch repo info for {}: {}", entry.hf_repo_id, e))?;
        info.siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|f| {
                f.ends_with(".safetensors")
                    || f.ends_with(".json")
                    || f.ends_with(".jinja")
                    || f.ends_with(".model")
            })
            .collect()
    };

    if filenames.is_empty() {
        return Err(format!("no downloadable files in {}", entry.hf_repo_id));
    }

    for filename in &filenames {
        match &progress {
            Some(tx) => {
                let reporter = NdjsonProgress {
                    tx: tx.clone(),
                    filename: filename.clone(),
                    total: 0,
                    completed: 0,
                };
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

/// Load a freshly pulled model into the running engine. The engine worker
/// exists from boot on (profiles contract: no implicit default profile);
/// pulls add to it like any other load.
async fn ensure_loaded(state: &Arc<AppState>, alias: &str) -> Result<(), String> {
    let entry = state.resolve_model(alias).map_err(|e| e.to_string())?;
    let spec = crate::llama_bridge::load_spec_for_route(&entry)?;

    let engine = state
        .engine()
        .ok_or_else(|| "engine not initialised".to_string())?;
    tokio::task::spawn_blocking(move || engine.load_model(spec))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    if let Ok(mut loaded) = state.loaded_models.lock() {
        if !loaded.iter().any(|a| a == alias) {
            loaded.push(alias.to_string());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DELETE /api/delete — remove a model from the registry, engine, and cache.
// ---------------------------------------------------------------------------

pub async fn delete_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteRequest>,
) -> axum::response::Response {
    let removed = {
        let mut registry = match state.registry.write() {
            Ok(r) => r,
            Err(_) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned")
                    .into_response()
            }
        };
        match registry.remove(&req.model) {
            Ok(entry) => entry,
            Err(e) => return api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
        }
    };

    // Unload from the engine if resident.
    if let Some(engine) = state.engine() {
        if let Err(e) = engine.remove_model(&removed.alias) {
            tracing::debug!("engine remove_model: {}", e);
        }
    }
    if let Ok(mut loaded) = state.loaded_models.lock() {
        loaded.retain(|a| a != &removed.alias);
    }

    // Remove cached weights if no other registry entry references the repo.
    let repo_still_used = state
        .registry
        .read()
        .map(|r| r.list().iter().any(|e| e.hf_repo_id == removed.hf_repo_id))
        .unwrap_or(true);
    if !repo_still_used {
        let cache = hf_hub::Cache::from_env();
        let repo_path = cache
            .path()
            .join(hf_hub::Repo::model(removed.hf_repo_id.clone()).folder_name());
        if repo_path.exists() {
            if let Err(e) = std::fs::remove_dir_all(&repo_path) {
                tracing::warn!(
                    "failed to remove cached weights {}: {}",
                    repo_path.display(),
                    e
                );
            }
        }
    }

    // Ollama returns 200 with no body on success.
    StatusCode::OK.into_response()
}

// ---------------------------------------------------------------------------
// POST /api/copy — duplicate a registry entry under a new alias.
// ---------------------------------------------------------------------------

pub async fn copy_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CopyRequest>,
) -> axum::response::Response {
    let mut registry = match state.registry.write() {
        Ok(r) => r,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    match registry.copy(&req.source, &req.destination) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    }
}

// ---------------------------------------------------------------------------
// POST /api/create — stub: Modelfile creation not supported in v0.
// ---------------------------------------------------------------------------

pub async fn create_handler(Json(_req): Json<CreateRequest>) -> axum::response::Response {
    api_error(
        StatusCode::NOT_IMPLEMENTED,
        "Modelfile creation is not supported. Configure models via config.toml [[model]] entries.",
    )
    .into_response()
}

// ---------------------------------------------------------------------------
// POST /api/push — stub: Push not supported.
// ---------------------------------------------------------------------------

pub async fn push_handler(Json(_req): Json<PushRequest>) -> axum::response::Response {
    api_error(StatusCode::NOT_IMPLEMENTED, "Push is not supported.").into_response()
}
