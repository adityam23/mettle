use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use crate::routes::error::api_error;
use mettle_core::config::RootConfig;
use mettle_core::engine::{Engine, EngineRequest};
use mettle_core::error::CoreError;
use mettle_core::hooks::HookRegistry;
use mettle_core::model_registry::{ModelRegistry, RegistryEntry};
use mettle_core::retrieval::CollectionStore;

/// Shared engine handle: any `mettle_core::engine::Engine` implementation
/// (currently `mettle_llama::LlamaEngine`).
pub type SharedEngine = Arc<dyn Engine>;

/// Server-side counters exposed at `/metrics` in Prometheus text format.
#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub search_requests_total: AtomicU64,
    pub search_latency_us_total: AtomicU64,
    pub ingest_documents_total: AtomicU64,
    pub ingest_chunks_total: AtomicU64,
}

/// Shared application state, accessible from every route handler via axum `State`.
pub struct AppState {
    /// The embedded llama.cpp engine. Installed once at boot (the worker
    /// owns the init-once `LlamaBackend`); every later load/unload goes
    /// through `Engine::load_slot`/`remove_model`. Shared with
    /// [`crate::profiles::ProfileManager`] behind the same Arc.
    pub engine: Arc<RwLock<Option<SharedEngine>>>,
    pub registry: Arc<RwLock<ModelRegistry>>,
    pub hooks: Arc<HookRegistry>,
    pub collections: Arc<Mutex<CollectionStore>>,
    pub config: RootConfig,
    pub startup_time: Instant,
    pub metrics: Metrics,
    /// Notified when the engine becomes available (used to lazily start
    /// the OpenAI/Anthropic listener after a first pull).
    pub engine_ready: tokio::sync::Notify,
    /// Aliases currently loaded into the engine (mirrors
    /// `engine.loaded()`, updated by pulls and the profile manager).
    pub loaded_models: Arc<Mutex<Vec<String>>>,
    /// Profile lifecycle events (SSE at `/engine/events`).
    pub events: Arc<crate::events::EventBus>,
    /// Runtime-switchable model profiles (`/engine/*` routes).
    pub profiles: Arc<crate::profiles::ProfileManager>,
    next_request_id: AtomicUsize,
}

impl AppState {
    pub fn new(config: RootConfig) -> Self {
        let data_dir = config.retrieval.data_dir.clone();
        let bit_width = config.retrieval.default_bit_width;
        let mut registry = ModelRegistry::new();
        registry.apply_user_models(&config.model);
        let engine = Arc::new(RwLock::new(None));
        let registry = Arc::new(RwLock::new(registry));
        let loaded_models = Arc::new(Mutex::new(Vec::new()));
        let events = Arc::new(crate::events::EventBus::new());
        let profiles = Arc::new(crate::profiles::ProfileManager::new(
            crate::profiles::ProfileDeps {
                engine: engine.clone(),
                registry: registry.clone(),
                loaded_models: loaded_models.clone(),
            },
            config.clone(),
            events.clone(),
            crate::profiles::default_state_path(),
        ));
        Self {
            engine,
            registry,
            hooks: Arc::new(HookRegistry::new()),
            collections: Arc::new(Mutex::new(CollectionStore::new(data_dir, bit_width))),
            config,
            startup_time: Instant::now(),
            metrics: Metrics::default(),
            engine_ready: tokio::sync::Notify::new(),
            loaded_models,
            events,
            profiles,
            next_request_id: AtomicUsize::new(0),
        }
    }

    /// Current engine handle, if a model is loaded.
    pub fn engine(&self) -> Option<SharedEngine> {
        self.engine.read().expect("engine lock poisoned").clone()
    }

    /// Install (or replace) the engine and wake anything waiting on it.
    pub fn set_engine(&self, engine: SharedEngine) {
        *self.engine.write().expect("engine lock poisoned") = Some(engine);
        self.engine_ready.notify_waiters();
    }

    /// Resolve a model name to an owned registry entry.
    pub fn resolve_model(&self, name: &str) -> Result<RegistryEntry, CoreError> {
        self.registry
            .read()
            .expect("registry lock poisoned")
            .resolve(name)
            .cloned()
    }

    /// Allocate a unique request id within this process.
    pub fn next_request_id(&self) -> usize {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Resolve a model name and obtain the engine handle for it — the shared
    /// preamble of every inference route. The error carries the HTTP status
    /// the route should return (404 unknown model, 503 no engine, 500 other).
    pub fn engine_for(
        &self,
        name: &str,
    ) -> Result<(RegistryEntry, SharedEngine), (StatusCode, String)> {
        let entry = self
            .resolve_model(name)
            .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
        let engine = self.engine().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "no models loaded; run `engine pull` first".to_string(),
            )
        })?;
        if !engine.loaded().iter().any(|a| a == &entry.alias) {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "model '{}' is not loaded; run `engine pull` first",
                    entry.alias
                ),
            ));
        }
        Ok((entry, engine))
    }

    /// Resolve a model name and obtain the engine sender for it.
    pub fn sender_for(
        &self,
        name: &str,
    ) -> Result<(RegistryEntry, tokio::sync::mpsc::Sender<EngineRequest>), (StatusCode, String)>
    {
        let (entry, engine) = self.engine_for(name)?;
        Ok((entry, engine.sender()))
    }

    // -- /v1 resolution (profiles contract §4) -------------------------------

    /// Structured unavailability for a registry-known model that is not
    /// resident. The body shape is part of the profiles contract; consumers
    /// branch on `code`/`reason`/`missing`.
    fn unavailable_response(&self, id: &str, engine_loaded: bool) -> Response {
        self.profiles
            .unavailability(id, engine_loaded)
            .into_response()
    }

    /// `/v1` engine handle: like [`Self::engine_for`], but not-resident is
    /// a structured 503 response instead of a generic message. `id` must be
    /// the resolved model id ([`crate::profiles::ProfileManager::
    /// resolve_alias`] already applied by the caller).
    pub fn v1_engine_for(&self, id: &str) -> Result<(RegistryEntry, SharedEngine), Response> {
        let entry = self
            .resolve_model(id)
            .map_err(|e| api_error(StatusCode::NOT_FOUND, e.to_string()).into_response())?;
        let engine = self
            .engine()
            .ok_or_else(|| self.unavailable_response(id, false))?;
        if !engine.loaded().iter().any(|a| a == &entry.alias) {
            return Err(self.unavailable_response(id, true));
        }
        // Single touch point for the JIT TTL sweeper.
        self.profiles.touch(&entry.alias);
        Ok((entry, engine))
    }

    /// `/v1` variant of [`Self::sender_for`] with structured unavailability.
    pub fn v1_sender_for(
        &self,
        id: &str,
    ) -> Result<(RegistryEntry, tokio::sync::mpsc::Sender<EngineRequest>), Response> {
        let (entry, engine) = self.v1_engine_for(id)?;
        Ok((entry, engine.sender()))
    }
}

impl Metrics {
    pub fn observe_search(&self, latency: std::time::Duration) {
        self.search_requests_total.fetch_add(1, Ordering::Relaxed);
        self.search_latency_us_total
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }
}
