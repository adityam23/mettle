use std::sync::Arc;

use axum::{extract::State, Json};
use mettle_core::ollama_types::{
    ModelDetails, PsModelInfo, PsResponse, TagModelInfo, TagsResponse,
};

use crate::state::AppState;

/// Build the Ollama `details` block from a registry entry's metadata.
fn details(e: &mettle_core::model_registry::RegistryEntry) -> ModelDetails {
    ModelDetails {
        parent_model: String::new(),
        format: e.metadata.format.clone(),
        family: e.metadata.family.clone(),
        families: e.metadata.families.clone(),
        parameter_size: e.metadata.parameter_size.clone(),
        quantization_level: e.metadata.quantization.clone(),
    }
}

/// GET /api/tags — list all registered models in Ollama's response shape.
pub async fn tags_handler(State(state): State<Arc<AppState>>) -> Json<TagsResponse> {
    let registry = state.registry.read().expect("registry lock poisoned");

    let models: Vec<TagModelInfo> = registry
        .list()
        .iter()
        .map(|e| TagModelInfo {
            name: e.alias.clone(),
            model: e.alias.clone(),
            modified_at: String::new(),
            size: 0,
            digest: String::new(),
            details: details(e),
        })
        .collect();

    Json(TagsResponse { models })
}

/// GET /api/ps — list models currently loaded into the engine.
pub async fn ps_handler(State(state): State<Arc<AppState>>) -> Json<PsResponse> {
    // The engine's own resident list is the single source of truth.
    let loaded: Vec<String> = state.engine().map(|e| e.loaded()).unwrap_or_default();
    let registry = state.registry.read().expect("registry lock poisoned");

    let models: Vec<PsModelInfo> = registry
        .list()
        .iter()
        .filter(|e| loaded.iter().any(|alias| alias == &e.alias))
        .map(|e| PsModelInfo {
            name: e.alias.clone(),
            model: e.alias.clone(),
            size: 0,
            digest: String::new(),
            details: details(e),
            expires_at: String::new(),
            size_vram: 0,
        })
        .collect();

    Json(PsResponse { models })
}
