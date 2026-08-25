use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use mettle_core::engine::embed_texts;
use mettle_core::ollama_types::{
    EmbedRequest, EmbedResponse, LegacyEmbedRequest, LegacyEmbedResponse,
};

use super::error::api_error;
use crate::state::AppState;

/// POST /api/embed — batched embeddings (Ollama v0.1.14+).
pub async fn embed_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> axum::response::Response {
    let (entry, engine) = match state.engine_for(&req.model) {
        Ok(v) => v,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let inputs = req.input.into_vec();
    let started = std::time::Instant::now();

    match embed_texts(engine.as_ref(), &entry.alias, &inputs).await {
        Ok(embeddings) => Json(EmbedResponse {
            model: req.model,
            embeddings,
            total_duration: started.elapsed().as_nanos() as u64,
            load_duration: 0,
            prompt_eval_count: inputs.len() as u32,
        })
        .into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

/// POST /api/embeddings — legacy single-input form.
pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LegacyEmbedRequest>,
) -> axum::response::Response {
    let (entry, engine) = match state.engine_for(&req.model) {
        Ok(v) => v,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    match embed_texts(engine.as_ref(), &entry.alias, &[req.prompt]).await {
        Ok(mut embeddings) if !embeddings.is_empty() => Json(LegacyEmbedResponse {
            embedding: embeddings.remove(0),
        })
        .into_response(),
        Ok(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "embedding failed").into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}
