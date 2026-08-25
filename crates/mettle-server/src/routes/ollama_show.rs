use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use mettle_core::ollama_types::{ModelDetails, ShowRequest, ShowResponse};

use super::error::api_error;
use crate::state::AppState;

/// POST /api/show — model metadata and capabilities.
pub async fn show_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShowRequest>,
) -> axum::response::Response {
    let entry = match state.resolve_model(&req.model) {
        Ok(e) => e,
        Err(e) => return api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };

    let capabilities = match &entry.loader_kind {
        mettle_core::model_registry::LoaderKind::GgufVision { .. } => {
            vec!["vision".into(), "text".into()]
        }
        mettle_core::model_registry::LoaderKind::Embedding { .. } => {
            vec!["embedding".into()]
        }
        mettle_core::model_registry::LoaderKind::Rerank { .. } => {
            vec!["rerank".into(), "embedding".into()]
        }
        _ => vec!["text".into()],
    };

    let details = ModelDetails {
        parent_model: String::new(),
        format: entry.metadata.format.clone(),
        family: entry.metadata.family.clone(),
        families: entry.metadata.families.clone(),
        parameter_size: entry.metadata.parameter_size.clone(),
        quantization_level: entry.metadata.quantization.clone(),
    };

    let vram_note = if entry.alias == "gemma4:26b-a4b" {
        "Exceeds 8 GB VRAM budget without offloading. All expert weights resident (~26B) even though ~4B active per token."
    } else {
        ""
    };

    Json(ShowResponse {
        modelfile: format!(
            "# Modelfile for {}\n# Managed by engine registry\n",
            entry.alias
        ),
        parameters: String::new(),
        template: String::new(),
        details,
        model_info: serde_json::json!({
            "hf_repo_id": entry.hf_repo_id,
            "revision": entry.revision,
            "vram_note": vram_note,
        }),
        capabilities,
    })
    .into_response()
}
