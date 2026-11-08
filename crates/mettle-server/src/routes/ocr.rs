//! `POST /v1/ocr` — image in, text out. The consumer rasterizes PDF pages
//! itself and posts base64-encoded images; the engine OCRs them with the
//! active profile's OCR slot, else the registry's default OCR model.

use std::sync::Arc;

use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use base64::Engine as _;
use mettle_core::config::ModelRole;
use mettle_core::engine::{EngineRequest, EngineResponse, RequestKind, SamplingParams};
use mettle_core::hooks::RequestCtx;
use mettle_core::ocr_types::{OcrRequest, OcrResponse, DEFAULT_OCR_INSTRUCTION};
use mettle_core::stream::{engine_accumulate, Accumulated};
use tokio::sync::mpsc;

use super::error::{api_error, api_error_code};
use crate::state::AppState;

/// POST /v1/ocr
pub async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Json(req): Json<OcrRequest>,
) -> axum::response::Response {
    // Validate the payload before touching the engine: a malformed image is
    // a client error regardless of model availability.
    let image = match base64::engine::general_purpose::STANDARD.decode(req.image) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        Ok(_) => {
            return api_error(StatusCode::BAD_REQUEST, "image is empty".to_string()).into_response()
        }
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("image is not valid base64: {e}"),
            )
            .into_response()
        }
    };

    // Not client-selectable: the active profile's highest-priority OCR slot,
    // else the registry default (503 `not_resident` unless it is loaded).
    let ocr_alias = match state.profiles.role_slot(ModelRole::Ocr) {
        Some(id) => id,
        None => match state.registry.read() {
            Ok(r) => r.default_ocr().alias.clone(),
            Err(_) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    String::from("lock poisoned"),
                )
                .into_response()
            }
        },
    };
    let (entry, sender) = match state.v1_sender_for(&ocr_alias) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let alias = entry.alias.clone();
    let model_name = alias.clone();
    let (tx, rx) = mpsc::channel::<EngineResponse>(256);
    let engine_req = EngineRequest {
        model: entry.alias,
        kind: RequestKind::Ocr {
            instruction: req
                .instruction
                .unwrap_or_else(|| DEFAULT_OCR_INSTRUCTION.to_string()),
            image,
            params: SamplingParams {
                // OCR pages can be long; cap generously.
                max_tokens: Some(8192),
                ..Default::default()
            },
            streaming: false,
        },
        response: tx,
    };
    if let Err(e) = sender.send(engine_req).await {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    match engine_accumulate(
        rx,
        state.hooks.clone(),
        ctx,
        model_name.clone(),
        |acc: Accumulated| {
            Ok(OcrResponse {
                text: acc.content,
                model: model_name,
            })
        },
    )
    .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err((status, msg)) => {
            state.profiles.note_runtime_failure(&alias, &msg);
            api_error_code(status, msg).into_response()
        }
    }
}
