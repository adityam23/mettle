//! Wiring that invokes the [`EngineHook`] contract at all four sites, plus
//! the bearer-auth middleware.
//!
//! # Hook sites
//! 1. **on_request** — axum middleware applied to every route.
//! 2. **on_token** — inside streaming response loops, per generated token.
//! 3. **on_response** — after building the final response, before sending.
//! 4. **on_retrieval** — after collecting search hits, before generation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use mettle_core::hooks::{HookOutcome, RequestCtx, RequestId};

use crate::state::AppState;

/// axum middleware that runs `on_request` hooks before every request.
/// Inserts the `RequestCtx` into request extensions for downstream use and
/// stamps `x-request-id` on every response (spec §3 / §7 guarantee 5).
pub async fn request_hook_middleware(
    State(state): State<Arc<AppState>>,
    mut req: Request,
    next: Next,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let model_id = req
        .headers()
        .get("x-model-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let mut ctx = RequestCtx {
        id: RequestId::new(),
        caller_identity: extract_bearer(&req),
        model_id,
        collection_ids: Vec::new(),
        metadata: HashMap::new(),
        start_time: SystemTime::now(),
    };
    let request_id = ctx.id.to_string();

    let mut response = match state.hooks.on_request(&mut ctx) {
        HookOutcome::Continue => {
            req.extensions_mut().insert(ctx);
            next.run(req).await
        }
        HookOutcome::Abort {
            status_code,
            message,
        } => {
            let status = axum::http::StatusCode::from_u16(status_code)
                .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
            (status, axum::Json(serde_json::json!({"error": message}))).into_response()
        }
    };

    if let Ok(value) = axum::http::HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

/// Bearer-token auth: when `server.api_keys` is non-empty, every request
/// must carry `Authorization: Bearer <key>` with a configured key.
/// Empty list = local trust (spec §8/§11).
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let keys = &state.config.server.api_keys;
    if keys.is_empty() {
        return next.run(req).await;
    }

    match extract_bearer(&req) {
        Some(token) if keys.iter().any(|k| k == &token) => next.run(req).await,
        _ => (
            axum::http::StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "missing or invalid API key"})),
        )
            .into_response(),
    }
}

/// Extract a Bearer token from the Authorization header, if present.
fn extract_bearer(req: &Request) -> Option<String> {
    req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_bearer_from_request() {
        let req = Request::builder()
            .header("authorization", "Bearer sk-test-123")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_bearer(&req), Some("sk-test-123".into()));
    }

    #[test]
    fn extract_bearer_missing() {
        let req = Request::builder().body(axum::body::Body::empty()).unwrap();
        assert_eq!(extract_bearer(&req), None);
    }
}
