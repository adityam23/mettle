//! `/engine` routes (agent-docs/profiles-contract.md): profile inspection,
//! activation, JIT slot load/unload, device snapshot, and the event SSE
//! stream. Mounted into the shared router in `app.rs`, so bearer auth and
//! request hooks apply exactly as they do to `/v1` (contract: same layer).

use std::convert::Infallible;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::Stream;
use serde::Deserialize;
use serde_json::json;

use crate::profiles::SlotOpError;
use crate::state::AppState;
use crate::unavailable::{with_retry_after, UnavailableReason};

// ---------------------------------------------------------------------------
// GET /engine/profiles — definitions + active + fit preview
// ---------------------------------------------------------------------------

pub async fn profiles_handler(State(state): State<std::sync::Arc<AppState>>) -> Response {
    // The fit preview parses GGUF headers: keep it off the executor.
    let profiles = std::sync::Arc::clone(&state.profiles);
    match tokio::task::spawn_blocking(move || profiles.profiles_view()).await {
        Ok(view) => Json(view).into_response(),
        Err(e) => crate::routes::error::api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("profiles view failed: {e}"),
        )
        .into_response(),
    }
}

// ---------------------------------------------------------------------------
// PUT /engine/profiles/active — activate (async; completion via events)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ActivateRequest {
    pub name: String,
}

pub async fn activate_handler(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<ActivateRequest>,
) -> Response {
    if !state.profiles.profile_exists(&req.name) {
        return crate::routes::error::api_error(
            StatusCode::NOT_FOUND,
            format!("unknown profile '{}'", req.name),
        )
        .into_response();
    }

    // A second activation supersedes an in-flight one (generation bump
    // inside activate()); 202 means accepted, not finished — subscribe to
    // /engine/events for profile.active / profile.degraded.
    let manager = std::sync::Arc::clone(&state.profiles);
    let name = req.name.clone();
    tokio::spawn(async move {
        if let Err(e) = manager.activate(name).await {
            tracing::warn!("activation failed: {e}");
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "activating", "profile": req.name })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /engine/status — per-slot residency
// ---------------------------------------------------------------------------

/// `GET /engine/status` — liveness + per-slot residency. Always 200 while
/// the process is up; `ready` and `phase` carry the readiness view so a
/// consumer that only polls this endpoint can still render the phase
/// (residency contract §1.4, amendment).
pub async fn status_handler(State(state): State<std::sync::Arc<AppState>>) -> Response {
    Json(state.profiles.status()).into_response()
}

/// `GET /engine/ready` — the single readiness gate. 200 when the engine
/// can serve the active profile, 503 with the same `{ready, phase}` body
/// otherwise. A 404 means the engine predates this endpoint.
pub async fn ready_handler(State(state): State<std::sync::Arc<AppState>>) -> Response {
    let readiness = state.profiles.readiness();
    let status = if readiness.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    // Only an in-flight `loading` is worth an automatic retry; a `failed`
    // phase needs an operator action first (§1.5 semantics).
    let retry_after = readiness.retry_after();
    with_retry_after((status, Json(readiness)).into_response(), retry_after)
}

// ---------------------------------------------------------------------------
// GET /engine/device — totals, free estimate, margin
// ---------------------------------------------------------------------------

pub async fn device_handler(State(state): State<std::sync::Arc<AppState>>) -> Response {
    Json(state.profiles.device_view()).into_response()
}

// ---------------------------------------------------------------------------
// POST /engine/slots/{id}/load | unload — JIT loading outside the profile
// ---------------------------------------------------------------------------

pub async fn slot_load_handler(
    State(state): State<std::sync::Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    match state.profiles.jit_load(&id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({ "status": "loaded", "id": id })),
        )
            .into_response(),
        Err(SlotOpError::AlreadyResident) => (
            StatusCode::CONFLICT,
            Json(json!({ "error": { "code": "already_resident" } })),
        )
            .into_response(),
        Err(other) => slot_error_response(other),
    }
}

pub async fn slot_unload_handler(
    State(state): State<std::sync::Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    match state.profiles.jit_unload(&id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({ "status": "unloaded", "id": id })),
        )
            .into_response(),
        Err(SlotOpError::NotResident) => (
            StatusCode::OK,
            Json(json!({ "status": "not_loaded", "id": id })),
        )
            .into_response(),
        Err(other) => slot_error_response(other),
    }
}

/// Map [`SlotOpError`] onto the wire contract: 404 unknown, 409 pinned /
/// switching conflicts carry plain codes; 503 VRAM refusals are structured
/// (`{error:{code,reason}}`) so consumers can distinguish them from generic
/// failures. Switch-in-progress responses additionally carry `Retry-After`.
fn slot_error_response(err: SlotOpError) -> Response {
    let (status, code, reason, retry_after) = match &err {
        SlotOpError::UnknownModel => (StatusCode::NOT_FOUND, "unknown_model", None, None),
        SlotOpError::Pinned => (
            StatusCode::CONFLICT,
            "pinned",
            Some("slot is a member of the active profile"),
            None,
        ),
        SlotOpError::SwitchInProgress => (
            StatusCode::SERVICE_UNAVAILABLE,
            "switch_in_progress",
            Some("a profile activation is in flight"),
            UnavailableReason::Loading.retry_after(),
        ),
        SlotOpError::Vram(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "vram",
            Some(reason.as_str()),
            None,
        ),
        SlotOpError::LoadFailed(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "load_failed",
            Some(reason.as_str()),
            None,
        ),
        SlotOpError::AlreadyResident | SlotOpError::NotResident => {
            unreachable!("handled by callers")
        }
    };

    let response = (
        status,
        Json(json!({ "error": { "code": code, "reason": reason } })),
    )
        .into_response();
    with_retry_after(response, retry_after)
}

// ---------------------------------------------------------------------------
// GET /engine/events — SSE, replaying the last 100 on connect
// ---------------------------------------------------------------------------

/// SSE keepalive interval. Contract with docket (QUESTIONS from docket Q6):
/// consumers bound proxy-thread lifetime by it, so it is pinned here rather
/// than inherited from axum's default.
const EVENTS_KEEPALIVE: std::time::Duration = std::time::Duration::from_secs(15);

pub async fn events_handler(
    State(state): State<std::sync::Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (replay, mut rx) = state.events.subscribe();

    let stream = async_stream::stream! {
        for json in replay {
            yield Ok(Event::default().data(json.to_string()));
        }
        loop {
            match rx.recv().await {
                Ok(json) => yield Ok(Event::default().data(json.to_string())),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!("event subscriber lagged, skipped {skipped}");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(EVENTS_KEEPALIVE))
}
