//! Structured `slot_unavailable` 503 — the one shape every surface uses to
//! say "this model cannot serve right now" (profiles-contract.md §4,
//! DOCKET_RESIDENCY_FAIL_LOUD_CONTRACT.md §1.5).
//!
//! `reason` is a closed set derived from the slot's real state, never a
//! guess: consumers branch on it (retry, re-activate, surface to the user).

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

/// Seconds a consumer should wait before retrying a `loading` slot.
const LOADING_RETRY_AFTER_SECS: &str = "2";

/// Why a registry-known model cannot serve. Serialises to the wire
/// strings of [`Self::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
    /// Known model, not in the active profile, never JIT-loaded.
    NotResident,
    /// Activation or JIT load in flight — retryable.
    Loading,
    /// The planner dropped it for budget, or it was evicted after an OOM.
    DroppedVram,
    /// A load was attempted and failed.
    LoadFailed,
    /// A rerank call's per-call context could not fit even one sequence.
    RerankContextVram,
    /// No engine is loaded at all.
    NoEngine,
}

impl UnavailableReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotResident => "not_resident",
            Self::Loading => "loading",
            Self::DroppedVram => "dropped_vram",
            Self::LoadFailed => "load_failed",
            Self::RerankContextVram => "rerank_context_vram",
            Self::NoEngine => "no_engine",
        }
    }

    /// `Retry-After` seconds — the one retry policy for every surface
    /// (`slot_unavailable` 503s, `/engine/ready`, switch-in-progress).
    /// Only an in-flight load is worth an automatic retry; every other
    /// reason needs an operator or consumer action first.
    pub fn retry_after(self) -> Option<&'static str> {
        matches!(self, Self::Loading).then_some(LOADING_RETRY_AFTER_SECS)
    }
}

/// One model's unavailability, ready to be rendered as the 503 payload.
#[derive(Debug, Clone)]
pub struct Unavailable {
    pub id: String,
    pub reason: UnavailableReason,
    /// Human-readable cause (planner message, load error), when known.
    pub detail: Option<String>,
    /// Whether the current session degraded a profile (dropped/shrunk).
    pub profile_degraded: bool,
}

impl Unavailable {
    pub fn payload(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "code": "slot_unavailable",
                "reason": self.reason,
                "detail": self.detail,
                "profile_degraded": self.profile_degraded,
                "missing": [self.id],
            }
        })
    }
}

impl IntoResponse for Unavailable {
    fn into_response(self) -> Response {
        let retry_after = self.reason.retry_after();
        let resp = (StatusCode::SERVICE_UNAVAILABLE, Json(self.payload())).into_response();
        with_retry_after(resp, retry_after)
    }
}

/// Stamp `Retry-After` on `resp` when the policy asked for one.
pub fn with_retry_after(mut resp: Response, secs: Option<&'static str>) -> Response {
    if let Some(secs) = secs {
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static(secs));
    }
    resp
}
