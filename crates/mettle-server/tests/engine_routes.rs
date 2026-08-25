//! HTTP-level tests for the `/engine/*` profile surface: wire shapes,
//! auth layering, the refusal path end-to-end through the router, and JIT
//! slot error mapping (agent-docs/profiles-handoff.md §7 route tests).
//!
//! No llama worker is started here: every exercised branch happens before
//! an engine handle is required, which is exactly the surface that must
//! stay correct without a GPU.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mettle_core::config::{ModelEntry, ProfileConfig, ProfileSlot, RootConfig};
use mettle_server::app::build_router;
use mettle_server::state::AppState;
use http_body_util::BodyExt;
use tower::ServiceExt;

fn profile(name: &str, slots: &[&str]) -> ProfileConfig {
    ProfileConfig {
        name: name.to_string(),
        slots: slots
            .iter()
            .map(|id| ProfileSlot {
                id: (*id).to_string(),
                priority: None,
                placement: None,
            })
            .collect(),
    }
}

fn test_state(
    data_dir: &std::path::Path,
    models: Vec<ModelEntry>,
    profiles: Vec<ProfileConfig>,
) -> Arc<AppState> {
    let mut config = RootConfig::default();
    config.retrieval.data_dir = data_dir.to_path_buf();
    config.model = models;
    config.profile = profiles;
    Arc::new(AppState::new(config))
}

async fn send(
    router: &axum::Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let req = match body {
        Some(json) => Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(json.to_string()))
            .unwrap(),
        None => Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap(),
    };
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// Status-only request for streaming endpoints (an SSE body never
/// completes, so it must not be collected).
async fn send_status_only(router: &axum::Router, method: &str, uri: &str) -> StatusCode {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    router.clone().oneshot(req).await.unwrap().status()
}

/// Metadata-only v3 GGUF (same recipe as `profiles.rs` unit tests): enough
/// real KV geometry for `estimate_vram` to succeed against an empty body.
fn write_synthetic_gguf(path: &std::path::Path) {
    const TYPE_STRING: u32 = 8;
    const TYPE_UINT32: u32 = 4;
    let arch = "qwen3";
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensors
    buf.extend_from_slice(&8u64.to_le_bytes()); // kv pairs
    let put_str = |buf: &mut Vec<u8>, s: &str| {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    };
    put_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&TYPE_STRING.to_le_bytes());
    put_str(&mut buf, arch);
    for (k, v) in [
        (format!("{arch}.block_count"), 36u32),
        (format!("{arch}.embedding_length"), 2560),
        (format!("{arch}.attention.head_count"), 32),
        (format!("{arch}.attention.head_count_kv"), 8),
        (format!("{arch}.attention.key_length"), 128),
        (format!("{arch}.attention.value_length"), 128),
        (format!("{arch}.vocab_size"), 151_936),
    ] {
        put_str(&mut buf, &k);
        buf.extend_from_slice(&TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

// ---------------------------------------------------------------------------
// GET /engine/profiles
// ---------------------------------------------------------------------------

#[tokio::test]
async fn profiles_lists_definitions_with_active_null() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), vec![profile("finance", &["qwen"])]);
    let router = build_router(state);

    let (status, json) = send(&router, "GET", "/engine/profiles", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["active"], serde_json::Value::Null);
    let profiles = json["profiles"].as_array().unwrap();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0]["name"], "finance");
    assert_eq!(profiles[0]["slots"][0]["id"], "qwen");
    assert_eq!(profiles[0]["slots"][0]["role"], "chat");
    assert_eq!(profiles[0]["slots"][0]["placement"], "gpu");
    // Slot unresolvable in this fixture: the fit preview degrades to null
    // rather than reporting a number it cannot stand behind.
    assert!(profiles[0]["slots"][0]["estimated_bytes"].is_null());
    assert!(profiles[0]["estimated_bytes"].is_null());
}

// ---------------------------------------------------------------------------
// GET /engine/status
// ---------------------------------------------------------------------------

#[tokio::test]
async fn status_starts_idle_with_no_slots() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), Vec::new());
    let router = build_router(state);

    let (status, json) = send(&router, "GET", "/engine/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["active"], serde_json::Value::Null);
    assert_eq!(json["switching"], false);
    assert_eq!(json["degraded"]["dropped"].as_array().unwrap().len(), 0);
    assert_eq!(json["degraded"]["shrunk"].as_array().unwrap().len(), 0);
    assert_eq!(json["slots"].as_array().unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// PUT /engine/profiles/active
// ---------------------------------------------------------------------------

#[tokio::test]
async fn activate_unknown_profile_returns_404() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), vec![profile("real", &["m1"])]);
    let router = build_router(state);

    let (status, json) = send(
        &router,
        "PUT",
        "/engine/profiles/active",
        Some(serde_json::json!({"name": "nope"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].is_string(), "{json}");
}

#[tokio::test]
async fn activate_refusal_is_visible_through_status_and_events() {
    let tmp = tempfile::TempDir::new().unwrap();
    // 'm1' exists nowhere: the ladder's first slot drops, idx == 0 ⇒ the
    // whole activation is refused.
    let state = test_state(tmp.path(), Vec::new(), vec![profile("big", &["m1"])]);
    let router = build_router(state.clone());

    let (status, json) = send(
        &router,
        "PUT",
        "/engine/profiles/active",
        Some(serde_json::json!({"name": "big"})),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(json["status"], "activating");

    // Completion arrives asynchronously; poll for the degradation record
    // (the durable observable outcome of this refusal).
    let mut refused = false;
    for _ in 0..500 {
        let (_, sj) = send(&router, "GET", "/engine/status", None).await;
        let dropped = sj["degraded"]["dropped"].as_array().unwrap();
        if dropped.iter().any(|v| v == "m1") {
            refused = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        refused,
        "activation refusal never surfaced in /engine/status"
    );

    let (_, sj) = send(&router, "GET", "/engine/status", None).await;
    assert_eq!(
        sj["active"],
        serde_json::Value::Null,
        "refusal keeps no profile"
    );
    let (replay, _rx) = state.events.subscribe();
    assert!(
        replay.iter().any(|j| j.contains("profile.activating")),
        "{replay:?}"
    );
    assert!(
        replay.iter().any(|j| j.contains("profile.degraded")),
        "{replay:?}"
    );
}

// ---------------------------------------------------------------------------
// POST /engine/slots/{id}/load | unload
// ---------------------------------------------------------------------------

#[tokio::test]
async fn jit_load_unknown_model_maps_to_structured_404() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), Vec::new());
    let router = build_router(state);

    let (status, json) = send(&router, "POST", "/engine/slots/ghost/load", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "unknown_model", "{json}");
}

#[tokio::test]
async fn jit_unload_not_resident_is_a_clean_noop() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf = tmp.path().join("known.gguf");
    write_synthetic_gguf(&gguf);
    let models = vec![ModelEntry {
        id: "known".into(),
        default: false,
        path: Some(gguf),
        n_gpu_layers: None,
        mmproj_path: None,
        role: None,
        min_ctx: None,
        draft: None,
    }];
    let state = test_state(tmp.path(), models, vec![profile("p", &["known"])]);
    let router = build_router(state);

    let (status, json) = send(&router, "POST", "/engine/slots/known/unload", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "not_loaded", "{json}");
}

#[tokio::test]
async fn jit_load_without_engine_fails_structured_503_and_marks_slot_failed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf = tmp.path().join("known.gguf");
    write_synthetic_gguf(&gguf);
    let models = vec![ModelEntry {
        id: "known".into(),
        default: false,
        path: Some(gguf),
        n_gpu_layers: None,
        mmproj_path: None,
        role: None,
        min_ctx: None,
        draft: None,
    }];
    let state = test_state(tmp.path(), models, Vec::new());
    let router = build_router(state);

    // Fit checks pass (synthetic model is tiny); the missing engine handle
    // turns the load into a structured 503, never a silent 200.
    let (status, json) = send(&router, "POST", "/engine/slots/known/load", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{json}");
    assert_eq!(json["error"]["code"], "load_failed", "{json}");
    assert!(json["error"]["reason"].is_string(), "{json}");

    let (_, sj) = send(&router, "GET", "/engine/status", None).await;
    let slots = sj["slots"].as_array().unwrap();
    let known = slots.iter().find(|s| s["id"] == "known").unwrap();
    assert_eq!(known["state"], "failed", "{sj}");
    assert!(known["reason"].is_string(), "{sj}");
}

// ---------------------------------------------------------------------------
// GET /engine/device
// ---------------------------------------------------------------------------

#[tokio::test]
async fn device_endpoint_reports_shape() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), Vec::new());
    let router = build_router(state);

    let (status, json) = send(&router, "GET", "/engine/device", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["available"].is_boolean(), "{json}");
    if json["available"] == true {
        assert!(json["total_bytes"].is_u64(), "{json}");
        assert!(json["free_bytes"].is_u64(), "{json}");
        assert!(json["margin_bytes"].is_u64(), "{json}");
    } else {
        assert!(json["backend"].is_null(), "{json}");
    }
}

// ---------------------------------------------------------------------------
// GET /engine/events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn events_endpoint_streams_sse() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), Vec::new());
    let router = build_router(state);

    let req = Request::builder()
        .method("GET")
        .uri("/engine/events")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()["content-type"],
        "text/event-stream",
        "SSE content type is part of the consumer contract"
    );
}

// ---------------------------------------------------------------------------
// Auth applies to the whole /engine surface
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bearer_auth_enforced_on_engine_routes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = RootConfig::default();
    config.retrieval.data_dir = tmp.path().to_path_buf();
    config.server.api_keys = vec!["sk-secret".into()];
    config.profile = vec![profile("p", &["m1"])];
    let state = Arc::new(AppState::new(config));
    let router = build_router(state);

    for (method, uri) in [
        ("GET", "/engine/status"),
        ("GET", "/engine/profiles"),
        ("GET", "/engine/device"),
        ("POST", "/engine/slots/m1/load"),
        ("POST", "/engine/slots/m1/unload"),
    ] {
        let code = send_status_only(&router, method, uri).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED, "{method} {uri}");
    }

    // Streaming endpoint: assert status without draining the body.
    let code = send_status_only(&router, "GET", "/engine/events").await;
    assert_eq!(code, StatusCode::UNAUTHORIZED);

    // Mutations reject bad tokens too.
    let req = Request::builder()
        .method("PUT")
        .uri("/engine/profiles/active")
        .header("content-type", "application/json")
        .header("authorization", "Bearer wrong")
        .body(Body::from("{\"name\":\"p\"}"))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Correct token passes through to the handler.
    let code = {
        let req = Request::builder()
            .method("GET")
            .uri("/engine/status")
            .header("authorization", "Bearer sk-secret")
            .body(Body::empty())
            .unwrap();
        router.clone().oneshot(req).await.unwrap().status()
    };
    assert_eq!(code, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// /v1 integration (profiles contract §4)
// ---------------------------------------------------------------------------

fn chat_request(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })
}

#[tokio::test]
async fn v1_chat_for_unloaded_model_is_structured_503_with_retry_after() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf = tmp.path().join("m1.gguf");
    write_synthetic_gguf(&gguf);
    let models = vec![ModelEntry {
        id: "m1".into(),
        default: false,
        path: Some(gguf),
        n_gpu_layers: None,
        mmproj_path: None,
        role: Some(mettle_core::config::ModelRole::Chat),
        min_ctx: None,
        draft: None,
    }];
    let state = test_state(tmp.path(), models, vec![profile("p", &["m1"])]);
    let router = build_router(state);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_request("m1").to_string()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        resp.headers().get("retry-after").is_some(),
        "structured unavailability carries Retry-After"
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "slot_unavailable", "{json}");
    assert_eq!(json["error"]["reason"], "vram");
    assert_eq!(json["error"]["missing"][0], "m1");
    // Nothing degraded yet: no profile ever activated in this session.
    assert_eq!(json["error"]["profile_degraded"], false);
}

#[tokio::test]
async fn v1_chat_after_refused_activation_reports_degradation() {
    let tmp = tempfile::TempDir::new().unwrap();
    // m1 is registered but nothing can load it here (no llama worker in
    // this harness): activating 'p' walks the ladder and refuses on its
    // only slot.
    let gguf = tmp.path().join("m1.gguf");
    write_synthetic_gguf(&gguf);
    let models = vec![ModelEntry {
        id: "m1".into(),
        default: false,
        path: Some(gguf),
        n_gpu_layers: None,
        mmproj_path: None,
        role: None,
        min_ctx: None,
        draft: None,
    }];
    let state = test_state(tmp.path(), models, vec![profile("p", &["m1"])]);
    let router = build_router(state.clone());

    let (status, _) = send(
        &router,
        "PUT",
        "/engine/profiles/active",
        Some(serde_json::json!({"name": "p"})),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    for _ in 0..500 {
        let (_, sj) = send(&router, "GET", "/engine/status", None).await;
        if sj["degraded"]["dropped"]
            .as_array()
            .is_some_and(|d| d.iter().any(|v| v == "m1"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let (_, json) = send(
        &router,
        "POST",
        "/v1/chat/completions",
        Some(chat_request("m1")),
    )
    .await;
    assert_eq!(json["error"]["code"], "slot_unavailable", "{json}");
    assert_eq!(json["error"]["profile_degraded"], true);
}

#[tokio::test]
async fn v1_role_alias_without_resolution_stays_a_generic_404() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path(), Vec::new(), vec![profile("p", &["m1"])]);
    let router = build_router(state);

    // No active profile ⇒ "role:chat" cannot resolve ⇒ passthrough ⇒
    // unknown-model lookup fails exactly as an explicit bogus id would.
    let (status, json) = send(
        &router,
        "POST",
        "/v1/chat/completions",
        Some(chat_request("role:chat")),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].is_string(), "{json}");
}

#[tokio::test]
async fn models_entries_carry_residency_and_vram_fields() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf = tmp.path().join("m1.gguf");
    write_synthetic_gguf(&gguf);
    let models = vec![ModelEntry {
        id: "m1".into(),
        default: false,
        path: Some(gguf),
        n_gpu_layers: None,
        mmproj_path: None,
        role: None,
        min_ctx: None,
        draft: None,
    }];
    let state = test_state(tmp.path(), models, Vec::new());
    let router = build_router(state);

    let (_, json) = send(&router, "GET", "/v1/models", None).await;
    let data = json["data"].as_array().unwrap();
    assert!(!data.is_empty());
    for entry in data {
        assert_eq!(entry["loaded"], false, "{entry}");
    }
    // Registered model with a readable header: planner estimate present.
    let m1 = data.iter().find(|e| e["id"] == "m1").unwrap();
    assert!(m1["vram_bytes"].is_u64(), "{m1}");
    // Built-in aliases without local weights: estimate unavailable, null.
    let ghost = data.iter().find(|e| e["id"] != "m1").unwrap();
    assert!(ghost["vram_bytes"].is_null(), "{ghost}");
}
