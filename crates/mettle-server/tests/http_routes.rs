//! HTTP-level integration tests: build the real axum router and drive it
//! with requests, asserting status codes and response bodies. This is what
//! actually verifies the endpoints respond — the serde shape tests in
//! `ollama_conformance.rs` never touch the router.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use mettle_core::config::RootConfig;
use mettle_server::app::build_router;
use mettle_server::state::AppState;
use tower::ServiceExt;

fn test_state(data_dir: &std::path::Path) -> Arc<AppState> {
    let mut config = RootConfig::default();
    config.retrieval.data_dir = data_dir.to_path_buf();
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

#[tokio::test]
async fn health_and_version() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, _) = send(&router, "GET", "/", None).await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = send(&router, "GET", "/api/version", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["version"].as_str().unwrap().contains("engine"));
}

#[tokio::test]
async fn tags_ps_show() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, body) = send(&router, "GET", "/api/tags", None).await;
    assert_eq!(status, StatusCode::OK);
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty(), "registry aliases should be listed");
    assert!(models.iter().any(|m| m["name"] == "gemma4:e4b"));

    let (status, body) = send(&router, "GET", "/api/ps", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["models"].is_array());

    let (status, body) = send(
        &router,
        "POST",
        "/api/show",
        Some(serde_json::json!({"model": "gemma4"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["details"]["family"], "gemma");

    let (status, body) = send(
        &router,
        "POST",
        "/api/show",
        Some(serde_json::json!({"model": "no-such-model"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn chat_without_model_returns_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, body) = send(
        &router,
        "POST",
        "/api/chat",
        Some(serde_json::json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "hi"}]
        })),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn generate_and_embed_without_model() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    // /api/generate: 503 without an engine, 404 for unknown models.
    let (status, body) = send(
        &router,
        "POST",
        "/api/generate",
        Some(serde_json::json!({"model": "gemma4", "prompt": "hi"})),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"].is_string());

    let (status, _) = send(
        &router,
        "POST",
        "/api/generate",
        Some(serde_json::json!({"model": "no-such:model", "prompt": "hi"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // /api/embed (batched) and /api/embeddings (legacy single).
    let (status, _) = send(
        &router,
        "POST",
        "/api/embed",
        Some(serde_json::json!({"model": "embed-gemma", "input": ["a", "b"]})),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

    let (status, _) = send(
        &router,
        "POST",
        "/api/embeddings",
        Some(serde_json::json!({"model": "embed-gemma", "prompt": "a"})),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

    let (status, _) = send(
        &router,
        "POST",
        "/api/embed",
        Some(serde_json::json!({"model": "no-such:model", "input": "a"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn create_and_push_are_structured_stubs() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, body) = send(
        &router,
        "POST",
        "/api/create",
        Some(serde_json::json!({"model": "x"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(body["error"].is_string());

    let (status, body) = send(
        &router,
        "POST",
        "/api/push",
        Some(serde_json::json!({"model": "x"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn collection_crud_ingest_search_delete() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    // Create.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // Duplicate collection → 409.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);

    // List.
    let (status, body) = send(&router, "GET", "/v1/collections", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, serde_json::json!(["docs"]));

    // Ingest.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({
            "id": "doc1",
            "content": "machine learning is a subset of artificial intelligence"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["chunks"].as_u64().unwrap() >= 1);

    // Duplicate document → 409.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({"id": "doc1", "content": "again"})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);

    // Keyword search finds it (with punctuation that must not 500).
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "machine learning?!", "mode": "keyword", "k": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = body.as_array().unwrap();
    assert!(!hits.is_empty(), "keyword search should match");
    assert_eq!(hits[0]["document_id"], "doc1");

    // Second doc with token-disjoint content (for OR-mode + hybrid pins).
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({
            "id": "doc2",
            "content": "quantum computing breakthroughs accelerate everything"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // OR-mode keyword (adoption contract §1.1): one matching term per doc
    // surfaces both. The old AND-expansion returned nothing here.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "machine quantum", "mode": "keyword", "k": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = body.as_array().unwrap();
    let hit_docs: std::collections::HashSet<&str> = hits
        .iter()
        .map(|h| h["document_id"].as_str().unwrap())
        .collect();
    assert!(hit_docs.contains("doc1"), "doc1 must match on 'machine'");
    assert!(hit_docs.contains("doc2"), "doc2 must match on 'quantum'");

    // Hybrid mode (adoption contract §1.2): keyword-matching doc still
    // surfaces through the fused pool (no engine here — dense leg scores are
    // the documented zero-vector degradation).
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "machine", "mode": "hybrid", "k": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = body.as_array().unwrap();
    assert!(
        hits.iter().any(|h| h["document_id"] == "doc1"),
        "hybrid must surface the keyword match"
    );

    // Allowlist excluding the document filters it out.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({
            "query": "machine learning",
            "mode": "keyword",
            "k": 5,
            "allowlist": ["other-doc"]
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().unwrap().is_empty());

    // Search on a missing collection → 404.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/nope/search",
        Some(serde_json::json!({"query": "x"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Delete the document; search no longer matches.
    let (status, _) = send(
        &router,
        "DELETE",
        "/v1/collections/docs/documents/doc1",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "machine learning", "mode": "keyword"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().unwrap().is_empty());

    // Delete the collection.
    let (status, _) = send(&router, "DELETE", "/v1/collections/docs", None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let (_, body) = send(&router, "GET", "/v1/collections", None).await;
    assert_eq!(body, serde_json::json!([]));
}

#[tokio::test]
async fn collections_persist_across_restart() {
    let tmp = tempfile::TempDir::new().unwrap();

    // First "process": create + ingest.
    {
        let router = build_router(test_state(tmp.path()));
        let (status, _) = send(
            &router,
            "POST",
            "/v1/collections",
            Some(serde_json::json!({"name": "persist", "embedding_model": "embed-gemma:latest"})),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let (status, _) = send(
            &router,
            "POST",
            "/v1/collections/persist/documents",
            Some(serde_json::json!({"id": "d1", "content": "turbovec persistence check"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    // Second "process": load from disk, search still works.
    {
        let state = test_state(tmp.path());
        let loaded = state.collections.lock().unwrap().load_all().unwrap();
        assert_eq!(loaded, 1);
        let router = build_router(state);

        let (status, body) = send(
            &router,
            "POST",
            "/v1/collections/persist/search",
            Some(serde_json::json!({"query": "persistence", "mode": "keyword"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!body.as_array().unwrap().is_empty());
    }
}

#[tokio::test]
async fn rag_chat_without_model_returns_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "kb", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, body) = send(
        &router,
        "POST",
        "/v1/rag/chat",
        Some(serde_json::json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "hi"}],
            "collections": ["kb"]
        })),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn copy_and_delete_models() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    // Copy a built-in alias.
    let (status, _) = send(
        &router,
        "POST",
        "/api/copy",
        Some(serde_json::json!({"source": "gemma4:e4b", "destination": "mymodel:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // The copy resolves through /api/show.
    let (status, body) = send(
        &router,
        "POST",
        "/api/show",
        Some(serde_json::json!({"model": "mymodel"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["details"]["family"], "gemma");

    // Delete it.
    let req = Request::builder()
        .method("DELETE")
        .uri("/api/delete")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model": "mymodel"}"#))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let (status, _) = send(
        &router,
        "POST",
        "/api/show",
        Some(serde_json::json!({"model": "mymodel"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Copying a nonexistent source 404s.
    let (status, _) = send(
        &router,
        "POST",
        "/api/copy",
        Some(serde_json::json!({"source": "nope:x", "destination": "y"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn embedding_model_listed_in_tags() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (_, body) = send(&router, "GET", "/api/tags", None).await;
    let models = body["models"].as_array().unwrap();
    assert!(
        models.iter().any(|m| m["name"] == "embed-gemma:latest"),
        "default embedding model should be registered"
    );
}

#[tokio::test]
async fn pull_rejected_in_offline_mode() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = RootConfig::default();
    config.retrieval.data_dir = tmp.path().to_path_buf();
    config.server.offline = true;
    let router = build_router(Arc::new(AppState::new(config)));

    let (status, body) = send(
        &router,
        "POST",
        "/api/pull",
        Some(serde_json::json!({"model": "gemma4"})),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(body["error"].as_str().unwrap().contains("offline"));
}

#[tokio::test]
async fn bearer_auth_enforced_when_configured() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = RootConfig::default();
    config.retrieval.data_dir = tmp.path().to_path_buf();
    config.server.api_keys = vec!["sk-secret".into()];
    let router = build_router(Arc::new(AppState::new(config)));

    // No key → 401.
    let (status, body) = send(&router, "GET", "/api/tags", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(body["error"].is_string());

    // Wrong key → 401.
    let req = Request::builder()
        .method("GET")
        .uri("/api/tags")
        .header("authorization", "Bearer wrong")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Correct key → 200.
    let req = Request::builder()
        .method("GET")
        .uri("/api/tags")
        .header("authorization", "Bearer sk-secret")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn request_id_header_on_every_response() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let req = Request::builder()
        .method("GET")
        .uri("/api/version")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let id = resp
        .headers()
        .get("x-request-id")
        .expect("x-request-id header must be present")
        .to_str()
        .unwrap();
    assert!(!id.is_empty());
}

#[tokio::test]
async fn metrics_exposed_and_counting() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "m", "embedding_model": "default"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/m/documents",
        Some(serde_json::json!({"id": "d", "content": "metrics test document"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = String::from_utf8(
        resp.into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains("engine_requests_total"));
    assert!(text.contains("engine_ingest_documents_total 1"));
    assert!(text.contains("engine_collection_chunks{collection=\"m\"} "));
    assert!(text.contains("engine_model_loaded 0"));
}

#[tokio::test]
async fn multipart_ingest() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "files", "embedding_model": "default"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let boundary = "X-ENGINE-TEST-BOUNDARY";
    let body = format!(
        "--{b}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"notes.md\"\r\ncontent-type: text/markdown\r\n\r\n# Notes\n\nmultipart ingestion works\r\n--{b}--\r\n",
        b = boundary
    );
    let req = Request::builder()
        .method("POST")
        .uri("/v1/collections/files/documents")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(body))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["document_id"], "notes.md");
    assert!(json["chunks"].as_u64().unwrap() >= 1);

    // The uploaded content is searchable.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/files/search",
        Some(serde_json::json!({"query": "multipart ingestion", "mode": "keyword"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.as_array().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Hook wiring through the real middleware stack
// ---------------------------------------------------------------------------

struct BlockEverything;
impl mettle_core::hooks::EngineHook for BlockEverything {
    fn on_request(
        &self,
        _ctx: &mut mettle_core::hooks::RequestCtx,
    ) -> mettle_core::hooks::HookOutcome {
        mettle_core::hooks::HookOutcome::Abort {
            status_code: 403,
            message: "blocked by policy".into(),
        }
    }
}

#[tokio::test]
async fn on_request_hook_abort_returns_hook_status() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = RootConfig::default();
    config.retrieval.data_dir = tmp.path().to_path_buf();
    let mut state = AppState::new(config);

    let mut hooks = mettle_core::hooks::HookRegistry::new();
    hooks.register(Box::new(BlockEverything));
    state.hooks = Arc::new(hooks);

    let router = build_router(Arc::new(state));

    let (status, body) = send(&router, "GET", "/api/version", None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"], "blocked by policy");
}

// --- /v1/rerank (agent-docs/rerank-contract.md) ------------------------------

async fn send_raw(router: &axum::Router, uri: &str, body: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
async fn rerank_empty_documents_is_ok() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, body) = send(
        &router,
        "POST",
        "/v1/rerank",
        Some(serde_json::json!({"model": "bge-reranker-v2-m3", "query": "q", "documents": []})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"], serde_json::json!([]));
}

#[tokio::test]
async fn rerank_error_codes_match_contract() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    // Unknown model -> 404.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/rerank",
        Some(serde_json::json!({"model": "no-such-model", "query": "q", "documents": ["d"]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());

    // Registered but not a reranker (generation model) -> 404 per contract
    // ("unknown/unloaded model -> 404").
    let (status, body) = send(
        &router,
        "POST",
        "/v1/rerank",
        Some(serde_json::json!({"model": "gemma4:e2b", "query": "q", "documents": ["d"]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());

    // Malformed JSON body -> 400 (not axum's 422).
    let (status, body) =
        send_raw(&router, "/v1/rerank", "{\"model\": \"bge-reranker-v2-m3\"").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].is_string());

    // Missing field -> also a malformed request -> 400.
    let (status, _) = send_raw(&router, "/v1/rerank", "{\"model\": \"bge-reranker-v2-m3\"}").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Registered reranker without a loaded engine -> 503 slot_unavailable
    // (structured body per profiles-contract.md, which supersedes the
    // plain-string error this endpoint used to return when not resident).
    let (status, body) = send(
        &router,
        "POST",
        "/v1/rerank",
        Some(serde_json::json!({"model": "bge-reranker-v2-m3", "query": "q", "documents": ["d"]})),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "slot_unavailable");
    assert_eq!(body["error"]["missing"][0], "bge-reranker-v2-m3");
}

#[tokio::test]
async fn embeddings_rejects_reranker_with_404() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));

    let (status, body) = send(
        &router,
        "POST",
        "/v1/embeddings",
        Some(serde_json::json!({"model": "bge-reranker-v2-m3", "input": "hi"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn collection_names_are_validated_and_never_reach_the_filesystem() {
    let tmp = tempfile::TempDir::new().unwrap();
    // A sibling directory a traversal would target.
    let victim = tmp.path().join("victim");
    std::fs::create_dir_all(&victim).unwrap();
    let data_dir = tmp.path().join("data");
    let router = build_router(test_state(&data_dir));

    for bad in ["../victim", ".hidden", "a/b", "", &"x".repeat(129)] {
        let (status, body) = send(
            &router,
            "POST",
            "/v1/collections",
            Some(serde_json::json!({ "name": bad, "embedding_model": "embed-gemma:latest" })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "name {bad:?}: {body}");
    }

    // Percent-encoded traversal in the path decodes to `../../victim`
    // inside a single segment; it must be a 404, not a remove_dir_all.
    let (status, _) = send(&router, "DELETE", "/v1/collections/..%2F..%2Fvictim", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = send(&router, "DELETE", "/v1/collections/never-created", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(victim.exists(), "traversal deleted a directory outside the data dir");

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({ "name": "Fin_bench-2.0", "embedding_model": "embed-gemma:latest" })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
}
