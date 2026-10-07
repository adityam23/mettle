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

mod support;
use support::FakeEmbedder;

fn test_state(data_dir: &std::path::Path) -> Arc<AppState> {
    let mut config = RootConfig::default();
    config.retrieval.data_dir = data_dir.to_path_buf();
    Arc::new(AppState::new(config))
}

/// Built-in embedders resident in a [`FakeEmbedder`] (registry dims).
const EMBEDDERS: &[(&str, usize)] = &[
    ("embed-gemma:latest", 768),
    ("qwen3-embedding-0.6b:latest", 1024),
];

/// `test_state` plus a resident fake embedder, so ingest and dense search
/// run without a GGUF. Returns the embedder for unload simulation.
fn test_state_with_embedder(data_dir: &std::path::Path) -> (Arc<AppState>, Arc<FakeEmbedder>) {
    let state = test_state(data_dir);
    let embedder = FakeEmbedder::new(EMBEDDERS);
    state.set_engine(embedder.clone());
    (state, embedder)
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
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

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
    // surfaces through the fused pool.
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
async fn collection_short_embedder_name_resolves_to_its_alias() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

    // "embed-gemma" only resolves through the registry's `:latest` fallback.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "short", "embedding_model": "embed-gemma"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // The engine keys its loaded models by the canonical alias, so ingest
    // and dense search must address "embed-gemma:latest", not the raw name.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/short/documents",
        Some(serde_json::json!({"id": "d1", "content": "short embedder name ingest"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "ingest failed: {body}");

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/short/search",
        Some(serde_json::json!({"query": "short embedder ingest", "mode": "dense", "k": 3})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "dense search failed: {body}");
    assert!(!body.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn collection_reindex_rebuilds_in_place_and_fails_loud() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    for (id, content) in [
        (
            "d1",
            "machine learning is a subset of artificial intelligence",
        ),
        (
            "d2",
            "quantum computing breakthroughs accelerate everything",
        ),
    ] {
        let (status, _) = send(
            &router,
            "POST",
            "/v1/collections/docs/documents",
            Some(serde_json::json!({"id": id, "content": content})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    // A freshly ingested collection is stamped as normalised.
    let (status, body) = send(&router, "GET", "/v1/collections/docs", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["needs_reindex"], serde_json::json!(false));
    assert_eq!(body["documents"], serde_json::json!(2));
    let chunks = body["chunks"].as_u64().unwrap();
    assert!(chunks >= 2);

    // Reindex rebuilds in place: same chunk count, still stamped, still
    // searchable (the int_ids survive, so dense search keeps working).
    let (status, body) = send(&router, "POST", "/v1/collections/docs/reindex", None).await;
    assert_eq!(status, StatusCode::OK, "reindex failed: {body}");
    assert_eq!(body["reindexed"], serde_json::json!(true));
    assert_eq!(body["chunks"].as_u64().unwrap(), chunks);

    let (status, body) = send(&router, "GET", "/v1/collections/docs", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["needs_reindex"], serde_json::json!(false));
    assert_eq!(body["chunks"].as_u64().unwrap(), chunks);

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "machine learning", "mode": "dense", "k": 3})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body.as_array()
            .unwrap()
            .iter()
            .any(|h| h["document_id"] == "d1"),
        "rebuilt index must still return the ingested document: {body}"
    );

    // Fail loud, never corrupt: with the embedder gone the reindex is a 503
    // and the existing index still answers keyword search.
    embedder.unload("embed-gemma:latest");
    let (status, body) = send(&router, "POST", "/v1/collections/docs/reindex", None).await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "reindex without a resident embedder must fail loud: {body}"
    );
    assert_eq!(body["error"]["code"], serde_json::json!("slot_unavailable"));

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "machine learning", "mode": "keyword", "k": 3})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.as_array().unwrap().is_empty());

    // Unknown collection → 404 on both new routes.
    let (status, _) = send(&router, "GET", "/v1/collections/nope", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = send(&router, "POST", "/v1/collections/nope/reindex", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn collections_persist_across_restart() {
    let tmp = tempfile::TempDir::new().unwrap();

    // First "process": create + ingest.
    {
        let (state, _embedder) = test_state_with_embedder(tmp.path());
        let router = build_router(state);
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
    // The embedder is resident (retrieval succeeds); the chat model is not.
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

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

/// RAG chat embeds the query once per distinct embedder in a request, not
/// once per collection.
#[tokio::test]
async fn rag_chat_embeds_query_once_per_embedder() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

    for name in ["kb1", "kb2"] {
        let (status, _) = send(
            &router,
            "POST",
            "/v1/collections",
            Some(serde_json::json!({"name": name, "embedding_model": "embed-gemma:latest"})),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
    }

    let before = embedder.embed_calls();
    // Retrieval runs (and succeeds) before the chat model's 503.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/rag/chat",
        Some(serde_json::json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "hi"}],
            "collections": ["kb1", "kb2"]
        })),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(embedder.embed_calls() - before, 1);
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
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

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
    // The fake embedder counts as a loaded engine.
    assert!(text.contains("engine_model_loaded 1"));
}

#[tokio::test]
async fn multipart_ingest() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

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

/// §1.1: a per-call rerank context that cannot fit is a structured 503
/// carrying `rerank_context_vram`, never a 500 and never a crash. A rerank
/// that fits keeps returning 200 scores.
#[tokio::test]
async fn rerank_context_vram_is_a_structured_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let engine = FakeEmbedder::new(&[("bge-reranker-v2-m3:latest", 768)]);
    state.set_engine(engine.clone());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "bge-reranker-v2-m3",
        "query": "q",
        "documents": ["d1", "d2"],
    });
    let (status, resp) = send(&router, "POST", "/v1/rerank", Some(body.clone())).await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    assert_eq!(resp["results"].as_array().unwrap().len(), 2);

    engine.fail_rerank_context_vram(
        "bge-reranker-v2-m3:latest",
        "rerank context needs ~1900 MiB (2048 tokens), 300 MiB free after margin",
    );
    let (status, resp) = send(&router, "POST", "/v1/rerank", Some(body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{resp}");
    assert_eq!(resp["error"]["code"], "slot_unavailable");
    assert_eq!(resp["error"]["reason"], "rerank_context_vram");
    assert_eq!(resp["error"]["missing"][0], "bge-reranker-v2-m3");
    assert!(resp["error"]["detail"].as_str().unwrap().contains("MiB"));
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
    assert!(
        victim.exists(),
        "traversal deleted a directory outside the data dir"
    );

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(
            serde_json::json!({ "name": "Fin_bench-2.0", "embedding_model": "embed-gemma:latest" }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
}

/// Residency contract §1.3: no zero-vector fallback on any path. An
/// unservable embedder fails ingest with the structured 503 and writes
/// nothing; dense/hybrid search fails the same way; keyword still works.
#[tokio::test]
async fn ingest_and_search_fail_loud_without_embedder() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state.clone());
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({"id": "kept", "content": "resident embedder wrote this"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    embedder.unload("embed-gemma:latest");

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({"id": "lost", "content": "zebra unicorn payload"})),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "slot_unavailable", "{body}");
    assert_eq!(body["error"]["reason"], "not_resident");
    assert_eq!(body["error"]["missing"][0], "embed-gemma:latest");

    // Nothing was written: no mode finds the rejected document.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "zebra unicorn", "mode": "keyword", "k": 10})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().unwrap().is_empty(), "{body}");

    for mode in ["dense", "hybrid"] {
        let (status, body) = send(
            &router,
            "POST",
            "/v1/collections/docs/search",
            Some(serde_json::json!({"query": "zebra", "mode": mode})),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{mode}: {body}");
        assert_eq!(body["error"]["code"], "slot_unavailable");
    }

    // A re-POST once the embedder is back is a clean first write (no 409
    // from a half-written earlier attempt).
    let (state2, _e) = test_state_with_embedder(tmp.path());
    state2.collections.lock().unwrap().load_all().unwrap();
    let router2 = build_router(state2);
    let (status, body) = send(
        &router2,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({"id": "lost", "content": "zebra unicorn payload"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn no_engine_ingest_is_a_structured_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let router = build_router(test_state(tmp.path()));
    send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents",
        Some(serde_json::json!({"id": "d", "content": "text"})),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["reason"], "no_engine", "{body}");
}

/// Batch ingest is all-or-nothing, the list endpoint reports what is
/// actually stored, and a batch is searchable once it lands (docket M3/M4).
#[tokio::test]
async fn batch_ingest_is_atomic_and_list_documents_reports_stored_state() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);

    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");

    // Two documents in one batch, one request.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents:batch",
        Some(serde_json::json!({"documents": [
            {"id": "a#1", "content": "quartz zebra one", "metadata": {"doc_id": "a", "page": "1"}},
            {"id": "b#1", "content": "quartz zebra two", "metadata": {"doc_id": "b", "page": "1"}},
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["documents"], 2, "{body}");
    assert_eq!(body["results"][0]["document_id"], "a#1");
    assert!(
        body["results"][0]["chunks"].as_u64().unwrap() >= 1,
        "{body}"
    );

    // The batch is searchable: both the FTS rows and the vectors landed.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/search",
        Some(serde_json::json!({"query": "quartz zebra", "mode": "keyword", "k": 10})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body.as_array().unwrap().len(), 2, "{body}");

    // The list reports ids, chunk counts and metadata.
    let (status, body) = send(&router, "GET", "/v1/collections/docs/documents", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let docs = body.as_array().unwrap();
    assert_eq!(docs.len(), 2, "{body}");
    assert_eq!(docs[0]["id"], "a#1");
    assert!(docs[0]["chunk_count"].as_u64().unwrap() >= 1, "{body}");
    assert_eq!(docs[0]["metadata"]["doc_id"], "a", "{body}");

    // A batch containing an already-stored id writes NOTHING: the new
    // document must not appear.
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents:batch",
        Some(serde_json::json!({"documents": [
            {"id": "c#1", "content": "quartz zebra three"},
            {"id": "a#1", "content": "quartz zebra one"},
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let (_, body) = send(&router, "GET", "/v1/collections/docs/documents", None).await;
    assert_eq!(body.as_array().unwrap().len(), 2, "batch leaked: {body}");

    // A duplicate INSIDE one batch fails the whole batch too.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents:batch",
        Some(serde_json::json!({"documents": [
            {"id": "d#1", "content": "quartz zebra four"},
            {"id": "d#1", "content": "quartz zebra five"},
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (_, body) = send(&router, "GET", "/v1/collections/docs/documents", None).await;
    assert_eq!(body.as_array().unwrap().len(), 2, "batch leaked: {body}");

    // Caps: more documents than one batch may carry -> 413, nothing written.
    let docs: Vec<serde_json::Value> = (0..1025)
        .map(|i| serde_json::json!({"id": format!("big#{i}"), "content": "x"}))
        .collect();
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents:batch",
        Some(serde_json::json!({"documents": docs})),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");

    // Caps: a body over the route's 8 MiB byte limit -> 413 from the
    // `DefaultBodyLimit` layer, nothing written.
    let oversized: String = "quartz zebra ".repeat(700_000); // ~8.7 MiB
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents:batch",
        Some(serde_json::json!({"documents": [{"id": "over#1", "content": oversized}]})),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    let (_, body) = send(&router, "GET", "/v1/collections/docs/documents", None).await;
    assert_eq!(
        body.as_array().unwrap().len(),
        2,
        "oversized batch leaked: {body}"
    );

    // A batch well over axum's default 2 MiB body limit still lands (the
    // route carries its own 8 MiB limit), so a whole filing fits.
    let big: String = "quantum zebra ".repeat(200_000); // ~2.6 MiB
    let (status, body) = send(
        &router,
        "POST",
        "/v1/collections/docs/documents:batch",
        Some(serde_json::json!({"documents": [{"id": "huge#1", "content": big}]})),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        body.to_string().chars().take(200).collect::<String>()
    );
    assert_eq!(
        body["documents"],
        1,
        "{}",
        body.to_string().chars().take(200).collect::<String>()
    );

    // Unknown collection is a 404, not an empty 200.
    let (status, _) = send(
        &router,
        "POST",
        "/v1/collections/nope/documents:batch",
        Some(serde_json::json!({"documents": [{"id": "z", "content": "z"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = send(&router, "GET", "/v1/collections/nope/documents", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Document ids are free-form; percent-encoded path segments decode to the
/// stored id (QUESTIONS from docket Q2: `acme#3` was lost as a fragment).
#[tokio::test]
async fn document_ids_round_trip_percent_encoded() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (state, _embedder) = test_state_with_embedder(tmp.path());
    let router = build_router(state);
    send(
        &router,
        "POST",
        "/v1/collections",
        Some(serde_json::json!({"name": "docs", "embedding_model": "embed-gemma:latest"})),
    )
    .await;
    for (id, encoded) in [("acme#3", "acme%233"), ("a/b c", "a%2Fb%20c")] {
        let (status, _) = send(
            &router,
            "POST",
            "/v1/collections/docs/documents",
            Some(serde_json::json!({"id": id, "content": format!("payload for {id}")})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let uri = format!("/v1/collections/docs/documents/{encoded}");
        let (status, body) = send(&router, "DELETE", &uri, None).await;
        assert_eq!(status, StatusCode::NO_CONTENT, "{id}: {body}");
    }
}
