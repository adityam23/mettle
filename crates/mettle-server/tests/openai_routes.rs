//! HTTP-level tests for the OpenAI-compatible `/v1/*` surface, Anthropic
//! `/v1/messages`, and the OCR endpoint: status codes, error shapes, and
//! serde wire shapes (matching what OpenAI/Anthropic SDKs and
//! agentic-assistant's `openai_compat.py` parse).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mettle_core::config::RootConfig;
use mettle_server::app::build_router;
use mettle_server::state::AppState;
use http_body_util::BodyExt;
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

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_lists_registered_aliases() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let (status, json) = send(&router, "GET", "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    let ids: Vec<String> = json["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&"gemma4:e2b".to_string()));
    assert!(ids.contains(&"embed-gemma:latest".to_string()));
    assert!(ids.contains(&"unlimited-ocr:latest".to_string()));
    for m in json["data"].as_array().unwrap() {
        assert_eq!(m["object"], "model");
        assert!(m["id"].is_string());
        assert!(m["owned_by"].is_string());
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_without_engine_returns_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "gemma4:e2b",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let (status, json) = send(&router, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json["error"]["code"], "slot_unavailable");
}

#[tokio::test]
async fn chat_completions_unknown_model_returns_404() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "no-such-model",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let (status, _) = send(&router, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn chat_completions_rejects_n_greater_than_one() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "gemma4:e2b",
        "messages": [{"role": "user", "content": "hi"}],
        "n": 2,
    });
    let (status, _) = send(&router, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_accepts_extra_openai_fields() {
    // Drop-in clients send fields we ignore (tools, seed, logprobs...).
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "gemma4:e2b",
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 16,
        "stop": ["\n", "END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "seed": 42,
        "logprobs": true,
        "user": "tester",
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    });
    // Parses fine (no 400/422); engine absent → 503, not a deserialization error.
    let (status, json) = send(&router, "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json["error"]["code"], "slot_unavailable");
}

// ---------------------------------------------------------------------------
// POST /v1/completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_rejects_multiple_prompts() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "gemma4:e2b",
        "prompt": ["one", "two"],
    });
    let (status, _) = send(&router, "POST", "/v1/completions", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embeddings_without_engine_returns_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "embed-gemma:latest",
        "input": ["hello", "world"],
    });
    let (status, json) = send(&router, "POST", "/v1/embeddings", Some(body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json["error"]["code"], "slot_unavailable");
}

// ---------------------------------------------------------------------------
// POST /v1/messages (Anthropic)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn messages_without_engine_returns_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({
        "model": "gemma4:e2b",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "system": "be brief",
        "stream": false,
    });
    let (status, json) = send(&router, "POST", "/v1/messages", Some(body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(json["error"].is_string());
}

// ---------------------------------------------------------------------------
// POST /v1/ocr
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ocr_invalid_base64_returns_400() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({"image": "not base64 !!!"});
    let (status, json) = send(&router, "POST", "/v1/ocr", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"].is_string());
}

#[tokio::test]
async fn ocr_empty_image_returns_400() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    let body = serde_json::json!({"image": ""});
    let (status, _) = send(&router, "POST", "/v1/ocr", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn ocr_without_engine_returns_503() {
    let tmp = tempfile::TempDir::new().unwrap();
    let state = test_state(tmp.path());
    let router = build_router(state);

    // Valid base64 of a tiny PNG.
    let body = serde_json::json!({"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="});
    let (status, json) = send(&router, "POST", "/v1/ocr", Some(body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json["error"]["code"], "slot_unavailable");
}

// ---------------------------------------------------------------------------
// Auth applies to the whole /v1 surface
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bearer_auth_enforced_on_v1_routes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = RootConfig::default();
    config.retrieval.data_dir = tmp.path().to_path_buf();
    config.server.api_keys = vec!["sk-secret".into()];
    let state = Arc::new(AppState::new(config));
    let router = build_router(state);

    let body = serde_json::json!({"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="});
    let req = Request::builder()
        .method("POST")
        .uri("/v1/ocr")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer sk-secret")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Serde wire shapes (what SDKs parse)
// ---------------------------------------------------------------------------

#[test]
fn chat_completion_response_shape() {
    let resp = mettle_core::openai_types::ChatCompletionResponse {
        id: "chatcmpl-1".into(),
        object: "chat.completion",
        created: 1,
        model: "gemma4:e2b".into(),
        choices: vec![mettle_core::openai_types::ChatChoice {
            index: 0,
            message: mettle_core::openai_types::ResponseMessage {
                role: "assistant".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            finish_reason: "stop".into(),
            logprobs: None,
        }],
        usage: mettle_core::openai_types::Usage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
        },
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 4);
}

#[test]
fn chat_completion_response_logprobs_shape() {
    // agent-assistant's trust layer parses choices[].logprobs.content[].logprob;
    // the field must serialize when requested and be absent otherwise.
    use mettle_core::openai_types as t;
    let with = t::ChatChoice {
        index: 0,
        message: t::ResponseMessage {
            role: "assistant".into(),
            content: "hi".into(),
            reasoning_content: None,
        },
        finish_reason: "stop".into(),
        logprobs: Some(t::ChatLogprobs {
            content: vec![t::LogprobContent {
                token: "hi".into(),
                logprob: -0.03,
                top_logprobs: vec![t::TopLogprob {
                    token: "hi".into(),
                    logprob: -0.03,
                }],
            }],
        }),
    };
    let v = serde_json::to_value(&with).unwrap();
    assert_eq!(v["logprobs"]["content"][0]["token"], "hi");
    assert_eq!(v["logprobs"]["content"][0]["logprob"], -0.03);
    assert_eq!(
        v["logprobs"]["content"][0]["top_logprobs"][0]["token"],
        "hi"
    );

    let without = t::ChatChoice {
        logprobs: None,
        ..with
    };
    let v = serde_json::to_value(&without).unwrap();
    assert!(v.get("logprobs").is_none());

    // Streaming: the per-token entry rides on each chunk's choice.
    let chunk = t::ChatChunkChoice {
        index: 0,
        delta: t::ChatDelta {
            role: Some("assistant".into()),
            content: Some("hi".into()),
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: with.logprobs,
    };
    let v = serde_json::to_value(&chunk).unwrap();
    assert_eq!(v["logprobs"]["content"][0]["logprob"], -0.03);
}

#[test]
fn chat_completion_chunk_shape() {
    let chunk = mettle_core::openai_types::ChatCompletionChunk {
        id: "chatcmpl-1".into(),
        object: "chat.completion.chunk",
        created: 1,
        model: "gemma4:e2b".into(),
        choices: vec![mettle_core::openai_types::ChatChunkChoice {
            index: 0,
            delta: mettle_core::openai_types::ChatDelta {
                role: Some("assistant".into()),
                content: Some("hi".into()),
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
    };
    let v = serde_json::to_value(&chunk).unwrap();
    assert_eq!(v["object"], "chat.completion.chunk");
    assert_eq!(v["choices"][0]["delta"]["content"], "hi");
    assert!(v["choices"][0].get("finish_reason").is_none());
}

#[test]
fn embedding_response_shape() {
    let resp = mettle_core::openai_types::EmbeddingResponse {
        object: "list",
        data: vec![mettle_core::openai_types::EmbeddingData {
            object: "embedding",
            embedding: vec![0.1, 0.2],
            index: 0,
        }],
        model: "embed-gemma:latest".into(),
        usage: mettle_core::openai_types::Usage {
            prompt_tokens: 1,
            completion_tokens: 0,
            total_tokens: 1,
        },
    };
    let v = serde_json::to_value(&resp).unwrap();
    let emb = v["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(emb.len(), 2);
    assert!((emb[0].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert!((emb[1].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert_eq!(v["data"][0]["index"], 0);
}

#[test]
fn anthropic_messages_response_shape() {
    let resp = mettle_core::anthropic_types::MessagesResponse {
        id: "msg_1".into(),
        type_: "message",
        role: "assistant",
        content: vec![mettle_core::anthropic_types::ContentBlock {
            type_: "text",
            text: "hi".into(),
        }],
        model: "gemma4:e2b".into(),
        stop_reason: "end_turn".into(),
        usage: mettle_core::anthropic_types::AnthropicUsage {
            input_tokens: 3,
            output_tokens: 1,
        },
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "hi");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["input_tokens"], 3);
}

#[test]
fn anthropic_stream_events_serialize_with_type_tag() {
    let evt = mettle_core::anthropic_types::MessageStreamEvent::ContentBlockDelta {
        index: 0,
        delta: mettle_core::anthropic_types::TextDelta {
            type_: "text_delta",
            text: "hi".into(),
        },
    };
    let v = serde_json::to_value(&evt).unwrap();
    assert_eq!(v["type"], "content_block_delta");
    assert_eq!(v["delta"]["type"], "text_delta");

    let stop = mettle_core::anthropic_types::MessageStreamEvent::MessageStop;
    let v = serde_json::to_value(&stop).unwrap();
    assert_eq!(v["type"], "message_stop");
}

#[test]
fn ocr_response_shape() {
    let resp = mettle_core::ocr_types::OcrResponse {
        text: "page text".into(),
        model: "unlimited-ocr:latest".into(),
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["text"], "page text");
    assert_eq!(v["model"], "unlimited-ocr:latest");
}
