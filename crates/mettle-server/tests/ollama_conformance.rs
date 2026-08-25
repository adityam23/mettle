//! Conformance tests for Ollama API response shapes.
//!
//! These are the definition of "drop-in": an Ollama client must receive
//! byte-level compatible JSON from every endpoint.

use mettle_core::ollama_types::*;

// ---------------------------------------------------------------------------
// /api/version
// ---------------------------------------------------------------------------

#[test]
fn version_response_shape() {
    let resp = VersionResponse {
        version: "engine 0.1.0".into(),
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert_eq!(json, r#"{"version":"engine 0.1.0"}"#);
}

// ---------------------------------------------------------------------------
// /api/chat — non-streaming
// ---------------------------------------------------------------------------

#[test]
fn chat_response_shape() {
    let resp = ChatResponse {
        model: "gemma4".into(),
        created_at: "2026-06-10T12:00:00Z".into(),
        message: ResponseMessage {
            role: "assistant".into(),
            content: "Hello!".into(),
            tool_calls: None,
        },
        done_reason: "stop".into(),
        done: true,
        timing: TimingInfo {
            total_duration: 1_000_000_000,
            load_duration: 0,
            prompt_eval_count: 10,
            prompt_eval_duration: 100_000_000,
            eval_count: 5,
            eval_duration: 900_000_000,
        },
    };

    let json = serde_json::to_string(&resp).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Key fields must be present.
    assert_eq!(parsed["model"], "gemma4");
    assert_eq!(parsed["done"], true);
    assert_eq!(parsed["done_reason"], "stop");
    assert_eq!(parsed["message"]["role"], "assistant");
    assert_eq!(parsed["message"]["content"], "Hello!");

    // Timing fields must use nanoseconds (whole numbers).
    assert_eq!(parsed["total_duration"], 1_000_000_000u64);
    assert_eq!(parsed["prompt_eval_count"], 10);
    assert_eq!(parsed["eval_count"], 5);
}

// ---------------------------------------------------------------------------
// /api/chat — streaming chunk
// ---------------------------------------------------------------------------

#[test]
fn chat_stream_chunk_shape() {
    let chunk = ChatStreamChunk {
        model: "gemma4".into(),
        created_at: "2026-06-10T12:00:00Z".into(),
        message: ChatStreamMessage {
            role: "assistant".into(),
            content: "Hi".into(),
            tool_calls: None,
        },
        done: false,
        done_reason: None,
        timing: None,
    };

    let json = serde_json::to_string(&chunk).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["done"], false);
    assert_eq!(parsed["message"]["content"], "Hi");
    // tool_calls should be absent when None.
    assert!(parsed["message"].get("tool_calls").is_none());
    // Intermediate chunks must not carry done_reason or timing fields.
    assert!(parsed.get("done_reason").is_none());
    assert!(parsed.get("eval_count").is_none());
}

#[test]
fn chat_stream_final_chunk_shape() {
    let chunk = ChatStreamChunk {
        model: "gemma4".into(),
        created_at: "2026-06-10T12:00:00Z".into(),
        message: ChatStreamMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: None,
        },
        done: true,
        done_reason: Some("stop".into()),
        timing: Some(TimingInfo {
            total_duration: 1_000_000_000,
            load_duration: 0,
            prompt_eval_count: 10,
            prompt_eval_duration: 100_000_000,
            eval_count: 5,
            eval_duration: 900_000_000,
        }),
    };

    let json = serde_json::to_string(&chunk).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Ollama's final streaming line carries done_reason and flattened timing.
    assert_eq!(parsed["done"], true);
    assert_eq!(parsed["done_reason"], "stop");
    assert_eq!(parsed["total_duration"], 1_000_000_000u64);
    assert_eq!(parsed["eval_count"], 5);
}

// ---------------------------------------------------------------------------
// /api/chat — request deserialization
// ---------------------------------------------------------------------------

#[test]
fn chat_request_deserialization() {
    let body = r#"{
        "model": "gemma4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ],
        "stream": true,
        "options": {"temperature": 0.7}
    }"#;

    let req: ChatRequest = serde_json::from_str(body).unwrap();
    assert_eq!(req.model, "gemma4");
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, "system");
    assert_eq!(req.messages[1].content, "Hello");
    assert_eq!(req.stream, Some(true));
    assert_eq!(
        req.options
            .unwrap()
            .get("temperature")
            .unwrap()
            .as_f64()
            .unwrap(),
        0.7
    );
}

// ---------------------------------------------------------------------------
// /api/generate — request deserialization
// ---------------------------------------------------------------------------

#[test]
fn generate_request_deserialization() {
    let body = r#"{
        "model": "gemma4",
        "prompt": "Why is the sky blue?",
        "system": "Be concise.",
        "stream": false
    }"#;

    let req: GenerateRequest = serde_json::from_str(body).unwrap();
    assert_eq!(req.model, "gemma4");
    assert_eq!(req.prompt, "Why is the sky blue?");
    assert_eq!(req.system, Some("Be concise.".into()));
    assert_eq!(req.stream, Some(false));
}

// ---------------------------------------------------------------------------
// /api/generate — response shape
// ---------------------------------------------------------------------------

#[test]
fn generate_response_shape() {
    let resp = GenerateResponse {
        model: "gemma4".into(),
        created_at: "2026-06-10T12:00:00Z".into(),
        response: "The sky is blue because...".into(),
        done: true,
        done_reason: "stop".into(),
        context: vec![1, 2, 3],
        timing: TimingInfo {
            total_duration: 500_000_000,
            load_duration: 0,
            prompt_eval_count: 8,
            prompt_eval_duration: 50_000_000,
            eval_count: 20,
            eval_duration: 450_000_000,
        },
    };

    let json = serde_json::to_string(&resp).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["done"], true);
    assert_eq!(parsed["response"], "The sky is blue because...");
    assert_eq!(parsed["context"].as_array().unwrap().len(), 3);
    assert_eq!(parsed["eval_count"], 20);
}

// ---------------------------------------------------------------------------
// /api/tags — response shape
// ---------------------------------------------------------------------------

#[test]
fn tags_response_shape() {
    let resp = TagsResponse {
        models: vec![TagModelInfo {
            name: "gemma4:latest".into(),
            model: "gemma4:latest".into(),
            modified_at: "2026-06-10T00:00:00Z".into(),
            size: 4_000_000_000,
            digest: "abc123".into(),
            details: ModelDetails {
                parent_model: "".into(),
                format: "gguf".into(),
                family: "gemma".into(),
                families: vec!["gemma".into()],
                parameter_size: "4B".into(),
                quantization_level: "Q4_K_M".into(),
            },
        }],
    };

    let json = serde_json::to_string(&resp).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["models"].as_array().unwrap().len(), 1);
    assert_eq!(parsed["models"][0]["name"], "gemma4:latest");
    assert_eq!(parsed["models"][0]["details"]["family"], "gemma");
}

// ---------------------------------------------------------------------------
// Format option — JSON mode and JSON schema
// ---------------------------------------------------------------------------

#[test]
fn format_option_json() {
    let body = r#""json""#;
    let fmt: FormatOption = serde_json::from_str(body).unwrap();
    assert!(matches!(fmt, FormatOption::Json));
}

#[test]
fn format_option_schema() {
    let body = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
    let fmt: FormatOption = serde_json::from_str(body).unwrap();
    assert!(matches!(fmt, FormatOption::Schema(_)));
}

// ---------------------------------------------------------------------------
// /api/embed — request with single and multiple inputs
// ---------------------------------------------------------------------------

#[test]
fn embed_request_single_input() {
    let body = r#"{"model": "gemma4", "input": "hello world"}"#;
    let req: EmbedRequest = serde_json::from_str(body).unwrap();
    let inputs = req.input.into_vec();
    assert_eq!(inputs, vec!["hello world"]);
}

#[test]
fn embed_request_multiple_inputs() {
    let body = r#"{"model": "gemma4", "input": ["hello", "world"]}"#;
    let req: EmbedRequest = serde_json::from_str(body).unwrap();
    let inputs = req.input.into_vec();
    assert_eq!(inputs, vec!["hello", "world"]);
}

// ---------------------------------------------------------------------------
// Error response shape
// ---------------------------------------------------------------------------

#[test]
fn error_response_shape() {
    // Ollama returns {"error": "message"} for errors.
    let err = serde_json::json!({"error": "model not found"});
    let json = serde_json::to_string(&err).unwrap();
    assert_eq!(json, r#"{"error":"model not found"}"#);
}
