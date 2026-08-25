//! End-to-end smoke test for the Phase 1 single-sequence engine.
//!
//! Ignored by default (needs a real GGUF). Run manually:
//!   METTLE_GGUF=/path/to/gemma-4-E4B-it-Q4_K_M.gguf \
//!     cargo test -p mettle-llama --test smoke -- --ignored --nocapture
//! Set METTLE_NGL=999 to offload to GPU (needs a CUDA build).
//!
//! Drives the engine purely through the public `Engine` trait + channel,
//! using blocking send/recv (no async runtime needed in the test thread).

use mettle_core::engine::{Engine, EngineRequest, EngineResponse, RequestKind, SamplingParams};
use mettle_core::ollama_types::ChatMessage;
use mettle_llama::{LlamaConfig, LlamaEngine};

fn gguf_path() -> std::path::PathBuf {
    std::env::var("METTLE_GGUF")
        .expect("set METTLE_GGUF to a Gemma 4 E4B GGUF path")
        .into()
}

#[test]
#[ignore = "needs a real GGUF; run with --ignored and METTLE_GGUF set"]
fn streaming_chat_generates() {
    let spec = mettle_core::engine::ModelLoadSpec {
        alias: "gemma4:e4b".into(),
        gguf_path: gguf_path(),
        kind: mettle_core::engine::ModelKind::Generation,
        n_gpu_layers: None,
        mmproj_path: None,
    };
    let cfg = LlamaConfig {
        n_gpu_layers: std::env::var("METTLE_NGL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        default_max_tokens: 32,
        ..LlamaConfig::default()
    };

    let engine = LlamaEngine::new(vec![spec], cfg).expect("engine init");
    assert_eq!(engine.loaded(), vec!["gemma4:e4b".to_string()]);

    let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineResponse>(64);
    let req = EngineRequest {
        model: "gemma4:e4b".into(),
        kind: RequestKind::Chat {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Reply with exactly the word: hello".into(),
                images: None,
                tool_calls: None,
            }],
            params: SamplingParams::default(),
            constraint: Default::default(),
            streaming: true,
        },
        response: tx,
    };
    engine.sender().blocking_send(req).expect("submit");

    let mut text = String::new();
    let mut finished = false;
    while let Some(resp) = rx.blocking_recv() {
        match resp {
            EngineResponse::Chunk {
                content,
                finish_reason,
                usage,
                ..
            } => {
                text.push_str(&content);
                if let Some(reason) = finish_reason {
                    finished = true;
                    eprintln!(
                        "\n[finish={reason} prompt_tok={} gen_tok={}]",
                        usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                        usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
                    );
                    break;
                }
            }
            EngineResponse::Error(e) => panic!("engine error: {e}"),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    eprintln!("generated: {text:?}");
    assert!(!text.trim().is_empty(), "expected non-empty generation");
    assert!(finished, "stream never finished");
}
