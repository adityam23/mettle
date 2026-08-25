//! End-to-end rerank scoring tests against the real GGUFs on disk.
//!
//! Ignored by default (needs the model files). Run manually:
//!   cargo test -p mettle-llama --test rerank_scores -- --ignored --nocapture
//!
//! These are the scoring half of the rerank contract's parity gate
//! (agent-docs/rerank-contract.md): HTTP status codes are covered by
//! mettle-server's `http_routes.rs`, the numbers here. Full HF-reference
//! parity lives in `/tmp/rerank_parity.py`; these assert the stable,
//! cheap invariants (ordering, score ranges, template-path activation).

use mettle_core::engine::{
    Engine, EngineRequest, EngineResponse, ModelKind, ModelLoadSpec, RequestKind,
};
use mettle_llama::{LlamaConfig, LlamaEngine};

fn gguf(rel: &str) -> std::path::PathBuf {
    let root = std::env::var_os("METTLE_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        });
    root.join(rel)
}

fn scores(engine: &LlamaEngine, alias: &str, query: &str, docs: &[&str]) -> Vec<f32> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineResponse>(64);
    let req = EngineRequest {
        model: alias.into(),
        kind: RequestKind::Rerank {
            query: query.into(),
            documents: docs.iter().map(|s| s.to_string()).collect(),
        },
        response: tx,
    };
    engine.sender().blocking_send(req).expect("submit");
    loop {
        match rx.blocking_recv().expect("engine dropped request") {
            EngineResponse::Scores(v) => return v,
            EngineResponse::Error(e) => panic!("rerank failed: {e}"),
            _ => continue,
        }
    }
}

#[test]
#[ignore = "needs models/bge-reranker-v2-m3 + qwen3-reranker-0.6b on disk; run with --ignored"]
fn rerankers_score_finance_pairs() {
    // One LlamaEngine (=> one LlamaBackend) per process; both rerankers ride
    // as separate aliases in a single engine.
    let mk_spec = |alias: &str, rel: &str| ModelLoadSpec {
        alias: alias.into(),
        gguf_path: gguf(rel),
        kind: ModelKind::Rerank,
        n_gpu_layers: Some(0),
        mmproj_path: None,
    };
    let cfg = LlamaConfig {
        n_gpu_layers: 0,
        ..LlamaConfig::default()
    };
    let engine = LlamaEngine::new(
        vec![
            mk_spec(
                "bge-reranker-v2-m3",
                "bge-reranker-v2-m3/bge-reranker-v2-m3-Q8_0.gguf",
            ),
            mk_spec(
                "qwen3-reranker-0.6b",
                "qwen3-reranker-0.6b/Qwen3-Reranker-0.6B-F16.gguf",
            ),
        ],
        cfg,
    )
    .expect("engine init");

    let query = "What was FY2018 capital expenditure?";
    let docs = [
        "Capital expenditures totaled $4.2 billion in fiscal 2018.",
        "The company held $900 million in cash and equivalents.",
        "Revenue grew 12% year over year to $31.5 billion.",
    ];

    // bge: XLM-R cross-encoder, raw logit scores.
    let bge_scores = scores(&engine, "bge-reranker-v2-m3", query, &docs);
    assert_eq!(bge_scores.len(), 3);
    assert!(
        bge_scores[0] > bge_scores[1] && bge_scores[0] > bge_scores[2],
        "relevant doc must rank first, got {bge_scores:?}"
    );

    // qwen3: decoder with GGUF "rerank" chat template => P(yes) in [0,1].
    // Without the template path the fallback layout produces degenerate
    // values for this model.
    let qwen_scores = scores(&engine, "qwen3-reranker-0.6b", query, &docs);
    assert_eq!(qwen_scores.len(), 3);
    assert!(
        qwen_scores.iter().all(|s| *s >= 0.0 && *s <= 1.0),
        "P(yes) scores must be probabilities, got {qwen_scores:?}"
    );
    assert!(
        qwen_scores[0] > 0.5,
        "relevant doc should be judged yes: {qwen_scores:?}"
    );
    assert!(
        qwen_scores[1] < 0.1 && qwen_scores[2] < 0.1,
        "irrelevant docs near no: {qwen_scores:?}"
    );
}
