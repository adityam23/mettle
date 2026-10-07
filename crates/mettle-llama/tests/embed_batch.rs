//! A multi-input embedding request (one shared context) must equal
//! per-input requests (within float tolerance) for every pooling mode —
//! causal last-token (Qwen3), bidirectional mean (embeddinggemma) and CLS
//! (bge-m3) — across mixed input lengths.
//!
//! Ignored by default (needs the model files). Run manually:
//!   cargo test -p mettle-llama --test embed_batch -- --ignored --nocapture

use mettle_core::engine::{
    Engine, EngineRequest, EngineResponse, ModelKind, ModelLoadSpec, RequestKind,
};
use mettle_core::model_registry::Pooling;
use mettle_llama::{LlamaConfig, LlamaEngine};

fn gguf(rel: &str) -> std::path::PathBuf {
    let root = std::env::var_os("METTLE_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        });
    root.join(rel)
}

fn embed(engine: &LlamaEngine, alias: &str, inputs: &[String]) -> Vec<Vec<f32>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineResponse>(1);
    let req = EngineRequest {
        model: alias.into(),
        kind: RequestKind::Embedding {
            inputs: inputs.to_vec(),
        },
        response: tx,
    };
    engine.sender().blocking_send(req).expect("submit");
    match rx.blocking_recv().expect("engine dropped request") {
        EngineResponse::Embeddings(v) => v,
        other => panic!("embedding failed: {other:?}"),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Mixed, irregular lengths (the shared context is sized for the longest
/// and reused for shorter inputs).
fn inputs() -> Vec<String> {
    let sentence = "Quarterly revenue rose on stronger cloud demand while operating costs fell. ";
    (0..21)
        .map(|i| {
            let extra = " filler".repeat(i * 5 % 11);
            match i % 7 {
                0 => format!("short query{extra}"),
                6 => sentence.repeat(150), // ~2k tokens: the longest input
                k => sentence.repeat(k * 12) + &extra,
            }
        })
        .collect()
}

#[test]
#[ignore = "needs models/{qwen3-embedding-0.6b,embeddinggemma-300m,bge-m3} on disk; run with --ignored"]
fn batched_embeddings_match_single() {
    let models = [
        (
            "qwen3-embed",
            "qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf",
            Pooling::Last,
        ),
        (
            "embed-gemma",
            "embeddinggemma-300m/embeddinggemma-300M-Q8_0.gguf",
            Pooling::Mean,
        ),
        ("bge-m3", "bge-m3/bge-m3-Q8_0.gguf", Pooling::Cls),
    ];
    let n_gpu_layers = std::env::var("METTLE_NGL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let specs = models
        .iter()
        .filter(|(_, rel, _)| gguf(rel).exists())
        .filter(|(alias, _, _)| {
            std::env::var("METTLE_EMBED_ONLY").map_or(true, |only| only == *alias)
        })
        .map(|(alias, rel, pooling)| ModelLoadSpec {
            alias: (*alias).into(),
            gguf_path: gguf(rel),
            kind: ModelKind::Embedding(*pooling),
            n_gpu_layers: Some(n_gpu_layers),
            mmproj_path: None,
        })
        .collect::<Vec<_>>();
    assert!(!specs.is_empty(), "no embedding GGUFs found");
    let aliases: Vec<String> = specs.iter().map(|s| s.alias.clone()).collect();
    let cfg = LlamaConfig {
        n_gpu_layers,
        ..LlamaConfig::default()
    };
    let engine = LlamaEngine::new(specs, cfg).expect("engine init");

    for alias in &aliases {
        // embeddinggemma's training context is 2048 tokens; keep every input
        // inside each model's limit by trimming the long ones for it.
        let inputs: Vec<String> = inputs()
            .into_iter()
            .map(|s| {
                if alias == "embed-gemma" {
                    s.chars().take(6000).collect()
                } else {
                    s
                }
            })
            .collect();
        let t0 = std::time::Instant::now();
        let batched = embed(&engine, alias, &inputs);
        let batched_ms = t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();
        let single: Vec<Vec<f32>> = inputs
            .iter()
            .map(|s| embed(&engine, alias, std::slice::from_ref(s)).remove(0))
            .collect();
        let single_ms = t1.elapsed().as_millis();
        assert_eq!(batched.len(), inputs.len());
        let mut worst = 1.0f32;
        let mut worst_abs = 0.0f32;
        for (b, s) in batched.iter().zip(&single) {
            assert_eq!(b.len(), s.len());
            worst = worst.min(cosine(b, s));
            let abs = b
                .iter()
                .zip(s)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            worst_abs = worst_abs.max(abs);
        }
        println!(
            "{alias}: {} inputs, batched {batched_ms} ms vs single {single_ms} ms, \
             min cosine {worst:.6}, max |diff| {worst_abs:.2e}",
            inputs.len()
        );
        assert!(worst > 0.9999, "{alias}: batched diverges (cosine {worst})");
    }
}
