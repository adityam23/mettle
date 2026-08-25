//! Engine-agnostic inference contract.
//!
//! This is the seam between the HTTP/route layer and whatever inference engine
//! backs it. Routes build an [`EngineRequest`] (which carries its own response
//! channel), send it on the [`Engine`]'s sender, and consume [`EngineResponse`]
//! events — exactly mirroring the shape the route loops already expect. The
//! concrete implementation lives in the `mettle-llama` crate; nothing here
//! depends on llama.cpp (or any engine), so the contract stays stable.

use tokio::sync::mpsc;

use crate::hooks::UsageInfo;
use crate::model_registry::Pooling;
use crate::ollama_types::ChatMessage;

/// Neutral sampling parameters, mapped from Ollama `options` / OpenAI fields.
#[derive(Debug, Clone, Default)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub max_tokens: Option<usize>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub stop: Vec<String>,
}

/// Output-format constraint. Phase 1 always uses `None`; Phase 4 wires JSON /
/// JSON-schema through to a llama.cpp grammar or llguidance.
#[derive(Debug, Clone, Default)]
pub enum GenConstraint {
    #[default]
    None,
    /// Free-form JSON object ("JSON mode").
    Json,
    /// Constrain output to a JSON Schema.
    JsonSchema(serde_json::Value),
}

/// A single generated token's logprob info (top-k alternatives when available).
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub top: Vec<(String, f64)>,
}

/// What kind of generation the request wants.
#[derive(Debug)]
pub enum RequestKind {
    /// Chat with message history; the engine applies the model's chat template.
    Chat {
        messages: Vec<ChatMessage>,
        params: SamplingParams,
        constraint: GenConstraint,
        streaming: bool,
    },
    /// Raw completion over `prompt` (+ optional `suffix` for fill-in-middle).
    Completion {
        prompt: String,
        suffix: Option<String>,
        params: SamplingParams,
        constraint: GenConstraint,
        streaming: bool,
    },
    /// Embed a single input with a resident embedding model.
    Embedding { input: String },
    /// Score `documents` against `query` with a resident reranker
    /// (cross-encoder). One raw relevance score per document, request order.
    Rerank {
        query: String,
        documents: Vec<String>,
    },
    /// OCR a single encoded image (PNG/JPEG) into text with a vision model.
    /// The engine inserts the image at the media marker in `instruction`.
    Ocr {
        instruction: String,
        /// Raw encoded image bytes (PNG/JPEG/BMP/GIF — stb_image formats).
        image: Vec<u8>,
        params: SamplingParams,
        streaming: bool,
    },
}

/// A request submitted to the engine. Carries the model alias and its own
/// response channel so the worker can stream results back.
#[derive(Debug)]
pub struct EngineRequest {
    /// Resident model alias to run against (resolved by the registry upstream).
    pub model: String,
    pub kind: RequestKind,
    /// Where the engine sends [`EngineResponse`] events for this request.
    pub response: mpsc::Sender<EngineResponse>,
}

/// An event emitted by the engine for a request.
#[derive(Debug, Clone)]
pub enum EngineResponse {
    /// Streaming token delta. The final streamed chunk carries `finish_reason`
    /// (and `usage`); intermediate chunks leave both `None`.
    Chunk {
        content: String,
        logprob: Option<TokenLogprob>,
        finish_reason: Option<String>,
        usage: Option<UsageInfo>,
    },
    /// Non-streaming completion: the whole content, final usage, and the full
    /// per-token logprob trace (so per-token hooks can still fire — §7 g.3).
    Done {
        content: String,
        finish_reason: String,
        usage: UsageInfo,
        logprobs: Vec<TokenLogprob>,
    },
    /// One embedding vector (for `RequestKind::Embedding`).
    Embeddings(Vec<f32>),
    /// Rerank scores for `RequestKind::Rerank` — one raw relevance score
    /// per document, in request order.
    Scores(Vec<f32>),
    /// Terminal error for this request.
    Error(String),
}

/// Errors from engine handle operations (not per-request generation errors,
/// which travel as [`EngineResponse::Error`]).
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("model '{0}' is not loaded")]
    ModelNotLoaded(String),
    #[error("{0}")]
    Other(String),
}

/// What role a loaded model plays; decides which request kinds it serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Generation,
    /// Embedding model pooled per its training recipe (never [`Pooling::Rank`]).
    Embedding(Pooling),
    /// Cross-encoder reranker — [`RequestKind::Rerank`] only.
    Rerank,
}

/// Everything the engine needs to load a model: the alias to register it under
/// and the local GGUF path (resolved upstream from the HF cache by the server).
#[derive(Debug, Clone)]
pub struct ModelLoadSpec {
    pub alias: String,
    pub gguf_path: std::path::PathBuf,
    pub kind: ModelKind,
    /// Per-model GPU offload override (defaults to the engine-wide value).
    pub n_gpu_layers: Option<u32>,
    /// Multimodal projector (mmproj) path for vision/OCR models.
    pub mmproj_path: Option<std::path::PathBuf>,
}

/// One profile/JIT slot's full load instruction for a running engine:
/// the model plus everything that must come up atomically with it.
/// [`Engine::load_model`] stays the plain-model path (`/api/pull`);
/// the profile manager uses [`Engine::load_slot`] so drafter attachment
/// and context-shrink degradation ride the same worker message.
#[derive(Debug, Clone)]
pub struct SlotLoadSpec {
    pub model: ModelLoadSpec,
    /// Same-size MTP drafter; loads with `model` and drives an MTP lane.
    pub mtp_draft: Option<ModelLoadSpec>,
    /// Standard speculative drafter (greedy acceptance in the scheduler);
    /// loads with `model`.
    pub draft: Option<ModelLoadSpec>,
    /// Context override for degradation-ladder shrink. `None` = engine
    /// default ([engine] n_ctx).
    pub n_ctx: Option<u32>,
}

/// Score `documents` against `query` with a resident reranker. One raw
/// relevance score per document, in request order. Shared by every rerank
/// surface (`/v1/rerank`).
pub async fn rerank_texts(
    engine: &dyn Engine,
    model_alias: &str,
    query: &str,
    documents: &[String],
) -> Result<Vec<f32>, String> {
    let sender = engine.sender();
    let (tx, mut rx) = mpsc::channel::<EngineResponse>(1);
    let req = EngineRequest {
        model: model_alias.to_string(),
        kind: RequestKind::Rerank {
            query: query.to_string(),
            documents: documents.to_vec(),
        },
        response: tx,
    };
    sender.send(req).await.map_err(|e| e.to_string())?;
    match rx.recv().await {
        Some(EngineResponse::Scores(v)) => Ok(v),
        Some(EngineResponse::Error(e)) => Err(e),
        other => Err(format!("unexpected rerank response: {other:?}")),
    }
}

/// Embed `texts` with a resident embedding model. One vector per input.
/// Shared by every embedding surface (Ollama, OpenAI, RAG).
pub async fn embed_texts(
    engine: &dyn Engine,
    model_alias: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    let sender = engine.sender();
    let mut out = Vec::with_capacity(texts.len());
    for text in texts {
        let (tx, mut rx) = mpsc::channel::<EngineResponse>(1);
        let req = EngineRequest {
            model: model_alias.to_string(),
            kind: RequestKind::Embedding {
                input: text.clone(),
            },
            response: tx,
        };
        sender.send(req).await.map_err(|e| e.to_string())?;
        match rx.recv().await {
            Some(EngineResponse::Embeddings(v)) => out.push(v),
            Some(EngineResponse::Error(e)) => return Err(e),
            other => return Err(format!("unexpected embedding response: {other:?}")),
        }
    }
    Ok(out)
}

/// The engine handle the server shares via `AppState`. Implemented by
/// `mettle_llama::LlamaEngine`. Cloning the sender is how routes submit work.
pub trait Engine: Send + Sync {
    /// A sender for submitting requests. Each [`EngineRequest`] names its model.
    fn sender(&self) -> mpsc::Sender<EngineRequest>;

    /// Aliases currently resident in the engine.
    fn loaded(&self) -> Vec<String>;

    /// Load a model into the running engine (used by `/api/pull`). Blocks until
    /// the load completes or fails.
    fn load_model(&self, spec: ModelLoadSpec) -> Result<(), EngineError>;

    /// Unload a model by alias. No-op-success if it was not resident.
    fn remove_model(&self, alias: &str) -> Result<(), EngineError>;

    /// Load a slot with engine-specific attachments (drafters, ctx
    /// override). Default: engines that don't implement slot-level
    /// overrides still load the model itself.
    fn load_slot(&self, slot: SlotLoadSpec) -> Result<(), EngineError> {
        self.load_model(slot.model)
    }
}
