//! OpenAI-compatible wire types (`/v1/*`), matching the shapes agentic
//! clients (including agent-assistant's `openai_compat.py`) expect.
//!
//! Request types deliberately do NOT use `deny_unknown_fields`: drop-in
//! clients send extra fields (tools, seed, logprobs, ...) that we accept and
//! ignore rather than reject.

use serde::{Deserialize, Serialize};

use crate::ollama_types::OneOrMany;

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Number of completions per prompt. Only `1` (or unset) is supported.
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Accepted for compatibility; structured output enforcement is Phase 4.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// OpenAI per-token logprobs flag. When true the response carries
    /// `choices[].logprobs.content[]` with each token's logprob.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Optional count of top alternatives per token (we always emit what the
    /// engine has when `logprobs` is true).
    #[serde(default)]
    pub top_logprobs: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// OpenAI `choices[].logprobs` payload.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<LogprobContent>,
}

/// One token's logprob entry (OpenAI `logprobs.content[]` element).
#[derive(Debug, Clone, Serialize)]
pub struct LogprobContent {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
    /// Reasoning-model thinking channel, separated from `content`
    /// (agent-docs/downstream-consumers.md). Absent for non-reasoning output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// One SSE `data:` frame for streaming chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Reserved for reasoning-model deltas; mettle streams the answer
    /// channel only, so this stays absent today.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

// ---------------------------------------------------------------------------
// POST /v1/completions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: OneOrMany<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: OneOrMany<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: u32,
}

// ---------------------------------------------------------------------------
// POST /v1/rerank (Cohere/Jina/llama.cpp convention — see
// agent-docs/rerank-contract.md)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    /// Return only the top `n` results (default: all, sorted).
    pub top_n: Option<usize>,
    /// Echo each result's document text back (default false — the consumer
    /// already has the text).
    pub return_documents: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    pub model: String,
    /// Sorted by `relevance_score` descending; `index` refers to the
    /// request's `documents` order.
    pub results: Vec<RerankResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResult {
    pub index: u32,
    pub relevance_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
    /// Profiles extension: residency right now. OpenAI SDKs ignore extra
    /// fields; mettle's consumer greys out unloaded models.
    pub loaded: bool,
    /// Measured-or-estimated VRAM footprint; null when unknown or CPU-only.
    pub vram_bytes: Option<u64>,
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    /// Map from the engine's neutral usage info.
    pub fn from_usage(u: &crate::hooks::UsageInfo) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}
