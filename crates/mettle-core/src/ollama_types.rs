use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Ollama uses nanoseconds everywhere for durations.
pub type Nanoseconds = u64;

/// One-or-many input, used by the embed endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(v) => vec![v],
            OneOrMany::Many(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/chat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub format: Option<FormatOption>,
    #[serde(default)]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// Streaming NDJSON chunk for /api/chat.
/// The final chunk (`done: true`) also carries `done_reason` and timing.
#[derive(Debug, Clone, Serialize)]
pub struct ChatStreamChunk {
    pub model: String,
    pub created_at: String,
    pub message: ChatStreamMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingInfo>,
}

/// Identical shape to [`ResponseMessage`]; kept as an alias so streaming
/// call sites read naturally.
pub type ChatStreamMessage = ResponseMessage;

/// Final response for /api/chat (non-streaming or final streaming chunk).
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ResponseMessage,
    pub done_reason: String,
    pub done: bool,
    #[serde(flatten)]
    pub timing: TimingInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TimingInfo {
    pub total_duration: Nanoseconds,
    pub load_duration: Nanoseconds,
    pub prompt_eval_count: u32,
    pub prompt_eval_duration: Nanoseconds,
    pub eval_count: u32,
    pub eval_duration: Nanoseconds,
}

// ---------------------------------------------------------------------------
// POST /api/generate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub format: Option<FormatOption>,
    #[serde(default)]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub raw: Option<bool>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub keep_alive: Option<String>,
}

/// Streaming NDJSON chunk for /api/generate.
/// The final chunk (`done: true`) also carries `done_reason` and timing.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateStreamChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingInfo>,
}

/// Final response for /api/generate.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub done_reason: String,
    pub context: Vec<u32>,
    #[serde(flatten)]
    pub timing: TimingInfo,
}

// ---------------------------------------------------------------------------
// POST /api/embed
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: OneOrMany<String>,
    #[serde(default)]
    pub truncate: Option<bool>,
    #[serde(default)]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub total_duration: Nanoseconds,
    pub load_duration: Nanoseconds,
    pub prompt_eval_count: u32,
}

// ---------------------------------------------------------------------------
// POST /api/embeddings (legacy)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct LegacyEmbedRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LegacyEmbedResponse {
    pub embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct TagsResponse {
    pub models: Vec<TagModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

// ---------------------------------------------------------------------------
// POST /api/show
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ShowRequest {
    pub model: String,
    #[serde(default)]
    pub verbose: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: ModelDetails,
    pub model_info: serde_json::Value,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

// ---------------------------------------------------------------------------
// GET /api/ps
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct PsResponse {
    pub models: Vec<PsModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PsModelInfo {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
    pub expires_at: String,
    pub size_vram: u64,
}

// ---------------------------------------------------------------------------
// POST /api/pull
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub model: String,
    #[serde(default)]
    pub insecure: Option<bool>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgress {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

// ---------------------------------------------------------------------------
// POST /api/copy, DELETE /api/delete
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeleteRequest {
    pub model: String,
}

// ---------------------------------------------------------------------------
// POST /api/create, POST /api/push (stubs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CreateRequest {
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PushRequest {
    pub model: String,
}

// ---------------------------------------------------------------------------
// GET /api/version
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct VersionResponse {
    pub version: String,
}

// ---------------------------------------------------------------------------
// Format option (shared by chat and generate)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum FormatOption {
    Json,
    Schema(serde_json::Value),
}

impl<'de> Deserialize<'de> for FormatOption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if value == serde_json::Value::String("json".into()) {
            Ok(FormatOption::Json)
        } else {
            Ok(FormatOption::Schema(value))
        }
    }
}
