//! Anthropic-compatible wire types for `POST /v1/messages` (+ SSE streaming),
//! matching the shape Anthropic SDKs expect (content blocks, stop_reason,
//! `message_delta` SSE events with usage).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// POST /v1/messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    /// Maximum tokens to generate (required by the Anthropic API).
    pub max_tokens: u32,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---------------------------------------------------------------------------
// SSE events (stream: true)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageStreamEvent {
    MessageStart {
        message: MessageStartBlock,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: TextDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaBlock,
        usage: AnthropicUsage,
    },
    MessageStop,
    Ping,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStartBlock {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextDelta {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaBlock {
    pub stop_reason: String,
}
