//! Anthropic-compatible `POST /v1/messages` (+ SSE streaming), matching the
//! shape Anthropic SDKs expect. System prompts become a system-role message;
//! `max_tokens` is required per the Anthropic API.

use std::sync::Arc;

use axum::{
    extract::{State, Extension},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use mettle_core::anthropic_types::{
    AnthropicMessage, AnthropicUsage, ContentBlock, MessageDeltaBlock, MessageStartBlock,
    MessagesRequest, MessagesResponse, MessageStreamEvent, TextDelta,
};
use mettle_core::engine::{EngineRequest, EngineResponse, GenConstraint, RequestKind, SamplingParams};
use mettle_core::hooks::RequestCtx;
use mettle_core::ollama_types::ChatMessage;
use mettle_core::stream::{engine_accumulate, engine_stream, Accumulated, FinishEvt, TokenEvt};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::error::{api_error, api_error_code};
use crate::response_stream::SseStream;
use crate::state::AppState;

/// POST /v1/messages
pub async fn messages_handler(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Json(req): Json<MessagesRequest>,
) -> axum::response::Response {
    let (entry, sender) = match state.sender_for(&req.model) {
        Ok(v) => v,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(system) = &req.system {
        if !system.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: system.clone(),
                images: None,
                tool_calls: None,
            });
        }
    }
    messages.extend(req.messages.into_iter().map(|m: AnthropicMessage| ChatMessage {
        role: m.role,
        content: m.content,
        images: None,
        tool_calls: None,
    }));
    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: Some(req.max_tokens as usize),
        stop: req.stop_sequences,
        ..Default::default()
    };

    let (tx, rx) = mpsc::channel::<EngineResponse>(256);
    let engine_req = EngineRequest {
        model: entry.alias,
        kind: RequestKind::Chat {
            messages,
            params,
            constraint: GenConstraint::None,
            streaming: req.stream.unwrap_or(false),
        },
        response: tx,
    };
    if let Err(e) = sender.send(engine_req).await {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    let model_name = req.model;
    let hooks = state.hooks.clone();
    let id = format!("msg_{}", Uuid::new_v4());

    if req.stream.unwrap_or(false) {
        let start_block = MessageStartBlock {
            id: id.clone(),
            type_: "message",
            role: "assistant",
            content: Vec::new(),
            model: model_name.clone(),
            stop_reason: None,
            usage: AnthropicUsage::default(),
        };
        let stream = engine_stream(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            {
                let start_block = start_block;
                let mut started = false;
                move |evt: TokenEvt<'_>| {
                    let mut events = Vec::new();
                    if !started {
                        started = true;
                        events.push(MessageStreamEvent::MessageStart {
                            message: start_block.clone(),
                        });
                        events.push(MessageStreamEvent::ContentBlockStart {
                            index: 0,
                            content_block: ContentBlock {
                                type_: "text",
                                text: String::new(),
                            },
                        });
                    }
                    if !evt.content.is_empty() {
                        events.push(MessageStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: TextDelta {
                                type_: "text_delta",
                                text: evt.content.to_string(),
                            },
                        });
                    }
                    Some(events)
                }
            },
            move |evt: FinishEvt| {
                vec![
                    MessageStreamEvent::ContentBlockStop { index: 0 },
                    MessageStreamEvent::MessageDelta {
                        delta: MessageDeltaBlock {
                            stop_reason: evt.finish_reason,
                        },
                        usage: AnthropicUsage {
                            input_tokens: evt.usage.prompt_tokens,
                            output_tokens: evt.usage.completion_tokens,
                        },
                    },
                    MessageStreamEvent::MessageStop,
                ]
            },
        );
        SseStream::new(stream, true).into_response()
    } else {
        match engine_accumulate(rx, hooks, ctx, model_name.clone(), move |acc: Accumulated| {
            Ok(MessagesResponse {
                id: id.clone(),
                type_: "message",
                role: "assistant",
                content: vec![ContentBlock {
                    type_: "text",
                    text: acc.content,
                }],
                model: model_name.clone(),
                stop_reason: acc.finish_reason,
                usage: AnthropicUsage {
                    input_tokens: acc.usage.prompt_tokens,
                    output_tokens: acc.usage.completion_tokens,
                },
            })
        })
        .await
        {
            Ok(resp) => Json(resp).into_response(),
            Err((status, msg)) => api_error_code(status, msg).into_response(),
        }
    }
}
