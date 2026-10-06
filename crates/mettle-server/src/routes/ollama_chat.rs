use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Extension, Json};
use mettle_core::engine::{EngineRequest, EngineResponse, RequestKind};
use mettle_core::hooks::RequestCtx;
use mettle_core::ollama_types::{
    ChatRequest, ChatResponse, ChatStreamChunk, ChatStreamMessage, ResponseMessage,
};
use mettle_core::stream::{engine_accumulate, engine_stream, Accumulated, FinishEvt, TokenEvt};
use mettle_core::timing::timestamp_iso8601;
use tokio::sync::mpsc;

use super::error::{api_error, api_error_code};
use crate::llama_bridge::{convert_format, convert_options, usage_to_timing};
use crate::response_stream::NdJsonStream;
use crate::state::AppState;

/// POST /api/chat
pub async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    let (entry, sender) = match state.sender_for(&req.model) {
        Ok(v) => v,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let is_streaming = req.stream.unwrap_or(false);
    let params = convert_options(&req.options);
    let constraint = convert_format(req.format.as_ref());

    let (tx, rx) = mpsc::channel::<EngineResponse>(256);
    let engine_req = EngineRequest {
        model: entry.alias,
        kind: RequestKind::Chat {
            messages: req.messages,
            params,
            constraint,
            streaming: is_streaming,
        },
        response: tx,
    };

    if let Err(e) = sender.send(engine_req).await {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    let model_name = req.model;
    let hooks = state.hooks.clone();

    if is_streaming {
        let stream = engine_stream(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            {
                let model_name = model_name.clone();
                move |evt: TokenEvt<'_>| Some(chat_token_chunk(model_name.clone(), evt))
            },
            move |evt: FinishEvt| chat_final_chunk(model_name.clone(), evt),
        );
        NdJsonStream::new(stream).into_response()
    } else {
        match engine_accumulate(rx, hooks, ctx, model_name.clone(), |acc: Accumulated| {
            Ok(chat_response(model_name, acc))
        })
        .await
        {
            Ok(resp) => Json(resp).into_response(),
            Err((status, msg)) => api_error_code(status, msg).into_response(),
        }
    }
}

fn chat_token_chunk(model: String, evt: TokenEvt<'_>) -> ChatStreamChunk {
    ChatStreamChunk {
        model,
        created_at: timestamp_iso8601(),
        message: ChatStreamMessage {
            role: "assistant".into(),
            content: evt.content.to_string(),
            tool_calls: None,
        },
        done: false,
        done_reason: None,
        timing: None,
    }
}

fn chat_final_chunk(model: String, evt: FinishEvt) -> ChatStreamChunk {
    ChatStreamChunk {
        model,
        created_at: timestamp_iso8601(),
        message: ChatStreamMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: None,
        },
        done: true,
        done_reason: Some(evt.finish_reason),
        timing: Some(usage_to_timing(&evt.usage)),
    }
}

fn chat_response(model: String, acc: Accumulated) -> ChatResponse {
    ChatResponse {
        model,
        created_at: timestamp_iso8601(),
        message: ResponseMessage {
            role: "assistant".into(),
            content: acc.content,
            tool_calls: None,
        },
        done_reason: acc.finish_reason,
        done: true,
        timing: usage_to_timing(&acc.usage),
    }
}
