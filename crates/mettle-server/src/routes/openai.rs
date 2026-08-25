//! OpenAI-compatible `/v1/*` routes: chat completions (SSE + non-stream),
//! text completions, embeddings, and model listing. Consumed by standard
//! OpenAI SDK clients.

use std::sync::Arc;

use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use mettle_core::engine::{
    embed_texts, rerank_texts, EngineRequest, EngineResponse, GenConstraint, RequestKind,
    SamplingParams,
};
use mettle_core::hooks::RequestCtx;
use mettle_core::ollama_types::{ChatMessage, OneOrMany};
use mettle_core::openai_types::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatLogprobs, CompletionChoice, CompletionChunk,
    CompletionChunkChoice, CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, LogprobContent, ModelInfo, ModelsResponse, RerankRequest, RerankResponse,
    RerankResult, ResponseMessage, TopLogprob, Usage,
};
use mettle_core::reasoning::{split_reasoning, ReasoningFilter};
use mettle_core::stream::{engine_accumulate, engine_stream, Accumulated, FinishEvt, TokenEvt};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::error::{api_error, api_error_code};
use crate::response_stream::SseStream;
use crate::state::AppState;
use futures::StreamExt;

/// Map OpenAI sampling fields to the neutral params.
fn openai_params(
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<usize>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
) -> SamplingParams {
    SamplingParams {
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        stop: stop.unwrap_or_default(),
        ..Default::default()
    }
}

/// Map `response_format` to an engine output constraint.
fn convert_response_format(v: Option<&serde_json::Value>) -> GenConstraint {
    let Some(v) = v else {
        return GenConstraint::None;
    };
    match v.get("type").and_then(|t| t.as_str()) {
        Some("json_object") => GenConstraint::Json,
        Some("json_schema") => v
            .get("json_schema")
            .cloned()
            .map(GenConstraint::JsonSchema)
            .unwrap_or(GenConstraint::Json),
        _ => GenConstraint::None,
    }
}

/// Build the per-token `logprobs` payload for one streamed token.
fn evt_logprobs(token: &str, lp: Option<&mettle_core::engine::TokenLogprob>) -> ChatLogprobs {
    let entry = match lp {
        Some(lp) => LogprobContent {
            token: lp.token.clone(),
            logprob: lp.logprob,
            top_logprobs: lp
                .top
                .iter()
                .map(|(t, p)| TopLogprob {
                    token: t.clone(),
                    logprob: *p,
                })
                .collect(),
        },
        None => LogprobContent {
            token: token.to_string(),
            logprob: 0.0,
            top_logprobs: Vec::new(),
        },
    };
    ChatLogprobs {
        content: vec![entry],
    }
}

/// POST /v1/chat/completions
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    if req.n.unwrap_or(1) != 1 {
        return api_error(StatusCode::BAD_REQUEST, "only n=1 is supported".to_string())
            .into_response();
    }
    let model_id = state.profiles.resolve_alias(&req.model);
    let (entry, sender) = match state.v1_sender_for(&model_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let alias = entry.alias.clone();
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            images: None,
            tool_calls: None,
        })
        .collect();
    let params = openai_params(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.stop.map(|s| s.into_vec()),
        req.presence_penalty,
        req.frequency_penalty,
    );
    let constraint = convert_response_format(req.response_format.as_ref());

    let (tx, rx) = mpsc::channel::<EngineResponse>(256);
    let engine_req = EngineRequest {
        model: entry.alias,
        kind: RequestKind::Chat {
            messages,
            params,
            constraint,
            streaming: req.stream.unwrap_or(false),
        },
        response: tx,
    };
    if let Err(e) = sender.send(engine_req).await {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    let model_name = req.model;
    let hooks = state.hooks.clone();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let want_logprobs = req.logprobs.unwrap_or(false);

    if req.stream.unwrap_or(false) {
        // Reasoning models: stream the answer channel only. The filter holds
        // back bytes while a thinking-channel tag is still possible; whatever
        // it still withholds at end of stream rides on the finish chunk's
        // delta (OpenAI clients concatenate deltas, so nothing is lost).
        let filter = std::sync::Arc::new(std::sync::Mutex::new(ReasoningFilter::new()));
        let stream = engine_stream(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            {
                let model_name = model_name.clone();
                let id = id.clone();
                let filter = filter.clone();
                move |evt: TokenEvt<'_>| {
                    let (out, merged) = {
                        let mut f = filter.lock().unwrap();
                        let out = f.push(evt.content);
                        (out, f.last_push_was_merged())
                    };
                    if out.is_empty() {
                        None
                    } else {
                        // A chunk that merges bytes held across pushes spans
                        // several tokens; attaching one token's logprob would
                        // fabricate a pairing, so we omit it instead.
                        Some(ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created: now(),
                            model: model_name.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: ChatDelta {
                                    role: Some("assistant".into()),
                                    content: Some(out),
                                    reasoning_content: None,
                                },
                                finish_reason: None,
                                logprobs: (want_logprobs && !merged)
                                    .then(|| evt_logprobs(evt.content, evt.logprob)),
                            }],
                        })
                    }
                }
            },
            move |evt: FinishEvt| {
                let held = filter.lock().unwrap().finish();
                ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created: now(),
                    model: model_name.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: (!held.is_empty()).then_some(held),
                            reasoning_content: None,
                        },
                        finish_reason: Some(evt.finish_reason),
                        logprobs: None,
                    }],
                }
            },
        );
        let stream = stream.map({
            let state = state.clone();
            let alias = alias.clone();
            move |item| {
                if let Err(e) = &item {
                    state.profiles.note_runtime_failure(&alias, e);
                }
                item
            }
        });
        SseStream::new(stream, false)
            .with_done("[DONE]")
            .into_response()
    } else {
        match engine_accumulate(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            move |acc: Accumulated| {
                // Reasoning models: `content` is the answer only; the thinking
                // channel moves to `message.reasoning_content` (contract:
                // agent-docs/downstream-consumers.md).
                let split = split_reasoning(&acc.content);
                // Non-streaming splits text after the fact, so the engine's
                // per-token logprob trace (still reasoning-token-aligned) no
                // longer matches `content`. Suppress it rather than pair it
                // wrongly — the consumer degrades a missing block to 'unknown'.
                let has_reasoning = split.reasoning.is_some();
                Ok(ChatCompletionResponse {
                    id: id.clone(),
                    object: "chat.completion",
                    created: now(),
                    model: model_name.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ResponseMessage {
                            role: "assistant".into(),
                            content: split.content,
                            reasoning_content: split.reasoning,
                        },
                        finish_reason: acc.finish_reason,
                        logprobs: (want_logprobs && !has_reasoning).then(|| ChatLogprobs {
                            content: acc
                                .logprobs
                                .iter()
                                .map(|lp| LogprobContent {
                                    token: lp.token.clone(),
                                    logprob: lp.logprob,
                                    top_logprobs: lp
                                        .top
                                        .iter()
                                        .map(|(t, p)| TopLogprob {
                                            token: t.clone(),
                                            logprob: *p,
                                        })
                                        .collect(),
                                })
                                .collect(),
                        }),
                    }],
                    usage: Usage::from_usage(&acc.usage),
                })
            },
        )
        .await
        {
            Ok(resp) => Json(resp).into_response(),
            Err((status, msg)) => {
                state.profiles.note_runtime_failure(&alias, &msg);
                api_error_code(status, msg).into_response()
            }
        }
    }
}

/// POST /v1/completions
pub async fn completions_handler(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    // Only the first prompt is supported (OpenAI allows batches); validate
    // the payload before touching the engine.
    let prompt = match req.prompt {
        OneOrMany::One(p) => p,
        OneOrMany::Many(mut v) if v.len() == 1 => v.remove(0),
        OneOrMany::Many(v) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("only one prompt is supported, got {}", v.len()),
            )
            .into_response()
        }
    };

    let model_id = state.profiles.resolve_alias(&req.model);
    let (entry, sender) = match state.v1_sender_for(&model_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let alias = entry.alias.clone();
    let params = openai_params(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.stop.map(|s| s.into_vec()),
        req.presence_penalty,
        req.frequency_penalty,
    );

    let (tx, rx) = mpsc::channel::<EngineResponse>(256);
    let engine_req = EngineRequest {
        model: entry.alias,
        kind: RequestKind::Completion {
            prompt,
            suffix: None,
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
    let id = format!("cmpl-{}", Uuid::new_v4());

    if req.stream.unwrap_or(false) {
        let stream = engine_stream(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            {
                let model_name = model_name.clone();
                let id = id.clone();
                move |evt: TokenEvt<'_>| {
                    if evt.content.is_empty() {
                        None
                    } else {
                        Some(CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created: now(),
                            model: model_name.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: evt.content.to_string(),
                                index: 0,
                                finish_reason: None,
                            }],
                        })
                    }
                }
            },
            move |evt: FinishEvt| CompletionChunk {
                id: id.clone(),
                object: "text_completion",
                created: now(),
                model: model_name.clone(),
                choices: vec![CompletionChunkChoice {
                    text: String::new(),
                    index: 0,
                    finish_reason: Some(evt.finish_reason),
                }],
            },
        );
        let stream = stream.map({
            let state = state.clone();
            let alias = alias.clone();
            move |item| {
                if let Err(e) = &item {
                    state.profiles.note_runtime_failure(&alias, e);
                }
                item
            }
        });
        SseStream::new(stream, false)
            .with_done("[DONE]")
            .into_response()
    } else {
        match engine_accumulate(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            move |acc: Accumulated| {
                Ok(CompletionResponse {
                    id: id.clone(),
                    object: "text_completion",
                    created: now(),
                    model: model_name.clone(),
                    choices: vec![CompletionChoice {
                        text: acc.content,
                        index: 0,
                        finish_reason: acc.finish_reason,
                    }],
                    usage: Usage::from_usage(&acc.usage),
                })
            },
        )
        .await
        {
            Ok(resp) => Json(resp).into_response(),
            Err((status, msg)) => {
                state.profiles.note_runtime_failure(&alias, &msg);
                api_error_code(status, msg).into_response()
            }
        }
    }
}

/// POST /v1/rerank — cross-encoder scoring in the Cohere/Jina/llama.cpp
/// shape (agent-docs/rerank-contract.md). Results sorted by
/// `relevance_score` descending; `index` maps back into the request's
/// `documents`. Raw model scores — not calibrated across queries.
pub async fn rerank_handler(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    // Parse manually so a malformed body is a 400 (contract: errors mirror
    // the OCR contract) instead of axum's default JSON-rejection 422.
    let req: RerankRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("invalid request body: {e}"),
            )
            .into_response()
        }
    };

    // Empty documents is a valid no-op per the contract.
    if req.documents.is_empty() {
        return Json(RerankResponse {
            model: req.model,
            results: Vec::new(),
        })
        .into_response();
    }

    // Resolve + kind gate BEFORE the availability checks: a model that
    // cannot serve this endpoint is a resolution error (404) even when no
    // engine is loaded.
    let model_id = state.profiles.resolve_alias(&req.model);
    let entry = match state.resolve_model(&model_id) {
        Ok(v) => v,
        Err(e) => {
            return api_error(StatusCode::NOT_FOUND, e.to_string()).into_response();
        }
    };
    let is_reranker = matches!(
        entry.loader_kind,
        mettle_core::model_registry::LoaderKind::Rerank { .. }
    );
    if !is_reranker {
        return api_error(
            StatusCode::NOT_FOUND,
            format!("model '{}' is not a reranker", entry.alias),
        )
        .into_response();
    }

    let (_entry, engine) = match state.v1_engine_for(&model_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match rerank_texts(engine.as_ref(), &entry.alias, &req.query, &req.documents).await {
        Ok(mut scores) => {
            if scores.len() != req.documents.len() {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(
                        "reranker returned {} scores for {} documents",
                        scores.len(),
                        req.documents.len()
                    ),
                )
                .into_response();
            }
            let return_documents = req.return_documents.unwrap_or(false);
            let mut results: Vec<RerankResult> = scores
                .drain(..)
                .enumerate()
                .map(|(i, score)| RerankResult {
                    index: i as u32,
                    relevance_score: score as f64,
                    document: return_documents.then(|| req.documents[i].clone()),
                })
                .collect();
            // Descending by score; stable so equal scores keep request order.
            results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
            if let Some(top_n) = req.top_n {
                results.truncate(top_n);
            }
            Json(RerankResponse {
                model: req.model,
                results,
            })
            .into_response()
        }
        Err(e) => {
            state.profiles.note_runtime_failure(&entry.alias, &e);
            api_error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response()
        }
    }
}

/// POST /v1/embeddings
pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> axum::response::Response {
    // Kind gate BEFORE availability: a reranker can never serve this
    // endpoint, so it is a resolution error (404) even with no engine loaded.
    let model_id = state.profiles.resolve_alias(&req.model);
    let entry = match state.resolve_model(&model_id) {
        Ok(v) => v,
        Err(e) => {
            return api_error(StatusCode::NOT_FOUND, e.to_string()).into_response();
        }
    };
    if matches!(
        entry.loader_kind,
        mettle_core::model_registry::LoaderKind::Rerank { .. }
    ) {
        return api_error(
            StatusCode::NOT_FOUND,
            format!("model '{}' is a reranker; use /v1/rerank", entry.alias),
        )
        .into_response();
    }

    let (_entry, engine) = match state.v1_engine_for(&model_id) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let inputs = req.input.into_vec();
    match embed_texts(engine.as_ref(), &entry.alias, &inputs).await {
        Ok(embeddings) => Json(EmbeddingResponse {
            object: "list",
            data: embeddings
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| EmbeddingData {
                    object: "embedding",
                    embedding,
                    index: i as u32,
                })
                .collect(),
            model: req.model,
            usage: Usage {
                prompt_tokens: inputs.len() as u32,
                completion_tokens: 0,
                total_tokens: inputs.len() as u32,
            },
        })
        .into_response(),
        Err(e) => {
            state.profiles.note_runtime_failure(&entry.alias, &e);
            api_error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response()
        }
    }
}

/// GET /v1/models — every registered alias (loaded or not), matching the
/// OpenAI `list models` shape.
pub async fn models_handler(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let resident = state.engine().map(|e| e.loaded()).unwrap_or_default();
    let data = match state.registry.read() {
        Ok(r) => r
            .list()
            .into_iter()
            .map(|e| ModelInfo {
                id: e.alias.clone(),
                object: "model",
                created: now(),
                owned_by: "mettle".into(),
                loaded: resident.iter().any(|a| a == &e.alias),
                vram_bytes: state.profiles.vram_bytes(&e.alias),
            })
            .collect::<Vec<_>>(),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                String::from("lock poisoned"),
            )
            .into_response()
        }
    };
    Json(ModelsResponse {
        object: "list",
        data,
    })
    .into_response()
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
