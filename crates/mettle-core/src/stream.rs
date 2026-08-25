//! Neutral consumption of [`EngineResponse`] streams — the single shared
//! implementation behind every inference route (Ollama chat/generate, RAG,
//! OpenAI chat/completions, Anthropic messages, OCR).
//!
//! Routes differ only in their wire types; this module owns the parts that
//! must behave identically everywhere:
//! * per-token `on_token` hook firing (streaming and non-streaming),
//! * content accumulation,
//! * the terminal `on_response` hook with abort propagation,
//! * finish-reason / usage bookkeeping.
//!
//! [`engine_stream`] maps `EngineResponse` events to wire chunks via two
//! closures (token chunks, final chunk); [`engine_accumulate`] maps the
//! whole response to one wire object. Hook helpers (`token_info`,
//! `fire_tokens_from_logprobs`, `response_ctx`, `fire_on_response`) live
//! here so routes and tests share one definition.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::engine::{EngineResponse, TokenLogprob};
use crate::hooks::{HookRegistry, RequestCtx, ResponseCtx, TokenInfo, UsageInfo};

// ---------------------------------------------------------------------------
// Hook helpers
// ---------------------------------------------------------------------------

/// Build the `on_token` payload from a token and its (optional) logprob.
pub fn token_info(token: &str, logprob: Option<&TokenLogprob>) -> TokenInfo {
    TokenInfo {
        token: token.to_string(),
        logprob: logprob.map(|l| l.logprob).unwrap_or(0.0),
        top_logprobs: logprob.map(|l| l.top.clone()),
    }
}

/// Fire `on_token` for every token in a non-streaming response's logprob
/// trace (spec §7 guarantee 3).
pub fn fire_tokens_from_logprobs(
    hooks: &HookRegistry,
    ctx: &RequestCtx,
    logprobs: &[TokenLogprob],
) {
    for lp in logprobs {
        hooks.on_token(ctx, &token_info(&lp.token, Some(lp)));
    }
}

/// Build the `on_response` context from the accumulated generation.
pub fn response_ctx(
    model_id: &str,
    content: String,
    finish_reason: String,
    usage: &UsageInfo,
) -> ResponseCtx {
    ResponseCtx {
        model_id: model_id.to_string(),
        content,
        finish_reason,
        usage: UsageInfo {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            prompt_eval_duration_ns: usage.prompt_eval_duration_ns,
            eval_duration_ns: usage.eval_duration_ns,
        },
        metadata: Default::default(),
    }
}

/// Run the response hook, propagating aborts with their status code.
pub fn fire_on_response(
    hooks: &HookRegistry,
    ctx: &RequestCtx,
    model_id: &str,
    content: String,
    finish_reason: String,
    usage: &UsageInfo,
) -> Result<ResponseCtx, (u16, String)> {
    let mut resp_ctx = response_ctx(model_id, content, finish_reason, usage);
    match hooks.on_response(ctx, &mut resp_ctx) {
        crate::hooks::HookOutcome::Abort {
            status_code,
            message,
        } => Err((status_code, message)),
        _ => Ok(resp_ctx),
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// A generated token delta, passed to the route's token-chunk mapper.
pub struct TokenEvt<'a> {
    pub content: &'a str,
    pub logprob: Option<&'a TokenLogprob>,
}

/// The terminal event: final `finish_reason` (post-hook) and usage.
pub struct FinishEvt {
    pub finish_reason: String,
    pub usage: UsageInfo,
}

/// Consume an [`EngineResponse`] channel and map it to wire chunks.
///
/// Fires `on_token` for every non-empty delta and `on_response` exactly once
/// on the terminal chunk, accumulating `full_content` for the hook. The
/// stream ends with the mapped final chunk; hook aborts surface as
/// `Err("aborted by hook: ...")`.
pub fn engine_stream<T, MT, MF>(
    mut rx: mpsc::Receiver<EngineResponse>,
    hooks: Arc<HookRegistry>,
    ctx: RequestCtx,
    model_name: String,
    mut map_token: MT,
    mut map_finish: MF,
) -> impl futures_core::Stream<Item = Result<T, String>>
where
    T: 'static,
    MT: FnMut(TokenEvt<'_>) -> Option<T> + 'static,
    MF: FnMut(FinishEvt) -> T + 'static,
{
    async_stream::stream! {
        let mut full_content = String::new();
        while let Some(response) = rx.recv().await {
            match response {
                EngineResponse::Chunk { content, logprob, finish_reason, usage } => {
                    if !content.is_empty() {
                        hooks.on_token(&ctx, &token_info(&content, logprob.as_ref()));
                        full_content.push_str(&content);
                        if let Some(chunk) = map_token(TokenEvt {
                            content: &content,
                            logprob: logprob.as_ref(),
                        }) {
                            yield Ok(chunk);
                        }
                    }

                    if let Some(reason) = finish_reason {
                        let usage = usage.unwrap_or_default();
                        let resp_ctx = match fire_on_response(
                            &hooks, &ctx, &model_name, full_content.clone(), reason, &usage,
                        ) {
                            Ok(r) => r,
                            Err((_, msg)) => {
                                yield Err(format!("aborted by hook: {msg}"));
                                return;
                            }
                        };
                        yield Ok(map_finish(FinishEvt {
                            finish_reason: resp_ctx.finish_reason,
                            usage,
                        }));
                        return;
                    }
                }
                EngineResponse::Error(e) => {
                    yield Err(e);
                    return;
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

/// The accumulated non-streaming result, passed to the route's mapper.
pub struct Accumulated {
    pub content: String,
    pub finish_reason: String,
    pub usage: UsageInfo,
    /// Full per-token logprob trace from the engine (may be empty if the
    /// backend did not produce one).
    pub logprobs: Vec<TokenLogprob>,
}

/// Consume an [`EngineResponse`] channel and produce one wire object.
///
/// Fires `on_token` per token (via the `Done` logprob trace and any chunks)
/// and `on_response` once at the end. Hook aborts and engine errors become
/// `Err((status, message))`.
pub async fn engine_accumulate<T>(
    rx: mpsc::Receiver<EngineResponse>,
    hooks: Arc<HookRegistry>,
    ctx: RequestCtx,
    model_name: String,
    map_done: impl FnOnce(Accumulated) -> Result<T, (u16, String)>,
) -> Result<T, (u16, String)> {
    let mut full_content = String::new();
    let mut usage = UsageInfo::default();
    let mut finish_reason = String::from("stop");
    let mut trace: Vec<TokenLogprob> = Vec::new();

    let mut rx = rx;
    while let Some(response) = rx.recv().await {
        match response {
            EngineResponse::Done {
                content,
                finish_reason: reason,
                usage: u,
                logprobs,
            } => {
                usage = u;
                full_content.push_str(&content);
                fire_tokens_from_logprobs(&hooks, &ctx, &logprobs);
                trace = logprobs;
                finish_reason = reason;
            }
            EngineResponse::Chunk {
                content, logprob, ..
            } => {
                if !content.is_empty() {
                    full_content.push_str(&content);
                    hooks.on_token(&ctx, &token_info(&content, logprob.as_ref()));
                }
            }
            EngineResponse::Error(e) => {
                return Err((500, e));
            }
            EngineResponse::Embeddings(_) | EngineResponse::Scores(_) => {}
        }
    }

    let resp_ctx = fire_on_response(
        &hooks,
        &ctx,
        &model_name,
        full_content,
        finish_reason,
        &usage,
    )
    .map_err(|(code, msg)| (code, format!("aborted by hook: {msg}")))?;

    map_done(Accumulated {
        content: resp_ctx.content,
        finish_reason: resp_ctx.finish_reason,
        usage,
        logprobs: trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineResponse;
    use crate::hooks::{EngineHook, HookOutcome, RequestCtx, ResponseCtx, TokenInfo};
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::time::SystemTime;
    use tokio::sync::mpsc;
    use tokio_stream::StreamExt;

    /// Records hook calls so tests can assert the shared stream logic fires
    /// `on_token` per token and `on_response` exactly once.
    #[derive(Default)]
    struct RecordingState {
        tokens: Mutex<Vec<String>>,
        responses: Mutex<Vec<(String, String)>>, // (content, finish_reason)
    }

    struct RecordingHook {
        state: Arc<RecordingState>,
        abort_response: bool,
    }

    impl RecordingHook {
        fn new(state: Arc<RecordingState>) -> Self {
            Self {
                state,
                abort_response: false,
            }
        }
    }

    impl EngineHook for RecordingHook {
        fn on_token(&self, _ctx: &RequestCtx, tok: &TokenInfo) {
            self.state.tokens.lock().push(tok.token.clone());
        }
        fn on_response(&self, _ctx: &RequestCtx, resp: &mut ResponseCtx) -> HookOutcome {
            self.state
                .responses
                .lock()
                .push((resp.content.clone(), resp.finish_reason.clone()));
            if self.abort_response {
                HookOutcome::Abort {
                    status_code: 403,
                    message: "blocked".into(),
                }
            } else {
                HookOutcome::Continue
            }
        }
    }

    fn ctx() -> RequestCtx {
        RequestCtx {
            id: crate::hooks::RequestId::new(),
            caller_identity: None,
            model_id: "gemma4:e2b".into(),
            collection_ids: Vec::new(),
            metadata: HashMap::new(),
            start_time: SystemTime::now(),
        }
    }

    fn usage() -> UsageInfo {
        UsageInfo {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn stream_maps_tokens_and_terminal() {
        let mut hooks = HookRegistry::new();
        let state = Arc::new(RecordingState::default());
        hooks.register(Box::new(RecordingHook::new(state.clone())));
        let (tx, rx) = mpsc::channel::<EngineResponse>(8);
        tx.send(EngineResponse::Chunk {
            content: "Hel".into(),
            logprob: None,
            finish_reason: None,
            usage: None,
        })
        .await
        .unwrap();
        tx.send(EngineResponse::Chunk {
            content: "lo".into(),
            logprob: None,
            finish_reason: None,
            usage: None,
        })
        .await
        .unwrap();
        tx.send(EngineResponse::Chunk {
            content: String::new(),
            logprob: None,
            finish_reason: Some("stop".into()),
            usage: Some(usage()),
        })
        .await
        .unwrap();
        drop(tx);

        let stream = engine_stream(
            rx,
            Arc::new(hooks),
            ctx(),
            "gemma4:e2b".into(),
            |evt: TokenEvt<'_>| Some(evt.content.to_string()),
            |evt: FinishEvt| format!("done:{}", evt.finish_reason),
        );
        let items: Vec<String> = stream.map(|r| r.unwrap()).collect().await;
        assert_eq!(items, vec!["Hel", "lo", "done:stop"]);

        // Hooks saw both tokens and the full accumulated response.
        assert_eq!(*state.tokens.lock(), vec!["Hel", "lo"]);
        let responses = state.responses.lock();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0], ("Hello".to_string(), "stop".to_string()));
    }

    #[tokio::test]
    async fn stream_surfaces_engine_error() {
        let (tx, rx) = mpsc::channel::<EngineResponse>(8);
        tx.send(EngineResponse::Chunk {
            content: "Hi".into(),
            logprob: None,
            finish_reason: None,
            usage: None,
        })
        .await
        .unwrap();
        tx.send(EngineResponse::Error("boom".into())).await.unwrap();
        drop(tx);

        let stream = engine_stream(
            rx,
            Arc::new(HookRegistry::new()),
            ctx(),
            "m".into(),
            |evt: TokenEvt<'_>| Some(evt.content.to_string()),
            |evt: FinishEvt| evt.finish_reason,
        );
        let items: Vec<Result<String, String>> = stream.collect().await;
        assert_eq!(items[0].as_deref(), Ok("Hi"));
        assert!(items[1].is_err());
        assert_eq!(items[1].as_ref().unwrap_err(), "boom");
    }

    #[tokio::test]
    async fn stream_aborts_on_hook_rejection() {
        let mut hooks = HookRegistry::new();
        let state = Arc::new(RecordingState::default());
        let mut hook = RecordingHook::new(state.clone());
        hook.abort_response = true;
        hooks.register(Box::new(hook));
        let (tx, rx) = mpsc::channel::<EngineResponse>(8);
        tx.send(EngineResponse::Chunk {
            content: "Hi".into(),
            logprob: None,
            finish_reason: None,
            usage: None,
        })
        .await
        .unwrap();
        tx.send(EngineResponse::Chunk {
            content: String::new(),
            logprob: None,
            finish_reason: Some("stop".into()),
            usage: Some(usage()),
        })
        .await
        .unwrap();
        drop(tx);

        let stream = engine_stream(
            rx,
            Arc::new(hooks),
            ctx(),
            "m".into(),
            |evt: TokenEvt<'_>| Some(evt.content.to_string()),
            |evt: FinishEvt| evt.finish_reason,
        );
        let items: Vec<Result<String, String>> = stream.collect().await;
        // Tokens already emitted stay; the abort cuts the stream at the
        // terminal event.
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_deref(), Ok("Hi"));
        assert!(items[1]
            .as_ref()
            .unwrap_err()
            .starts_with("aborted by hook:"));
    }

    #[tokio::test]
    async fn accumulate_combines_done_and_chunks() {
        let mut hooks = HookRegistry::new();
        let state = Arc::new(RecordingState::default());
        hooks.register(Box::new(RecordingHook::new(state.clone())));
        let (tx, rx) = mpsc::channel::<EngineResponse>(8);
        tx.send(EngineResponse::Chunk {
            content: "Hi".into(),
            logprob: None,
            finish_reason: None,
            usage: None,
        })
        .await
        .unwrap();
        tx.send(EngineResponse::Done {
            content: " there".into(),
            finish_reason: "length".into(),
            usage: usage(),
            logprobs: Vec::new(),
        })
        .await
        .unwrap();
        drop(tx);

        let acc = engine_accumulate(rx, Arc::new(hooks), ctx(), "m".into(), |a: Accumulated| {
            Ok((a.content, a.finish_reason, a.usage.total_tokens))
        })
        .await
        .unwrap();
        assert_eq!(acc, ("Hi there".to_string(), "length".to_string(), 6));
        assert_eq!(*state.tokens.lock(), vec!["Hi"]);
        let responses = state.responses.lock();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].1, "length");
    }

    #[tokio::test]
    async fn accumulate_error_maps_to_500() {
        let (tx, rx) = mpsc::channel::<EngineResponse>(8);
        tx.send(EngineResponse::Error("boom".into())).await.unwrap();
        drop(tx);

        let result = engine_accumulate(
            rx,
            Arc::new(HookRegistry::new()),
            ctx(),
            "m".into(),
            |a: Accumulated| Ok(a.content),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, 500);
    }

    #[tokio::test]
    async fn accumulate_hook_abort_propagates_status() {
        let mut hooks = HookRegistry::new();
        let state = Arc::new(RecordingState::default());
        let mut hook = RecordingHook::new(state.clone());
        hook.abort_response = true;
        hooks.register(Box::new(hook));
        let (tx, rx) = mpsc::channel::<EngineResponse>(8);
        tx.send(EngineResponse::Done {
            content: "x".into(),
            finish_reason: "stop".into(),
            usage: usage(),
            logprobs: Vec::new(),
        })
        .await
        .unwrap();
        drop(tx);

        let result = engine_accumulate(rx, Arc::new(hooks), ctx(), "m".into(), |a: Accumulated| {
            Ok(a.content)
        })
        .await;
        let err = result.unwrap_err();
        assert!(err.1.starts_with("aborted by hook:"));
    }
}
