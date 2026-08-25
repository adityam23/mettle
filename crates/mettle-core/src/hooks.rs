use std::collections::HashMap;
use std::time::SystemTime;

/// Unique request identifier, traceable through logs, metrics, and hook callbacks.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Context provided to every hook invocation for a request.
/// Hooks may mutate the metadata map; all other fields are read-only
/// (or should be treated as such).
#[derive(Debug, Clone)]
pub struct RequestCtx {
    pub id: RequestId,
    pub caller_identity: Option<String>,
    pub model_id: String,
    pub collection_ids: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub start_time: SystemTime,
}

/// Information about a single generated token, surfaced to streaming hooks.
#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub token: String,
    pub logprob: f64,
    pub top_logprobs: Option<Vec<(String, f64)>>,
}

/// A retrieval hit from a collection search. Structurally identical to
/// [`crate::retrieval::ScoredChunk`]; the alias keeps the §7 hook-contract
/// name stable while avoiding field-by-field conversions.
pub type Hit = crate::retrieval::ScoredChunk;

/// Context for the final response, mutable by `on_response` hooks.
#[derive(Debug, Clone)]
pub struct ResponseCtx {
    pub model_id: String,
    pub content: String,
    pub finish_reason: String,
    pub usage: UsageInfo,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_eval_duration_ns: u64,
    pub eval_duration_ns: u64,
}

/// Returned by every hook method to signal whether processing should continue
/// or the request should be short-circuited.
#[derive(Debug, Clone)]
pub enum HookOutcome {
    /// Proceed normally.
    Continue,
    /// Abort the request with an HTTP status code and message.
    Abort { status_code: u16, message: String },
}

/// The extension hook contract.
///
/// Stable from v0.1, semver-protected. Every method has a default no-op
/// implementation so implementors only override what they need.
///
/// Hooks fire on **all** API surfaces uniformly: Ollama routes,
/// OpenAI/Anthropic routes, and RAG routes.
pub trait EngineHook: Send + Sync {
    /// Inspect, rewrite, or reject an inbound request.
    /// Return `Abort` to short-circuit with an error response.
    fn on_request(&self, _ctx: &mut RequestCtx) -> HookOutcome {
        HookOutcome::Continue
    }

    /// Filter or re-rank retrieval results before they are passed to
    /// the model or returned to the caller.
    fn on_retrieval(&self, _ctx: &RequestCtx, _hits: &mut Vec<Hit>) {}

    /// Streaming tap — fires for every generated token.
    /// Token logprobs are always available regardless of whether
    /// the client requested them.
    fn on_token(&self, _ctx: &RequestCtx, _tok: &TokenInfo) {}

    /// Annotate, transform, or block the final response.
    fn on_response(&self, _ctx: &RequestCtx, _resp: &mut ResponseCtx) -> HookOutcome {
        HookOutcome::Continue
    }
}

/// Ordered registry of compiled-in [`EngineHook`] implementations.
/// Hooks are invoked in registration order.
pub struct HookRegistry {
    hooks: Vec<Box<dyn EngineHook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a hook. Hooks fire in registration order.
    pub fn register(&mut self, hook: Box<dyn EngineHook>) {
        self.hooks.push(hook);
    }

    /// Run `on_request` across all hooks. The first `Abort` short-circuits.
    pub fn on_request(&self, ctx: &mut RequestCtx) -> HookOutcome {
        for hook in &self.hooks {
            match hook.on_request(ctx) {
                HookOutcome::Continue => continue,
                abort => return abort,
            }
        }
        HookOutcome::Continue
    }

    /// Run `on_retrieval` across all hooks. Each hook may mutate `hits`.
    pub fn on_retrieval(&self, ctx: &RequestCtx, hits: &mut Vec<Hit>) {
        for hook in &self.hooks {
            hook.on_retrieval(ctx, hits);
        }
    }

    /// Run `on_token` across all hooks (streaming tap).
    pub fn on_token(&self, ctx: &RequestCtx, tok: &TokenInfo) {
        for hook in &self.hooks {
            hook.on_token(ctx, tok);
        }
    }

    /// Run `on_response` across all hooks. The first `Abort` short-circuits.
    pub fn on_response(&self, ctx: &RequestCtx, resp: &mut ResponseCtx) -> HookOutcome {
        for hook in &self.hooks {
            match hook.on_response(ctx, resp) {
                HookOutcome::Continue => continue,
                abort => return abort,
            }
        }
        HookOutcome::Continue
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LoggingHook;
    impl EngineHook for LoggingHook {
        fn on_request(&self, ctx: &mut RequestCtx) -> HookOutcome {
            ctx.metadata
                .insert("seen_by_hook".into(), "true".into());
            HookOutcome::Continue
        }

        fn on_response(&self, _ctx: &RequestCtx, resp: &mut ResponseCtx) -> HookOutcome {
            resp.metadata
                .insert("annotated".into(), "1".into());
            HookOutcome::Continue
        }
    }

    struct BlockHook;
    impl EngineHook for BlockHook {
        fn on_request(&self, _ctx: &mut RequestCtx) -> HookOutcome {
            HookOutcome::Abort {
                status_code: 403,
                message: "blocked".into(),
            }
        }
    }

    #[test]
    fn hook_registry_invocation_order() {
        let mut reg = HookRegistry::new();
        reg.register(Box::new(LoggingHook));

        let mut ctx = RequestCtx {
            id: RequestId::new(),
            caller_identity: None,
            model_id: "test".into(),
            collection_ids: vec![],
            metadata: HashMap::new(),
            start_time: SystemTime::now(),
        };

        let outcome = reg.on_request(&mut ctx);
        assert!(matches!(outcome, HookOutcome::Continue));
        assert_eq!(ctx.metadata.get("seen_by_hook").unwrap(), "true");
    }

    #[test]
    fn hook_registry_abort_short_circuits() {
        let mut reg = HookRegistry::new();
        reg.register(Box::new(BlockHook));
        reg.register(Box::new(LoggingHook)); // never reached

        let mut ctx = RequestCtx {
            id: RequestId::new(),
            caller_identity: None,
            model_id: "test".into(),
            collection_ids: vec![],
            metadata: HashMap::new(),
            start_time: SystemTime::now(),
        };

        let outcome = reg.on_request(&mut ctx);
        assert!(matches!(outcome, HookOutcome::Abort { .. }));
        // LoggingHook never ran
        assert!(!ctx.metadata.contains_key("seen_by_hook"));
    }

    #[test]
    fn on_response_hook() {
        let mut reg = HookRegistry::new();
        reg.register(Box::new(LoggingHook));

        let ctx = RequestCtx {
            id: RequestId::new(),
            caller_identity: None,
            model_id: "test".into(),
            collection_ids: vec![],
            metadata: HashMap::new(),
            start_time: SystemTime::now(),
        };

        let mut resp = ResponseCtx {
            model_id: "test".into(),
            content: "hello".into(),
            finish_reason: "stop".into(),
            usage: UsageInfo::default(),
            metadata: HashMap::new(),
        };

        let outcome = reg.on_response(&ctx, &mut resp);
        assert!(matches!(outcome, HookOutcome::Continue));
        assert_eq!(resp.metadata.get("annotated").unwrap(), "1");
    }
}
