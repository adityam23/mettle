//! Shared per-sequence generation state, emission, and draft verification —
//! the single definition used by both the continuous-batching scheduler and
//! the MTP speculative lane.
//!
//! [`verify_draft`] implements llama.cpp's `sample_and_accept_n` semantics:
//! run the slot sampler at each draft position, accept on match, resample on
//! the first divergence (exact for greedy sampling), and emit accepted
//! tokens through the normal hook/stop/max path.

use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc;

use mettle_core::engine::{EngineResponse, TokenLogprob};
use mettle_core::hooks::UsageInfo;

/// Outcome of emitting one generated token.
#[derive(Debug, PartialEq)]
pub(crate) enum Emit {
    Continue,
    Stop(&'static str),
    /// Client dropped the response channel; abort quietly.
    Aborted,
}

/// Outcome of one sampling step.
#[derive(Debug, PartialEq)]
pub(crate) enum SampleOutcome {
    Continue,
    Stop(&'static str),
    Aborted,
}

/// Per-sequence generation state shared by every execution path.
pub(crate) struct GenState {
    pub streaming: bool,
    pub response: mpsc::Sender<EngineResponse>,
    pub full: String,
    pub decoder: encoding_rs::Decoder,
    pub n_decoded: usize,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub logprobs: Vec<TokenLogprob>,
    /// The sampled token waiting to be decoded next.
    pub pending: Option<LlamaToken>,
    pub finished: bool,
    pub t_eval: Option<Instant>,
}

impl GenState {
    pub fn new(
        streaming: bool,
        response: mpsc::Sender<EngineResponse>,
        max_tokens: usize,
        stop: Vec<String>,
    ) -> Self {
        Self {
            streaming,
            response,
            full: String::new(),
            decoder: encoding_rs::UTF_8.new_decoder(),
            n_decoded: 0,
            max_tokens,
            stop,
            logprobs: Vec::new(),
            pending: None,
            finished: false,
            t_eval: Some(Instant::now()),
        }
    }

    /// Emit one piece: accumulate, stream chunk, stop/max checks.
    pub fn emit(&mut self, piece: &str, logprob: Option<&TokenLogprob>) -> Emit {
        self.full.push_str(piece);
        self.n_decoded += 1;

        if self.streaming && !piece.is_empty() {
            if self
                .response
                .blocking_send(EngineResponse::Chunk {
                    content: piece.to_string(),
                    logprob: logprob.cloned(),
                    finish_reason: None,
                    usage: None,
                })
                .is_err()
            {
                return Emit::Aborted;
            }
        }

        if let Some(s) = self.stop.iter().find(|s| self.full.ends_with(s.as_str())) {
            self.full.truncate(self.full.len() - s.len());
            return Emit::Stop("stop");
        }
        if self.n_decoded >= self.max_tokens {
            return Emit::Stop("length");
        }
        Emit::Continue
    }

    /// Seed the generation with a reasoning opener that the model's chat
    /// template pre-filled (e.g. Qwen3.5's template ends its generation prompt
    /// with `<|im_start|>assistant\n<think>\n`, so the model continues INSIDE
    /// the thinking block and never emits `<think>` itself). The opener is
    /// part of the assistant-turn text: it counts no tokens and carries no
    /// logprob, and on the streaming surface the route's ReasoningFilter
    /// consumes it like any opener, so clients never see it as content.
    pub fn seed_reasoning_opener(&mut self, tag: &str) {
        let opener = format!("<{tag}>");
        self.full.push_str(&opener);
        if self.streaming {
            let _ = self.response.blocking_send(EngineResponse::Chunk {
                content: opener,
                logprob: None,
                finish_reason: None,
                usage: None,
            });
        }
    }

    /// Terminal bookkeeping: send Done / final chunk. Clears `pending`.
    pub fn finish(&mut self, reason: &str, prompt_len: usize, prompt_ns: u64) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.pending = None;
        let eval_ns = self
            .t_eval
            .take()
            .map(|t| t.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        let usage = UsageInfo {
            prompt_tokens: prompt_len as u32,
            completion_tokens: self.n_decoded as u32,
            total_tokens: (prompt_len + self.n_decoded) as u32,
            prompt_eval_duration_ns: prompt_ns,
            eval_duration_ns: eval_ns,
        };
        if self.streaming {
            let _ = self.response.blocking_send(EngineResponse::Chunk {
                content: String::new(),
                logprob: None,
                finish_reason: Some(reason.to_string()),
                usage: Some(usage),
            });
        } else {
            let _ = self.response.blocking_send(EngineResponse::Done {
                content: std::mem::take(&mut self.full),
                finish_reason: reason.to_string(),
                usage,
                logprobs: std::mem::take(&mut self.logprobs),
            });
        }
    }
}

/// Result of a draft-verification round.
#[derive(Debug)]
pub(crate) struct VerifyOutcome {
    /// Draft tokens positionally accepted (before the divergence/stop).
    pub accepted: usize,
    /// Generation ended mid-verification (EOG / stop sequence / hook stop).
    pub stop_reason: Option<&'static str>,
    pub aborted: bool,
}

/// Verify staged drafts against the target (`sample_and_accept_n` semantics).
///
/// `chain[i]` = the token at position `base + i` (`chain[0]` is the anchor);
/// sampling its batch index predicts position `base + i + 1`:
/// * `i = 0` samples `anchor_idx`, compared against `draft[0]`;
/// * `i > 0` samples `draft_idx[i-1]`, compared against `draft[i]`;
/// * `i == k` samples the extra token when every draft matched.
///
/// Accepted drafts are emitted immediately through [`GenState::emit`]; the
/// final token (replacement or extra) is left in `gs.pending` for the caller
/// to emit after rolling back the unverified tail.
pub(crate) fn verify_draft(
    gs: &mut GenState,
    sampler: &mut LlamaSampler,
    ctx: &LlamaContext<'_>,
    model: &LlamaModel,
    anchor_idx: i32,
    draft_idx: &[i32],
    draft: &[LlamaToken],
) -> VerifyOutcome {
    let k = draft.len();
    let mut accepted = 0usize;
    let mut stop_reason = None;

    for i in 0..=k {
        let idx = if i == 0 { anchor_idx } else { draft_idx[i - 1] };
        // `llama_sampler_sample` accepts internally (0.1.154's C side); an
        // explicit accept here would double-feed grammar/penalty state.
        let s = sampler.sample(ctx, idx);

        if i == k {
            // Every draft matched: the extra token from the last draft
            // position's logits.
            accepted = k;
            gs.pending = Some(s);
            break;
        }

        let d = draft[i];
        if s != d {
            // First divergence: the target's own choice is the next token.
            accepted = i;
            gs.pending = Some(s);
            break;
        }

        // Positional accept of draft[i].
        if model.is_eog_token(d) {
            accepted = i + 1;
            stop_reason = Some("stop");
            break;
        }
        let logprob = crate::logprobs_for(ctx, idx, d, model);
        let piece = model
            .token_to_piece(d, &mut gs.decoder, false, None)
            .unwrap_or_default();
        if piece.contains("<end_of_turn>") || piece.contains("<start_of_turn>") {
            accepted = i + 1;
            stop_reason = Some("stop");
            break;
        }
        if let Some(lp) = &logprob {
            gs.logprobs.push(lp.clone());
        }
        match gs.emit(&piece, logprob.as_ref()) {
            Emit::Continue => {}
            Emit::Stop(r) => {
                accepted = i + 1; // this draft token stays in KV
                stop_reason = Some(r);
                break;
            }
            Emit::Aborted => {
                return VerifyOutcome {
                    accepted,
                    stop_reason: None,
                    aborted: true,
                };
            }
        }
    }

    VerifyOutcome {
        accepted,
        stop_reason,
        aborted: false,
    }
}

/// One non-speculative sampling step: sample, detokenize, emit, checks.
/// Sets `gs.pending` on success; `finished` on abort.
pub(crate) fn sample_one(
    gs: &mut GenState,
    sampler: &mut LlamaSampler,
    ctx: &LlamaContext<'_>,
    model: &LlamaModel,
    idx: i32,
) -> SampleOutcome {
    // `llama_sampler_sample` accepts internally; the explicit accept was
    // removed because it double-fed grammar/penalty state (0.1.154).
    let token = sampler.sample(ctx, idx);

    if model.is_eog_token(token) {
        return SampleOutcome::Stop("stop");
    }
    let logprob = crate::logprobs_for(ctx, idx, token, model);
    let piece = model
        .token_to_piece(token, &mut gs.decoder, false, None)
        .unwrap_or_default();
    if piece.contains("<end_of_turn>") || piece.contains("<start_of_turn>") {
        return SampleOutcome::Stop("stop");
    }
    if let Some(lp) = &logprob {
        gs.logprobs.push(lp.clone());
    }
    match gs.emit(&piece, logprob.as_ref()) {
        Emit::Continue => {
            gs.pending = Some(token);
            SampleOutcome::Continue
        }
        Emit::Stop(r) => SampleOutcome::Stop(r),
        Emit::Aborted => {
            gs.finished = true;
            SampleOutcome::Aborted
        }
    }
}

/// Sample after C-helper (mtmd) decodes. `llama_sampler_sample` validates the
/// batch at the C level, so the sampled row is correct even though the wrapper
/// never saw the decode; logprobs are skipped because the wrapper's
/// `initialized_logits` does not track C-helper decodes.
pub(crate) fn sample_one_mtmd(
    gs: &mut GenState,
    sampler: &mut LlamaSampler,
    ctx: &LlamaContext<'_>,
    model: &LlamaModel,
    idx: i32,
) -> SampleOutcome {
    let token = sampler.sample(ctx, idx);

    if model.is_eog_token(token) {
        return SampleOutcome::Stop("stop");
    }
    let piece = model
        .token_to_piece(token, &mut gs.decoder, false, None)
        .unwrap_or_default();
    if piece.contains("<end_of_turn>") || piece.contains("<start_of_turn>") {
        return SampleOutcome::Stop("stop");
    }
    match gs.emit(&piece, None) {
        Emit::Continue => {
            gs.pending = Some(token);
            SampleOutcome::Continue
        }
        Emit::Stop(r) => SampleOutcome::Stop(r),
        Emit::Aborted => {
            gs.finished = true;
            SampleOutcome::Aborted
        }
    }
}
