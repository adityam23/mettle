//! MTP speculative lane (Phase 6): exclusive single-sequence generation
//! using llama-cpp-2's `MtpSpeculative` wrapper (same-model MTP drafters
//! sharing the target's KV via `ctx_other`).
//!
//! The wrapper binds speculative state to sequence 0, so the lane serves one
//! request at a time (a queue absorbs the rest). This matches the
//! single-consumer deployment (agent-assistant) and is the fastest decode
//! path: each verification round decodes `[anchor, draft[0..k])` in ONE
//! target pass; the MTP heads produced the drafts in the drafter's own pass.
//!
//! Protocol (per wrapper contract):
//!   1. `begin(prompt)` at admission,
//!   2. decode the prompt, then `process(&batch)` — every decode must be
//!      processed so the MTP state streams,
//!   3. sample the first token normally,
//!   4. per round: `draft(n_past, id_last, prompt)` → decode
//!      `[id_last, drafts..]` → `process` → verify → `accept(n_accepted)`,
//!   5. partial acceptance rolls back the shared KV (target and draft
//!      contexts share one cache).
//!
//! # Soundness note
//! `MtpSpeculative<'static>` owns both contexts (lifetimes shortened from
//! local borrows — the model outlives them because [`MtpLane`] declares
//! `draft_model` before `spec`, so it drops after).

use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::time::Instant;

use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
use llama_cpp_2::token::LlamaToken;

use mettle_core::engine::{EngineRequest, RequestKind};

use crate::prompt;
use crate::state::{self, GenState, SampleOutcome};

/// Runtime knobs for the MTP lane.
#[derive(Debug, Clone)]
pub struct MtpParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub default_max_tokens: usize,
    pub kv_cache_q8: bool,
    pub flash_attn: bool,
    pub gamma: usize,
}

/// One in-flight MTP generation.
struct ActiveGen {
    prompt: Vec<LlamaToken>,
    /// Next free position in the shared KV for this sequence.
    n_past: i32,
    sampler: LlamaSampler,
    gs: GenState,
    t_prefill: Option<Instant>,
    prompt_ns: u64,
    prefill_done: bool,
    /// Staged draft awaiting verification (empty = normal round).
    draft: Vec<LlamaToken>,
    draft_idx: Vec<i32>,
    draft_staged: bool,
}

/// Exclusive single-sequence MTP lane for one generation model.
pub struct MtpLane {
    /// Declared before `spec`: the drafter outlives its contexts. Never read;
    /// the field exists so it drops AFTER the spec's contexts (drop-order
    /// invariant for the `'static` transmute). Boxed: the draft context holds
    /// a `&LlamaModel` reference into it, so it must not move.
    #[allow(dead_code)]
    draft_model: Box<LlamaModel>,
    spec: MtpSpeculative<'static>,
    queue: VecDeque<EngineRequest>,
    active: Option<ActiveGen>,
    n_batch: usize,
    default_max_tokens: usize,
}

impl MtpLane {
    /// Build the lane: target context over `model`, MTP draft context over
    /// `draft_model` bound via `ctx_other` (shared KV).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: &LlamaBackend,
        model: &LlamaModel,
        draft_model: Box<LlamaModel>,
        params: &MtpParams,
    ) -> Result<Self> {
        use llama_cpp_2::context::params::KvCacheType;
        // Never over-extend beyond the model's trained context (extended rope
        // degrades to garbage; see Scheduler::new for the rationale).
        let n_ctx = params.n_ctx.min(model.n_ctx_train()).max(2048);
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(params.n_threads)
            .with_n_batch(params.n_batch)
            .with_n_seq_max(1)
            .with_flash_attention_policy(crate::scheduler::flash_policy(params.flash_attn));
        if params.kv_cache_q8 {
            ctx_params = ctx_params
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);
        }

        let target = model.new_context(backend, ctx_params.clone())?;
        let target =
            unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(target) };

        let draft_ctx_params = ctx_params.with_context_type(LlamaContextType::Mtp);
        let draft_ctx =
            draft_model.new_context_with_ctx_other(backend, draft_ctx_params, &target)?;
        let draft_ctx =
            unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(draft_ctx) };

        let spec_params = MtpSpeculativeParams {
            n_max: params.gamma as i32,
            n_min: 0,
            p_min: 0.0,
        };
        let spec = MtpSpeculative::new(target, draft_ctx, spec_params)
            .map_err(|e| anyhow!("mtp init: {e}"))?;

        Ok(Self {
            draft_model,
            spec,
            queue: VecDeque::new(),
            active: None,
            n_batch: params.n_batch as usize,
            default_max_tokens: params.default_max_tokens,
        })
    }

    pub fn queue_request(&mut self, req: EngineRequest) {
        self.queue.push_back(req);
    }

    pub fn busy(&self) -> bool {
        self.active.is_some() || !self.queue.is_empty()
    }

    /// Run one iteration. Returns the number of decodes performed.
    pub fn iterate(&mut self, model: &LlamaModel) -> Result<usize> {
        if self.active.is_none() {
            if let Some(req) = self.queue.pop_front() {
                if let Err(e) = self.begin(req, model) {
                    tracing::warn!("mtp admit failed: {e:#}");
                }
            }
        }
        let Some(mut active) = self.active.take() else {
            return Ok(0);
        };
        if active.gs.finished {
            return Ok(0);
        }

        let decodes = if !active.prefill_done {
            self.prefill(&mut active, model)?
        } else {
            self.generation_round(&mut active, model)?
        };

        if active.gs.finished {
            // Terminal was sent; clear the lane state.
            let _ = self
                .spec
                .target_context_mut()
                .clear_kv_cache_seq(Some(0), None, None);
        } else {
            self.active = Some(active);
        }
        Ok(decodes)
    }

    /// One generation round: stage drafts, decode, verify.
    fn generation_round(&mut self, active: &mut ActiveGen, model: &LlamaModel) -> Result<usize> {
        let mut decodes = 0;

        if !active.draft_staged {
            let id_last = active
                .gs
                .pending
                .ok_or_else(|| anyhow!("no pending token in generation round"))?;
            // The wrapper decodes `id_last` at position `n_past` in the draft
            // context (speculative.cpp: `common_batch_add(batch, dp.id_last,
            // dp.n_past, ...)`), so n_past = the pending token's position =
            // the next free KV slot of the (shared) target context.
            let drafts = self
                .spec
                .draft(active.n_past, id_last, &active.prompt)
                .map_err(|e| anyhow!("mtp draft: {e}"))?;
            active.draft = drafts;
            active.draft_staged = true;
        }

        let pending = active
            .gs
            .pending
            .take()
            .ok_or_else(|| anyhow!("no pending token"))?;

        if active.draft.is_empty() {
            // Normal round: decode the pending token alone.
            let mut b = LlamaBatch::new(1, 1);
            b.add(pending, active.n_past, &[0], true)?;
            self.spec.target_context_mut().decode(&mut b)?;
            self.spec.process(&b)?;
            active.n_past += 1;
            decodes += 1;
            let outcome = state::sample_one(
                &mut active.gs,
                &mut active.sampler,
                self.spec.target_context(),
                model,
                0,
            );
            self.apply_outcome(active, model, outcome)?;
            return Ok(decodes);
        }

        // Spec round: [anchor @ n_past, drafts @ n_past+1..]. `base_pos` is
        // the ANCHOR's position: with accepted < k the rollback keeps the
        // anchor + accepted drafts in KV and clears from base+accepted+1
        // (the replacement is sampled from position base+accepted's logits,
        // predicting base+accepted+1).
        let k = active.draft.len();
        let base_pos = active.n_past; // position of the anchor
        let mut b = LlamaBatch::new(1 + k, 1);
        let anchor_idx = b.n_tokens() as i32;
        b.add(pending, active.n_past, &[0], true)?;
        active.draft_idx.clear();
        for (i, tok) in active.draft.iter().enumerate() {
            let idx = b.n_tokens() as i32;
            b.add(*tok, active.n_past + 1 + i as i32, &[0], true)?;
            active.draft_idx.push(idx);
        }
        self.spec.target_context_mut().decode(&mut b)?;
        self.spec.process(&b)?;
        active.n_past += 1 + k as i32;
        decodes += 1;

        let outcome = state::verify_draft(
            &mut active.gs,
            &mut active.sampler,
            self.spec.target_context(),
            model,
            anchor_idx,
            &active.draft_idx,
            &active.draft,
        );
        // The wrapper requires accept() before the next draft() call.
        self.spec
            .accept(outcome.accepted as u16)
            .map_err(|e| anyhow!("mtp accept: {e}"))?;
        // The staged draft is consumed (accepted or not); a full acceptance
        // doesn't roll back, so clear the flag here or the next round would
        // re-stage the stale draft at shifted positions.
        active.draft_staged = false;

        if outcome.aborted {
            active.gs.finished = true;
            return Ok(decodes);
        }
        if let Some(reason) = outcome.stop_reason {
            self.rollback(active, base_pos + outcome.accepted as i32 + 1);
            active
                .gs
                .finish(reason, active.prompt.len(), active.prompt_ns);
            return Ok(decodes);
        }

        // Emit the final token (replacement or extra), then roll back the
        // unverified tail on partial acceptance.
        let final_tok = active.gs.pending.expect("verification sets pending");
        let final_idx = if outcome.accepted == 0 {
            anchor_idx
        } else {
            active.draft_idx[outcome.accepted - 1]
        };
        let final_logprob =
            crate::logprobs_for(self.spec.target_context(), final_idx, final_tok, model);
        let piece = model
            .token_to_piece(final_tok, &mut active.gs.decoder, false, None)
            .unwrap_or_default();
        if let Some(lp) = &final_logprob {
            active.gs.logprobs.push(lp.clone());
        }
        let emit = active.gs.emit(&piece, final_logprob.as_ref());

        if outcome.accepted < k {
            self.rollback(active, base_pos + outcome.accepted as i32 + 1);
        }
        match emit {
            state::Emit::Continue => {}
            state::Emit::Stop(reason) => {
                active
                    .gs
                    .finish(reason, active.prompt.len(), active.prompt_ns)
            }
            state::Emit::Aborted => active.gs.finished = true,
        }
        Ok(decodes)
    }

    /// Admission: tokenize, build the sampler, `begin` the MTP state.
    fn begin(&mut self, req: EngineRequest, model: &LlamaModel) -> Result<()> {
        let EngineRequest { kind, response, .. } = req;
        let (prompt, opens_reasoning, params, constraint, streaming) = match kind {
            RequestKind::Chat {
                messages,
                params,
                constraint,
                streaming,
            } => {
                let built = prompt::build_chat_tokens(model, &messages)?;
                (
                    built.tokens,
                    built.opens_reasoning,
                    params,
                    constraint,
                    streaming,
                )
            }
            RequestKind::Completion {
                prompt,
                params,
                constraint,
                streaming,
                ..
            } => (
                model.str_to_token(&prompt, AddBos::Always)?,
                None,
                params,
                constraint,
                streaming,
            ),
            other => {
                return Err(anyhow!(
                    "MTP lane does not serve {:?}",
                    std::mem::discriminant(&other)
                ));
            }
        };
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");

        let sampler = crate::build_sampler_with(model, &params, &constraint);
        self.spec
            .begin(&prompt)
            .map_err(|e| anyhow!("mtp begin: {e}"))?;

        self.active = Some(ActiveGen {
            prompt,
            n_past: 0,
            sampler,
            gs: GenState::new(
                streaming,
                response,
                params.max_tokens.unwrap_or(self.default_max_tokens),
                params.stop.clone(),
            ),
            t_prefill: Some(Instant::now()),
            prompt_ns: 0,
            prefill_done: false,
            draft: Vec::new(),
            draft_idx: Vec::new(),
            draft_staged: false,
        });
        // Chat-template reasoning prefill: seed the turn (see scheduler.rs).
        if let Some(tag) = opens_reasoning {
            if let Some(active) = self.active.as_mut() {
                active.gs.seed_reasoning_opener(tag);
            }
        }
        Ok(())
    }

    /// Decode the prompt through the MTP process hook and sample the first
    /// token. `llama_decode` asserts batch size <= n_batch, so a prompt
    /// longer than n_batch is decoded in chunks (logits on every position:
    /// the MTP hidden-state streaming mirrors them into the draft context).
    fn prefill(&mut self, active: &mut ActiveGen, model: &LlamaModel) -> Result<usize> {
        let n = active.prompt.len();
        let mut decodes = 0;
        let mut last_idx = 0i32;
        for (c, chunk) in active.prompt.chunks(self.n_batch).enumerate() {
            let base = c * self.n_batch;
            let mut b = LlamaBatch::new(chunk.len(), 1);
            for (i, tok) in chunk.iter().enumerate() {
                b.add(*tok, (base + i) as i32, &[0], true)?;
            }
            self.spec.target_context_mut().decode(&mut b)?;
            self.spec.process(&b)?;
            // Batch index of the chunk's last token in THIS decode.
            last_idx = (chunk.len() - 1) as i32;
            decodes += 1;
        }
        active.n_past = n as i32;
        active.prefill_done = true;
        active.t_prefill.take().map(|t| {
            active.prompt_ns = t.elapsed().as_nanos() as u64;
        });

        let outcome = state::sample_one(
            &mut active.gs,
            &mut active.sampler,
            self.spec.target_context(),
            model,
            last_idx,
        );
        self.apply_outcome(active, model, outcome)?;
        Ok(decodes)
    }

    fn apply_outcome(
        &mut self,
        active: &mut ActiveGen,
        _model: &LlamaModel,
        outcome: SampleOutcome,
    ) -> Result<()> {
        match outcome {
            SampleOutcome::Continue => {}
            SampleOutcome::Stop(reason) => {
                active
                    .gs
                    .finish(reason, active.prompt.len(), active.prompt_ns)
            }
            SampleOutcome::Aborted => active.gs.finished = true,
        }
        Ok(())
    }

    /// Roll the shared KV back to `pos` (positions >= pos removed).
    fn rollback(&mut self, active: &mut ActiveGen, pos: i32) {
        let _ = self
            .spec
            .target_context_mut()
            .clear_kv_cache_seq(Some(0), Some(pos as u32), None);
        active.n_past = pos;
        active.draft_staged = false;
    }
}
