//! Continuous-batching scheduler (Phase 2) + standard draft-model
//! speculation (Phase 6).
//!
//! Each generation model owns one persistent [`LlamaContext`] and a slot
//! table (`n_seq_max` slots). Every iteration:
//!   1. admit queued requests into free slots (chunked prompt prefill;
//!      mtmd OCR evals run outside the combined batch),
//!   2. build ONE combined [`LlamaBatch`] across all active slots
//!      (prefill chunks + one continuation token per slot + draft tokens),
//!   3. decode once, then sample/verify/detokenize/emit per slot,
//!   4. free finished slots (KV `seq_rm`).
//!
//! Speculative decoding mirrors llama.cpp's `common_speculative_draft_model`
//! + `sample_and_accept_n` semantics (see `common/speculative.cpp`):
//! * the draft context decodes the anchor token and each proposal, staying
//!   in KV lockstep with the target sequence;
//! * the target decodes `[anchor, draft[0..k])` in ONE batch;
//! * acceptance runs the slot sampler per draft position (exact for greedy);
//! * partial acceptance rolls back BOTH contexts from the same position.
//!
//! # Soundness note
//! [`LlamaContext`] borrows its [`LlamaModel`]; both live on the worker
//! thread. Schedulers store `LlamaContext<'static>` obtained by shortening
//! the borrow — sound because the owning [`crate::ModelRuntime`] declares
//! `model` before `sched`, so the context is dropped before the model.

use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::time::Instant;

use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdInputChunks};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use mettle_core::engine::{EngineRequest, EngineResponse, GenConstraint, RequestKind};

use crate::prompt;
use crate::state::{self, GenState, SampleOutcome};

/// Map the config knob to llama.cpp's flash-attention policy (AUTO lets
/// llama.cpp enable it when the backend supports it).
pub(crate) fn flash_policy(enabled: bool) -> llama_cpp_sys_2::llama_flash_attn_type {
    if enabled {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO
    } else {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED
    }
}

/// Runtime knobs for one generation model's scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_seq_max: usize,
    pub n_threads: i32,
    pub default_max_tokens: usize,
    /// q8_0 KV quantisation (llama.cpp has no FP8; q8_0 is the 8-bit
    /// stand-in agreed in the migration plan).
    pub kv_cache_q8: bool,
    pub flash_attn: bool,
    /// Max draft tokens per verification round (Phase 6).
    pub gamma: usize,
    /// Allow the scheduler to move ops to higher-priority backends (GPU).
    /// False for CPU-only models: keeps their compute on CPU so no CUDA
    /// compute buffer is reserved (see load_runtime in lib.rs).
    pub op_offload: bool,
}

/// Staged OCR input: mtmd chunks + the bitmap. The bitmap field is never
/// read directly but must outlive `chunks` (chunk data points into it).
struct OcrPrefill {
    chunks: MtmdInputChunks,
    #[allow(dead_code)]
    bitmap: MtmdBitmap,
}

/// Per-sequence draft-context state (KV mirror of a target slot).
struct DraftSeq {
    n_past: i32,
    active: bool,
}

/// Private draft-model context for standard speculation.
struct DraftRuntime {
    /// Boxed for the same reason as `ModelRuntime.model`: the draft context
    /// holds a `&LlamaModel` reference into this, so it must not move.
    model: Box<LlamaModel>,
    ctx: LlamaContext<'static>,
    seqs: Vec<DraftSeq>,
    gamma: usize,
}

struct SpecState {
    /// Draft tokens proposed for the current verification round.
    draft: Vec<LlamaToken>,
    /// Batch indices of each draft token in the last combined decode.
    verify_idx: Vec<i32>,
    /// True once proposals are staged (anchor is in the batch).
    armed: bool,
}

/// One in-flight generation.
struct Slot {
    seq_id: i32,
    prompt: Vec<LlamaToken>,
    /// Decoded token history (prompt + accepted generated) — the KV prefix.
    tokens: Vec<LlamaToken>,
    /// Next free position in the context for this sequence.
    n_past: i32,
    /// Prompt tokens not yet decoded (chunked prefill).
    prefill: VecDeque<LlamaToken>,
    sampler: LlamaSampler,
    ocr: Option<OcrPrefill>,
    spec: Option<SpecState>,
    gs: GenState,
    t_prefill: Option<Instant>,
    prompt_ns: u64,
}

/// The per-model scheduler.
pub struct Scheduler {
    ctx: LlamaContext<'static>,
    mtmd: Option<MtmdContext>,
    slots: Vec<Option<Slot>>,
    queue: VecDeque<EngineRequest>,
    n_batch: usize,
    n_seq_max: usize,
    default_max_tokens: usize,
    draft: Option<DraftRuntime>,
}

impl Scheduler {
    /// Create the persistent context for a generation model. `mmproj` (if
    /// any) loads vision/OCR support; `draft_model` (if any) arms standard
    /// draft-model speculation. The caller guarantees the model outlives the
    /// returned scheduler.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: &LlamaBackend,
        model: &LlamaModel,
        mmproj: Option<&str>,
        params: &SchedulerParams,
        draft_model: Option<Box<LlamaModel>>,
    ) -> Result<Self> {
        use llama_cpp_2::context::params::KvCacheType;
        // Never over-extend a model beyond its trained context (extended rope
        // degrades to garbage): the config's n_ctx applies to the generation
        // model; smaller models (e.g. OCR's 32768) get their own ceiling.
        let n_ctx = params.n_ctx.min(model.n_ctx_train()).max(2048);
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_threads(params.n_threads)
            .with_n_batch(params.n_batch)
            .with_n_seq_max(params.n_seq_max as u32)
            // Flash attention is only safe for offloaded (GPU) contexts: on
            // CPU-only contexts (e.g. OCR) it drops embedding-token
            // contributions, so the model never sees the image.
            .with_flash_attention_policy(flash_policy(params.flash_attn && params.op_offload))
            .with_op_offload(params.op_offload);
        if params.kv_cache_q8 {
            ctx_params = ctx_params
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);
        }
        let ctx = model.new_context(backend, ctx_params.clone())?;
        // Sound: the model (owned by the caller's ModelRuntime) outlives this.
        let ctx = unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(ctx) };

        let mtmd = match mmproj {
            Some(path) => {
                let p = llama_cpp_2::mtmd::MtmdContextParams {
                    use_gpu: false,
                    print_timings: false,
                    n_threads: params.n_threads.max(1),
                    media_marker: std::ffi::CString::new(llama_cpp_2::mtmd::mtmd_default_marker())
                        .expect("marker has no NUL"),
                    image_min_tokens: -1,
                    image_max_tokens: -1,
                };
                Some(
                    llama_cpp_2::mtmd::MtmdContext::init_from_file(path, model, &p)
                        .map_err(|e| anyhow!("mtmd init: {e}"))?,
                )
            }
            None => None,
        };

        let draft = match draft_model {
            Some(model) => {
                let dctx = model.new_context(backend, ctx_params)?;
                let dctx =
                    unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(dctx) };
                Some(DraftRuntime {
                    model,
                    ctx: dctx,
                    seqs: (0..params.n_seq_max)
                        .map(|_| DraftSeq {
                            n_past: 0,
                            active: false,
                        })
                        .collect(),
                    gamma: params.gamma,
                })
            }
            None => None,
        };

        Ok(Self {
            ctx,
            mtmd,
            slots: (0..params.n_seq_max).map(|_| None).collect(),
            queue: VecDeque::new(),
            n_batch: params.n_batch as usize,
            n_seq_max: params.n_seq_max,
            default_max_tokens: params.default_max_tokens,
            draft,
        })
    }

    pub fn queue_request(&mut self, req: EngineRequest) {
        self.queue.push_back(req);
    }

    /// True while any slot is active or work is queued.
    pub fn busy(&self) -> bool {
        self.slots.iter().any(|s| s.is_some()) || !self.queue.is_empty()
    }

    /// Run one iteration. Returns the number of decodes performed.
    pub fn iterate(&mut self, model: &LlamaModel) -> Result<usize> {
        let mut decodes = 0;

        // --- Admission ----------------------------------------------------
        while self.slots.iter().flatten().count() < self.n_seq_max {
            let Some(req) = self.queue.pop_front() else {
                break;
            };
            let Some(free) = self.slots.iter().position(|s| s.is_none()) else {
                break;
            };
            if let Err(e) = self.admit(free as i32, req, model) {
                tracing::warn!("admit failed: {e:#}");
            }
        }

        // --- OCR prefill (mtmd eval decodes outside the combined batch) ---
        let mut finished: Vec<i32> = Vec::new();
        let ocr_slots: Vec<i32> = self
            .slots
            .iter()
            .flatten()
            .filter(|s| s.ocr.is_some())
            .map(|s| s.seq_id)
            .collect();
        for seq_id in ocr_slots {
            let Some(mut slot) = self.slots[seq_id as usize].take() else {
                continue;
            };
            let mtmd = self.mtmd.as_ref().expect("OCR slot without mtmd");
            let ocr = slot.ocr.take().expect("checked");
            let new_past = match ocr.chunks.eval_chunks(
                mtmd,
                &self.ctx,
                slot.n_past,
                slot.seq_id,
                self.n_batch as i32,
                true,
            ) {
                Ok(n) => n,
                Err(e) => {
                    // On eval failure the slot was taken out of `slots`; free
                    // its KV seq and report the error instead of dropping the
                    // slot (which would leak stale KV into the next request
                    // that reuses this seq).
                    let e = anyhow!("mtmd eval: {e}");
                    let _ = slot
                        .gs
                        .response
                        .blocking_send(EngineResponse::Error(e.to_string()));
                    Self::finish_slot(&mut slot, "error");
                    finished.push(seq_id);
                    continue;
                }
            };
            slot.n_past = new_past;
            // The C helper decoded every prompt token; drop the staged prefill
            // so the combined-batch section does not decode it a second time.
            slot.prefill.clear();
            // All prompt tokens are now in the KV (the C helper decoded them).
            slot.tokens = slot.prompt.clone();
            decodes += 1;
            // The eval marked logits on the last token of the final chunk
            // (logits_last=true); that token is at batch index n_tokens-1 of
            // the last decode.
            let last_chunk = ocr
                .chunks
                .get(ocr.chunks.len() - 1)
                .expect("eval chunks non-empty");
            let logits_idx =
                i32::try_from(last_chunk.n_tokens()).expect("chunk token count fits i32") - 1;
            let outcome = state::sample_one_mtmd(
                &mut slot.gs,
                &mut slot.sampler,
                &self.ctx,
                model,
                logits_idx,
            );
            slot.t_prefill.take().map(|t| {
                slot.prompt_ns = t.elapsed().as_nanos() as u64;
            });
            Self::apply_outcome(&mut slot, outcome);
            if slot.gs.finished {
                finished.push(seq_id);
            } else {
                self.slots[seq_id as usize] = Some(slot);
            }
        }

        // --- Stage speculative proposals ----------------------------------
        let spec_seqs: Vec<i32> = self
            .slots
            .iter()
            .flatten()
            .filter(|s| s.spec.is_some())
            .map(|s| s.seq_id)
            .collect();
        for seq_id in spec_seqs {
            let (has_pending, armed) = {
                let slot = self.slots[seq_id as usize].as_ref().expect("checked");
                (
                    slot.gs.pending.is_some(),
                    slot.spec.as_ref().is_some_and(|s| s.armed),
                )
            };
            if has_pending && !armed {
                self.propose_drafts(seq_id, model)?;
            }
        }

        // --- Build the combined batch -------------------------------------
        let mut batch = LlamaBatch::new(
            self.n_batch + self.n_seq_max * (self.gamma() + 1) + 16,
            self.n_seq_max as i32,
        );
        // (seq_id, batch index whose logits predict the slot's next token)
        let mut to_sample: Vec<(i32, i32)> = Vec::new();

        for slot in self.slots.iter_mut().flatten() {
            if slot.gs.finished {
                continue;
            }
            if slot.n_past as u32 >= self.ctx.n_ctx() {
                Self::finish_slot(slot, "length");
                finished.push(slot.seq_id);
                continue;
            }

            // `llama_decode` asserts the batch never exceeds n_batch, so each
            // slot contributes at most the remaining budget; slots that don't
            // fit wait for the next iteration.
            let used = batch.n_tokens() as usize;
            if used >= self.n_batch {
                continue;
            }
            let budget = self.n_batch - used;

            let spec = slot.spec.is_some();
            let mut anchor_idx: Option<i32> = None;

            if slot.prefill.is_empty() {
                // Continuation: decode the pending token.
                if let Some(tok) = slot.gs.pending.take() {
                    let idx = batch.n_tokens() as i32;
                    batch.add(tok, slot.n_past, &[slot.seq_id], true)?;
                    slot.tokens.push(tok);
                    slot.n_past += 1;
                    anchor_idx = Some(idx);
                }
            } else {
                // Prefill chunk. Request output logits ONLY on the final token
                // of the whole prompt (the chunk that empties the queue), never
                // on the last token of an intermediate chunk. Marking
                // intermediate prefill positions as outputs corrupts
                // hybrid-recurrent (Gated Delta Net) generation across chunked
                // prefill: the model degenerates to immediate EOS on long
                // prompts split over 4+ decode calls. llama-server likewise
                // sets output on the last prompt token only (server-context.cpp:
                // `batch.set_output(size-1, true)` gated on the whole prompt
                // being processed; all other prompt tokens use `need_embd()`).
                let take = slot.prefill.len().min(budget);
                let chunk: Vec<LlamaToken> = slot.prefill.drain(..take).collect();
                let completes_prefill = slot.prefill.is_empty();
                let last = chunk.len() - 1;
                for (i, tok) in chunk.iter().enumerate() {
                    let idx = batch.n_tokens() as i32;
                    let want_logits = completes_prefill && i == last;
                    batch.add(*tok, slot.n_past, &[slot.seq_id], want_logits)?;
                    slot.tokens.push(*tok);
                    slot.n_past += 1;
                    if want_logits {
                        anchor_idx = Some(idx);
                    }
                }
                if completes_prefill {
                    // Prefill complete: first sample comes from the last
                    // prompt token's logits.
                    slot.t_prefill.take().map(|t| {
                        slot.prompt_ns = t.elapsed().as_nanos() as u64;
                    });
                }
            }

            // Speculative drafts follow the anchor in the same batch.
            if spec && anchor_idx.is_some() {
                if let Some(s) = slot.spec.as_mut() {
                    if s.armed && budget >= 1 + s.draft.len() {
                        s.verify_idx.clear();
                        for tok in &s.draft {
                            let idx = batch.n_tokens() as i32;
                            batch.add(*tok, slot.n_past, &[slot.seq_id], true)?;
                            slot.tokens.push(*tok);
                            slot.n_past += 1;
                            s.verify_idx.push(idx);
                        }
                    } else if s.armed {
                        // Not enough budget for the full draft this round:
                        // verify the anchor alone; propose_drafts re-stages
                        // once a pending token exists again.
                        s.armed = false;
                    }
                }
            }

            if let Some(idx) = anchor_idx {
                to_sample.push((slot.seq_id, idx));
            }
        }

        if batch.n_tokens() == 0 {
            return Ok(decodes);
        }
        self.ctx.decode(&mut batch)?;
        decodes += 1;

        // --- Sample / verify / emit ----------------------------------------
        for (seq_id, anchor_idx) in to_sample {
            let Some(mut slot) = self.slots[seq_id as usize].take() else {
                continue;
            };
            if slot.gs.finished {
                self.slots[seq_id as usize] = Some(slot);
                continue;
            }
            let spec_round = slot
                .spec
                .as_ref()
                .is_some_and(|s| s.armed && !s.draft.is_empty());
            let result = if spec_round {
                self.step_slot_spec(&mut slot, model, anchor_idx)
            } else {
                self.step_slot(&mut slot, model, anchor_idx)
            };
            match result {
                Ok(()) => {}
                Err(e) => {
                    let _ = slot
                        .gs
                        .response
                        .blocking_send(EngineResponse::Error(e.to_string()));
                    slot.gs.finished = true;
                }
            }
            if slot.gs.finished {
                finished.push(seq_id);
            } else {
                self.slots[seq_id as usize] = Some(slot);
            }
        }

        for seq in finished {
            self.free_slot(seq);
        }
        Ok(decodes)
    }

    fn gamma(&self) -> usize {
        self.draft.as_ref().map(|d| d.gamma).unwrap_or(0)
    }

    /// Admit one request: tokenize, build the sampler (with output
    /// constraint), stage prefill, and (for speculative slots) decode the
    /// prompt on the draft context.
    fn admit(&mut self, seq_id: i32, req: EngineRequest, model: &LlamaModel) -> Result<()> {
        // A newly admitted seq owns its KV outright; clear any residue left by
        // a prior occupant that failed to free cleanly (e.g. an OCR eval that
        // errored mid-prefill). No-op when the seq is already clean.
        let _ = self.ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        if let Some(draft) = self.draft.as_mut() {
            let _ = draft
                .ctx
                .clear_kv_cache_seq(Some(seq_id as u32), None, None);
            let dseq = &mut draft.seqs[seq_id as usize];
            dseq.active = false;
            dseq.n_past = 0;
        }
        let EngineRequest { kind, response, .. } = req;
        let (prompt, opens_reasoning, params, constraint, streaming, ocr) = match kind {
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
                    None,
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
                None,
            ),
            RequestKind::Ocr {
                instruction,
                image,
                params,
                streaming,
            } => {
                let mtmd = self
                    .mtmd
                    .as_ref()
                    .ok_or_else(|| anyhow!("model has no vision projector"))?;
                let bitmap = prompt::image_bitmap(mtmd, &image)?;
                let chunks = prompt::ocr_input_chunks(mtmd, &instruction, &bitmap)?;
                let tokens =
                    model.str_to_token(&prompt::ocr_input_text(&instruction), AddBos::Always)?;
                (
                    tokens,
                    None,
                    params,
                    GenConstraint::None,
                    streaming,
                    Some(OcrPrefill { chunks, bitmap }),
                )
            }
            RequestKind::Embedding { .. } | RequestKind::Rerank { .. } => {
                return Err(anyhow!("embedding/rerank requests are not scheduled"));
            }
        };
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");

        let sampler = crate::build_sampler_with(model, &params, &constraint);
        // Speculation is exact only for greedy sampling.
        let speculate = self.draft.is_some() && params.temperature.unwrap_or(0.0) <= 0.0;

        let mut slot = Slot {
            seq_id,
            prompt,
            tokens: Vec::new(),
            n_past: 0,
            prefill: VecDeque::new(),
            sampler,
            ocr,
            spec: speculate.then(|| SpecState {
                draft: Vec::new(),
                verify_idx: Vec::new(),
                armed: false,
            }),
            gs: GenState::new(
                streaming,
                response,
                params.max_tokens.unwrap_or(self.default_max_tokens),
                params.stop.clone(),
            ),
            t_prefill: Some(Instant::now()),
            prompt_ns: 0,
        };
        // The chat template pre-filled a reasoning opener (e.g. Qwen3.5 ends
        // its generation prompt with `<think>\n`): seed the turn with it so
        // the reasoning-channel splitter sees a well-formed generation.
        if let Some(tag) = opens_reasoning {
            slot.gs.seed_reasoning_opener(tag);
        }
        slot.prefill = slot.prompt.iter().copied().collect();

        // Draft prompt prefill keeps the draft KV in lockstep.
        if speculate {
            if let Some(draft) = self.draft.as_mut() {
                let dseq = &mut draft.seqs[seq_id as usize];
                dseq.active = true;
                dseq.n_past = 0;
                let mut b = LlamaBatch::new(slot.prompt.len().max(1), 1);
                for (i, tok) in slot.prompt.iter().enumerate() {
                    b.add(*tok, i as i32, &[seq_id], false)?;
                }
                draft.ctx.decode(&mut b)?;
                dseq.n_past = slot.prompt.len() as i32;
            }
        }

        self.slots[seq_id as usize] = Some(slot);
        Ok(())
    }

    /// Propose greedy drafts from the draft model: decode the anchor, then
    /// up to `gamma` proposals, mirroring `common_speculative_draft_model`.
    fn propose_drafts(&mut self, seq_id: i32, model: &LlamaModel) -> Result<()> {
        let Some(draft) = self.draft.as_mut() else {
            return Ok(());
        };
        let Some(anchor) = self.slots[seq_id as usize]
            .as_ref()
            .and_then(|s| s.gs.pending)
        else {
            return Ok(());
        };
        let dseq = &mut draft.seqs[seq_id as usize];
        anyhow::ensure!(dseq.active, "draft seq not initialised");

        let mut greedy = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Decode the anchor at the draft position (lockstep with target).
        let mut b = LlamaBatch::new(1, 1);
        b.add(anchor, dseq.n_past, &[seq_id], true)?;
        draft.ctx.decode(&mut b)?;
        dseq.n_past += 1;

        let mut proposed: Vec<LlamaToken> = Vec::with_capacity(draft.gamma);
        for i in 0..draft.gamma {
            let tok = greedy.sample(&draft.ctx, 0);
            greedy.accept(tok);
            if model.is_eog_token(tok) {
                break;
            }
            let piece = draft
                .model
                .token_to_piece(tok, &mut decoder, false, None)
                .unwrap_or_default();
            if piece.contains("<end_of_turn>") || piece.contains("<start_of_turn>") {
                break;
            }
            proposed.push(tok);
            if i + 1 >= draft.gamma {
                break;
            }
            let mut nb = LlamaBatch::new(1, 1);
            nb.add(tok, dseq.n_past, &[seq_id], true)?;
            draft.ctx.decode(&mut nb)?;
            dseq.n_past += 1;
        }

        if let Some(slot) = self.slots[seq_id as usize].as_mut() {
            if let Some(spec) = slot.spec.as_mut() {
                spec.draft = proposed;
                spec.verify_idx.clear();
                spec.armed = true;
            }
        }
        Ok(())
    }

    /// One sampling step for a non-speculative slot.
    fn step_slot(&mut self, slot: &mut Slot, model: &LlamaModel, idx: i32) -> Result<()> {
        let outcome = state::sample_one(&mut slot.gs, &mut slot.sampler, &self.ctx, model, idx);
        Self::apply_outcome(slot, outcome);
        Ok(())
    }

    /// Verify a staged draft (`sample_and_accept_n`), emit accepted tokens,
    /// resample on divergence, roll back both contexts on partial acceptance.
    fn step_slot_spec(
        &mut self,
        slot: &mut Slot,
        model: &LlamaModel,
        anchor_idx: i32,
    ) -> Result<()> {
        let Some(spec) = slot.spec.as_mut() else {
            return self.step_slot(slot, model, anchor_idx);
        };
        let k = spec.draft.len();
        let base_pos = slot.n_past - 1 - k as i32; // position of the anchor

        let outcome = state::verify_draft(
            &mut slot.gs,
            &mut slot.sampler,
            &self.ctx,
            model,
            anchor_idx,
            &spec.verify_idx,
            &spec.draft,
        );

        if outcome.aborted {
            slot.gs.finished = true;
            return Ok(());
        }
        if let Some(reason) = outcome.stop_reason {
            // Finished mid-verification: drop the unverified tail.
            self.rollback_spec(slot, base_pos + outcome.accepted as i32 + 1);
            Self::finish_slot(slot, reason);
            return Ok(());
        }

        // Emit the pending final token (replacement or extra) and roll back
        // the unverified draft tail on partial acceptance.
        let final_tok = slot.gs.pending.expect("verification sets pending");
        let final_idx = if outcome.accepted == 0 {
            anchor_idx
        } else {
            spec.verify_idx[outcome.accepted - 1]
        };
        let final_logprob = crate::logprobs_for(&self.ctx, final_idx, final_tok, model);
        let piece = model
            .token_to_piece(final_tok, &mut slot.gs.decoder, false, None)
            .unwrap_or_default();
        if let Some(lp) = &final_logprob {
            slot.gs.logprobs.push(lp.clone());
        }
        let emit = slot.gs.emit(&piece, final_logprob.as_ref());

        if outcome.accepted < k {
            self.rollback_spec(slot, base_pos + outcome.accepted as i32 + 1);
        }
        match emit {
            state::Emit::Continue => {}
            state::Emit::Stop(reason) => Self::finish_slot(slot, reason),
            state::Emit::Aborted => slot.gs.finished = true,
        }
        Ok(())
    }

    /// Route a sampling outcome: finish on stop/abort, keep the pending
    /// token otherwise.
    fn apply_outcome(slot: &mut Slot, outcome: SampleOutcome) {
        match outcome {
            SampleOutcome::Continue => {}
            SampleOutcome::Stop(reason) => Self::finish_slot(slot, reason),
            SampleOutcome::Aborted => slot.gs.finished = true,
        }
    }

    /// Roll the target and draft KV back to `pos` (positions >= pos removed)
    /// and fix up the slot bookkeeping.
    fn rollback_spec(&mut self, slot: &mut Slot, pos: i32) {
        let _ = self
            .ctx
            .clear_kv_cache_seq(Some(slot.seq_id as u32), Some(pos as u32), None);
        if let Some(draft) = self.draft.as_mut() {
            let dseq = &mut draft.seqs[slot.seq_id as usize];
            if dseq.active && dseq.n_past > pos {
                let _ =
                    draft
                        .ctx
                        .clear_kv_cache_seq(Some(slot.seq_id as u32), Some(pos as u32), None);
                dseq.n_past = pos;
            }
        }
        slot.tokens.truncate(pos as usize);
        slot.n_past = pos;
    }

    /// Terminal bookkeeping: send Done / final chunk.
    fn finish_slot(slot: &mut Slot, reason: &'static str) {
        let prompt_len = slot.prompt.len();
        let prompt_ns = slot.prompt_ns;
        slot.gs.finish(reason, prompt_len, prompt_ns);
    }

    fn free_slot(&mut self, seq_id: i32) {
        let _ = self.ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        if let Some(draft) = self.draft.as_mut() {
            let dseq = &mut draft.seqs[seq_id as usize];
            if dseq.active {
                let _ = draft
                    .ctx
                    .clear_kv_cache_seq(Some(seq_id as u32), None, None);
                dseq.active = false;
                dseq.n_past = 0;
            }
        }
        self.slots[seq_id as usize] = None;
    }
}
