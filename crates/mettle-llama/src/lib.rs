//! llama.cpp-backed [`Engine`] implementation: continuous-batching
//! scheduler (Phase 2), logprobs, mtmd vision/OCR, structured output
//! (Phase 4), and speculative decoding (Phase 6, standard draft + MTP).
//!
//! Threading model — no tokio runtime inside the engine:
//! * a **bridge thread** `blocking_recv`s [`EngineRequest`]s off the tokio
//!   channel that routes use and forwards them to the worker;
//! * a **worker thread** owns the (`!Send`) [`LlamaBackend`], all loaded
//!   models, and one scheduler (or MTP lane) per generation model. It runs
//!   decode iterations and streams results via
//!   `tokio::mpsc::Sender::blocking_send` (valid because this thread runs no
//!   runtime). Control messages (load/remove) ride the same worker channel.
//!
//! Per-model GPU offload comes from [`ModelLoadSpec::n_gpu_layers`]
//! (defaults to [`LlamaConfig::n_gpu_layers`]).

mod mtp;
pub mod prompt;
mod scheduler;
mod state;
pub mod vram;

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc;

use mettle_core::engine::{
    Engine, EngineError, EngineRequest, EngineResponse, GenConstraint, ModelKind, ModelLoadSpec,
    RequestKind, SamplingParams, SlotLoadSpec, TokenLogprob,
};
use mettle_core::model_registry::Pooling;

use mtp::{MtpLane, MtpParams};
// Re-exported for integration tests (crash_repro drives the scheduler
// directly instead of through the engine worker).
pub use scheduler::{Scheduler, SchedulerParams};

/// The JSON output grammar (llama.cpp's `json.gbnf`) for `format: json`.
const JSON_GBNF: &str = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null")

object ::=
  "{" (
            string ":" value
    ("," string ":" value)*
  )? "}"

array  ::=
  "[" (
            value
    ("," value)*
  )? "]"

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\""

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
"#;

/// Speculative-decoding configuration (Phase 6).
#[derive(Debug, Clone, Default)]
pub struct SpecConfig {
    /// Max draft tokens per verification round.
    pub gamma: usize,
    /// target model alias → MTP drafter load spec (same-model MTP lane).
    pub mtp: Vec<(String, ModelLoadSpec)>,
    /// target model alias → standard draft model load spec.
    pub draft: Vec<(String, ModelLoadSpec)>,
}

/// Runtime knobs for the engine, derived from `RootConfig` by the server.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Default layers to offload to the GPU (999 = all). CUDA-first default.
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_seq_max: usize,
    pub n_threads: i32,
    pub default_max_tokens: usize,
    /// q8_0 KV quantisation (llama.cpp has no FP8; q8_0 is the stand-in).
    pub kv_cache_q8: bool,
    pub flash_attn: bool,
    pub speculative: SpecConfig,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        Self {
            n_gpu_layers: 999,
            n_ctx: 8192,
            n_batch: 512,
            n_seq_max: 16,
            n_threads,
            default_max_tokens: 512,
            kv_cache_q8: true,
            flash_attn: true,
            speculative: SpecConfig::default(),
        }
    }
}

/// Control + work messages multiplexed onto the single worker channel.
enum WorkerMsg {
    Generate(EngineRequest),
    Load {
        slot: SlotLoadSpec,
        ack: std_mpsc::Sender<Result<(), String>>,
    },
    Remove {
        alias: String,
        ack: std_mpsc::Sender<()>,
    },
}

/// A model resident in the worker.
///
/// Field order matters: `sched`/`mtp` (contexts) are declared BEFORE `model`
/// so the model drops LAST — the documented soundness invariant for the
/// `'static` context lifetimes.
///
/// `model` is **boxed**: the scheduler/MTP contexts hold `&LlamaModel`
/// references, and the box keeps the referent at a stable heap address even
/// though `ModelRuntime` itself is moved around the worker's map. (A plain
/// field would move the value on every runtime move, dangling the contexts'
/// references — observed as a SIGSEGV in `llama_vocab::n_tokens`.)
struct ModelRuntime {
    sched: Option<Scheduler>,
    mtp: Option<MtpLane>,
    model: Box<LlamaModel>,
    kind: ModelKind,
    /// The GGUF's named "rerank" chat template, resolved at load time
    /// (`Some` only for reranker models that embed one).
    rerank_template: Option<String>,
}

/// llama.cpp-backed engine handle shared via `AppState`.
pub struct LlamaEngine {
    /// Request channel handed to routes (they `send().await`).
    req_tx: mpsc::Sender<EngineRequest>,
    /// Worker control channel (load/remove).
    worker_tx: std_mpsc::Sender<WorkerMsg>,
    /// Aliases currently resident (mirrors the worker's model map).
    loaded: Arc<Mutex<Vec<String>>>,
}

impl LlamaEngine {
    /// Build the engine and load `specs`. Blocks until models are resident
    /// (or a load fails). The `LlamaBackend` is initialised on the worker
    /// thread (it is `!Send`); only one engine may exist per process.
    pub fn new(specs: Vec<ModelLoadSpec>, cfg: LlamaConfig) -> anyhow::Result<Self> {
        let aliases: Vec<String> = specs.iter().map(|s| s.alias.clone()).collect();
        let loaded = Arc::new(Mutex::new(aliases));

        let (worker_tx, worker_rx) = std_mpsc::channel::<WorkerMsg>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<(), String>>();

        // Worker thread: owns backend + models, runs no async runtime.
        let worker_cfg = cfg.clone();
        std::thread::Builder::new()
            .name("llama-worker".into())
            .spawn(move || worker_main(specs, worker_cfg, worker_rx, ready_tx))?;

        // Surface backend-init / initial-load failures synchronously.
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => anyhow::bail!("llama engine init failed: {e}"),
            Err(_) => anyhow::bail!("llama worker died during init"),
        }

        // Bridge thread: tokio request channel -> worker channel.
        let (req_tx, mut req_rx) = mpsc::channel::<EngineRequest>(256);
        let bridge_tx = worker_tx.clone();
        std::thread::Builder::new()
            .name("llama-bridge".into())
            .spawn(move || {
                while let Some(req) = req_rx.blocking_recv() {
                    if bridge_tx.send(WorkerMsg::Generate(req)).is_err() {
                        break; // worker gone
                    }
                }
            })?;

        Ok(Self {
            req_tx,
            worker_tx,
            loaded,
        })
    }
}

impl Engine for LlamaEngine {
    fn sender(&self) -> mpsc::Sender<EngineRequest> {
        self.req_tx.clone()
    }

    fn loaded(&self) -> Vec<String> {
        self.loaded.lock().map(|l| l.clone()).unwrap_or_default()
    }

    fn load_model(&self, spec: ModelLoadSpec) -> Result<(), EngineError> {
        self.load_slot(SlotLoadSpec {
            model: spec,
            mtp_draft: None,
            draft: None,
            n_ctx: None,
        })
    }

    fn load_slot(&self, slot: SlotLoadSpec) -> Result<(), EngineError> {
        let alias = slot.model.alias.clone();
        let (ack, ack_rx) = std_mpsc::channel();
        self.worker_tx
            .send(WorkerMsg::Load { slot, ack })
            .map_err(|_| EngineError::Other("engine worker gone".into()))?;
        match ack_rx.recv() {
            Ok(Ok(())) => {
                if let Ok(mut l) = self.loaded.lock() {
                    if !l.contains(&alias) {
                        l.push(alias);
                    }
                }
                Ok(())
            }
            Ok(Err(e)) => Err(EngineError::Other(e)),
            Err(_) => Err(EngineError::Other("engine worker gone".into())),
        }
    }

    fn remove_model(&self, alias: &str) -> Result<(), EngineError> {
        let (ack, ack_rx) = std_mpsc::channel();
        self.worker_tx
            .send(WorkerMsg::Remove {
                alias: alias.to_string(),
                ack,
            })
            .map_err(|_| EngineError::Other("engine worker gone".into()))?;
        let _ = ack_rx.recv();
        if let Ok(mut l) = self.loaded.lock() {
            l.retain(|a| a != alias);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

fn worker_main(
    specs: Vec<ModelLoadSpec>,
    cfg: LlamaConfig,
    worker_rx: std_mpsc::Receiver<WorkerMsg>,
    ready: std_mpsc::Sender<Result<(), String>>,
) {
    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            let _ = ready.send(Err(format!("backend init: {e}")));
            return;
        }
    };

    let mut runtimes: HashMap<String, ModelRuntime> = HashMap::new();
    for spec in specs {
        let alias = spec.alias.clone();
        let slot = SlotLoadSpec {
            model: spec,
            mtp_draft: None,
            draft: None,
            n_ctx: None,
        };
        match load_runtime(&backend, &slot, &cfg) {
            Ok(rt) => {
                runtimes.insert(alias, rt);
            }
            Err(e) => {
                // Tolerant init: a model that cannot stand alone (e.g. an MTP
                // drafter, which needs ctx_other) must not take the whole
                // engine down. The server starts with whatever loaded.
                tracing::warn!("skipping model '{}': {e:#}", slot.model.alias);
            }
        }
    }
    let _ = ready.send(Ok(()));

    let mut pending: Vec<EngineRequest> = Vec::new();

    loop {
        // Drain control + request messages.
        while let Ok(msg) = worker_rx.try_recv() {
            match msg {
                WorkerMsg::Generate(req) => pending.push(req),
                WorkerMsg::Load { slot, ack } => {
                    let r = load_runtime(&backend, &slot, &cfg)
                        .map(|rt| {
                            runtimes.insert(slot.model.alias.clone(), rt);
                        })
                        .map_err(|e| format!("{e:#}"));
                    let _ = ack.send(r);
                }
                WorkerMsg::Remove { alias, ack } => {
                    runtimes.remove(&alias);
                    let _ = ack.send(());
                }
            }
        }

        // Route pending requests.
        for req in pending.drain(..) {
            route_request(&mut runtimes, &backend, req);
        }

        // One decode iteration per runtime.
        let mut any_work = false;
        for (alias, rt) in runtimes.iter_mut() {
            if let Some(lane) = rt.mtp.as_mut() {
                any_work |= lane.busy();
                if let Err(e) = lane.iterate(&rt.model) {
                    tracing::warn!("mtp lane {alias} iteration failed: {e:#}");
                }
            }
            if let Some(sched) = rt.sched.as_mut() {
                any_work |= sched.busy();
                if let Err(e) = sched.iterate(&rt.model) {
                    tracing::warn!("scheduler {alias} iteration failed: {e:#}");
                }
            }
        }

        if !any_work {
            match worker_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(WorkerMsg::Generate(req)) => pending.push(req),
                Ok(WorkerMsg::Load { slot, ack }) => {
                    let r = load_runtime(&backend, &slot, &cfg)
                        .map(|rt| {
                            runtimes.insert(slot.model.alias.clone(), rt);
                        })
                        .map_err(|e| format!("{e:#}"));
                    let _ = ack.send(r);
                }
                Ok(WorkerMsg::Remove { alias, ack }) => {
                    runtimes.remove(&alias);
                    let _ = ack.send(());
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    }
}

/// Load a slot and build its runtime (scheduler, MTP lane, or inline
/// embedding/rerank context).
///
/// Drafter and context overrides come from the [`SlotLoadSpec`] first
/// (profile/JIT loads); the engine-wide `cfg.speculative` wiring remains
/// the fallback for boot loads built from config.
fn load_runtime(
    backend: &LlamaBackend,
    slot: &SlotLoadSpec,
    cfg: &LlamaConfig,
) -> Result<ModelRuntime> {
    let spec = &slot.model;
    // Boxed: contexts hold references into the model, so its address must be
    // stable for the runtime's lifetime.
    let model = Box::new(load_one(backend, spec, cfg)?);

    match spec.kind {
        ModelKind::Generation => {}
        // Embedding/rerank models never enter the generation scheduler; they
        // run inline on fresh contexts per request.
        kind @ (ModelKind::Embedding(_) | ModelKind::Rerank) => {
            let rerank_template = match kind {
                ModelKind::Rerank => gguf_rerank_template(&spec.gguf_path),
                _ => None,
            };
            return Ok(ModelRuntime {
                sched: None,
                mtp: None,
                model,
                kind,
                rerank_template,
            });
        }
    }

    // CPU-only models must not get op-offload: llama.cpp 0.1.154's scheduler
    // otherwise moves ops onto the CUDA backend for ANY context (reserving a
    // ~0.5 GiB CUDA compute buffer even for a 0-layer model), which OOMs on
    // this 6 GB card once GPU models are resident.
    let op_offload = spec.n_gpu_layers.unwrap_or(cfg.n_gpu_layers) > 0;
    // CPU-only models use F16 KV: RAM is plentiful, and q8 K/V on image
    // embeddings degrades vision features enough that OCR models lean toward
    // EOS. GPU models keep q8 to save VRAM.
    // Degradation ladder may shrink ONE slot's context without touching the
    // engine-wide default ([SlotLoadSpec::n_ctx] override).
    let n_ctx = slot.n_ctx.unwrap_or(cfg.n_ctx);
    let kv_cache_q8 = cfg.kv_cache_q8 && op_offload;
    let mut sched_params = SchedulerParams {
        n_ctx,
        n_batch: cfg.n_batch,
        n_seq_max: cfg.n_seq_max,
        n_threads: cfg.n_threads,
        default_max_tokens: cfg.default_max_tokens,
        kv_cache_q8,
        flash_attn: cfg.flash_attn,
        gamma: cfg.speculative.gamma.max(1),
        op_offload,
    };
    // Hybrid SSM+attention ("recurrent") architectures corrupt generation
    // under multi-sequence KV in this llama.cpp build: with n_seq_max > 1 a
    // Qwen3.5 generator degenerates to immediate EOS on real (long) prompts
    // regardless of KV quant or batch size, while n_seq_max = 1 generates
    // correctly. Single-slot also matches how these models are consumed
    // (`/v1` is single-user). Gemma4 is unaffected (its MTP lane is
    // sequence-0-only anyway).
    if gguf_architecture(&spec.gguf_path)
        .is_some_and(|arch| HYBRID_SINGLE_SEQ_ARCHS.contains(&arch.as_str()))
    {
        if sched_params.n_seq_max > 1 {
            tracing::warn!(
                alias = %spec.alias,
                "hybrid-recurrent architecture: clamping scheduler to 1 concurrent sequence (multi-seq KV corrupts generation)"
            );
            sched_params.n_seq_max = 1;
        }
    }

    // Same-model MTP drafter takes precedence over a standard draft model.
    // Slot-level wiring (profile/JIT loads) wins over boot-time config wiring.
    let mtp_spec = slot.mtp_draft.as_ref().or_else(|| {
        cfg.speculative
            .mtp
            .iter()
            .find(|(a, _)| a == &spec.alias)
            .map(|(_, s)| s)
    });
    if let Some(mtp_spec) = mtp_spec {
        let draft_model = Box::new(load_one(backend, mtp_spec, cfg)?);
        let mtp_params = MtpParams {
            n_ctx,
            n_batch: cfg.n_batch,
            n_threads: cfg.n_threads,
            default_max_tokens: cfg.default_max_tokens,
            kv_cache_q8: cfg.kv_cache_q8,
            flash_attn: cfg.flash_attn,
            gamma: cfg.speculative.gamma.max(1),
        };
        let lane = MtpLane::new(backend, &model, draft_model, &mtp_params)?;
        return Ok(ModelRuntime {
            sched: None,
            mtp: Some(lane),
            model,
            kind: ModelKind::Generation,
            rerank_template: None,
        });
    }

    let draft_model = match slot.draft.as_ref().or_else(|| {
        cfg.speculative
            .draft
            .iter()
            .find(|(a, _)| a == &spec.alias)
            .map(|(_, s)| s)
    }) {
        Some(draft_spec) => Some(Box::new(load_one(backend, draft_spec, cfg)?)),
        None => None,
    };
    let mmproj = spec.mmproj_path.as_ref().and_then(|p| p.to_str());
    let sched = Scheduler::new(backend, &model, mmproj, &sched_params, draft_model)?;
    Ok(ModelRuntime {
        sched: Some(sched),
        mtp: None,
        model,
        kind: ModelKind::Generation,
        rerank_template: None,
    })
}

/// Resolve a GGUF's named "rerank" chat template (`tokenizer.chat_template.rerank`).
///
/// We read the metadata key directly instead of calling
/// `LlamaModel::chat_template(Some(name))`: llama-cpp-2 0.1.154 passes a
/// dangling CString for the name argument (upstream bug), so every named
/// lookup fails and the rerank prompt would silently fall back to the
/// cross-encoder layout.
fn gguf_rerank_template(path: &std::path::Path) -> Option<String> {
    let ctx = llama_cpp_2::gguf::GgufContext::from_file(path)?;
    let idx = ctx.find_key("tokenizer.chat_template.rerank");
    if idx < 0 {
        return None;
    }
    ctx.val_str(idx).map(str::to_string)
}

/// Architectures whose generation corrupts under n_seq_max > 1 in this
/// llama.cpp build (hybrid SSM+attention / recurrent models). Mirrors
/// llama.cpp's `llm_arch_is_recurrent`-style hybrid set.
const HYBRID_SINGLE_SEQ_ARCHS: &[&str] = &[
    "qwen35",
    "qwen35moe",
    "qwen3next",
    "kimi_linear",
    "nemotron_h",
    "nemotron_h_moe",
];

/// Read a GGUF's `general.architecture` metadata value.
fn gguf_architecture(path: &std::path::Path) -> Option<String> {
    let ctx = llama_cpp_2::gguf::GgufContext::from_file(path)?;
    let idx = ctx.find_key("general.architecture");
    if idx < 0 {
        return None;
    }
    ctx.val_str(idx).map(str::to_string)
}

fn load_one(backend: &LlamaBackend, spec: &ModelLoadSpec, cfg: &LlamaConfig) -> Result<LlamaModel> {
    let n_gpu_layers = spec.n_gpu_layers.unwrap_or(cfg.n_gpu_layers);
    let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model = LlamaModel::load_from_file(backend, &spec.gguf_path, &params)?;
    Ok(model)
}

/// Route a request: embeddings run inline on a fresh context; generation
/// goes to the model's MTP lane or scheduler queue.
fn route_request(
    runtimes: &mut HashMap<String, ModelRuntime>,
    backend: &LlamaBackend,
    req: EngineRequest,
) {
    let EngineRequest {
        model: alias,
        kind,
        response,
    } = req;
    let Some(rt) = runtimes.get(&alias) else {
        let _ = response.blocking_send(EngineResponse::Error(format!(
            "model '{alias}' is not loaded"
        )));
        return;
    };
    match kind {
        RequestKind::Embedding { input } => {
            let pooling = match rt.kind {
                ModelKind::Embedding(p) => p,
                _ => {
                    let _ = response.blocking_send(EngineResponse::Error(format!(
                        "model '{alias}' is not an embedding model"
                    )));
                    return;
                }
            };
            match embed(backend, &rt.model, pooling, &input) {
                Ok(v) => {
                    let _ = response.blocking_send(EngineResponse::Embeddings(v));
                }
                Err(e) => {
                    let _ = response.blocking_send(EngineResponse::Error(e.to_string()));
                }
            }
        }
        RequestKind::Rerank { query, documents } => {
            if rt.kind != ModelKind::Rerank {
                let _ = response.blocking_send(EngineResponse::Error(format!(
                    "model '{alias}' is not a reranker"
                )));
                return;
            }
            match rerank(
                backend,
                &rt.model,
                rt.rerank_template.as_deref(),
                &query,
                &documents,
            ) {
                Ok(v) => {
                    let _ = response.blocking_send(EngineResponse::Scores(v));
                }
                Err(e) => {
                    let _ = response.blocking_send(EngineResponse::Error(e.to_string()));
                }
            }
        }
        // Generation requests go to the model's MTP lane or scheduler queue.
        kind @ (RequestKind::Chat { .. }
        | RequestKind::Completion { .. }
        | RequestKind::Ocr { .. }) => {
            let runtime = runtimes.get_mut(&alias).expect("checked above");
            if let Some(lane) = runtime.mtp.as_mut() {
                lane.queue_request(EngineRequest {
                    model: alias,
                    kind,
                    response,
                });
            } else if let Some(sched) = runtime.sched.as_mut() {
                sched.queue_request(EngineRequest {
                    model: alias,
                    kind,
                    response,
                });
            } else {
                let _ = response.blocking_send(EngineResponse::Error(format!(
                    "model '{alias}' has no generation path"
                )));
            }
        }
    }
}

/// Embed a single input on a fresh embedding context.
fn embed(
    backend: &LlamaBackend,
    model: &LlamaModel,
    pooling: Pooling,
    input: &str,
) -> Result<Vec<f32>> {
    let tokens = model.str_to_token(input, AddBos::Always)?;
    anyhow::ensure!(!tokens.is_empty(), "empty embedding input");

    // The context is sized to the model's training length. A longer input
    // would trip llama.cpp's encoder assert (`n_ubatch >= n_tokens`) and
    // abort the whole process — fail cleanly instead.
    let limit = model.n_ctx_train() as usize;
    anyhow::ensure!(
        tokens.len() <= limit,
        "embedding input too long: {} tokens > model context {}",
        tokens.len(),
        limit
    );

    let mut ctx = model.new_context(backend, embed_ctx_params(pooling, tokens.len(), 1))?;
    let mut batch = LlamaBatch::new(tokens.len().max(8), 1);
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], false)?;
    }
    ctx.decode(&mut batch)?;

    let emb = ctx
        .embeddings_seq_ith(0)
        .map_err(|e| anyhow::anyhow!("embeddings: {e}"))?;
    Ok(emb.to_vec())
}

/// Score (query, document) pairs with a cross-encoder reranker.
///
/// Pair layout follows upstream llama.cpp (`format_prompt_rerank`): use the
/// GGUF's embedded "rerank" chat template when present (Qwen3-Reranker),
/// else `[BOS] query [EOS] [SEP] doc [EOS]`. Under RANK pooling the
/// classification head is part of the graph, so `embeddings_seq_ith(s)`
/// yields `n_cls_out` floats — score = first one (`P(yes)` for Qwen3's
/// softmaxed yes/no head, the raw logit for bge-reranker).
fn rerank(
    backend: &LlamaBackend,
    model: &LlamaModel,
    template: Option<&str>,
    query: &str,
    documents: &[String],
) -> Result<Vec<f32>> {
    let limit = model.n_ctx_train() as usize;
    let bos = model.token_bos();
    let eos = model.token_eos();
    let sep = model.token_sep();

    // One token sequence per document; truncate docs so each pair fits the
    // training context (positions beyond it are out-of-distribution).
    let mut seqs: Vec<Vec<LlamaToken>> = Vec::with_capacity(documents.len());
    for doc in documents {
        let seq = if let Some(template) = &template {
            // Substitute THEN tokenize the whole prompt: splitting the
            // template around {document} would tokenize head and doc as
            // separate strings and break BPE merges across the seam
            // (e.g. ": " + "Jane" never forms ": Jane"), shifting scores.
            let prompt = template
                .replace("{query}", query)
                .replace("{document}", doc);
            let seq = model.str_to_token(&prompt, AddBos::Never)?;
            anyhow::ensure!(
                seq.len() < limit,
                "rerank pair too long: {} tokens > model context {}",
                seq.len(),
                limit
            );
            seq
        } else {
            let query_tokens = model.str_to_token(query, AddBos::Never)?;
            anyhow::ensure!(!query_tokens.is_empty(), "empty rerank query");
            anyhow::ensure!(
                query_tokens.len() + 3 < limit,
                "rerank query too long: {} tokens > model context {}",
                query_tokens.len(),
                limit
            );
            let doc_tokens = model.str_to_token(doc, AddBos::Never)?;
            let budget = limit - query_tokens.len() - 3;
            let mut seq = Vec::with_capacity(query_tokens.len() + budget + 3);
            if bos.0 != -1 {
                seq.push(bos);
            }
            seq.extend_from_slice(&query_tokens);
            if eos.0 != -1 {
                seq.push(eos);
            }
            if sep.0 != -1 {
                seq.push(sep);
            }
            seq.extend_from_slice(&doc_tokens[..doc_tokens.len().min(budget)]);
            if eos.0 != -1 {
                seq.push(eos);
            }
            seq
        };
        anyhow::ensure!(!seq.is_empty(), "empty rerank input");
        seqs.push(seq);
    }
    if seqs.is_empty() {
        return Ok(Vec::new());
    }
    // Decode each pair in its own context/decode: multi-sequence batching
    // perturbs decoder-arch rerankers (Qwen3) at the sub-0.1% score level —
    // enough to swap near-zero-scored documents vs the HF reference. Pair
    // counts are small, so the throughput cost is negligible next to exact,
    // request-shape-independent scores.
    let mut scores = Vec::with_capacity(seqs.len());
    for seq in &seqs {
        let params = embed_ctx_params(Pooling::Rank, seq.len(), 1);
        let mut ctx = model.new_context(backend, params)?;
        let mut batch = LlamaBatch::new(seq.len().max(8), 1);
        for (pos, tok) in seq.iter().enumerate() {
            batch.add(*tok, pos as i32, &[0], false)?;
        }
        ctx.decode(&mut batch)?;
        let emb = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("rerank scores: {e}"))?;
        anyhow::ensure!(
            !emb.is_empty(),
            "model produced no rerank score (missing classifier head?)"
        );
        scores.push(emb[0]);
    }
    Ok(scores)
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Build the sampler chain for a request: penalties → top-k → top-p →
/// min-p → output constraint → temperature/distribution (or greedy).
pub(crate) fn build_sampler_with(
    model: &LlamaModel,
    p: &SamplingParams,
    constraint: &GenConstraint,
) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::new();

    let repeat = p.repeat_penalty.unwrap_or(1.0);
    let freq = p.frequency_penalty.unwrap_or(0.0);
    let presence = p.presence_penalty.unwrap_or(0.0);
    if repeat != 1.0 || freq != 0.0 || presence != 0.0 {
        // Last-n window matches Ollama's default (64); penalties run on the
        // top-k/top-p-filtered candidates (llama.h guidance).
        chain.push(LlamaSampler::penalties(64, repeat, freq, presence));
    }
    if let Some(k) = p.top_k {
        chain.push(LlamaSampler::top_k(k as i32));
    }
    if let Some(tp) = p.top_p {
        chain.push(LlamaSampler::top_p(tp as f32, 1));
    }
    if let Some(mp) = p.min_p {
        chain.push(LlamaSampler::min_p(mp as f32, 1));
    }

    match constraint {
        GenConstraint::Json => {
            chain.push(
                LlamaSampler::grammar(model, JSON_GBNF, "root")
                    .expect("bundled JSON grammar is valid"),
            );
        }
        GenConstraint::JsonSchema(schema) => {
            #[cfg(feature = "llguidance")]
            {
                if let Ok(s) = LlamaSampler::llguidance(model, "json_schema", &schema.to_string()) {
                    chain.push(s);
                } else {
                    tracing::warn!("llguidance rejected schema; falling back to JSON mode");
                    chain.push(
                        LlamaSampler::grammar(model, JSON_GBNF, "root")
                            .expect("bundled JSON grammar is valid"),
                    );
                }
            }
            #[cfg(not(feature = "llguidance"))]
            {
                let _ = schema;
                tracing::warn!(
                    "JSON-schema constraints need the llguidance feature; using JSON mode"
                );
                chain.push(
                    LlamaSampler::grammar(model, JSON_GBNF, "root")
                        .expect("bundled JSON grammar is valid"),
                );
            }
        }
        GenConstraint::None => {}
    }

    let temp = p.temperature.unwrap_or(0.0);
    if temp <= 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        chain.push(LlamaSampler::temp(temp as f32));
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        chain.push(LlamaSampler::dist(seed));
    }
    LlamaSampler::chain_simple(chain)
}

// ---------------------------------------------------------------------------
// Logprobs
// ---------------------------------------------------------------------------

/// Single-pass softmax over the full vocabulary for the chosen token: the
/// token's log-probability plus the top-5 alternatives (spec §7 g.3 — hooks
/// always see logprobs).
pub(crate) fn logprobs_for(
    ctx: &llama_cpp_2::context::LlamaContext<'_>,
    idx: i32,
    chosen: LlamaToken,
    model: &LlamaModel,
) -> Option<TokenLogprob> {
    use std::cmp::Ordering;

    let mut max_logit = f32::NEG_INFINITY;
    let mut chosen_logit = f32::NEG_INFINITY;
    let mut top: [(f32, LlamaToken); 5] = [(f32::NEG_INFINITY, LlamaToken(0)); 5];
    let mut n_vocab = 0usize;

    for td in ctx.candidates_ith(idx) {
        let logit = td.logit();
        n_vocab += 1;
        if td.id() == chosen {
            chosen_logit = logit;
        }
        if logit > max_logit {
            max_logit = logit;
        }
        if logit > top[4].0 {
            top[4] = (logit, td.id());
            // Re-sort descending (bubble the new entry up; arrays are tiny).
            let mut j = 4;
            while j > 0 && top[j].0 > top[j - 1].0 {
                top.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    if n_vocab == 0 {
        return None;
    }

    // Softmax denominator (shifted for stability).
    let mut sum = 0f64;
    for td in ctx.candidates_ith(idx) {
        sum += ((td.logit() - max_logit) as f64).exp();
    }

    let chosen_p = ((chosen_logit - max_logit) as f64).exp() / sum;
    let chosen_piece = model
        .token_to_piece(chosen, &mut encoding_rs::UTF_8.new_decoder(), false, None)
        .unwrap_or_default();

    let mut top_pieces = Vec::with_capacity(5);
    let mut top_sorted: Vec<(f32, LlamaToken)> = top.to_vec();
    top_sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    for (logit, tok) in top_sorted.iter().take(5) {
        if !logit.is_finite() {
            break;
        }
        let piece = model
            .token_to_piece(*tok, &mut encoding_rs::UTF_8.new_decoder(), false, None)
            .unwrap_or_default();
        let prob = ((logit - max_logit) as f64).exp() / sum;
        top_pieces.push((piece, prob.ln()));
    }

    Some(TokenLogprob {
        token: chosen_piece,
        logprob: chosen_p.ln(),
        top: top_pieces,
    })
}

// ---------------------------------------------------------------------------
// Context params
// ---------------------------------------------------------------------------

fn embed_ctx_params(pooling: Pooling, n_tokens: usize, n_seqs: usize) -> LlamaContextParams {
    // Size to the tokens actually in flight: KV/compute buffers scale with
    // n_ctx, so a reranker/embedder request must not reserve the model's full
    // training context. Positions beyond the training length are
    // out-of-distribution — callers truncate first. Encoder contexts cannot
    // split a sequence across ubatches (llama.cpp asserts `n_ubatch >=
    // n_tokens`), so n_batch/n_ubatch cover the whole pass.
    let n_ctx = (n_tokens.max(1)) as u32;
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_seq_max(n_seqs.max(1) as u32)
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_n_threads(
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4),
        )
        .with_embeddings(true)
        .with_pooling_type(match pooling {
            Pooling::Mean => LlamaPoolingType::Mean,
            Pooling::Cls => LlamaPoolingType::Cls,
            Pooling::Last => LlamaPoolingType::Last,
            // Classification head in the graph — the rerank score path.
            Pooling::Rank => LlamaPoolingType::Rank,
        })
}

// ---------------------------------------------------------------------------
// GGUF inspection (for `mettled tune` partial-offload sizing)
// ---------------------------------------------------------------------------

/// Read the transformer layer count from a GGUF file header. `None` when
/// the file is unreadable or lacks the key (e.g. embedding models).
pub fn gguf_layer_count(path: &std::path::Path) -> Option<u32> {
    let ctx = llama_cpp_2::gguf::GgufContext::from_file(path)?;
    let idx = ctx.find_key("llama.block_count");
    (idx >= 0).then(|| ctx.val_u32(idx))
}
