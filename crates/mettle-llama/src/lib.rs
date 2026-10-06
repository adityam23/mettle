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
    RequestKind, RerankError, SamplingParams, SlotLoadSpec, TokenLogprob,
};
use mettle_core::model_registry::Pooling;

use crate::scheduler::flash_policy;
use crate::vram::RerankGeometry;
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
    /// Rerank per-call packed-token cap (residency contract §1.1). Bounds
    /// the per-call rerank compute buffer; a slot may override it lower.
    pub rerank_micro_tokens: usize,
    /// Margin held back from the device's free memory when preflighting an
    /// inline (rerank/embed) context, from `[engine] vram_margin_bytes`
    /// (default [`DEFAULT_VRAM_MARGIN_BYTES`]). Mirrors the planner's
    /// default so inline and planned reservations budget alike.
    pub vram_margin_bytes: u64,
    pub speculative: SpecConfig,
}

/// Default rerank per-call packed-token cap ([engine] rerank_micro_tokens).
pub const DEFAULT_RERANK_MICRO_TOKENS: usize = 2048;

/// Default inline-context VRAM margin ([engine] vram_margin_bytes unset):
/// 512 MiB, matching the planner's default.
pub const DEFAULT_VRAM_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// Floor the planner will shrink a rerank slot's packed-token cap to
/// before dropping the slot (residency contract §1.1). One sequence of a
/// typical query/document pair still fits at this cap.
pub const MIN_RERANK_MICRO_TOKENS: usize = 256;

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
            rerank_micro_tokens: DEFAULT_RERANK_MICRO_TOKENS,
            vram_margin_bytes: DEFAULT_VRAM_MARGIN_BYTES,
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
    /// Whether ops run on the GPU for this model's inline (embed/rerank)
    /// contexts. `placement = "cpu"` slots set this false so their compute
    /// buffers never land on CUDA0 (residency contract §1.2).
    op_offload: bool,
    /// Per-call packed-token cap for this reranker (slot override, else
    /// engine default). Ignored for non-rerank kinds.
    rerank_micro_tokens: usize,
    /// Cost coefficients for this reranker's per-call context, read at load
    /// time. `None` for non-rerank kinds or a CPU-placed reranker (no GPU
    /// context to preflight).
    rerank_geometry: Option<RerankGeometry>,
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
            rerank_micro_tokens: None,
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
            rerank_micro_tokens: None,
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
            route_request(&mut runtimes, &backend, req, &cfg);
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
            let op_offload = spec.n_gpu_layers.unwrap_or(cfg.n_gpu_layers) > 0;
            // Rerank preflight coefficients (§1.1); only meaningful when the
            // context will hold GPU buffers.
            let rerank_geometry = (matches!(kind, ModelKind::Rerank) && op_offload)
                .then(|| vram::rerank_geometry(&spec.gguf_path, cfg.kv_cache_q8))
                .flatten();
            return Ok(ModelRuntime {
                sched: None,
                mtp: None,
                model,
                kind,
                rerank_template,
                // §1.2: CPU-placed models must not reserve CUDA compute
                // buffers, so op offload follows the same n_gpu_layers
                // signal the generation path uses.
                op_offload,
                rerank_micro_tokens: slot.rerank_micro_tokens.unwrap_or(cfg.rerank_micro_tokens),
                rerank_geometry,
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
            op_offload,
            rerank_micro_tokens: cfg.rerank_micro_tokens,
            rerank_geometry: None,
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
        op_offload,
        rerank_micro_tokens: cfg.rerank_micro_tokens,
        rerank_geometry: None,
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
    cfg: &LlamaConfig,
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
            match embed(backend, &rt.model, pooling, &input, rt.op_offload) {
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
                rt.rerank_micro_tokens,
                rt.rerank_geometry,
                cfg.vram_margin_bytes,
                rt.op_offload,
            ) {
                Ok(v) => {
                    let _ = response.blocking_send(EngineResponse::Scores(v));
                }
                Err(RerankError::ContextVram(e)) => {
                    let _ = response.blocking_send(EngineResponse::ContextVram(e));
                }
                Err(RerankError::Other(e)) => {
                    let _ = response.blocking_send(EngineResponse::Error(e));
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
    op_offload: bool,
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

    let mut ctx = model.new_context(
        backend,
        embed_ctx_params(pooling, tokens.len(), tokens.len(), 1, op_offload),
    )?;
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

/// Rerank micro-batch shape: at most this many sequences per context
/// (bounded further by the packed-token cap). Matches the cap the planner
/// budgets for (residency contract §1.1).
pub const RERANK_MICRO_BATCH: usize = 8;

/// Cached accelerator probe for the inline-context preflight (residency
/// contract §1.1). Probed once; `memory()` shells out per call.
static DEVICE_PROBE: std::sync::LazyLock<Option<Box<dyn mettle_core::device::Device>>> =
    std::sync::LazyLock::new(mettle_core::device::detect);

/// Current free VRAM minus `margin`, or `None` when no accelerator is
/// reachable (CPU-only host: no GPU context to preflight).
///
/// Reads the device each time it is called — the probe shells out, so
/// callers read it once per request and never cache it across requests
/// (a stale reading could admit an over-committing batch).
fn inline_vram_budget(margin: u64) -> Option<u64> {
    let dev = DEVICE_PROBE.as_ref()?;
    Some(dev.memory()?.free_bytes.saturating_sub(margin))
}

/// Estimated cost of the context one rerank micro-batch of `chunk` needs.
fn rerank_batch_cost(geo: &RerankGeometry, chunk: &[Vec<LlamaToken>]) -> vram::RerankContextCost {
    let total: usize = chunk.iter().map(|s| s.len()).sum();
    let max_len = chunk.iter().map(|s| s.len()).max().unwrap_or(1);
    let kv_cells = max_len.div_ceil(256) * 256 * chunk.len();
    geo.cost(kv_cells, total)
}

/// Shrink a greedy-filled batch (whole sequences only) until its estimated
/// per-call context fits `budget`. Returns `(n, fits)`: the number of
/// leading sequences to use and whether that batch fits. `budget = None`
/// (no reachable accelerator) keeps the batch unchanged.
fn fit_rerank_batch(
    geo: &RerankGeometry,
    chunk: &[Vec<LlamaToken>],
    budget: Option<u64>,
) -> (usize, bool) {
    let mut n = chunk.len();
    let Some(budget) = budget else {
        return (n, true);
    };
    while n > 1 && rerank_batch_cost(geo, &chunk[..n]).total() > budget {
        n = n.div_ceil(2);
    }
    (n, rerank_batch_cost(geo, &chunk[..n]).total() <= budget)
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
    micro_tokens: usize,
    geometry: Option<RerankGeometry>,
    vram_margin_bytes: u64,
    op_offload: bool,
) -> Result<Vec<f32>, mettle_core::engine::RerankError> {
    // Everything that can fail is an ordinary engine error except the §1.1
    // preflight refusal, which returns `ContextVram`.
    fn other(e: impl std::fmt::Display) -> RerankError {
        RerankError::Other(e.to_string())
    }
    let seqs = rerank_sequences(model, template, query, documents).map_err(other)?;
    if seqs.is_empty() {
        return Ok(Vec::new());
    }
    // §1.1 preflight budget: read the device ONCE per call, before any
    // micro-batch is planned. The probe shells out, and this runs inline on
    // the worker thread that also decodes every generation slot — reading
    // it per micro-batch stalled all other slots for a process round-trip.
    // It is still read per request (never cached across requests): a stale
    // reading could admit an over-committing batch.
    let vram_budget = geometry.and_then(|_| inline_vram_budget(vram_margin_bytes));
    let mut scores = Vec::with_capacity(seqs.len());
    let mut batch_seqs: &[Vec<LlamaToken>] = &seqs;
    while !batch_seqs.is_empty() {
        // Greedy fill: take whole sequences while both caps hold (at least
        // one sequence per context, even when it alone exceeds the token cap).
        let mut n = 0usize;
        let mut tokens = 0usize;
        for seq in batch_seqs {
            let len = seq.len();
            if n > 0 && (n >= RERANK_MICRO_BATCH || tokens + len > micro_tokens) {
                break;
            }
            n += 1;
            tokens += len;
        }
        // §1.1 preflight: a ggml/CUDA allocation failure aborts the whole
        // process, so never attempt a context we cannot fit. Halve the
        // batch (whole sequences) until it fits; if even one sequence does
        // not, refuse with the structured reason and keep the process up.
        if let Some(geo) = geometry {
            let (fit_n, fits) = fit_rerank_batch(&geo, &batch_seqs[..n], vram_budget);
            n = fit_n;
            if !fits {
                let cost = rerank_batch_cost(&geo, &batch_seqs[..n]);
                let budget = vram_budget.unwrap_or(0);
                return Err(RerankError::ContextVram(format!(
                    "rerank context needs ~{} MiB ({} tokens), {} MiB free after margin",
                    cost.total() / (1024 * 1024),
                    batch_seqs[..n].iter().map(|s| s.len()).sum::<usize>(),
                    budget / (1024 * 1024)
                )));
            }
        }
        let (chunk, rest) = batch_seqs.split_at(n);
        batch_seqs = rest;
        let total: usize = chunk.iter().map(|s| s.len()).sum();
        let max_len = chunk.iter().map(|s| s.len()).max().unwrap_or(1);
        let kv_cells = max_len.div_ceil(256) * 256 * chunk.len();
        let params = embed_ctx_params(Pooling::Rank, kv_cells, total, chunk.len(), op_offload);
        let t0 = std::time::Instant::now();
        let mut ctx = model.new_context(backend, params).map_err(other)?;
        let create_ms = t0.elapsed().as_millis();
        let mut batch = LlamaBatch::new(total.max(8), chunk.len() as i32);
        for (seq_i, seq) in chunk.iter().enumerate() {
            for (pos, tok) in seq.iter().enumerate() {
                batch
                    .add(*tok, pos as i32, &[seq_i as i32], false)
                    .map_err(other)?;
            }
        }
        let t1 = std::time::Instant::now();
        ctx.decode(&mut batch).map_err(other)?;
        tracing::debug!(
            rerank_batch = %format!(
                "n_seq={} tokens={total} ctx_create={create_ms}ms decode={}ms",
                chunk.len(),
                t1.elapsed().as_millis()
            ),
            "rerank micro-batch"
        );
        for seq_i in 0..chunk.len() {
            let emb = ctx
                .embeddings_seq_ith(seq_i as i32)
                .map_err(|e| other(format!("rerank scores: {e}")))?;
            if emb.is_empty() {
                return Err(other(
                    "model produced no rerank score (missing classifier head?)",
                ));
            }
            scores.push(emb[0]);
        }
    }
    Ok(scores)
}

/// Build one token sequence per (query, document) pair, truncating docs so
/// each pair fits the training context (positions beyond it are
/// out-of-distribution).
fn rerank_sequences(
    model: &LlamaModel,
    template: Option<&str>,
    query: &str,
    documents: &[String],
) -> Result<Vec<Vec<LlamaToken>>> {
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
    Ok(seqs)
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

fn embed_ctx_params(
    pooling: Pooling,
    kv_cells: usize,
    pp_tokens: usize,
    n_seqs: usize,
    op_offload: bool,
) -> LlamaContextParams {
    // `pp_tokens` sizes the prefill compute buffer (n_batch / n_ubatch must
    // cover the whole pass — encoder contexts cannot split a sequence across
    // ubatches, llama.cpp asserts `n_ubatch >= n_tokens`); `kv_cells` sizes
    // the KV cache (with kv_unified=false, kv_cells/n_seq_max cells are
    // reserved PER sequence, so multi-seq callers pass
    // kv_cells = max_len * n_seqs). Measured on the bench GPU (2026-09-10,
    // adoption contract §1.3): per-token decode cost is FLAT in sequence
    // count (~0.55 ms/token for the 0.6B reranker) under either buffer
    // layout, and SPLIT KV preserves score ordering better than unified
    // (probe mean rho .99898 / min .9945 vs unified .9941 / .90) — so the
    // micro-batch keeps kv_unified=false and batches to cut context churn,
    // not for compute amortization. Reserving either buffer for a
    // shape-skewed worst case OOMs the 6 GB card (a 4608-cell KV shape
    // wanted a 2.5 GiB compute buffer).
    // Positions beyond the training length are out-of-distribution —
    // callers truncate first.
    let n_ctx = (kv_cells.max(1)) as u32;
    let n_pp = (pp_tokens.max(1)) as u32;
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_seq_max(n_seqs.max(1) as u32)
        .with_kv_unified(false)
        .with_n_batch(n_pp)
        .with_n_ubatch(n_pp)
        .with_n_threads(
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4),
        )
        // §1.2: a CPU-placed model must not reserve a CUDA compute buffer;
        // op offload follows the same signal as the generation path (the
        // scheduler sets flash attention from it too).
        .with_flash_attention_policy(flash_policy(op_offload))
        .with_op_offload(op_offload)
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
/// Read an embedding model's embedding dimension from a GGUF file header
/// (`{arch}.embedding_length`). Used to size a turbovec collection to the
/// actual model — no hardcoded 768. `None` when the file is unreadable or
/// lacks the key. The arch prefix is resolved from `general.architecture`
/// exactly as [`vram::estimate_vram`] does.
pub fn gguf_embedding_dim(path: &std::path::Path) -> Option<u32> {
    let ctx = llama_cpp_2::gguf::GgufContext::from_file(path)?;
    let arch_idx = ctx.find_key("general.architecture");
    let arch = (arch_idx >= 0)
        .then(|| ctx.val_str(arch_idx))
        .flatten()
        .unwrap_or("llama")
        .to_owned();
    let idx = ctx.find_key(&format!("{arch}.embedding_length"));
    (idx >= 0).then(|| ctx.val_u32(idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seqs(lens: &[usize]) -> Vec<Vec<LlamaToken>> {
        lens.iter()
            .map(|n| (0..*n).map(|i| LlamaToken::new(i as i32)).collect())
            .collect()
    }

    /// §1.1: the preflight halves the batch while the estimated context does
    /// not fit, and reports failure (rather than allocating) when even one
    /// sequence cannot fit.
    #[test]
    fn rerank_preflight_shrinks_then_refuses() {
        // 1 KiB/token compute, no base, no KV: cost = 2048 tokens * 1 KiB.
        let geo = RerankGeometry::with_coefficients(0, 1024, 0);
        let batch = seqs(&[512, 512, 512, 512]);

        // Everything fits.
        let (n, fits) = fit_rerank_batch(&geo, &batch, Some(4096 * 1024));
        assert_eq!((n, fits), (4, true));

        // 2 sequences (1024 tokens = 1 MiB) fit, 4 (2 MiB) do not.
        let (n, fits) = fit_rerank_batch(&geo, &batch, Some(1500 * 1024));
        assert_eq!((n, fits), (2, true));

        // One sequence (512 tokens = 512 KiB) is the last rung that fits.
        let (n, fits) = fit_rerank_batch(&geo, &batch, Some(600 * 1024));
        assert_eq!((n, fits), (1, true));

        // Below one sequence: refuse instead of allocating.
        let (n, fits) = fit_rerank_batch(&geo, &batch, Some(100 * 1024));
        assert_eq!((n, fits), (1, false));

        // No reachable accelerator: no preflight at all.
        let (n, fits) = fit_rerank_batch(&geo, &batch, None);
        assert_eq!((n, fits), (4, true));
    }
}
