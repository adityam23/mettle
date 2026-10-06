//! Profile manager (agent-docs/profiles-contract.md): runtime-switchable,
//! VRAM-aware model collections with graceful degradation.
//!
//! Owns three questions the routes ask:
//! * **fit** — will a slot's weights + KV + compute buffers fit free VRAM?
//!   Estimated from the GGUF header ([`mettle_llama::vram::estimate_vram`]),
//!   corrected by post-load measurement persisted in `state.json`.
//! * **degradation** — full ctx → halved ctx down to `min_ctx` → drop;
//!   never silently, always as [`EngineEvent`]s.
//! * **residency** — what is loaded, under which profile, and what happens
//!   when a switch supersedes an in-flight one (generation counter).
//!
//! # Threading / locking
//! `activate()` runs as a spawned task; every blocking `load_model` call is
//! wrapped in `spawn_blocking`. Because the llama worker is a single thread
//! (`LlamaBackend` is `!Send`, init-once), a model load blocks ALL
//! generation — "activating" is a first-class visible state (`switching`,
//! `SlotState::Loading`), and in-flight requests stall rather than fail.
//!
//! # Known limitation (accepted, documented)
//! Supersede uses cooperative cancellation: each activation carries a
//! generation number checked before every slot operation, and a late
//! completing load rolls itself back. The residual race — an unload decided
//! by generation N landing after generation N+1 re-targeted the same slot —
//! is narrowed to a single blocking op wide and self-heals by
//! re-activating the profile (status shows the slot Unloaded; `/v1`
//! answers a clean structured 503). Single-user surface; not worth
//! phase-serializing activations yet.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use mettle_core::config::{
    ModelEntry, ModelRole, Placement, ProfileConfig, ProfileSlot, RootConfig,
};
use mettle_core::device::Device;
use mettle_core::engine::{ModelLoadSpec, SlotLoadSpec};
use mettle_core::model_registry::{LoaderKind, ModelRegistry, RegistryEntry};
use mettle_llama::vram::estimate_vram;

use crate::events::{EngineEvent, EventBus};
use crate::llama_bridge;
use crate::state::SharedEngine;
use crate::unavailable::{Unavailable, UnavailableReason};

/// Planner margin when `[engine] vram_margin_bytes` is unset: absorbs CUDA
/// workspace/cuda-graph overhead the header estimate does not model plus
/// other processes' usage. Measured gap on the reference card was ~850 MiB
/// including those effects; 512 MiB + measurement reconciliation covers a
/// single-user box.
const DEFAULT_VRAM_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// `state.json` schema version marker — bump when the meaning of a stored
/// byte count changes. Entries carrying a different version are discarded.
const STATE_ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Idle time before an unpinned JIT slot is unloaded again (handoff §5;
/// per-request TTL extension deliberately deferred).
const JIT_TTL_SECS: u64 = 600;
/// TTL sweep cadence.
const SWEEP_INTERVAL_SECS: u64 = 30;

/// Substrings identifying CUDA out-of-memory failures at decode time.
/// Brittle by nature; kept in one place (profiles-contract.md §6) so the
/// match list is auditable.
const OOM_PATTERNS: &[&str] = &["out of memory", "cudaMalloc"];

fn is_oom_error(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    OOM_PATTERNS.iter().any(|p| m.contains(p))
}

fn parse_role(s: &str) -> Option<ModelRole> {
    match s {
        "chat" => Some(ModelRole::Chat),
        "embed" => Some(ModelRole::Embed),
        "rerank" => Some(ModelRole::Rerank),
        "ocr" => Some(ModelRole::Ocr),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shared handles
// ---------------------------------------------------------------------------

/// Handles into [`crate::state::AppState`] internals the manager needs.
/// Passed separately (instead of `Weak<AppState>`) so there is no
/// manager ↔ state reference cycle.
#[derive(Clone)]
pub struct ProfileDeps {
    pub engine: Arc<std::sync::RwLock<Option<SharedEngine>>>,
    pub registry: Arc<std::sync::RwLock<ModelRegistry>>,
    pub loaded_models: Arc<Mutex<Vec<String>>>,
}

// ---------------------------------------------------------------------------
// Slot state + persisted measurements
// ---------------------------------------------------------------------------

/// Runtime residency of one catalog model.
#[derive(Debug, Clone)]
pub enum SlotState {
    Unloaded,
    Loading,
    Loaded(Residency),
    Failed { cause: FailCause, reason: String },
}

/// Where and how a loaded slot lives.
#[derive(Debug, Clone, Default)]
pub struct Residency {
    pub ctx: u32,
    /// Planner estimate for THIS load's context (None for CPU slots,
    /// which are not budgeted against VRAM).
    pub estimated_bytes: Option<u64>,
    /// Post-load device-memory delta, when a device was reachable.
    pub measured_bytes: Option<u64>,
    pub shrunk: bool,
    /// Effective per-call packed-token cap for a rerank slot (the rung
    /// the ladder admitted, which may be below the configured cap);
    /// `None` for every other kind (residency contract §1.1).
    pub micro_tokens: Option<usize>,
    /// Effective placement: true when the weights live in system RAM —
    /// forced by `placement = "cpu"` or reached by demoting an `auto` slot.
    pub on_cpu: bool,
}

/// Why a slot ended up [`SlotState::Failed`]: the planner refused it for
/// budget (`dropped_vram` on the wire) or the load itself could not happen
/// (`load_failed`). Kept separate from the human-readable reason so the
/// `/v1` unavailability payload never has to parse text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailCause {
    Vram,
    Load,
}

impl SlotState {
    /// Wire label for `/engine/status` (`loading/loaded/unloaded/failed`).
    pub fn label(&self) -> &'static str {
        match self {
            Self::Unloaded => "unloaded",
            Self::Loading => "loading",
            Self::Loaded { .. } => "loaded",
            Self::Failed { .. } => "failed",
        }
    }
}

/// Readiness phase rendered by `/engine/status` and `/engine/ready`
/// (residency contract §1.4, merged with the first-run surface contract).
/// `phase` is a closed set; the rest carry whatever that phase knows.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Phase {
    /// `downloading` | `loading` | `failed` | `ready`.
    pub phase: &'static str,
    /// Model the phase is about (the boot target, or the download in flight).
    pub model: Option<String>,
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
    /// Error text for `failed`.
    pub error: Option<String>,
    /// Closed reason for `failed` (`dropped_vram` / `load_failed` / …).
    pub reason: Option<String>,
    /// Human-readable cause paired with `reason`.
    pub detail: Option<String>,
}

impl Default for Phase {
    fn default() -> Self {
        Self {
            phase: "loading",
            model: None,
            bytes_done: None,
            bytes_total: None,
            error: None,
            reason: None,
            detail: None,
        }
    }
}

/// Default bounded boot-retry window (seconds) before `/engine/ready`
/// reports `failed` (residency contract §1.4). Overridable via
/// `[engine] ready_timeout_secs`.
const DEFAULT_READY_TIMEOUT_SECS: u64 = 60;

/// One persisted post-load measurement (`state.json`, XDG data dir). Stale
/// entries (different engine version or GGUF mtime) are discarded on use.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeasuredEntry {
    pub bytes: u64,
    /// Context the measurement was taken at (the KV slope lets a fresh
    /// measurement project to other contexts).
    pub ctx: u32,
    pub engine_version: String,
    pub gguf_mtime_secs: u64,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct PersistedState {
    active_profile: Option<String>,
    #[serde(default)]
    measured: HashMap<String, MeasuredEntry>,
}

// ---------------------------------------------------------------------------
// Fit model
// ---------------------------------------------------------------------------

/// Cost of a slot as `(anchor bytes at anchor ctx) + kv slope`. Both the
/// header estimate (anchor = weights + compute @ full ctx) and a fresh
/// measurement (anchor = measured total @ its ctx) project through the same
/// formula, so the ladder logic is identical either way.
#[derive(Debug, Clone, Copy)]
pub struct SlotCost {
    pub anchor_bytes: u64,
    pub anchor_ctx: u32,
    pub kv_per_token: u64,
}

impl SlotCost {
    /// Projected cost at `ctx` (saturating — never negative).
    pub fn at(&self, ctx: u32) -> u64 {
        let delta = i128::from(ctx) - i128::from(self.anchor_ctx);
        let total = i128::from(self.anchor_bytes) + delta * i128::from(self.kv_per_token);
        u64::try_from(total.max(0)).unwrap_or(u64::MAX)
    }
}

/// Context sizes the degradation ladder walks: full, then halves, clamped
/// to end exactly at `min_ctx`. Empty tail when `min_ctx` is unset or ≥
/// full (a `min_ctx` above the full context cannot rescue a full-context
/// failure — shrinking past it is meaningless).
fn ctx_rungs(full: u32, min_ctx: Option<u32>) -> Vec<u32> {
    let mut rungs = vec![full];
    if let Some(floor) = min_ctx.filter(|m| *m < full) {
        let mut cur = full;
        while cur > floor {
            let next = (cur / 2).max(floor);
            if *rungs.last().unwrap() == next {
                break;
            }
            rungs.push(next);
            cur = next;
        }
    }
    rungs
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Inner {
    active: Option<String>,
    slots: HashMap<String, SlotState>,
    dropped: Vec<String>,
    shrunk: Vec<String>,
    /// `auto` slots running on CPU because the GPU could not hold them.
    demoted: Vec<String>,
}

pub struct ProfileManager {
    deps: ProfileDeps,
    config: RootConfig,
    events: Arc<EventBus>,
    device: Option<Box<dyn Device>>,
    inner: Mutex<Inner>,
    measured: Mutex<HashMap<String, MeasuredEntry>>,
    /// JIT-slot activity timestamps for the TTL sweeper (task 5 of the
    /// handoff; recorded here from `touch()`).
    last_used: Mutex<HashMap<String, Instant>>,
    /// Activation generation: bumped by every `activate()`; a run observing
    /// a foreign generation stops touching the world.
    generation: AtomicU64,
    switching: AtomicBool,
    /// Set while a runtime-OOM eviction is in flight; concurrent OOM
    /// reports collapse into the first attempt.
    recovering: AtomicBool,
    state_path: PathBuf,
    /// Boot readiness phase (§1.4). Live slot state wins for `ready`;
    /// this carries what only boot knows (`loading` retries, `failed`,
    /// and later `downloading`).
    phase: Mutex<Phase>,
}

impl ProfileManager {
    pub fn new(
        deps: ProfileDeps,
        config: RootConfig,
        events: Arc<EventBus>,
        state_path: PathBuf,
    ) -> Self {
        let device = mettle_core::device::detect();
        Self {
            deps,
            config,
            events,
            device,
            inner: Mutex::new(Inner {
                active: None,
                slots: HashMap::new(),
                dropped: Vec::new(),
                shrunk: Vec::new(),
                demoted: Vec::new(),
            }),
            measured: Mutex::new(HashMap::new()),
            generation: AtomicU64::new(1),
            last_used: Mutex::new(HashMap::new()),
            switching: AtomicBool::new(false),
            recovering: AtomicBool::new(false),
            state_path,
            phase: Mutex::new(Phase::default()),
        }
    }

    // -- configuration accessors ------------------------------------------

    fn profile(&self, name: &str) -> Option<ProfileConfig> {
        self.config.profile.iter().find(|p| p.name == name).cloned()
    }

    /// Does `name` name a configured profile? Route-level 404 check
    /// (before a generation bump).
    pub fn profile_exists(&self, name: &str) -> bool {
        self.profile(name).is_some()
    }

    fn catalog_entry(&self, id: &str) -> Option<ModelEntry> {
        self.config.model.iter().find(|m| m.id == id).cloned()
    }

    fn full_ctx(&self) -> u32 {
        self.config.engine.n_ctx.unwrap_or(8192)
    }

    fn margin_bytes(&self) -> u64 {
        self.config
            .engine
            .vram_margin_bytes
            .unwrap_or(DEFAULT_VRAM_MARGIN_BYTES)
    }

    fn kv_q8(&self) -> bool {
        matches!(
            self.config.engine.kv_cache,
            mettle_core::config::KVCacheSetting::Fp8
        )
    }

    fn n_seq_max(&self) -> usize {
        self.config.engine.max_concurrent_seqs
    }

    fn stale(&self, gen: u64) -> bool {
        self.generation.load(Ordering::SeqCst) != gen
    }

    fn switching(&self) -> bool {
        self.switching.load(Ordering::SeqCst)
    }

    /// Effective role: explicit catalog value wins, else inferred from the
    /// loader kind (GGUF→chat, vision→ocr, embedding→embed, rerank→rerank).
    fn effective_role(&self, id: &str) -> ModelRole {
        let catalog = self.catalog_entry(id);
        if let Some(role) = catalog.as_ref().and_then(|m| m.role) {
            return role;
        }
        let reg = self.deps.registry.read().ok();
        let kind = reg
            .as_ref()
            .and_then(|r| r.resolve(id).ok())
            .map(|e| e.loader_kind.clone());
        match kind {
            Some(LoaderKind::Embedding { .. }) => ModelRole::Embed,
            Some(LoaderKind::Rerank { .. }) => ModelRole::Rerank,
            Some(LoaderKind::GgufVision { .. }) => ModelRole::Ocr,
            _ => ModelRole::Chat,
        }
    }

    // -- fit --------------------------------------------------------------

    /// Drafter wiring for a chat slot: an explicit catalog `draft` wins
    /// (standard draft); otherwise the implicit same-size gemma4 `-mtp`
    /// drafter preserves today's boot-time MTP behavior.
    fn draft_wiring(&self, reg_entry: &RegistryEntry) -> Option<ModelLoadSpec> {
        if let Some(draft_id) = self.catalog_entry(&reg_entry.alias).and_then(|m| m.draft) {
            let reg = self.deps.registry.read().ok()?;
            if let Ok(drafter) = reg.resolve(&draft_id) {
                return llama_bridge::load_spec_for_route(&drafter).ok();
            }
            return None;
        }
        let reg = self.deps.registry.read().ok()?;
        let drafter = llama_bridge::resolve_mtp_drafter(&reg, reg_entry)?;
        llama_bridge::load_spec_for_route(&drafter).ok()
    }

    /// Whether `wiring` rides as MTP (own full-size KV → scales with the
    /// target's ctx) or as a standard drafter (short contexts → constant).
    fn wiring_is_mtp(&self, reg_entry: &RegistryEntry, wiring: &ModelLoadSpec) -> bool {
        if let Some(draft_id) = self.catalog_entry(&reg_entry.alias).and_then(|m| m.draft) {
            let _ = draft_id;
            return false; // explicit catalog draft: standard drafting
        }
        let reg = match self.deps.registry.read() {
            Ok(r) => r,
            Err(_) => return false,
        };
        llama_bridge::resolve_mtp_drafter(&reg, reg_entry).is_some_and(|d| d.alias == wiring.alias)
    }

    /// Configured per-call rerank packed-token cap, or `None` for every
    /// other kind (residency contract §1.1: a rerank slot's KV is per-call,
    /// so its cost is the packed-token cap's context, not the resident ctx).
    fn rerank_cap(&self, reg_entry: &RegistryEntry) -> Option<usize> {
        (self.effective_role(&reg_entry.alias) == ModelRole::Rerank).then(|| {
            self.config
                .engine
                .rerank_micro_tokens
                .unwrap_or(mettle_llama::DEFAULT_RERANK_MICRO_TOKENS)
        })
    }

    /// Cap rungs: the configured cap, then halved down to
    /// [`mettle_llama::MIN_RERANK_MICRO_TOKENS`]. Lowering the cap shrinks
    /// the per-call context; it is preferred over dropping a rerank slot.
    fn rerank_cap_rungs(cap: usize) -> Vec<usize> {
        let mut rungs = vec![cap];
        let mut c = cap;
        while c > mettle_llama::MIN_RERANK_MICRO_TOKENS {
            c = (c / 2).max(mettle_llama::MIN_RERANK_MICRO_TOKENS);
            rungs.push(c);
        }
        rungs.dedup();
        rungs
    }

    /// Header-derived fit model for a slot (weights + mmproj + drafters +
    /// KV slope + compute heuristic + a rerank slot's per-call context).
    /// `None` when the GGUF is unreadable or lacks geometry keys — the slot
    /// is then unloadable (GPU-or-refuse cannot verify fit).
    fn estimate_cost(
        &self,
        reg_entry: &RegistryEntry,
        rerank_cap: Option<usize>,
    ) -> Option<SlotCost> {
        let gguf = llama_bridge::resolve_gguf_path(reg_entry).ok()?;
        let est = estimate_vram(&gguf, self.full_ctx(), self.kv_q8(), self.n_seq_max())?;

        let mut anchor = est.weights_bytes + est.compute_bytes;
        let mut slope = est.kv_bytes_per_token;
        if let Some(mmproj) = llama_bridge::resolve_mmproj_path(reg_entry) {
            anchor += std::fs::metadata(mmproj).map(|m| m.len()).unwrap_or(0);
        }
        if let Some(wiring) = self.draft_wiring(reg_entry) {
            if let Some(d_est) = estimate_vram(
                &wiring.gguf_path,
                self.full_ctx(),
                self.kv_q8(),
                self.n_seq_max(),
            ) {
                anchor += d_est.weights_bytes + d_est.compute_bytes;
                if self.wiring_is_mtp(reg_entry, &wiring) {
                    slope += d_est.kv_bytes_per_token;
                } else {
                    // standard drafter: short-context KV, treat as constant
                    anchor += d_est.kv_bytes_per_token * 1024;
                }
            }
        }

        // §1.1: a rerank slot's KV lives in the per-call context, not in a
        // resident cache — budget that context at the packed-token cap and
        // drop the ctx-projected slope (its resident memory is weights +
        // compute only).
        if let Some(cap) = rerank_cap {
            let per_call = mettle_llama::vram::rerank_cap_cost(
                &gguf,
                cap,
                mettle_llama::RERANK_MICRO_BATCH,
                self.kv_q8(),
            )?;
            anchor += per_call.total();
            slope = 0;
        }

        Some(SlotCost {
            anchor_bytes: anchor + slope * u64::from(self.full_ctx()),
            anchor_ctx: self.full_ctx(),
            kv_per_token: slope,
        })
    }

    /// Fresh persisted measurement for this exact binary + file, projected
    /// through the header KV slope. `None` = stale/absent → use the header
    /// estimate.
    fn measured_cost(&self, id: &str, reg_entry: &RegistryEntry) -> Option<SlotCost> {
        let gguf = llama_bridge::resolve_gguf_path(reg_entry).ok()?;
        let mtime = gguf_mtime_secs(&gguf)?;
        let measured = self.measured.lock().ok()?;
        let m = measured.get(id)?;
        if m.engine_version != STATE_ENGINE_VERSION || m.gguf_mtime_secs != mtime {
            return None;
        }
        // Slope comes from the header when it is readable (measurements
        // don't change it); a measurement without a header projects flat.
        let kv_per_token = estimate_vram(&gguf, self.full_ctx(), self.kv_q8(), self.n_seq_max())
            .map(|e| e.kv_bytes_per_token)
            .unwrap_or(0);
        Some(SlotCost {
            anchor_bytes: m.bytes,
            anchor_ctx: m.ctx,
            kv_per_token,
        })
    }

    fn slot_cost(&self, reg_entry: &RegistryEntry, rerank_cap: Option<usize>) -> Option<SlotCost> {
        match self.measured_cost(&reg_entry.alias, reg_entry) {
            // A measurement sees the resident delta at load; a rerank slot's
            // per-call context is created per call, so the cap cost must be
            // added here. The cached measurement carries no ctx-projected
            // slope for a rerank slot (`kv_per_token = 0`).
            Some(mut cost) => {
                if let Some(cap) = rerank_cap {
                    let gguf = llama_bridge::resolve_gguf_path(reg_entry).ok()?;
                    let per_call = mettle_llama::vram::rerank_cap_cost(
                        &gguf,
                        cap,
                        mettle_llama::RERANK_MICRO_BATCH,
                        self.kv_q8(),
                    )?;
                    cost.anchor_bytes = cost.anchor_bytes.saturating_add(per_call.total());
                    cost.kv_per_token = 0;
                }
                Some(cost)
            }
            // Header estimate: cap-aware, so the rerank slot is charged
            // weights + compute + ONE capped context and never the resident
            // full-ctx KV it does not hold (§1.1). Passing the cap here (and
            // not adding it again) is what keeps the planner honest: the
            // full-ctx slope on top of the cap double-charged the slot by
            // `kv_per_token * full_ctx` — ~900 MiB on the qwen3 reranker.
            None => self.estimate_cost(reg_entry, rerank_cap),
        }
    }

    // -- engine plumbing ----------------------------------------------------

    fn engine_handle(&self) -> Result<SharedEngine, String> {
        self.deps
            .engine
            .read()
            .expect("engine lock poisoned")
            .clone()
            .ok_or_else(|| "engine not initialised".to_string())
    }

    async fn load_via_engine(&self, spec: SlotLoadSpec) -> Result<(), String> {
        let engine = self.engine_handle()?;
        tokio::task::spawn_blocking(move || engine.load_slot(spec).map_err(|e| e.to_string()))
            .await
            .map_err(|e| format!("load task panicked: {e}"))?
    }

    async fn unload_via_engine(&self, alias: &str) -> Result<(), String> {
        let engine = self.engine_handle()?;
        let alias = alias.to_string();
        tokio::task::spawn_blocking(move || engine.remove_model(&alias).map_err(|e| e.to_string()))
            .await
            .map_err(|e| format!("unload task panicked: {e}"))?
    }

    fn remember_loaded(&self, alias: &str) {
        if let Ok(mut l) = self.deps.loaded_models.lock() {
            if !l.iter().any(|a| a == alias) {
                l.push(alias.to_string());
            }
        }
    }

    fn remember_unloaded(&self, alias: &str) {
        if let Ok(mut l) = self.deps.loaded_models.lock() {
            l.retain(|a| a != alias);
        }
    }

    fn set_slot(&self, id: &str, state: SlotState) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.slots.insert(id.to_string(), state);
        }
    }

    /// Build the full load instruction for one slot: placement-forced
    /// offload, drafter wiring, ctx override.
    fn slot_spec(
        &self,
        reg_entry: &RegistryEntry,
        placement: Placement,
        n_ctx: Option<u32>,
        rerank_micro_tokens: Option<usize>,
    ) -> Result<SlotLoadSpec, String> {
        let mut model = llama_bridge::load_spec_for_route(reg_entry)?;
        // Placement is the only offload knob (profiles-contract.md).
        // `Auto` here means "the GPU attempt"; its CPU fallback calls this
        // with `Placement::Cpu`.
        model.n_gpu_layers = Some(match placement {
            Placement::Gpu | Placement::Auto => 999,
            Placement::Cpu => 0,
        });
        let wiring = self.draft_wiring(reg_entry);
        let (mtp_draft, draft) =
            if wiring.is_some() && self.wiring_is_mtp(reg_entry, wiring.as_ref().unwrap()) {
                (wiring, None)
            } else {
                (None, wiring)
            };
        Ok(SlotLoadSpec {
            model,
            mtp_draft,
            draft,
            n_ctx,
            rerank_micro_tokens,
        })
    }

    // -- activation ---------------------------------------------------------

    /// Boot-time re-activation: load `state.json`; if it names a profile
    /// that still exists, activate it (degradation applies — VRAM may have
    /// changed across restarts). Otherwise — amendment, adoption-contract
    /// grilling Q11 (2026-09-10): activate the config's `default = true`
    /// profile (the shipped `bench-retrieval` shape) instead of idling, so
    /// "shipped default" means what it says. Only when NOTHING was ever
    /// persisted (fresh install); an explicit profile switch always wins.
    pub async fn boot_activate(self: Arc<Self>) {
        // Pick the boot target: the persisted profile, else the shipped
        // `default = true` one (adoption-contract grilling Q11).
        let target = match self.load_persisted_active() {
            Some(name) => Some(name),
            None => self
                .config
                .profile
                .iter()
                .find(|p| p.default)
                .map(|p| p.name.clone()),
        };
        let Some(name) = target else {
            tracing::info!("no active profile persisted; engine idles until one is activated");
            self.set_phase(Phase {
                phase: "failed",
                reason: Some(UnavailableReason::NotResident.as_str().to_string()),
                detail: Some("no profile configured to activate".into()),
                error: Some("no profile configured to activate".into()),
                ..Default::default()
            });
            return;
        };

        // First run: mettle owns its weights. Download anything the target
        // profile pins that is not on disk yet, reporting progress through
        // the readiness phase, before the planner tries to size it. A
        // failure is recorded, NOT fatal: this process must still load the
        // weights that ARE on disk (a partial profile can cover every role
        // it declares), and the acquisition is retried inside the window
        // below so a transient failure needs no restart.
        let mut acquire_err = self.ensure_profile_weights(&name).await.err();
        if let Some((model, error)) = &acquire_err {
            tracing::error!(
                "boot: weight acquisition for profile '{name}' failed for '{}': {error}",
                model.as_deref().unwrap_or("unknown")
            );
        }

        // Bounded retry with backoff (§1.4): another process holding VRAM
        // at boot is transient, so keep the phase `loading` for the window
        // and only report `failed` once it runs out.
        let window = self.ready_timeout();
        let started = Instant::now();
        let mut delay = Duration::from_secs(1);
        let mut attempt = 0u32;
        // Generation this boot loop left behind. An operator switch
        // (`PUT /engine/profiles/active`) during the window bumps the
        // generation and owns the engine from then on — the boot loop must
        // stop rather than supersede it on the next retry (and it must not
        // touch the phase, which the newer run settles).
        let mut last_gen: Option<u64> = None;
        loop {
            let seen_gen = self.generation.load(Ordering::SeqCst);
            if last_gen.is_some_and(|prev| seen_gen != prev) {
                tracing::info!(
                    "boot: another profile was activated during the retry window; \
                     stopping retries for '{name}'"
                );
                return;
            }
            attempt += 1;
            self.set_phase(Phase {
                phase: "loading",
                model: Some(name.clone()),
                detail: (attempt > 1).then(|| format!("retry {attempt}")),
                ..Default::default()
            });
            tracing::info!("boot: activating profile '{name}' (attempt {attempt})");
            if let Err(e) = Arc::clone(&self).activate_inner(name.clone(), true).await {
                tracing::warn!("boot activation attempt {attempt} failed: {e}");
            }
            // Our own run must be the newest one. Anything else means a
            // concurrent activation took the engine mid-attempt.
            let now_gen = self.generation.load(Ordering::SeqCst);
            if now_gen != seen_gen + 1 {
                tracing::info!(
                    "boot: activation superseded during attempt {attempt} \
                     (generation {seen_gen} -> {now_gen}); stopping retries for '{name}'"
                );
                return;
            }
            last_gen = Some(now_gen);
            if self.is_serving() {
                self.set_phase(Phase {
                    phase: "ready",
                    ..Default::default()
                });
                tracing::info!("boot: profile '{name}' ready after {attempt} attempt(s)");
                return;
            }
            if started.elapsed() + delay >= window {
                break;
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(15));
            // Between attempts: acquisition may succeed now (a transient
            // network failure, or the operator restored the cache), and the
            // next activation then sizes the real weights.
            match self.ensure_profile_weights(&name).await {
                Ok(()) => acquire_err = None,
                Err(e) => acquire_err = Some(e),
            }
        }
        // A failed acquisition is the honest reason a profile that never
        // came up is not serving — report its model and error instead of
        // the generic slot-derived reason.
        if let Some((model, error)) = acquire_err {
            tracing::warn!("boot: profile '{name}' not ready; weight acquisition failed: {error}");
            self.set_phase(Phase {
                phase: "failed",
                model,
                error: Some(error.clone()),
                reason: Some(UnavailableReason::NotResident.as_str().to_string()),
                detail: Some(error),
                ..Default::default()
            });
            return;
        }
        let (reason, detail) = self.failure_reason();
        tracing::warn!(
            "boot: profile '{name}' not ready within {}s ({}: {})",
            window.as_secs(),
            reason.as_deref().unwrap_or("unknown"),
            detail.as_deref().unwrap_or("no detail")
        );
        self.set_phase(Phase {
            phase: "failed",
            model: Some(name),
            error: detail.clone(),
            reason,
            detail,
            ..Default::default()
        });
    }

    /// Acquire any weights the named profile pins that are not on disk yet,
    /// one model at a time, reporting `downloading` progress (§1.1–§1.3 of
    /// the first-run contract). Local-path entries are never downloaded: a
    /// missing one stays the config error the activation path already
    /// reports (§M6 decision).
    ///
    /// Every pinned model is attempted even when one fails — an unreachable
    /// repo must not skip the rest — and the first failure is returned as
    /// `(model, error)` so the caller can name it in the readiness phase.
    /// Callers treat a failure as non-fatal: whatever is on disk still
    /// loads, and a partial profile can still cover every role it declares.
    async fn ensure_profile_weights(
        self: &Arc<Self>,
        name: &str,
    ) -> Result<(), (Option<String>, String)> {
        let Some(profile) = self.profile(name) else {
            return Ok(());
        };
        let offline = self.config.server.offline;
        let pinned: Vec<RegistryEntry> = {
            let reg = match self.deps.registry.read() {
                Ok(r) => r,
                Err(_) => return Err((None, "registry lock poisoned".into())),
            };
            profile
                .slots
                .iter()
                .filter_map(|slot| reg.resolve(&slot.id).ok().cloned())
                .collect()
        };
        let mut first_err: Option<(Option<String>, String)> = None;
        for entry in pinned {
            if crate::weights::model_files_present(&entry) {
                continue;
            }
            // A resolvable GGUF with only a projector missing is not a config
            // error: it is repairable when the entry is downloadable, and a
            // local-path entry must not be reported as "path does not exist".
            if crate::llama_bridge::resolve_gguf_path(&entry).is_ok()
                && !crate::weights::is_downloadable(&entry)
            {
                continue;
            }
            // Not downloadable: never a boot failure (other slots may cover
            // the role), but never silent either - a slot pinned by the
            // active profile that cannot be acquired is a config error and
            // shows up as an unloaded slot in `/engine/status`.
            if let Some(p) = &entry.local_path {
                tracing::warn!(
                    "profile '{name}': slot '{}' is not on disk and its configured local path '{}' does not exist",
                    entry.alias,
                    p.display()
                );
                continue;
            }
            if !crate::weights::is_downloadable(&entry) {
                tracing::warn!(
                    "profile '{name}': slot '{}' has no downloadable GGUF (repo '{}', no quant file); \
                     skipping - configure a local path to serve it",
                    entry.alias,
                    entry.hf_repo_id
                );
                continue;
            }
            let model = entry.alias.clone();
            self.set_phase(Phase {
                phase: "downloading",
                model: Some(model.clone()),
                ..Default::default()
            });
            let manager = Arc::clone(self);
            let phase_model = model.clone();
            let sink: crate::weights::ProgressSink =
                Arc::new(move |p: crate::weights::WeightProgress| {
                    manager.set_phase(Phase {
                        phase: "downloading",
                        model: Some(phase_model.clone()),
                        bytes_done: Some(p.bytes_done),
                        bytes_total: Some(p.bytes_total),
                        ..Default::default()
                    });
                });
            if let Err(e) = crate::weights::ensure_weights(&entry, offline, Some(sink)).await {
                let msg = format!("{e}");
                // The consumer wants to know WHICH model failed, not which
                // profile was booting.
                self.set_phase(Phase {
                    phase: "failed",
                    model: Some(model.clone()),
                    error: Some(msg.clone()),
                    reason: Some(UnavailableReason::NotResident.as_str().to_string()),
                    detail: Some(msg.clone()),
                    ..Default::default()
                });
                tracing::warn!("profile '{name}': acquiring '{model}' failed: {msg}");
                if first_err.is_none() {
                    first_err = Some((Some(model.clone()), msg));
                }
                continue;
            }
            self.set_phase(Phase {
                phase: "loading",
                model: Some(model),
                ..Default::default()
            });
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(
            self.config
                .engine
                .ready_timeout_secs
                .unwrap_or(DEFAULT_READY_TIMEOUT_SECS),
        )
    }

    fn load_persisted_active(&self) -> Option<String> {
        let bytes = std::fs::read(&self.state_path).ok()?;
        let state: PersistedState = serde_json::from_slice(&bytes).ok()?;
        // Refresh the measured cache with non-stale entries.
        if let Ok(mut measured) = self.measured.lock() {
            for (id, entry) in state.measured {
                if entry.engine_version == STATE_ENGINE_VERSION {
                    measured.insert(id, entry);
                }
            }
        }
        let active = state.active_profile?;
        self.profile(&active).is_some().then_some(active)
    }

    fn persist_state(&self) {
        let payload = {
            let inner = match self.inner.lock() {
                Ok(i) => i,
                Err(_) => return,
            };
            let measured = match self.measured.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            PersistedState {
                active_profile: inner.active.clone(),
                measured: measured.clone(),
            }
        };
        let tmp = self.state_path.with_extension("json.tmp");
        let write = || -> std::io::Result<()> {
            if let Some(parent) = self.state_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&tmp, serde_json::to_vec(&payload).unwrap_or_default())?;
            std::fs::rename(&tmp, &self.state_path)
        };
        if let Err(e) = write() {
            tracing::warn!("failed to persist engine state: {e}");
        }
    }

    /// Activate `name`: diff-and-unload, then load target slots by
    /// priority through the degradation ladder. Completion is announced by
    /// events (`profile.active` / `profile.degraded`); the returned Result
    /// is for the caller's logs.
    pub async fn activate(self: Arc<Self>, name: String) -> Result<(), String> {
        // Weights are ensured on every activation, not only at boot: a
        // download that failed (offline, transient network) must be
        // recoverable by re-activating instead of restarting the daemon.
        // A failure is reported through the phase and does not abort the
        // activation — whatever is on disk may still cover every role.
        if let Err((model, error)) = self.ensure_profile_weights(&name).await {
            tracing::warn!(
                "activation of '{name}': weight acquisition failed for '{}': {error}",
                model.as_deref().unwrap_or("unknown")
            );
        }
        self.activate_inner(name, false).await
    }

    /// `boot` suppresses the terminal `failed` transition so the boot retry
    /// loop can keep the phase `loading` for its whole window (§1.4).
    async fn activate_inner(self: Arc<Self>, name: String, boot: bool) -> Result<(), String> {
        let my_gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.switching.store(true, Ordering::SeqCst);
        self.set_phase(Phase {
            phase: "loading",
            model: Some(name.clone()),
            ..Default::default()
        });
        self.events
            .emit(EngineEvent::ProfileActivating { name: name.clone() });

        let result = self.run_activation(&name, my_gen).await;

        // Only the newest run owns the switching flag and the phase.
        if self.generation.load(Ordering::SeqCst) == my_gen {
            self.switching.store(false, Ordering::SeqCst);
            if !boot {
                self.settle_phase();
            }
        }
        result
    }

    /// Terminal phase after an activation completes outside the boot
    /// window: `ready` when it can serve, else `failed` with the real
    /// reason.
    fn settle_phase(&self) {
        if self.is_serving() {
            self.set_phase(Phase {
                phase: "ready",
                ..Default::default()
            });
            return;
        }
        let (reason, detail) = self.failure_reason();
        self.set_phase(Phase {
            phase: "failed",
            error: detail.clone(),
            reason,
            detail,
            ..Default::default()
        });
    }

    fn set_phase(&self, phase: Phase) {
        if let Ok(mut p) = self.phase.lock() {
            *p = phase;
        }
    }

    /// True when the active profile is settled and still serves every role
    /// it declares (residency contract §1.4): no activation in flight, and
    /// for each declared role at least one slot is `Loaded`. A dropped
    /// slot is fine as long as another slot covers its role; a role with
    /// no surviving slot means the profile cannot answer — not ready.
    fn is_serving(&self) -> bool {
        let Ok(inner) = self.inner.lock() else {
            return false;
        };
        let Some(active) = inner.active.as_ref() else {
            return false;
        };
        let Some(profile) = self.profile(active) else {
            return false;
        };
        let mut roles: Vec<ModelRole> = Vec::new();
        let mut loaded_roles: Vec<ModelRole> = Vec::new();
        for slot in &profile.slots {
            let role = self.effective_role(&slot.id);
            if !roles.contains(&role) {
                roles.push(role);
            }
            match inner.slots.get(&slot.id) {
                Some(SlotState::Loading) => return false,
                Some(SlotState::Loaded(_)) => {
                    if !loaded_roles.contains(&role) {
                        loaded_roles.push(role);
                    }
                }
                _ => {}
            }
        }
        !roles.is_empty() && roles.iter().all(|r| loaded_roles.contains(r))
    }

    /// Closed reason + detail for a profile that cannot serve, derived
    /// from its real slot state (shared by `failed` phase and §1.5).
    fn failure_reason(&self) -> (Option<String>, Option<String>) {
        let Ok(inner) = self.inner.lock() else {
            return (None, None);
        };
        // A failed slot is the most specific cause.
        if let Some((id, SlotState::Failed { cause, reason })) = inner
            .slots
            .iter()
            .find(|(_, s)| matches!(s, SlotState::Failed { .. }))
        {
            let label = match cause {
                FailCause::Vram => UnavailableReason::DroppedVram,
                FailCause::Load => UnavailableReason::LoadFailed,
            };
            return (
                Some(label.as_str().to_string()),
                Some(format!("'{id}': {reason}")),
            );
        }
        if !inner.dropped.is_empty() {
            return (
                Some(UnavailableReason::DroppedVram.as_str().to_string()),
                Some(format!("dropped for VRAM: {}", inner.dropped.join(", "))),
            );
        }
        if inner.active.is_none() {
            return (
                Some(UnavailableReason::NotResident.as_str().to_string()),
                Some("no profile active".into()),
            );
        }
        (None, None)
    }

    async fn run_activation(&self, name: &str, my_gen: u64) -> Result<(), String> {
        let Some(profile) = self.profile(name) else {
            return Err(format!("unknown profile '{name}'"));
        };

        // Highest priority first; unprioritized slots follow in declaration
        // order (stable sort — config.rs documents the semantics).
        let mut targets: Vec<(usize, &ProfileSlot)> = profile.slots.iter().enumerate().collect();
        targets.sort_by_key(|(i, s)| (s.priority.unwrap_or(u32::MAX), *i));

        // 1. Diff: unload everything resident that the target doesn't name.
        let target_ids: HashSet<&str> = profile.slots.iter().map(|s| s.id.as_str()).collect();
        let resident: Vec<String> = match self.inner.lock() {
            Ok(inner) => inner
                .slots
                .iter()
                .filter(|(_, st)| matches!(st, SlotState::Loaded(_) | SlotState::Loading))
                .map(|(id, _)| id.clone())
                .collect(),
            Err(_) => return Err("slot state poisoned".into()),
        };
        for id in resident {
            if self.stale(my_gen) {
                return Ok(()); // superseded; the newer run diffs from scratch
            }
            if target_ids.contains(id.as_str()) {
                continue; // overlapping models stay resident
            }
            match self.unload_via_engine(&id).await {
                Ok(()) => {
                    self.set_slot(&id, SlotState::Unloaded);
                    self.remember_unloaded(&id);
                    self.events.emit(EngineEvent::SlotUnloaded { id });
                }
                Err(e) => tracing::warn!("unload '{id}' failed during switch: {e}"),
            }
        }

        // 2. Load target slots in priority order through the ladder.
        let mut loaded_any = false;
        let mut dropped: Vec<String> = Vec::new();
        let mut shrunk: Vec<String> = Vec::new();
        let mut demoted: Vec<String> = Vec::new();
        let mut refusal: Option<String> = None;

        for (idx, (_, slot)) in targets.iter().enumerate() {
            if self.stale(my_gen) {
                return Ok(());
            }
            if matches!(
                self.inner
                    .lock()
                    .ok()
                    .and_then(|i| i.slots.get(&slot.id).cloned()),
                Some(SlotState::Loaded(_))
            ) {
                loaded_any = true; // overlap: already resident
                continue;
            }
            match self
                .load_through_ladder(&slot.id, slot.placement, my_gen)
                .await
            {
                LadderOutcome::Loaded {
                    shrunk: was_shrunk,
                    demoted: was_demoted,
                } => {
                    loaded_any = true;
                    if was_shrunk {
                        shrunk.push(slot.id.clone());
                    }
                    if was_demoted {
                        demoted.push(slot.id.clone());
                    }
                }
                LadderOutcome::Dropped(_, reason) => {
                    dropped.push(slot.id.clone());
                    // First slot refused ⇒ refuse the whole activation
                    // (profiles-contract.md); current state stands.
                    if idx == 0 && !loaded_any {
                        refusal = Some(reason);
                        break;
                    }
                }
                LadderOutcome::Aborted => return Ok(()),
            }
        }

        if let Some(reason) = refusal {
            // Slots after the refused first one were never attempted; record
            // the refusal on them so `/v1` reports why they are missing.
            let first_cause = targets.first().and_then(|(_, first)| {
                match self.inner.lock().ok()?.slots.get(&first.id) {
                    Some(SlotState::Failed { cause, .. }) => Some(*cause),
                    _ => None,
                }
            });
            for (_, slot) in targets.iter().skip(1) {
                let resident = matches!(
                    self.inner
                        .lock()
                        .ok()
                        .and_then(|i| i.slots.get(&slot.id).cloned()),
                    Some(SlotState::Loaded(_))
                );
                if !resident {
                    self.fail_slot(
                        &slot.id,
                        first_cause.unwrap_or(FailCause::Load),
                        &format!("not attempted: activation of '{name}' refused ({reason})"),
                    );
                }
            }
            if let Ok(mut inner) = self.inner.lock() {
                inner.dropped = profile.slots.iter().map(|s| s.id.clone()).collect();
                inner.shrunk.clear();
                inner.demoted.clear();
            }
            self.events.emit(EngineEvent::ProfileDegraded {
                name: name.to_string(),
                dropped: profile.slots.iter().map(|s| s.id.clone()).collect(),
                shrunk: Vec::new(),
                demoted: Vec::new(),
                reason: reason.clone(),
            });
            return Err(format!("activation of '{name}' refused: {reason}"));
        }

        // 3. Commit.
        if self.stale(my_gen) {
            return Ok(());
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.active = Some(name.to_string());
            inner.dropped = dropped.clone();
            inner.shrunk = shrunk.clone();
            inner.demoted = demoted.clone();
        }
        self.persist_state();

        if !dropped.is_empty() || !shrunk.is_empty() || !demoted.is_empty() {
            self.events.emit(EngineEvent::ProfileDegraded {
                name: name.to_string(),
                dropped,
                shrunk,
                demoted,
                reason: "vram".into(),
            });
        }
        self.events.emit(EngineEvent::ProfileActive {
            name: name.to_string(),
        });
        Ok(())
    }

    /// The per-slot degradation ladder. `gpu`: full ctx → halved ctx down
    /// to `min_ctx` (and, for rerankers, a shrinking per-call cap) → drop.
    /// `cpu`: loaded from RAM unconditionally, never budgeted. `auto`: the
    /// GPU ladder first, then CPU instead of a drop (docket M1). The slot
    /// is failed in exactly one place, so an `auto` slot that lands on CPU
    /// never emits a `slot.failed` for its abandoned GPU attempt.
    async fn load_through_ladder(
        &self,
        id: &str,
        placement: Option<Placement>,
        my_gen: u64,
    ) -> LadderOutcome {
        self.set_slot(id, SlotState::Loading);
        let outcome = match self.catalog_and_registry(id) {
            Err(reason) => LadderOutcome::Dropped(FailCause::Load, reason),
            Ok((catalog, reg_entry)) => match placement.unwrap_or(Placement::Gpu) {
                Placement::Cpu => self.load_on_cpu(id, &reg_entry).await,
                Placement::Gpu => self.load_on_gpu(id, &catalog, &reg_entry, my_gen).await,
                Placement::Auto => match self.load_on_gpu(id, &catalog, &reg_entry, my_gen).await {
                    LadderOutcome::Dropped(_, gpu_reason) if !self.stale(my_gen) => {
                        tracing::info!("demoting '{id}' to CPU: {gpu_reason}");
                        match self.load_on_cpu(id, &reg_entry).await {
                            LadderOutcome::Loaded { shrunk, .. } => LadderOutcome::Loaded {
                                shrunk,
                                demoted: true,
                            },
                            LadderOutcome::Dropped(cause, cpu_reason) => LadderOutcome::Dropped(
                                cause,
                                format!("{gpu_reason}; CPU fallback failed: {cpu_reason}"),
                            ),
                            aborted => aborted,
                        }
                    }
                    other => other,
                },
            },
        };
        if let LadderOutcome::Dropped(cause, reason) = &outcome {
            self.fail_slot(id, *cause, reason);
        }
        outcome
    }

    /// Catalog + registry entry for a profile slot, or why it can't load.
    fn catalog_and_registry(&self, id: &str) -> Result<(ModelEntry, RegistryEntry), String> {
        let catalog = self
            .catalog_entry(id)
            .ok_or_else(|| format!("model '{id}' is not in the catalog"))?;
        let reg_entry = self
            .deps
            .registry
            .read()
            .map_err(|_| "registry lock poisoned".to_string())?
            .resolve(id)
            .cloned()
            .map_err(|e| e.to_string())?;
        Ok((catalog, reg_entry))
    }

    /// System-RAM residency: no budget check, no shrink ladder.
    async fn load_on_cpu(&self, id: &str, reg_entry: &RegistryEntry) -> LadderOutcome {
        let spec = match self.slot_spec(reg_entry, Placement::Cpu, None, None) {
            Ok(s) => s,
            Err(reason) => return LadderOutcome::Dropped(FailCause::Load, reason),
        };
        match self.load_via_engine(spec).await {
            Ok(()) => {
                self.record_load_success(
                    id,
                    Residency {
                        ctx: self.full_ctx(),
                        on_cpu: true,
                        ..Default::default()
                    },
                );
                LadderOutcome::Loaded {
                    shrunk: false,
                    demoted: false,
                }
            }
            Err(reason) => LadderOutcome::Dropped(FailCause::Load, reason),
        }
    }

    /// The VRAM ladder. Returns the outcome without failing the slot; the
    /// caller decides whether a drop is final or falls back to CPU.
    async fn load_on_gpu(
        &self,
        id: &str,
        catalog: &ModelEntry,
        reg_entry: &RegistryEntry,
        my_gen: u64,
    ) -> LadderOutcome {
        // GPU path: budget = free − margin (fresh snapshot per slot).
        let Some(device) = self.device.as_ref() else {
            let reason = "no accelerator detected (GPU-or-refuse)".to_string();
            return LadderOutcome::Dropped(FailCause::Load, reason);
        };
        device.clear_cache();
        let Some(mem_before) = device.memory() else {
            let reason = "accelerator unreachable".to_string();
            return LadderOutcome::Dropped(FailCause::Load, reason);
        };
        let budget = mem_before.free_bytes.saturating_sub(self.margin_bytes());

        // Rung plan: ctx halving, crossed with the packed-token cap for a
        // rerank slot (§1.1 — its per-call context, not its resident ctx,
        // is what the cap moves; the ctx slope is zero for that kind, so
        // the ctx rungs collapse to one).
        let rerank_cap = self.rerank_cap(reg_entry);
        let cap_rungs: Vec<Option<usize>> = match rerank_cap {
            Some(cap) => Self::rerank_cap_rungs(cap).into_iter().map(Some).collect(),
            None => vec![None],
        };
        let ctx_rungs: Vec<u32> = if rerank_cap.is_some() {
            vec![self.full_ctx()]
        } else {
            ctx_rungs(self.full_ctx(), catalog.min_ctx)
        };
        let rungs: Vec<(Option<usize>, u32)> = cap_rungs
            .into_iter()
            .flat_map(|cap| ctx_rungs.iter().map(move |ctx| (cap, *ctx)))
            .collect();

        // Fast planner pass: when no rung fits even on paper, refuse the
        // slot without a doomed cudaMalloc round-trip.
        let mut costs = Vec::with_capacity(rungs.len());
        for (cap, _ctx) in &rungs {
            match self.slot_cost(reg_entry, *cap) {
                Some(cost) => costs.push(cost),
                None => {
                    let reason =
                        format!("cannot estimate fit for '{}' (unreadable GGUF header)", id);
                    return LadderOutcome::Dropped(FailCause::Load, reason);
                }
            }
        }
        let gguf_for_measure = llama_bridge::resolve_gguf_path(reg_entry).ok();
        if rungs
            .iter()
            .zip(&costs)
            .all(|((_, ctx), cost)| cost.at(*ctx) > budget)
        {
            let reason = format!(
                "'{}' needs {} bytes at full ctx, {} free after margin (no rung fits)",
                id,
                costs[0].at(self.full_ctx()),
                budget
            );
            return LadderOutcome::Dropped(FailCause::Vram, reason);
        }
        for (rung_idx, ((cap, ctx), cost)) in rungs.iter().zip(&costs).enumerate() {
            let ctx = *ctx;
            if self.stale(my_gen) {
                return LadderOutcome::Aborted;
            }
            let projected = cost.at(ctx);
            if projected > budget {
                continue; // planner filter: this rung cannot fit
            }
            let shrunk = rung_idx > 0 || ctx != self.full_ctx();
            let mut residency = Residency {
                ctx,
                estimated_bytes: Some(projected),
                shrunk,
                micro_tokens: *cap,
                ..Default::default()
            };
            let spec = match self.slot_spec(reg_entry, Placement::Gpu, Some(ctx), *cap) {
                Ok(s) => s,
                Err(reason) => {
                    return LadderOutcome::Dropped(FailCause::Load, reason);
                }
            };
            match self.load_via_engine(spec).await {
                Ok(()) => {
                    if self.stale(my_gen) {
                        // Late finish after a supersede: the load is REAL —
                        // record it so residency views stay truthful. A
                        // slot the newer profile never wanted becomes a
                        // visible non-pinned resident (the TTL sweeper /
                        // next activation diff cleans it up).
                        self.record_load_success(id, residency);
                        return LadderOutcome::Aborted;
                    }
                    let measured = device.memory().and_then(|after| {
                        let delta =
                            i128::from(mem_before.free_bytes) - i128::from(after.free_bytes);
                        u64::try_from(delta.max(0)).ok()
                    });
                    if let (Some(bytes), Some(gguf)) = (measured, gguf_for_measure.as_ref()) {
                        if let Some(mtime) = gguf_mtime_secs(gguf) {
                            if let Ok(mut store) = self.measured.lock() {
                                store.insert(
                                    id.to_string(),
                                    MeasuredEntry {
                                        bytes,
                                        ctx,
                                        engine_version: STATE_ENGINE_VERSION.into(),
                                        gguf_mtime_secs: mtime,
                                    },
                                );
                            }
                            self.persist_state();
                        }
                    }
                    if let Some(after) = device.memory() {
                        if after.free_bytes < self.margin_bytes() {
                            self.events.emit(EngineEvent::VramPressure {
                                free_bytes: after.free_bytes,
                            });
                        }
                    }
                    residency.measured_bytes = measured;
                    self.record_load_success(id, residency);
                    return LadderOutcome::Loaded {
                        shrunk,
                        demoted: false,
                    };
                }
                // Load-time OOM (cudaMalloc) lands here: slide down the
                // ladder rather than giving up on the slot.
                Err(reason) => {
                    tracing::warn!("load '{id}' at ctx={ctx} cap={cap:?} failed: {reason}");
                }
            }
        }

        let reason = format!(
            "'{}' needs {} bytes at full ctx, {} free after margin (all rungs failed)",
            id,
            costs[0].at(self.full_ctx()),
            budget
        );
        LadderOutcome::Dropped(FailCause::Vram, reason)
    }

    fn record_load_success(&self, id: &str, residency: Residency) {
        self.set_slot(id, SlotState::Loaded(residency));
        self.remember_loaded(id);
        self.touch(id);
        self.events
            .emit(EngineEvent::SlotLoaded { id: id.to_string() });
    }

    fn fail_slot(&self, id: &str, cause: FailCause, reason: &str) {
        self.set_slot(
            id,
            SlotState::Failed {
                cause,
                reason: reason.to_string(),
            },
        );
        self.remember_unloaded(id);
        self.events.emit(EngineEvent::SlotFailed {
            id: id.to_string(),
            reason: reason.to_string(),
        });
    }

    // -- JIT slots -----------------------------------------------------------

    /// Load a catalog model OUTSIDE the active profile. Must fit free VRAM
    /// without evicting anything; never touches pinned members.
    pub async fn jit_load(&self, id: &str) -> Result<(), SlotOpError> {
        if self.is_pinned(id) {
            return Err(SlotOpError::Pinned);
        }
        if self.switching() {
            return Err(SlotOpError::SwitchInProgress);
        }
        if matches!(
            self.inner
                .lock()
                .ok()
                .and_then(|i| i.slots.get(id).cloned()),
            Some(SlotState::Loaded(_)) | Some(SlotState::Loading)
        ) {
            return Err(SlotOpError::AlreadyResident);
        }

        // The registry is the loadable universe (user catalog entries plus
        // built-ins); a JIT slot need not appear in any [[profile]] or even
        // in `config.model`.
        let reg_entry = {
            let reg = self
                .deps
                .registry
                .read()
                .map_err(|_| SlotOpError::LoadFailed("registry lock poisoned".into()))?;
            match reg.resolve(id).cloned() {
                Ok(entry) => entry,
                Err(_) => return Err(SlotOpError::UnknownModel),
            }
        };

        // JIT is a GPU operation (placement is profile intent; these slots
        // have no profile).
        let Some(device) = self.device.as_ref() else {
            return Err(SlotOpError::Vram("no accelerator detected".into()));
        };
        device.clear_cache();
        let mem = device
            .memory()
            .ok_or_else(|| SlotOpError::Vram("accelerator unreachable".into()))?;
        let budget = mem.free_bytes.saturating_sub(self.margin_bytes());

        let rerank_cap = self.rerank_cap(&reg_entry);
        let cost = self.slot_cost(&reg_entry, rerank_cap).ok_or_else(|| {
            SlotOpError::Vram("cannot estimate fit (unreadable GGUF header)".into())
        })?;
        let full = self.full_ctx();
        let needed = cost.at(full);
        if needed > budget {
            return Err(SlotOpError::Vram(format!(
                "'{id}' needs {needed} bytes, {budget} available after margin"
            )));
        }

        self.set_slot(id, SlotState::Loading);
        let spec = self
            .slot_spec(&reg_entry, Placement::Gpu, None, rerank_cap)
            .map_err(SlotOpError::LoadFailed)?;
        match self.load_via_engine(spec).await {
            Ok(()) => {
                self.record_load_success(
                    id,
                    Residency {
                        ctx: full,
                        estimated_bytes: Some(needed),
                        measured_bytes: None,
                        shrunk: false,
                        micro_tokens: rerank_cap,
                        on_cpu: false,
                    },
                );
                Ok(())
            }
            Err(reason) => {
                self.fail_slot(id, FailCause::Load, &reason);
                Err(SlotOpError::LoadFailed(reason))
            }
        }
    }

    /// Unload a JIT slot. Pinned members refuse (409 upstream).
    pub async fn jit_unload(&self, id: &str) -> Result<(), SlotOpError> {
        let known = self
            .deps
            .registry
            .read()
            .map(|reg| reg.resolve(id).is_ok())
            .unwrap_or(false);
        if !known {
            return Err(SlotOpError::UnknownModel);
        }
        if self.is_pinned(id) {
            return Err(SlotOpError::Pinned);
        }
        if self.switching() {
            return Err(SlotOpError::SwitchInProgress);
        }
        let resident = matches!(
            self.inner
                .lock()
                .ok()
                .and_then(|i| i.slots.get(id).cloned()),
            Some(SlotState::Loaded(_)) | Some(SlotState::Loading)
        );
        if !resident {
            return Err(SlotOpError::NotResident);
        }
        self.unload_via_engine(id)
            .await
            .map_err(SlotOpError::LoadFailed)?;
        self.set_slot(id, SlotState::Unloaded);
        self.remember_unloaded(id);
        self.events
            .emit(EngineEvent::SlotUnloaded { id: id.to_string() });
        Ok(())
    }

    /// Is `id` a member of the currently active profile?
    fn is_pinned(&self, id: &str) -> bool {
        let active = self.inner.lock().ok().and_then(|i| i.active.clone());
        active
            .and_then(|name| self.profile(&name))
            .is_some_and(|p| p.slots.iter().any(|s| s.id == id))
    }

    /// Request-resolution touch point for the future JIT TTL sweeper.
    pub fn touch(&self, id: &str) {
        if let Ok(mut lu) = self.last_used.lock() {
            lu.insert(id.to_string(), Instant::now());
        }
    }

    // -- /v1 role aliases ----------------------------------------------------

    /// `"role:<name>"` → the active profile's highest-priority slot serving
    /// that role. Anything else — including an unresolvable role name —
    /// passes through unchanged, so explicit ids keep working verbatim.
    pub fn resolve_alias(&self, name: &str) -> String {
        let Some(role_name) = name.strip_prefix("role:") else {
            return name.to_string();
        };
        let Some(role) = parse_role(role_name) else {
            return name.to_string();
        };
        self.resolve_role_slot(role)
            .unwrap_or_else(|| name.to_string())
    }

    /// Highest-priority resident-slot candidate for `role`: same ordering
    /// semantics as activation (priority first, declaration order breaking
    /// ties). Membership in the ACTIVE profile is what counts — residency
    /// is checked separately by availability.
    fn resolve_role_slot(&self, role: ModelRole) -> Option<String> {
        let active = self.inner.lock().ok()?.active.clone()?;
        let profile = self.profile(&active)?;
        let mut ordered: Vec<&ProfileSlot> = profile.slots.iter().collect();
        ordered.sort_by_key(|s| s.priority.unwrap_or(u32::MAX)); // stable ⇒ declaration tiebreak
        ordered
            .into_iter()
            .find(|s| self.effective_role(&s.id) == role)
            .map(|s| s.id.clone())
    }

    // -- /v1 structured unavailability ---------------------------------------

    /// Why `id` cannot serve right now, derived from its real slot state
    /// (residency contract §1.5). `engine_loaded` is the caller's view of
    /// whether any engine exists at all.
    pub fn unavailability(&self, id: &str, engine_loaded: bool) -> Unavailable {
        let (state, dropped, degraded) = self
            .inner
            .lock()
            .map(|i| {
                (
                    i.slots.get(id).cloned(),
                    i.dropped.iter().any(|d| d == id),
                    !i.dropped.is_empty() || !i.shrunk.is_empty(),
                )
            })
            .unwrap_or((None, false, false));
        let (reason, detail) = match state {
            _ if !engine_loaded => (UnavailableReason::NoEngine, None),
            Some(SlotState::Loading) => (UnavailableReason::Loading, None),
            Some(SlotState::Failed { cause, reason }) => (
                match cause {
                    FailCause::Vram => UnavailableReason::DroppedVram,
                    FailCause::Load => UnavailableReason::LoadFailed,
                },
                Some(reason),
            ),
            // Unloaded but recorded as dropped: evicted by OOM recovery.
            _ if dropped => (
                UnavailableReason::DroppedVram,
                Some(format!(
                    "'{id}' was evicted to recover from a CUDA out-of-memory error"
                )),
            ),
            _ => (UnavailableReason::NotResident, None),
        };
        Unavailable {
            id: id.to_string(),
            reason,
            detail,
            profile_degraded: degraded,
        }
    }

    /// Whether this session's activation degraded the profile (dropped or
    /// shrunk a slot). Carried on every `slot_unavailable` 503.
    pub fn profile_degraded(&self) -> bool {
        self.inner
            .lock()
            .map(|i| !i.dropped.is_empty() || !i.shrunk.is_empty())
            .unwrap_or(false)
    }

    /// VRAM footprint for `/v1/models`: a fresh measurement wins over the
    /// planner estimate; non-resident models report the header estimate and
    /// CPU-resident or in-flight slots report nothing.
    pub fn vram_bytes(&self, id: &str) -> Option<u64> {
        if let Ok(inner) = self.inner.lock() {
            match inner.slots.get(id) {
                Some(SlotState::Loaded(r)) => return r.measured_bytes.or(r.estimated_bytes),
                Some(SlotState::Loading) | Some(SlotState::Failed { .. }) => return None,
                _ => {}
            }
        }
        let entry = self
            .deps
            .registry
            .read()
            .ok()
            .and_then(|reg| reg.resolve(id).ok().cloned())?;
        let cap = self.rerank_cap(&entry);
        self.slot_cost(&entry, cap).map(|c| c.at(self.full_ctx()))
    }

    // -- JIT TTL sweeper -------------------------------------------------------

    /// Background loop unloading unpinned JIT slots idle beyond
    /// [`JIT_TTL_SECS`]. Spawned once at boot.
    pub async fn ttl_sweeper_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(SWEEP_INTERVAL_SECS));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            self.sweep_jit_slots().await;
        }
    }

    /// Unload every resident non-profile slot whose last use predates the
    /// TTL. Pinned members are never touched; a switch in flight defers the
    /// whole sweep to the next tick.
    async fn sweep_jit_slots(&self) {
        if self.switching() {
            return;
        }
        let deadline = Instant::now() - Duration::from_secs(JIT_TTL_SECS);
        let pinned: HashSet<String> = self
            .inner
            .lock()
            .ok()
            .and_then(|i| i.active.clone())
            .and_then(|name| self.profile(&name))
            .map(|p| p.slots.iter().map(|s| s.id.clone()).collect())
            .unwrap_or_default();
        let expired: Vec<String> = {
            let inner = match self.inner.lock() {
                Ok(i) => i,
                Err(_) => return,
            };
            let lu = match self.last_used.lock() {
                Ok(l) => l,
                Err(_) => return,
            };
            inner
                .slots
                .iter()
                .filter(|(id, st)| {
                    matches!(st, SlotState::Loaded(_))
                        && !pinned.contains(*id)
                        && lu.get(*id).is_none_or(|t| *t < deadline)
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        for id in expired {
            if self.switching() {
                return; // superseded mid-sweep; next tick re-derives
            }
            match self.unload_via_engine(&id).await {
                Ok(()) => {
                    self.set_slot(&id, SlotState::Unloaded);
                    self.remember_unloaded(&id);
                    self.events.emit(EngineEvent::SlotUnloaded { id });
                }
                Err(e) => tracing::warn!("TTL unload '{id}' failed: {e}"),
            }
        }
    }

    // -- runtime OOM recovery ---------------------------------------------------

    /// Decode-error funnel (`profiles-contract.md` §6): cheap sync pattern
    /// check on the hot path; actual recovery runs as its own task. Safe to
    /// call with any error text — non-OOM messages are ignored.
    pub fn note_runtime_failure(self: &Arc<Self>, failing_alias: &str, msg: &str) {
        if !is_oom_error(msg) {
            return;
        }
        tracing::warn!("CUDA OOM while serving '{failing_alias}'; recovering");
        let this = self.clone();
        let failing = failing_alias.to_string();
        tokio::spawn(this.on_runtime_oom(failing));
    }

    /// One OOM recovery attempt: evict exactly one slot — preferring idle
    /// JIT residents (oldest use first), else the active profile's
    /// lowest-priority member — never the failing slot itself. The failing
    /// request has already failed; it is not retried server-side.
    ///
    /// Simplification (same accepted tradeoff as activation): the llama
    /// worker is sequential, so when this runs no other slot is EXECUTING;
    /// requests merely queued against the victim fail their next turn with
    /// a structured 503. Boot re-activation remains the crash-level path.
    pub async fn on_runtime_oom(self: Arc<Self>, failing: String) {
        if self.recovering.swap(true, Ordering::SeqCst) {
            return; // one recovery at a time
        }
        let outcome: Result<(), String> = async {
            let Some(victim) = self.pick_oom_victim(&failing) else {
                // Nothing evictable: report pressure and stand down. Boot
                // re-activation handles the crash-level case.
                self.events.emit(EngineEvent::VramPressure {
                    free_bytes: self
                        .device
                        .as_ref()
                        .and_then(|d| d.memory())
                        .map(|m| m.free_bytes)
                        .unwrap_or(0),
                });
                return Ok(());
            };
            let member_of_active = self.is_pinned(&victim);
            self.unload_via_engine(&victim).await?;
            self.set_slot(&victim, SlotState::Unloaded);
            self.remember_unloaded(&victim);
            self.events
                .emit(EngineEvent::SlotUnloaded { id: victim.clone() });
            // An `auto` member moves to CPU instead of disappearing
            // (docket M1: reach a steady state rather than drop).
            if member_of_active
                && self.slot_placement(&victim) == Placement::Auto
                && self.demote_to_cpu(&victim).await
            {
                return Ok(());
            }
            // Losing a profile member is degradation; announce it so the
            // consumer can surface "OCR unavailable" rather than guess.
            if member_of_active {
                if let Ok(mut inner) = self.inner.lock() {
                    if !inner.dropped.contains(&victim) {
                        inner.dropped.push(victim.clone());
                    }
                }
                if let Some(name) = self.inner.lock().ok().and_then(|i| i.active.clone()) {
                    self.events.emit(EngineEvent::ProfileDegraded {
                        name,
                        dropped: vec![victim],
                        shrunk: Vec::new(),
                        demoted: Vec::new(),
                        reason: "vram".into(),
                    });
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = outcome {
            tracing::warn!("runtime OOM recovery failed: {e}");
        }
        self.recovering.store(false, Ordering::SeqCst);
    }

    /// Reload an already-unloaded `auto` member on CPU after an OOM
    /// eviction. `false` (slot failed with the reason) when even the CPU
    /// load is impossible; the caller then records it as dropped.
    async fn demote_to_cpu(&self, id: &str) -> bool {
        let outcome = match self.catalog_and_registry(id) {
            Ok((_, reg_entry)) => self.load_on_cpu(id, &reg_entry).await,
            Err(reason) => LadderOutcome::Dropped(FailCause::Load, reason),
        };
        match outcome {
            LadderOutcome::Loaded { .. } => {
                tracing::warn!("demoted '{id}' to CPU after a CUDA out-of-memory error");
                if let Ok(mut inner) = self.inner.lock() {
                    if !inner.demoted.iter().any(|d| d == id) {
                        inner.demoted.push(id.to_string());
                    }
                }
                if let Some(name) = self.inner.lock().ok().and_then(|i| i.active.clone()) {
                    self.events.emit(EngineEvent::ProfileDegraded {
                        name,
                        dropped: Vec::new(),
                        shrunk: Vec::new(),
                        demoted: vec![id.to_string()],
                        reason: "vram".into(),
                    });
                }
                true
            }
            LadderOutcome::Dropped(cause, reason) => {
                self.fail_slot(
                    id,
                    cause,
                    &format!("CPU demotion after OOM failed: {reason}"),
                );
                false
            }
            LadderOutcome::Aborted => false,
        }
    }

    /// Configured placement of `id`: the active profile's slot wins, then
    /// any profile declaring it; `Gpu` when unspecified (JIT slots).
    fn slot_placement(&self, id: &str) -> Placement {
        let active = self.inner.lock().ok().and_then(|i| i.active.clone());
        let from = |p: &ProfileConfig| {
            p.slots
                .iter()
                .find(|s| s.id == id)
                .and_then(|s| s.placement)
        };
        active
            .and_then(|name| self.profile(&name))
            .and_then(|p| from(&p))
            .or_else(|| self.config.profile.iter().find_map(from))
            .unwrap_or(Placement::Gpu)
    }

    /// Eviction choice: oldest-touched JIT resident wins; otherwise the
    /// lowest-priority (highest priority number, latest declared) profile
    /// member. The failing slot itself is never chosen.
    fn pick_oom_victim(&self, failing: &str) -> Option<String> {
        let now = Instant::now();
        let inner = self.inner.lock().ok()?;
        let members: HashMap<String, (u32, usize)> = inner
            .active
            .clone()
            .and_then(|name| self.profile(&name))
            .map(|p| {
                p.slots
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.id.clone(), (s.priority.unwrap_or(u32::MAX), i)))
                    .collect()
            })
            .unwrap_or_default();

        let mut jits: Vec<(String, Instant)> = Vec::new();
        let mut pinned_members: Vec<(String, u32, usize)> = Vec::new();
        for (id, st) in inner.slots.iter() {
            // Only GPU residents free VRAM when evicted.
            let on_gpu = matches!(st, SlotState::Loaded(r) if !r.on_cpu);
            if id == failing || !on_gpu {
                continue;
            }
            match members.get(id) {
                Some(&(prio, idx)) => pinned_members.push((id.clone(), prio, idx)),
                None => {
                    let touched = self
                        .last_used
                        .lock()
                        .ok()
                        .and_then(|lu| lu.get(id).copied())
                        .unwrap_or(now);
                    jits.push((id.clone(), touched));
                }
            }
        }

        jits.sort_by_key(|(_, t)| *t);
        if let Some((id, _)) = jits.first() {
            return Some(id.clone());
        }
        pinned_members.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        pinned_members.first().map(|(id, _, _)| id.clone())
    }

    // -- views ---------------------------------------------------------------

    /// `(ready, {ready, phase:{…}})` — the readiness gate shared by
    /// `/engine/ready` and the extra fields on `/engine/status`
    /// (residency contract §1.4). Live slot state decides `ready`; the
    /// tracked phase supplies what only boot knows (`loading` retries,
    /// `failed`, `downloading`). `engine_loaded` is the caller's view.
    ///
    /// `switching` is part of the gate (§1.4: ready is true only when "no
    /// activation is in progress"): while a target profile is still loading,
    /// the slots of the OUTGOING profile may all be resident and serving, so
    /// `is_serving()` alone would advertise the model the caller is waiting
    /// to replace. The phase then reads `loading`, which is what bounds the
    /// poll (`Retry-After`).
    pub fn readiness(&self, engine_loaded: bool) -> (bool, serde_json::Value) {
        let ready = engine_loaded && !self.switching() && self.is_serving();
        let mut phase = self.phase.lock().map(|p| p.clone()).unwrap_or_default();
        match (ready, phase.phase) {
            (true, _) => phase.phase = "ready",
            // Serving state regressed (unloaded/deactivated): never keep
            // advertising a stale `ready`.
            (false, "ready") => phase.phase = "loading",
            _ => {}
        }
        let body = serde_json::json!({
            "ready": ready,
            "phase": {
                "phase": phase.phase,
                "model": phase.model,
                "bytes_done": phase.bytes_done,
                "bytes_total": phase.bytes_total,
                "error": phase.error,
                "reason": phase.reason,
                "detail": phase.detail,
            },
        });
        (ready, body)
    }

    /// `GET /engine/status` payload — liveness plus per-slot residency.
    /// Always 200 while the process is up; `ready` + `phase` are the
    /// readiness view (residency contract §1.4, amendment).
    pub fn status(&self, engine_loaded: bool) -> serde_json::Value {
        let inner = self.inner.lock().map(|i| i.clone()).unwrap_or_default();
        let mut ids: Vec<String> = inner.slots.keys().cloned().collect();
        for p in &self.config.profile {
            for s in &p.slots {
                if !ids.contains(&s.id) {
                    ids.push(s.id.clone());
                }
            }
        }

        let slots: Vec<serde_json::Value> = ids
            .iter()
            .map(|id| {
                let st = inner.slots.get(id);
                let (label, ctx, estimated, measured, shrunk, reason, micro_tokens) = match st {
                    Some(SlotState::Loaded(r)) => (
                        "loaded",
                        Some(r.ctx),
                        r.estimated_bytes,
                        r.measured_bytes,
                        r.shrunk,
                        None,
                        r.micro_tokens,
                    ),
                    Some(SlotState::Failed { reason, .. }) => (
                        "failed",
                        None,
                        None,
                        None,
                        false,
                        Some(reason.clone()),
                        None,
                    ),
                    Some(SlotState::Loading) => ("loading", None, None, None, false, None, None),
                    _ => ("unloaded", None, None, None, false, None, None),
                };
                // Where the weights actually live once loaded (§1.2: the
                // reported placement must describe real residency).
                let configured = self.slot_placement(id);
                let placement = match st {
                    Some(SlotState::Loaded(r)) if r.on_cpu => Placement::Cpu,
                    Some(SlotState::Loaded(_)) => Placement::Gpu,
                    _ => configured,
                };
                let priority = self
                    .config
                    .profile
                    .iter()
                    .flat_map(|p| p.slots.iter())
                    .find(|s| s.id == *id)
                    .and_then(|s| s.priority);
                serde_json::json!({
                    "id": id,
                    "role": self.effective_role(id),
                    "priority": priority,
                    "placement": placement,
                    "configured_placement": configured,
                    "demoted": inner.demoted.contains(id),
                    "state": label,
                    "ctx": ctx,
                    "estimated_bytes": estimated,
                    "measured_bytes": measured,
                    "shrunk": shrunk,
                    "pinned": self.is_pinned(id),
                    "reason": reason,
                    "micro_tokens": micro_tokens,
                })
            })
            .collect();

        let (ready, readiness) = self.readiness(engine_loaded);
        serde_json::json!({
            "active": inner.active,
            "switching": self.switching(),
            "degraded": {
                "dropped": inner.dropped,
                "shrunk": inner.shrunk,
                "demoted": inner.demoted,
            },
            "slots": slots,
            "ready": ready,
            "phase": readiness["phase"],
        })
    }

    /// `GET /engine/device` payload.
    pub fn device_view(&self) -> serde_json::Value {
        match self.device.as_ref().and_then(|d| d.memory()) {
            Some(mem) => serde_json::json!({
                "available": true,
                "backend": self.device.as_ref().map(|d| d.name()),
                "total_bytes": mem.total_bytes,
                "free_bytes": mem.free_bytes,
                "margin_bytes": self.margin_bytes(),
            }),
            None => serde_json::json!({ "available": false, "backend": null }),
        }
    }

    /// `GET /engine/profiles` payload: definitions + active + per-slot /
    /// per-profile fit previews. Overlapping models across profiles are
    /// counted per profile (a preview, not an accounting identity).
    pub fn profiles_view(&self) -> serde_json::Value {
        let active = self.inner.lock().ok().and_then(|i| i.active.clone());

        let profiles: Vec<serde_json::Value> = self
            .config
            .profile
            .iter()
            .map(|p| {
                let mut total = Some(0u64);
                let slots: Vec<serde_json::Value> = p
                    .slots
                    .iter()
                    .map(|s| {
                        let est = self
                            .deps
                            .registry
                            .read()
                            .ok()
                            .and_then(|reg| reg.resolve(&s.id).ok().cloned())
                            .and_then(|entry| {
                                let cap = self.rerank_cap(&entry);
                                self.slot_cost(&entry, cap)
                            })
                            .map(|c| c.at(self.full_ctx()));
                        if est.is_none() {
                            total = None;
                        } else {
                            total = total.map(|t| t + est.unwrap_or(0));
                        }
                        serde_json::json!({
                            "id": s.id,
                            "priority": s.priority,
                            "placement": s.placement.unwrap_or(Placement::Gpu),
                            "role": self.effective_role(&s.id),
                            "estimated_bytes": est,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "name": p.name,
                    "slots": slots,
                    "estimated_bytes": total,
                })
            })
            .collect();

        serde_json::json!({ "active": active, "profiles": profiles })
    }
}

// helpers ----------------------------------------------------------------

enum LadderOutcome {
    /// `demoted`: an `auto` slot that fell back to CPU.
    Loaded {
        shrunk: bool,
        demoted: bool,
    },
    Dropped(FailCause, String),
    Aborted,
}

/// Errors the slot routes map onto HTTP statuses.
#[derive(Debug)]
pub enum SlotOpError {
    UnknownModel,       // 404
    Pinned,             // 409
    SwitchInProgress,   // 503 + Retry-After
    AlreadyResident,    // 409 (load of an already-loaded JIT slot)
    NotResident,        // unload no-op
    Vram(String),       // 503 structured
    LoadFailed(String), // 503
}

fn gguf_mtime_secs(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    meta.modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// `state.json` under [`data_dir`](mettle_core::config::data_dir),
/// overridable via `METTLE_STATE` (tests point this at a tempfile).
pub fn default_state_path() -> PathBuf {
    if let Ok(p) = std::env::var("METTLE_STATE") {
        return PathBuf::from(p);
    }
    mettle_core::config::data_dir().join("state.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mettle_core::device::DeviceMemory;
    use mettle_core::engine::{Engine, EngineError, EngineRequest};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    fn active_name(m: &ProfileManager) -> Option<String> {
        m.inner.lock().unwrap().active.clone()
    }

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    /// Metadata-only v3 GGUF with real KV-geometry KVs, so `estimate_vram`
    /// produces a genuine KV slope; a 4 MiB body makes the file size real.
    fn write_synthetic_gguf(path: &Path) {
        const TYPE_STRING: u32 = 8;
        const TYPE_UINT32: u32 = 4;
        let arch = "qwen3";
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensors
        buf.extend_from_slice(&8u64.to_le_bytes()); // kv pairs
        let put_str = |buf: &mut Vec<u8>, s: &str| {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        };
        put_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&TYPE_STRING.to_le_bytes());
        put_str(&mut buf, arch);
        for (k, v) in [
            (format!("{arch}.block_count"), 36u32),
            (format!("{arch}.embedding_length"), 2560),
            (format!("{arch}.attention.head_count"), 32),
            (format!("{arch}.attention.head_count_kv"), 8),
            (format!("{arch}.attention.key_length"), 128),
            (format!("{arch}.attention.value_length"), 128),
            (format!("{arch}.vocab_size"), 151_936),
        ] {
            put_str(&mut buf, &k);
            buf.extend_from_slice(&TYPE_UINT32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
        f.write_all(&vec![0u8; 4 * MIB as usize]).unwrap();
        f.sync_all().unwrap();
    }

    struct FakeDevice {
        free: Arc<AtomicU64>,
    }
    impl Device for FakeDevice {
        fn name(&self) -> &str {
            "fake"
        }
        fn memory(&self) -> Option<DeviceMemory> {
            Some(DeviceMemory {
                total_bytes: 8 * GIB,
                free_bytes: self.free.load(AtomicOrdering::SeqCst),
            })
        }
    }

    #[derive(Default)]
    struct FakeEngine {
        loaded: Mutex<Vec<String>>,
        ops: Mutex<Vec<String>>,
        fail: Mutex<HashMap<String, String>>,
        /// One-shot park gate: `(nth, signal)` parks exactly load attempt
        /// `nth` of the alias (0 = first attempt).
        park_nth: parking_lot::Mutex<HashMap<String, (usize, std::sync::mpsc::Receiver<()>)>>,
        loads_started: parking_lot::Mutex<HashMap<String, usize>>,
    }
    impl FakeEngine {
        fn op_log(&self) -> Vec<String> {
            self.ops.lock().unwrap().clone()
        }
        fn wait_for_op(&self, needle: &str) {
            for _ in 0..5000 {
                if self.op_log().iter().any(|op| op.contains(needle)) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            panic!("no op containing '{needle}'; saw {:?}", self.op_log());
        }
    }
    impl Engine for FakeEngine {
        fn sender(&self) -> tokio::sync::mpsc::Sender<EngineRequest> {
            let (tx, _rx) = tokio::sync::mpsc::channel(1);
            tx
        }
        fn loaded(&self) -> Vec<String> {
            self.loaded.lock().unwrap().clone()
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
            // Attempt numbers make parking deterministic: exactly the Nth
            // load of an alias blocks, so a racing second loader can never
            // steal the gate (the old map-keyed gate deadlocked on exactly
            // that race).
            let attempt = {
                let mut starts = self.loads_started.lock();
                let n = *starts.entry(alias.clone()).or_insert(0);
                starts.insert(alias.clone(), n + 1);
                n
            };
            self.ops.lock().unwrap().push(format!(
                "load:{alias}@ctx={:?},ngl={:?},cap={:?}",
                slot.n_ctx, slot.model.n_gpu_layers, slot.rerank_micro_tokens
            ));
            let parks = {
                let gate = self.park_nth.lock();
                gate.get(&alias).is_some_and(|(nth, _)| *nth == attempt)
            };
            if parks {
                let (_, rx) = self.park_nth.lock().remove(&alias).unwrap();
                let _ = rx.recv(); // held until the test releases us
            }
            if let Some(reason) = self.fail.lock().unwrap().get(&alias) {
                return Err(EngineError::Other(reason.clone()));
            }
            // Mirrors LlamaEngine::load_slot: the resident list dedupes.
            let mut loaded = self.loaded.lock().unwrap();
            if !loaded.contains(&alias) {
                loaded.push(alias);
            }
            Ok(())
        }
        fn remove_model(&self, alias: &str) -> Result<(), EngineError> {
            self.ops.lock().unwrap().push(format!("unload:{alias}"));
            self.loaded.lock().unwrap().retain(|a| a != alias);
            Ok(())
        }
    }

    struct Fixture {
        manager: Arc<ProfileManager>,
        engine: Arc<FakeEngine>,
        events: Arc<EventBus>,
        dir: tempfile::TempDir,
        /// Live free-VRAM counter the fake probe reads.
        free: Arc<AtomicU64>,
    }

    impl Fixture {
        /// Each profile is `(name, slot ids in order)` over the default
        /// model pair m1/m2.
        fn build(profiles: &[(&str, &[&str])], free_bytes: u64) -> Self {
            Self::build_fiddled(&["m1", "m2"], profiles, free_bytes, |_| {})
        }

        /// Explicit model universe: every id gets a synthetic GGUF and a
        /// catalog entry.
        fn build_with_models(ids: &[&str], profiles: &[(&str, &[&str])], free_bytes: u64) -> Self {
            Self::build_fiddled(ids, profiles, free_bytes, |_| {})
        }

        /// Fixture with a pre-Arc config tweak (roles, priorities) that
        /// cannot be applied through the `Arc<ProfileManager>` afterwards.
        fn build_fiddled(
            ids: &[&str],
            profiles: &[(&str, &[&str])],
            free_bytes: u64,
            fiddle: impl FnOnce(&mut RootConfig),
        ) -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let mut config = RootConfig::default();
            let mut models = Vec::new();
            for id in ids {
                let gguf = dir.path().join(format!("{id}.gguf"));
                write_synthetic_gguf(&gguf);
                models.push(ModelEntry {
                    id: (*id).into(),
                    default: false,
                    path: Some(gguf),
                    n_gpu_layers: None,
                    mmproj_path: None,
                    role: None,
                    min_ctx: None,
                    draft: None,
                });
            }
            config.model = models;
            config.profile = profiles
                .iter()
                .map(|(name, slots)| ProfileConfig {
                    name: (*name).to_string(),
                    default: false,
                    slots: slots
                        .iter()
                        .map(|id| ProfileSlot {
                            id: (*id).to_string(),
                            priority: None,
                            placement: None,
                        })
                        .collect(),
                })
                .collect();
            fiddle(&mut config);

            let engine = Arc::new(FakeEngine::default());
            let mut registry = ModelRegistry::new();
            registry.apply_user_models(&config.model);
            let deps = ProfileDeps {
                engine: Arc::new(std::sync::RwLock::new(Some(engine.clone() as SharedEngine))),
                registry: Arc::new(std::sync::RwLock::new(registry)),
                loaded_models: Arc::new(Mutex::new(Vec::new())),
            };
            let events = Arc::new(EventBus::new());
            let manager =
                ProfileManager::new(deps, config, events.clone(), dir.path().join("state.json"));
            let mut manager = Box::new(manager);
            // Private field: inject the fake probe before wrapping in Arc.
            let free = Arc::new(AtomicU64::new(free_bytes));
            manager.device = Some(Box::new(FakeDevice { free: free.clone() }));
            Self {
                manager: Arc::new(*manager),
                engine,
                events,
                dir,
                free,
            }
        }

        fn replay_contains(&self, needle: &str) -> bool {
            let (replay, _rx) = self.events.subscribe();
            replay.iter().any(|j| j.contains(needle))
        }
    }

    // -- pure planner ----------------------------------------------------------

    #[test]
    fn ctx_rungs_halve_down_to_min_ctx() {
        assert_eq!(ctx_rungs(8192, None), vec![8192]);
        assert_eq!(ctx_rungs(8192, Some(2048)), vec![8192, 4096, 2048]);
        assert_eq!(ctx_rungs(8192, Some(6000)), vec![8192, 6000]);
        assert_eq!(ctx_rungs(1000, Some(2000)), vec![1000]);
    }

    #[test]
    fn rerank_cap_rungs_halve_down_to_the_floor() {
        assert_eq!(
            ProfileManager::rerank_cap_rungs(2048),
            vec![2048, 1024, 512, 256]
        );
        assert_eq!(ProfileManager::rerank_cap_rungs(256), vec![256]);
        assert_eq!(ProfileManager::rerank_cap_rungs(600), vec![600, 300, 256]);
    }

    /// §1.1: a rerank slot's fit must include the per-call packed-token
    /// context, and when only a smaller cap fits the ladder loads that cap
    /// instead of dropping the slot.
    #[tokio::test]
    async fn rerank_slot_budgets_per_call_context_and_shrinks_cap() {
        let fx = Fixture::build_fiddled(&["r1"], &[("p", &["r1"])], 8 * GIB, |cfg| {
            cfg.model[0].role = Some(mettle_core::config::ModelRole::Rerank);
            cfg.engine.rerank_micro_tokens = Some(2048);
        });
        let m = fx.manager.clone();
        let entry = m
            .deps
            .registry
            .read()
            .unwrap()
            .resolve("r1")
            .unwrap()
            .clone();
        let full = m.full_ctx();
        // The cap moves the cost: a rerank slot's KV is per-call.
        let resident_only = m.slot_cost(&entry, None).unwrap().at(full);
        let at_floor = m.slot_cost(&entry, Some(256)).unwrap().at(full);
        let at_default = m.slot_cost(&entry, Some(2048)).unwrap().at(full);
        assert!(
            at_floor > resident_only && at_default > at_floor,
            "per-call context must be budgeted: {resident_only} {at_floor} {at_default}"
        );

        // Budget that admits the floor cap only.
        fx.free
            .store(at_floor + m.margin_bytes(), AtomicOrdering::SeqCst);
        m.clone().activate("p".into()).await.unwrap();

        let ops = fx.engine.op_log();
        let load = ops.iter().find(|o| o.starts_with("load:r1")).unwrap();
        assert!(load.contains("cap=Some(256)"), "{ops:?}");
        // A lowered cap is a shrunk slot, reported as degradation.
        assert!(m.profile_degraded());
        assert!(
            m.inner.lock().unwrap().shrunk.iter().any(|s| s == "r1"),
            "slot recorded as shrunk"
        );
        // The effective cap is visible to the consumer (§1.1).
        let status = m.status(true);
        assert_eq!(status["slots"][0]["micro_tokens"], 256, "{status}");
    }

    // -- activation --------------------------------------------------------------

    #[tokio::test]
    async fn activate_loads_all_slots_and_persists() {
        let fx = Fixture::build(&[("p", &["m1", "m2"])], 8 * GIB);

        fx.manager.clone().activate("p".into()).await.unwrap();

        assert_eq!(active_name(&fx.manager), Some("p".into()));
        let ops = fx.engine.op_log();
        assert_eq!(ops.len(), 2, "{ops:?}");
        assert!(ops.iter().all(|op| op.contains("ngl=Some(999)")), "{ops:?}");
        assert!(
            ops.iter().any(|op| op.contains("load:m1@ctx=Some(8192)")),
            "{ops:?}"
        );
        assert!(fx.replay_contains("\"profile.activating\""));
        assert!(fx.replay_contains("\"profile.active\""));
        assert!(fx.dir.path().join("state.json").exists(), "state persisted");
    }

    #[tokio::test]
    async fn boot_reactivates_persisted_profile() {
        let fx = Fixture::build(&[("p", &["m1", "m2"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();
        let ops_before = fx.engine.op_log().len();

        let mut registry = ModelRegistry::new();
        registry.apply_user_models(&fx.manager.config.model);
        let registry = Arc::new(std::sync::RwLock::new(registry));
        let fresh = Arc::new(ProfileManager::new(
            ProfileDeps {
                engine: Arc::new(std::sync::RwLock::new(Some(
                    fx.engine.clone() as SharedEngine
                ))),
                registry,
                loaded_models: Arc::new(Mutex::new(Vec::new())),
            },
            fx.manager.config.clone(),
            fx.events.clone(),
            fx.dir.path().join("state.json"),
        ));
        fresh.clone().boot_activate().await;

        assert_eq!(active_name(&fresh), Some("p".into()));
        assert!(fx.engine.op_log().len() > ops_before, "engine reloaded");
    }

    #[tokio::test]
    async fn boot_activates_shipped_default_profile() {
        // Grilling Q11 amendment (adoption contract §1.4): with NOTHING ever
        // persisted, a `default = true` profile in the config activates on
        // first boot instead of idling the engine.
        let fx = Fixture::build_fiddled(&["m1", "m2"], &[("p", &["m1"])], 8 * GIB, |cfg| {
            cfg.profile[0].default = true;
        });

        let fresh = Arc::new(ProfileManager::new(
            ProfileDeps {
                engine: Arc::new(std::sync::RwLock::new(Some(
                    fx.engine.clone() as SharedEngine
                ))),
                registry: Arc::new(std::sync::RwLock::new({
                    let mut r = ModelRegistry::new();
                    r.apply_user_models(&fx.manager.config.model);
                    r
                })),
                loaded_models: Arc::new(Mutex::new(Vec::new())),
            },
            fx.manager.config.clone(),
            fx.events.clone(),
            fx.dir.path().join("state.json"),
        ));
        fresh.clone().boot_activate().await;

        assert_eq!(active_name(&fresh), Some("p".into()));
    }

    #[tokio::test]
    async fn boot_without_default_marker_still_idles() {
        // No persisted state and no default-marked profile -> the engine
        // idles (profiles-contract behavior preserved for non-default
        // configs).
        let fx = Fixture::build(&[("p", &["m1"])], 8 * GIB);
        let fresh = Arc::new(ProfileManager::new(
            ProfileDeps {
                engine: Arc::new(std::sync::RwLock::new(Some(
                    fx.engine.clone() as SharedEngine
                ))),
                registry: Arc::new(std::sync::RwLock::new({
                    let mut r = ModelRegistry::new();
                    r.apply_user_models(&fx.manager.config.model);
                    r
                })),
                loaded_models: Arc::new(Mutex::new(Vec::new())),
            },
            fx.manager.config.clone(),
            fx.events.clone(),
            fx.dir.path().join("state.json"),
        ));
        fresh.clone().boot_activate().await;

        assert_eq!(active_name(&fresh), None);
    }

    // -- readiness phase (§1.4) --------------------------------------------------

    #[tokio::test]
    async fn boot_reaches_ready_and_reports_phase() {
        let fx = Fixture::build_fiddled(&["m1", "m2"], &[("p", &["m1"])], 8 * GIB, |cfg| {
            cfg.profile[0].default = true;
        });
        fx.manager.clone().boot_activate().await;

        let (ready, body) = fx.manager.readiness(true);
        assert!(ready, "{body}");
        assert_eq!(body["ready"], true);
        assert_eq!(body["phase"]["phase"], "ready");
        assert_eq!(body["phase"]["error"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn readiness_is_false_while_a_switch_is_in_flight() {
        // §1.4: ready only when "no activation is in progress". The
        // outgoing profile's slots stay resident while the target loads, so
        // `is_serving()` alone advertises the model being replaced
        // (observed live: 200 with the old profile, then a 503 on the
        // target the caller had just switched to).
        let fx = Fixture::build(&[("p", &["m1"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();
        assert!(fx.manager.readiness(true).0);

        fx.manager.switching.store(true, Ordering::SeqCst);
        let (ready, body) = fx.manager.readiness(true);
        assert!(!ready, "{body}");
        assert_eq!(body["phase"]["phase"], "loading", "{body}");

        fx.manager.switching.store(false, Ordering::SeqCst);
        assert!(fx.manager.readiness(true).0, "cleared switch must re-ready");
    }

    #[tokio::test]
    async fn boot_reports_failed_with_reason_after_window() {
        // Nothing fits the budget, so every slot is dropped and the window
        // cannot end in `ready`: the phase must settle on `failed` with the
        // real reason, not stay `loading` forever.
        let fx = Fixture::build_fiddled(&["m1"], &[("p", &["m1"])], 0, |cfg| {
            cfg.profile[0].default = true;
            cfg.engine.ready_timeout_secs = Some(1);
        });
        fx.manager.clone().boot_activate().await;

        let (ready, body) = fx.manager.readiness(true);
        assert!(!ready);
        assert_eq!(body["phase"]["phase"], "failed", "{body}");
        assert_eq!(body["phase"]["reason"], "dropped_vram", "{body}");
        assert!(body["phase"]["detail"].is_string(), "{body}");
    }

    #[tokio::test]
    async fn readiness_needs_a_live_engine_handle() {
        let fx = Fixture::build(&[("p", &["m1"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();

        // Slots are loaded, but with no engine handle the process cannot
        // serve: `/engine/ready` must gate on both.
        assert!(fx.manager.readiness(true).0);
        let (ready, body) = fx.manager.readiness(false);
        assert!(!ready, "{body}");
        assert_eq!(body["phase"]["phase"], "loading");
    }

    #[tokio::test]
    async fn overlap_stays_resident_across_switch() {
        let fx = Fixture::build(&[("p1", &["m1", "m2"]), ("p2", &["m1"])], 8 * GIB);
        fx.manager.clone().activate("p1".into()).await.unwrap();

        // p2 shares m1 with p1; only m2 should be touched.
        fx.manager.clone().activate("p2".into()).await.unwrap();

        let ops = fx.engine.op_log();
        assert_eq!(
            ops.iter().filter(|op| op.starts_with("load:")).count(),
            2,
            "{ops:?}"
        );
        assert!(ops.contains(&"unload:m2".to_string()), "{ops:?}");
        assert!(!ops.contains(&"unload:m1".to_string()), "{ops:?}");
        assert_eq!(active_name(&fx.manager), Some("p2".into()));
    }

    #[tokio::test]
    async fn refusal_keeps_previous_profile_active() {
        // 'big' must name a NON-resident slot: an overlap slot would be
        // skipped as already resident and never reach the ladder.
        let fx = Fixture::build_with_models(
            &["m1", "m2", "m3"],
            &[("ok", &["m1", "m2"]), ("big", &["m3"])],
            8 * GIB,
        );
        fx.manager.clone().activate("ok".into()).await.unwrap();
        // Make 'big''s only slot unloadable.
        fx.engine
            .fail
            .lock()
            .unwrap()
            .insert("m3".into(), "simulated load failure".into());

        let err = fx.manager.clone().activate("big".into()).await.unwrap_err();
        assert!(err.contains("m3"), "{err}");
        assert_eq!(
            active_name(&fx.manager),
            Some("ok".into()),
            "previous profile kept"
        );
        assert!(fx.replay_contains("\"profile.degraded\""));
    }

    #[tokio::test]
    async fn supersede_aborts_inflight_activation() {
        let fx = Fixture::build(&[("p1", &["m1", "m2"]), ("p2", &["m1"])], 8 * GIB);
        // Park m1's load under p1.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        fx.engine.park_nth.lock().insert("m1".into(), (0, rx));

        let p1 = fx.manager.clone();
        let j1 = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(p1.activate("p1".into()))
        });
        fx.engine.wait_for_op("load:m1");

        // p2 wants only m1; it supersedes p1 mid-load.
        fx.manager.clone().activate("p2".into()).await.unwrap();
        drop(tx); // release p1's parked load
        let _r1 = j1.await.unwrap();

        assert_eq!(active_name(&fx.manager), Some("p2".into()));
        // m1 is resident exactly once despite the double load attempt.
        let resident = fx.engine.loaded();
        assert_eq!(
            resident.iter().filter(|a| *a == "m1").count(),
            1,
            "{resident:?}"
        );
    }

    #[test]
    fn stale_measured_entry_is_discarded() {
        let fx = Fixture::build(&[("p", &["m1", "m2"])], 8 * GIB);
        let entry = fx
            .manager
            .deps
            .registry
            .read()
            .unwrap()
            .resolve("m1")
            .unwrap()
            .clone();
        let gguf = llama_bridge::resolve_gguf_path(&entry).unwrap();
        let mtime = gguf_mtime_secs(&gguf).unwrap();

        // Fresh measurement: accepted.
        fx.manager.measured.lock().unwrap().insert(
            "m1".into(),
            MeasuredEntry {
                bytes: 100 * MIB,
                ctx: 4096,
                engine_version: STATE_ENGINE_VERSION.to_string(),
                gguf_mtime_secs: mtime,
            },
        );
        assert!(fx.manager.measured_cost("m1", &entry).is_some());

        // Wrong mtime: discarded, planner falls back to the header estimate.
        fx.manager
            .measured
            .lock()
            .unwrap()
            .get_mut("m1")
            .unwrap()
            .gguf_mtime_secs = mtime + 1;
        assert!(fx.manager.measured_cost("m1", &entry).is_none());
    }

    #[tokio::test]
    async fn jit_load_pins_profile_members_and_serves_others() {
        let fx = Fixture::build(&[("p", &["m1", "m2"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();

        // Profile members are pinned: JIT load refuses.
        let err = fx.manager.clone().jit_load("m1").await.unwrap_err();
        assert!(matches!(err, SlotOpError::Pinned), "{err:?}");

        // A catalog model outside every profile loads on demand.
        fx.manager
            .deps
            .registry
            .write()
            .unwrap()
            .copy("m1", "m3")
            .unwrap();
        fx.manager.clone().jit_load("m3").await.unwrap();
        assert!(fx.engine.loaded().contains(&"m3".to_string()));
    }

    // -- /v1 role aliases -------------------------------------------------------

    #[test]
    fn role_alias_resolves_by_priority_and_falls_through() {
        let fx = Fixture::build_fiddled(&["m1", "m2"], &[("p", &["m1", "m2"])], 8 * GIB, |cfg| {
            cfg.model[0].role = Some(ModelRole::Embed);
            cfg.profile[0].slots[0].priority = Some(1);
        });
        // Nothing active: passthrough.
        assert_eq!(fx.manager.resolve_alias("role:chat"), "role:chat");

        // Simulate an active profile without loading anything: m1 is the
        // prioritized embed slot, m2 the unprioritized chat fallback.
        fx.manager.inner.lock().unwrap().active = Some("p".into());

        assert_eq!(fx.manager.resolve_alias("role:embed"), "m1");
        assert_eq!(fx.manager.resolve_alias("role:chat"), "m2");
        assert_eq!(fx.manager.resolve_alias("role:nope"), "role:nope");
        assert_eq!(fx.manager.resolve_alias("gemma4:e2b"), "gemma4:e2b");
    }

    #[test]
    fn unavailable_reason_follows_slot_state() {
        use crate::unavailable::UnavailableReason as R;
        let fx = Fixture::build(&[("p", &["m1"])], 8 * GIB);
        let m = &fx.manager;
        let reason = |id: &str, engine: bool| m.unavailability(id, engine).reason;

        assert_eq!(reason("m9", true), R::NotResident);
        assert_eq!(
            reason("m9", false),
            R::NoEngine,
            "no engine wins over slot state"
        );

        m.set_slot("m9", SlotState::Loading);
        assert_eq!(reason("m9", true), R::Loading);

        m.fail_slot(
            "m9",
            FailCause::Vram,
            "needs 9 bytes, 1 free (no rung fits)",
        );
        let u = m.unavailability("m9", true);
        assert_eq!(u.reason, R::DroppedVram);
        assert_eq!(
            u.detail.as_deref(),
            Some("needs 9 bytes, 1 free (no rung fits)")
        );

        m.fail_slot("m9", FailCause::Load, "bad gguf");
        assert_eq!(reason("m9", true), R::LoadFailed);

        // OOM eviction leaves the slot Unloaded but recorded as dropped.
        m.set_slot("m9", SlotState::Unloaded);
        m.inner.lock().unwrap().dropped.push("m9".into());
        let u = m.unavailability("m9", true);
        assert_eq!(u.reason, R::DroppedVram);
        assert!(u.profile_degraded);

        let j = u.payload();
        assert_eq!(j["error"]["code"], "slot_unavailable", "{j}");
        assert_eq!(j["error"]["reason"], "dropped_vram");
        assert_eq!(j["error"]["missing"][0], "m9");
    }

    #[tokio::test]
    async fn refused_activation_marks_untried_slots_with_the_refusal() {
        // 1 MiB free: the first slot cannot fit, so the whole profile is
        // refused and the second slot is never attempted.
        let fx = Fixture::build(&[("p", &["m1", "m2"])], MIB);
        assert!(fx.manager.clone().activate("p".into()).await.is_err());
        let first = fx.manager.unavailability("m1", true);
        assert_eq!(
            first.reason,
            crate::unavailable::UnavailableReason::DroppedVram
        );
        let second = fx.manager.unavailability("m2", true);
        assert_eq!(
            second.reason,
            crate::unavailable::UnavailableReason::DroppedVram
        );
        assert!(
            second
                .detail
                .as_deref()
                .unwrap_or("")
                .contains("not attempted"),
            "{second:?}"
        );
    }

    // -- JIT TTL sweeper ---------------------------------------------------------

    #[tokio::test]
    async fn ttl_sweeper_unloads_idle_jit_slots_only() {
        let fx = Fixture::build(&[("p", &["m1", "m2"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();
        fx.manager
            .deps
            .registry
            .write()
            .unwrap()
            .copy("m1", "m3")
            .unwrap();
        fx.manager.clone().jit_load("m3").await.unwrap();
        assert!(fx.engine.loaded().contains(&"m3".to_string()));

        // Backdate m3's last use past the TTL. Pinned m1/m2 have no such
        // concept — they survive regardless of their timestamps.
        let old = Instant::now() - Duration::from_secs(JIT_TTL_SECS + 60);
        fx.manager
            .last_used
            .lock()
            .unwrap()
            .insert("m3".into(), old);

        fx.manager.sweep_jit_slots().await;

        assert!(
            !fx.engine.loaded().contains(&"m3".to_string()),
            "{:?}",
            fx.engine.loaded()
        );
        assert!(fx.engine.loaded().contains(&"m1".to_string()));
        assert!(
            fx.engine.op_log().contains(&"unload:m3".to_string()),
            "{:?}",
            fx.engine.op_log()
        );
        assert!(fx.replay_contains("\"slot.unloaded\""));
    }

    #[tokio::test]
    async fn ttl_sweeper_spares_freshly_used_slots() {
        let fx = Fixture::build(&[("p", &["m1"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();
        fx.manager
            .deps
            .registry
            .write()
            .unwrap()
            .copy("m1", "m3")
            .unwrap();
        fx.manager.clone().jit_load("m3").await.unwrap();

        assert!(
            fx.engine.loaded().contains(&"m3".to_string()),
            "{:?}",
            fx.engine.loaded()
        );
    }

    // -- runtime OOM recovery ------------------------------------------------------

    #[tokio::test]
    async fn oom_recovery_evicts_lowest_priority_member_not_the_failure() {
        let fx = Fixture::build_fiddled(&["m1", "m2"], &[("p", &["m1", "m2"])], 8 * GIB, |cfg| {
            cfg.profile[0].slots[0].priority = Some(10); // lowest priority
            cfg.profile[0].slots[1].priority = Some(1); // highest priority
        });
        fx.manager.clone().activate("p".into()).await.unwrap();

        fx.manager.clone().on_runtime_oom("m2".into()).await;

        // m1 (lowest priority) evicted; the failing slot itself untouched.
        let ops = fx.engine.op_log();
        assert!(ops.contains(&"unload:m1".to_string()), "{ops:?}");
        assert!(!ops.contains(&"unload:m2".to_string()), "{ops:?}");
        assert!(fx.replay_contains("\"profile.degraded\""));
        assert!(fx
            .manager
            .inner
            .lock()
            .unwrap()
            .dropped
            .contains(&"m1".to_string()));
    }

    // -- placement = "auto" (docket M1) ----------------------------------------

    #[tokio::test]
    async fn auto_slot_demotes_to_cpu_instead_of_dropping() {
        // 1 MiB free: no GPU rung fits, so the auto slot lands on CPU.
        let fx = Fixture::build_fiddled(&["m1"], &[("p", &["m1"])], MIB, |cfg| {
            cfg.profile[0].slots[0].placement = Some(Placement::Auto);
        });
        fx.manager.clone().activate("p".into()).await.unwrap();

        let ops = fx.engine.op_log();
        assert!(
            ops.iter()
                .any(|o| o.starts_with("load:m1@") && o.contains("ngl=Some(0)")),
            "{ops:?}"
        );
        assert!(
            fx.manager.readiness(true).0,
            "a demoted profile still serves"
        );
        assert!(
            !fx.replay_contains("\"slot.failed\""),
            "the abandoned GPU attempt must not surface as a failure"
        );
        assert!(fx.replay_contains("\"demoted\":[\"m1\"]"));

        let status = fx.manager.status(true);
        let slot = &status["slots"][0];
        assert_eq!(slot["placement"], "cpu", "{status}");
        assert_eq!(slot["configured_placement"], "auto");
        assert_eq!(slot["demoted"], true);
        assert_eq!(status["degraded"]["demoted"][0], "m1");
    }

    #[tokio::test]
    async fn gpu_slot_still_drops_when_nothing_fits() {
        let fx = Fixture::build(&[("p", &["m1"])], MIB);
        assert!(fx.manager.clone().activate("p".into()).await.is_err());
        let ops = fx.engine.op_log();
        assert!(!ops.iter().any(|o| o.contains("ngl=Some(0)")), "{ops:?}");
    }

    #[tokio::test]
    async fn oom_recovery_demotes_an_auto_victim_to_cpu() {
        let fx = Fixture::build_fiddled(&["m1", "m2"], &[("p", &["m1", "m2"])], 8 * GIB, |cfg| {
            cfg.profile[0].slots[0].priority = Some(10); // lowest priority
            cfg.profile[0].slots[0].placement = Some(Placement::Auto);
            cfg.profile[0].slots[1].priority = Some(1);
        });
        fx.manager.clone().activate("p".into()).await.unwrap();

        fx.manager.clone().on_runtime_oom("m2".into()).await;

        let ops = fx.engine.op_log();
        let unload = ops
            .iter()
            .position(|o| o == "unload:m1")
            .expect("evicted from GPU");
        assert!(
            ops[unload..]
                .iter()
                .any(|o| o.starts_with("load:m1@") && o.contains("ngl=Some(0)")),
            "{ops:?}"
        );
        let inner = fx.manager.inner.lock().unwrap().clone();
        assert!(inner.demoted.contains(&"m1".to_string()));
        assert!(!inner.dropped.contains(&"m1".to_string()));
        assert!(matches!(inner.slots.get("m1"), Some(SlotState::Loaded(r)) if r.on_cpu));
    }

    #[tokio::test]
    async fn oom_recovery_never_evicts_a_cpu_resident() {
        // Evicting a CPU slot frees no VRAM; with only the failing GPU slot
        // and a CPU slot resident there is nothing worth evicting.
        let fx = Fixture::build_fiddled(&["m1", "m2"], &[("p", &["m1", "m2"])], 8 * GIB, |cfg| {
            cfg.profile[0].slots[0].priority = Some(10);
            cfg.profile[0].slots[0].placement = Some(Placement::Cpu);
            cfg.profile[0].slots[1].priority = Some(1);
        });
        fx.manager.clone().activate("p".into()).await.unwrap();

        fx.manager.clone().on_runtime_oom("m2".into()).await;

        let ops = fx.engine.op_log();
        assert!(!ops.iter().any(|o| o.starts_with("unload:")), "{ops:?}");
    }

    #[tokio::test]
    async fn oom_recovery_prefers_jit_residents_over_members() {
        let fx = Fixture::build(&[("p", &["m1"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();
        fx.manager
            .deps
            .registry
            .write()
            .unwrap()
            .copy("m1", "m2")
            .unwrap();
        fx.manager.clone().jit_load("m2").await.unwrap();

        fx.manager.clone().on_runtime_oom("m1".into()).await;

        // The JIT resident goes first; the profile member stays.
        let ops = fx.engine.op_log();
        assert!(ops.contains(&"unload:m2".to_string()), "{ops:?}");
        assert!(
            fx.engine.loaded().contains(&"m1".to_string()),
            "{:?}",
            fx.engine.loaded()
        );
        assert!(!fx.replay_contains("\"profile.degraded\""));
    }

    #[tokio::test]
    async fn note_runtime_failure_only_reacts_to_oom_errors() {
        let fx = Fixture::build(&[("p", &["m1", "m2"])], 8 * GIB);
        fx.manager.clone().activate("p".into()).await.unwrap();

        fx.manager
            .note_runtime_failure("m1", "some unrelated decode failure");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !fx.engine
                .op_log()
                .iter()
                .any(|op| op.starts_with("unload:")),
            "{:?}",
            fx.engine.op_log()
        );

        fx.manager
            .note_runtime_failure("m1", "CUDA error: cudaMalloc failed: out of memory");
        // A real OOM on m1 recovers by evicting the OTHER resident (the
        // failing slot itself is never chosen). Recovery runs as a spawned
        // task on this current-thread runtime: await (not a blocking sleep)
        // or it would never get polled.
        for _ in 0..500 {
            if fx
                .engine
                .op_log()
                .iter()
                .any(|op| op.starts_with("unload:m2"))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            fx.engine
                .op_log()
                .iter()
                .any(|op| op.starts_with("unload:m2")),
            "{:?}",
            fx.engine.op_log()
        );
    }
}
