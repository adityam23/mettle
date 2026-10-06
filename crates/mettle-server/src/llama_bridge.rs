//! Bridge between wire types and the engine-agnostic `mettle_core::engine`
//! contract, plus construction of the llama.cpp-backed engine.
//!
//! Hook helpers (`token_info`, `fire_tokens_from_logprobs`,
//! `fire_on_response`, ...) and the shared response consumers live in
//! `mettle_core::stream`; embedding plumbing in `mettle_core::engine`.
//! This module keeps only the Ollama-flavoured conversions and engine
//! construction (GGUF resolution, config wiring, speculative drafting).

use std::collections::HashMap;
use std::path::PathBuf;

use mettle_core::config::{KVCacheSetting, RootConfig, SpeculativeSetting};
use mettle_core::engine::{GenConstraint, ModelKind, ModelLoadSpec, SamplingParams};
use mettle_core::model_registry::{LoaderKind, ModelRegistry, RegistryEntry};
use mettle_core::ollama_types::{FormatOption, TimingInfo};

use mettle_llama::{LlamaConfig, SpecConfig};

// ---------------------------------------------------------------------------
// Ollama options/format -> neutral request params
// ---------------------------------------------------------------------------

/// Map Ollama `options` to neutral [`SamplingParams`].
pub fn convert_options(options: &Option<HashMap<String, serde_json::Value>>) -> SamplingParams {
    let mut p = SamplingParams::default();
    let Some(opts) = options else { return p };

    if let Some(v) = opts.get("temperature").and_then(|v| v.as_f64()) {
        p.temperature = Some(v);
    }
    if let Some(v) = opts.get("top_k").and_then(|v| v.as_u64()) {
        p.top_k = Some(v as usize);
    }
    if let Some(v) = opts.get("top_p").and_then(|v| v.as_f64()) {
        p.top_p = Some(v);
    }
    if let Some(v) = opts.get("min_p").and_then(|v| v.as_f64()) {
        p.min_p = Some(v);
    }
    // Ollama uses `num_predict`; OpenAI-ish clients may send `max_tokens`.
    if let Some(v) = opts
        .get("num_predict")
        .or_else(|| opts.get("max_tokens"))
        .and_then(|v| v.as_u64())
    {
        p.max_tokens = Some(v as usize);
    }
    if let Some(v) = opts.get("frequency_penalty").and_then(|v| v.as_f64()) {
        p.frequency_penalty = Some(v as f32);
    }
    if let Some(v) = opts.get("presence_penalty").and_then(|v| v.as_f64()) {
        p.presence_penalty = Some(v as f32);
    }
    if let Some(v) = opts.get("repeat_penalty").and_then(|v| v.as_f64()) {
        p.repeat_penalty = Some(v as f32);
    }
    if let Some(v) = opts.get("stop") {
        p.stop = match v {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
    }
    p
}

/// Map an Ollama `format` to a neutral output constraint (enforced by the
/// engine's grammar/llguidance samplers).
pub fn convert_format(format: Option<&FormatOption>) -> GenConstraint {
    match format {
        None | Some(FormatOption::Schema(serde_json::Value::Null)) => GenConstraint::None,
        Some(FormatOption::Json) => GenConstraint::Json,
        Some(FormatOption::Schema(schema)) => GenConstraint::JsonSchema(schema.clone()),
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Convert engine [`UsageInfo`] to Ollama [`TimingInfo`] (all nanoseconds).
pub fn usage_to_timing(usage: &mettle_core::hooks::UsageInfo) -> TimingInfo {
    TimingInfo {
        total_duration: usage.prompt_eval_duration_ns + usage.eval_duration_ns,
        load_duration: 0,
        prompt_eval_count: usage.prompt_tokens,
        prompt_eval_duration: usage.prompt_eval_duration_ns,
        eval_count: usage.completion_tokens,
        eval_duration: usage.eval_duration_ns,
    }
}

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------

/// Build the llama runtime config from the server config, wiring
/// n_gpu_layers / n_ctx / n_seq_max / KV quant and resolving the
/// speculative-drafting plan (Phase 5/6).
pub fn build_llama_config(config: &RootConfig) -> LlamaConfig {
    let mut cfg = LlamaConfig {
        n_gpu_layers: config.engine.n_gpu_layers.unwrap_or(999),
        n_ctx: config.engine.n_ctx.unwrap_or(8192),
        n_batch: config.engine.n_batch.unwrap_or(512),
        n_seq_max: config.engine.max_concurrent_seqs,
        n_threads: std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4),
        default_max_tokens: 512,
        kv_cache_q8: matches!(config.engine.kv_cache, KVCacheSetting::Fp8),
        flash_attn: true,
        rerank_micro_tokens: config
            .engine
            .rerank_micro_tokens
            .unwrap_or(mettle_llama::DEFAULT_RERANK_MICRO_TOKENS),
        speculative: SpecConfig {
            gamma: 5,
            mtp: Vec::new(),
            draft: Vec::new(),
        },
    };

    let registry = {
        let mut r = ModelRegistry::new();
        r.apply_user_models(&config.model);
        r
    };

    for m in &config.model {
        let Some(entry) = registry.resolve(&m.id).ok() else {
            continue;
        };
        // MTP drafter: same-size gemma4 drafter (gemma4:<size>-mtp), used
        // when the drafter's GGUF is resolvable.
        if let Some(drafter) = resolve_mtp_drafter(&registry, entry) {
            if let Some(spec) = load_spec_for(&drafter) {
                cfg.speculative.mtp.push((entry.alias.clone(), spec));
                continue;
            }
        }
        // Standard draft model: gemma4:e2b drafts larger gemma4 targets. The
        // gemma4:e2b drafter's vocabulary (262144) is only sound for gemma4
        // targets — a non-gemma4 target (OCR, embeddings) has a different
        // vocabulary, so an out-of-vocab draft token crashes the target
        // decode ("invalid token"). MTP already covers gemma4 targets whose
        // `-mtp` drafter resolves, so this is a fallback for gemma4 targets
        // whose MTP drafter is missing.
        let target_is_gemma4 = entry.alias.starts_with("gemma4:");
        let drafting = match &config.engine.speculative {
            SpeculativeSetting::Off => false,
            SpeculativeSetting::Model(_) => target_is_gemma4 && entry.alias != "gemma4:e2b",
            SpeculativeSetting::Auto => target_is_gemma4 && entry.alias != "gemma4:e2b",
        };
        if drafting {
            if let Some(drafter) = registry.resolve("gemma4:e2b").ok() {
                if let Some(spec) = load_spec_for(drafter) {
                    cfg.speculative.draft.push((entry.alias.clone(), spec));
                }
            }
        }
    }
    cfg
}

/// Same-size MTP drafter for a gemma4 target (`gemma4:e2b` →
/// `gemma4:e2b-mtp`), if the registry knows one. Also consulted by the
/// profile manager when wiring a chat slot without an explicit catalog
/// `draft` (preserves boot-time MTP behavior under profiles).
pub fn resolve_mtp_drafter(
    registry: &ModelRegistry,
    entry: &RegistryEntry,
) -> Option<RegistryEntry> {
    let size = entry
        .alias
        .strip_prefix("gemma4:")?
        .split(':')
        .next()
        .unwrap_or_default();
    let drafter_alias = format!("gemma4:{size}-mtp");
    registry.resolve(&drafter_alias).ok().cloned()
}

/// Resolve a registry entry to a local GGUF path: config-local path first,
/// then the HF cache.
pub fn resolve_gguf_path(entry: &RegistryEntry) -> Result<PathBuf, String> {
    if let Some(p) = &entry.local_path {
        if p.exists() {
            return Ok(p.clone());
        }
        return Err(format!(
            "model '{}' configured local path '{}' does not exist",
            entry.alias,
            p.display()
        ));
    }
    let cache = hf_hub::Cache::from_env();
    if let Some(filename) = &entry.hf_quant_filename {
        if let Some(path) = cache.model(entry.hf_repo_id.clone()).get(filename) {
            return Ok(path);
        }
        // A named file must resolve to THAT file. Falling through to
        // "whatever GGUF this repo has" would silently substitute a
        // different model: the gemma4 MTP drafter names
        // `mtp-gemma-4-*` in the target's own repo, which no normal
        // pull or first-run download ever fetches, so the substituted
        // sibling is the target itself - wired as its own drafter.
        return Err(format!(
            "GGUF '{}' for '{}' not in cache; pull it first",
            filename, entry.alias
        ));
    }
    // Entries that name no quant file (user-defined GGUF entries, repos with
    // a single quant) take whatever GGUF this repo already has in the cache.
    if let Some(path) = crate::weights::cached_gguf(entry) {
        return Ok(path);
    }
    Err(match &entry.hf_quant_filename {
        Some(filename) => format!(
            "GGUF '{}' for '{}' not in cache; pull it first",
            filename, entry.alias
        ),
        None => format!(
            "no GGUF for '{}' in the HF cache; pull it first",
            entry.alias
        ),
    })
}

/// Resolve a vision entry's mmproj file: config-local path first, then the
/// HF cache (the cache may be cleared once models live in the repo).
pub fn resolve_mmproj_path(entry: &RegistryEntry) -> Option<PathBuf> {
    let LoaderKind::GgufVision {
        mmproj_filename, ..
    } = &entry.loader_kind
    else {
        return None;
    };
    if let Some(p) = &entry.mmproj_path {
        if p.exists() {
            return Some(p.clone());
        }
        tracing::warn!(
            "model '{}' configured mmproj path '{}' does not exist",
            entry.alias,
            p.display()
        );
    }
    hf_hub::Cache::from_env()
        .model(entry.hf_repo_id.clone())
        .get(mmproj_filename)
}

/// Map a registry entry to a load spec; hard error when the GGUF is not
/// resolvable (route-side loads must fail loudly).
pub fn load_spec_for_route(entry: &RegistryEntry) -> Result<ModelLoadSpec, String> {
    let gguf_path = resolve_gguf_path(entry)?;
    let kind = match &entry.loader_kind {
        LoaderKind::GGUF { .. } | LoaderKind::GgufVision { .. } => ModelKind::Generation,
        LoaderKind::Embedding { pooling, .. } => ModelKind::Embedding(*pooling),
        LoaderKind::Rerank { .. } => ModelKind::Rerank,
    };
    Ok(ModelLoadSpec {
        alias: entry.alias.clone(),
        gguf_path,
        kind,
        n_gpu_layers: entry.n_gpu_layers,
        mmproj_path: resolve_mmproj_path(entry),
    })
}

/// Map a registry entry to a load spec (None if it can't be resolved yet).
fn load_spec_for(entry: &RegistryEntry) -> Option<ModelLoadSpec> {
    match load_spec_for_route(entry) {
        Ok(spec) => Some(spec),
        Err(e) => {
            tracing::debug!("skipping model '{}': {}", entry.alias, e);
            None
        }
    }
}
