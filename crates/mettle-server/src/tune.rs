//! `mettled tune` — hardware-matched config generation.
//!
//! Detects free VRAM (nvidia-smi), sizes each configured model against the
//! budget, and writes `config.toml` with per-model `n_gpu_layers`:
//! `999` = all layers on GPU, `0` = CPU, in between = partial offload from
//! the GGUF's layer count. Falls back to a CPU-only config when no NVIDIA
//! GPU is visible.

use std::path::PathBuf;

use mettle_core::config::RootConfig;
use mettle_core::model_registry::ModelRegistry;

/// Headroom reserved for the KV cache, activations, and the mmproj
/// projector on top of the summed model weights.
const VRAM_HEADROOM_GB: f32 = 1.0;
/// Approximate extra weight of an mmproj projector (vision models).
const MMPROJ_GB: f32 = 0.5;
/// Bytes per parameter by quantization family (used when the GGUF file is
/// not yet downloaded, so tuning works before `engine pull`).
fn bytes_per_param(quant: &str) -> f32 {
    let q = quant.to_ascii_uppercase();
    if q.contains("F16") || q.contains("FP16") {
        2.05
    } else if q.contains("Q8") {
        1.05
    } else if q.contains("Q4") {
        0.55
    } else if q.contains("Q5") {
        0.68
    } else if q.contains("Q6") {
        0.8
    } else {
        0.6
    }
}

/// Parse "2B" / "0.3B" / "7.6B" parameter sizes.
fn params_from_size(s: &str) -> Option<f32> {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix("B") {
        (n, 1e9)
    } else if let Some(n) = s.strip_suffix("M") {
        (n, 1e6)
    } else {
        (s, 1e9)
    };
    num.trim().parse::<f32>().ok().map(|v| v * mult)
}

/// Free VRAM in GB on the first visible NVIDIA GPU, if any.
fn detect_vram_gb() -> Option<f32> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let max_mib = text
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .fold(0.0f32, f32::max);
    (max_mib > 0.0).then_some(max_mib / 1024.0)
}

/// Size of one model in GB: the resolved GGUF's file size when present,
/// else an estimate from parameter count and quantization.
fn model_size_gb(alias: &str, registry: &ModelRegistry, is_vision: bool) -> f32 {
    let file_gb = registry
        .resolve(alias)
        .ok()
        .and_then(|e| {
            let p = if let Some(local) = &e.local_path {
                local.clone()
            } else {
                let filename = e.hf_quant_filename.as_ref()?;
                hf_hub::Cache::from_env()
                    .model(e.hf_repo_id.clone())
                    .get(filename)?
            };
            std::fs::metadata(&p).ok().map(|m| m.len() as f32 / 1e9)
        })
        .unwrap_or(0.0);
    if file_gb > 0.0 {
        return file_gb + if is_vision { MMPROJ_GB } else { 0.0 };
    }
    let est = registry
        .resolve(alias)
        .ok()
        .and_then(|e| {
            params_from_size(&e.metadata.parameter_size)
                .map(|p| p * bytes_per_param(&e.metadata.quantization) / 1e9)
        })
        .unwrap_or(1.5);
    est + if is_vision { MMPROJ_GB } else { 0.0 }
}

/// GPU layers for a model given the remaining budget.
fn layers_for(size_gb: f32, budget_gb: &mut f32, alias: &str, registry: &ModelRegistry) -> u32 {
    if *budget_gb >= size_gb {
        *budget_gb -= size_gb;
        return 999;
    }
    if *budget_gb >= size_gb * 0.25 {
        // Partial offload: layer count from the GGUF header when available.
        let n_layers = registry
            .resolve(alias)
            .ok()
            .and_then(|e| {
                let p = e.local_path.clone().or_else(|| {
                    let filename = e.hf_quant_filename.as_ref()?;
                    hf_hub::Cache::from_env()
                        .model(e.hf_repo_id.clone())
                        .get(filename)
                })?;
                mettle_llama::gguf_layer_count(&p)
            })
            .unwrap_or(50);
        let layers = ((*budget_gb / size_gb) * n_layers as f32).ceil() as u32;
        *budget_gb = 0.0;
        layers.clamp(1, n_layers)
    } else {
        0
    }
}

/// Write a VRAM-matched config and return a process exit code.
pub fn run(config: &RootConfig, output: Option<PathBuf>) -> i32 {
    let Some(vram_gb) = detect_vram_gb() else {
        eprintln!(
            "no NVIDIA GPU detected (nvidia-smi missing or no output); \
             writing a CPU-only config"
        );
        return write_config(config, output, 0.0);
    };
    eprintln!("detected {vram_gb:.1} GB free VRAM");
    write_config(config, output, vram_gb)
}

fn write_config(config: &RootConfig, output: Option<PathBuf>, vram_gb: f32) -> i32 {
    // Include the config's [[model]] entries so local `path` files are
    // sized from real file sizes, not param-count estimates.
    let mut registry = ModelRegistry::new();
    registry.apply_user_models(&config.model);
    let mut budget = (vram_gb - VRAM_HEADROOM_GB).max(0.0);

    let mut models: Vec<mettle_core::config::ModelEntry> = if config.model.is_empty() {
        // Adoption contract §1.4: the shipped default catalog is the
        // bench-proven retrieval set + gemma4 chat (JIT), matching
        // RootConfig::default().
        mettle_core::config::RootConfig::default().model
    } else {
        config.model.clone()
    };
    // The shipped-default profile rides along (below); its slot ids must
    // resolve in the emitted catalog or `RootConfig::validate` rejects the
    // generated config — so absent ids are appended here and sized like any
    // other model.
    let profile = mettle_core::config::RootConfig::default().profile.remove(0);
    for slot in &profile.slots {
        if !models.iter().any(|m| m.id == slot.id) {
            models.push(mettle_core::config::ModelEntry {
                id: slot.id.clone(),
                default: false,
                path: None,
                n_gpu_layers: None,
                mmproj_path: None,
                role: None,
                min_ctx: None,
                draft: None,
            });
        }
    }

    // Chat models first (they matter most), embeddings and OCR last.
    let mut lines = Vec::new();
    let mut ordered: Vec<_> = models.iter().collect();
    ordered.sort_by_key(|m| {
        let vision = registry.resolve(&m.id).is_ok_and(|e| {
            matches!(
                e.loader_kind,
                mettle_core::model_registry::LoaderKind::GgufVision { .. }
            )
        });
        (vision, m.id.starts_with("embed"))
    });
    for m in ordered {
        let is_vision = registry.resolve(&m.id).is_ok_and(|e| {
            matches!(
                e.loader_kind,
                mettle_core::model_registry::LoaderKind::GgufVision { .. }
            )
        });
        let size = model_size_gb(&m.id, &registry, is_vision);
        let layers = layers_for(size, &mut budget, &m.id, &registry);
        eprintln!(
            "  {}: ~{size:.1} GB -> n_gpu_layers = {}{}",
            m.id,
            layers,
            if layers == 999 { " (all)" } else { "" }
        );
        lines.push(format!(
            "[[model]]\nid = \"{}\"\ndefault = {}\n{}\n{}",
            m.id,
            m.default,
            m.path
                .as_ref()
                .map(|p| format!("path = \"{}\"", p.display()))
                .unwrap_or_default(),
            format!("n_gpu_layers = {layers}"),
        ));
    }

    // Engine shape per adoption contract §1.4: n_ctx 8192, fp8 KV, all
    // layers on GPU by default; speculative passthrough. The bench-retrieval
    // profile definition rides along (default = activated on first boot) so
    // the generated config ships the proven retrieval residency.
    let mut profile_lines = format!(
        "[[profile]]\nname = \"{}\"\ndefault = {}\n",
        profile.name, profile.default
    );
    for slot in &profile.slots {
        profile_lines.push_str(&format!(
            "\n[[profile.slots]]\nid = \"{}\"\npriority = {}\n",
            slot.id,
            slot.priority.unwrap_or(u32::MAX)
        ));
        // Placement is profile intent (`auto` = CPU fallback allowed);
        // regenerating the config must not silently drop it.
        if let Some(placement) = slot.placement {
            let value = match placement {
                mettle_core::config::Placement::Gpu => "gpu",
                mettle_core::config::Placement::Cpu => "cpu",
                mettle_core::config::Placement::Auto => "auto",
            };
            profile_lines.push_str(&format!("placement = \"{value}\"\n"));
        }
    }
    let config_toml = format!(
        "# Generated by `mettled tune` (llama.cpp backend; VRAM-matched)\n\
         [server]\n\
         host = \"127.0.0.1\"\n\
         port = {port}\n\
         \n\
         [engine]\n\
         max_concurrent_seqs = {max_seqs}\n\
         n_ctx = 8192\n\
         kv_cache = \"fp8\"\n\
         prefix_cache = true\n\
         speculative = \"{speculative}\"\n\
         n_gpu_layers = 999\n\
         \n\
         {models}\n\
         \n\
         {profile_lines}",
        port = config.server.port,
        max_seqs = config.engine.max_concurrent_seqs,
        models = lines.join("\n\n"),
        speculative = serde_json::to_string(&config.engine.speculative)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string(),
    );
    let path = output.unwrap_or_else(|| mettle_core::config::config_dir().join("config.toml"));
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, &config_toml) {
        Ok(()) => {
            eprintln!("wrote {}", path.display());
            0
        }
        Err(e) => {
            eprintln!("failed to write {}: {}", path.display(), e);
            1
        }
    }
}
