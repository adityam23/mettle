use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Base URL of a locally running engine with default host/port — the
/// default target for every CLI command.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

/// Top-level engine configuration, deserialized from TOML.
/// Zero-config first run: `mettled` starts fine with no config file present.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootConfig {
    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub engine: EngineOptions,

    #[serde(default = "default_models")]
    pub model: Vec<ModelEntry>,

    /// Named model collections (agent-docs/profiles-contract.md). Empty =
    /// no profiles defined; the engine then loads nothing until one is
    /// activated. Validated by [`RootConfig::validate`].
    #[serde(default)]
    pub profile: Vec<ProfileConfig>,

    #[serde(default)]
    pub retrieval: RetrievalConfig,
}

impl Default for RootConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineOptions::default(),
            model: default_models(),
            profile: vec![default_profile()],
            retrieval: RetrievalConfig::default(),
        }
    }
}

impl RootConfig {
    /// Cross-entry validation, run on every TOML load (hard errors, never
    /// silent skips — profiles-contract.md):
    /// - profile names unique and non-empty
    /// - every profile has slots; slot ids resolve to `[[model]]` catalog
    ///   entries; a slot id appears at most once per profile
    /// - `min_ctx` > 0
    /// - `draft` targets resolve to catalog entries
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        let catalog: std::collections::HashSet<&str> =
            self.model.iter().map(|m| m.id.as_str()).collect();
        let mut seen_names = std::collections::HashSet::new();
        for p in &self.profile {
            if p.name.is_empty() {
                return Err(crate::error::CoreError::Config(
                    "profile name must not be empty".into(),
                ));
            }
            if !seen_names.insert(p.name.as_str()) {
                return Err(crate::error::CoreError::Config(format!(
                    "duplicate profile name '{}'",
                    p.name
                )));
            }
            if p.slots.is_empty() {
                return Err(crate::error::CoreError::Config(format!(
                    "profile '{}' has no slots",
                    p.name
                )));
            }
            let mut seen_slots = std::collections::HashSet::new();
            for s in &p.slots {
                if !catalog.contains(s.id.as_str()) {
                    return Err(crate::error::CoreError::Config(format!(
                        "profile '{}' references unknown model id '{}'",
                        p.name, s.id
                    )));
                }
                if !seen_slots.insert(s.id.as_str()) {
                    return Err(crate::error::CoreError::Config(format!(
                        "profile '{}' lists model '{}' twice",
                        p.name, s.id
                    )));
                }
            }
        }
        for m in &self.model {
            if let Some(ctx) = m.min_ctx {
                if ctx == 0 {
                    return Err(crate::error::CoreError::Config(format!(
                        "model '{}': min_ctx must be > 0",
                        m.id
                    )));
                }
            }
            if let Some(d) = &m.draft {
                if !catalog.contains(d.as_str()) {
                    return Err(crate::error::CoreError::Config(format!(
                        "model '{}': draft target '{}' is not a catalog entry",
                        m.id, d
                    )));
                }
            }
        }
        Ok(())
    }

    /// Load from a TOML file path. Errors if the file is malformed or
    /// fails cross-entry validation.
    pub fn load_from(path: &std::path::Path) -> Result<Self, crate::error::CoreError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            crate::error::CoreError::Config(format!("failed to read config file: {}", e))
        })?;
        let cfg: RootConfig = toml::from_str(&contents).map_err(|e| {
            crate::error::CoreError::Config(format!("failed to parse config: {}", e))
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Build from an explicit path, or return defaults if the file does not exist.
    pub fn load_or_default(path: &std::path::Path) -> Result<Self, crate::error::CoreError> {
        if path.exists() {
            Self::load_from(path)
        } else {
            Ok(Self::default())
        }
    }
}

// ---------------------------------------------------------------------------
// Roles, placement, profiles
// ---------------------------------------------------------------------------

/// The role a catalog model plays in a profile. Closed enum by design
/// (profiles-contract.md): new roles arrive as code changes, and an unknown
/// role string is a parse error, never a silently-ignored slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelRole {
    Chat,
    Embed,
    Rerank,
    Ocr,
}

/// Where a slot's weights live. `Gpu` is the default policy
/// (GPU-or-refuse); `Cpu` is explicit config intent only — degradation
/// never falls back to CPU on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Placement {
    Gpu,
    Cpu,
}

/// One model inside a `[[profile]]`. `priority` orders degradation and
/// load order within the profile; `None` keeps declaration order
/// (earlier = higher priority).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSlot {
    /// Catalog (`[[model]]`) id. Existence is validated by
    /// [`RootConfig::validate`].
    pub id: String,
    #[serde(default)]
    pub priority: Option<u32>,
    #[serde(default)]
    pub placement: Option<Placement>,
}

/// A named, ordered model collection. One profile is active at a time;
/// activation state lives outside this file (engine state, not config).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfig {
    pub name: String,
    /// Shipped-default marker: activated on first boot when no active
    /// profile was ever persisted (adoption-contract grilling Q11,
    /// 2026-09-10 — the one deliberate amendment to profiles-contract's
    /// "no implicit default profile").
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub slots: Vec<ProfileSlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    /// Registry alias (`gemma4:e2b`). Unknown aliases resolve to a
    /// user-defined entry with this id as the HF repo id.
    pub id: String,

    #[serde(default)]
    pub default: bool,

    /// Local GGUF path; takes precedence over the HF-cache resolution for
    /// the registry's built-in repos. Lets self-hosted installs point at
    /// files they already have without downloading.
    #[serde(default)]
    pub path: Option<PathBuf>,

    /// Per-model GPU offload override (layers); defaults to `[engine]
    /// n_gpu_layers`. Set 0 to force CPU (e.g. a large OCR model on a
    /// small card).
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,

    /// Local multimodal-projector (mmproj) GGUF for vision models; takes
    /// precedence over the HF-cache resolution. Required once the HF cache
    /// is cleared (models moved into the repo).
    #[serde(default)]
    pub mmproj_path: Option<PathBuf>,

    /// Profile role override. `None` infers from the loader kind
    /// (GGUF→chat, vision→ocr, Embedding→embed, Rerank→rerank) at
    /// registry level; an explicit value wins and an unknown string is a
    /// parse error (closed enum — profiles-contract.md).
    #[serde(default)]
    pub role: Option<ModelRole>,

    /// Lowest context this slot may be shrunk to under VRAM pressure
    /// (degradation ladder step 2). `None` = never shrunk.
    #[serde(default)]
    pub min_ctx: Option<u32>,

    /// Speculative drafter attached to this (chat) model; a catalog id
    /// that loads and unloads atomically with this entry.
    #[serde(default)]
    pub draft: Option<String>,
}

fn default_models() -> Vec<ModelEntry> {
    // Adoption contract §1.4 (2026-09-10): the shipped default catalog is
    // the bench-proven qwen3 retrieval set plus the gemma4 chat model (chat
    // stays JIT — the profile pins only retrieval models). `path: None`
    // entries clone the registry builtins (HF repos + quant filenames), so
    // fresh installs resolve via the HF cache and this repo's `models/`
    // checkouts resolve via config paths when a user pins them.
    fn entry(id: &str, default: bool) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            default,
            path: None,
            n_gpu_layers: None,
            mmproj_path: None,
            role: None,
            min_ctx: None,
            draft: None,
        }
    }
    vec![
        entry("gemma4:e2b", true),
        entry("qwen3-embedding-0.6b:latest", false),
        entry("qwen3-reranker-0.6b:latest", false),
        entry("embed-gemma:latest", false),
        entry("bge-m3:latest", false),
        entry("bge-reranker-v2-m3:latest", false),
    ]
}

/// The shipped-default profile: the bench-retrieval shape (n_ctx 8192,
/// KV fp8, `n_gpu_layers 999`, all retrieval models GPU-resident; chat
/// models JIT). Priorities order the degradation ladder: the qwen3 defaults
/// are kept longest, the secondary embedders/reranker evict first.
fn default_profile() -> ProfileConfig {
    let slot = |id: &str, priority: u32| ProfileSlot {
        id: id.into(),
        priority: Some(priority),
        placement: None,
    };
    ProfileConfig {
        name: "bench-retrieval".into(),
        default: true,
        slots: vec![
            slot("qwen3-embedding-0.6b:latest", 10),
            slot("qwen3-reranker-0.6b:latest", 20),
            slot("embed-gemma:latest", 30),
            slot("bge-m3:latest", 40),
            slot("bge-reranker-v2-m3:latest", 50),
        ],
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    pub port: u16,

    #[serde(default)]
    pub api_keys: Vec<String>,

    #[serde(default)]
    pub offline: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_keys: Vec::new(),
            offline: false,
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".into()
}

fn default_port() -> u16 {
    11434
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineOptions {
    #[serde(default = "default_max_concurrent_seqs")]
    pub max_concurrent_seqs: usize,

    #[serde(default)]
    pub kv_cache: KVCacheSetting,

    #[serde(default = "default_true")]
    pub prefix_cache: bool,

    #[serde(default)]
    pub speculative: SpeculativeSetting,

    /// Engine-wide GPU offload (layers, 0 = CPU, 999 = all); written by
    /// `mettled tune` from VRAM detection. Per-model override via
    /// `[[model]] n_gpu_layers`.
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,

    /// Batch size for decode/prefill (default 512). Lower values shrink
    /// each context's compute-buffer reservation; on small GPUs this is the
    /// knob that lets an MTP lane + OCR + embedding model coexist.
    #[serde(default)]
    pub n_batch: Option<u32>,

    /// Generation context length (default 8192). KV scales at ~35 KB/token
    /// for gemma4 q8_0; 64K measures ~4.8 GB total on the 6 GB reference GPU.
    #[serde(default)]
    pub n_ctx: Option<u32>,

    /// Safety margin subtracted from the device free-memory estimate before
    /// the profile planner admits a slot (profiles-contract.md "Budget").
    /// Absorbs CUDA workspace/cuda-graph overhead the GGUF-header estimate
    /// does not model, plus other processes' usage. Default: 512 MiB.
    #[serde(default)]
    pub vram_margin_bytes: Option<u64>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            max_concurrent_seqs: default_max_concurrent_seqs(),
            kv_cache: KVCacheSetting::default(),
            prefix_cache: true,
            speculative: SpeculativeSetting::default(),
            n_gpu_layers: None,
            n_batch: None,
            n_ctx: None,
            vram_margin_bytes: None,
        }
    }
}

fn default_max_concurrent_seqs() -> usize {
    16
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KVCacheSetting {
    Fp16,
    #[default]
    Fp8,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeculativeSetting {
    #[default]
    Auto,
    Off,
    Model(String),
}

impl<'de> Deserialize<'de> for SpeculativeSetting {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "auto" => Ok(SpeculativeSetting::Auto),
            "off" => Ok(SpeculativeSetting::Off),
            other => Ok(SpeculativeSetting::Model(other.to_owned())),
        }
    }
}

// ---------------------------------------------------------------------------
// Retrieval / vector store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    #[serde(default = "default_bit_width")]
    pub default_bit_width: usize,

    #[serde(default = "default_chunk_tokens")]
    pub chunk_tokens: usize,

    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            default_bit_width: default_bit_width(),
            chunk_tokens: default_chunk_tokens(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

fn default_data_dir() -> PathBuf {
    data_dir()
}

/// Mutable state root (`state.json`, retrieval stores): `$XDG_DATA_HOME`
/// or `~/.local/share`, namespaced under `mettle/`.
pub fn data_dir() -> PathBuf {
    xdg_base("XDG_DATA_HOME", ".local/share").join("mettle")
}

/// Configuration root (`config.toml`): `$XDG_CONFIG_HOME` or `~/.config`,
/// namespaced under `mettle/`.
pub fn config_dir() -> PathBuf {
    xdg_base("XDG_CONFIG_HOME", ".config").join("mettle")
}

fn xdg_base(var: &str, fallback_rel: &str) -> PathBuf {
    if let Ok(dir) = std::env::var(var) {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(fallback_rel)
}

fn default_bit_width() -> usize {
    4
}

fn default_chunk_tokens() -> usize {
    512
}

fn default_chunk_overlap() -> usize {
    64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrips() {
        let cfg = RootConfig::default();
        assert_eq!(cfg.server.port, 11434);
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert!(cfg.engine.prefix_cache);
        assert!(matches!(cfg.engine.kv_cache, KVCacheSetting::Fp8));
        // gemma4 chat + the qwen3 retrieval set (adoption contract §1.4).
        assert_eq!(cfg.model.len(), 6);
        assert_eq!(cfg.model[0].id, "gemma4:e2b");
        assert!(cfg.model[0].default);
        assert!(cfg.model[0].path.is_none());
        assert!(cfg
            .model
            .iter()
            .any(|m| m.id == "qwen3-embedding-0.6b:latest"));
        // The shipped default profile: bench-retrieval, activation-default,
        // qwen3 slots kept longest (lowest priority number wins role alias).
        let profile = &cfg.profile[0];
        assert_eq!(profile.name, "bench-retrieval");
        assert!(profile.default);
        assert_eq!(profile.slots[0].id, "qwen3-embedding-0.6b:latest");
        assert_eq!(profile.slots[1].id, "qwen3-reranker-0.6b:latest");
    }

    #[test]
    fn parse_minimal_config() {
        let toml = r#"
[server]
host = "0.0.0.0"
port = 9999
"#;
        let cfg: RootConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 9999);
        // rest are defaults
        assert_eq!(cfg.engine.max_concurrent_seqs, 16);
    }

    #[test]
    fn parse_full_config() {
        let toml = r#"
[server]
host = "127.0.0.1"
port = 11434
api_keys = ["sk-test"]
offline = true

[engine]
max_concurrent_seqs = 8
kv_cache = "fp16"
prefix_cache = false
speculative = "off"

[[model]]
id = "gemma4:e2b"

[[model]]
id = "gemma4:e4b"
default = true
path = "/models/gemma-4-E4B-it-Q4_K_M.gguf"
n_gpu_layers = 20

[retrieval]
data_dir = "/tmp/engine-test"
chunk_tokens = 256
chunk_overlap = 32
"#;
        let cfg: RootConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.server.api_keys, vec!["sk-test"]);
        assert!(cfg.server.offline);
        assert!(matches!(cfg.engine.kv_cache, KVCacheSetting::Fp16));
        assert!(!cfg.engine.prefix_cache);
        assert!(matches!(cfg.engine.speculative, SpeculativeSetting::Off));
        assert_eq!(cfg.model.len(), 2);
        assert_eq!(
            cfg.model[1].path,
            Some(PathBuf::from("/models/gemma-4-E4B-it-Q4_K_M.gguf"))
        );
        assert_eq!(cfg.model[1].n_gpu_layers, Some(20));
        assert_eq!(cfg.retrieval.chunk_tokens, 256);
        assert_eq!(cfg.retrieval.data_dir, PathBuf::from("/tmp/engine-test"));
    }

    #[test]
    fn speculative_setting_parsing() {
        let toml = r#"
[engine]
speculative = "gemma4:e2b"
"#;
        let cfg: RootConfig = toml::from_str(toml).unwrap();
        assert!(matches!(
            cfg.engine.speculative,
            SpeculativeSetting::Model(ref m) if m == "gemma4:e2b"
        ));
    }

    #[test]
    fn parse_profiles_and_catalog_roles() {
        let toml = r#"
[[model]]
id = "qwen35-4b-opus-distill"
path = "/models/qwen.Q4_K_M.gguf"
role = "chat"
min_ctx = 2048
draft = "qwen35-mtp"

[[model]]
id = "qwen35-mtp"
path = "/models/qwen-mtp.gguf"

[[model]]
id = "bge-reranker-v2-m3"
path = "/models/bge-reranker.gguf"
role = "rerank"

[[profile]]
name = "finance"
[[profile.slots]]
id = "qwen35-4b-opus-distill"
priority = 1
[[profile.slots]]
id = "bge-reranker-v2-m3"
placement = "cpu"

[[profile]]
name = "retrieval-bench"
[[profile.slots]]
id = "bge-reranker-v2-m3"
"#;
        let cfg: RootConfig = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model[0].role, Some(ModelRole::Chat));
        assert_eq!(cfg.model[0].min_ctx, Some(2048));
        assert_eq!(cfg.model[0].draft.as_deref(), Some("qwen35-mtp"));
        assert_eq!(cfg.model[1].role, None, "role inferred at registry level");
        assert_eq!(cfg.profile.len(), 2);
        assert_eq!(cfg.profile[0].name, "finance");
        assert_eq!(cfg.profile[0].slots.len(), 2);
        assert_eq!(cfg.profile[0].slots[0].priority, Some(1));
        assert_eq!(cfg.profile[0].slots[0].placement, None);
        assert_eq!(cfg.profile[0].slots[1].placement, Some(Placement::Cpu));
        assert_eq!(cfg.profile[1].slots[0].priority, None);
    }

    #[test]
    fn unknown_role_and_unknown_slot_field_rejected() {
        // Closed role enum: an unknown role is a parse error, never silent.
        let toml = r#"
[[model]]
id = "x"
role = "sae"
"#;
        assert!(toml::from_str::<RootConfig>(toml).is_err());

        // deny_unknown_fields on profile tables.
        let toml = r#"
[[model]]
id = "x"

[[profile]]
name = "p"
ttl_secs = 60
[[profile.slots]]
id = "x"
"#;
        assert!(toml::from_str::<RootConfig>(toml).is_err());
    }

    #[test]
    fn validate_rejects_broken_profiles() {
        let base = r#"
[[model]]
id = "chat-a"

[[model]]
id = "chat-b"
"#;
        let mk = |profile: &str| format!("{base}\n{profile}");

        // Unknown slot id.
        let cfg: RootConfig = toml::from_str(&mk(
            "[[profile]]\nname = \"p\"\n[[profile.slots]]\nid = \"nope\"\n",
        ))
        .unwrap();
        assert!(cfg.validate().is_err());

        // Duplicate profile name.
        let dup = "[[profile]]\nname = \"p\"\n[[profile.slots]]\nid = \"chat-a\"\n";
        let cfg: RootConfig = toml::from_str(&mk(&format!("{dup}{dup}"))).unwrap();
        assert!(cfg.validate().is_err());

        // Empty slots.
        let cfg: RootConfig = toml::from_str(&mk("[[profile]]\nname = \"p\"\n")).unwrap();
        assert!(cfg.validate().is_err());

        // Same model twice in one profile.
        let twice = "[[profile]]\nname = \"p\"\n[[profile.slots]]\nid = \"chat-a\"\n[[profile.slots]]\nid = \"chat-a\"\n";
        let cfg: RootConfig = toml::from_str(&mk(twice)).unwrap();
        assert!(cfg.validate().is_err());

        // Draft target must be a catalog entry.
        let draft = "[[model]]\nid = \"chat-c\"\ndraft = \"missing-draft\"\n";
        let cfg: RootConfig = toml::from_str(&mk(draft)).unwrap();
        assert!(cfg.validate().is_err());

        // min_ctx must be positive.
        let zero = "[[model]]\nid = \"chat-d\"\nmin_ctx = 0\n";
        let cfg: RootConfig = toml::from_str(&mk(zero)).unwrap();
        assert!(cfg.validate().is_err());
    }
}
