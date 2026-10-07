use std::collections::HashMap;

use crate::error::CoreError;

/// Describes how a model should be loaded.
///
/// The llama.cpp world is GGUF-only: text models, vision (mmproj) models,
/// and embedding models. The safetensors-era kinds (`Normal`, `Multimodal`)
/// are gone with mistralrs.
/// How per-sequence outputs are pooled for non-generative models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Average over all tokens (embeddinggemma).
    Mean,
    /// First token (bge-m3 dense embeddings).
    Cls,
    /// Last token (Qwen3 embedding models).
    Last,
    /// Classification head attached to the graph — the only pooling that
    /// yields a rerank score (`embeddings_seq_ith` returns n_cls_out floats).
    Rank,
}

#[derive(Debug, Clone)]
pub enum LoaderKind {
    /// GGUF quantised checkpoint — single file with embedded quant format.
    GGUF { quant_filename: String },
    /// Vision-language model: a text GGUF plus a multimodal projector
    /// (`mmproj`) file for image input (e.g. Unlimited-OCR).
    GgufVision {
        quant_filename: String,
        mmproj_filename: String,
    },
    /// Embedding-only model; pooled with [`Pooling::Mean`]/[`Pooling::Cls`]/
    /// [`Pooling::Last`] per the model's training recipe. `quant_filename`
    /// names the GGUF inside `hf_repo_id` (`None` = config-local path only).
    Embedding {
        pooling: Pooling,
        quant_filename: Option<String>,
    },
    /// Cross-encoder reranker — scored with [`Pooling::Rank`]. Same
    /// `quant_filename` semantics as [`LoaderKind::Embedding`].
    Rerank { quant_filename: Option<String> },
}

/// Rich metadata for a model, surfaced in /api/show and /api/tags.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization: String,
    pub format: String,
}

/// One entry in the model registry, mapping an Ollama-style alias to
/// a Hugging Face repo and loader configuration.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub alias: String,
    pub hf_repo_id: String,
    pub hf_quant_filename: Option<String>,
    pub loader_kind: LoaderKind,
    pub revision: Option<String>,
    pub is_default: bool,
    pub metadata: ModelMetadata,
    /// True when this entry came from the user config rather than built-ins.
    pub user_defined: bool,
    /// Local GGUF path from `[[model]] path`; takes precedence over the
    /// HF-cache resolution.
    pub local_path: Option<std::path::PathBuf>,
    /// Per-model GPU offload override from `[[model]] n_gpu_layers`.
    pub n_gpu_layers: Option<u32>,
    /// Local mmproj path from `[[model]] mmproj_path`; takes precedence
    /// over the HF-cache resolution for vision models.
    pub mmproj_path: Option<std::path::PathBuf>,
    /// Embedding dimension for `Embedding`-kinds — the real vector length
    /// this model produces. `None` for non-embedding models or unknown
    /// (read the GGUF header's `{arch}.embedding_length` at collection
    /// creation instead). Kept per-model so a collection can be sized to
    /// its embedder (bge-m3=1024, qwen3-embedding=1024, embed-gemma=768),
    /// not a hardcoded global.
    pub embedding_dim: Option<usize>,
}

/// Where an entry's weight file comes from.
///
/// The single statement of the "local path / named HF file / any cached GGUF"
/// rule: path resolution, first-run acquisition and `/api/pull` all match on
/// this instead of re-deriving it from `local_path` / `hf_quant_filename`.
/// Derived on demand from the entry's fields, so it can never disagree with
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSource<'a> {
    /// `[[model]] path`: operator-provided. Never downloaded — a missing
    /// file is a config error.
    Local(&'a std::path::Path),
    /// A named GGUF in a Hugging Face repo: resolved from the HF cache and
    /// downloadable. A named file resolves to THAT file only.
    HfFile { repo: &'a str, file: &'a str },
    /// No named file: whatever GGUF this repo already has in the HF cache.
    /// Not downloadable. A `[[model]]` id that matches no built-in alias
    /// lands here with the id itself as the repo id (a bare HF repo id such
    /// as `org/model-GGUF` resolves from the cache this way).
    HfAnyGguf { repo: &'a str },
}

impl WeightSource<'_> {
    /// Can this source be fetched from Hugging Face?
    pub fn is_downloadable(&self) -> bool {
        matches!(self, WeightSource::HfFile { .. })
    }
}

impl RegistryEntry {
    /// Where this entry's weight file comes from (see [`WeightSource`]).
    pub fn weight_source(&self) -> WeightSource<'_> {
        match (&self.local_path, &self.hf_quant_filename) {
            (Some(path), _) => WeightSource::Local(path),
            (None, Some(file)) => WeightSource::HfFile {
                repo: &self.hf_repo_id,
                file,
            },
            (None, None) => WeightSource::HfAnyGguf {
                repo: &self.hf_repo_id,
            },
        }
    }

    /// The vision projector's filename in `hf_repo_id`, for
    /// [`LoaderKind::GgufVision`] entries.
    pub fn mmproj_filename(&self) -> Option<&str> {
        match &self.loader_kind {
            LoaderKind::GgufVision {
                mmproj_filename, ..
            } => Some(mmproj_filename),
            _ => None,
        }
    }

    /// True for embedding-only models (the only kind a collection can use).
    pub fn is_embedding(&self) -> bool {
        matches!(self.loader_kind, LoaderKind::Embedding { .. })
    }

    /// The registry-declared vector dimension of an embedding model: the
    /// fallback for sizing a collection when the GGUF header (the
    /// authoritative source) is not readable. `None` for non-embeddings and
    /// for embedders without a declared dim.
    pub fn declared_embedding_dim(&self) -> Option<usize> {
        if self.is_embedding() {
            self.embedding_dim
        } else {
            None
        }
    }
}

/// The model registry holds built-in aliases (Gemma 4 family) and
/// user-supplied overrides. It resolves Ollama-style `model:tag` names
/// to Hugging Face repos.
#[derive(Clone)]
pub struct ModelRegistry {
    entries: HashMap<String, RegistryEntry>,
    default_id: String,
    default_embedding_id: String,
    default_rerank_id: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a built-in registry entry.
fn builtin(
    alias: &str,
    hf_repo_id: &str,
    loader_kind: LoaderKind,
    is_default: bool,
    family: &str,
    parameter_size: &str,
    quantization: &str,
) -> RegistryEntry {
    let (format, hf_quant_filename) = match &loader_kind {
        LoaderKind::GGUF { quant_filename } => ("gguf", Some(quant_filename.clone())),
        LoaderKind::GgufVision { quant_filename, .. } => {
            ("gguf-vision", Some(quant_filename.clone()))
        }
        LoaderKind::Embedding { quant_filename, .. } => ("gguf", quant_filename.clone()),
        LoaderKind::Rerank { quant_filename } => ("gguf", quant_filename.clone()),
    };
    RegistryEntry {
        alias: alias.into(),
        hf_repo_id: hf_repo_id.into(),
        hf_quant_filename,
        loader_kind,
        revision: None,
        is_default,
        metadata: ModelMetadata {
            family: family.into(),
            families: vec![family.into()],
            parameter_size: parameter_size.into(),
            quantization: quantization.into(),
            format: format.into(),
        },
        user_defined: false,
        local_path: None,
        n_gpu_layers: None,
        mmproj_path: None,
        embedding_dim: None,
    }
}

impl ModelRegistry {
    /// Build the registry pre-populated with the GGUF-backed built-in
    /// aliases. Project scope: Gemma 4 E2B/E4B, embeddinggemma, Unlimited-OCR.
    pub fn new() -> Self {
        use LoaderKind::*;

        let e2b = builtin(
            "gemma4:e2b",
            "ggml-org/gemma-4-E2B-it-GGUF",
            GGUF {
                quant_filename: "gemma-4-E2B-it-Q4_0.gguf".into(),
            },
            true,
            "gemma",
            "2B",
            "Q4_0",
        );

        let builtins = [
            // "gemma4" and "gemma4:latest" are lookup keys for the SAME
            // canonical entry (alias stays gemma4:e2b — the engine registers
            // models by alias, so lookups must resolve to it).
            ("gemma4:e2b".to_string(), e2b.clone()),
            ("gemma4:latest".to_string(), e2b.clone()),
            ("gemma4".to_string(), e2b),
            (
                "gemma4:e4b".to_string(),
                builtin(
                    "gemma4:e4b",
                    "ggml-org/gemma-4-E4B-it-GGUF",
                    GGUF {
                        quant_filename: "gemma-4-E4B-it-Q4_0.gguf".into(),
                    },
                    false,
                    "gemma",
                    "4B",
                    "Q4_0",
                ),
            ),
            // Default embedding model: embeddinggemma-300m (768-dim).
            (
                "embed-gemma:latest".to_string(),
                builtin(
                    "embed-gemma:latest",
                    "ggml-org/embeddinggemma-300M-GGUF",
                    Embedding {
                        pooling: Pooling::Mean,
                        quant_filename: Some("embeddinggemma-300M-Q8_0.gguf".into()),
                    },
                    false,
                    "gemma",
                    "300M",
                    "Q8_0",
                ),
            ),
            // OCR: Baidu Unlimited-OCR (DeepSeek-OCR arch) via GGUF + mmproj.
            (
                "unlimited-ocr:latest".to_string(),
                builtin(
                    "unlimited-ocr:latest",
                    "sahilchachra/Unlimited-OCR-GGUF",
                    GgufVision {
                        quant_filename: "Unlimited-OCR-Q4_K_M.gguf".into(),
                        mmproj_filename: "mmproj-Unlimited-OCR-F16.gguf".into(),
                    },
                    false,
                    "unlimited-ocr",
                    "3B",
                    "Q4_K_M",
                ),
            ),
            // MTP drafters (Phase 6): same-model drafter GGUFs paired with
            // their gemma4 target via the engine's MTP lane.
            (
                "gemma4:e2b-mtp".to_string(),
                builtin(
                    "gemma4:e2b-mtp",
                    "ggml-org/gemma-4-E2B-it-GGUF",
                    GGUF {
                        quant_filename: "mtp-gemma-4-E2B-it-Q4_0.gguf".into(),
                    },
                    false,
                    "gemma",
                    "MTP",
                    "Q4_0",
                ),
            ),
            (
                "gemma4:e4b-mtp".to_string(),
                builtin(
                    "gemma4:e4b-mtp",
                    "ggml-org/gemma-4-E4B-it-GGUF",
                    GGUF {
                        quant_filename: "mtp-gemma-4-E4B-it-Q8_0.gguf".into(),
                    },
                    false,
                    "gemma",
                    "MTP",
                    "Q8_0",
                ),
            ),
            // Retrieval-track models for agent-assistant (rerank contract
            // 2026-08-22): two cross-encoder rerankers on /v1/rerank and two
            // candidate embedders on /v1/embeddings. Serving one at a time
            // is fine — the consumer swaps via config between runs.
            (
                "bge-reranker-v2-m3:latest".to_string(),
                builtin(
                    "bge-reranker-v2-m3:latest",
                    "gpustack/bge-reranker-v2-m3-GGUF",
                    Rerank {
                        quant_filename: Some("bge-reranker-v2-m3-Q8_0.gguf".into()),
                    },
                    false,
                    "bge",
                    "568M",
                    "Q8_0",
                ),
            ),
            // No upstream GGUF: converted from the official safetensors with
            // llama.cpp's convert_hf_to_gguf.py so cls.output.weight (yes/no)
            // is present; resolved via config-local path.
            (
                "qwen3-reranker-0.6b:latest".to_string(),
                builtin(
                    "qwen3-reranker-0.6b:latest",
                    "Qwen/Qwen3-Reranker-0.6B",
                    Rerank {
                        quant_filename: None,
                    },
                    false,
                    "qwen3",
                    "600M",
                    "F16",
                ),
            ),
            (
                "bge-m3:latest".to_string(),
                builtin(
                    "bge-m3:latest",
                    "gpustack/bge-m3-GGUF",
                    Embedding {
                        pooling: Pooling::Cls,
                        quant_filename: Some("bge-m3-Q8_0.gguf".into()),
                    },
                    false,
                    "bge",
                    "568M",
                    "Q8_0",
                ),
            ),
            (
                "qwen3-embedding-0.6b:latest".to_string(),
                builtin(
                    "qwen3-embedding-0.6b:latest",
                    "Qwen/Qwen3-Embedding-0.6B-GGUF",
                    Embedding {
                        pooling: Pooling::Last,
                        quant_filename: Some("Qwen3-Embedding-0.6B-Q8_0.gguf".into()),
                    },
                    false,
                    "qwen3",
                    "600M",
                    "Q8_0",
                ),
            ),
        ];

        let mut entries: HashMap<String, RegistryEntry> = builtins.into_iter().collect();
        // Stamp the embedding dims so a resolved entry carries its vector
        // length (collection sizing falls back to this when the GGUF is not
        // yet cached). Only Embedding-kind entries are touched.
        for (alias, dim) in Self::EMBED_DIMS {
            if let Some(e) = entries.get_mut(*alias) {
                e.embedding_dim = Some(*dim);
            }
        }

        Self {
            default_id: "gemma4:e2b".into(),
            // Adoption contract §1.4 (2026-09-10): the bench-winning qwen3
            // config becomes the shipped default. embed-gemma and bge-m3
            // stay registered as comparison/secondary models.
            default_embedding_id: "qwen3-embedding-0.6b:latest".into(),
            default_rerank_id: "qwen3-reranker-0.6b:latest".into(),
            entries,
        }
    }

    /// The built-in embedding models' vector lengths. `None` when the
    /// GGUF is the authoritative source (read `{arch}.embedding_length`
    /// at collection creation), these are the deterministic fallback so a
    /// collection can be sized before the model is pulled/cached.
    const EMBED_DIMS: &'static [(&'static str, usize)] = &[
        ("embed-gemma:latest", 768),
        ("bge-m3:latest", 1024),
        ("qwen3-embedding-0.6b:latest", 1024),
    ];

    /// Resolve an Ollama-style name to a registry entry.
    ///
    /// Resolution order:
    /// 1. Exact match in the registry (`gemma4:e4b`).
    /// 2. Append `:latest` and retry (`gemma4` → `gemma4:latest`).
    /// 3. If the name contains `:`, try as a direct HF repo id.
    /// 4. Return `ModelNotFound`.
    pub fn resolve(&self, name: &str) -> Result<&RegistryEntry, CoreError> {
        // 1. Exact match
        if let Some(entry) = self.entries.get(name) {
            return Ok(entry);
        }

        // 2. Try :latest
        if !name.contains(':') {
            let latest = format!("{}:latest", name);
            if let Some(entry) = self.entries.get(&latest) {
                return Ok(entry);
            }
        }

        // 3. Direct HF repo fallback (name contains `:` → treat as HF repo)
        if name.contains(':') {
            // This could be an unknown `model:tag` — don't guess, return not found
            return Err(CoreError::ModelNotFound(format!(
                "unknown model '{}'; configure it via config.toml [[model]] entries or use a known alias",
                name
            )));
        }

        Err(CoreError::ModelNotFound(format!(
            "unknown model '{}' and no matching Hugging Face repo found",
            name
        )))
    }

    /// Return the default model entry.
    pub fn default(&self) -> &RegistryEntry {
        self.entries
            .get(&self.default_id)
            .expect("default model must be in registry")
    }

    /// Return the default rerank model entry (adoption contract §1.4:
    /// qwen3-reranker-0.6b is the shipped default rerank alias).
    pub fn default_rerank(&self) -> &RegistryEntry {
        self.entries
            .get(&self.default_rerank_id)
            .expect("default rerank model must be in registry")
    }
    /// Return the default embedding model entry (spec resolved decision #3).
    pub fn default_embedding(&self) -> &RegistryEntry {
        self.entries
            .get(&self.default_embedding_id)
            .expect("default embedding model must be in registry")
    }

    /// Duplicate an entry under a new alias (`/api/copy`). Overwrites an
    /// existing destination, matching Ollama semantics.
    pub fn copy(&mut self, source: &str, destination: &str) -> Result<(), CoreError> {
        let mut entry = self.resolve(source)?.clone();
        entry.alias = destination.to_string();
        entry.is_default = false;
        entry.user_defined = true;
        self.entries.insert(destination.to_string(), entry);
        Ok(())
    }

    /// Remove an entry by alias (`/api/delete`). Tries the exact alias,
    /// then the `:latest` form.
    pub fn remove(&mut self, alias: &str) -> Result<RegistryEntry, CoreError> {
        if let Some(entry) = self.entries.remove(alias) {
            return Ok(entry);
        }
        if !alias.contains(':') {
            let latest = format!("{}:latest", alias);
            if let Some(entry) = self.entries.remove(&latest) {
                return Ok(entry);
            }
        }
        Err(CoreError::ModelNotFound(format!(
            "unknown model '{}'",
            alias
        )))
    }

    /// Apply user-defined model entries from config. These override
    /// built-in entries with the same alias; unknown aliases become
    /// user-defined entries resolved from the local `path` (or the HF cache
    /// by repo id).
    pub fn apply_user_models(&mut self, models: &[crate::config::ModelEntry]) {
        for m in models {
            let mut entry = self
                .entries
                .get(&m.id)
                .cloned()
                .unwrap_or_else(|| RegistryEntry {
                    alias: m.id.clone(),
                    hf_repo_id: m.id.clone(),
                    hf_quant_filename: None,
                    loader_kind: LoaderKind::GGUF {
                        quant_filename: String::new(),
                    },
                    revision: None,
                    is_default: m.default,
                    metadata: ModelMetadata {
                        family: String::new(),
                        families: vec![],
                        parameter_size: String::new(),
                        quantization: String::new(),
                        format: "gguf".into(),
                    },
                    user_defined: true,
                    local_path: None,
                    n_gpu_layers: None,
                    mmproj_path: None,
                    embedding_dim: None,
                });

            entry.user_defined = true;
            // A config-local GGUF path overrides HF-cache resolution.
            entry.local_path = m.path.clone();
            entry.n_gpu_layers = m.n_gpu_layers;
            entry.mmproj_path = m.mmproj_path.clone();
            if m.default {
                self.default_id = m.id.clone();
            }
            self.entries.insert(m.id.clone(), entry);
        }
    }

    /// List all registered model aliases (for /api/tags). Multiple lookup
    /// keys can map to the same entry (e.g. `gemma4` → `gemma4:e4b`), so the
    /// list is deduplicated by alias.
    pub fn list(&self) -> Vec<&RegistryEntry> {
        let mut v: Vec<&RegistryEntry> = self.entries.values().collect();
        v.sort_by(|a, b| a.alias.cmp(&b.alias));
        v.dedup_by(|a, b| a.alias == b.alias);
        v
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no entries are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Alias of the OCR-capable model (the first vision entry), if any.
    /// `/v1/ocr` targets this model; user configs can override it by
    /// registering their own `GgufVision` entry (or none).
    pub fn ocr_alias(&self) -> Option<String> {
        self.entries
            .values()
            .find(|e| matches!(e.loader_kind, LoaderKind::GgufVision { .. }))
            .map(|e| e.alias.clone())
    }

    /// The registry default alias for `role` when `name` asks for it: an
    /// empty name or the literal `"default"`. `None` means `name` is an
    /// explicit model id the caller resolves itself. This is the one place
    /// the "empty or `default` means the registry default" rule lives.
    pub fn default_alias_for(&self, name: &str, role: DefaultRole) -> Option<&str> {
        if !name.is_empty() && name != "default" {
            return None;
        }
        let entry = match role {
            DefaultRole::Embedding => self.default_embedding(),
            DefaultRole::Rerank => self.default_rerank(),
        };
        Some(entry.alias.as_str())
    }
}

/// Which registry default an empty / `"default"` model name selects
/// ([`ModelRegistry::default_alias_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultRole {
    Embedding,
    Rerank,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_alias_for_only_maps_empty_and_default() {
        let reg = ModelRegistry::new();
        for name in ["", "default"] {
            assert_eq!(
                reg.default_alias_for(name, DefaultRole::Embedding),
                Some(reg.default_embedding().alias.as_str())
            );
            assert_eq!(
                reg.default_alias_for(name, DefaultRole::Rerank),
                Some(reg.default_rerank().alias.as_str())
            );
        }
        assert_eq!(
            reg.default_alias_for("bge-m3", DefaultRole::Embedding),
            None
        );
        assert_eq!(reg.default_alias_for("Default", DefaultRole::Rerank), None);
    }

    #[test]
    fn declared_embedding_dim_is_embedder_only() {
        let reg = ModelRegistry::new();
        let embedder = reg.default_embedding();
        assert!(embedder.is_embedding());
        assert_eq!(embedder.declared_embedding_dim(), embedder.embedding_dim);
        assert!(embedder.declared_embedding_dim().is_some());
        let chat = reg.resolve("gemma4:e4b").unwrap();
        assert!(!chat.is_embedding());
        assert_eq!(chat.declared_embedding_dim(), None);
    }

    #[test]
    fn resolve_exact_match() {
        let reg = ModelRegistry::new();
        let entry = reg.resolve("gemma4:e4b").unwrap();
        assert_eq!(entry.hf_repo_id, "ggml-org/gemma-4-E4B-it-GGUF");
        assert!(matches!(entry.loader_kind, LoaderKind::GGUF { .. }));
    }

    #[test]
    fn resolve_bare_as_latest() {
        let reg = ModelRegistry::new();
        let entry = reg.resolve("gemma4").unwrap();
        // gemma4 -> gemma4:latest -> gemma4:e2b
        assert!(entry.is_default);
        assert_eq!(entry.alias, "gemma4:e2b");
    }

    #[test]
    fn resolve_e2b() {
        let reg = ModelRegistry::new();
        let entry = reg.resolve("gemma4:e2b").unwrap();
        assert_eq!(entry.hf_repo_id, "ggml-org/gemma-4-E2B-it-GGUF");
        assert_eq!(entry.metadata.parameter_size, "2B");
    }

    #[test]
    fn resolve_ocr_vision_entry() {
        let reg = ModelRegistry::new();
        let entry = reg.resolve("unlimited-ocr").unwrap();
        assert!(matches!(entry.loader_kind, LoaderKind::GgufVision { .. }));
        assert_eq!(entry.hf_repo_id, "sahilchachra/Unlimited-OCR-GGUF");
    }

    #[test]
    fn resolve_unknown_returns_err() {
        let reg = ModelRegistry::new();
        let result = reg.resolve("nonexistent:model");
        assert!(result.is_err());
    }

    #[test]
    fn default_embedding_is_qwen3() {
        let reg = ModelRegistry::new();
        let e = reg.default_embedding();
        assert_eq!(e.hf_repo_id, "Qwen/Qwen3-Embedding-0.6B-GGUF");
        assert!(matches!(
            e.loader_kind,
            LoaderKind::Embedding {
                pooling: Pooling::Last,
                ..
            }
        ));
        // Resolvable by bare name via :latest fallback.
        assert!(reg.resolve("qwen3-embedding-0.6b").is_ok());
        // embed-gemma stays registered as the comparison/secondary embedder.
        assert!(reg.resolve("embed-gemma").is_ok());
        // qwen3-reranker-0.6b is the default rerank alias (§1.4).
        assert_eq!(reg.default_rerank().alias, "qwen3-reranker-0.6b:latest");
    }

    #[test]
    fn rerank_contract_models_resolve() {
        let reg = ModelRegistry::new();
        let reranker = reg.resolve("bge-reranker-v2-m3").unwrap();
        assert_eq!(reranker.hf_repo_id, "gpustack/bge-reranker-v2-m3-GGUF");
        assert!(matches!(reranker.loader_kind, LoaderKind::Rerank { .. }));
        let qwen_reranker = reg.resolve("qwen3-reranker-0.6b").unwrap();
        assert!(matches!(
            qwen_reranker.loader_kind,
            LoaderKind::Rerank { .. }
        ));
        let bge_m3 = reg.resolve("bge-m3").unwrap();
        assert!(matches!(
            bge_m3.loader_kind,
            LoaderKind::Embedding {
                pooling: Pooling::Cls,
                ..
            }
        ));
        let qwen_emb = reg.resolve("qwen3-embedding-0.6b").unwrap();
        assert!(matches!(
            qwen_emb.loader_kind,
            LoaderKind::Embedding {
                pooling: Pooling::Last,
                ..
            }
        ));
    }

    #[test]
    fn copy_creates_alias() {
        let mut reg = ModelRegistry::new();
        reg.copy("gemma4:e4b", "mymodel:latest").unwrap();
        let entry = reg.resolve("mymodel").unwrap();
        assert_eq!(entry.hf_repo_id, "ggml-org/gemma-4-E4B-it-GGUF");
        assert!(entry.user_defined);
        assert!(reg.copy("nonexistent:x", "y").is_err());
    }

    #[test]
    fn remove_deletes_entry() {
        let mut reg = ModelRegistry::new();
        reg.copy("gemma4:e4b", "doomed:latest").unwrap();
        reg.remove("doomed").unwrap();
        assert!(reg.resolve("doomed").is_err());
        assert!(reg.remove("doomed").is_err());
    }

    #[test]
    fn default_is_e2b() {
        let reg = ModelRegistry::new();
        let def = reg.default();
        assert_eq!(def.alias, "gemma4:e2b");
    }

    #[test]
    fn user_override_default() {
        let mut reg = ModelRegistry::new();
        reg.apply_user_models(&[crate::config::ModelEntry {
            default: true,
            ..crate::config::ModelEntry::new("gemma4:e2b")
        }]);
        let def = reg.default();
        assert_eq!(def.alias, "gemma4:e2b");
    }

    #[test]
    fn weight_source_is_derived_from_the_entry() {
        let mut reg = ModelRegistry::new();
        // A built-in naming its quant file: a downloadable HF file.
        let e2b = reg.resolve("gemma4:e2b").unwrap().clone();
        assert_eq!(
            e2b.weight_source(),
            WeightSource::HfFile {
                repo: "ggml-org/gemma-4-E2B-it-GGUF",
                file: "gemma-4-E2B-it-Q4_0.gguf",
            }
        );
        assert!(e2b.weight_source().is_downloadable());
        // A built-in with no quant file: any cached GGUF, not downloadable.
        let qwen = reg.resolve("qwen3-reranker-0.6b").unwrap();
        assert_eq!(
            qwen.weight_source(),
            WeightSource::HfAnyGguf {
                repo: "Qwen/Qwen3-Reranker-0.6B"
            }
        );
        assert!(!qwen.weight_source().is_downloadable());

        // A configured path wins over the built-in's HF file, and an unknown
        // id resolves from the cache under the id itself.
        let local = std::path::PathBuf::from("/models/e2b.gguf");
        reg.apply_user_models(&[
            crate::config::ModelEntry {
                path: Some(local.clone()),
                ..crate::config::ModelEntry::new("gemma4:e2b")
            },
            crate::config::ModelEntry::new("org/custom-GGUF"),
        ]);
        let e2b = reg.resolve("gemma4:e2b").unwrap();
        assert_eq!(e2b.weight_source(), WeightSource::Local(&local));
        assert!(!e2b.weight_source().is_downloadable());
        let custom = reg.resolve("org/custom-GGUF").unwrap();
        assert_eq!(
            custom.weight_source(),
            WeightSource::HfAnyGguf {
                repo: "org/custom-GGUF"
            }
        );
        assert!(!custom.weight_source().is_downloadable());
    }

    #[test]
    fn mmproj_filename_only_for_vision_entries() {
        let reg = ModelRegistry::new();
        assert_eq!(
            reg.resolve("unlimited-ocr").unwrap().mmproj_filename(),
            Some("mmproj-Unlimited-OCR-F16.gguf")
        );
        assert_eq!(reg.resolve("gemma4:e2b").unwrap().mmproj_filename(), None);
    }

    #[test]
    fn list_returns_all() {
        let reg = ModelRegistry::new();
        let list = reg.list();
        // Deduplicated by alias: e2b (once, despite 3 lookup keys), e4b,
        // embed-gemma, unlimited-ocr, e2b-mtp, e4b-mtp, plus the rerank
        // contract's bge-reranker-v2-m3, qwen3-reranker-0.6b, bge-m3,
        // qwen3-embedding-0.6b.
        assert_eq!(list.len(), 10);
        assert_eq!(list.iter().filter(|e| e.alias == "gemma4:e2b").count(), 1);
        assert_eq!(
            list.iter()
                .filter(|e| e.alias == "unlimited-ocr:latest")
                .count(),
            1
        );
    }
}
