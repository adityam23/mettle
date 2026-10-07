//! GGUF-header VRAM estimation for profile planning
//! (agent-docs/profiles-contract.md: "estimate to plan, measure to
//! correct").
//!
//! Reads only GGUF metadata (`GgufContext` — no tensor load), so it is
//! safe to call for every catalog entry at config/activation time. The
//! numbers are deliberately conservative-ish heuristics: weights dominate
//! and are exact (file size), KV scales exactly with the header's
//! geometry, and the compute-buffer term is a documented approximation
//! that the planner's measured-reconciliation loop replaces with real
//! post-load usage after the first activation.

use std::path::Path;

use llama_cpp_2::gguf::GgufContext;

/// Estimated VRAM cost of one model at a given context size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramEstimate {
    /// Weights: the GGUF file size (tensor data dominates; metadata is a
    /// few KB). An mmproj projector's file size is added by the caller
    /// for vision models.
    pub weights_bytes: u64,
    /// KV-cache growth per context token across all layers, in bytes
    /// (both K and V, all heads). Multiply by `n_ctx`.
    pub kv_bytes_per_token: u64,
    /// Compute/workspace buffers at the planner's `n_ctx` — logits plus
    /// an activation heuristic. The least exact term; measurement
    /// corrects it.
    pub compute_bytes: u64,
}

impl VramEstimate {
    /// Total estimated VRAM at `n_ctx` context tokens.
    pub fn total_bytes(&self, n_ctx: u32) -> u64 {
        self.weights_bytes + self.kv_bytes_per_token * u64::from(n_ctx) + self.compute_bytes
    }
}

/// q8_0 KV blocks: 32 elements in 34 bytes (32 + 2-byte scale).
const Q8_0_BYTES_PER_ELEM: f64 = 34.0 / 32.0;
/// f16 KV elements.
const F16_BYTES_PER_ELEM: f64 = 2.0;
/// Activation/workspace heuristic: ~8 embedding-wide f32 rows per batch
/// token (attention Q/K/V/O plus FFN intermediates dominate in practice).
const ACTIVATION_ROWS_PER_BATCH_TOKEN: u64 = 8;
/// Batch the compute estimate assumes (matches `[engine] n_batch`
/// default; the true reservation is corrected by measurement).
const ESTIMATE_BATCH: u64 = 512;
/// Fixed CUDA workspace/context overhead heuristic.
const COMPUTE_BASE_BYTES: u64 = 64 * 1024 * 1024;
/// Vocab fallback when the header lacks `{arch}.vocab_size`.
const DEFAULT_VOCAB: u64 = 131_072;

/// Attention/state geometry read from a GGUF header.
struct Geometry {
    n_embd: u64,
    n_vocab: u64,
    /// Bytes of KV per context token, over the layers that actually hold a
    /// KV cache (attention layers only — see `full_attention_interval`).
    kv_bytes_per_token: u64,
    /// Recurrent (SSM/GDN) state per sequence in bytes; 0 for pure
    /// attention models. Fixed per sequence, not per token (§1.7).
    recurrent_state_per_seq: u64,
}

/// Read the geometry keys, resolving the arch prefix from
/// `general.architecture` (falling back to `llama`).
fn geometry(ctx: &GgufContext, kv_q8: bool) -> Option<Geometry> {
    let arch_idx = ctx.find_key("general.architecture");
    let arch = (arch_idx >= 0)
        .then(|| ctx.val_str(arch_idx))
        .flatten()
        .unwrap_or("llama")
        .to_owned();
    let key = |name: &str| {
        let idx = ctx.find_key(&format!("{arch}.{name}"));
        (idx >= 0).then(|| ctx.val_u32(idx))
    };

    let n_layer = u64::from(key("block_count")?);
    let n_embd = u64::from(key("embedding_length")?);
    let n_head = u64::from(key("attention.head_count")?);
    if n_head == 0 || n_layer == 0 {
        return None;
    }
    // head_count_kv absent (or 0) means MHA — every head carries KV.
    let n_head_kv = match key("attention.head_count_kv") {
        Some(0) | None => n_head,
        Some(v) => u64::from(v),
    };
    // Explicit K/V dims (`{arch}.attention.key_length` — Gemma, Qwen3.5,
    // MLA models) else embd/head.
    let k_dim = match key("attention.key_length") {
        Some(v) => u64::from(v),
        None => n_embd / n_head,
    };
    let v_dim = match key("attention.value_length") {
        Some(v) => u64::from(v),
        None => k_dim,
    };
    let n_vocab = key("vocab_size").map(u64::from).unwrap_or(DEFAULT_VOCAB);

    // §1.7: hybrid-recurrent architectures (qwen35, qwen3next, …) interleave
    // full-attention layers with linear-attention (SSM/GDN) layers that hold
    // NO KV cache. `{arch}.full_attention_interval = k` means every k-th
    // layer is full attention; counting all `block_count` layers overcounts
    // the KV by that factor (qwen35-4b: 32 layers → 8, 69632 → 17408
    // B/token). The recurrent state is a fixed per-sequence cost instead.
    let interval = key("full_attention_interval").filter(|v| *v > 1);
    let n_layer_kv = match interval {
        Some(k) => n_layer.div_ceil(u64::from(k)),
        None => n_layer,
    };
    let bytes_per_elem = if kv_q8 {
        Q8_0_BYTES_PER_ELEM
    } else {
        F16_BYTES_PER_ELEM
    };
    let kv_bytes_per_token = (n_layer_kv * (k_dim + v_dim) * n_head_kv) as f64 * bytes_per_elem;

    let recurrent_state_per_seq = match interval {
        Some(_) => {
            let recurrent_layers = n_layer.saturating_sub(n_layer_kv);
            let state_size = key("ssm.state_size").map(u64::from);
            let inner = key("ssm.inner_size").map(u64::from);
            match (state_size, inner) {
                // GDN state per layer: inner_size × state_size f32 elements
                // ([v_heads, k_dim, v_dim] with v_heads*v_dim = inner_size).
                (Some(state), Some(inner)) => recurrent_layers * state * inner * 4,
                _ => 0,
            }
        }
        None => 0,
    };

    Some(Geometry {
        n_embd,
        n_vocab,
        kv_bytes_per_token: kv_bytes_per_token as u64,
        recurrent_state_per_seq,
    })
}

/// Estimate the VRAM cost of `gguf` at `n_ctx` context tokens.
///
/// `kv_q8` mirrors `[engine] kv_cache = "fp8"` (llama.cpp q8_0 KV);
/// `n_seq_max` scales the per-sequence logits buffer and the recurrent
/// state. Returns `None` when the file is unreadable or lacks the
/// architecture geometry keys.
pub fn estimate_vram(
    gguf: &Path,
    _n_ctx: u32,
    kv_q8: bool,
    n_seq_max: usize,
) -> Option<VramEstimate> {
    let ctx = GgufContext::from_file(gguf)?;
    let weights_bytes = std::fs::metadata(gguf).ok()?.len();
    let g = geometry(&ctx, kv_q8)?;

    // Logits: one f32 vocab row per sequence. Activations: the heuristic
    // above at the estimate batch width. Recurrent state is a fixed
    // per-sequence cost (§1.7).
    let n_seq = n_seq_max.max(1) as u64;
    let logits = g.n_vocab * 4 * n_seq;
    let activations = ESTIMATE_BATCH * g.n_embd * 4 * ACTIVATION_ROWS_PER_BATCH_TOKEN;
    let recurrent = g.recurrent_state_per_seq * n_seq;
    let compute_bytes = COMPUTE_BASE_BYTES + logits + activations + recurrent;

    Some(VramEstimate {
        weights_bytes,
        kv_bytes_per_token: g.kv_bytes_per_token,
        compute_bytes,
    })
}

/// Measured CUDA compute-buffer cost per packed rerank token at hidden
/// width 1024 (residency contract §0: a 2040-token, 6-sequence pass on the
/// 0.6B Qwen3 reranker reserved 1490.48 MiB ⇒ ~0.73 MiB/token). Cross-encoder
/// activation buffers scale ~linearly with the hidden width, so this is
/// scaled by `{arch}.embedding_length / 1024`.
const RERANK_COMPUTE_BYTES_PER_TOKEN_AT_1024: u64 = 768 * 1024;

/// Precomputed rerank-context cost coefficients for one model (read once
/// at load; the per-call preflight stays arithmetic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RerankGeometry {
    base_bytes: u64,
    compute_bytes_per_token: u64,
    kv_bytes_per_token: u64,
}

impl RerankGeometry {
    /// Explicit coefficients — unit tests only (production reads them from
    /// the GGUF header in [`rerank_geometry`]).
    #[cfg(test)]
    pub fn with_coefficients(
        base_bytes: u64,
        compute_bytes_per_token: u64,
        kv_bytes_per_token: u64,
    ) -> Self {
        Self {
            base_bytes,
            compute_bytes_per_token,
            kv_bytes_per_token,
        }
    }

    /// Cost of a packed pass of `tokens` tokens over a KV of `kv_cells`
    /// (split-KV: `max_len` rounded to 256 times the sequence count).
    pub fn cost(&self, kv_cells: usize, tokens: usize) -> RerankContextCost {
        RerankContextCost {
            kv_bytes: self.kv_bytes_per_token * kv_cells as u64,
            compute_bytes: self.base_bytes + self.compute_bytes_per_token * tokens as u64,
        }
    }
}

impl RerankGeometry {
    /// [`rerank_cap_cost`] over already-read coefficients: pure arithmetic,
    /// so a planner walking several cap rungs parses the header once.
    pub fn cap_cost(&self, cap_tokens: usize, micro_batch: usize) -> RerankContextCost {
        let cap = cap_tokens.max(1);
        let n_seqs = micro_batch.max(1).min(cap);
        let max_len = cap.div_ceil(n_seqs);
        let kv_cells = max_len.div_ceil(256) * 256 * n_seqs;
        self.cost(kv_cells, cap)
    }
}

/// Estimated cost of one rerank call's context (residency contract §1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RerankContextCost {
    pub kv_bytes: u64,
    pub compute_bytes: u64,
}

impl RerankContextCost {
    pub fn total(&self) -> u64 {
        self.kv_bytes.saturating_add(self.compute_bytes)
    }
}

/// Read the rerank-context cost coefficients for `gguf`.
pub fn rerank_geometry(gguf: &Path, kv_q8: bool) -> Option<RerankGeometry> {
    let ctx = GgufContext::from_file(gguf)?;
    let g = geometry(&ctx, kv_q8)?;
    Some(RerankGeometry {
        base_bytes: COMPUTE_BASE_BYTES,
        compute_bytes_per_token: RERANK_COMPUTE_BYTES_PER_TOKEN_AT_1024 * g.n_embd / 1024,
        kv_bytes_per_token: g.kv_bytes_per_token,
    })
}

/// Worst-case-per-call cost a rerank slot must budget at admission for a
/// packed-token cap of `cap_tokens` under a `micro_batch`-sequence shape:
/// a full pass at the cap with the greedy fill's nominal (roughly even)
/// sequence lengths. The per-call preflight re-checks the real geometry.
pub fn rerank_cap_cost(
    gguf: &Path,
    cap_tokens: usize,
    micro_batch: usize,
    kv_q8: bool,
) -> Option<RerankContextCost> {
    rerank_geometry(gguf, kv_q8).map(|g| g.cap_cost(cap_tokens, micro_batch))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal metadata-only GGUF (v3): magic, version, zero
    /// tensors, then string/u32 KV pairs.
    fn write_synthetic_gguf(path: &Path, arch: &str) {
        use std::io::Write;
        const TYPE_STRING: u32 = 8;
        const TYPE_UINT32: u32 = 4;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&8u64.to_le_bytes()); // kv count

        let put_str = |buf: &mut Vec<u8>, s: &str| {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        };
        let put_kv_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
            put_str(buf, k);
            buf.extend_from_slice(&TYPE_STRING.to_le_bytes());
            put_str(buf, v);
        };
        let put_kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            put_str(buf, k);
            buf.extend_from_slice(&TYPE_UINT32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };

        put_kv_str(&mut buf, "general.architecture", arch);
        put_kv_u32(&mut buf, &format!("{arch}.block_count"), 36);
        put_kv_u32(&mut buf, &format!("{arch}.embedding_length"), 2560);
        put_kv_u32(&mut buf, &format!("{arch}.attention.head_count"), 32);
        put_kv_u32(&mut buf, &format!("{arch}.attention.head_count_kv"), 8);
        put_kv_u32(&mut buf, &format!("{arch}.attention.key_length"), 128);
        put_kv_u32(&mut buf, &format!("{arch}.attention.value_length"), 128);
        put_kv_u32(&mut buf, &format!("{arch}.vocab_size"), 151_936);

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
        // Pad so weights (file size) is non-trivially distinct.
        f.write_all(&vec![0u8; 1_000_000]).unwrap();
        f.sync_all().unwrap();
    }

    #[test]
    fn estimates_kv_geometry_from_arch_prefixed_header() {
        let dir = std::env::temp_dir().join("mettle-vram-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic-qwen3.gguf");
        write_synthetic_gguf(&path, "qwen3");
        let file_len = std::fs::metadata(&path).unwrap().len();

        // qwen3: 36 layers, GQA 32:8, K/V dim 128 →
        // 36 * (128+128) * 8 = 73_728 bytes/token at f16.
        let est = estimate_vram(&path, 8192, false, 16).unwrap();
        assert_eq!(est.weights_bytes, file_len);
        assert_eq!(est.kv_bytes_per_token, 36 * (128 + 128) * 8 * 2);
        let total = est.total_bytes(8192);
        assert!(total > file_len + 73_728 * 8192);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn q8_kv_and_missing_arch_fallback() {
        let dir = std::env::temp_dir().join("mettle-vram-test");
        std::fs::create_dir_all(&dir).unwrap();

        // q8 KV: 34/32 bytes per element.
        let path = dir.join("synthetic-q8.gguf");
        write_synthetic_gguf(&path, "qwen3");
        let est = estimate_vram(&path, 4096, true, 16).unwrap();
        let expected = (36.0 * 256.0 * 8.0 * (34.0 / 32.0)) as u64;
        assert_eq!(est.kv_bytes_per_token, expected);
        std::fs::remove_file(&path).ok();

        // No general.architecture: falls back to the `llama.` prefix.
        let path = dir.join("synthetic-llama.gguf");
        write_synthetic_gguf(&path, "llama");
        assert!(estimate_vram(&path, 4096, false, 16).is_some());
        std::fs::remove_file(&path).ok();

        // Not a GGUF at all: soft failure.
        let path = dir.join("not-a-gguf.bin");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(estimate_vram(&path, 4096, false, 16).is_none());
        std::fs::remove_file(&path).ok();
    }

    /// Sanity against a real GGUF (qwen35-4b Q4_K_M from the finance
    /// benchmark dir). Ignored by default — needs the model on disk.
    /// Run: `cargo test -p mettle-llama --lib -- --ignored real_gguf --nocapture`
    #[test]
    #[ignore]
    fn real_gguf_estimate() {
        let p = std::path::PathBuf::from(
            std::env::var("METTLE_QWEN_GGUF").expect("set METTLE_QWEN_GGUF to a real GGUF"),
        );
        if !p.exists() {
            return; // model not present on this host
        }
        let est = estimate_vram(&p, 8192, true, 16).unwrap();
        // Real header: 32 layers, 16:4 GQA, K/V dim 256 → q8 KV
        // 32*(256+256)*4*34/32 = 69_632 bytes/token.
        assert_eq!(est.kv_bytes_per_token, 69_632);
        // Q4_K_M 4B weights ≈ 2.5–3 GB.
        assert!(est.weights_bytes > 2_000_000_000, "{}", est.weights_bytes);
        let total = est.total_bytes(8192);
        // Must land in the ballpark we measured on the real card (~4.1 GB).
        assert!(total < 6_000_000_000, "{}", total);
        println!(
            "qwen35-4b @8k q8-KV: weights={} MiB, kv/token={} B, compute={} MiB, total={} MiB",
            est.weights_bytes / 1024 / 1024,
            est.kv_bytes_per_token,
            est.compute_bytes / 1024 / 1024,
            total / 1024 / 1024
        );
    }
}
