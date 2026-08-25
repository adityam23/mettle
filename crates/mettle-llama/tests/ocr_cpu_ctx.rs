//! Repro: does a ngl=0 (CPU) model context reserve CUDA compute buffers in
//! llama.cpp 0.1.154? Observed: the engine's OCR context reserves ~540 MiB
//! CUDA even with op_offload=false, while a standalone context reserves 0.
//! Hypothesis: prior context creations in the same process change the
//! scheduler's placement (backend state).

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

fn models_root() -> std::path::PathBuf {
    std::env::var_os("METTLE_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"))
}

fn gguf(rel: &str) -> std::path::PathBuf {
    models_root().join(rel)
}

#[test]
fn cpu_model_context_reserves_cuda() {
    let backend = LlamaBackend::init().unwrap();

    // First: a GPU model + context, like the engine's MTP lane target.
    let gemma = LlamaModel::load_from_file(
        &backend,
        gguf("gemma-4-e2b/gemma-4-E2B-it-Q4_0.gguf"),
        &LlamaModelParams::default().with_n_gpu_layers(999),
    )
    .expect("load gemma");
        
        
    let gemma_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(8192).unwrap()))
        .with_n_batch(512)
        .with_n_seq_max(1)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
        .with_n_threads(16);
    let _gemma_ctx = gemma.new_context(&backend, gemma_params).expect("gemma ctx");

    // Then the CPU model + context (op_offload=false like the engine).
    let model = LlamaModel::load_from_file(
        &backend,
        gguf("unlimited-ocr/Unlimited-OCR-Q4_K_M.gguf"),
        &LlamaModelParams::default().with_n_gpu_layers(0),
    )
    .expect("load OCR model with 0 GPU layers");
    use llama_cpp_2::context::params::KvCacheType;
    let params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(8192).unwrap()))
        .with_n_batch(512)
        .with_n_seq_max(4)
        .with_type_k(KvCacheType::Q8_0)
        .with_type_v(KvCacheType::Q8_0)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
        .with_n_threads(16)
        .with_op_offload(false);
    let _ctx = model.new_context(&backend, params).expect("context");
    eprintln!("ocr_cpu_ctx: context created OK");
}
