//! Chat-template path selection against real GGUFs.
//!
//! Ignored by default (needs model files). Run manually:
//!   cargo test -p mettle-llama --test chat_template_paths -- --ignored --nocapture
//!
//! Contract: agent-docs/downstream-consumers.md ("Serving different models
//! properly"). Arbitrary instruct/reasoning GGUFs must get their embedded
//! jinja template applied (`build_jinja_chat_tokens`), while curated gemma4
//! models keep the hand-spliced Gemma layout (their jinja template makes the
//! FFI fail).

use mettle_core::ollama_types::ChatMessage;
use mettle_llama::prompt::build_chat_tokens;

fn models_root() -> std::path::PathBuf {
    std::env::var_os("METTLE_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
}

fn gguf(rel: &str) -> String {
    models_root().join(rel).to_string_lossy().into_owned()
}

/// The Qwen distill GGUF lives outside this repo (agent-assistant
/// benchmarks); overridable for hosts without it.
fn qwen_gguf() -> String {
    std::env::var("METTLE_QWEN_GGUF").unwrap_or_else(|_| {
        "/personal-projects/agentic-assistant/benchmarks/financebench/models/qwen35-4b-opus-distill.Q4_K_M.gguf"
            .into()
    })
}

fn piece(model: &llama_cpp_2::model::LlamaModel, t: llama_cpp_2::token::LlamaToken) -> String {
    #[allow(deprecated)]
    model
        .token_to_str(t, llama_cpp_2::model::Special::Tokenize)
        .unwrap_or_default()
}

fn render(
    model: &llama_cpp_2::model::LlamaModel,
    toks: &[llama_cpp_2::token::LlamaToken],
) -> String {
    toks.iter().map(|t| piece(model, *t)).collect()
}

static BACKEND: std::sync::LazyLock<llama_cpp_2::llama_backend::LlamaBackend> =
    std::sync::LazyLock::new(|| llama_cpp_2::llama_backend::LlamaBackend::init().unwrap());

fn llama_backend() -> &'static llama_cpp_2::llama_backend::LlamaBackend {
    &BACKEND
}

fn llama_model(path: &str) -> llama_cpp_2::model::LlamaModel {
    llama_cpp_2::model::LlamaModel::load_from_file(
        llama_backend(),
        std::path::Path::new(path),
        &Default::default(),
    )
    .unwrap()
}

fn messages(user: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".into(),
            content: "You are helpful.".into(),
            images: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".into(),
            content: user.into(),
            images: None,
            tool_calls: None,
        },
    ]
}

/// Qwen3.5-4B Opus distill (the FinanceBench generator): its embedded jinja
/// template must be applied, producing `<|im_start|>` framing instead of the
/// Gemma turn layout that caused reasoning leak + repetition loops.
#[test]
#[ignore = "needs the qwen35-4b distill GGUF on disk; run with --ignored"]
fn qwen_distill_uses_embedded_template() {
    let path = qwen_gguf();
    let model = llama_model(&path);
    let built = build_chat_tokens(&model, &messages("hi")).unwrap();
    let toks = &built.tokens;

    let first = piece(&model, toks[0]);
    assert_eq!(
        first, "<|im_start|>",
        "jinja template path should open with <|im_start|>, got {first:?}"
    );
    let rendered: String = toks[..3].iter().map(|t| piece(&model, *t)).collect();
    assert!(
        rendered.starts_with("<|im_start|>system"),
        "expected system-role framing, got {rendered:?}"
    );
}

/// Curated gemma4 keeps the hand-spliced layout: explicit BOS then
/// `<start_of_turn>` (id 105).
#[test]
#[ignore = "needs models/gemma-4-e2b on disk; run with --ignored"]
fn gemma_keeps_spliced_layout() {
    let model = llama_model(&gguf("gemma-4-e2b/gemma-4-E2B-it-Q4_0.gguf"));
    let toks = build_chat_tokens(&model, &messages("hi")).unwrap().tokens;
    assert_eq!(toks[0], model.token_bos());
    assert_eq!(toks[1], llama_cpp_2::token::LlamaToken(105));
}

/// Contract ask (2026-08-24 regression): a LONG real-prompt case. The old
/// tests passed on short prompts while real serving failed, because the
/// legacy ChatML render drops the template's `<think>\n` generation prefill —
/// invisible at ~60 tokens, degenerate (0-token EOS, stray tokens, unclosed
/// thinking) on grounded multi-chunk context.
#[test]
#[ignore = "needs the qwen35-4b distill GGUF on disk; run with --ignored"]
fn qwen_distill_long_grounded_prompt_keeps_reasoning_prefill() {
    let path = qwen_gguf();
    let model = llama_model(&path);

    // Multi-chunk grounded context in the shape agent-assistant sends
    // (well past 2000 prompt tokens).
    let chunk = "3M's capital expenditures for fiscal year 2023 were $1,577 million, \
primarily directed toward manufacturing capacity and R&D facilities. \
The company reported operating margins of 19.2% with revenue of $32.7 billion. \
Depreciation and amortization totaled $614 million for the period. \
Research and development spending reached $1.8 billion, up 4% year over year. ";
    let user = format!(
        "Answer the question using ONLY the provided context. Cite sources as [n].\n\n\
         Question: What was 3M's capital expenditure in fiscal 2023?\n\nContext:\n{}",
        chunk.repeat(30)
    );
    assert!(
        user.len() > 8000,
        "context must be long enough to reproduce"
    );

    let built = build_chat_tokens(&model, &messages(&user)).unwrap();
    assert!(
        built.tokens.len() >= 2000,
        "expected >=2k prompt tokens, got {}",
        built.tokens.len()
    );

    // The generation prompt must end inside the prefilled thinking block:
    // `<|im_start|>assistant\n<think>\n`.
    let tail = render(&model, &built.tokens[built.tokens.len() - 12..]);
    assert!(
        tail.ends_with("<|im_start|>assistant\n<think>\n"),
        "long prompt must keep the template's <think> prefill; tail was {tail:?}"
    );
    assert_eq!(built.opens_reasoning, Some("think"));
}

/// Root-cause evidence for the 2026-08-24 long-prompt regression: dropping
/// the template's `<think>\n` prefill changes greedy decoding's FIRST token
/// on a long grounded prompt (the old ChatML render forces the model to emit
/// its own opener — or worse, an immediate end-of-turn token, as observed on
/// the real FinanceBench prompts), while the true jinja render continues
/// inside the prefilled block as trained.
#[test]
#[ignore = "needs the qwen35-4b distill GGUF on disk; CPU prefill takes ~1min"]
fn qwen_legacy_render_flips_first_token_to_eos_on_long_prompt() {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::model::AddBos as AB;
    use llama_cpp_2::model::LlamaChatMessage;

    let path = qwen_gguf();
    let model = llama_model(&path);
    let backend = llama_backend();

    let chunk = "3M's capital expenditures for fiscal year 2023 were $1,577 million, \
primarily directed toward manufacturing capacity and R&D facilities. \
The company reported operating margins of 19.2% with revenue of $32.7 billion. ";
    let user = format!(
        "Answer the question using ONLY the provided context.\n\nContext:\n{}",
        chunk.repeat(60)
    );

    // True jinja render (the fixed path).
    let fixed = build_chat_tokens(&model, &messages(&user)).unwrap().tokens;
    assert!(fixed.len() >= 2000);

    // Legacy ChatML render (the old broken path).
    let tmpl = model.chat_template(None).unwrap();
    let chat = vec![
        LlamaChatMessage::new("system".into(), "You are helpful.".into()).unwrap(),
        LlamaChatMessage::new("user".into(), user).unwrap(),
    ];
    let legacy_prompt = model.apply_chat_template(&tmpl, &chat, true).unwrap();
    let legacy = model.str_to_token(&legacy_prompt, AB::Never).unwrap();
    assert_ne!(fixed.len(), legacy.len(), "renders must differ");

    let first_token = |toks: &[llama_cpp_2::token::LlamaToken]| -> String {
        let n = toks.len() as u32;
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(std::num::NonZeroU32::new(n + 8).unwrap()))
            .with_n_batch(n + 8)
            .with_n_ubatch(n + 8)
            .with_n_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get() as i32)
                    .unwrap_or(4),
            );
        let mut ctx = model.new_context(backend, params).unwrap();
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(toks.len(), 1);
        for (i, t) in toks.iter().enumerate() {
            batch.add(*t, i as i32, &[0], i == toks.len() - 1).unwrap();
        }
        ctx.decode(&mut batch).unwrap();
        let logits = ctx.get_logits_ith((toks.len() - 1) as i32);
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        piece(&model, llama_cpp_2::token::LlamaToken(best as i32))
    };

    let broken = first_token(&legacy);
    let good = first_token(&fixed);
    println!("legacy first token: {broken:?}; jinja first token: {good:?}");
    assert_ne!(broken, good, "prefill must change greedy continuation");
}

/// Dump mettle's exact rendered prompt for a request payload so it can be
/// diffed against llama-server --jinja (`--verbose-prompt`) and the jinja2
/// reference render. Writes /tmp/infengine_render.txt.
#[test]
#[ignore = "needs the qwen35-4b distill GGUF on disk"]
fn dump_rendered_prompt_from_payload() {
    let path = qwen_gguf();
    let model = llama_model(&path);
    let payload: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/qwen-e2e/realctx.json").unwrap())
            .unwrap();
    let msgs: Vec<ChatMessage> = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| ChatMessage {
            role: m["role"].as_str().unwrap().into(),
            content: m["content"].as_str().unwrap().into(),
            images: None,
            tool_calls: None,
        })
        .collect();
    let built = build_chat_tokens(&model, &msgs).unwrap();
    let text = render(&model, &built.tokens);
    std::fs::write("/tmp/infengine_render.txt", &text).unwrap();
    println!(
        "tokens={} opens_reasoning={:?}",
        built.tokens.len(),
        built.opens_reasoning
    );
    println!("tail: {}", &text[text.len() - 120..]);
}
