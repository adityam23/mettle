//! Prompt construction: jinja chat templating, Gemma fallback, OCR (mtmd).

use std::collections::BTreeMap;

use anyhow::Result;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdInputText};
use llama_cpp_2::token::LlamaToken;
use minijinja::Environment;

use mettle_core::ollama_types::ChatMessage;

/// A constructed chat prompt: the tokens to prefill plus whether the model's
/// chat template leaves the generation **inside an unclosed reasoning block**
/// (e.g. Qwen3.5's template ends the generation prompt with `<think>\n`).
/// When set, the engine seeds the generation with that opener so downstream
/// reasoning-channel splitting sees a well-formed turn (see [`GenState`-
/// side seeding in `scheduler.rs`]).
#[derive(Debug, Clone)]
pub struct ChatPrompt {
    pub tokens: Vec<LlamaToken>,
    pub opens_reasoning: Option<&'static str>,
}

/// Reasoning tags recognised for opener-prefill detection (`<tag>` at the
/// very end of the rendered generation prompt, never closed).
const REASONING_TAGS: [&str; 4] = ["think", "thinking", "reasoning", "analysis"];

/// Build the chat prompt: the model's embedded jinja chat template when
/// llama.cpp recognises it, else the hand-spliced Gemma layout.
///
/// Rendering uses a real jinja engine (minijinja + pycompat), matching what
/// `llama-server --jinja`, vLLM and HF transformers do — NOT the legacy C
/// `llama_chat_apply_template`, which only pattern-matches a hardcoded list of
/// template shapes and silently reduces unknown templates to generic ChatML.
/// For Qwen3.5-style reasoning models that difference is fatal: their template
/// prefills `<|im_start|>assistant\n<think>\n`, the legacy render drops the
/// `<think>\n`, and on real (long) prompts the model degenerates at the
/// untrained seam (0-token EOS, stray tokens, unclosed thinking). See
/// agent-docs/downstream-consumers.md ("REGRESSION — long prompts").
pub fn build_chat_tokens(model: &LlamaModel, messages: &[ChatMessage]) -> Result<ChatPrompt> {
    // Gate on the legacy path first so routing stays 1:1 with the previous
    // behaviour: models whose embedded template llama.cpp cannot detect (Gemma)
    // keep the hand-spliced layout — rendering them with jinja would produce
    // turn-marker strings this build cannot tokenize back to special ids.
    if let Some(tokens) = build_legacy_chat_tokens(model, messages)? {
        if let Some(prompt) = build_minijinja_chat_tokens(model, messages)? {
            return Ok(prompt);
        }
        return Ok(ChatPrompt {
            tokens,
            opens_reasoning: None,
        });
    }
    Ok(ChatPrompt {
        tokens: build_gemma_chat_tokens(model, messages)?,
        opens_reasoning: None,
    })
}

/// Legacy fallback: apply the model's embedded jinja chat template (GGUF
/// `tokenizer.chat_template`) through llama.cpp's `llama_chat_apply_template`.
///
/// This C function does NOT parse jinja — it pattern-matches a hardcoded list
/// of known template shapes (`llama-chat.cpp: llm_chat_detect_template`) and
/// reduces anything containing `<|im_start|>` to generic ChatML, dropping
/// reasoning prefills like Qwen's `<think>\n`. Kept only as (a) the routing
/// probe that decides jinja-vs-Gemma-layout and (b) the degradation path when
/// minijinja cannot render a template. Returns `Ok(None)` whenever the path
/// doesn't apply (no embedded template, FFI rejection, unusable output).
fn build_legacy_chat_tokens(
    model: &LlamaModel,
    messages: &[ChatMessage],
) -> Result<Option<Vec<LlamaToken>>> {
    let Ok(tmpl) = model.chat_template(None) else {
        return Ok(None);
    };
    let chat = chat_messages(messages);
    if chat.is_empty() {
        return Ok(None);
    }
    // add_ass=true leaves the assistant-turn opener hanging so generation
    // continues inside the assistant turn instead of emitting one.
    let Ok(prompt) = model.apply_chat_template(&tmpl, &chat, true) else {
        return Ok(None);
    };
    if prompt.is_empty() {
        return Ok(None);
    }
    // The rendered template owns its special tokens (<|im_start|> etc.);
    // str_to_token parses specials, so no synthetic BOS goes on top — matching
    // llama-server --jinja for Qwen-style models.
    Ok(Some(model.str_to_token(&prompt, AddBos::Never)?))
}

/// Render the GGUF's embedded jinja chat template with a real jinja engine
/// (minijinja + pycompat), then tokenize. This is what `llama-server --jinja`,
/// vLLM and HF transformers do; it reproduces the template byte-for-byte,
/// including generation-prompt reasoning prefills (`<|im_start|>assistant\n
/// <think>\n` for Qwen3.5) that the legacy pattern-matcher silently drops.
fn build_minijinja_chat_tokens(
    model: &LlamaModel,
    messages: &[ChatMessage],
) -> Result<Option<ChatPrompt>> {
    let Ok(tmpl) = model.chat_template(None) else {
        return Ok(None);
    };
    let src = tmpl.to_str()?;
    if chat_messages(messages).is_empty() {
        return Ok(None);
    }

    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // HF templates expect plain message objects plus the vocab's special-token
    // strings; `tools` stays undefined so `{% if tools %}` takes the else arm.
    let msgs: Vec<BTreeMap<&str, minijinja::Value>> = messages
        .iter()
        .filter(|m| !m.content.is_empty())
        .map(|m| {
            BTreeMap::from([
                ("role", minijinja::Value::from(m.role.as_str())),
                ("content", minijinja::Value::from(m.content.as_str())),
            ])
        })
        .collect();
    let ctx = minijinja::context! {
        messages => msgs,
        add_generation_prompt => true,
        bos_token => special_token_str(model, model.token_bos()),
        eos_token => special_token_str(model, model.token_eos()),
    };

    let rendered = env.template_from_str(src)?.render(ctx)?;
    if rendered.is_empty() {
        return Ok(None);
    }
    let tokens = model.str_to_token(&rendered, AddBos::Never)?;
    Ok(Some(ChatPrompt {
        tokens,
        opens_reasoning: detect_reasoning_prefill(&rendered),
    }))
}

/// `Some(tag)` when the rendered generation prompt ends inside an unclosed
/// `<tag>` opener (e.g. Qwen3.5 ends with `<|im_start|>assistant\n<think>\n`):
/// the model continues INSIDE that reasoning block and must not re-emit the
/// opener for the channel splitter to recognise it.
fn detect_reasoning_prefill(rendered: &str) -> Option<&'static str> {
    let tail = rendered.trim_end();
    REASONING_TAGS
        .iter()
        .copied()
        .find(|&tag| tail.ends_with(&format!("<{tag}>")))
}

/// Wire messages into llama.cpp's legacy FFI type (empty content dropped —
/// the legacy renderer cannot express it).
fn chat_messages(messages: &[ChatMessage]) -> Vec<llama_cpp_2::model::LlamaChatMessage> {
    messages
        .iter()
        .filter(|m| !m.content.is_empty())
        .filter_map(|m| {
            llama_cpp_2::model::LlamaChatMessage::new(m.role.clone(), m.content.clone()).ok()
        })
        .collect()
}

/// Decode a special token id to its marker string ("" when absent/unmappable).
fn special_token_str(model: &LlamaModel, token: LlamaToken) -> String {
    #[allow(deprecated)]
    model
        .token_to_str(token, llama_cpp_2::model::Special::Tokenize)
        .unwrap_or_default()
}

/// Hand-spliced Gemma chat prompt — the fallback for models whose embedded
/// jinja template llama.cpp cannot apply (Gemma 4: `llama_chat_apply_template`
/// FFI returns -1), where `str_to_token`'s `parse_special` also does not map
/// the `<start_of_turn>`/`<end_of_turn>` marker *strings* to their special ids
/// in this build. So we splice the ids directly (105/106, stable across
/// Gemma 2/3/4) and tokenize only plain-text segments. Gemma has no system
/// role; system content is folded into the first user turn.
fn build_gemma_chat_tokens(
    model: &LlamaModel,
    messages: &[ChatMessage],
) -> Result<Vec<LlamaToken>> {
    const SOT: LlamaToken = LlamaToken(105); // <start_of_turn>
    const EOT: LlamaToken = LlamaToken(106); // <end_of_turn>
    let text = |s: &str| model.str_to_token(s, AddBos::Never);

    let mut toks = vec![model.token_bos()];
    let mut pending_system = String::new();
    for m in messages {
        match m.role.as_str() {
            "system" => {
                pending_system.push_str(&m.content);
                pending_system.push('\n');
            }
            role => {
                let tag = if role == "assistant" || role == "model" {
                    "model\n"
                } else {
                    "user\n"
                };
                toks.push(SOT);
                toks.extend(text(tag)?);
                if tag == "user\n" && !pending_system.is_empty() {
                    toks.extend(text(&format!("{pending_system}\n"))?);
                    pending_system.clear();
                }
                toks.extend(text(&m.content)?);
                toks.push(EOT);
                toks.extend(text("\n")?);
            }
        }
    }
    toks.push(SOT);
    toks.extend(text("model\n")?);
    Ok(toks)
}

/// Decode an encoded image (PNG/JPEG/BMP/GIF) into an mtmd bitmap.
pub fn image_bitmap(mtmd: &MtmdContext, bytes: &[u8]) -> Result<MtmdBitmap> {
    MtmdBitmap::from_buffer(mtmd, bytes, false).map_err(|e| anyhow::anyhow!("image decode: {e}"))
}

/// Build the mtmd tokenization input for an OCR request: the instruction
/// followed by the media marker (image-first, matching DeepSeek-OCR's
/// canonical `<image>\nConvert the document to markdown...` prompt shape).
pub fn ocr_input_text(instruction: &str) -> String {
    format!("{}\n{}", mtmd_default_marker(), instruction)
}

/// Tokenize OCR text + bitmap via mtmd. The returned chunks evaluate the
/// image embeddings and prompt into the target context.
pub fn ocr_input_chunks(
    mtmd: &MtmdContext,
    instruction: &str,
    bitmap: &MtmdBitmap,
) -> Result<llama_cpp_2::mtmd::MtmdInputChunks> {
    let text = MtmdInputText {
        text: ocr_input_text(instruction),
        add_special: true,
        parse_special: true,
    };
    mtmd.tokenize(text, &[bitmap])
        .map_err(|e| anyhow::anyhow!("mtmd tokenize: {e}"))
}

#[cfg(test)]
mod tests {
    use super::detect_reasoning_prefill;

    #[test]
    fn detects_think_prefill() {
        // Qwen3.5's generation prompt ends inside the prefilled think block.
        let p = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(detect_reasoning_prefill(p), Some("think"));
    }

    #[test]
    fn closed_block_is_not_a_prefill() {
        let p = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(detect_reasoning_prefill(p), None);
    }

    #[test]
    fn opener_in_history_is_not_a_prefill() {
        // A `<think>` mentioned in the user turn must not trip the detector:
        // only an opener at the very end of the rendered prompt counts.
        let p = "<|im_start|>user\nwhat does <think> mean?<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(detect_reasoning_prefill(p), None);
    }

    #[test]
    fn plain_prompt_has_no_prefill() {
        assert_eq!(detect_reasoning_prefill("<s>[INST] hi [/INST]"), None);
    }
}
