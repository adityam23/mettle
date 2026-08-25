# mettle

Local inference engine for GGUF models. Rust around an embedded llama.cpp core (`llama-cpp-2` FFI). One daemon serves chat, completions, embeddings, reranking and vision-OCR over OpenAI-, Ollama- and Anthropic-compatible HTTP APIs, and schedules models against a fixed VRAM budget instead of dying on it.

Built for small GPUs: a 6 GB card runs a chat generator plus an embedding model plus an OCR model concurrently, because the profile manager treats VRAM as a scheduled resource.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/adityam23/mettle/main/scripts/install.sh | sh
```

Installs `mettle` (CLI) and `mettled` (daemon) into `~/.local/bin`, creating `~/.config/mettle` and `~/.local/share/mettle` on first use. `METTLE_VARIANT=cuda` selects the CUDA-enabled Linux build. Uninstall: `scripts/uninstall.sh` (add `--purge` to delete config/state).

From source:

```sh
cargo build --release                    # CPU build
cargo build --release --features cuda    # Linux + NVIDIA toolchain
# binaries: target/release/mettle, target/release/mettled
```

## Quickstart

```sh
mettle tune              # detect GPU, write a tuned config.toml
mettle serve             # execs mettled in the foreground
mettle pull gemma4:e2b   # fetch a registry model from Hugging Face
```

```sh
curl -s http://127.0.0.1:11434/v1/chat/completions -d '{
  "model": "gemma4:e2b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "logprobs": true,
  "stream": true
}'
```

OpenAI-compatible `/v1/*` is the primary surface and includes per-token logprobs and streaming deltas. Native reasoning-model output lands in `message.reasoning_content` (chat template applied from the GGUF's embedded jinja).

## Configuration

`~/.config/mettle/config.toml` (override: `METTLE_CONFIG`). Zero-config first run works; everything below has defaults.

| Section | Key | Default | Notes |
|---|---|---|---|
| `[server]` | `host` / `port` | `127.0.0.1` / `11434` | |
| | `api_keys` | `[]` | Bearer auth when non-empty |
| | `offline` | `false` | Refuse network fetches |
| `[engine]` | `max_concurrent_seqs` | `16` | |
| | `kv_cache` | `fp8` | `fp8` or `fp16` |
| | `prefix_cache` | `true` | |
| | `speculative` | `auto` | `auto` \| `off` \| drafter alias |
| | `n_gpu_layers` | auto | Layers offloaded; `mettle tune` writes this |
| | `n_batch` | `512` | Lower shrinks compute-buffer reservations |
| | `n_ctx` | `8192` | KV ≈ 35 KB/token (gemma4 q8_0); 64K ≈ 4.8 GB |
| | `vram_margin_bytes` | `536870912` | Safety margin subtracted before admitting slots |
| `[[model]]` | `id` | — | Registry alias or HF repo id |
| | `path` / `mmproj_path` | — | Local GGUF overrides HF resolution |
| | `n_gpu_layers` / `min_ctx` / `draft` / `role` | inferred | Per-model overrides |
| `[[profile]]` | `name`, `slots` | — | Named model groups; one active at a time |
| `[retrieval]` | `data_dir` | XDG data dir | Vector stores |
| | `chunk_tokens` / `chunk_overlap` / `default_bit_width` | `512` / `64` / `4` | |

Environment: `METTLE_CONFIG` (config path), `METTLE_STATE` (state file), `RUST_LOG` (tracing filter).

## How it works

- **Slots, not processes.** `mettled` embeds llama.cpp behind a channel-driven worker (`LlamaEngine`). Models are slots loaded/unloaded atomically by alias.
- **Profiles with a degradation ladder.** A profile declares ordered slots with roles (chat/embed/rerank/ocr). Admission compares header-derived cost against free VRAM minus the safety margin; under pressure the ladder retries smaller context rungs down to each slot's `min_ctx`, then drops the slot. Measured post-load footprints persist in `~/.local/share/mettle/state.json` (invalidated by binary version or GGUF mtime) so second activations fit precisely.
- **Chunked prefill.** Long prompts decode incrementally across bounded batches instead of reserving for worst case.
- **Events.** Profile activation is async: `PUT /engine/profiles/active` answers `202`; completion arrives on the SSE bus at `/engine/events` (last 100 events replayed on connect).

## API surface

| Prefix | Style |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` | OpenAI |
| `/api/generate`, `/api/chat`, `/api/embeddings`, `/api/pull`, tags/copy/delete/show, `/api/metrics` | Ollama |
| `/v1/messages` | Anthropic |
| `/v1/rerank` | Cohere-style reranking |
| `/v1/ocr` | Vision-OCR (image in, text out) |
| `/engine/profiles`, `/engine/status`, `/engine/device`, `/engine/slots/{id}/load\|unload`, `/engine/events` | native profile control |

## Benchmarks

No published numbers — they are meaningless without your model and GPU. The CLI ships the harness: `mettle bench` drives streaming chat (TTFT, aggregate tokens/s) and optional collection search latency (p50/p95), emitting JSON.

## Backends

| Feature flag | Status |
|---|---|
| CPU | Default; always built |
| `cuda` | Release artifacts bundle CUDA runtime libs; arches 80/86/89 |
| `metal` | Enabled in the macOS arm64 release artifact |
| `vulkan`, `rocm` | Passthrough flags exist; not exercised in CI |

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). Links llama.cpp (MIT) via [llama-cpp-2](https://crates.io/crates/llama-cpp-2).
