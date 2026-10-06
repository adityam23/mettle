# BEST_CONFIG_VERDICT — mettle best-config parity + turbovec ablation (2026-09-10)

Executed per `BEST_CONFIG_CONTRACT.md`. Inputs: 150 FinanceBench questions /
84 filings / 36,920 chunks, embedder `qwen3-embedding-0.6b:latest`,
reranker `qwen3-reranker-0.6b:latest`. Served on the CUDA build of `mettled`
(all layers `dev = CUDA0`, 2,796 MiB GPU-resident, `n_gpu_layers=999`; verified
via `nvidia-smi` + `ggml_backend_cuda_*` symbols in the binary — no CPU fallback).

Outputs: `results/mettle_best_config.json` (Part A+B),
`results/turbovec_ablation.json` (Part C), state files alongside.

## 1. Head-to-head: mettle vs docket (recall@k, t0.5, plus @20 t0.8 + mbo)

| arm | @3 | @6 | @10 | @20_t0.8 | mbo | docket same arm @10 | Δ@10 |
|---|---|---|---|---|---|---|---|
| exact | .627 | .727 | **.793** | .760 | .9106 | .793 | **.000** |
| dense (served) | .620 | .747 | .787 | .773 | .9109 | — (sanity ≈ exact ✓) | |
| hybrid (served) | .620 | .747 | .787 | .773 | .9109 | .667 | **+.120** |
| keyword_or (FTS5 OR + bm25) | .320 | .433 | **.487** | .500 | .7523 | bm25 .380 | **+.107** |
| rerank_exact(qwen3-rr) | .760 | .840 | **.880** | .813 | .9106 | .880 | **.000** |
| rerank_hybrid(qwen3-rr) | .747 | .827 | .867 | .813 | .9109 | **.913** (docket winner) | **−.046** |

Latency (p50, this host): exact 36.3 ms (local rank) / 43.8 ms (served dense) /
44.5 ms (served hybrid) / 3.2 ms (keyword_or) / query embed 40.6 ms /
rerank call 12.5 s (p95 15.6 s, 40 docs, GPU).

**Q1 — does mettle match docket's winning config?** On every arm the two
systems share, they agree to the third decimal (exact and rerank_exact are
bit-for-bit parity: .793/.880 @10, identical @3/@6/@20 and mbo). What does NOT
transfer is the *winner itself*: docket's `.913` rerank_hybrid rides on its RRF
hybrid candidate pool, whose diversity is what the reranker converts into
recall (+.246 over its .667 hybrid base). Mettle's hybrid is FTS5-candidates →
dense rerank, which collapses to dense (.787 == dense), so reranking it yields
only +.08 → `.867`. **Mettle's best config is rerank_exact(.qwen3-rr) = .880@10
(.813@20t0.8) — full parity with docket's rerank_exact arm, 4.6 points below
docket's overall winner.** To close that gap mettle would need a diverse
candidate pool (e.g. RRF-style union of dense + keyword top-40) — a hybrid
fusion change, not a reranker change.

**Q2 — FTS5 keyword leg salvageable? YES, and it outperforms docket's bm25.**
OR-mode + `bm25()` ranking: **.487@10 / .752 mbo** vs docket `rank_bm25`
.380@10 / .6503 mbo (Δ +.107; decision rule: within .02 of .380 → salvage,
here mettle exceeds it). Current production keyword mode is still AND-expansion
(0.000). **Keyword stays in mettle with a named fix: add an OR-mode FTS5
search mode** (same query path as Part B: tokenized alnum OR-join, bm25
ordering). Caveats disclosed: FTS5 unicode61 tokenizer vs `[a-z0-9]+` metric
tokenization; indexed text carries the 2000-char embed_clean cap.

**Q3 — turbovec vs the field (N=36,920 × 1024 f32, 150 real queries):**

| engine | recall vs exact (top-10 overlap) | unfiltered p50 | index bytes | 4× scale p50 | filtered 1% p50 / match | deletes |
|---|---|---|---|---|---|---|
| exact_numpy | 1.0 (reference) | 6.58 ms (5.92 1-thread) | 151,224,448 (f32 raw) | 24.1 ms | 0.11 ms / 1.0 | no |
| **turbovec bw4 (mettle-core)** | **.954** (metric Δ +.007) | **0.555 ms** | **20,459,859 (7.4×)** | 2.71 ms | 0.19 / 0.70 | **yes, O(1)** |
| turbovec bw2 | .861 | 0.272 ms | 10,474,227 | — | 0.14 / 0.28 | yes, O(1) |
| faiss_flat_ip | 1.0 | 9.58 ms | 151,224,365 | 22.5 ms | 5.46 / 0.99 | no |
| faiss_hnsw (M32, ef64) | .989 | **0.23 ms** | 161,274,754 (8× fat) | 0.30 ms | 1.07 / 0.99 | no |
| faiss_ivfpq (768 lists, PQ 256×4bit) | .616 | 0.40 ms | 8,238,708 | — | 0.68 / 0.02 | no |
| hnswlib (M32, ef64) | .993 | 0.32 ms | 161,419,908 | — | 1.55 / 0.76 | soft (mark_deleted) |

- **turbovec beats numpy brute force on this corpus: 11.8× faster at N, 8.9×
  at 4× scale**, while costing ~4.6% top-10 overlap (bw4) that the end-to-end
  metric cannot see (+.0067 recall@10 vs exact). At N=36,920 numpy would need
  6.6 ms/query of pure CPU — a poor trade for zero storage savings.
- turbovec bw8 does not exist in the wheel (supports 2/3/4) — bw8 row skipped;
  bw2 degrades filtered matching badly (0.28), confirming **bw4 is the right
  default** (mettle-core's current default).
- Graph ANN (FAISS HNSW / hnswlib) matches the speed but costs **8× the
  memory of f32 raw** (vs turbovec's 7.4× SMALLER) and cannot delete ids.
- IVFPQ wins disk (8.2 MB) but loses .38 of exact overlap and is useless
  filtered (match .02) — and needs a train phase (breaks online ingest).
- turbovec's in-kernel allowlist (0.194 ms) is quantization-approximate
  (match .70 vs subset-exact) — the same approximation as its unfiltered
  search, no extra degradation; FAISS/HNSW need post-filter/overfetch at 5–28×
  its filtered latency to get exactness.

**Turbovec verdict: STAYS in mettle-core** — 7.4× compression with metric-
invisible recall loss, 10× faster than the exact reference with a widening
lead at scale, in-kernel allowlist filtering, O(1) deletes, no training phase
(online ingest), no dependency on faiss's weight or rebuild-on-delete model.

## 2. Component-residence table (contract §8, filled)

| Component | Resides today | Deciding number | Decision |
|---|---|---|---|
| Chunking / ingestion | docket | shared frozen substrate; settled | **stays in docket** |
| Embedding-model serving | mettled GPU, one engine for all surfaces | query embed p50 40.6 ms served; fidelity already gated (min_cos ≥ .9998); CUDA-verified this run | **stays in mettle** |
| Dense vector index | turbovec IdMapIndex bw4 in mettle-core | 0.555 ms p50 vs numpy 6.58 ms (11.8×), 7.4× smaller than f32, bw4 recall metric Δ +.0067 vs exact; graph ANN only ties speed at 8× memory and no deletes | **stays in mettle (turbovec bw4)** |
| Keyword/BM25 leg | FTS5 in mettle | FTS5-OR .487@10 vs docket bm25 .380@10 (AND-mode today: .000) | **stays in mettle with named fix: OR-mode FTS5 search mode** |
| Hybrid fusion | mettle (FTS5 ∩ dense) | hybrid == dense .787@10 / mbo .9109; rerank_hybrid .867 < rerank_exact .880 | **stays, but rerank pipelines should draw candidates from exact/dense; diversity (RRF-union) is the gap to docket's .913 winner** |
| Reranker weights | mettled GPU `/v1/rerank` | scores identical to docket (.880/.813); served call p50 12.5 s vs HF-local 2.45 s — per-pair decode by design | **stays in mettle; batching fix is a separate decision** |
| Orchestration / API | mettle `/v1` | consumer coupling exists | **stays in mettle** |

## 3. Deviations (also recorded in the output JSONs)

1. Search-arm latencies exclude query embedding (measured separately as
   `embed_query`); docket's per-arm latencies include ~44 ms query embed — add
   ~41 ms when comparing to docket's latency rows.
2. turbovec `bit_width=8` skipped: the 1.0.0 wheel supports 2/3/4 only (the
   contract asked for 2/4/8); bw3 exists in the wheel but was not contracted.
3. `sqlite_vec` skipped (optional per contract §5.7; not installable in this env).
4. mettle_bench.py's `score_hits` nests `@k_t` flags under `"hits"`; its own
   `summarize()` drops them in the v1 summary (latent bug, file unmodified per
   contract). best_config_bench digs them from `hits` — mettle v1's published
   summary is missing recall@k rows that the state file does contain.
5. All vectors L2-normalized for the IP engines (cosine ≡ IP; disclosed in
   provenance). Query vectors embedded once via `/v1/embeddings` and persisted
   (contract §2.8), shared with Part C.
6. Serve path was strictly GPU (verified): `mettled` CUDA build, every layer
   `dev = CUDA0`, KV + compute buffers on CUDA0, 2,796 MiB resident; Part C ran
   with mettled stopped (contract §5).
