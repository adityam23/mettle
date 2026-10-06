# Mettle best-config parity + turbovec ablation contract (2026-09-10)

**Question being answered.** The first parity round (2026-09-04 mettle side,
2026-09-10 docket side, both complete) found docket's winning configuration:
**qwen3-embedding-0.6b dense/hybrid + qwen3-reranker-0.6b rerank**, reaching
**recall@10 = 0.913** (docket `rerank_hybrid`). Three decisions remain, all to
be settled by measurement, not argument:

1. **Does mettle match docket's winning config?** Run the identical
   qwen3-embedder + qwen3-reranker pipeline natively on mettle
   (`dense/hybrid` collections + `/v1/rerank` over hybrid top-40) and pair it
   against docket's winner.
2. **Is mettle's FTS5 keyword leg salvageable?** The first run measured
   keyword = 0.000 at every k (FTS5 AND-expansion of long natural questions).
   Test whether FTS5 in OR-mode with `bm25()` ranking approaches docket's
   `rank_bm25` arm. If yes, the keyword leg stays in mettle (fix later); if
   no, it moves to docket.
3. **What does turbovec actually provide** over exact brute force and over
   the standard ANN alternatives (FAISS, hnswlib, sqlite-vec)? Measure recall
   vs exact, latency, memory, ingest, deletes, and in-kernel allowlist
   filtering on the real 36,920-vector corpus. This decides whether the
   turbovec dependency stays in mettle-core.

**Authority.** Written by the mettle-side agent (2026-09-10) after the first
parity cycle. The executing agent MUST follow §2 (equivalence rules) and §5
(environment rules) to the letter; every deviation MUST be disclosed in the
output JSON's `deviations` field. Docket code is READ-ONLY. No mettle server
code changes in this contract — measurement only.

---

## 1. What is already measured (do not re-run)

- **Mettle v1** (2026-09-04, `results/mettle_results.json` + per-question
  rows in `results/mettle_state.json`): all arms for 3 embedders × 2
  rerankers, CUDA build, 150 q / 84 filings / 36,920 chunks. Key rows:
  qwen3-emb rerank_dense(qwen3-rr) = .747/.827/.867@10/.813@20t8.
- **Docket v1** (2026-09-10,
  `/personal-projects/docket/benchmarks/financebench/results/docket_parity.json`):
  the reference numbers for pairing. Embedder-agnostic arms (bm25) are shared
  across embedders (docket indexes embed_clean text once).

**Docket targets to pair against (recall@k, t0.5, @20 at t0.8):**

| docket arm (qwen3-emb unless noted) | @3 | @6 | @10 | @20_t0.8 | mbo |
|---|---|---|---|---|---|
| exact | .627 | .727 | .793 | .760 | .9106 |
| bm25 (all embedders) | .287 | .333 | .380 | .367 | .6503 |
| hybrid | .413 | .540 | .667 | .680 | .8982 |
| rerank_exact(qwen3-rr) | .760 | .840 | .880 | .813 | .9106 |
| **rerank_hybrid(qwen3-rr) — the winner** | **.767** | **.840** | **.913** | **.807** | .8963 |

Docket latency (per query, p50): exact 88.3 ms (incl. 44.0 ms query embed);
bm25 3.0 ms; hybrid 91.5 ms. Served rerank call (≤40 docs, from mettle):
qwen3-rr p50 12.7 s.

## 2. Equivalence rules (binding, inherited from METTLE_PARITY_CONTRACT.md)

1. **Questions + gold truth**: `benchmarks/financebench/data/financebench_merged.jsonl`
   (150 rows). Gold = `evidence[].evidence_text` token sets, overlap
   coefficient `|a∩b|/min(|a|,|b|)`, hit iff ≥ threshold. Use
   `parity_common.py` verbatim (`gold_evidence`, `overlap_coeff`,
   `recall_hit`, `score_hits` from `mettle_bench.py`).
2. **Chunks**: frozen `.sweep_cache/<embedder-slug>/<DOC>/corpus.jsonl`
   (36,920 chunks / 84 filings). Scoring on RAW `text`; mettle-stored/embed
   text = `embed_clean(text)` (cap 2000).
3. **Metric**: recall@k for k ∈ {3,5,6,10,20} × t ∈ {0.5,0.8} +
   `mean_best_overlap`. Report ALL k/threshold pairs.
4. **Scoping**: per-filing retrieval (question retrieves within its own
   filing), except where an arm is explicitly global.
5. **Candidates**: rerank arms rerank the top-40 (`parity_common.CANDIDATES`).
6. **Timing**: query latency = wall time question → ranked hits, including
   query embedding. Rerank latency = wall time of the `/v1/rerank` call.
7. **Models**: only `qwen3-embedding-0.6b:latest` (1024-d) and
   `qwen3-reranker-0.6b:latest` (P(yes)) this round — the winning pair.
   bge arms are NOT re-run (v1 numbers exist).
8. **Vectors**: frozen sweep-cache vectors only. Mettle's fidelity gate
   (min_cos ≥ 0.9998 vs current binary) is already verified — do not re-embed
   the corpus. Query vectors: embed each of the 150 questions ONCE via
   `POST /v1/embeddings` (qwen3-emb), persist into the state file, and reuse
   everywhere (Part A exact arm + Part C ablation queries).
9. **Provenance**: record mettle git head + dirty flag, mettled binary mtime,
   GGUF file sizes, host, python package versions (turbovec, faiss, hnswlib),
   and the state-file path in every output JSON.

## 3. Part A — best-config arms on mettle (mettled required)

New script: `benchmarks/retrieval_parity/best_config_bench.py`.
Self-contained, resumable (state `results/mettle_best_config_state.json`,
per-question checkpoint after each arm, 503-wait + retry helpers copied from
`mettle_bench.py`). Do NOT modify `mettle_bench.py`.

Reuse the existing per-filing collections in
`.state/collections/fbpar-qwen3-embedding-0-6b-latest-<doc>` (ingested by the
v1 run; same name scheme). Verify one collection's document count against
`load_filing_corpus` before trusting it; re-ingest only on mismatch (and
disclose).

Arms, per question (150), embedder = qwen3-embedding-0.6b:

| arm | what | pairs against (docket) |
|---|---|---|
| `exact` | brute-force cosine over the filing's frozen vectors, top-40 (`pc.rank_by_query`) | docket `exact` — expect ±0.005 (larger gap = bug, investigate first) |
| `dense` | `POST /collections/{c}/search mode=dense k=40` | mettle v1 dense; sanity ≈ exact |
| `hybrid` | `POST /collections/{c}/search mode=hybrid k=40` | mettle v1 hybrid; expect ≈ dense |
| `rerank_exact(qwen3-rr)` | exact top-40 texts (embed_clean) → `POST /v1/rerank` | docket `rerank_exact(qwen3-rr)` |
| `rerank_hybrid(qwen3-rr)` | **hybrid** top-40 texts → `POST /v1/rerank` | docket `rerank_hybrid(qwen3-rr)` — the head-to-head |
| `keyword_or` | §4 salvage test | docket `bm25` |

Record per-question: arm hits (all k/t), mean_best_overlap, `*_ms` latencies,
`rerank_secs`. Aggregate to the docket JSON's shape (`arms`, `latency` with
n/mean_ms/p50_ms/p95_ms).

**Estimated rerank cost**: qwen3-rr ≈ 13 s per 40-doc call →
150 × 2 arms × 13 s ≈ 65 min. Run rerank arms last so the cheaper arms land
first, and checkpoint after every question.

## 4. Part B — FTS5 keyword-leg salvage test (same script, no server change)

Read-only SQLite over one of the existing per-filing collection DBs
(`.state/collections/<coll>/metadata.db`, FTS5 table). For each question:

1. tokens = `re.findall(r"[a-z0-9]+", question.lower())` (same tokenization
   as the gold metric);
2. `SELECT ... FROM <fts> WHERE <fts> MATCH 't1 OR t2 OR ...' ORDER BY bm25(<fts>)`
   — all tokens, OR-joined, no sanitization beyond lowercasing alnum;
3. map rowids back to chunk ids via the `chunks` table, take top-40, score
   with the parity metric on RAW text.

Disclose: FTS5's default unicode61 tokenizer vs the metric's `[a-z0-9]+`
regex (underscore/unicode edge cases), and the 2000-char embed_clean cap on
indexed text.

**Decision rule** (write it into the output JSON's `keyword_verdict` field):
recall@10 within 0.02 of docket's bm25 (0.380) → FTS5 leg salvageable, stays
in mettle pending an OR-mode search mode; 0.380 − recall@10 > 0.02 → declare
the keyword leg unsalvageable on FTS5, keyword moves to docket.

## 5. Part C — turbovec ablation (mettled needed only once, for query vectors)

New script: `benchmarks/retrieval_parity/turbovec_ablation.py`.
State: `results/turbovec_ablation_state.json`. After the 150 query vectors
are embedded and persisted, the rest must run WITHOUT mettled (pure Python
engines on the same vectors).

**Data**: qwen3-emb vectors from the sweep cache: 36,920 × 1024 f32, plus the
150 persisted query vectors, plus each filing's chunk-id list.

**Engines** (each a row in the output; versions recorded):

1. `exact_numpy` — f32 matmul + argsort (reference; also docket's Corpus
   equivalent).
2. `turbovec_idmap_bw4` — turbovec `IdMapIndex(dim=1024, bit_width=4)`
   (mettle-core default). Also rows for `bw2` and `bw8`.
3. `faiss_flat_ip` — `IndexFlatIP` (exact f32).
4. `faiss_hnsw` — `IndexHNSWFlat` (M=32, efConstruction=200, efSearch=64),
   inner-product via metric rescale or `IndexHNSWFlat` on L2 with
   normalized vectors (disclose choice).
5. `faiss_ivfpq` — trained compressed ANN (nlist ≈ 4·√N, PQ m=256/4-bit code
   to match turbovec's ~4-bit bitrate), train on a 20% sample; disclose
   training time and its effect on online-ingest semantics.
6. `hnswlib` — ip space, M=32, ef=64.
7. `sqlite_vec` — OPTIONAL: only if `uv run --with sqlite-vec` installs
   cleanly; otherwise skip with a stated reason (per the repo dependency
   rule: a heavy dep adopted for a single measurement is a user decision).

**Measurements per engine** at N = 36,920 (global stack of all filings):

- **Ingest**: wall time to add all 36,920 vectors, vectors/s; note any train
  phase (IVF/PQ) separately.
- **Memory**: index bytes on disk (after save) + peak RSS delta during build
  (`resource.getrusage(RUSAGE_SELF).ru_maxrss`), f32 raw bytes as reference
  (151,224,320 B).
- **Unfiltered query latency**: 150 queries, k=10, p50/p95; disclose thread
  config (run one extra single-thread pass for `exact_numpy` and
  `faiss_flat_ip` vs `turbovec`).
- **Recall vs exact**: (a) top-10 id overlap with `exact_numpy`'s top-10,
  mean over queries; (b) end-to-end recall@k on the real 150-question metric
  (unfiltered global: each question retrieves globally, k=10; report the
  recall@10 delta vs the exact arm run through the same scoring).
- **Filtered search (the allowlist question)**: for each question, allowlist
  = its own filing's chunk ids (≈440 ids, ~1.2% selectivity) → retrieve
  k=10 within the allowlist. Reference = brute force over the allowed
  subset. For each engine: latency + "filtered correctness" = fraction of
  queries whose top-10 (by the engine) equals top-10 of brute-force-over-
  allowed-subset. turbovec uses `search_with_allowlist`; FAISS flat uses
  post-filter or `IDSelector`; HNSW uses overfetch (ef×4, re-query if
  short) — disclose each engine's strategy and whether its output is
  guaranteed top-k of the allowed set.
- **Deletes**: turbovec: time 1,000 `remove()` calls + verify search still
  correct. FAISS/HNSW: report "no id-removal" (rebuild required) as a fact.
- **Scale probe (synthetic, disclose)**: replicate the 36,920 vectors ×4
  with offset ids (147,680 rows) — latency trend ONLY, no recall claims —
  for `exact_numpy`, `turbovec_bw4`, `faiss_flat_ip`, `faiss_hnsw`.

Dependencies are ad-hoc (`uv run --with turbovec --with faiss-cpu --with
hnswlib python ...`), NOT added to any project manifest — this is a
measurement, not a dependency adoption.

## 6. Output contract

```jsonc
// results/mettle_best_config.json
{
  "generated": "<iso8601>",
  "n_questions": 150, "n_filings": 84,
  "embedder": "qwen3-embedding-0.6b:latest",
  "arms": {
    "exact":            {"@3_t0.5": 0.0, "@20_t0.8": 0.0, "mean_best_overlap": 0.0},
    "dense": {...}, "hybrid": {...}, "keyword_or": {...},
    "rerank_exact(qwen3-reranker-0.6b:latest)": {...},
    "rerank_hybrid(qwen3-reranker-0.6b:latest)": {...}
  },
  "latency": {
    "exact": {"n":0,"mean_ms":0,"p50_ms":0,"p95_ms":0}, "dense": {...},
    "hybrid": {...}, "keyword_or": {...},
    "embed_query": {...},
    "rerank_call(qwen3-reranker-0.6b:latest)": {...}
  },
  "keyword_verdict": {"docket_bm25@10": 0.38, "mettle_fts5_or@10": 0.0,
                       "delta": 0.0, "salvageable": true},
  "deviations": [], "provenance": {...}
}

// results/turbovec_ablation.json
{
  "generated": "<iso8601>",
  "n_vectors": 36920, "dim": 1024,
  "query_vectors": "persisted in state, embedded via /v1/embeddings once",
  "engines": {
    "<engine>": {
      "ingest": {"wall_s": 0.0, "vecs_per_s": 0.0, "train_s": 0.0},
      "memory": {"index_bytes": 0, "peak_rss_bytes": 0},
      "unfiltered": {"p50_ms": 0.0, "p95_ms": 0.0, "top10_overlap_vs_exact": 0.0,
                      "recall10_metric_delta_vs_exact": 0.0},
      "filtered_1pct": {"p50_ms": 0.0, "top10_match_vs_subset_exact": 0.0,
                         "strategy": "in-kernel|post-filter|overfetch"},
      "filtered_5pct": {...}, "filtered_48pct": {...},
      "deletes": {"supported": true, "ms_per_1000": 0.0, "post_delete_ok": true},
      "scale_4x_p50_ms": 0.0
    }
  },
  "turbovec_verdict": "<filled by the verdict, see §7>",
  "deviations": [], "provenance": {...}
}
```

## 7. Deliverable + verdict

`results/mettle_best_config.json`, `results/turbovec_ablation.json`, plus a
short `results/BEST_CONFIG_VERDICT.md` that fills the component-residence
table (§8) with the measured numbers and states, per component:
**stays in mettle / moves to docket / stays with a named fix** — each with
its deciding number. stdout summary mirrors the docket contract's.

## 8. Component-residence decision frame (fill from measurements)

| Component | Resides today | Evidence so far | What this contract decides |
|---|---|---|---|
| Chunking / document ingestion | docket | frozen corpus is the shared substrate of both systems | settled — stays in docket |
| Embedding-model serving | mettled, GPU, one engine for all surfaces | fidelity min_cos ≥ 0.9998; both sides consumed the same served models | settled — stays in mettle |
| Dense vector index | turbovec `IdMapIndex` (4-bit) inside mettle-core | 4-bit = 7.5× smaller than f32; ANN recall ≈ exact (v1); but exact numpy p50 27.6 ms beat served dense 42.9 ms at 36,920 | does turbovec beat numpy/FAISS on THIS corpus? at what corpus size does 16× compression start to matter? is in-kernel allowlist measurably better than post-filter/overfetch? |
| Keyword/BM25 leg | FTS5 in mettle (SQLite) | AND-expansion → 0.000 recall; docket rank_bm25 = .380@10 | salvageable via OR-mode (Part B), or move keyword to docket |
| Hybrid fusion | mettle (FTS5 candidates → dense; falls back to dense) | mettle .833@10 vs docket RRF .613@10 | confirm hybrid ≈ dense on qwen3-emb; fusion stays in mettle |
| Reranker weights | mettled GPU (`/v1/rerank`) | scores ρ=0.997 vs HF; latency 12.9 s served vs 2.45 s HF-local (per-pair decode, by design) | record served latency as-is; a mettle-side batching fix is a SEPARATE decision, out of scope here |
| Orchestration / API | mettle `/v1` (agent-assistant consumes it) | consumer coupling already exists | settled — stays in mettle |

## 9. Environment

1. mettled (Part A + query-vector embedding only):
   ```bash
   cd /personal-projects/mettle
   METTLE_CONFIG=/personal-projects/mettle/benchmarks/retrieval_parity/mettle-bench-profile.toml \
     target/release/mettled &
   curl -X PUT http://127.0.0.1:11434/engine/profiles/active \
     -H 'Content-Type: application/json' -d '{"name":"bench-retrieval"}'
   ```
   n_ctx=8192 profile — do not raise it (profile manager drops models).
2. Python STRICTLY via uv: `uv run python ...` from
   `benchmarks/retrieval_parity/` (its `.venv`/`uv.lock`). NEVER bare
   `python`/`python3`. Ad-hoc deps via `--with`, never added to manifests.
3. Docket repo READ-ONLY: sweep cache and questions file are inputs only.
4. Scripts must be resumable: state file per part, checkpoint after every
   question (Part A) and every engine × phase (Part C). Rerank arms are the
   long pole (~65 min) — checkpoint them granularly.
5. Sanity gate before any recall number: self-search (chunk's own text must
   return itself top-1 through the arm's own path — `mettle_bench.py` does
   this for ingest; replicate for dense/hybrid/keyword_or). Exit 3 on
   failure (matches the v1 wrapper convention).
