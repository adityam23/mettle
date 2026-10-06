#!/usr/bin/env python
"""Rerank ρ probe: fixed (query, 40-doc) calls for the ADOPTION contract §1.3 gate.

Deterministic rerank probe set: 20 calls sampled with a fixed seed from the
bench workload (records whose doc has a collection + gold evidence), each pool
being the exact-cosine top-40 for that question built from the persisted query
vectors in mettle_best_config_state.json (no server calls needed to build a
pool, so inputs are byte-identical across captures).

Modes:
  capture (default) — call the CURRENT served binary, write probe inputs +
      per-doc relevance scores + call seconds to results/rerank_probe_baseline.json.
      MUST run against the pre-batching binary: no per-doc score vectors exist
      in any other results artifact (mettle_best_config.json is aggregates only).
  compare — re-run the same calls, compute per-call Spearman ρ of score
      ordering vs the baseline, mean ρ, and served p50 latency (§1.3 gates:
      p50 ≤ 4 s per 40-doc call, ρ ≥ 0.99).

Run from benchmarks/retrieval_parity/:  uv run python rerank_probe.py [compare]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mettle_bench as mb  # noqa: E402
import parity_common as pc  # noqa: E402

BASE_DEFAULT = "http://127.0.0.1:11434/v1"
RESULTS = Path(__file__).with_name("results")
BASELINE_PATH = RESULTS / "rerank_probe_baseline.json"
STATE_PATH = RESULTS / "mettle_best_config_state.json"

SEED = 20260910
N_CALLS = 20
EMBEDDER = "qwen3-embedding-0.6b:latest"
RERANKER = "qwen3-reranker-0.6b:latest"
POOL_SIZES = [40, 40, 40, 40, 40, 7, 40, 40, 20, 40,
              13, 40, 5, 40, 40, 3, 40, 40, 40, 12]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_calls() -> list[dict]:
    """Deterministic probe set: 20 (filing, question, top-N texts) triples.

    Pool sizes follow POOL_SIZES in sample order: mostly full 40-doc calls,
    with deliberately uneven ones (3/5/7/12/13/20) — uniform 40-doc pools
    never exercise the uneven tails where micro-batch padding/bucketing is
    most likely to shift score order (§1.3 accepts order preservation, not
    bit-exact logits)."""
    state = json.loads(STATE_PATH.read_text())
    qv = state["query_vectors"]
    filings_avail = set(pc.filing_list())
    records = [r for r in pc.load_questions()
               if r["doc_name"] in filings_avail and pc.gold_evidence(r)
               and r["financebench_id"] in qv]
    rng = random.Random(SEED)
    sampled = rng.sample(records, N_CALLS)

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in sampled:
        by_doc[r["doc_name"]].append(r)

    calls: list[dict] = []
    for i, rec in enumerate(sampled):
        corpus = pc.load_filing_corpus(EMBEDDER, rec["doc_name"])
        raw_text = {r["chunk_id"]: r["text"] for r in corpus}
        vecs = [r["vector"] for r in corpus]
        ranked = pc.rank_by_query(vecs, qv[rec["financebench_id"]])
        n = min(POOL_SIZES[i], len(corpus))
        cand = [corpus[j]["chunk_id"] for j in ranked][:n]
        texts = [pc.embed_clean(raw_text.get(cid, "")) for cid in cand]
        calls.append({"qid": rec["financebench_id"], "filing": rec["doc_name"],
                      "query": rec["question"], "texts": texts})
    return calls

def run_calls(base: str, calls: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(calls):
        t0 = time.perf_counter()
        resp = mb.post(f"{base}/rerank", {"model": RERANKER, "query": c["query"],
                                          "documents": c["texts"]},
                       what=f"probe rerank {i}")
        secs = time.perf_counter() - t0
        # results are score-descending, stable (openai.rs:516); no top_n sent,
        # so exactly len(texts) entries come back.
        by_index = {r["index"]: r["relevance_score"] for r in resp["results"]}
        if len(by_index) != len(c["texts"]):
            raise RuntimeError(f"probe {i}: {len(by_index)} scores for "
                               f"{len(c['texts'])} docs")
        out.append({"qid": c["qid"], "filing": c["filing"], "secs": round(secs, 3),
                    "order": [r["index"] for r in resp["results"]],
                    "scores": [by_index[i] for i in range(len(c["texts"]))]})
        print(f"  [probe {i + 1}/{len(calls)}] {c['qid']} {c['filing']} {secs:.2f}s",
              flush=True)
    return out


def provenance() -> dict:
    prov = mb.provenance()
    prov["probe_seed"] = SEED
    prov["n_calls"] = N_CALLS
    prov["reranker"] = RERANKER
    prov["pool_recipe"] = ("exact-cosine top-40 from persisted query vectors "
                           "(mettle_best_config_state.json), embed_clean texts")
    return prov


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman ρ over raw scores (values only matter through rank)."""
    n = len(a)
    ra = [0.0] * n
    rb = [0.0] * n
    for scores, ranks in ((a, ra), (b, rb)):
        order = sorted(range(n), key=lambda i: scores[i])
        for pos, i in enumerate(order):
            ranks[i] = pos + 1
    da = [(x - 1 - (n - 1) / 2) for x in ra]
    db = [(x - 1 - (n - 1) / 2) for x in rb]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return num / den if den else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE_DEFAULT)
    ap.add_argument("mode", nargs="?", default="capture",
                    choices=["capture", "compare"])
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    if args.mode == "compare":
        # Replay the BASELINE's exact calls verbatim; never re-derive pools.
        # mettle_best_config_state.json is a live artifact the gate run
        # rewrites — re-deriving would silently change the inputs and the
        # rho number would look valid while measuring something else.
        baseline = json.loads(BASELINE_PATH.read_text())
        calls = baseline["calls"]
    else:
        calls = build_calls()
    print(f"{len(calls)} probe calls (seed {SEED}, {RERANKER})", flush=True)

    if args.mode == "capture":
        results = run_calls(base, calls)
        doc = {"generated": now(), "calls": calls, "results": results,
               "provenance": provenance()}
        RESULTS.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(doc, indent=1))
        secs = sorted(r["secs"] for r in results)
        print(f"wrote {BASELINE_PATH}; p50={secs[len(secs) // 2]:.2f}s "
              f"max={secs[-1]:.2f}s")
        return

    base_by_qid = {r["qid"]: r for r in baseline["results"]}
    if len(calls) != len(base_by_qid):
        raise RuntimeError(
            f"baseline drift: {len(calls)} calls vs {len(base_by_qid)} baseline results")
    results = run_calls(base, calls)
    rhos = [spearman(base_by_qid[r["qid"]]["scores"], r["scores"]) for r in results]
    secs = sorted(r["secs"] for r in results)
    p50 = secs[len(secs) // 2]
    base_secs = sorted(r["secs"] for r in baseline["results"])
    base_p50 = base_secs[len(base_secs) // 2]
    mean_rho = sum(rhos) / len(rhos)
    # Gate per contract amendment 2026-09-10: batched p50 <= serial baseline
    # p50 AND score ordering mean rho >= 0.99 (the original 4 s target was
    # measured unreachable in this engine; see the contract's §1.3 note).
    ok = p50 <= base_p50 and mean_rho >= 0.99
    print(f"\np50={p50:.2f}s (gate <= baseline {base_p50:.2f}s)  "
          f"p95={secs[int(len(secs) * .95)]:.2f}s  "
          f"mean rho={mean_rho:.6f} (gate >= 0.99)  "
          f"min rho={min(rhos):.6f}")
    print("GATE", "PASS" if ok else "FAIL")
    out = {"generated": now(), "p50_s": p50, "baseline_p50_s": base_p50,
           "mean_rho": mean_rho,
           "per_call_rho": {r["qid"]: rho for r, rho in zip(results, rhos)},
           "gate_pass": ok}
    (RESULTS / "rerank_probe_compare.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
