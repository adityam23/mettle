#!/usr/bin/env python
"""Diversity diagnostics for fusion candidate cells (offline, no engine).

Companion to fusion_probe.py: for each candidate fusion cell, report
recall@10_t0.5 AND pool-level diversity metrics — the §1.2 objective is a
diverse candidate pool for the reranker, so a cell that wins recall by
collapsing to dense ordering is a cosmetic pass.

WARNING (2026-09-11): the `uniq%` column here counts fused top-40 members
absent from dense's TOP-40, which conflates two different things — genuinely
keyword-only candidates AND dense ranks 40..79 leaking in when the dense pool
is deeper than 40. It reported 11.6% for a cell whose pool in fact contained
ZERO keyword-only chunks, which is how a dense collapse nearly shipped.
Use fusion_composition.py (decomposes the pool by source: dense<40 /
dense ranks 40+ / keyword-only) for any decision; treat uniq% as a
sanity-only number.

gold-in-pool uses the gate's own predicate (mb.score_hits @40_t0.5 over gold
token sets), never set algebra on ids.

Run:  cd benchmarks/retrieval_parity && uv run --with numpy python diversity_probe.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mettle_bench as mb  # noqa: E402
import parity_common as pc  # noqa: E402

RESULTS = Path(__file__).with_name("results")
EMBEDDER = "qwen3-embedding-0.6b:latest"

# (dense pool, keyword pool, dense weight, keyword weight)
CELLS = [(80, 15, 3.0, 1.0), (120, 15, 3.0, 1.0), (80, 80, 2.0, 0.5),
         (40, 20, 2.0, 1.0), (80, 40, 2.0, 1.0), (40, 40, 1.0, 1.0)]


def keyword_rank(db_path: Path, question: str, k: int) -> list[str]:
    tokens = pc.tokenize(question)
    if not tokens:
        return []
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT c.document_id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (" OR ".join(tokens), k)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    return [r[0] for r in rows]


def main() -> None:
    state = json.loads((RESULTS / "mettle_best_config_state.json").read_text())
    records = pc.load_questions(None)
    avail = set(pc.filing_list())
    recs = [r for r in records if r["doc_name"] in avail and pc.gold_evidence(r)]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_doc[r["doc_name"]].append(r)

    stats = {c: {"rec": 0, "gold40": [], "gold40_dense": [], "uniq": [], "n": 0}
             for c in CELLS}
    for doc, doc_recs in by_doc.items():
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        raw_text = {r["chunk_id"]: r["text"] for r in corpus}
        vecs = np.array([r["vector"] for r in corpus], dtype=np.float32)
        vecn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        db = Path(mb.STATE_DIR / "collections" / mb.coll_name(EMBEDDER, doc) / "metadata.db")
        for rec in doc_recs:
            qid = rec["financebench_id"]
            if qid not in state["questions"]:
                continue
            golds = pc.gold_evidence(rec)
            qv = np.array(state["query_vectors"][qid], dtype=np.float32)
            qv = qv / np.linalg.norm(qv)
            dense_ids = [corpus[i]["chunk_id"] for i in np.argsort(-(vecn @ qv))]
            kw_ids = keyword_rank(db, rec["question"], 200)
            dense40 = set(dense_ids[:40])
            s_ref = mb.score_hits(dense_ids[:40], raw_text, golds, ks=(40,), thresholds=(0.5,))
            for cell in CELLS:
                pool, kwp, wd, wk = cell
                fused: dict[str, float] = {}
                for rank, cid in enumerate(dense_ids[:pool]):
                    fused[cid] = fused.get(cid, 0.0) + wd / (60.0 + rank + 1)
                for rank, cid in enumerate(kw_ids[:kwp]):
                    fused[cid] = fused.get(cid, 0.0) + wk / (60.0 + rank + 1)
                merged = sorted(fused, key=lambda c: -fused[c])[:40]
                s = mb.score_hits(merged, raw_text, golds, ks=(10, 40), thresholds=(0.5,))
                st = stats[cell]
                st["rec"] += int(s["hits"]["@10_t0.5"])
                st["gold40"].append(s["hits"]["@40_t0.5"])
                st["gold40_dense"].append(s_ref["hits"]["@40_t0.5"])
                st["uniq"].append(len(set(merged) - dense40) / 40)
                st["n"] += 1

    print(f"{'cell':22s} {'rec@10':>7s} {'n':>4s} | {'uniq%':>6s} "
          f"{'gold@40 fused':>13s} {'gold@40 dense':>13s}")
    for cell, st in sorted(stats.items()):
        print(f"p{cell[0]}_kw{cell[1]}_w{cell[2]:g}:{cell[3]:g}     "
              f"{st['rec'] / st['n']:.4f} {st['n']:4d} | "
              f"{100 * sum(st['uniq']) / st['n']:5.1f}% "
              f"{sum(st['gold40']) / st['n']:13.4f} {sum(st['gold40_dense']) / st['n']:13.4f}")


if __name__ == "__main__":
    main()
