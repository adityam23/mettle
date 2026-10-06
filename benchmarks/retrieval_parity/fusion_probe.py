#!/usr/bin/env python
"""Offline hybrid-fusion grid probe — diagnosis for the §1.2 gate failure.

The served RRF-union (k=60, equal-weight keyword∪dense top-40 pools) measured
hybrid @10_t0.5 = .72 vs the ≥.787 gate (old collapsing hybrid = .7867). This
probe replays the same data offline (persisted query vectors, frozen sweep-cache
corpus vectors, collection FTS5 DBs — no mettled, no models) and measures
recall@10_t0.5 for fusion variants, so the minimal recipe change is picked from
evidence, not taste.

Run:  cd benchmarks/retrieval_parity && uv run python fusion_probe.py
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
KWCAP = 200


def keyword_rank(db_path: Path, question: str, k: int) -> list[str]:
    tokens = pc.tokenize(question)
    if not tokens:
        return []
    match = " OR ".join(tokens)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT c.document_id FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    return [r[0] for r in rows]


def main() -> None:
    state = json.loads((RESULTS / "mettle_best_config_state.json").read_text())
    questions = state["questions"]

    records = pc.load_questions(None)
    filings_avail = set(pc.filing_list())
    recs = [r for r in records
            if r["doc_name"] in filings_avail and pc.gold_evidence(r)]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_doc[r["doc_name"]].append(r)

    variants: dict[str, dict[str, int]] = {}
    n = 0
    for doc, doc_recs in by_doc.items():
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        raw_text = {r["chunk_id"]: r["text"] for r in corpus}
        vecs = np.array([r["vector"] for r in corpus], dtype=np.float32)
        vecn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        db = Path(mb.STATE_DIR / "collections" / mb.coll_name(EMBEDDER, doc) / "metadata.db")
        for rec in doc_recs:
            qid = rec["financebench_id"]
            row = questions.get(qid)
            if row is None or "dense" not in row:
                continue
            golds = pc.gold_evidence(rec)
            qv = np.array(state["query_vectors"][qid], dtype=np.float32)
            qv = qv / np.linalg.norm(qv)
            sims = vecn @ qv
            dense_ids = [corpus[i]["chunk_id"] for i in np.argsort(-sims)]
            kw_ids = keyword_rank(db, rec["question"], KWCAP)

            for arm, ids in (("dense", dense_ids), ("keyword_or", kw_ids)):
                s = mb.score_hits(ids, raw_text, golds)
                key = f"check/{arm}"
                variants.setdefault(key, {"@10": 0, "n": 0})
                variants[key]["@10"] += int(s["hits"]["@10_t0.5"])
                variants[key]["n"] += 1

            for pool in (40, 80, 120):
                for kw_pool in (pool, 15, 25, pool // 2):
                    kw_list = kw_ids[:kw_pool]
                    for wd, wk in ((1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0),
                                   (1.0, 0.5), (1.0, 0.25), (2.0, 0.5)):
                        fused: dict[str, float] = {}
                        for rank, cid in enumerate(dense_ids[:pool]):
                            fused[cid] = fused.get(cid, 0.0) + wd / (60.0 + rank + 1)
                        for rank, cid in enumerate(kw_list):
                            fused[cid] = fused.get(cid, 0.0) + wk / (60.0 + rank + 1)
                        merged = sorted(fused, key=lambda c: -fused[c])
                        s = mb.score_hits(merged, raw_text, golds)
                        key = f"pool{pool}_kw{kw_pool}_wd{wd:g}_wk{wk:g}"
                        variants.setdefault(key, {"@10": 0, "n": 0})
                        variants[key]["@10"] += int(s["hits"]["@10_t0.5"])
                        variants[key]["n"] += 1
            n += 1
            if n % 25 == 0:
                print(f"[probe {n}] questions replayed", flush=True)

    print(f"\n=== fusion grid ({n} questions) — recall@10_t0.5 ===")
    rows = sorted(variants.items(), key=lambda kv: -kv[1]["@10"] / kv[1]["n"])
    for key, v in rows:
        print(f"{key:32s} @10={v['@10'] / v['n']:.4f}  ({v['@10']}/{v['n']})")


if __name__ == "__main__":
    main()
