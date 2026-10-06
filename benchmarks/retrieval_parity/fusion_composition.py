"""Decompose a fusion cell's top-40: where does each candidate come from?

Motivated by an inconsistency between the RRF score arithmetic and the
`uniq%` column of diversity_probe.py: with dense weight 2.0 / keyword weight
0.5, a keyword-only candidate's RRF score (max 0.5/61 = 0.0082) cannot exceed
a dense candidate's at rank 39 (2.0/100 = 0.0200), so it should never enter
the fused top-40 — yet the probe reported 11.6% of the pool as "novel".

This prints, per cell and averaged over the replayed questions:
  dense<40    fused top-40 members that are in dense's true top-40
  dense40-79  fused top-40 members from dense ranks 40..79
  kw-only     fused top-40 members absent from dense's top-80 pool entirely
plus the score of the 40th fused candidate and the best keyword-only score, so
the entry mechanism is visible.

Run: cd benchmarks/retrieval_parity && uv run --with numpy python fusion_composition.py
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
CELLS = [(80, 80, 2.0, 0.5), (40, 40, 1.0, 1.0), (80, 15, 3.0, 1.0)]
KWCAP = 200


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

    stats = {c: {"dense<40": 0, "dense40+": 0, "kw-only": 0, "n": 0} for c in CELLS}
    for doc, doc_recs in by_doc.items():
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        raw_text = {r["chunk_id"]: r["text"] for r in corpus}
        vecs = np.array([r["vector"] for r in corpus], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        zero = norms[:, 0] == 0
        norms[zero] = 1.0
        vecn = vecs / norms
        db = Path(mb.STATE_DIR / "collections" / mb.coll_name(EMBEDDER, doc) / "metadata.db")
        for rec in doc_recs:
            qid = rec["financebench_id"]
            if qid not in state["questions"]:
                continue
            qv = np.array(state["query_vectors"][qid], dtype=np.float32)
            qv = qv / np.linalg.norm(qv)
            sims = vecn @ qv
            order = np.argsort(-sims)
            dense_ids = [corpus[i]["chunk_id"] for i in order]
            kw_ids = keyword_rank(db, rec["question"], KWCAP)
            for cell in CELLS:
                pool, kwp, wd, wk = cell
                fused: dict[str, float] = {}
                for rank, cid in enumerate(dense_ids[:pool]):
                    fused[cid] = fused.get(cid, 0.0) + wd / (60.0 + rank + 1)
                for rank, cid in enumerate(kw_ids[:kwp]):
                    fused[cid] = fused.get(cid, 0.0) + wk / (60.0 + rank + 1)
                merged = sorted(fused, key=lambda c: -fused[c])[:40]
                d40 = set(dense_ids[:40])
                d80 = set(dense_ids[:80])
                st = stats[cell]
                st["dense<40"] += sum(1 for c in merged if c in d40)
                st["dense40+"] += sum(1 for c in merged if c not in d40 and c in d80)
                st["kw-only"] += sum(1 for c in merged if c not in d80)
                st["n"] += 1
            _ = raw_text

    print(f"{'cell':20s} {'n':>4s} | {'dense<40':>9s} {'dense40+':>9s} {'kw-only':>8s}")
    for cell, st in sorted(stats.items()):
        if not st["n"]:
            continue
        print(f"p{cell[0]}_kw{cell[1]}_w{cell[2]:g}:{cell[3]:g}   {st['n']:4d} | "
              f"{st['dense<40'] / st['n']:9.1f} {st['dense40+'] / st['n']:9.1f} "
              f"{st['kw-only'] / st['n']:8.1f}")


if __name__ == "__main__":
    main()
