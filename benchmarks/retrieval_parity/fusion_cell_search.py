"""Grid search for a fusion cell that satisfies BOTH 1.2 goals offline.

Findings this script answers (2026-09-11):
* the served dense-weighted cell (80/80, 2:0.5) puts ZERO keyword-only
  candidates in the fused top-40 — its "novelty" was dense ranks 40-79
  leaking through a deeper dense pool (see fusion_composition.py);
* the equal-weight cell (40/40, 1:1) puts ~10 keyword-only candidates in the
  top-40 and drove rerank_hybrid to .9067, but its keyword leg also invades
  the top-10 and hybrid@10 craters to .72 (gate >= .787).

Mechanism to exploit: with dense weight 2.0, a keyword-only candidate at
keyword-rank 0 (score wk/61) stays below dense rank 9 (2/70 = .0286) while
exceeding dense rank 39 (2/100 = .02) as long as wk is in (1.22, 1.74). Those
keyword candidates then occupy ranks ~10-40 without displacing the dense
top-10 — pool diversity for the reranker with dense-quality top-10 recall.
Dual-list candidates additionally get both RRF terms, which can promote golds
into the top-10 relative to pure dense ordering.

Reports per cell: recall@10_t0.5, gold@40_t0.5 (gate predicate), and the
fused-top-40 composition (dense<40 / dense ranks 40-79 / keyword-only).

Run: cd benchmarks/retrieval_parity && uv run --with numpy python fusion_cell_search.py
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

# (dense pool, keyword pool, dense weight, keyword weight)
CELLS = [
    (80, 80, 2.0, 0.5),    # shipped: hybrid .8067 served, rerank .88 (0 kw-only)
    (40, 40, 1.0, 1.0),    # equal weight: rerank .9067, hybrid .72 (fail)
    (80, 40, 2.0, 0.6),
    (80, 40, 2.0, 0.7),
    (80, 40, 2.0, 0.8),
    (80, 40, 2.0, 0.9),
    (80, 40, 2.0, 1.0),
    (80, 40, 2.0, 1.1),
    (80, 80, 2.0, 0.7),
    (80, 80, 2.0, 0.9),
    (80, 80, 2.0, 1.0),
]


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

    stats = {c: {"rec": 0, "gold40": 0, "dense<40": 0, "dense40+": 0,
                 "kw-only": 0, "n": 0} for c in CELLS}
    for doc, doc_recs in by_doc.items():
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        raw_text = {r["chunk_id"]: r["text"] for r in corpus}
        vecs = np.array([r["vector"] for r in corpus], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms[:, 0] == 0] = 1.0
        vecn = vecs / norms
        db = Path(mb.STATE_DIR / "collections" / mb.coll_name(EMBEDDER, doc) / "metadata.db")
        for rec in doc_recs:
            qid = rec["financebench_id"]
            if qid not in state["questions"]:
                continue
            golds = pc.gold_evidence(rec)
            qv = np.array(state["query_vectors"][qid], dtype=np.float32)
            qv = qv / np.linalg.norm(qv)
            dense_ids = [corpus[i]["chunk_id"] for i in np.argsort(-(vecn @ qv))]
            kw_ids = keyword_rank(db, rec["question"], KWCAP)
            for cell in CELLS:
                pool, kwp, wd, wk = cell
                fused: dict[str, float] = {}
                for rank, cid in enumerate(dense_ids[:pool]):
                    fused[cid] = fused.get(cid, 0.0) + wd / (60.0 + rank + 1)
                for rank, cid in enumerate(kw_ids[:kwp]):
                    fused[cid] = fused.get(cid, 0.0) + wk / (60.0 + rank + 1)
                merged = sorted(fused, key=lambda c: -fused[c])
                s = mb.score_hits(merged, raw_text, golds, ks=(10, 40), thresholds=(0.5,))
                d40, d80 = set(dense_ids[:40]), set(dense_ids[:80])
                top40 = merged[:40]
                st = stats[cell]
                st["rec"] += int(s["hits"]["@10_t0.5"])
                st["gold40"] += int(s["hits"]["@40_t0.5"])
                st["dense<40"] += sum(1 for c in top40 if c in d40)
                st["dense40+"] += sum(1 for c in top40 if c not in d40 and c in d80)
                st["kw-only"] += sum(1 for c in top40 if c not in d80)
                st["n"] += 1

    print(f"{'cell':20s} {'rec@10':>7s} {'gold@40':>8s} | {'dense<40':>8s} "
          f"{'dense40+':>8s} {'kw-only':>7s}")
    for cell, st in sorted(stats.items(), key=lambda kv: -kv[1]["rec"] / max(kv[1]["n"], 1)):
        n = st["n"]
        if not n:
            continue
        print(f"p{cell[0]}_kw{cell[1]}_w{cell[2]:g}:{cell[3]:g}   "
              f"{st['rec'] / n:7.4f} {st['gold40'] / n:8.4f} | "
              f"{st['dense<40'] / n:8.1f} {st['dense40+'] / n:8.1f} {st['kw-only'] / n:7.1f}")


if __name__ == "__main__":
    main()
