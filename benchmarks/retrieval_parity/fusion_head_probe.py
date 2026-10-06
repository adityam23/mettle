"""Head-preserving fusion probe: keep the dense ordering at the top, gain pool
diversity in the tail.

Measured conflict this script addresses (2026-09-11, all offline unless noted):
* dense-weighted RRF (80/80, 2:0.5) -- SERVED hybrid .8067 (gate >= .787 PASS)
  but rerank pool has 0 keyword-only candidates; SERVED rerank_hybrid .880
  (gate >= .90 FAIL);
* equal-weight RRF (40/40, 1:1) -- SERVED rerank_hybrid .9067 (PASS) but
  SERVED hybrid .7200 (FAIL).
Additive RRF cannot do both: the keyword weight that lifts a keyword-only
candidate above dense's tail ALSO adds a second RRF term to dual-list chunks,
promoting them into the top-10 and displacing dense's head golds.

Recipe under test: leave positions 1..HEAD in dense order (so recall@10 equals
dense's own), and fill positions HEAD+1..40 by RRF over the remaining dense
pool and the keyword pool. The reranker then sees dense's head plus
keyword-only tail candidates.

Reports per variant: recall@10_t0.5, gold@40_t0.5, and the fused-top-40
composition (dense<40 / dense ranks 40-79 / keyword-only) so the pool
diversity delivered to the reranker is explicit.

Run: cd benchmarks/retrieval_parity && uv run --with numpy python fusion_head_probe.py
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

# (head, dense pool, keyword pool, dense weight, keyword weight)
VARIANTS = [
    (0, 40, 40, 1.0, 1.0),      # reference: the failing equal-weight cell
    (5, 40, 40, 1.0, 1.0),
    (8, 40, 40, 1.0, 1.0),
    (10, 40, 40, 1.0, 1.0),     # shipped option C
    (12, 40, 40, 1.0, 1.0),
    (10, 80, 40, 1.0, 1.0),
    (10, 80, 80, 1.0, 1.0),
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

    stats = {v: {"rec": 0, "gold40": 0, "dense<40": 0, "dense40+": 0,
                 "kw-only": 0, "n": 0} for v in VARIANTS}
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
            for var in VARIANTS:
                head, pool, kwp, wd, wk = var
                fused: dict[str, float] = {}
                for rank, cid in enumerate(dense_ids[:pool]):
                    fused[cid] = fused.get(cid, 0.0) + wd / (60.0 + rank + 1)
                for rank, cid in enumerate(kw_ids[:kwp]):
                    fused[cid] = fused.get(cid, 0.0) + wk / (60.0 + rank + 1)
                ranked = sorted(fused, key=lambda c: -fused[c])
                if head:
                    # dense head pinned, RRF ordering fills the rest
                    pinned = dense_ids[:head]
                    rest = [c for c in ranked if c not in set(pinned)]
                    merged = pinned + rest
                else:
                    merged = ranked
                s = mb.score_hits(merged, raw_text, golds, ks=(10, 40), thresholds=(0.5,))
                d40, d80 = set(dense_ids[:40]), set(dense_ids[:80])
                top40 = merged[:40]
                st = stats[var]
                st["rec"] += int(s["hits"]["@10_t0.5"])
                st["gold40"] += int(s["hits"]["@40_t0.5"])
                st["dense<40"] += sum(1 for c in top40 if c in d40)
                st["dense40+"] += sum(1 for c in top40 if c not in d40 and c in d80)
                st["kw-only"] += sum(1 for c in top40 if c not in d80)
                st["n"] += 1

    print(f"{'variant':24s} {'rec@10':>7s} {'gold@40':>8s} | {'dense<40':>8s} "
          f"{'dense40+':>8s} {'kw-only':>7s}")
    for var, st in sorted(stats.items(), key=lambda kv: -kv[1]["rec"] / max(kv[1]["n"], 1)):
        head, pool, kwp, wd, wk = var
        n = st["n"]
        print(f"h{head}_p{pool}_kw{kwp}_w{wd:g}:{wk:g}".ljust(24) +
              f" {st['rec'] / n:7.4f} {st['gold40'] / n:8.4f} | "
              f"{st['dense<40'] / n:8.1f} {st['dense40+'] / n:8.1f} {st['kw-only'] / n:7.1f}")


if __name__ == "__main__":
    main()
