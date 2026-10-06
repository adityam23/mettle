#!/usr/bin/env python
"""BEST_CONFIG_CONTRACT.md Part A + B: best-config arms + FTS5 keyword salvage.

Runs ONLY the winning embedder (qwen3-embedding-0.6b) arms on mettle and pairs
them against docket's v1 winner (rerank_hybrid(qwen3-rr) recall@10 = .913):

  exact                 brute-force cosine over the filing's frozen vectors
  dense                 POST /collections/{c}/search mode=dense k=40
  hybrid                POST /collections/{c}/search mode=hybrid k=40
  keyword_or            Part B — FTS5 OR-mode + bm25() read from metadata.db
  rerank_exact(qwen3-rr)  exact top-40 (embed_clean) -> POST /v1/rerank
  rerank_hybrid(qwen3-rr) hybrid top-40 (embed_clean) -> POST /v1/rerank

Rules (BEST_CONFIG_CONTRACT.md §2, binding):
- questions/gold/metric/chunks/scoping/candidates via parity_common verbatim;
  score_hits imported from mettle_bench.py (NOT modified).
- query vectors embedded ONCE via /v1/embeddings, persisted in state, reused
  everywhere (this arm set + turbovec_ablation.py Part C).
- resumable: state results/mettle_best_config_state.json, checkpoint after
  every question; rerank arms last (the ~65 min long pole).
- sanity gate (exit 3): chunk-0 self-search through each arm's own path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mettle_bench as mb  # noqa: E402  (score_hits/post/coll_name imported, mettle_bench NOT modified)
import parity_common as pc  # noqa: E402

BASE_DEFAULT = "http://127.0.0.1:11434/v1"
RESULTS = Path(__file__).with_name("results")
OUT_PATH = RESULTS / "mettle_best_config.json"
STATE_PATH = RESULTS / "mettle_best_config_state.json"

EMBEDDER = "qwen3-embedding-0.6b:latest"
RERANKER = "qwen3-reranker-0.6b:latest"

# docket v1 reference rows (BEST_CONFIG_CONTRACT.md §1)
DOCKET = {
    "exact": {"@3": 0.627, "@6": 0.727, "@10": 0.793, "@20_t0.8": 0.760, "mbo": 0.9106},
    "bm25": {"@3": 0.287, "@6": 0.333, "@10": 0.380, "@20_t0.8": 0.367, "mbo": 0.6503},
    "hybrid": {"@3": 0.413, "@6": 0.540, "@10": 0.667, "@20_t0.8": 0.680, "mbo": 0.8982},
    f"rerank_exact({RERANKER})": {"@3": 0.760, "@6": 0.840, "@10": 0.880, "@20_t0.8": 0.813, "mbo": 0.9106},
    f"rerank_hybrid({RERANKER})": {"@3": 0.767, "@6": 0.840, "@10": 0.913, "@20_t0.8": 0.807, "mbo": 0.8963},
}

ARMS = ("exact", "dense", "hybrid", "keyword_or",
        f"rerank_exact({RERANKER})", f"rerank_hybrid({RERANKER})")
CHEAP_ARMS = ("exact", "dense", "hybrid", "keyword_or")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"started": now(), "query_vectors": {}, "embed_query_ms": {},
            "collections": {}, "sanity": {}, "questions": {}, "deviations": []}


def save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=1))


def provenance() -> dict:
    prov = mb.provenance()
    prov.update({
        "state_file": str(STATE_PATH),
        "contract": "BEST_CONFIG_CONTRACT.md (2026-09-10)",
        "embedder": EMBEDDER, "reranker": RERANKER,
        "python_packages": python_packages(),
    })
    return prov


def python_packages() -> dict:
    out = {}
    try:  # turbovec pypi wheel version, if importable in this env
        import turbovec  # type: ignore
        out["turbovec"] = getattr(turbovec, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        out["turbovec"] = "not installed in this venv (collection index.bin built by mettle-core turbovec 0.8.1)"
    return out


# ---------------------------------------------------------------------------
# query vectors (embedded ONCE, persisted — contract §2.8)
# ---------------------------------------------------------------------------

def ensure_query_vectors(base: str, state: dict, records: list[dict]) -> None:
    todo = [r for r in records if r["financebench_id"] not in state["query_vectors"]]
    if not todo:
        print(f"[qvect] all {len(records)} query vectors already persisted", flush=True)
        return
    print(f"[qvect] embedding {len(todo)} queries ONCE via /v1/embeddings", flush=True)
    for i, rec in enumerate(todo):
        qid = rec["financebench_id"]
        t0 = time.perf_counter()
        vecs = mb.embed_texts(base, EMBEDDER, [rec["question"]])[0]
        state["embed_query_ms"][qid] = round((time.perf_counter() - t0) * 1000, 1)
        state["query_vectors"][qid] = vecs[0]
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            save_state(state)
            print(f"  [qvect {i+1}/{len(todo)}] {qid}", flush=True)


# ---------------------------------------------------------------------------
# collection verification (contract §3: verify counts, re-ingest on mismatch)
# ---------------------------------------------------------------------------

def verify_collections(state: dict, filings: list[str], base: str) -> None:
    embedder = EMBEDDER
    for doc in filings:
        corpus = pc.load_filing_corpus(embedder, doc)
        name = mb.coll_name(embedder, doc)
        db = filing_db(name)
        if not db.is_file():
            print(f"[verify] {name}: MISSING — re-ingesting", flush=True)
            rep = mb.ingest_filing(base, embedder, doc, corpus)
            state["collections"][name] = {"status": "reingested", **rep}
            continue
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            n = con.execute("select count(*) from chunks").fetchone()[0]
        if n != len(corpus):
            print(f"[verify] {name}: {n} stored != {len(corpus)} corpus — re-ingesting",
                  flush=True)
            rep = mb.ingest_filing(base, embedder, doc, corpus)
            state["collections"][name] = {"status": "reingested_mismatch",
                                          "stored": n, "corpus": len(corpus), **rep}
        else:
            state["collections"].setdefault(
                name, {"status": "verified", "chunks": n})
    save_state(state)
    bad = {k: v for k, v in state["collections"].items() if "reingest" in v["status"]}
    print(f"[verify] {len(state['collections'])} collections checked, "
          f"{len(bad)} re-ingested", flush=True)


# ---------------------------------------------------------------------------
# sanity gates (contract §9.5, exit 3 on failure)
# ---------------------------------------------------------------------------

def sanity_gate(base: str, state: dict, filings: list[str]) -> None:
    if state["sanity"].get("passed"):
        return
    embedder = EMBEDDER
    fails = []
    for doc in filings:
        corpus = pc.load_filing_corpus(embedder, doc)
        chunk0 = corpus[0]
        name = mb.coll_name(embedder, doc)
        q = pc.embed_clean(chunk0["text"])
        dense = mb.post(f"{base}/collections/{name}/search",
                        {"query": q, "k": 1, "mode": "dense"}, what=f"sanity {name} dense")
        if not dense or dense[0].get("document_id") != chunk0["chunk_id"]:
            fails.append(f"{doc}: dense self-search top-1 failed")
        hybrid = mb.post(f"{base}/collections/{name}/search",
                         {"query": q, "k": 10, "mode": "hybrid"}, what=f"sanity {name} hybrid")
        hids = [h.get("document_id") for h in hybrid]
        if chunk0["chunk_id"] not in hids:
            fails.append(f"{doc}: hybrid self-search missed top-10")
        kw = keyword_or_search(filing_db(name), chunk0["text"], 10)
        if chunk0["chunk_id"] not in [c for c, _ in kw]:
            fails.append(f"{doc}: keyword_or self-search missed top-10")
        if fails:
            break
        state["sanity"][doc] = True
        save_state(state)
    if fails:
        print("SANITY GATE FAILED (exit 3):", *fails, sep="\n  ", flush=True)
        sys.exit(3)
    state["sanity"]["passed"] = True
    state["sanity"]["note"] = ("dense = strict top-1; hybrid/keyword_or = chunk-0 "
                               "present in top-10 (top-1 not guaranteed for "
                               "rerank/OR paths)")
    save_state(state)
    print("[sanity] all filings pass self-search through their own arm path", flush=True)


# ---------------------------------------------------------------------------
# Part B — FTS5 OR-mode keyword salvage (read-only SQLite, contract §4)
# ---------------------------------------------------------------------------

def filing_db(name: str) -> Path:
    return mb.STATE_DIR / "collections" / name / "metadata.db"


def keyword_or_search(db_path: Path, question: str, k: int) -> list[tuple[str, float]]:
    """FTS5 OR-mode + bm25() ranking over one filing's collection DB.
    Returns [(chunk_id, bm25_rank)] best-first."""
    tokens = pc.tokenize(question)
    if not tokens:
        return []
    match = " OR ".join(tokens)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # chunks_fts.chunk_id is mettle's internal chunk id ("<doc>/chunk/<i>");
        # join back to chunks so we return the DOCUMENT id (== docket chunk_id).
        rows = con.execute(
            "SELECT c.document_id, bm25(chunks_fts) FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    return rows


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def run_arms(base: str, state: dict, filings: list[str],
             by_doc: dict[str, list[dict]], phases: set[str]) -> None:
    """Arm-aware resume: a question is skipped per-arm, so a rerank-only pass
    after the cheap pass fills only the rerank arms."""
    rerank_arms = [a for a in ARMS if a.startswith("rerank")]
    for doc in filings:
        recs = by_doc.get(doc, [])
        if not recs:
            continue
        todo = [r for r in recs
                if any((a in CHEAP_ARMS and "cheap" in phases)
                       or (a in rerank_arms and "rerank" in phases)
                       for a in ARMS if a not in state["questions"].get(r["financebench_id"], {}))]
        if not todo:
            print(f"[arms] {doc}: nothing pending", flush=True)
            continue
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        raw_text = {r["chunk_id"]: r["text"] for r in corpus}
        vecs = [r["vector"] for r in corpus]
        name = mb.coll_name(EMBEDDER, doc)
        db = filing_db(name)
        # per-filing warmup (excluded from latency): lazy paths
        mb.post(f"{base}/collections/{name}/search",
                {"query": todo[0]["question"], "k": 1, "mode": "dense"},
                what=f"warmsearch {name}")
        for rec in todo:
            qid = rec["financebench_id"]
            golds = pc.gold_evidence(rec)
            qv = state["query_vectors"][qid]
            row: dict = state["questions"].setdefault(qid, {"qid": qid})
            need_exact = "exact" not in row  # exact ids feed both rerank arms
            if need_exact:
                # exact — verbatim docket-equivalent ranking (pc.rank_by_query)
                t0 = time.perf_counter()
                ranked = pc.rank_by_query(vecs, qv)
                exact_ids = [corpus[i]["chunk_id"] for i in ranked][:pc.CANDIDATES]
                row["exact_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                row["exact"] = mb.score_hits(exact_ids, raw_text, golds)
            else:
                ranked = pc.rank_by_query(vecs, qv)
                exact_ids = [corpus[i]["chunk_id"] for i in ranked][:pc.CANDIDATES]
            if "cheap" in phases:
                # served dense + hybrid
                for mode in ("dense", "hybrid"):
                    if mode in row:
                        continue
                    t0 = time.perf_counter()
                    hits = mb.post(f"{base}/collections/{name}/search",
                                   {"query": rec["question"], "k": pc.CANDIDATES,
                                    "mode": mode}, what=f"search {name} {mode}")
                    row[f"{mode}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    ids = [h["document_id"] for h in hits]
                    row[mode] = mb.score_hits(ids, raw_text, golds)
                # Part B keyword_or
                if "keyword_or" not in row:
                    t0 = time.perf_counter()
                    kw = keyword_or_search(db, rec["question"], pc.CANDIDATES)
                    row["keyword_or_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    row["keyword_or"] = mb.score_hits([c for c, _ in kw], raw_text, golds)
            if "rerank" in phases:
                # rerank arms: hybrid top-40 + exact top-40 texts -> /v1/rerank
                if f"rerank_hybrid({RERANKER})" in row and f"rerank_exact({RERANKER})" in row:
                    continue
                t0 = time.perf_counter()
                hybrid_hits = mb.post(f"{base}/collections/{name}/search",
                                      {"query": rec["question"], "k": pc.CANDIDATES,
                                       "mode": "hybrid"}, what=f"search {name} hybrid (rerank cand)")
                cand_time = round((time.perf_counter() - t0) * 1000, 1)
                hybrid_ids = [h["document_id"] for h in hybrid_hits]
                for arm, cand_ids in ((f"rerank_hybrid({RERANKER})", hybrid_ids),
                                      (f"rerank_exact({RERANKER})", exact_ids)):
                    if arm in row:
                        continue
                    texts = [pc.embed_clean(raw_text.get(cid, "")) for cid in cand_ids]
                    order, secs = mb.rerank_top(base, RERANKER, rec["question"], texts)
                    s = mb.score_hits([cand_ids[i] for i in order], raw_text, golds)
                    s["rerank_secs"] = round(secs, 2)
                    s["candidate_search_ms"] = cand_time if "hybrid" in arm else row.get("exact_ms")
                    row[arm] = s
            save_state(state)
        print(f"[arms] {doc}: {len(todo)} questions "
              f"({'+'.join(sorted(p for p in phases if p != 'cheap')) or 'cheap'})",
              flush=True)




# ---------------------------------------------------------------------------
# aggregation (docket JSON shape — contract §6)
# ---------------------------------------------------------------------------

def agg(state: dict) -> dict:
    rows = list(state["questions"].values())
    arms: dict = {}
    for arm in ARMS:
        vals = [r[arm] for r in rows if arm in r]
        if not vals:
            continue
        # score_hits nests the @k_t map under "hits" (mettle_bench.py score_hits)
        flags = [k for k in vals[0]["hits"] if k.startswith("@")]
        out = {k: round(sum(1 for v in vals if v["hits"][k]) / len(vals), 4)
               for k in flags}
        out["mean_best_overlap"] = round(statistics.fmean(v["best_overlap"] for v in vals), 4)
        out["n"] = len(vals)
        arms[arm] = out
    lat: dict = {}
    for key in ("exact", "dense", "hybrid", "keyword_or"):
        samples = [r[f"{key}_ms"] / 1000 for r in rows if f"{key}_ms" in r]
        if samples:
            lat[key] = mb.lat_stats(samples)
    samples = [state["embed_query_ms"][qid] / 1000 for qid in state["embed_query_ms"]]
    if samples:
        lat["embed_query"] = mb.lat_stats(samples)
    for arm in ARMS:
        if not arm.startswith("rerank"):
            continue
        secs = [r[arm]["rerank_secs"] for r in rows if arm in r and "rerank_secs" in r[arm]]
        if secs:
            lat[f"rerank_call({RERANKER})"] = mb.lat_stats(secs)
            break
    return arms, lat


def summarize(state: dict) -> dict:
    arms, lat = agg(state)
    out = {"arms": arms, "latency": lat}
    kw = arms.get("keyword_or", {})
    if kw:
        met = kw.get("@10_t0.5", 0.0)
        dock = DOCKET["bm25"]["@10"]
        out["keyword_verdict"] = {
            "docket_bm25@10": dock, "mettle_fts5_or@10": met,
            "delta": round(dock - met, 4),
            "salvageable": (dock - met) <= 0.02,
            "rule": "recall@10 within 0.02 of docket bm25 (0.380) => salvageable",
        }
    return out


def print_report(res: dict) -> None:
    print(f"\n=== mettle best-config ({res['n_questions']} q / {res['n_filings']} filings) ===")
    for arm, vals in res["arms"].items():
        ref = DOCKET.get(arm.split("(")[0])
        d = f"  docket@10={ref['@10']}" if ref else ""
        print(f"{arm:44s} @10_t0.5={vals.get('@10_t0.5')} mbo={vals.get('mean_best_overlap')}{d}")
    for key, s in res["latency"].items():
        print(f"lat {key:42s} p50={s['p50_ms']}ms p95={s['p95_ms']}ms n={s['n']}")
    if "keyword_verdict" in res:
        v = res["keyword_verdict"]
        print(f"keyword verdict: mettle FTS5-OR @10={v['mettle_fts5_or@10']} vs docket bm25 "
              f"@10={v['docket_bm25@10']} -> salvageable={v['salvageable']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-filings", type=int, default=0)
    ap.add_argument("--phase", default="all",
                    choices=["all", "cheap", "rerank"])
    ap.add_argument("--smoke", action="store_true",
                    help="2 filings / 8 questions sanity run (separate state)")
    args = ap.parse_args()

    global STATE_PATH, OUT_PATH  # noqa: PLW0603
    if args.smoke:
        STATE_PATH = RESULTS / "mettle_best_config_state_smoke.json"
        OUT_PATH = RESULTS / "mettle_best_config_smoke.json"
        args.max_filings = args.max_filings or 2
        args.limit = args.limit or 8
        phases = {"cheap"}
    else:
        phases = {"cheap"} if args.phase == "cheap" else \
                 {"rerank"} if args.phase == "rerank" else {"cheap", "rerank"}

    RESULTS.mkdir(parents=True, exist_ok=True)
    state = load_state()
    base = args.base_url.rstrip("/")

    records = pc.load_questions(args.limit or None)
    filings_avail = set(pc.filing_list())
    recs = [r for r in records if r["doc_name"] in filings_avail and pc.gold_evidence(r)]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_doc[r["doc_name"]].append(r)
    filings = sorted(by_doc)
    if args.max_filings:
        filings = filings[: args.max_filings]
        by_doc = {d: by_doc[d] for d in filings}
    n_q = sum(len(v) for v in by_doc.values())
    print(f"{n_q} questions across {len(filings)} filings (embedder {EMBEDDER})", flush=True)

    state["provenance"] = provenance()
    ensure_query_vectors(base, state, [r for d in filings for r in by_doc[d]])
    verify_collections(state, filings, base)
    sanity_gate(base, state, filings)
    if phases & {"cheap"}:
        run_arms(base, state, filings, by_doc, {"cheap"})
    if phases & {"rerank"}:
        run_arms(base, state, filings, by_doc, {"rerank"})

    summary = summarize(state)
    res = {"generated": now(), "n_questions": n_q, "n_filings": len(filings),
           "embedder": EMBEDDER, **summary,
           "sanity": state.get("sanity"),
           "collections": state.get("collections"),
           "deviations": state.get("deviations", []),
           "provenance": state.get("provenance")}
    OUT_PATH.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT_PATH}")
    print_report(res)


if __name__ == "__main__":
    main()
