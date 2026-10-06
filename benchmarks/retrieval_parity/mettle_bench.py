#!/usr/bin/env python
"""Mettle side of the mettle-vs-docket retrieval parity benchmark.

Measures mettle's native retrieval stack — turbovec ANN dense index, SQLite
FTS5 keyword, hybrid mode, /v1/rerank — on the REAL FinanceBench data (150
questions, 84 filings, 36,920 frozen docket chunks), with the SAME metric as
docket's sweep_recall.py (gold-evidence token-overlap recall@k). The docket
side is specified by METTLE_PARITY_CONTRACT.md in the docket repo.

Arms (per embedder unless noted):
  exact        local brute-force cosine over the cached mettle-produced vectors
               (= what docket's Corpus.dense_search does; the ANN reference)
  dense        mettle collection /search mode=dense (turbovec 4-bit ANN)
  keyword      mettle collection /search mode=keyword (SQLite FTS5 BM25 order)
  hybrid       mettle collection /search mode=hybrid (FTS5 top-100 -> dense)
  rerank/exact candidates = exact top-40 -> POST /v1/rerank (both rerankers)
  rerank/dense candidates = turbovec top-40 -> POST /v1/rerank (mettle composite)
  global       one collection, all 36,920 chunks, allowlist-filtered search
               (embed-gemma only; production-scale ANN data point)
  embed-thruput /v1/embeddings batch=64 over a fixed 3-filing subset
  ingest       chunks/sec of collection ingest (server-side embed + index add)

Usage (from this directory):
  uv run python -u mettle_bench.py --smoke          # 2 filings, quick sanity
  uv run python -u mettle_bench.py                  # full 150-Q run (resume-safe)

Requires: mettled running with mettle-bench-profile.toml (see that file).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_common as pc  # noqa: E402

BASE_DEFAULT = "http://127.0.0.1:11434/v1"
RESULTS = Path(__file__).with_name("results")
OUT_PATH = RESULTS / "mettle_results.json"
STATE_PATH = RESULTS / "mettle_state.json"
STATE_DIR = Path("/personal-projects/mettle/benchmarks/retrieval_parity/.state")

COLL_PREFIX = "fbpar"
GLOBAL_COLL = "fbpar-global"
FIDELITY_N = 256
GLOBAL_FILINGS_MAX = 0  # 0 = all filings in the global collection

http = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def post(url: str, body: dict, tries: int = 6, what: str = "request") -> dict | list:
    """POST with wait-retry on transport errors / 503 slot_unavailable."""
    for attempt in range(tries):
        try:
            r = http.post(url, json=body)
            if r.status_code == 503:
                wait = float(r.headers.get("Retry-After", "2") or 2)
                print(f"    [503] {what}: waiting {wait}s ({r.text[:120]})", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == tries - 1:
                raise
            print(f"    [retry] {what}: {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{what}: exhausted retries")


def get(url: str) -> dict:
    r = http.get(url)
    r.raise_for_status()
    return r.json()


def lat_stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0}
    s = sorted(samples)
    q = lambda f: s[min(len(s) - 1, int(round((len(s) - 1) * f)))]  # noqa: E731
    return {
        "n": len(s),
        "mean_ms": round(statistics.fmean(s) * 1000, 2),
        "p50_ms": round(q(0.5) * 1000, 2),
        "p95_ms": round(q(0.95) * 1000, 2),
    }


def coll_name(embedder: str, doc: str) -> str:
    slug = pc.EMBEDDER_CACHE_SLUG[embedder].replace(".", "-")
    return f"{COLL_PREFIX}-{slug}-{doc.lower().replace('_', '-')}"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# state / checkpointing
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"ingested": {}, "search": {}, "rerank": {}, "global": {},
            "fidelity": {}, "embed_thruput": {}, "started": now()}


def save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=1))


def provenance() -> dict:
    gguf = {}
    for m in Path("/personal-projects/mettle/models").rglob("*.gguf"):
        gguf[str(m)] = m.stat().st_size
    try:
        head = subprocess.run(
            ["git", "-C", "/personal-projects/mettle", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", "/personal-projects/mettle", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        head, dirty = "unknown", ""
    binary = Path("/personal-projects/mettle/target/release/mettled")
    return {
        "mettle_git_head": head,
        "mettle_git_dirty": bool(dirty),
        "mettled_mtime": dt.datetime.fromtimestamp(binary.stat().st_mtime).isoformat()
        if binary.exists() else None,
        "gguf_files": gguf,
        "host": "RTX 3060 Laptop 6GB / Ryzen 9 5900HX",
        "questions_file": str(pc.QUESTIONS_PATH),
        "chunk_source": str(pc.SWEEP_CACHE) + " (frozen docket chunks, 36,920)",
        "config": str(Path(__file__).with_name("mettle-bench-profile.toml")),
        "metric": "gold-evidence token-overlap recall@k (verbatim docket sweep_recall.py)",
    }


# ---------------------------------------------------------------------------
# phase 0: environment
# ---------------------------------------------------------------------------

def probe(base: str) -> dict:
    models = get(f"{base}/models")
    ids = [m.get("id") for m in models.get("data", [])]
    print(f"server models: {ids}", flush=True)
    need = set(pc.EMBEDDER_CACHE_SLUG) | set(pc.RERANKERS)
    missing = sorted(need - set(ids))
    if missing:
        print(f"WARNING: models not listed by server (will JIT-load?): {missing}", flush=True)
    # warm every model so first-use JIT load is outside timed sections
    for m in sorted(need):
        if m in pc.RERANKERS:
            t0 = time.perf_counter()
            post(f"{base}/rerank", {"model": m, "query": "warmup",
                                    "documents": ["alpha", "beta"]}, what=f"warmup {m}")
            print(f"  warm {m}: {time.perf_counter()-t0:.1f}s", flush=True)
        else:
            t0 = time.perf_counter()
            post(f"{base}/embeddings", {"model": m, "input": ["warmup"]}, what=f"warmup {m}")
            print(f"  warm {m}: {time.perf_counter()-t0:.1f}s", flush=True)
    try:
        status = get(base.replace("/v1", "") + "/engine/status")
    except Exception as e:  # noqa: BLE001
        status = {"error": str(e)}
    return {"listed_models": ids, "engine_status_probe": status}


def measured_bytes() -> dict:
    try:
        st = get("http://127.0.0.1:11434/engine/status")
        return {sl.get("model", sl.get("id", "?")): sl.get("measured_bytes", 0)
                for sl in st.get("slots", [])}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# phase 1: cache fidelity (are the frozen vectors still what this binary emits?)
# ---------------------------------------------------------------------------

def embed_texts(base: str, model: str, texts: list[str]) -> tuple[list[list[float]], float]:
    t0 = time.perf_counter()
    resp = post(f"{base}/embeddings", {"model": model, "input": texts},
                what=f"embed {model} x{len(texts)}")
    dt_ = time.perf_counter() - t0
    rows = sorted(resp["data"], key=lambda r: r["index"])
    return [r["embedding"] for r in rows], dt_


def fidelity(base: str, state: dict, filings: list[str], force: bool = False) -> None:
    for embedder in pc.EMBEDDER_CACHE_SLUG:
        if not force and embedder in state["fidelity"]:
            continue
        import random
        rng = random.Random(42)
        pairs = []
        for doc in filings:
            corpus = pc.load_filing_corpus(embedder, doc)
            for row in corpus:
                pairs.append((doc, row))
        sample = rng.sample(pairs, min(FIDELITY_N, len(pairs)))
        texts = [pc.embed_clean(r["text"]) for _, r in sample]
        vecs, secs = [], 0.0
        for i in range(0, len(texts), 64):
            v, d = embed_texts(base, embedder, texts[i:i + 64])
            vecs.extend(v)
            secs += d
        cos = [pc.cosine(v, r["vector"]) for (_, r), v in zip(sample, vecs)]
        rep = {
            "n": len(cos),
            "min_cos": round(min(cos), 6),
            "mean_cos": round(statistics.fmean(cos), 6),
            "embed_secs": round(secs, 2),
            "chunks_per_sec": round(len(cos) / secs, 2) if secs else 0,
            "checked_at": now(),
        }
        state["fidelity"][embedder] = rep
        save_state(state)
        print(f"[fidelity] {embedder}: min_cos={rep['min_cos']} "
              f"mean_cos={rep['mean_cos']} ({rep['n']} chunks, "
              f"{rep['chunks_per_sec']} ch/s)", flush=True)
        if rep["min_cos"] < 0.999:
            print(f"WARNING: {embedder} cache fidelity low — vectors were produced "
                  f"by a different engine state; exact-arm numbers still valid "
                  f"(they use these fresh vectors? NO — exact arm uses cache). "
                  f"See contract caveats.", flush=True)


# ---------------------------------------------------------------------------
# phase 2: ingest (per embedder, per filing collection)
# ---------------------------------------------------------------------------

def ensure_collection(base: str, name: str, embedder: str) -> None:
    http.delete(f"{base}/collections/{name}", timeout=30.0)
    r = http.post(f"{base}/collections",
                  json={"name": name, "embedding_model": embedder}, timeout=60.0)
    assert r.status_code in (200, 201, 409), f"create {name}: {r.status_code} {r.text[:200]}"


def ingest_filing(base: str, embedder: str, doc: str, corpus: list[dict]) -> dict:
    name = coll_name(embedder, doc)
    ensure_collection(base, name, embedder)
    t0 = time.perf_counter()
    for row in corpus:
        post(f"{base}/collections/{name}/documents",
             {"id": row["chunk_id"], "content": pc.embed_clean(row["text"]),
              "metadata": {"doc_id": doc}},
             what=f"ingest {name} {row['chunk_id']}")
    secs = time.perf_counter() - t0
    # sanity: searching the first chunk's text must return that chunk top-1.
    # mettle's returned chunk_id is "<doc.id>/chunk/<i>"; the POSTed document id
    # (== docket chunk_id) comes back as document_id, so key on that.
    s = post(f"{base}/collections/{name}/search",
             {"query": pc.embed_clean(corpus[0]["text"]), "k": 1, "mode": "dense"},
             what=f"sanity {name}")
    ok = bool(s) and s[0].get("document_id") == corpus[0]["chunk_id"]
    return {"chunks": len(corpus), "secs": round(secs, 2),
            "chunks_per_sec": round(len(corpus) / secs, 2), "self_search_ok": ok}


def ingest_phase(base: str, state: dict, filings: list[str], embedders: list[str],
                 by_doc: dict[str, list[dict]]) -> None:
    for embedder in embedders:
        done = state["ingested"].setdefault(embedder, {})
        todo = [d for d in filings if d not in done]
        if not todo:
            print(f"[ingest] {embedder}: all {len(filings)} filings already ingested", flush=True)
            continue
        print(f"[ingest] {embedder}: {len(todo)} filings to ingest", flush=True)
        for i, doc in enumerate(todo):
            corpus = pc.load_filing_corpus(embedder, doc)
            rep = ingest_filing(base, embedder, doc, corpus)
            if not rep["self_search_ok"]:
                print(f"  FATAL: self-search sanity failed for {doc} ({embedder}) — "
                      f"embeddings likely degraded to zero vectors. Stop and inspect.",
                      flush=True)
                sys.exit(3)
            done[doc] = rep
            save_state(state)
            print(f"  [{i+1}/{len(todo)}] {doc}: {rep['chunks']} chunks in "
                  f"{rep['secs']}s ({rep['chunks_per_sec']} ch/s)", flush=True)


def index_bytes(embedder: str) -> int:
    """Sum of per-filing turbovec index.bin bytes for one embedder."""
    slug = pc.EMBEDDER_CACHE_SLUG[embedder].replace(".", "-")
    total = 0
    for p in STATE_DIR.glob(f"collections/{COLL_PREFIX}-{slug}-*/index.bin"):
        total += p.stat().st_size
    return total


# ---------------------------------------------------------------------------
# phase 3: search arms (per embedder, per question)
# ---------------------------------------------------------------------------

def score_hits(hit_ids: list[str], raw_text: dict[str, str], golds: list[set],
               ks=pc.KS, thresholds=pc.THRESHOLDS) -> dict:
    """recall@k bookkeeping; hit texts are the RAW chunk texts (identical to
    what docket scores), looked up by chunk_id."""
    token_sets = [set(tokenize_raw(raw_text.get(cid, ""))) for cid in hit_ids]
    out = {}
    for k in ks:
        for t in thresholds:
            out[f"@{k}_t{t}"] = pc.recall_hit(token_sets[:k], golds, t)
    best = 0.0
    for cs in token_sets:
        for g in golds:
            best = max(best, pc.overlap_coeff(cs, g))
    return {"hits": out, "best_overlap": best}


def tokenize_raw(text: str) -> list[str]:
    return pc.tokenize(text)


def search_phase(base: str, state: dict, filings: list[str], embedders: list[str],
                 by_doc: dict[str, list[dict]]) -> None:
    for embedder in embedders:
        done = state["search"].setdefault(embedder, {})
        for doc in filings:
            recs = by_doc.get(doc, [])
            recs = [r for r in recs if r["financebench_id"] not in done]
            if not recs:
                continue
            corpus = pc.load_filing_corpus(embedder, doc)
            raw_text = {r["chunk_id"]: r["text"] for r in corpus}
            vecs = [r["vector"] for r in corpus]
            name = coll_name(embedder, doc)
            # warmup search (excluded): first call may touch lazy paths
            post(f"{base}/collections/{name}/search",
                 {"query": recs[0]["question"], "k": 1, "mode": "dense"},
                 what=f"warmsearch {name}")
            for rec in recs:
                qid = rec["financebench_id"]
                golds = pc.gold_evidence(rec)
                row: dict = {"qid": qid}
                # query vector (timed — docket pays this too via embed_texts)
                t0 = time.perf_counter()
                qv, _ = embed_texts(base, embedder, [rec["question"]])
                row["embed_query_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                qv = qv[0]
                # exact reference (local brute-force cosine, as docket Corpus does)
                t0 = time.perf_counter()
                exact_ids = [corpus[i]["chunk_id"] for i in pc.rank_by_query(vecs, qv)]
                row["exact_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                row["exact"] = score_hits(exact_ids[:pc.CANDIDATES], raw_text, golds)
                # mettle search modes
                for mode in ("dense", "keyword", "hybrid"):
                    t0 = time.perf_counter()
                    hits = post(f"{base}/collections/{name}/search",
                                {"query": rec["question"], "k": pc.CANDIDATES,
                                 "mode": mode},
                                what=f"search {name} {mode}")
                    row[f"{mode}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    ids = [h["document_id"] for h in hits]
                    row[mode] = score_hits(ids, raw_text, golds)
                done[qid] = row
            save_state(state)
            print(f"[search] {embedder} {doc}: {len(recs)} questions scored", flush=True)


# ---------------------------------------------------------------------------
# phase 4: rerank arms
# ---------------------------------------------------------------------------

def rerank_top(base: str, model: str, query: str, texts: list[str]) -> tuple[list[int], float]:
    """Returns (candidate indices best-first, call seconds)."""
    t0 = time.perf_counter()
    resp = post(f"{base}/rerank", {"model": model, "query": query, "documents": texts},
                what=f"rerank {model} x{len(texts)}")
    secs = time.perf_counter() - t0
    order = [r["index"] for r in resp["results"]]
    return order, secs


def rerank_phase(base: str, state: dict, filings: list[str], embedders: list[str],
                 by_doc: dict[str, list[dict]]) -> None:
    for embedder in embedders:
        for rr in pc.RERANKERS:
            key = f"{embedder}|{rr}"
            done = state["rerank"].setdefault(key, {})
            for doc in filings:
                recs = [r for r in by_doc.get(doc, [])
                        if r["financebench_id"] not in done]
                if not recs:
                    continue
                corpus = pc.load_filing_corpus(embedder, doc)
                raw_text = {r["chunk_id"]: r["text"] for r in corpus}
                vecs = [r["vector"] for r in corpus]
                name = coll_name(embedder, doc)
                for rec in recs:
                    qid = rec["financebench_id"]
                    golds = pc.gold_evidence(rec)
                    qv = embed_texts(base, embedder, [rec["question"]])[0][0]
                    exact_ids = [corpus[i]["chunk_id"]
                                 for i in pc.rank_by_query(vecs, qv)][:pc.CANDIDATES]
                    dense_hits = post(f"{base}/collections/{name}/search",
                                      {"query": rec["question"], "k": pc.CANDIDATES,
                                       "mode": "dense"},
                                      what=f"search {name} dense (rerank cand)")
                    dense_ids = [h["document_id"] for h in dense_hits]
                    row = {"qid": qid}
                    for src, ids in (("exact", exact_ids), ("dense", dense_ids)):
                        texts = [pc.embed_clean(raw_text.get(cid, "")) for cid in ids]
                        order, secs = rerank_top(base, rr, rec["question"], texts)
                        ranked = [ids[i] for i in order]
                        s = score_hits(ranked, raw_text, golds)
                        s["rerank_secs"] = round(secs, 2)
                        row[f"rerank_{src}"] = s
                    done[qid] = row
                save_state(state)
                print(f"[rerank] {key} {doc}: {len(recs)} questions", flush=True)


# ---------------------------------------------------------------------------
# phase 5: global collection (embed-gemma only) — production scale
# ---------------------------------------------------------------------------

def global_phase(base: str, state: dict, filings: list[str],
                 by_doc: dict[str, list[dict]], embedder: str = "embed-gemma:latest") -> None:
    g = state["global"]
    if "ingest" not in g:
        print(f"[global] building {GLOBAL_COLL} with ALL chunks of {len(filings)} filings", flush=True)
        ensure_collection(base, GLOBAL_COLL, embedder)
        t0 = time.perf_counter()
        n = 0
        for doc in filings:
            corpus = pc.load_filing_corpus(embedder, doc)
            for row in corpus:
                post(f"{base}/collections/{GLOBAL_COLL}/documents",
                     {"id": row["chunk_id"], "content": pc.embed_clean(row["text"]),
                      "metadata": {"doc_id": doc}},
                     what=f"global ingest {row['chunk_id']}")
                n += 1
            print(f"  [global] {doc}: {n} chunks total, "
                  f"{round(time.perf_counter()-t0)}s", flush=True)
        g["ingest"] = {"chunks": n, "secs": round(time.perf_counter() - t0, 1)}
        p = STATE_DIR / GLOBAL_COLL / "index.bin"
        g["index_bytes"] = p.stat().st_size if p.exists() else None
        save_state(state)
    if "search" not in g:
        done = {}
        allow = {doc: [r["chunk_id"]
                       for r in pc.load_filing_corpus(embedder, doc)]
                 for doc in filings}
        for doc in filings:
            for rec in by_doc.get(doc, []):
                qid = rec["financebench_id"]
                golds = pc.gold_evidence(rec)
                row = {"qid": qid, "doc": doc}
                for mode in ("dense", "keyword", "hybrid"):
                    t0 = time.perf_counter()
                    hits = post(f"{base}/collections/{GLOBAL_COLL}/search",
                                {"query": rec["question"], "k": 20, "mode": mode,
                                 "allowlist": allow[doc]},
                                what=f"global search {mode}")
                    row[f"{mode}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    row[mode] = score_hits([h["document_id"] for h in hits],
                                           {r["chunk_id"]: r["text"]
                                            for r in pc.load_filing_corpus(embedder, doc)},
                                           golds, ks=pc.KS, thresholds=pc.THRESHOLDS)
                done[qid] = row
            g["search"] = done
            save_state(state)
            print(f"[global] searched {doc}", flush=True)


# ---------------------------------------------------------------------------
# phase 6: embedding throughput (docket-client-equivalent: batch=64)
# ---------------------------------------------------------------------------

def embed_thruput_phase(base: str, state: dict, filings: list[str]) -> None:
    subset = filings[:3]  # fixed, named in the contract
    for embedder in pc.EMBEDDER_CACHE_SLUG:
        if embedder in state["embed_thruput"]:
            continue
        texts = []
        for doc in subset:
            for row in pc.load_filing_corpus(embedder, doc):
                texts.append(pc.embed_clean(row["text"]))
        # warmup batch excluded
        embed_texts(base, embedder, texts[:64])
        batch_times = []
        t_all0 = time.perf_counter()
        for i in range(0, len(texts), 64):
            _, d = embed_texts(base, embedder, texts[i:i + 64])
            batch_times.append(d)
        wall = time.perf_counter() - t_all0
        rep = {
            "subset_filings": subset,
            "texts": len(texts),
            "batch": 64,
            "wall_secs": round(wall, 1),
            "chunks_per_sec": round(len(texts) / wall, 2),
            "batch_latency": lat_stats(batch_times),
            "per_text_ms": round(wall * 1000 / len(texts), 1),
        }
        state["embed_thruput"][embedder] = rep
        save_state(state)
        print(f"[embed-thruput] {embedder}: {rep['chunks_per_sec']} ch/s "
              f"({rep['per_text_ms']} ms/text, batch=64)", flush=True)


# ---------------------------------------------------------------------------
# aggregation / report
# ---------------------------------------------------------------------------

def agg_recall(rows: list[dict], arm: str) -> dict:
    """Mean recall over questions for one arm's @k_t flags + mean best overlap."""
    if not rows:
        return {}
    flags = [k for k in rows[0][arm] if k.startswith("@")]
    out = {k: round(sum(1 for r in rows if r[arm][k]) / len(rows), 4) for k in flags}
    if "best_overlap" in rows[0][arm]:
        out["mean_best_overlap"] = round(
            statistics.fmean(r[arm]["best_overlap"] for r in rows), 4)
    return out


def summarize(state: dict) -> dict:
    out: dict = {"arms": {}}
    for embedder, qrows in state["search"].items():
        rows = list(qrows.values())
        out["arms"].setdefault(embedder, {})
        for arm in ("exact", "dense", "keyword", "hybrid"):
            out["arms"][embedder][arm] = agg_recall(rows, arm)
        ms = {arm: lat_stats([r[f"{arm}_ms"] / 1000 for r in rows if f"{arm}_ms" in r])
              for arm in ("dense", "keyword", "hybrid")}
        ms["exact_local"] = lat_stats([r["exact_ms"] / 1000 for r in rows])
        ms["embed_query"] = lat_stats([r["embed_query_ms"] / 1000 for r in rows])
        out["arms"][embedder]["latency"] = ms
    for key, qrows in state["rerank"].items():
        emb, rr = key.split("|")
        rows = list(qrows.values())
        node = out["arms"].setdefault(emb, {})
        for src in ("exact", "dense"):
            arm = f"rerank_{src}({rr})"
            node[arm] = agg_recall(rows, f"rerank_{src}")
        node.setdefault("latency", {})[f"rerank({rr})"] = lat_stats(
            [r["rerank_exact"]["rerank_secs"] for r in rows])
    if state["global"].get("search"):
        rows = list(state["global"]["search"].values())
        g = {}
        for arm in ("dense", "keyword", "hybrid"):
            g[arm] = agg_recall(rows, arm)
        g["latency"] = {arm: lat_stats([r[f"{arm}_ms"] / 1000 for r in rows])
                        for arm in ("dense", "keyword", "hybrid")}
        g["ingest"] = state["global"].get("ingest")
        g["index_bytes"] = state["global"].get("index_bytes")
        out["global_collection(embed-gemma)"] = g
    return out


def print_report(res: dict) -> None:
    ks_head = [f"@{k}_t0.5" for k in pc.KS]
    print("\n=== recall@k (t0.5) — mettle side, real FinanceBench ===")
    for emb, node in res.get("arms", {}).items():
        print(f"\n[{emb}]")
        hdr = f"{'arm':<38}" + "".join(f"{k:>9}" for k in ks_head)
        print(hdr)
        for arm, rec in node.items():
            if arm == "latency" or not isinstance(rec, dict):
                continue
            if not any(k in rec for k in ks_head):
                continue
            print(f"{arm:<38}" + "".join(
                f"{rec.get(k, float('nan')):>9.3f}" if k in rec else f"{'-':>9}"
                for k in ks_head))
        lat = node.get("latency", {})
        if lat:
            print("  latency(ms): " + "  ".join(
                f"{k} p50={v['p50_ms']:.0f} mean={v['mean_ms']:.0f}"
                for k, v in lat.items()))
    g = res.get("global_collection(embed-gemma)")
    if g:
        print(f"\n[global collection, embed-gemma, 36,920 chunks, allowlist-filtered]")
        for arm in ("dense", "keyword", "hybrid"):
            rec = g.get(arm, {})
            print(f"{arm:<38}" + "".join(f"{rec.get(k, float('nan')):>9.3f}" for k in ks_head))
        print(f"  latency: {g.get('latency')}")
        print(f"  index_bytes={g.get('index_bytes')}  ingest={g.get('ingest')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="questions to score (0=all)")
    ap.add_argument("--max-filings", type=int, default=0, help="filings to use (0=all)")
    ap.add_argument("--embedders",
                    default=",".join(pc.EMBEDDER_CACHE_SLUG))
    ap.add_argument("--smoke", action="store_true",
                    help="2 filings / 8 questions sanity run (separate state)")
    ap.add_argument("--skip-global", action="store_true")
    ap.add_argument("--phase", default="all",
                    choices=["all", "probe", "fidelity", "ingest", "search",
                             "rerank", "global", "embed-thruput", "report"])
    args = ap.parse_args()

    if args.smoke:
        global STATE_PATH, OUT_PATH  # noqa: PLW0603
        STATE_PATH = RESULTS / "mettle_state_smoke.json"
        OUT_PATH = RESULTS / "mettle_results_smoke.json"
        args.max_filings = args.max_filings or 2
        args.limit = args.limit or 8

    RESULTS.mkdir(parents=True, exist_ok=True)
    state = load_state()
    base = args.base_url.rstrip("/")
    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]

    # questions + filings
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
    print(f"{n_q} questions across {len(filings)} filings "
          f"(embedders: {', '.join(embedders)})", flush=True)

    ph = args.phase
    if ph in ("all", "probe"):
        state["env"] = probe(base)
        state["provenance"] = provenance()
        save_state(state)
    if ph in ("all", "fidelity"):
        fidelity(base, state, filings)
    if ph in ("all", "ingest"):
        ingest_phase(base, state, filings, embedders, by_doc)
        state["index_bytes_by_embedder"] = {
            e: index_bytes(e) for e in embedders}
        state["engine_status_bytes"] = measured_bytes()
        save_state(state)
    if ph in ("all", "search"):
        search_phase(base, state, filings, embedders, by_doc)
    if ph in ("all", "rerank"):
        rerank_phase(base, state, filings, embedders, by_doc)
    if ph in ("all", "global") and not args.skip_global:
        global_phase(base, state, filings, by_doc)
    if ph in ("all", "embed-thruput"):
        embed_thruput_phase(base, state, filings)

    summary = summarize(state)
    res = {
        "generated": now(),
        "n_questions": n_q,
        "n_filings": len(filings),
        "filings": filings,
        "embedders": embedders,
        **summary,
        "ingest": state.get("ingested"),
        "fidelity": state.get("fidelity"),
        "embed_thruput": state.get("embed_thruput"),
        "index_bytes_by_embedder": state.get("index_bytes_by_embedder"),
        "engine_status_bytes": state.get("engine_status_bytes"),
        "provenance": state.get("provenance"),
    }
    OUT_PATH.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT_PATH}")
    print_report(res)


if __name__ == "__main__":
    main()
