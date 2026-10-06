#!/usr/bin/env python
"""BEST_CONFIG_CONTRACT.md Part C — turbovec ablation (pure Python, no mettled).

Compares mettle-core's turbovec IdMapIndex against exact numpy and the standard
ANN alternatives on the real 36,920-vector qwen3-embedding corpus (frozen docket
sweep cache) + the 150 persisted query vectors from Part A
(results/mettle_best_config_state.json). mettled must NOT be running-work here;
query vectors are read from the state file.

Run ad-hoc (NOT project deps — measurement only):
  cd benchmarks/retrieval_parity
  uv run --with turbovec --with faiss-cpu --with hnswlib --with numpy \
    python turbovec_ablation.py [--smoke]

State: results/turbovec_ablation_state.json (checkpoint per engine x phase).
Output: results/turbovec_ablation.json (contract §6).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_common as pc  # noqa: E402

RESULTS = Path(__file__).with_name("results")
OUT_PATH = RESULTS / "turbovec_ablation.json"
STATE_PATH = RESULTS / "turbovec_ablation_state.json"
QVEC_STATE = RESULTS / "mettle_best_config_state.json"

EMBEDDER = "qwen3-embedding-0.6b:latest"
DIM = 1024
K = 10
NQUERIES = 150
RAW_F32_BYTES = 36_920 * DIM * 4  # 151,224,320

# allowlist selectivities (contract §5/§6): 1% (own filing), 5%, 48%
SELIVITIES = (("filtered_1pct", 1), ("filtered_5pct", 5), ("filtered_48pct", 48))

DEV_NOTE = []


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB


def lat(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    q = lambda f: s[min(len(s) - 1, int(round((len(s) - 1) * f)))]  # noqa: E731
    return {"n": len(s), "mean_ms": round(statistics.fmean(s), 3),
            "p50_ms": round(q(0.5), 3), "p95_ms": round(q(0.95), 3)}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"started": now(), "engines": {}, "deviations": []}


def save_state(st: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=1))


def provenance() -> dict:
    import turbovec  # type: ignore
    import faiss  # type: ignore
    import hnswlib  # type: ignore
    return {
        "turbovec": getattr(turbovec, "__version__", "unknown"),
        "faiss": getattr(faiss, "__version__", "unknown"),
        "hnswlib": getattr(hnswlib, "__version__", "unknown"),
        "numpy": np.__version__,
        "corpus": f"{pc.SWEEP_CACHE}/{pc.EMBEDDER_CACHE_SLUG[EMBEDDER]} (frozen)",
        "raw_f32_bytes": RAW_F32_BYTES,
        "single_thread_pass": True,
        "note": "cosine == IP on normalized vectors; all vectors L2-normalized "
                "before IP engines (queries too); exact_numpy ranks by normalized IP",
    }


# ---------------------------------------------------------------------------
# data: global stack of all filings + 150 persisted query vectors
# ---------------------------------------------------------------------------

def load_data(smoke: bool = False) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[int]]]:
    qst = json.loads(QVEC_STATE.read_text())
    qids = sorted(qst["query_vectors"])
    if len(qids) < NQUERIES:
        raise RuntimeError(f"only {len(qids)} query vectors in {QVEC_STATE}; run Part A first")
    Q = np.array([qst["query_vectors"][q] for q in qids], dtype=np.float32)

    vecs, ids, filing_ids = [], [], {}
    filings = sorted(p.name for p in
                     (pc.SWEEP_CACHE / pc.EMBEDDER_CACHE_SLUG[EMBEDDER]).iterdir()
                     if p.is_dir())
    for doc in filings:
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        fids = []
        for row in corpus:
            vecs.append(row["vector"])
            ids.append(row["chunk_id"])
            fids.append(len(ids) - 1)
        filing_ids[doc] = fids
    X = np.array(vecs, dtype=np.float32)
    # normalize so inner product == cosine (disclosed in provenance)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    if smoke:
        Qn = Qn[:NQUERIES]
    print(f"data: {X.shape} vectors, {len(Qn)} queries, {len(filings)} filings",
          flush=True)
    return X, Qn, ids, filing_ids


def exact_topk(X: np.ndarray, Qn: np.ndarray, k: int = K) -> np.ndarray:
    """exact_numpy reference: scores = X @ q, top-k ids best-first."""
    return np.argsort(-(X @ Qn.T), axis=0)[:k].T  # (nq, k)


# ---------------------------------------------------------------------------
# engine wrappers — each exposes: build(X), search(q, k), search_filtered
# (q, k, allowlist), save(path) -> bytes, remove(ids)
# ---------------------------------------------------------------------------

class Engine:
    name = "?"
    strategy = "post-filter"
    supports_delete = False

    def __init__(self, X: np.ndarray):
        self.X = X

    def build(self) -> tuple[float, float]:  # (wall_s, train_s)
        raise NotImplementedError

    def search(self, q: np.ndarray, k: int) -> np.ndarray:  # returns ids
        raise NotImplementedError

    def search_filtered(self, q: np.ndarray, k: int, allow: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> int | None:
        return None

    def ensure_built(self) -> None:
        """Rebuild (unmeasured) when this phase runs in a fresh process after a
        state resume — the built index only lives in the process that built it."""
        if not getattr(self, "_built", False):
            self.build()
            self._built = True

    def delete(self, ids: list[int], k: int, probe_q: np.ndarray) -> dict:
        return {"supported": False,
                "note": "no id-removal (rebuild required)"}


class ExactNumpy(Engine):
    name = "exact_numpy"
    strategy = "subset brute force (exact by construction)"

    def build(self):
        return 0.0, 0.0  # no index; matrix is the reference copy

    def search(self, q, k):
        return np.argsort(-(self.X @ q))[:k]

    def search_filtered(self, q, k, allow):
        return allow[np.argsort(-(self.X[allow] @ q))[:k]]

    def save(self, path):
        p = path.with_suffix(".npy")
        np.save(p, self.X)
        return p.stat().st_size



class Turbovec(Engine):
    """turbovec 1.0.0 wheel (the same IdMapIndex mettle-core uses; the pypi
    wheel postdates the 0.8.1 crate pinned in mettle-core — versions recorded).

    API (verified): add_with_ids(X, ids) batched; search(queries, k,
    allowlist=None) -> (scores, ids); remove(id); to_bytes/from_bytes/write/load.
    """

    def __init__(self, X: np.ndarray, bit_width: int):
        super().__init__(X)
        self.bw = bit_width
        self.name = f"turbovec_idmap_bw{bit_width}"
        self.strategy = "in-kernel allowlist (search allowlist= kwarg)"
        self.supports_delete = True

    def build(self):
        import turbovec
        t0 = time.perf_counter()
        self.idx = turbovec.IdMapIndex(DIM, self.bw)
        self.idx.add_with_ids(self.X, np.arange(len(self.X), dtype=np.uint64))
        return time.perf_counter() - t0, 0.0

    def search(self, q, k):
        _, ids = self.idx.search(q[None, :], k)
        return ids[0]

    def search_filtered(self, q, k, allow):
        _, ids = self.idx.search(q[None, :], k,
                                 allowlist=allow.astype(np.uint64))
        return ids[0]

    def save(self, path):
        data = self.idx.to_bytes()
        path.write_bytes(data)
        return len(data)

    def delete(self, ids, k, probe_q):
        t0 = time.perf_counter()
        for i in ids:
            self.idx.remove(i)
        ms = (time.perf_counter() - t0) * 1000
        ok = not any(self.idx.contains(int(i)) for i in ids)
        _, got = self.idx.search(probe_q[None, :], k)
        clean = not (set(got[0].tolist()) & set(int(i) for i in ids))
        return {"supported": True, "ms_per_1000": round(ms / len(ids) * 1000, 3),
                "n": len(ids), "post_delete_ok": bool(ok and clean),
                "note": "remove() deletes by id (contains()==False afterwards); "
                        "search verified clean"}


class FaissFlat(Engine):
    name = "faiss_flat_ip"
    strategy = "post-filter over full exact scores (IDSelector-free; exact)"

    def build(self):
        import faiss
        t0 = time.perf_counter()
        self.idx = faiss.IndexFlatIP(DIM)
        self.idx.add(self.X)
        return time.perf_counter() - t0, 0.0

    def search(self, q, k):
        _, I = self.idx.search(q[None, :], k)
        return I[0]

    def search_filtered(self, q, k, allow):
        mask = np.zeros(len(self.X), dtype=bool)
        mask[allow] = True
        scores = self.X @ q
        scores[~mask] = -np.inf
        return np.argsort(-scores)[:k]

    def save(self, path):
        import faiss
        p = str(path) + ".faiss"
        faiss.write_index(self.idx, p)
        return Path(p).stat().st_size


class FaissHnsw(Engine):
    name = "faiss_hnsw"
    strategy = ("overfetch ef escalation 256/1024/4096, exact post-filter "
                "fallback if still short — output NOT guaranteed exact top-k")

    def build(self):
        import faiss
        t0 = time.perf_counter()
        idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 64
        idx.add(self.X)
        self.idx = idx
        return time.perf_counter() - t0, 0.0

    def search(self, q, k):
        _, I = self.idx.search(q[None, :], k)
        return I[0]

    def search_filtered(self, q, k, allow):
        mask = np.zeros(len(self.X), dtype=bool)
        mask[allow] = True
        want = min(len(allow), k * 4)
        for ef in (256, 1024, 4096):
            self.idx.hnsw.efSearch = ef
            _, I = self.idx.search(q[None, :], want)
            hits = [i for i in I[0] if i >= 0 and mask[i]][:k]
            if len(hits) >= k:
                self.idx.hnsw.efSearch = 64
                return np.array(hits)
        self.idx.hnsw.efSearch = 64
        # disclosed fallback: exact brute force over the allowed subset
        return allow[np.argsort(-(self.X[allow] @ q))[:k]]

    def save(self, path):
        import faiss
        p = str(path) + ".faiss"
        faiss.write_index(self.idx, p)
        return Path(p).stat().st_size


class FaissIvfPq(Engine):
    name = "faiss_ivfpq"
    strategy = "post-filter over nprobe=8 IVF results (compressed, approximate)"

    def __init__(self, X: np.ndarray):
        super().__init__(X)
        self.nlist = 768  # ~4*sqrt(36920)
        self.train_frac = 0.2

    def build(self):
        import faiss
        n = len(self.X)
        rng = np.random.default_rng(0)
        train = self.X[rng.choice(n, size=max(int(n * self.train_frac), 4096),
                                  replace=False)]
        quantizer = faiss.IndexFlatIP(DIM)
        idx = faiss.IndexIVFPQ(quantizer, DIM, self.nlist, 256, 4,
                               faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter()
        idx.train(train)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        idx.add(self.X)
        add_s = time.perf_counter() - t0
        idx.nprobe = 8
        self.idx = idx
        return add_s, train_s

    def search(self, q, k):
        _, I = self.idx.search(q[None, :], k)
        return I[0]

    def search_filtered(self, q, k, allow):
        mask = np.zeros(len(self.X), dtype=bool)
        mask[allow] = True
        _, I = self.idx.search(q[None, :], min(len(allow), k * 8))
        hits = [i for i in I[0] if i >= 0 and mask[i]][:k]
        return np.array(hits)

    def save(self, path):
        import faiss
        p = str(path) + ".faiss"
        faiss.write_index(self.idx, p)
        return Path(p).stat().st_size


class HnswLib(Engine):
    name = "hnswlib"
    strategy = "overfetch ef x4 + re-query if short; mark_deleted soft-removal"
    supports_delete = True

    def build(self):
        import hnswlib
        t0 = time.perf_counter()
        idx = hnswlib.Index(space="ip", dim=DIM)
        idx.init_index(max_elements=len(self.X) * 4, ef_construction=200, M=32)
        idx.set_ef(64)
        idx.add_items(self.X, np.arange(len(self.X)))
        self.idx = idx
        return time.perf_counter() - t0, 0.0

    def search(self, q, k):
        return self.idx.knn_query(q[None, :], k=k)[0][0]

    def search_filtered(self, q, k, allow):
        mask = np.zeros(len(self.X), dtype=bool)
        mask[allow] = True
        for ef in (256, 1024, len(allow)):
            self.idx.set_ef(min(ef, len(allow)))
            I = self.idx.knn_query(q[None, :], k=min(len(allow), k * 4))[0][0]
            hits = [i for i in I if mask[i]][:k]
            if len(hits) >= k:
                return np.array(hits)
        return np.array(hits)

    def save(self, path):
        p = str(path) + ".hnsw"
        self.idx.save_index(p)
        return Path(p).stat().st_size

    def delete(self, ids, k, probe_q):
        t0 = time.perf_counter()
        for i in ids:
            self.idx.mark_deleted(i)
        ms = (time.perf_counter() - t0) * 1000
        res = set(self.idx.knn_query(probe_q[None, :], k=k)[0][0].tolist())
        return {"supported": True, "ms_per_1000": round(ms / len(ids) * 1000, 3),
                "n": len(ids), "post_delete_ok": not (res & set(ids)),
                "note": "mark_deleted soft-deletes (no space reclaim)"}


def make_engines(X: np.ndarray, smoke: bool) -> list[Engine]:
    engs: list[Engine] = [ExactNumpy(X)]
    import turbovec
    for bw in (4, 2, 8):
        try:
            turbovec.IdMapIndex(DIM, bw)
        except ValueError as e:
            DEV_NOTE.append(f"turbovec bit_width={bw} skipped: {e} "
                            "(turbovec 1.0.0 wheel supports 2/3/4; contract asked for 2/4/8)")
            continue
        engs.append(Turbovec(X, bw))
    engs += [FaissFlat(X), FaissHnsw(X), FaissIvfPq(X), HnswLib(X)]
    if smoke:
        engs = engs[:3]
    try:
        import sqlite_vec  # type: ignore  # noqa: F401
        DEV_NOTE.append("sqlite_vec present but not measured (optional per contract §5.7)")
    except ImportError:
        DEV_NOTE.append("sqlite_vec skipped: optional dep not installable in this env")
    return engs


# ---------------------------------------------------------------------------
# measurement phases (checkpoint per engine x phase)
# ---------------------------------------------------------------------------

def phase_ingest(state: dict, e: Engine) -> None:
    node = state["engines"].setdefault(e.name, {})
    if "ingest" in node:
        return
    rss0 = rss_kb()
    wall_s, train_s = e.build()
    e._built = True
    rss1 = rss_kb()
    node["ingest"] = {"wall_s": round(wall_s, 3), "vecs_per_s": round(len(e.X) / max(wall_s, 1e-9), 1),
                      "train_s": round(train_s, 3),
                      "note": "train phase separate" if train_s else "no training phase"}
    node["memory"] = {"peak_rss_bytes": max(0, (rss1 - rss0)) * 1024}
    save_state(state)
    print(f"[ingest] {e.name}: {wall_s:.1f}s train={train_s:.1f}s", flush=True)


def phase_memory(state: dict, e: Engine, tag: str) -> None:
    node = state["engines"].setdefault(e.name, {})
    if "index_bytes" in node.get("memory", {}):
        return
    p = RESULTS / f"_tmp_idx_{e.name}"
    nbytes = e.save(p) or -1
    p.with_suffix(".npy").unlink(missing_ok=True)
    for suf in (".faiss", ".hnsw"):
        (RESULTS / f"_tmp_idx_{e.name}{suf}").unlink(missing_ok=True)
    node["memory"]["index_bytes"] = nbytes
    save_state(state)
    print(f"[memory] {e.name}: {nbytes} bytes", flush=True)


def phase_unfiltered(state: dict, e: Engine, Qn: np.ndarray,
                     exact_ref: np.ndarray, records: dict,
                     chunk_ids: list[str]) -> None:
    node = state["engines"].setdefault(e.name, {})
    e.ensure_built()
    if "unfiltered" in node:
        return
    samples = []
    for q in Qn:
        t0 = time.perf_counter()
        e.search(q, K)
        samples.append((time.perf_counter() - t0) * 1000)
    ids = ids_from(e, Qn)
    overlap = float(np.mean([
        len(set(ids[i]) & set(exact_ref[i].tolist())) / K for i in range(len(Qn))]))
    # end-to-end metric recall@10 (unfiltered global), exact arm scored identically
    rec10 = e2e_recall10(ids, records, chunk_ids)
    ref10 = e2e_recall10([r.tolist() for r in exact_ref], records, chunk_ids)
    node["unfiltered"] = {**lat(samples),
                          "top10_overlap_vs_exact": round(overlap, 4),
                          "recall10_metric": round(rec10, 4),
                          "recall10_metric_delta_vs_exact": round(rec10 - ref10, 4)}
    # single-thread pass for the latency-sensitive engines
    if e.name in ("exact_numpy", "faiss_flat_ip", "turbovec_idmap_bw4"):
        node["unfiltered_single_thread"] = single_thread_pass(e, Qn)
    save_state(state)
    print(f"[unfiltered] {e.name}: {node['unfiltered']['p50_ms']}ms p50, "
          f"overlap={overlap:.3f}, rec10={rec10:.4f}", flush=True)


def ids_from(e: Engine, Qn: np.ndarray) -> list[list[int]]:
    return [e.search(q, K).tolist() for q in Qn]


def e2e_recall10(top_ids: list[list[int]], records: dict,
                 chunk_ids: list[str]) -> float:
    """recall@10 on the real metric: global top-10 scored like Part A.
    top_ids are global integer row ids -> chunk text via chunk_ids[row]."""
    qids = records["qids"]
    raw_text = records["raw_text"]
    gold_by_qid = records["gold_by_qid"]
    hits = 0
    for row, qid in zip(top_ids, qids):
        golds = gold_by_qid[qid]
        token_sets = [set(pc.tokenize(raw_text[chunk_ids[int(i)]])) for i in row]
        gold_sets = [set(g) for g in golds]
        if pc.recall_hit(token_sets, gold_sets, 0.5):
            hits += 1
    return hits / len(top_ids)


def single_thread_pass(e: Engine, Qn: np.ndarray) -> dict:
    import os
    old = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    samples = []
    for q in Qn:
        t0 = time.perf_counter()
        e.search(q, K)
        samples.append((time.perf_counter() - t0) * 1000)
    if old is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = old
    try:
        import faiss
        faiss.omp_set_num_threads(0)
    except Exception:  # noqa: BLE001
        pass
    return lat(samples)


def phase_filtered(state: dict, e: Engine, Qn: np.ndarray, filing_ids: dict,
                   ids: list[str], id_to_filing: dict[int, str],
                   rng: np.random.Generator) -> None:
    node = state["engines"].setdefault(e.name, {})
    e.ensure_built()
    all_ids = np.arange(len(ids))
    for sel_key, pct in SELIVITIES:
        if sel_key in node:
            continue
        lat_s, match_n, n = [], 0, 0
        for qi, q in enumerate(Qn):
            own = np.array(filing_ids[id_to_filing[qi]], dtype=np.int64)
            extra = int(round(len(all_ids) * pct / 100)) - len(own)
            if pct == 1:
                allow = own
            else:
                extra_ids = rng.choice(np.setdiff1d(all_ids, own), size=max(extra, 0),
                                       replace=False)
                allow = np.union1d(own, extra_ids)
            ref = allow[np.argsort(-(e.X[allow] @ q))[:K]]
            t0 = time.perf_counter()
            got = e.search_filtered(q, K, allow)
            lat_s.append((time.perf_counter() - t0) * 1000)
            if set(got.tolist()) == set(ref.tolist()):
                match_n += 1
            n += 1
        node[sel_key] = {"p50_ms": lat(lat_s)["p50_ms"],
                         "top10_match_vs_subset_exact": round(match_n / n, 4),
                         "strategy": e.strategy,
                         "n_allow": int(len(allow))}
        print(f"[filtered] {e.name} {sel_key}: {node[sel_key]['p50_ms']}ms "
              f"match={node[sel_key]['top10_match_vs_subset_exact']}", flush=True)
    save_state(state)


def phase_deletes(state: dict, e: Engine, Qn: np.ndarray) -> None:
    node = state["engines"].setdefault(e.name, {})
    e.ensure_built()
    if "deletes" in node:
        return
    probe = Qn[0]
    ids = [int(i) for i in e.search(probe, K)]
    node["deletes"] = e.delete(ids, K, probe)
    save_state(state)
    print(f"[deletes] {e.name}: {node['deletes']}", flush=True)


def phase_scale(state: dict, e: Engine, Qn: np.ndarray) -> None:
    node = state["engines"].setdefault(e.name, {})
    if "scale_4x_p50_ms" in node:
        return
    if e.name not in ("exact_numpy", "turbovec_idmap_bw4", "faiss_flat_ip", "faiss_hnsw"):
        node["scale_4x_p50_ms"] = None  # not in the contract's probe set
        save_state(state)
        return
    # replicate corpus x4 with offset ids — latency trend ONLY, no recall claims
    big = np.concatenate([e.X] * 4)
    if isinstance(e, Turbovec):
        shadow = Turbovec(big, e.bw)
    elif isinstance(e, ExactNumpy):
        shadow = ExactNumpy(big)
    elif isinstance(e, FaissFlat):
        shadow = FaissFlat(big)
    elif isinstance(e, FaissHnsw):
        shadow = FaissHnsw(big)
    else:
        node["scale_4x_p50_ms"] = None
        save_state(state)
        return
    wall_s, _ = shadow.build() if not isinstance(shadow, ExactNumpy) else (0.0, 0.0)
    samples = []
    for q in Qn:
        t0 = time.perf_counter()
        shadow.search(q, K)
        samples.append((time.perf_counter() - t0) * 1000)
    node["scale_4x_p50_ms"] = lat(samples)["p50_ms"]
    node["scale_4x_build_s"] = round(wall_s, 3)
    node["scale_4x_n"] = int(len(big))
    save_state(state)
    print(f"[scale4x] {e.name}: {node['scale_4x_p50_ms']}ms p50", flush=True)


# ---------------------------------------------------------------------------
# turbovec verdict (filled from measured numbers)
# ---------------------------------------------------------------------------

def turbovec_verdict(state: dict) -> str:
    e = state["engines"]
    t4, exact, flat = e.get("turbovec_idmap_bw4", {}), e.get("exact_numpy", {}), e.get("faiss_flat_ip", {})
    if not (t4 and exact):
        return "insufficient data"
    ov = t4.get("unfiltered", {}).get("top10_overlap_vs_exact")
    mem = t4.get("memory", {}).get("index_bytes")
    raw = RAW_F32_BYTES
    comp = round(raw / mem, 1) if mem and mem > 0 else None
    f1 = t4.get("filtered_1pct", {})
    p50 = t4.get("unfiltered", {}).get("p50_ms")
    ep50 = exact.get("unfiltered", {}).get("p50_ms")
    return (f"turbovec bw4: recall vs exact = {ov}; index = {mem} B "
            f"({comp}x smaller than f32 raw {raw} B); unfiltered p50 {p50} ms vs "
            f"exact_numpy {ep50} ms; in-kernel allowlist top10 match vs subset-exact "
            f"= {f1.get('top10_match_vs_subset_exact')} at {f1.get('p50_ms')} ms; "
            f"id-removal supported in place")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global NQUERIES  # noqa: PLW0603
    if args.smoke:
        NQUERIES = 8
        global OUT_PATH, STATE_PATH  # noqa: PLW0603
        OUT_PATH = RESULTS / "turbovec_ablation_smoke.json"
        STATE_PATH = RESULTS / "turbovec_ablation_state_smoke.json"
    X, Qn, ids, filing_ids = load_data(args.smoke)
    records = build_records(Qn, ids, filing_ids)
    state = load_state()
    state.setdefault("n_vectors", int(len(X)))
    state.setdefault("dim", DIM)
    state.setdefault("provenance", provenance())

    exact_ref = exact_topk(X, Qn)
    id_to_filing = {i: doc for doc, fids in filing_ids.items() for i in fids}
    # align questions with filings: query i belongs to its own filing via Part A order
    # records map qid -> filing; allowlist for qi = filing of that question
    qid_by_qi = records["qids"]
    qi_filing = [records["qid_filing"][qid] for qid in qid_by_qi]

    rng = np.random.default_rng(0)
    for e in make_engines(X, args.smoke):
        phase_ingest(state, e)
        phase_memory(state, e, e.name)
        phase_unfiltered(state, e, Qn, exact_ref, records, ids)
        phase_filtered(state, e, Qn, filing_ids, ids, qi_filing, rng)
        phase_deletes(state, e, Qn)
        phase_scale(state, e, Qn)

    state["generated"] = now()
    state["turbovec_verdict"] = turbovec_verdict(state)
    state["deviations"] = (state.get("deviations", []) or []) + DEV_NOTE
    save_state(state)
    res = {"generated": state["generated"], "n_vectors": len(X), "dim": DIM,
           "query_vectors": "persisted in state, embedded via /v1/embeddings once (Part A)",
           "engines": state["engines"],
           "turbovec_verdict": state["turbovec_verdict"],
           "deviations": state["deviations"],
           "provenance": state["provenance"]}
    OUT_PATH.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT_PATH}")
    for name, node in res["engines"].items():
        u = node.get("unfiltered", {})
        f1 = node.get("filtered_1pct", {})
        print(f"{name:22s} p50={u.get('p50_ms')}ms overlap={u.get('top10_overlap_vs_exact')} "
              f"idx={node.get('memory', {}).get('index_bytes')}B "
              f"f1match={f1.get('top10_match_vs_subset_exact')} f1p50={f1.get('p50_ms')}ms")


def build_records(Qn: np.ndarray, ids: list[str], filing_ids: dict) -> dict:
    """Per-question scoring inputs: gold token sets + filing, aligned to Qn rows."""
    qst = json.loads(QVEC_STATE.read_text())
    qids = sorted(qst["query_vectors"])[:NQUERIES]
    rec_by_qid = {r["financebench_id"]: r for r in pc.load_questions(None)}
    raw_text: dict[str, str] = {}
    gold_by_qid: dict[str, list[list[str]]] = {}
    qid_filing: dict[str, str] = {}
    for doc in filing_ids:
        corpus = pc.load_filing_corpus(EMBEDDER, doc)
        for row in corpus:
            raw_text[row["chunk_id"]] = row["text"]
    for qid in qids:
        rec = rec_by_qid[qid]
        gold_by_qid[qid] = [sorted(g) for g in pc.gold_evidence(rec)]
        qid_filing[qid] = rec["doc_name"]
    return {"qids": qids, "raw_text": raw_text, "gold_by_qid": gold_by_qid,
            "qid_filing": qid_filing, "query_vectors_by_qid": qst["query_vectors"]}


if __name__ == "__main__":
    main()
