"""Shared metric + data loading for the mettle/docket retrieval parity benchmark.

The metric functions below are VERBATIM copies of the docket harness
(/personal-projects/docket/benchmarks/financebench/sweep_recall.py and
docket/retrieval/retriever.py, state 2026-08-29). Equivalence of the
benchmark depends on both sides scoring with the exact same functions;
any change here MUST be mirrored there (see METTLE_PARITY_CONTRACT.md in
the docket repo).

Provenance of each copy is noted inline. Do not "improve" them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths into the docket repo (READ-ONLY — this harness never writes there).
# ---------------------------------------------------------------------------
DOCKET_ROOT = Path("/personal-projects/docket")
FINANCEBENCH = DOCKET_ROOT / "benchmarks" / "financebench"
QUESTIONS_PATH = FINANCEBENCH / "data" / "financebench_merged.jsonl"
SWEEP_CACHE = FINANCEBENCH / ".sweep_cache"
FILINGS_DIR = FINANCEBENCH / "filings"

# embedder model id -> docket sweep-cache dir slug (their naming, verbatim)
EMBEDDER_CACHE_SLUG = {
    "embed-gemma:latest": "embed-gemma-latest",
    "bge-m3:latest": "bge-m3-latest",
    "qwen3-embedding-0.6b:latest": "qwen3-embedding-0-6b-latest",
}

RERANKERS = ["bge-reranker-v2-m3:latest", "qwen3-reranker-0.6b:latest"]

KS = (3, 5, 6, 10, 20)
THRESHOLDS = (0.5, 0.8)
CANDIDATES = 40  # docket Retriever default candidate pool feeding rerank

# ---------------------------------------------------------------------------
# Metric functions — VERBATIM from docket.
# ---------------------------------------------------------------------------

# docket/retrieval/retriever.py:28,34-36
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# sweep_recall.py:54-59 (_EMBED_CHAR_CAP / _CTRL / _embed_clean)
_EMBED_CHAR_CAP = 2000
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def embed_clean(t: str) -> str:
    return _CTRL.sub(" ", t)[:_EMBED_CHAR_CAP]


# sweep_recall.py:98-105 (_gold_evidence)
def gold_evidence(rec: dict) -> list[set]:
    out = []
    for e in rec.get("evidence") or []:
        if isinstance(e, dict) and e.get("evidence_text"):
            toks = set(tokenize(e["evidence_text"]))
            if toks:
                out.append(toks)
    return out


# sweep_recall.py:108-113 (_overlap_coeff)
def overlap_coeff(a: set, b: set) -> float:
    """|a∩b| / min(|a|,|b|) — size-robust: a chunk fully inside a big evidence
    table still scores 1.0, so table-shaped evidence isn't unfairly penalised."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# sweep_recall.py:116-121 (_recall_hit)
def recall_hit(hit_token_sets: list[set], golds: list[set], thresh: float) -> bool:
    for cs in hit_token_sets:
        for g in golds:
            if overlap_coeff(cs, g) >= thresh:
                return True
    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    """FinanceBench 150-Q open subset, as used by the docket harness."""
    recs = [json.loads(l) for l in open(QUESTIONS_PATH) if l.strip()]
    if limit:
        recs = recs[:limit]
    return recs


def load_filing_corpus(embedder: str, doc_name: str) -> list[dict]:
    """Frozen docket chunks (+ their cached mettle-produced vectors) for one
    filing. Rows: {"chunk_id","text","page","vector"}.

    text is the RAW stored chunk text (what docket's BM25 + recall scoring
    use); vectors were produced by the same GGUF models this benchmark serves.
    """
    slug = EMBEDDER_CACHE_SLUG[embedder]
    p = SWEEP_CACHE / slug / doc_name / "corpus.jsonl"
    if not p.is_file():
        raise FileNotFoundError(p)
    out = []
    for line in open(p):
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "page": row.get("page"),
                "vector": row["vector"],
            }
        )
    return out


def filing_list() -> list[str]:
    """All filings present in the sweep cache (84), sorted like docket does."""
    slug = EMBEDDER_CACHE_SLUG["embed-gemma:latest"]
    return sorted(p.name for p in (SWEEP_CACHE / slug).iterdir() if p.is_dir())


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def rank_by_query(vecs: list[list[float]], qv: list[float]) -> list[int]:
    """Exact cosine ranking (docket Corpus.dense_search equivalent): returns
    chunk indices best-first. Pure-python brute force — same algorithm as
    docket's production lite path."""
    scored = sorted(
        range(len(vecs)), key=lambda i: cosine(vecs[i], qv), reverse=True
    )
    return scored
