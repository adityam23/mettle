"""Compare a rerank arm's recall on the same questions between bench states.

Why: `best_config_bench.py` scores questions filing-by-filing, so a partial
re-run covers only a prefix of the filings. Comparing that prefix against the
SAME question ids in the pre-change state separates a real regression from
filing-order noise. The pre-change state is the rotated
`results/prechange/mettle_best_config_state.json`.

Also reports which candidate pool each rerank call saw (the pool the new
`hybrid` mode returns), since rerank recall is bounded by pool quality.

Usage: uv run python rerank_arm_compare.py [arm-name]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
NEW = RESULTS / "mettle_best_config_state.json"
OLD = RESULTS / "prechange" / "mettle_best_config_state.json"

DEFAULT_ARM = "rerank_hybrid(qwen3-reranker-0.6b:latest)"


def hits(rows: dict[str, dict], qids: list[str], arm: str) -> tuple[int, int]:
    got = [rows[q][arm]["hits"]["@10_t0.5"] for q in qids if arm in rows.get(q, {})]
    return sum(1 for g in got if g), len(got)


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARM
    new = json.loads(NEW.read_text())["questions"]
    old = json.loads(OLD.read_text())["questions"]

    done = [q for q in new if arm in new[q]]
    if not done:
        raise SystemExit(f"no {arm!r} rows in the re-run state")

    print(f"arm: {arm}")
    print(f"re-run coverage: {len(done)}/150 questions (filing-order prefix)\n")

    for label, rows in (("re-run   ", new), ("prechange", old)):
        n, tot = hits(rows, done, arm)
        print(f"{label} same {tot:3d} q -> {n:3d}/{tot} = {n / tot:.4f}" if tot else
              f"{label} same   0 q -> n/a")

    n_all, tot_all = hits(old, list(old), arm)
    print(f"\nprechange all 150 q -> {n_all}/{tot_all} = {n_all / tot_all:.4f}")

    # Where the two runs disagree, on the shared prefix.
    flips = [(q, old[q][arm]["hits"]["@10_t0.5"], new[q][arm]["hits"]["@10_t0.5"])
             for q in done if q in old and arm in old[q]
             and old[q][arm]["hits"]["@10_t0.5"] != new[q][arm]["hits"]["@10_t0.5"]]
    up = sum(1 for _, o, n in flips if n and not o)
    down = sum(1 for _, o, n in flips if o and not n)
    print(f"\nper-question flips on the prefix: +{up} / -{down} ({len(flips)} total)")
    for q, o, n in flips[:12]:
        print(f"  {q}: {o} -> {n}")

    # Pool composition: how much of the reranked candidate list is novel vs the
    # dense top-40 the question already had.
    pool_sizes = [len(new[q][arm].get("candidate_ids", [])) for q in done
                  if "candidate_ids" in new[q][arm]]
    if pool_sizes:
        print(f"\nrerank pool sizes recorded: {sorted(set(pool_sizes))}")


if __name__ == "__main__":
    main()
