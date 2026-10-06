"""Reset the served arms of the bench state so they recompute against a rebuilt binary.

`best_config_bench.py` skips any question whose arm verdict is already present,
so a stale state file strands a re-run on the previous binary's numbers. This
strips the named arms (default: the two that depend on the hybrid search
geometry) plus the self-search sanity block, which also exercises hybrid.

Existing artifact: the pre-change state is rotated to
results/prechange/ and the previous gate run's aggregates live in
results/mettle_best_config.json, so nothing is lost.

Usage:
  uv run python strip_state.py                    # strip hybrid + rerank_hybrid + sanity
  uv run python strip_state.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).with_name("results")
STATE = RESULTS / "mettle_best_config_state.json"

# Arm-key prefixes to invalidate. `hybrid` and `hybrid_ms` are the stage-1 arm;
# rerank_hybrid(...) is the rerank arm whose candidate pool comes from it.
PREFIXES = ("hybrid", "rerank_hybrid(")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    state = json.loads(STATE.read_text())
    removed: dict[str, int] = {}
    for row in state["questions"].values():
        for key in list(row):
            if key.startswith(PREFIXES):
                row.pop(key)
                removed[key.split("(")[0]] = removed.get(key.split("(")[0], 0) + 1
    had_sanity = bool(state.get("sanity"))
    state["sanity"] = {}

    print(f"questions: {len(state['questions'])}")
    print(f"removed: {removed}")
    print(f"sanity block cleared: {had_sanity}")
    if args.dry_run:
        print("dry run — not written")
        return
    STATE.write_text(json.dumps(state, indent=2))
    print(f"wrote {STATE}")


if __name__ == "__main__":
    main()
