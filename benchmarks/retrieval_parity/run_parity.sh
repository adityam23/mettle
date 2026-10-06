#!/usr/bin/env bash
# Resumable wrapper for the mettle-side retrieval parity benchmark.
# Survives session exit (run detached). Harness resumes from results/mettle_state.json.
set -u
cd /personal-projects/mettle/benchmarks/retrieval_parity || exit 1
for attempt in $(seq 1 40); do
  uv run python -u mettle_bench.py 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "RUN COMPLETE"; exit 0; fi
  if [ $rc -eq 3 ]; then echo "FATAL SANITY FAILURE — aborting wrapper"; exit 3; fi
  echo "[wrapper] attempt $attempt exited rc=$rc — resuming in 60s"
  sleep 60
done
echo "wrapper gave up after 40 attempts"
exit 1
