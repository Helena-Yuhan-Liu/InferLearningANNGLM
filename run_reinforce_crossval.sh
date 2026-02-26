#!/usr/bin/env bash
set -euo pipefail

for sd in 99 41 42 43; do
  for fold in 0 1 2 3 4; do
    jobid="run_fold${fold}_sd${sd}"
    echo "Running ${jobid}"
    python3 run_reinforce.py --fold "${fold}" --seed "${sd}" \
      > "stdout_${jobid}.log" \
      2> "stderr_${jobid}.log"
  done
done