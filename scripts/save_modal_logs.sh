#!/usr/bin/env bash
# Save the console logs of this revision's Modal jobs, minus progress-bar noise.
set -euo pipefail
dest="$(dirname "$0")/../artifacts/logs/modal_jobs"
mkdir -p "$dest"
copy() { # copy <src> <name>
  if [ -f "$1" ]; then grep -v -E "sampling loop time step|it/s\]|MB/s\]|Created Function|Uploaded|Downloading|Installing" "$1" > "$dest/$2.log" || true; fi
}
copy /tmp/j1_smoke.log p1_smoke
copy /tmp/j1.log p1_paper_reproduction_128
copy /tmp/j2.log retrain_first_attempt_stopped
copy /tmp/j4.log p1b_paper_reproduction_256
copy /tmp/j5.log p1c_fp32_control
copy /tmp/j_sweep.log p2_step_sweep
copy /tmp/j_warm_start.log p2_warm_start
copy /tmp/j_overfit.log p2_overfit
copy /tmp/j_mse.log p2_noise_mse
copy /tmp/j_score.log p2_score_snapshots
copy /tmp/j_respace.log p2_respaced_validation
for a in A Astar B C; do copy /tmp/j_xai_$a.log xai_v2_$a; done
for f in /tmp/j_extra_*.log; do [ -f "$f" ] && copy "$f" "$(basename "$f" .log | sed 's/^j_extra_//')"; done
ls "$dest"
