"""Per-job spend ledger for this revision, built from the saved Modal job logs.

The volume's run_registry.jsonl cannot be trusted for this: concurrent containers
each append to their own snapshot of the file and the last commit wins, so most
jobs of a parallel batch are missing from it. Instead every job's console log is
saved under artifacts/logs/modal_jobs/ and the wall-clock printed by the
container wrapper (`EXIT <code> in <s>s`) is used.

Cost is an estimate: container wall-clock seconds times the per-hour price in
dm_ahr_vit.budget (modal.com/pricing as recorded in reports/compute_plan.md). It
excludes container start-up, image builds and volume storage, so it is a floor,
not an invoice. Jobs that were stopped by the operator have no EXIT line; their
seconds come from `modal_jobs/overrides.json` and are flagged as estimates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.budget import USD_PER_HOUR  # noqa: E402
from dm_ahr_vit.config import save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402

CAP_USD = 6.30
EXIT_RE = re.compile(r"^EXIT (\d+) in ([0-9.]+)s", re.M)
GPU_RE = re.compile(r"--gpu (\S+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=CAP_USD)
    args = ap.parse_args()
    d = artifact_dir("logs") / "modal_jobs"
    overrides = json.loads((d / "overrides.json").read_text()) if (d / "overrides.json").exists() else {}
    rows = []
    for p in sorted(d.glob("*.log")):
        text = p.read_text(errors="ignore")
        gpu_m = GPU_RE.search(text)
        gpu = gpu_m.group(1) if gpu_m else "A100-40GB"   # gpu_script default
        exits = EXIT_RE.findall(text)
        # A run_chain job prints several EXITs; the container wrapper's is the last one.
        if exits:
            code, secs, basis = int(exits[-1][0]), float(exits[-1][1]), "container wall-clock (EXIT line)"
        elif p.stem in overrides:
            code, secs, basis = -1, float(overrides[p.stem]["seconds"]), "ESTIMATE: " + overrides[p.stem]["why"]
        else:
            code, secs, basis = -2, 0.0, "no EXIT line: unfinished or failed, not counted"
        usd = secs / 3600.0 * USD_PER_HOUR[gpu]
        rows.append({"job": p.stem, "gpu": gpu, "exit_code": code, "seconds": round(secs, 1),
                     "usd_estimate": round(usd, 3), "basis": basis})
    ledger = pd.DataFrame(rows)
    ledger.to_csv(artifact_dir("logs") / "spend_ledger.csv", index=False)
    total = float(ledger["usd_estimate"].sum())
    save_json({"jobs": int(len(ledger)), "estimated_usd_total": round(total, 3), "cap_usd": args.cap,
               "within_cap": bool(total <= args.cap),
               "basis": "container wall-clock x recorded Modal per-hour price; excludes start-up, image builds and storage"},
              artifact_dir("logs") / "spend_summary.json")
    print(ledger.to_string(index=False))
    print(f"estimated total ${total:.2f} (cap ${args.cap:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
