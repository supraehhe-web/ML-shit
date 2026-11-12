"""Per-job spend ledger for this revision, built from artifacts/logs/run_registry.jsonl.

Cost is an estimate: wall-clock seconds of each recorded job times the per-hour
price in dm_ahr_vit.budget (modal.com/pricing as recorded in reports/compute_plan.md).
It excludes container start-up, image builds and volume storage, so it is a
floor, not an invoice. Jobs before `--since` belong to earlier phases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.budget import USD_PER_HOUR  # noqa: E402
from dm_ahr_vit.config import save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402

CAP_USD = 5.50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-12T14:40:00+00:00")
    ap.add_argument("--cap", type=float, default=CAP_USD)
    args = ap.parse_args()
    path = artifact_dir("logs") / "run_registry.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df = df[df["timestamp"] >= args.since].copy()
    gpu_jobs = df[df["gpu"].isin(USD_PER_HOUR)].copy()
    gpu_jobs["usd_estimate"] = gpu_jobs.apply(
        lambda r: r["seconds"] / 3600.0 * USD_PER_HOUR[r["gpu"]], axis=1)
    keep = [c for c in ("timestamp", "job", "gpu", "seconds", "usd_estimate", "arm", "seed", "stage",
                        "tag", "iterations", "images") if c in gpu_jobs.columns]
    ledger = gpu_jobs[keep].sort_values("timestamp")
    ledger.round(4).to_csv(artifact_dir("logs") / "spend_ledger.csv", index=False)
    total = float(ledger["usd_estimate"].sum())
    save_json({"since": args.since, "jobs": int(len(ledger)), "estimated_usd_total": round(total, 3),
               "cap_usd": args.cap, "within_cap": bool(total <= args.cap),
               "basis": "wall-clock seconds x recorded Modal per-hour price; excludes start-up and storage"},
              artifact_dir("logs") / "spend_summary.json")
    print(ledger.round(3).to_string(index=False))
    print(f"estimated total ${total:.2f} (cap ${args.cap:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
