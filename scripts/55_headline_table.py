"""Headline test metrics per arm, pooled over every available seed, against the reviewer's ranges.

Reads artifacts/predictions/metrics_*.json and pred_*_full.csv. The ranges are a reviewer's rough
estimates (reports/prereg_v4_review_response.md, section 7), shown for context, not as targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.paths import artifact_dir  # noqa: E402

PRED = artifact_dir("predictions")
ARMS = {"A": 8, "Astar": 8, "B": 8, "Cv2": 8, "Creleased": 3, "C": 3}
SEEDS_ALL = (42, 123, 2026, 7, 11, 17, 23, 31)
RANGES = {"accuracy": (0.82, 0.90), "macro_f1": (0.65, 0.80), "balanced_accuracy": (0.65, 0.80),
          "auc_macro": (0.94, 0.98), "melanoma_recall": (0.55, 0.80)}
SD_LIMIT = 0.02


def seeds_for(arm: str) -> list[int]:
    return [s for s in SEEDS_ALL if (PRED / f"metrics_{arm}_seed{s}.json").exists()]


def main() -> int:
    rows = []
    for arm in ARMS:
        seeds = seeds_for(arm)
        vals = {k: [] for k in ("accuracy", "macro_f1", "balanced_accuracy", "auc_macro", "melanoma_recall")}
        for s in seeds:
            m = json.loads((PRED / f"metrics_{arm}_seed{s}.json").read_text())["metrics"][f"{arm}|{s}|full"]
            for k in ("accuracy", "macro_f1", "balanced_accuracy", "auc_macro"):
                vals[k].append(m[k])
            d = pd.read_csv(PRED / f"pred_{arm}_seed{s}_full.csv")
            mel = d[d["label"] == "mel"]
            vals["melanoma_recall"].append(float((mel["pred"] == "mel").mean()))
        row = {"arm": arm, "seeds": len(seeds)}
        for k, v in vals.items():
            row[f"{k}_mean"], row[f"{k}_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
            lo, hi = RANGES[k]
            row[f"{k}_vs_range"] = ("below" if row[f"{k}_mean"] < lo else
                                    "above" if row[f"{k}_mean"] > hi else "inside")
        row["macro_f1_seed_sd_within_0.02"] = row["macro_f1_sd"] <= SD_LIMIT
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(artifact_dir("logs") / "headline_metrics_pooled.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["arm", "seeds"] + [f"{k}_{x}" for k in RANGES for x in ("mean", "vs_range")] + [
        "macro_f1_sd", "macro_f1_seed_sd_within_0.02", "melanoma_recall_sd"]
    print(df[show].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
