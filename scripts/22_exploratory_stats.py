"""Exploratory statistics for the released-checkpoint arm (Creleased).

Creleased was added after the pre-registered protocol was frozen, so it is
reported separately: the planned family in `09_stats.py` is left untouched and
these rows carry an explicit exploratory label.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()

LABEL = ("released-checkpoint arm, 25-step respaced sampling, exploratory")
COMPARISONS = [("A", "Creleased"), ("B", "Creleased")]


def load_stats_module():
    """Import 09_stats.py so the frozen methodology is reused verbatim."""
    path = Path(__file__).resolve().parent / "09_stats.py"
    spec = importlib.util.spec_from_file_location("stats09", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_stats_module()
    mod.ARMS = ["A", "B", "Creleased"]

    protocol = load_config("configs/protocol.json")
    seeds = list(protocol["seeds"])
    reps = int(protocol["bootstrap_replicates"])
    bseed = int(protocol["bootstrap_seed"])

    out: list = []
    with timed("exploratory_stats", log) as t:
        for subset in ("full", "hair_present"):
            out += mod.classification_endpoint(seeds, subset, reps, bseed, COMPARISONS)

        p = artifact_dir("logs") / "paired_psnr_ssim_test.csv"
        if p.exists():
            df = pd.read_csv(p)
            for metric in ("psnr", "ssim"):
                col = f"Creleased_{metric}"
                if col not in df.columns or not df[col].notna().any():
                    continue
                out += mod.paired_column_endpoint(
                    df, col, f"B_{metric}", f"paired_{metric}", reps, bseed,
                    [("Creleased", "B")])
                out += mod.paired_column_endpoint(
                    df, col, f"hairy_input_{metric}", f"paired_{metric}", reps, bseed,
                    [("Creleased", "hairy input")])

    if not out:
        log.error("no exploratory comparisons could be computed")
        return 1

    table = pd.DataFrame(out)
    table["family"] = "exploratory_creleased"
    table["p_holm"] = np.nan
    for _, grp in table.groupby(["endpoint", "subset"]):
        table.loc[grp.index, "p_holm"] = holm(grp["p_raw"].tolist())
    table["note"] = LABEL
    table["negligible"] = (table["p_holm"] < 0.05) & (table["effect_size_dz"].abs() < 0.2)

    cols = ["endpoint", "subset", "family", "comparison", "arm_a", "arm_b",
            "arm_a_mean", "arm_b_mean", "difference", "ci_low", "ci_high",
            "p_raw", "p_holm", "effect_size_dz", "wilcoxon_p_raw",
            "wilcoxon_p_holm", "wilcoxon_statistic", "wilcoxon_n_groups",
            "n_images", "n_groups", "replicates", "test", "negligible", "note"]
    table = table[[c for c in cols if c in table.columns]]
    table = table.sort_values(["endpoint", "subset", "comparison"])
    table.to_csv(artifact_dir("logs") / "statistical_comparisons_exploratory.csv", index=False)
    save_json({"seconds": t["seconds"], "rows": int(len(table)), "note": LABEL},
              artifact_dir("logs") / "stats_exploratory_summary.json")
    record_run("exploratory_stats", gpu=None, seconds=t["seconds"], rows=int(len(table)))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
