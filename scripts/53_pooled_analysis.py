"""Stage E pooled analysis over seeds (frozen design in reports/prereg_v4_review_response.md).

Arms A, A*, B, Cv2 at eight seeds each. Two-level bootstrap (lesion groups and seeds),
10,000 replicates, seed 20261113. Endpoints: test macro-F1 and per-image log-loss, full and
hair-present subsets. Families: planned = A-B; secondary control = A-A*; exploratory = Cv2 rows.
Also writes the per-seed table and a seed-fixed (earlier-style) CI beside the two-level one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import CLASSES, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.stats.pooled import (  # noqa: E402
    compare,
    draw_weights,
    f1_scores,
    group_counts,
    group_loss,
    loss_scores,
    macro_f1,
    one_level,
    two_level,
)
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()
ARMS = ("A", "Astar", "B", "Cv2")
SEEDS = (42, 123, 2026, 7, 11, 17, 23, 31)
REPS, BOOT_SEED = 10_000, 20261113
COMPARISONS = [("A", "B", "planned"), ("A", "Astar", "secondary_control"),
               ("A", "Cv2", "exploratory"), ("B", "Cv2", "exploratory"),
               ("Cv2", "Astar", "exploratory")]


def load(arm: str, seed: int, subset: str) -> pd.DataFrame:
    return pd.read_csv(artifact_dir("predictions") / f"pred_{arm}_seed{seed}_{subset}.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--commit", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rows, per_seed = [], []
    for subset in ("full", "hair_present"):
        data = {(a, s): load(a, s, subset) for a in ARMS for s in seeds}
        ids = None
        for df in data.values():
            ids = set(df["image_id"]) if ids is None else ids & set(df["image_id"])
        data = {k: v[v["image_id"].isin(ids)].sort_values("image_id").reset_index(drop=True)
                for k, v in data.items()}
        ref = data[(ARMS[0], seeds[0])]
        for df in data.values():
            assert list(df["image_id"]) == list(ref["image_id"]), "arms differ in test images"
        groups = sorted(ref["group_id"].unique())
        w = draw_weights(len(groups), REPS, BOOT_SEED)
        ones = np.ones((1, len(groups)), dtype=np.float32)
        counts = {a: [group_counts(data[(a, s)], groups) for s in seeds] for a in ARMS}
        losses = {a: [group_loss(data[(a, s)], groups) for s in seeds] for a in ARMS}
        obs_f1 = {a: f1_scores(counts[a], ones)[:, 0] for a in ARMS}
        obs_ll = {a: loss_scores(losses[a], ones)[:, 0] for a in ARMS}
        for a in ARMS:
            for j, s in enumerate(seeds):
                d = data[(a, s)]
                per_seed.append({"arm": a, "seed": s, "subset": subset, "n": len(d),
                                 "macro_f1": obs_f1[a][j], "log_loss": obs_ll[a][j],
                                 "accuracy": float((d["label"] == d["pred"]).mean())})
        rng = np.random.default_rng(BOOT_SEED + (1 if subset == "full" else 2))
        for endpoint, obs, scores_of in (("macro_f1", obs_f1, lambda a: f1_scores(counts[a], w)),
                                         ("log_loss", obs_ll, lambda a: loss_scores(losses[a], w))):
            scores = {a: scores_of(a) for a in ARMS}
            dist2 = {a: two_level(scores[a], rng) for a in ARMS}
            dist1 = {a: one_level(scores[a]) for a in ARMS}
            for a, b, family in COMPARISONS:
                r = compare(dist2[a], dist2[b], float(obs[a].mean()), float(obs[b].mean()))
                fixed = compare(dist1[a], dist1[b], float(obs[a].mean()), float(obs[b].mean()))
                row = {"endpoint": endpoint, "subset": subset, "family": family,
                       "comparison": f"{a}-{b}", "arm_a_mean": float(obs[a].mean()),
                       "arm_b_mean": float(obs[b].mean()), "arm_a_seed_sd": float(obs[a].std(ddof=1)),
                       "arm_b_seed_sd": float(obs[b].std(ddof=1)), **r,
                       "ci_low_seeds_fixed": fixed["ci_low"], "ci_high_seeds_fixed": fixed["ci_high"],
                       "n_seeds": len(seeds), "replicates": REPS}
                if endpoint == "log_loss":
                    ga = np.mean([ls / np.maximum(n, 1) for ls, n in losses[a]], axis=0)
                    gb = np.mean([ls / np.maximum(n, 1) for ls, n in losses[b]], axis=0)
                    d = ga - gb
                    row["dz_over_lesion_groups"] = float(d.mean() / d.std(ddof=1))
                rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, g in out.groupby(["endpoint", "subset", "family"]):
        out.loc[g.index, "p_holm"] = holm(g["p_raw"].tolist())
    out["established"] = (out["p_holm"] < 0.05) & ((out["ci_low"] > 0) | (out["ci_high"] < 0))
    out["underpowered_for_0.02"] = out["mdd_80pct"] > 0.02
    ps = pd.DataFrame(per_seed)

    # sanity: per-seed macro-F1 here equals the metric files written at evaluation time
    worst = 0.0
    for a in ARMS:
        for s in seeds:
            m = json.loads((artifact_dir("predictions") / f"metrics_{a}_seed{s}.json").read_text())
            ref = m["metrics"][f"{a}|{s}|full"]["macro_f1"]
            mine = float(ps[(ps.arm == a) & (ps.seed == s) & (ps.subset == "full")]["macro_f1"].iloc[0])
            worst = max(worst, abs(ref - mine))
    assert worst < 1e-6, f"per-seed macro-F1 disagrees with metric files by {worst}"

    commit = args.commit or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                           text=True, cwd=ROOT).stdout.strip()
    logs = artifact_dir("logs")
    out.to_csv(logs / "pooled_seed_analysis.csv", index=False)
    ps.to_csv(logs / "pooled_per_seed.csv", index=False)
    save_json({"seeds": seeds, "arms": list(ARMS), "replicates": REPS, "bootstrap_seed": BOOT_SEED,
               "git_commit": commit, "max_abs_diff_vs_metric_files": worst,
               "design": "two-level bootstrap: lesion groups shared across arms, seeds resampled "
                         "within arm; p = 2*min tail with +1 smoothing; Holm per endpoint, subset, family"},
              logs / "pooled_seed_analysis.json")
    pd.set_option("display.width", 250)
    print(out[["endpoint", "subset", "family", "comparison", "arm_a_mean", "arm_b_mean", "difference",
               "ci_low", "ci_high", "ci_low_seeds_fixed", "ci_high_seeds_fixed", "p_holm", "mdd_80pct",
               "established"]].round(4).to_string(index=False))
    print(ps[ps.subset == "full"].groupby("arm")["macro_f1"].agg(["mean", "std", "min", "max", "count"]).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
