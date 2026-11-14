"""Assemble the statistical comparison table.

Paired lesion-group bootstrap with the same resample shared across arms and
seeds, Holm correction across the three planned comparisons per endpoint, plus
group-aggregated Wilcoxon tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm, wilcoxon_per_group  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
ARMS = ["A", "B", "C", "Astar"]
# Stage C scores a second diffusion arm in the role of C; see main(--c-arm).
C_ARM = "C"
SUFFIX = ""


def as_c_arm(pairs: list) -> list:
    return [tuple(C_ARM if x == "C" else x for x in pair) for pair in pairs]


def _long_from_labels(df: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """Long (arm, group, value) frame from same-row columns given as (col, label)."""
    frames = []
    for col, label in pairs:
        if col not in df.columns:
            continue
        g = df.dropna(subset=["group_id", col]).groupby("group_id")[col].mean()
        frames.append(pd.DataFrame({"arm": label, "group_id": g.index,
                                    "value": g.to_numpy(dtype=float)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _long_from_arm(df: pd.DataFrame, value_col: str, arm_col: str) -> pd.DataFrame:
    """Long (arm, group, value) frame from a tidy per-image table."""
    g = (df.dropna(subset=["group_id", value_col])
         .groupby([arm_col, "group_id"])[value_col].mean().reset_index())
    return g.rename(columns={arm_col: "arm", value_col: "value"})


def attach_wilcoxon(rows: list, long_df: pd.DataFrame, comparisons: list) -> list:
    """Attach group-aggregated Wilcoxon results, Holm-corrected within the family.

    The proposal asks for a Wilcoxon signed-rank test beside the bootstrap CI.
    Values are collapsed to one per lesion group before testing.
    """
    if not rows or long_df is None or long_df.empty:
        return rows
    res, pvals = {}, []
    for a, b in comparisons:
        r = wilcoxon_per_group(long_df, a, b)
        res[f"{a}-{b}"] = r
        pvals.append(r["wilcoxon_p_raw"])
    adj = holm(pvals)
    keys = list(res)
    for row in rows:
        r = res.get(row["comparison"])
        if r is None:
            continue
        row.update(r)
        row["wilcoxon_p_holm"] = adj[keys.index(row["comparison"])]
        row["wilcoxon_test"] = "wilcoxon_signed_rank_group_aggregated"
    return rows


def load_predictions(arm: str, subset: str, seeds: list[int]):
    frames = []
    for seed in seeds:
        p = artifact_dir("predictions") / f"pred_{arm}_seed{seed}_{subset}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def group_class_counts(df: pd.DataFrame, groups: list[str]) -> dict:
    """Per-group TP/FP/FN per class so a bootstrap becomes one matmul."""
    gidx = {g: i for i, g in enumerate(groups)}
    n_classes = len(CLASSES)
    tp = np.zeros((len(groups), n_classes))
    fp = np.zeros((len(groups), n_classes))
    fn = np.zeros((len(groups), n_classes))
    y_true = np.array([CLASSES.index(v) for v in df["label"]])
    y_pred = np.array([CLASSES.index(v) for v in df["pred"]])
    gi = df["group_id"].map(gidx).to_numpy()
    for k in range(len(df)):
        g, t, p = int(gi[k]), int(y_true[k]), int(y_pred[k])
        if t == p:
            tp[g, t] += 1.0
        else:
            fp[g, p] += 1.0
            fn[g, t] += 1.0
    return {"tp": tp, "fp": fp, "fn": fn}


def macro_f1_from_counts(tp, fp, fn) -> np.ndarray:
    denom = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return per_class.mean(axis=1)


def draw_counts(n_groups: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(reps, n_groups))
    counts = np.zeros((reps, n_groups), dtype=np.float32)
    for r in range(reps):
        np.add.at(counts[r], idx[r], 1.0)
    return counts


def compare(dist_a: np.ndarray, dist_b: np.ndarray, alpha: float = 0.05,
            paired_diffs: np.ndarray | None = None) -> dict:
    """Bootstrap CI/p for a difference; Cohen's dz from observed pairs.

    dz is computed on the observed paired differences, not on the bootstrap
    distribution, whose spread shrinks with replicate count and would inflate
    the effect size without bound.
    """
    diff = dist_a - dist_b
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"difference": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_raw": float("nan"),
                "effect_size_dz": float("nan")}
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
    mean_diff = float(np.nanmean(dist_a) - np.nanmean(dist_b))
    dz = float("nan")
    if paired_diffs is not None:
        d = np.asarray(paired_diffs, dtype=float)
        d = d[np.isfinite(d)]
        if d.size > 1 and d.std(ddof=1) > 0:
            dz = float(d.mean() / d.std(ddof=1))
    return {
        "difference": mean_diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_raw": float(min(1.0, p)),
        "effect_size_dz": dz,
    }


def bootstrap_arm_scores(per_seed_counts: list, W: np.ndarray) -> np.ndarray:
    """Bootstrap distribution of the seed-averaged macro-F1 for one arm."""
    scores = []
    for counts in per_seed_counts:
        tp = W @ counts["tp"]
        fp = W @ counts["fp"]
        fn = W @ counts["fn"]
        scores.append(macro_f1_from_counts(tp, fp, fn))
    return np.mean(np.vstack(scores), axis=0)


def classification_endpoint(seeds: list[int], subset: str, reps: int, bseed: int,
                            comparisons: list) -> list:
    preds = {arm: load_predictions(arm, subset, seeds) for arm in ARMS}
    available = [a for a in ARMS if preds[a] is not None]
    if len(available) < 2:
        log.warning("classification/%s: only arms %s available", subset, available)
        return []
    common = None
    for arm in available:
        ids = set(preds[arm]["image_id"])
        common = ids if common is None else (common & ids)
    common = sorted(common)
    groups = sorted({g for arm in available
                     for g in preds[arm].loc[preds[arm]["image_id"].isin(common),
                                             "group_id"].unique()})
    W = draw_counts(len(groups), reps, bseed)

    dists, observed = {}, {}
    per_group_f1: dict[str, np.ndarray] = {}
    correct_long: list[dict] = []
    for arm in available:
        per_seed, seed_scores = [], []
        correct_tables = []
        for seed in seeds:
            sub = preds[arm][(preds[arm]["seed"] == seed)
                             & (preds[arm]["image_id"].isin(common))]
            if sub.empty:
                continue
            counts = group_class_counts(sub, groups)
            per_seed.append(counts)
            seed_scores.append(float(macro_f1_from_counts(
                counts["tp"].sum(0)[None], counts["fp"].sum(0)[None],
                counts["fn"].sum(0)[None])[0]))
            correct = (sub["label"] == sub["pred"]).astype(float)
            correct_tables.append(correct.groupby(sub["group_id"]).mean())
        if not per_seed:
            continue
        dists[arm] = bootstrap_arm_scores(per_seed, W)
        observed[arm] = float(np.mean(seed_scores))
        # Per-group macro-F1, averaged over seeds, for the observed effect size.
        per_group_f1[arm] = np.mean(
            [macro_f1_from_counts(c["tp"], c["fp"], c["fn"]) for c in per_seed],
            axis=0,
        )
        if correct_tables:
            avg = pd.concat(correct_tables, axis=1).mean(axis=1)
            correct_long.extend({"arm": arm, "group_id": g, "value": float(v)}
                                for g, v in avg.items())

    rows = []
    for a, b in comparisons:
        if a not in dists or b not in dists:
            continue
        diffs = None
        if a in per_group_f1 and b in per_group_f1:
            diffs = per_group_f1[a] - per_group_f1[b]
        rows.append({
            "endpoint": "test_macro_f1",
            "subset": subset,
            "comparison": f"{a}-{b}",
            "arm_a": a,
            "arm_b": b,
            "arm_a_mean": observed[a],
            "arm_b_mean": observed[b],
            "n_images": len(common),
            "n_groups": len(groups),
            "replicates": reps,
            **compare(dists[a], dists[b], paired_diffs=diffs),
            "test": "lesion-group paired bootstrap, percentile CI, two-sided",
        })
    rows = attach_wilcoxon(rows, pd.DataFrame(correct_long), comparisons)
    save_json(
        {"subset": subset, "observed_macro_f1": observed,
         "arms": available, "n_images": len(common), "n_groups": len(groups)},
        artifact_dir("logs") / f"stats_classification_{subset}{SUFFIX}.json",
    )
    return rows


def bootstrap_group_means(df: pd.DataFrame, value: str, arm_col: str,
                          W: np.ndarray, groups: list, gidx: dict) -> dict:
    """Bootstrap mean of `value` per arm, resampling lesion groups."""
    dists, obs = {}, {}
    for arm, sub in df.groupby(arm_col):
        sub = sub.dropna(subset=[value])
        if sub.empty:
            continue
        gi = sub["group_id"].map(gidx).to_numpy()
        vals = sub[value].to_numpy(dtype=float)
        sums = np.zeros(len(groups))
        cnts = np.zeros(len(groups))
        np.add.at(sums, gi, vals)
        np.add.at(cnts, gi, 1.0)
        dists[arm] = _safe_ratio(W @ sums, W @ cnts)
        obs[arm] = float(vals.mean())
    return {"dists": dists, "obs": obs}


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Mean per bootstrap replicate, ignoring replicates that drew no members.

    A group-level resample can miss a group entirely; dividing by a clamped
    epsilon there would fabricate astronomically large values.
    """
    out = np.full(num.shape, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def paired_column_endpoint(df: pd.DataFrame, col_a: str, col_b: str, endpoint: str,
                           reps: int, bseed: int, comparisons: list,
                           arm_col=None) -> list:
    """Bootstrap a paired column comparison (two conditions in the same rows)."""
    if df.empty or col_a not in df.columns or col_b not in df.columns:
        return []
    df = df.dropna(subset=[col_a, col_b, "group_id"])
    if df.empty:
        return []
    groups = sorted(df["group_id"].unique())
    gidx = {g: i for i, g in enumerate(groups)}
    W = draw_counts(len(groups), reps, bseed)
    rows = []
    if arm_col is not None and arm_col in df.columns:
        stats_by_arm = bootstrap_group_means(df, col_a, arm_col, W, groups, gidx)
        for x, y in comparisons:
            if x not in stats_by_arm["dists"] or y not in stats_by_arm["dists"]:
                continue
            rows.append({
                "endpoint": endpoint, "subset": "test_x1_tier",
                "comparison": f"{x}-{y}", "arm_a": x, "arm_b": y,
                "arm_a_mean": stats_by_arm["obs"][x],
                "arm_b_mean": stats_by_arm["obs"][y],
                "n_images": int(df[df[arm_col] == x].shape[0]),
                "n_groups": len(groups), "replicates": reps,
                **compare(stats_by_arm["dists"][x], stats_by_arm["dists"][y]),
                "test": "lesion-group paired bootstrap, percentile CI, two-sided",
            })
        return attach_wilcoxon(
            rows, _long_from_arm(df, col_a, arm_col),
            [c for c in comparisons if c[0] in stats_by_arm["dists"]
             and c[1] in stats_by_arm["dists"]])

    gi = df["group_id"].map(gidx).to_numpy()
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)
    sa, sb = np.zeros(len(groups)), np.zeros(len(groups))
    ca, cb = np.zeros(len(groups)), np.zeros(len(groups))
    np.add.at(sa, gi, a)
    np.add.at(sb, gi, b)
    np.add.at(ca, gi, 1.0)
    np.add.at(cb, gi, 1.0)
    da = _safe_ratio(W @ sa, W @ ca)
    db = _safe_ratio(W @ sb, W @ cb)
    observed_diffs = a - b
    for x, y in comparisons:
        rows.append({
            "endpoint": endpoint, "subset": "held_out_pairs",
            "comparison": f"{x}-{y}", "arm_a": x, "arm_b": y,
            "arm_a_mean": float(a.mean()), "arm_b_mean": float(b.mean()),
            "n_images": int(len(df)), "n_groups": len(groups),
            "replicates": reps, **compare(da, db, paired_diffs=observed_diffs),
            "test": "lesion-group paired bootstrap, percentile CI, two-sided",
        })
    label_pairs = [(col_a, comparisons[0][0]), (col_b, comparisons[0][1])] \
        if comparisons else [(col_a, col_a), (col_b, col_b)]
    return attach_wilcoxon(rows, _long_from_labels(df, label_pairs), comparisons)


def xai_rows(metric: str, reps: int, bseed: int, comparisons: list) -> list:
    # `_v2_` files come from a separate diagnostic analysis with different
    # columns; mixing them into the reported endpoints would average two
    # different measurements of the same arm.
    files = sorted(f for f in artifact_dir("logs").glob("xai_localization_*.csv")
                   if "_v2_" not in f.name)
    if not files:
        return []
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # XAI runs recorded before group_id was written to the CSV: recover it from
    # the frozen split so explainability endpoints still aggregate by lesion
    # rather than by image. Backfill per row, because a mixed set of files can
    # have the column present but empty for the older runs.
    if "group_id" not in df.columns:
        df["group_id"] = pd.NA
    if df["group_id"].isna().any():
        masks = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
        mapping = dict(zip(masks["image_id"], masks["group_id"]))
        df["group_id"] = df["group_id"].fillna(df["image_id"].map(mapping))
    if metric not in df.columns or "group_id" not in df.columns:
        return []
    # One value per (arm, seed, group) before resampling.
    agg = df.groupby(["arm", "seed", "group_id"])[metric].mean().reset_index()
    # Average seeds first so seeds are not counted as independent samples.
    per_group = agg.groupby(["arm", "group_id"])[metric].mean().reset_index()
    groups = sorted(per_group["group_id"].unique())
    gidx = {g: i for i, g in enumerate(groups)}
    W = draw_counts(len(groups), reps, bseed)
    res = bootstrap_group_means(per_group, metric, "arm", W, groups, gidx)
    rows = []
    for a, b in comparisons:
        if a not in res["dists"] or b not in res["dists"]:
            continue
        rows.append({
            "endpoint": f"x1_{metric}", "subset": "test_x1",
            "comparison": f"{a}-{b}", "arm_a": a, "arm_b": b,
            "arm_a_mean": res["obs"][a], "arm_b_mean": res["obs"][b],
            "n_images": int((df["arm"] == a).sum()),
            "n_groups": len(groups), "replicates": reps,
            **compare(res["dists"][a], res["dists"][b]),
            "test": "lesion-group paired bootstrap, percentile CI, two-sided",
        })
    return attach_wilcoxon(rows, _long_from_arm(per_group, metric, "arm"), comparisons)


def main() -> int:
    import argparse

    global ARMS, C_ARM, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-arm", default="C", help="arm that plays the role of C (e.g. Cv2)")
    C_ARM = ap.parse_args().c_arm
    SUFFIX = "" if C_ARM == "C" else f"_{C_ARM}"
    ARMS = ["A", "B", C_ARM, "Astar"]
    protocol = load_config("configs/protocol.json")
    seeds = list(protocol["seeds"])
    reps = int(protocol["bootstrap_replicates"])
    bseed = int(protocol["bootstrap_seed"])
    comparisons = as_c_arm([tuple(c) for c in protocol["planned_comparisons"]])
    out: list = []

    with timed("stats", log) as t:
        for subset in ("full", "hair_present"):
            out += classification_endpoint(seeds, subset, reps, bseed, comparisons)

        # Secondary control: A vs A* isolates the 128px resolution change that
        # Arm C introduces. Reported separately from the planned family.
        for subset in ("full", "hair_present"):
            out += classification_endpoint(
                seeds, subset, reps, bseed, [("A", "Astar")])
        # Secondary control: C vs A* compares the diffusion arm against the
        # resolution-matched original, so the 128px round-trip is held constant.
        for subset in ("full", "hair_present"):
            out += classification_endpoint(
                seeds, subset, reps, bseed, [(C_ARM, "Astar")])

        p = artifact_dir("logs") / f"paired_psnr_ssim_test{SUFFIX}.csv"
        if p.exists():
            df = pd.read_csv(p)
            for metric in ("psnr", "ssim"):
                out += paired_column_endpoint(
                    df, f"B_{metric}", f"hairy_input_{metric}",
                    f"paired_{metric}", reps, bseed, [("B", "hairy input")])
                if f"{C_ARM}_{metric}" in df.columns and df[f"{C_ARM}_{metric}"].notna().any():
                    out += paired_column_endpoint(
                        df, f"{C_ARM}_{metric}", f"B_{metric}",
                        f"paired_{metric}", reps, bseed, [(C_ARM, "B")])
                    out += paired_column_endpoint(
                        df, f"{C_ARM}_{metric}", f"hairy_input_{metric}",
                        f"paired_{metric}", reps, bseed, [(C_ARM, "hairy input")])

        for metric in ("gradcam_overlap20", "rollout_overlap20"):
            out += xai_rows(metric, reps, bseed, comparisons)
        # Resolution control for explanation overlap, added with the Stage C analysis
        # after the planned rows were first seen: A* is the control named in the addendum.
        for metric in ("gradcam_overlap20", "rollout_overlap20"):
            out += xai_rows(metric, reps, bseed, [("A", "Astar"), (C_ARM, "Astar")])

    if C_ARM != "C":
        # A second diffusion arm: keep only rows that involve it, plus the A-A* explanation
        # control (A-B and A-A* classification rows live in the original files unchanged).
        out = [r for r in out if C_ARM in (r["arm_a"], r["arm_b"])
               or (r["endpoint"].startswith("x1_") and r["comparison"] == "A-Astar")]
        for r in out:
            # Per-group macro-F1 is not additive, so its dz can contradict the difference
            # (reports/prereg_v4_review_response.md, S2). Differences with CIs are reported instead.
            if r["endpoint"] == "test_macro_f1":
                r["effect_size_dz"] = float("nan")
    if not out:
        log.error("no statistical comparisons could be computed")
        return 1
    table = pd.DataFrame(out)
    # A-A* is a secondary resolution control, not one of the three planned
    # comparisons, so it gets its own Holm family and cannot inflate the
    # planned family's correction.
    table["family"] = np.where(
        table["comparison"].str.contains("Astar"), "secondary_control", "planned")
    if C_ARM != "C":
        # Cv2 was designed after the Stage B gate failed (D34): every row involving it is exploratory.
        table.loc[table["arm_a"].eq(C_ARM) | table["arm_b"].eq(C_ARM), "family"] = "exploratory"
    table["p_holm"] = np.nan
    for _, grp in table.groupby(["endpoint", "subset", "family"]):
        table.loc[grp.index, "p_holm"] = holm(grp["p_raw"].tolist())
    cols = ["endpoint", "subset", "family", "comparison", "arm_a", "arm_b", "arm_a_mean",
            "arm_b_mean", "difference", "ci_low", "ci_high", "p_raw", "p_holm",
            "effect_size_dz", "wilcoxon_p_raw", "wilcoxon_p_holm", "wilcoxon_statistic",
            "wilcoxon_n_groups", "n_images", "n_groups", "replicates", "test"]
    table = table[[c for c in cols if c in table.columns]]
    table = table.sort_values(["endpoint", "subset", "comparison"])
    # A clear p-value on a tiny difference is not a meaningful finding. Flag it
    # so the prose cannot call a negligible effect important.
    table["negligible"] = (
        (table["p_holm"] < 0.05)
        & (table["effect_size_dz"].abs() < 0.2)
    )
    table.to_csv(artifact_dir("logs") / f"statistical_comparisons{SUFFIX}.csv", index=False)
    save_json({"seconds": t["seconds"], "rows": int(len(table)),
               "endpoints": sorted(table["endpoint"].unique().tolist())},
              artifact_dir("logs") / f"stats_summary{SUFFIX}.json")
    record_run("stats", gpu=None, seconds=t["seconds"], rows=int(len(table)))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




