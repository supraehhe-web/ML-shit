"""Hair-coverage reanalysis (CPU only, from existing per-image predictions).

Replaces the per-decile macro-F1 of the first version: about 150 images spread over
seven classes per decile made every cell too noisy to interpret.

1. Pooled, lesion-clustered model. Per-image correctness (mean over the three seeds,
   so the response is a fraction) is regressed on arm, image-level hair coverage (per
   10 percentage points) and their interaction: a fractional logit (binomial GLM) with
   cluster-robust standard errors by lesion group. Arm A is the reference, so the
   arm term is the difference at zero coverage and the interaction term is how the
   gap changes per +10 pp of coverage. Sensitivity: group-mean coverage per lesion.
   Predicted accuracy gaps at fixed coverages use the delta method.
2. Per-decile balanced accuracy (macro recall over the classes present in the
   decile) with lesion-cluster bootstrap CIs, for each arm and for each arm's gap to A.
3. Internal control. Images screened as hair-absent pass through unchanged in every
   arm, so arm differences on them cannot come from hair removal; they show how large
   between-arm differences are from training composition and seeds alone.

Outputs (artifacts/logs): hair_coverage_pooled_model.csv, hair_coverage_marginal_gaps.csv,
hair_coverage_decile_balacc.csv, hair_coverage_decile_gaps.csv, hair_coverage_control.json
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()
ARMS = ["A", "Astar", "B", "C", "Creleased", "Cfixed"]
SEEDS = [42, 123, 2026]
CIDX = {c: i for i, c in enumerate(CLASSES)}
K = len(CLASSES)
REPS = 2000
BSEED = 20260101


def load(arm: str) -> pd.DataFrame | None:
    frames = []
    for seed in SEEDS:
        p = artifact_dir("predictions") / f"pred_{arm}_seed{seed}_full.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        d["seed"] = seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def arm_counts(df: pd.DataFrame, ids: set, groups: list) -> list[tuple[np.ndarray, np.ndarray]]:
    gidx = {g: i for i, g in enumerate(groups)}
    out = []
    for _, d in df[df["image_id"].isin(ids)].groupby("seed"):
        t = np.zeros((len(groups), K))
        c = np.zeros((len(groups), K))
        gi = d["group_id"].map(gidx).to_numpy()
        y = d["label"].map(CIDX).to_numpy()
        p = d["pred"].map(CIDX).to_numpy()
        np.add.at(t, (gi, y), 1.0)
        np.add.at(c, (gi, y), (y == p).astype(float))
        out.append((t, c))
    return out


def balanced_accuracy(W: np.ndarray, counts: list) -> np.ndarray:
    """Seed-averaged balanced accuracy for each weight row of W (macro recall, absent classes skipped)."""
    vals = []
    for t, c in counts:
        tt, cc = W @ t, W @ c
        rec = np.where(tt > 0, cc / np.maximum(tt, 1e-12), np.nan)
        vals.append(np.nanmean(rec, axis=1))
    return np.mean(vals, axis=0)


def draw_weights(n_groups: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(reps, n_groups))
    w = np.zeros((reps, n_groups))
    for r in range(reps):
        w[r] = np.bincount(idx[r], minlength=n_groups)
    return w


def pooled_model(long: pd.DataFrame, cov_col: str, variant: str):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    codes = pd.factorize(long["group_id"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.glm(f"y ~ C(arm, Treatment('A')) * {cov_col}", data=long,
                      family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": codes})
    ci = res.conf_int()
    table = pd.DataFrame({"variant": variant, "term": res.params.index, "coef_logit": res.params.values,
                          "se": res.bse.values, "ci_low": ci[0].values, "ci_high": ci[1].values,
                          "p": res.pvalues.values, "n_obs": int(res.nobs),
                          "n_groups": int(len(np.unique(codes)))})
    return res, table


def marginal_gaps(res, arms: list[str], cov_col: str, points: list[float], variant: str) -> pd.DataFrame:
    import re

    from scipy.special import expit
    from scipy.stats import norm

    V = np.asarray(res.cov_params())
    beta = res.params.values
    names = list(res.params.index)

    def x(arm: str, cov10: float) -> np.ndarray:
        out = []
        for n in names:
            if n == "Intercept":
                out.append(1.0)
            elif n == cov_col:
                out.append(cov10)
            else:
                level = re.search(r"\[T\.([^\]]+)\]", n).group(1)
                v = 1.0 if level == arm else 0.0
                out.append(v * cov10 if n.endswith(":" + cov_col) else v)
        return np.asarray(out)

    rows = []
    for arm in arms:
        if arm == "A":
            continue
        for cov in points:
            xa, xb = x(arm, cov / 0.10), x("A", cov / 0.10)
            ma, mb = expit(xa @ beta), expit(xb @ beta)
            g = ma * (1 - ma) * xa - mb * (1 - mb) * xb
            se = float(np.sqrt(g @ V @ g))
            gap = float(ma - mb)
            rows.append({"variant": variant, "comparison": f"{arm}-A", "coverage": cov, "gap": gap,
                         "ci_low": gap - 1.96 * se, "ci_high": gap + 1.96 * se,
                         "p": float(2 * (1 - norm.cdf(abs(gap / se)))) if se > 0 else np.nan,
                         "acc_arm": float(ma), "acc_A": float(mb)})
    return pd.DataFrame(rows)


def main() -> int:
    hair = pd.read_csv(manifest_dir() / "hair_screening.csv")
    th = hair[hair["split"] == "test"].set_index("image_id")
    preds = {a: load(a) for a in ARMS}
    preds = {a: d for a, d in preds.items() if d is not None}
    if "A" not in preds:
        log.error("Arm A predictions missing")
        return 1
    arms = list(preds)

    # ---- 1. pooled lesion-clustered fractional logit -------------------------
    frames = []
    for a, d in preds.items():
        d = d.copy()
        d["correct"] = (d["label"] == d["pred"]).astype(float)
        m = d.groupby(["image_id", "group_id"])["correct"].mean().reset_index().rename(columns={"correct": "y"})
        m["arm"] = a
        frames.append(m)
    long = pd.concat(frames, ignore_index=True)
    long["coverage"] = long["image_id"].map(th["hair_coverage"])
    long["cov10"] = long["coverage"] / 0.10
    gm = long.groupby("group_id")["coverage"].mean()
    long["cov10_group"] = long["group_id"].map(gm) / 0.10
    long = long.dropna(subset=["cov10"])
    pts = [0.0, 0.02, 0.06, 0.16, 0.30]
    tables, gaps = [], []
    for col, name in (("cov10", "image_coverage"), ("cov10_group", "group_mean_coverage")):
        res, t = pooled_model(long, col, name)
        tables.append(t)
        gaps.append(marginal_gaps(res, arms, col, pts, name))
    pd.concat(tables).to_csv(artifact_dir("logs") / "hair_coverage_pooled_model.csv", index=False)
    pd.concat(gaps).round(5).to_csv(artifact_dir("logs") / "hair_coverage_marginal_gaps.csv", index=False)
    log.info("pooled model\n%s", pd.concat(tables).round(4).to_string(index=False))

    # ---- 2. per-decile balanced accuracy -------------------------------------
    cov = th["hair_coverage"].dropna()
    edges = np.unique(np.quantile(cov, np.linspace(0, 1, 11)))
    dec = pd.cut(cov, bins=edges, include_lowest=True, labels=False)
    dec_rows, gap_rows = [], []
    for k in sorted(dec.dropna().unique()):
        ids = set(dec[dec == k].index)
        groups = sorted(set(preds["A"].loc[preds["A"]["image_id"].isin(ids), "group_id"]))
        W = draw_weights(len(groups), REPS, BSEED + int(k))
        counts = {a: arm_counts(preds[a], ids, groups) for a in arms}
        point, boot = {}, {}
        for a in arms:
            point[a] = float(balanced_accuracy(np.ones((1, len(groups))), counts[a])[0])
            boot[a] = balanced_accuracy(W, counts[a])
        sub = cov[list(ids)]
        pass_through = float((~th.loc[list(ids), "hair_present"].astype(bool)).mean())
        for a in arms:
            lo, hi = np.nanpercentile(boot[a], [2.5, 97.5])
            dec_rows.append({"arm": a, "decile": int(k), "coverage_lo": float(sub.min()),
                             "coverage_hi": float(sub.max()), "n_images": len(ids), "n_groups": len(groups),
                             "pass_through_fraction": pass_through, "balanced_accuracy": point[a],
                             "ci_low": float(lo), "ci_high": float(hi)})
        for a in arms:
            if a == "A":
                continue
            d = boot[a] - boot["A"]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            p = 2 * min(float(np.nanmean(d <= 0)), float(np.nanmean(d >= 0)))
            gap_rows.append({"comparison": f"{a}-A", "decile": int(k), "n_images": len(ids),
                             "pass_through_fraction": pass_through, "gap": point[a] - point["A"],
                             "ci_low": float(lo), "ci_high": float(hi), "p_raw": min(1.0, p)})
    pd.DataFrame(dec_rows).round(5).to_csv(artifact_dir("logs") / "hair_coverage_decile_balacc.csv", index=False)
    gdf = pd.DataFrame(gap_rows)
    gdf["p_holm"] = np.nan
    for _, g in gdf.groupby("comparison"):
        gdf.loc[g.index, "p_holm"] = holm(g["p_raw"].tolist())
    gdf.round(5).to_csv(artifact_dir("logs") / "hair_coverage_decile_gaps.csv", index=False)

    # ---- 3. internal control: hair-absent (pass-through) images ----------------
    absent = set(th.index[~th["hair_present"].astype(bool)])
    groups = sorted(set(preds["A"].loc[preds["A"]["image_id"].isin(absent), "group_id"]))
    W = draw_weights(len(groups), REPS, BSEED + 99)
    counts = {a: arm_counts(preds[a], absent, groups) for a in arms}
    boot = {a: balanced_accuracy(W, counts[a]) for a in arms}
    pt = {a: float(balanced_accuracy(np.ones((1, len(groups))), counts[a])[0]) for a in arms}
    control = {"n_images": len(absent), "n_groups": len(groups),
               "statement": "hair-absent images are the original image in every arm (pass-through), so "
                            "differences between arms here reflect training composition and seeds, not hair removal",
               "balanced_accuracy": pt, "pairwise": {}}
    for a, b in (("B", "A"), ("Astar", "A"), ("C", "A"), ("C", "Astar"), ("B", "Astar")):
        if a in boot and b in boot:
            d = boot[a] - boot[b]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            control["pairwise"][f"{a}-{b}"] = {"gap": pt[a] - pt[b], "ci_low": float(lo), "ci_high": float(hi)}
    save_json(control, artifact_dir("logs") / "hair_coverage_control.json")
    record_run("hair_coverage_v2", gpu=None, seconds=0.0, arms=len(arms))
    print(pd.DataFrame(dec_rows).round(3).to_string(index=False))
    print(gdf.round(3).to_string(index=False))
    print(control["pairwise"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
