"""Paired cluster bootstrap and Holm correction.

All resampling is done at the lesion-group level, and the same resample is
shared across arms and seeds so comparisons stay paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment. NaN inputs are ignored in the family."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not np.isnan(p)]
    m = len(idx)
    out = [float("nan")] * len(pvalues)
    if m == 0:
        return out
    ordered = sorted(idx, key=lambda i: pvalues[i])
    running = 0.0
    for rank, i in enumerate(ordered):
        adj = (m - rank) * float(pvalues[i])
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


def paired_cluster_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    groups: np.ndarray,
    statistic=None,
    replicates: int = 10000,
    seed: int = 20260101,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap the paired difference a - b, resampling lesion groups.

    `values_a`/`values_b` hold one observation per row (same row alignment).
    For image-level metric tables, aggregate to one row per group first.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    g = np.asarray(groups)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[ok], b[ok], g[ok]
    if len(a) == 0:
        return {"n": 0, "note": "no finite paired observations"}

    stat = statistic or (lambda x: float(np.mean(x)))
    uniq = np.unique(g)
    index_by_group = {gu: np.nonzero(g == gu)[0] for gu in uniq}
    observed = stat(a) - stat(b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(replicates, dtype=float)
    for r in range(replicates):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index_by_group[gu] for gu in pick])
        diffs[r] = stat(a[rows]) - stat(b[rows])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    sd = float(np.std(diffs, ddof=1)) if replicates > 1 else float("nan")
    return {
        "n": int(len(a)),
        "n_groups": int(len(uniq)),
        "stat_a": float(stat(a)),
        "stat_b": float(stat(b)),
        "difference": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_raw": float(min(1.0, p_two)),
        "effect_size_dz": float(observed / sd) if sd and sd > 0 else float("nan"),
        "replicates": int(replicates),
        "seed": int(seed),
        "alpha": alpha,
    }


def cohens_dz(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


def wilcoxon_group_paired(df: pd.DataFrame, value: str, arm_col: str,
                          group_col: str = "group_id") -> dict:
    """Group-aggregated Wilcoxon signed-rank between the two arms in `df`."""
    from scipy.stats import wilcoxon

    wide = df.pivot_table(index=group_col, columns=arm_col, values=value, aggfunc="mean")
    wide = wide.dropna()
    if wide.shape[1] != 2 or len(wide) < 6:
        return {"n_groups": int(len(wide)), "p_raw": float("nan"),
                "note": "insufficient paired groups"}
    arms = list(wide.columns)
    try:
        res = wilcoxon(wide[arms[0]], wide[arms[1]])
        p = float(res.pvalue)
        stat = float(res.statistic)
    except Exception as exc:  # noqa: BLE001
        return {"n_groups": int(len(wide)), "p_raw": float("nan"), "note": repr(exc)}
    return {
        "test": "wilcoxon_signed_rank_group_aggregated",
        "arm_a": arms[0],
        "arm_b": arms[1],
        "n_groups": int(len(wide)),
        "statistic": stat,
        "p_raw": p,
        "median_difference": float((wide[arms[0]] - wide[arms[1]]).median()),
    }


def wilcoxon_per_group(long_df: pd.DataFrame, arm_a: str, arm_b: str,
                       value_col: str = "value",
                       group_col: str = "group_id") -> dict:
    """Wilcoxon signed-rank on lesion-group-level paired values.

    `long_df` carries one row per (arm, group). Collapsing to a single value per
    lesion first keeps the many images of one lesion from being treated as
    independent replicates.
    """
    from scipy.stats import wilcoxon

    out = {"wilcoxon_n_groups": 0, "wilcoxon_statistic": float("nan"),
           "wilcoxon_p_raw": float("nan")}
    if long_df is None or long_df.empty:
        return out
    sub = long_df[long_df["arm"].isin([arm_a, arm_b])]
    if sub.empty:
        return out
    wide = sub.pivot_table(index=group_col, columns="arm",
                           values=value_col, aggfunc="mean").dropna()
    out["wilcoxon_n_groups"] = int(len(wide))
    if wide.shape[1] != 2 or len(wide) < 6:
        return out
    a = wide[arm_a].to_numpy(dtype=float)
    b = wide[arm_b].to_numpy(dtype=float)
    if np.allclose(a - b, 0.0):
        out["wilcoxon_statistic"] = 0.0
        out["wilcoxon_p_raw"] = 1.0
        return out
    try:
        res = wilcoxon(a, b)
    except ValueError:
        return out
    out["wilcoxon_statistic"] = float(res.statistic)
    out["wilcoxon_p_raw"] = float(res.pvalue)
    return out

