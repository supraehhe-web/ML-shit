"""Pooled-over-seeds statistics with a two-level bootstrap (lesion groups and training seeds).

Pure numpy/pandas so it runs and is tested without Modal. A scores array has shape
(seeds, replicates): the score of one arm and seed on each lesion-group bootstrap replicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CLASSES

LOSS_FLOOR = 1e-12


def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return per_class.mean(axis=1)


def group_counts(df: pd.DataFrame, groups: list[str]) -> dict:
    """Per-lesion-group TP/FP/FN per class, as in scripts/09_stats.py."""
    gidx = {g: i for i, g in enumerate(groups)}
    k = len(CLASSES)
    tp, fp, fn = (np.zeros((len(groups), k)) for _ in range(3))
    y = df["label"].map(CLASSES.index).to_numpy()
    p = df["pred"].map(CLASSES.index).to_numpy()
    g = df["group_id"].map(gidx).to_numpy()
    for gi, yi, pi in zip(g, y, p):
        if yi == pi:
            tp[gi, yi] += 1
        else:
            fp[gi, pi] += 1
            fn[gi, yi] += 1
    return {"tp": tp, "fp": fp, "fn": fn}


def group_loss(df: pd.DataFrame, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Summed per-image log-loss and image count per lesion group."""
    probs = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    true = df["label"].map(CLASSES.index).to_numpy()
    loss = -np.log(np.maximum(probs[np.arange(len(df)), true], LOSS_FLOOR))
    gidx = {g: i for i, g in enumerate(groups)}
    gi = df["group_id"].map(gidx).to_numpy()
    ls = np.bincount(gi, weights=loss, minlength=len(groups))
    n = np.bincount(gi, minlength=len(groups)).astype(float)
    return ls, n


def draw_weights(n_groups: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(reps, n_groups))
    w = np.zeros((reps, n_groups), dtype=np.float32)
    for r in range(reps):
        np.add.at(w[r], idx[r], 1.0)
    return w


def f1_scores(counts_by_seed: list[dict], w: np.ndarray) -> np.ndarray:
    return np.stack([macro_f1(w @ c["tp"], w @ c["fp"], w @ c["fn"]) for c in counts_by_seed])


def loss_scores(loss_by_seed: list[tuple[np.ndarray, np.ndarray]], w: np.ndarray) -> np.ndarray:
    return np.stack([(w @ ls) / np.maximum(w @ n, 1.0) for ls, n in loss_by_seed])


def two_level(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample seeds with replacement independently on every replicate, then average."""
    s, r = scores.shape
    idx = rng.integers(0, s, size=(r, s))
    return scores[idx, np.arange(r)[:, None]].mean(axis=1)


def one_level(scores: np.ndarray) -> np.ndarray:
    """Seeds held fixed (the earlier analysis): average the seeds on each replicate."""
    return scores.mean(axis=0)


def compare(dist_a: np.ndarray, dist_b: np.ndarray, obs_a: float, obs_b: float) -> dict:
    diff = dist_a - dist_b
    reps = diff.size
    lo, hi = np.percentile(diff, [2.5, 97.5])
    k_le, k_ge = int((diff <= 0).sum()), int((diff >= 0).sum())
    p = min(1.0, 2 * min((k_le + 1) / (reps + 1), (k_ge + 1) / (reps + 1)))
    se = float(diff.std(ddof=1))
    return {"difference": float(obs_a - obs_b), "ci_low": float(lo), "ci_high": float(hi),
            "p_raw": float(p), "se_diff": se, "mdd_80pct": 2.8 * se}
