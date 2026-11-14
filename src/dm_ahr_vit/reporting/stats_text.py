"""Per-image log-loss differences between arms (replaces the standardised effect size dz).

For two arms, the per-image log-loss difference (a minus b), averaged over the seeds both arms
have, with a lesion-group bootstrap interval; per-seed differences are returned alongside.
Negative means arm a has the lower (better) loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CLASSES

FLOOR = 1e-12


def _loss(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)
    p = d[[f"prob_{c}" for c in CLASSES]].to_numpy()
    true = d["label"].map(CLASSES.index).to_numpy()
    return pd.DataFrame({"image_id": d["image_id"], "group_id": d["group_id"],
                         "loss": -np.log(np.maximum(p[np.arange(len(d)), true], FLOOR))})


def logloss_diff(pred_dir: Path, a: str, b: str, seeds=(42, 123, 2026), reps: int = 2000,
                 rng_seed: int = 20261113, subset: str = "full") -> dict:
    per_seed, diffs, groups = {}, [], None
    for s in seeds:
        fa, fb = Path(pred_dir) / f"pred_{a}_seed{s}_{subset}.csv", Path(pred_dir) / f"pred_{b}_seed{s}_{subset}.csv"
        if not (fa.exists() and fb.exists()):
            continue
        la, lb = _loss(fa), _loss(fb)
        assert (la["image_id"].to_numpy() == lb["image_id"].to_numpy()).all()
        d = la["loss"].to_numpy() - lb["loss"].to_numpy()
        per_seed[s] = float(d.mean())
        diffs.append(d)
        groups = la["group_id"].to_numpy()
    if not diffs:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "per_seed": {}}
    d = np.mean(diffs, axis=0)
    uniq, gi = np.unique(groups, return_inverse=True)
    sums = np.bincount(gi, weights=d, minlength=len(uniq))
    n = np.bincount(gi, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    boot = sums[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"difference": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "per_seed": per_seed}
