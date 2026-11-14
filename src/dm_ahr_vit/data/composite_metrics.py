"""Per-item metrics for hair removal on composites with known clean targets.

All arrays are at the same resolution (the diffusion model's 128 px). `alpha` is the soft
strand coverage of the composite, `lesion` a boolean lesion mask. Lesion damage is measured
only on lesion pixels the hair did not touch, where any change from the input is damage.
"""

from __future__ import annotations

import cv2
import numpy as np

CHANGE_LEVELS = 10
HAIR_ALPHA = 0.02
MIN_LESION_PIXELS = 50


def item_metrics(out: np.ndarray, target: np.ndarray, comp: np.ndarray, alpha: np.ndarray,
                 lesion: np.ndarray) -> dict:
    out, target, comp = (np.asarray(x, np.float64) for x in (out, target, comp))
    mse = float(np.mean((out - target) ** 2))
    psnr = float(10.0 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")
    w = alpha[..., None].astype(np.float64)
    denom = float(w.sum() * 3 + 1e-9)
    e_floor = float((np.abs(comp - target) * w).sum() / denom)
    e_out = float((np.abs(out - target) * w).sum() / denom)
    removal = 1.0 - e_out / max(e_floor, 1e-9)

    hair_region = cv2.dilate((alpha > HAIR_ALPHA).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    clear_lesion = lesion & ~hair_region
    clear_skin = ~lesion & ~hair_region
    changed = np.abs(out - comp).max(axis=2) > CHANGE_LEVELS
    err = np.abs(out - target).mean(axis=2)
    nan = float("nan")
    return {
        "psnr": psnr,
        "hair_removal": removal,
        "lesion_pixels_clear_of_hair": int(clear_lesion.sum()),
        "lesion_mae": float(err[clear_lesion].mean()) if clear_lesion.sum() >= MIN_LESION_PIXELS else nan,
        "lesion_changed": float(changed[clear_lesion].mean()) if clear_lesion.sum() >= MIN_LESION_PIXELS else nan,
        "skin_mae": float(err[clear_skin].mean()) if clear_skin.sum() >= MIN_LESION_PIXELS else nan,
        "skin_changed": float(changed[clear_skin].mean()) if clear_skin.sum() >= MIN_LESION_PIXELS else nan,
    }


def bootstrap_mean_ci(x: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (reps, x.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)
