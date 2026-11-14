"""Harvest real hair from dermoscopic images as thin-structure masks.

The frozen detector (`screening.py`) is built to flag *whether* hair is present and is
deliberately generous: its mask is dilated and picks up lesion texture. For compositing real
hair onto clean images we need the opposite: only long, thin, dark strands.

A dark thin component is kept as hair only when all of these hold:
  * strong response (well above the screening threshold), so faint texture is excluded;
  * long skeleton and small mean width (strands, not blobs or short dashes);
  * few junctions per unit length (crossing strands yes, the dense mesh of pigment network no);
  * not blue or purple (pen ink) and not hugging the image border (dermoscope frame edge).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .screening import hair_response

HARVEST_THRESHOLD = 30
MIN_PEAK_RESPONSE = 80
MIN_COMPONENT_AREA = 40
MIN_SKELETON_LENGTH = 40
MAX_MEAN_WIDTH = 7.0
MAX_BRANCH_DENSITY = 0.03
MAX_BLUE_OVER_RED = 8.0
MAX_BORDER_FRACTION = 0.5
BORDER_PX = 4
MAX_FRACTION = 0.25


def thin_hair_mask(image_rgb: np.ndarray, cfg: dict,
                   min_skeleton: int = MIN_SKELETON_LENGTH,
                   max_width: float = MAX_MEAN_WIDTH,
                   threshold: int = HARVEST_THRESHOLD,
                   min_peak: int = MIN_PEAK_RESPONSE,
                   max_branch_density: float = MAX_BRANCH_DENSITY) -> tuple[np.ndarray, dict]:
    """Return a uint8 {0,1} mask of accepted strands and a stats dict."""
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark, _ = hair_response(gray, {**cfg, "light_hair": False})
    mask = (dark > int(threshold)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    skel = skeletonize(mask > 0)
    skel_u8 = skel.astype(np.uint8)
    skel_len = np.bincount(labels[skel], minlength=n).astype(np.float64)
    nb = cv2.filter2D(skel_u8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    junction = (skel & ((nb - skel_u8) >= 3)).astype(np.uint8)
    # Count branch *points* (clusters of junction pixels), not junction pixels.
    n_j, jlab = cv2.connectedComponents(junction, connectivity=8)
    branches = np.zeros(n, dtype=np.float64)
    if n_j > 1:
        ys, xs = np.nonzero(junction)
        _, first = np.unique(jlab[ys, xs], return_index=True)
        branches = np.bincount(labels[ys[first], xs[first]], minlength=n).astype(np.float64)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    peak = np.zeros(n, dtype=np.float64)
    np.maximum.at(peak, labels.ravel(), dark.ravel().astype(np.float64))
    flat = labels.ravel()
    mean_rgb = [np.bincount(flat, weights=image_rgb[..., c].ravel().astype(np.float64),
                            minlength=n) / np.maximum(area, 1) for c in range(3)]
    border = np.zeros((h, w), bool)
    border[:BORDER_PX], border[-BORDER_PX:] = True, True
    border[:, :BORDER_PX], border[:, -BORDER_PX:] = True, True
    border_frac = np.bincount(labels[border], minlength=n) / np.maximum(area, 1)

    keep = np.zeros(n, dtype=bool)
    reasons = {"small": 0, "short": 0, "wide": 0, "weak": 0, "mesh": 0, "ink": 0, "border": 0}
    for i in range(1, n):
        if area[i] < MIN_COMPONENT_AREA:
            reasons["small"] += 1
        elif skel_len[i] < min_skeleton:
            reasons["short"] += 1
        elif area[i] / skel_len[i] > max_width:
            reasons["wide"] += 1
        elif peak[i] < min_peak:
            reasons["weak"] += 1
        elif branches[i] / skel_len[i] > max_branch_density:
            reasons["mesh"] += 1
        elif mean_rgb[2][i] - mean_rgb[0][i] > MAX_BLUE_OVER_RED:
            reasons["ink"] += 1
        elif border_frac[i] > MAX_BORDER_FRACTION:
            reasons["border"] += 1
        else:
            keep[i] = True
    out = keep[labels].astype(np.uint8)
    fraction = float(out.mean())
    info = {"components_total": int(n - 1), "components_kept": int(keep.sum()),
            "hair_pixels": int(out.sum()), "fraction": fraction, "rejected": reasons}
    if fraction > MAX_FRACTION:
        out = np.zeros_like(out)
        info.update(components_kept=0, hair_pixels=0, fraction=0.0)
    return out, info
