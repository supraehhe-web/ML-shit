"""Full-resolution paste-back of a low-resolution diffusion fill.

The diffusion model works at 128 px and regenerates the whole image. Here its output is used
only where it removed dark hair: flagged pixels (fill brighter than input) are replaced, and
every other pixel stays the original, full-resolution value.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_GRAY = np.array([0.299, 0.587, 0.114], np.float32)


def hair_flag(in_small: np.ndarray, fill_small: np.ndarray, tau: float) -> np.ndarray:
    """True where the fill is brighter than the input by more than `tau` gray levels."""
    gin = in_small.astype(np.float32) @ _GRAY
    gout = fill_small.astype(np.float32) @ _GRAY
    return (gout - gin) > float(tau)


def pasteback(orig: np.ndarray, fill_small: np.ndarray, tau: float = 16.0, dilate: int = 1,
              det_mask: np.ndarray | None = None, refine_dilate: int = 2
              ) -> tuple[np.ndarray, dict]:
    """Blend `fill_small` (square, e.g. 128 px) into `orig` (H x W x 3 uint8).

    `det_mask` is an optional full-resolution binary hair mask; when given, the replaced
    region is limited to it (dilated by `refine_dilate` px).
    """
    h, w = orig.shape[:2]
    s = fill_small.shape[0]
    in_small = np.asarray(Image.fromarray(orig).resize((s, s), Image.Resampling.BICUBIC))
    flag = hair_flag(in_small, fill_small, tau).astype(np.uint8)
    if dilate > 0:
        flag = cv2.dilate(flag, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=int(dilate))
    m = cv2.resize(flag.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if det_mask is not None:
        det = (det_mask > 0).astype(np.uint8)
        if refine_dilate > 0:
            k = 2 * int(refine_dilate) + 1
            det = cv2.dilate(det, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = m * det.astype(np.float32)
    fill = np.asarray(Image.fromarray(fill_small).resize((w, h), Image.Resampling.BICUBIC),
                      dtype=np.float32)
    m3 = m[..., None]
    out = orig.astype(np.float32) * (1.0 - m3) + fill * m3
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out, {"mask_fraction": float(m.mean()), "changed_fraction": float((m > 0).mean())}
