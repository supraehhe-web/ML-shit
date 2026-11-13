"""Mask geometry that matches what the ViT actually sees.

The classifier's eval transform is Resize(shorter side -> 255) then CenterCrop(224)
(`data.dataset.build_transforms(train=False)`). A saliency map lives in that
224x224 model-input space, so a lesion mask must go through the same geometry
before the two are compared. 128x128 files (A*/C hair-present images) are
stretched copies of the full field, so their masks are stretched the same way first.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 224
RESIZE = int(SIZE * 1.14)  # 255


def nn_resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.NEAREST))


def stretched_224(mask: np.ndarray) -> np.ndarray:
    """The whole mask stretched to 224x224: what the first XAI run scored against."""
    return nn_resize(mask, SIZE, SIZE)


def _resized_dims(w: int, h: int) -> tuple[int, int]:
    return (RESIZE, int(RESIZE * h / w)) if w <= h else (int(RESIZE * w / h), RESIZE)


def model_space(mask: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """The mask under the geometry the ViT sees for an image file of size `image_wh` = (w, h)."""
    w, h = image_wh
    m = nn_resize(mask, w, h)
    nw, nh = _resized_dims(w, h)
    m = nn_resize(m, nw, nh)
    left, top = int(round((nw - SIZE) / 2.0)), int(round((nh - SIZE) / 2.0))
    return m[top:top + SIZE, left:left + SIZE]


def retained_fraction(mask: np.ndarray, image_wh: tuple[int, int]) -> float:
    """Share of the lesion that survives the centre crop."""
    w, h = image_wh
    nw, nh = _resized_dims(w, h)
    total = float((nn_resize(nn_resize(mask, w, h), nw, nh) > 0).sum())
    if total == 0:
        return float("nan")
    return float((model_space(mask, image_wh) > 0).sum() / total)
