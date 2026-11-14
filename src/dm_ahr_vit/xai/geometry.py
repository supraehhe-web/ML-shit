"""The one place that defines the model's view of an image and its lesion mask.

The ViT never sees the stored photo: it sees the photo resized so the shorter side is
``int(size * 1.14)`` and then centre-cropped to ``size`` x ``size``. Saliency maps live in that
cropped frame, so a lesion mask has to be carried through the identical resize and crop before
it is compared with a map. The classifier's evaluation transform (``data.dataset``) and every
explanation script call the functions below, so the two cannot drift apart.

Stored image sizes differ by arm AND by image: the original and DullRazor arms keep the 600 x 450
photo; in the A*, C, Creleased and Cv2 arms the hair-present images are stored at 128 x 128 (an
aspect-ratio squash) while hair-absent images pass through unchanged at 600 x 450 (see
`scripts/02_hair_pipeline.py`). The geometry therefore reads each image's stored size from the file.
Masks are stored at 600 x 450 and are squashed to that size first.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

EVAL_SCALE = 1.14
FULL_SIZE = (600, 450)
SQUASHED_SIZE = (128, 128)
ARM_STORED_SIZES: dict[str, frozenset[tuple[int, int]]] = {
    "A": frozenset({FULL_SIZE}), "B": frozenset({FULL_SIZE}),
    "Astar": frozenset({FULL_SIZE, SQUASHED_SIZE}), "C": frozenset({FULL_SIZE, SQUASHED_SIZE}),
    "Creleased": frozenset({FULL_SIZE, SQUASHED_SIZE}), "Cv2": frozenset({FULL_SIZE, SQUASHED_SIZE}),
}
FRAME_VERSION = "model_frame_v2"


def eval_resize_size(size: int = 224) -> int:
    return int(size * EVAL_SCALE)


def eval_ops(size: int = 224, interpolation=InterpolationMode.BILINEAR) -> list:
    """The resize and centre-crop of the classifier's evaluation transform."""
    return [transforms.Resize(eval_resize_size(size), interpolation=interpolation),
            transforms.CenterCrop(size)]


def to_model_frame(img: Image.Image, size: int = 224, is_mask: bool = False) -> Image.Image:
    """Carry a PIL image (or mask, with nearest-neighbour resampling) into the model's frame."""
    mode = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
    for op in eval_ops(size, mode):
        img = op(img)
    return img


def mask_to_model_frame(mask: Image.Image, stored_size_wh: tuple[int, int] | None,
                        size: int = 224) -> np.ndarray:
    """Binary {0,1} lesion mask in the model's frame, for an arm whose images are stored at
    ``stored_size_wh``. ``None`` means the mask is already at the stored size."""
    m = mask.convert("L")
    if stored_size_wh is not None and m.size != tuple(stored_size_wh):
        m = m.resize(tuple(int(v) for v in stored_size_wh), Image.Resampling.NEAREST)
    m = to_model_frame(m, size, is_mask=True)
    return (np.asarray(m) > 127).astype(np.uint8)


def shift_zero_fill(a, dx: int, dy: int):
    """Translate an (..., H, W) numpy array or tensor by (dx, dy) pixels, filling with zeros."""
    is_t = isinstance(a, torch.Tensor)
    out = torch.zeros_like(a) if is_t else np.zeros_like(a)
    h, w = a.shape[-2], a.shape[-1]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    if abs(dx) < w and abs(dy) < h:
        out[..., yd, xd] = a[..., ys, xs]
    return out
