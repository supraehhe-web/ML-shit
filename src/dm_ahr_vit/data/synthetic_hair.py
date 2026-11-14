"""Seeded synthetic hair rendering for paired hair-removal evaluation."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


def _rng(image_id: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{image_id}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def render_hair(img: Image.Image, image_id: str, seed: int = 0):
    """Draw random dark curved hairs; return (hairy image, binary mask, params)."""
    rng = _rng(image_id, seed)
    w, h = img.size
    n_hairs = int(rng.integers(8, 20))
    blur_sigma = float(rng.uniform(0.5, 1.5))
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n_hairs):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.4, 1.2) * max(w, h)
        curve = rng.uniform(-0.6, 0.6)
        thickness = int(rng.integers(1, 4))
        steps = 48
        xs = np.linspace(0, length, steps)
        px = x0 + xs * np.cos(angle)
        py = y0 + xs * np.sin(angle)
        perp = (np.linspace(0, 1, steps) ** 2) * curve * 40.0
        px = px + perp * np.sin(angle + np.pi / 2)
        py = py + perp * np.cos(angle + np.pi / 2)
        for x, y in zip(px, py):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                mask[max(0, yi - thickness // 2):yi + thickness // 2 + 1,
                     max(0, xi - thickness // 2):xi + thickness // 2 + 1] = 255
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    m = mask > 0
    k = max(1, int(round(blur_sigma)))
    if k > 1:
        from scipy.ndimage import uniform_filter
        soft = uniform_filter(mask.astype(np.float64), size=k)
        m = soft > 0.05
    darkness = rng.uniform(0.25, 0.55)
    out = arr.copy()
    out[m] = arr[m] * darkness
    hairy = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return hairy, Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)), {
        "n_hairs": n_hairs, "blur_sigma": blur_sigma, "seed": seed, "source": "synthetic",
    }
