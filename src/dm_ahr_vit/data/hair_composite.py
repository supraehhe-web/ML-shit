"""Training pairs made by compositing REAL hair onto clean dermoscopic images.

A clean-ish target image (very low detector hair coverage) gets real hair strands, harvested
by `hair/harvest.py` from a different hairy image, pasted on at full resolution. The pair
(target, composite) is then resized to the diffusion model's native size with the same
bicubic resize the rest of the pipeline uses. Targets and sources always come from the same
split, and only the train and validation splits are accepted: the test split cannot be used.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .synthetic_hair import render_hair

CLEAN_MAX_COVERAGE = 0.0167
HAIR_MIN_COVERAGE = 0.03
# Layers with very little harvested hair were where review found ink, bubble rims and lesion dots.
MIN_LAYER_FRACTION = 0.004
ALLOWED_SPLITS = ("train", "val")


def select_pools(screening: pd.DataFrame, split: str) -> tuple[list[str], list[str]]:
    """Target ids (clean-ish) and candidate source ids (clearly hairy) for one split."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} is not allowed; only {ALLOWED_SPLITS}")
    df = screening[screening["split"] == split]
    targets = df.loc[df["hair_coverage"] <= CLEAN_MAX_COVERAGE, "image_id"]
    sources = df.loc[df["hair_coverage"] >= HAIR_MIN_COVERAGE, "image_id"]
    return sorted(targets.tolist()), sorted(sources.tolist())


class HairCompositeDataset(Dataset):
    """(composite, clean) pairs; items match `PairedDataset`: HR = clean, SR = composite."""

    def __init__(self, split: str, targets: list[str], sources: list[str],
                 images_dir: Path, layers_dir: Path, size: int = 128,
                 length: int | None = None, fixed: bool = False, seed: int = 0,
                 simple_prob: float = 0.3, max_shift: int = 120, max_sources: int = 3,
                 groups: dict[str, str] | None = None):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split {split!r} is not allowed; only {ALLOWED_SPLITS}")
        self.split = split
        self.images_dir = Path(images_dir)
        self.layers_dir = Path(layers_dir)
        self.targets = list(targets)
        self.sources = [s for s in sources if (self.layers_dir / f"{s}.png").exists()]
        if not self.targets or not self.sources:
            raise ValueError("no targets or no harvested sources for this split")
        self.size, self.fixed, self.seed = int(size), bool(fixed), int(seed)
        self.simple_prob, self.max_shift = float(simple_prob), int(max_shift)
        self.max_sources = max(1, int(max_sources))
        self.length = int(length) if length else len(self.targets)
        self.groups = groups or {}
        self._rng_cache: np.random.Generator | None = None

    def __len__(self) -> int:
        return self.length

    def _rng(self, idx: int) -> np.random.Generator:
        if self.fixed:
            return np.random.default_rng([self.seed, int(idx)])
        if self._rng_cache is None:
            self._rng_cache = np.random.default_rng(
                [self.seed, os.getpid(), int(torch.initial_seed() % (2**31))])
        return self._rng_cache

    def compose(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Full-resolution target, composite, soft alpha, and what was used."""
        rng = self._rng(idx)
        tid = self.targets[int(idx) % len(self.targets)]
        k = int(rng.integers(1, self.max_sources + 1))
        sids = [self.sources[int(rng.integers(len(self.sources)))] for _ in range(k)]
        target = np.asarray(Image.open(self.images_dir / f"{tid}.jpg").convert("RGB"))
        h, w = target.shape[:2]
        comp = target.astype(np.float32)
        alpha_tot = np.zeros((h, w), np.float32)
        for sid in sids:
            src = np.asarray(Image.open(self.images_dir / f"{sid}.jpg").convert("RGB"))
            mask = cv2.imread(str(self.layers_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
            if src.shape[:2] != (h, w):
                src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if rng.random() < 0.5:
                src, mask = src[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                src, mask = src[::-1], mask[::-1]
            dy = int(rng.integers(-self.max_shift, self.max_shift + 1))
            dx = int(rng.integers(-self.max_shift, self.max_shift + 1))
            src = np.roll(src, (dy, dx), axis=(0, 1))
            mask = np.roll(mask, (dy, dx), axis=(0, 1))
            m = (mask > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            alpha = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7), 0.0, 1.0)
            a3 = alpha[..., None]
            comp = comp * (1 - a3) + src.astype(np.float32) * a3
            alpha_tot = 1.0 - (1.0 - alpha_tot) * (1.0 - alpha)
        comp = np.clip(np.rint(comp), 0, 255).astype(np.uint8)
        simple = False
        if self.simple_prob > 0 and rng.random() < self.simple_prob:
            hairy, _, _ = render_hair(Image.fromarray(comp), tid, int(rng.integers(2**31)))
            comp, simple = np.asarray(hairy), True
        return target, comp, alpha_tot, {"target": tid, "sources": sids, "simple": simple}

    def _resize(self, arr: np.ndarray) -> torch.Tensor:
        im = Image.fromarray(arr).resize((self.size, self.size), Image.Resampling.BICUBIC)
        x = np.asarray(im, dtype=np.float32)
        return torch.from_numpy(x / 127.5 - 1.0).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        target, comp, _, _ = self.compose(idx)
        return {"HR": self._resize(target), "SR": self._resize(comp)}

    def manifest(self) -> dict:
        ids = sorted(set(self.targets) | set(self.sources))
        grp = {self.groups[i] for i in ids if i in self.groups}
        return {"n": int(self.length), "target_ids": sorted(self.targets),
                "source_ids": sorted(self.sources), "splits": [self.split],
                "groups": sorted(grp)}


def build_datasets(screening_csv: Path, images_dir: Path, layers_dir: Path,
                   size: int = 128, val_length: int = 192, seed: int = 0,
                   min_layer_fraction: float = MIN_LAYER_FRACTION):
    """Train and validation composite datasets from the screening table."""
    scr = pd.read_csv(screening_csv)
    layer_stats = pd.read_csv(Path(layers_dir) / "summary.csv")
    usable = set(layer_stats.loc[layer_stats["fraction"] >= min_layer_fraction, "image_id"])
    groups = dict(zip(scr["image_id"], scr["group_id"]))
    tr_t, tr_s = select_pools(scr, "train")
    va_t, va_s = select_pools(scr, "val")
    tr_s, va_s = [i for i in tr_s if i in usable], [i for i in va_s if i in usable]
    train = HairCompositeDataset("train", tr_t, tr_s, images_dir, layers_dir, size,
                                 fixed=False, seed=seed, groups=groups)
    val = HairCompositeDataset("val", va_t, va_s, images_dir, layers_dir, size,
                               length=min(val_length, len(va_t)), fixed=True,
                               seed=seed + 1, groups=groups)
    return train, val
