"""Arm datasets over the frozen splits, with the matched ViT recipe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..config import CLASS_TO_IDX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(cfg: dict, train: bool):
    """Eval: square resize + ImageNet normalize. Train: the recipe in vit.json."""
    size = int(cfg["image_size"])
    aug = cfg.get("augmentation", {})
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=tuple(aug.get("random_resized_crop_scale", (0.75, 1.0))),
                ratio=tuple(aug.get("random_resized_crop_ratio", (3.0 / 4.0, 4.0 / 3.0))),
            ),
            transforms.RandomHorizontalFlip(float(aug.get("horizontal_flip", 0.5))),
            transforms.RandomVerticalFlip(float(aug.get("vertical_flip", 0.5))),
            transforms.RandomRotation(float(aug.get("rotation_degrees", 15))),
            transforms.ColorJitter(
                *[float(v) for v in aug.get("color_jitter", [0.15, 0.15, 0.1, 0.02])]
            ),
            transforms.ToTensor(),
            norm,
        ])
    # Eval: resize the shorter side to int(size * 1.14) = 255, then centre-crop.
    # This is the pipeline the recorded runs used; squashing to a square or
    # resizing to 256 changes predictions on these 600x450 images.
    from ..xai.geometry import eval_ops  # one definition of the model's frame (see xai/geometry)

    return transforms.Compose([*eval_ops(size), transforms.ToTensor(), norm])


class ArmDataset(Dataset):
    """One arm of the study over one split.

    Arm A reads the original ISIC JPEGs; every processed arm (B, C, Astar,
    Creleased, ...) reads ``processed_root/<arm>/<image_id>.png``.
    """

    def __init__(self, arm: str, split: str, splits_df: pd.DataFrame,
                 raw_images_dir, processed_root, transform, image_ids=None):
        df = splits_df[splits_df["split"] == split]
        if image_ids is not None:
            df = df[df["image_id"].isin(set(image_ids))]
        self.records = df.reset_index(drop=True)
        self.arm = arm
        self.raw_images_dir = Path(raw_images_dir)
        self.processed_root = Path(processed_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, image_id: str) -> Path:
        if self.arm == "A":
            return self.raw_images_dir / f"{image_id}.jpg"
        return self.processed_root / self.arm / f"{image_id}.png"

    def __getitem__(self, idx: int) -> dict:
        r = self.records.iloc[idx]
        with Image.open(self._path(r["image_id"])) as fh:
            img = fh.convert("RGB")
            x = self.transform(img)
        return {
            "image": x,
            "label": int(CLASS_TO_IDX[r["dx"]]),
            "image_id": str(r["image_id"]),
            "group_id": str(r["group_id"]),
            "dx": str(r["dx"]),
        }


def collate(batch: list) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "image_id": [b["image_id"] for b in batch],
        "group_id": [b["group_id"] for b in batch],
        "dx": [b["dx"] for b in batch],
    }
