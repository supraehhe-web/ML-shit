"""Lesion-mask loading (X1 tier) and the mask manifest builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..paths import data_root

X1_PROVENANCE = (
    "HAM10000_segmentations_lesion_tschandl (Harvard Dataverse doi:10.7910/DVN/DBW86T)"
)


def resolve(path) -> Path:
    """Re-root container-recorded ``/__modal/...`` paths onto the local data root."""
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def load_mask_224(path, size: int = 224) -> np.ndarray:
    """Grayscale mask -> binary {0,1} array resized with NEAREST."""
    with Image.open(resolve(path)) as fh:
        m = fh.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def load_mask_registered(path, image_size_wh=None, size: int = 224) -> np.ndarray:
    """Lesion mask in the model's frame; delegates to `xai.geometry.mask_to_model_frame`."""
    from ..xai.geometry import mask_to_model_frame

    with Image.open(resolve(path)) as fh:
        return mask_to_model_frame(fh, image_size_wh, size)


def build_x1_manifest(masks_dir: Path, split_df: pd.DataFrame, man: pd.DataFrame,
                      out_path: Path) -> pd.DataFrame:
    """One row per split image: mask presence, geometry checks, foreground fraction."""
    sizes = dict(zip(man["image_id"], zip(man["width"], man["height"])))
    dx = dict(zip(split_df["image_id"], split_df["dx"]))
    group = dict(zip(split_df["image_id"], split_df["group_id"]))
    split = dict(zip(split_df["image_id"], split_df["split"]))
    rows = []
    for image_id in split_df["image_id"]:
        p = masks_dir / f"{image_id}_segmentation.png"
        row = {
            "image_id": image_id,
            "tier": "X1",
            "split": split[image_id],
            "dx": dx[image_id],
            "group_id": group[image_id],
            "has_mask": bool(p.exists()),
            "mask_path": str(p),
            "provenance": X1_PROVENANCE,
            "expert_traced_claimed": False,
        }
        status = "ok"
        if p.exists():
            with Image.open(p) as fh:
                arr = np.array(fh.convert("L"))
            w, h = arr.shape[1], arr.shape[0]
            binary = set(np.unique(arr)) <= {0, 255}
            non_empty = bool((arr > 127).any())
            fg = float((arr > 127).mean())
            iw, ih = sizes.get(image_id, (None, None))
            size_matches = bool(iw == w and ih == h)
            row.update({
                "width": w, "height": h, "binary": bool(binary),
                "non_empty": non_empty, "foreground_fraction": fg,
                "size_matches": size_matches,
            })
            if not (binary and non_empty and size_matches):
                status = "degraded"
        else:
            row.update({"width": None, "height": None, "binary": None,
                        "non_empty": None, "foreground_fraction": None,
                        "size_matches": None})
            status = "missing_mask"
        row["status"] = status
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
