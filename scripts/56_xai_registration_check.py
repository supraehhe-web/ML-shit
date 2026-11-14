"""Do the lesion masks line up with what the ViT sees? Mask-only check on the validation split.

The recorded XAI runs scored saliency (computed on the 224x224 centre crop the model sees)
against a mask squashed from the whole frame. This measures how far apart those two frames are,
per arm geometry, and what the centred-Gaussian baseline scores under each. No model, no test data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.data.masks import load_mask_224, load_mask_registered, resolve  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.xai.vit_xai import (  # noqa: E402
    gaussian_chance_map,
    lesion_overlap,
    pointing_game,
    salient_in_lesion,
)

SPLIT = "val"            # never "test": the geometry is a property of the masks, not of results
GEOMETRIES = {"orig600x450 (A, B)": None, "sq128 (A*, C, Cv2, Creleased)": (128, 128)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def centroid_offset(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    c = (mask.shape[1] - 1) / 2.0
    return float(np.hypot(xs.mean() - c, ys.mean() - c) / mask.shape[1])


def crop_loss_600x450(path) -> float:
    """Share of lesion area that the model's centre crop cuts away, for 600x450 inputs."""
    with Image.open(resolve(path)) as fh:
        m = np.array(fh.convert("L").resize((340, 255), Image.Resampling.NEAREST)) > 127
    top, left = int(round((255 - 224) / 2.0)), int(round((340 - 224) / 2.0))
    kept = m[top:top + 224, left:left + 224].sum()
    return float(1.0 - kept / max(m.sum(), 1))


def main() -> int:
    cfg = load_config("configs/vit.json")
    size = int(cfg["image_size"])
    frac = float(load_config("configs/sampling.json")["xai_top_fraction"])
    x1 = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
    x1 = x1[(x1["split"] == SPLIT) & (x1["status"] == "ok")]
    gauss = gaussian_chance_map(size)

    rows = []
    for _, r in x1.iterrows():
        legacy = load_mask_224(r["mask_path"], size)
        row = {"image_id": r["image_id"], "legacy_area": float(legacy.mean()),
               "legacy_recall": lesion_overlap(gauss, legacy, frac),
               "legacy_precision": salient_in_lesion(gauss, legacy, frac),
               "legacy_pointing": pointing_game(gauss, legacy),
               "crop_loss_600x450": crop_loss_600x450(r["mask_path"])}
        for name, wh in GEOMETRIES.items():
            reg = load_mask_registered(r["mask_path"], wh, size)
            key = name.split(" ")[0]
            row.update({
                f"{key}_area": float(reg.mean()),
                f"{key}_iou_vs_legacy": iou(legacy, reg),
                f"{key}_recall": lesion_overlap(gauss, reg, frac),
                f"{key}_precision": salient_in_lesion(gauss, reg, frac),
                f"{key}_pointing": pointing_game(gauss, reg),
                f"{key}_centroid_offset": centroid_offset(reg) if reg.sum() else float("nan"),
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    out = artifact_dir("logs") / "xai_registration_val.csv"
    df.to_csv(out, index=False)

    summary = {"split": SPLIT, "n": int(len(df)), "top_fraction": frac,
               "means": {c: float(df[c].mean()) for c in df.columns if c != "image_id"},
               "centroid_offset_quantiles_orig600x450": df["orig600x450_centroid_offset"]
               .quantile([0.1, 0.5, 0.9]).round(4).to_dict(),
               "share_lesion_cut_by_crop_over_10pct": float((df["crop_loss_600x450"] > 0.10).mean())}
    save_json(summary, artifact_dir("logs") / "xai_registration_val.json")
    print(pd.Series(summary["means"]).round(3).to_string())
    print(summary["centroid_offset_quantiles_orig600x450"], summary["share_lesion_cut_by_crop_over_10pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
