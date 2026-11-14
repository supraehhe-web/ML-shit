"""Narrow the eval resize size by sweeping candidates against recorded predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode as IM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LIMIT = 400


def main() -> int:
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    test = splits[splits["split"] == "test"].reset_index(drop=True).head(LIMIT)
    recorded = pd.read_csv(artifact_dir("predictions") / "pred_A_seed42_full.csv")
    recorded = recorded.set_index("image_id").loc[test["image_id"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=False).to(device)
    load_checkpoint(artifact_dir("checkpoints", "vit", "A_seed42") / "best.pt", model, map_location=device)
    model.eval()

    imgs = []
    for iid in test["image_id"]:
        with Image.open(isic_images_dir() / f"{iid}.jpg") as fh:
            imgs.append(fh.convert("RGB"))
    print("image sizes:", sorted({im.size for im in imgs})[:5], flush=True)

    rec_pred = recorded["pred"].to_numpy()
    rec_prob = recorded[[f"prob_{c}" for c in CLASSES]].to_numpy()
    norm = transforms.Normalize(MEAN, STD)
    rows = []
    cands = [(s, IM.BILINEAR) for s in (224, 232, 240, 248, 255, 256, 257, 264, 272, 288, 300)]
    cands += [(256, IM.BICUBIC), (256, IM.LANCZOS)]
    for short, interp in cands:
        tfm = transforms.Compose([transforms.Resize(short, interpolation=interp),
                                  transforms.CenterCrop(224), transforms.ToTensor(), norm])
        batch = torch.stack([tfm(im) for im in imgs]).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(batch)
        p = torch.softmax(logits.float(), 1).cpu().numpy()
        pred = np.array([CLASSES[i] for i in p.argmax(1)])
        rows.append({"short": short, "interp": str(interp).split(".")[-1],
                     "mismatches": int((pred != rec_pred).sum()),
                     "mean_prob_absdiff": float(np.abs(p - rec_prob).mean())})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values(["mismatches", "mean_prob_absdiff"])
    df.to_csv(artifact_dir("logs") / "resize_sweep.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
