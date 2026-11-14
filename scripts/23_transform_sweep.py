"""Find the eval transform that reproduces the recorded predictions exactly.

Diagnostic only: sweeps candidate preprocessing pipelines against a recorded
prediction CSV and reports how many per-image predictions disagree.
"""

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

from dm_ahr_vit.config import CLASSES, load_config  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def variants(size: int = 224) -> dict:
    norm = transforms.Normalize(MEAN, STD)
    t = transforms.ToTensor()
    return {
        "r256_cc224": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), t, norm]),
        "r256_bicubic_cc224": transforms.Compose([transforms.Resize(256, interpolation=IM.BICUBIC), transforms.CenterCrop(size), t, norm]),
        "r256_lanczos_cc224": transforms.Compose([transforms.Resize(256, interpolation=IM.LANCZOS), transforms.CenterCrop(size), t, norm]),
        "r256_nearest_cc224": transforms.Compose([transforms.Resize(256, interpolation=IM.NEAREST), transforms.CenterCrop(size), t, norm]),
        "r256sq_cc224": transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(size), t, norm]),
        "r224_cc224": transforms.Compose([transforms.Resize(224), transforms.CenterCrop(size), t, norm]),
        "r232_cc224": transforms.Compose([transforms.Resize(232), transforms.CenterCrop(size), t, norm]),
        "r256_cc224_noaa": transforms.Compose([transforms.Resize(256, antialias=False), transforms.CenterCrop(size), t, norm]),
        "r224sq": transforms.Compose([transforms.Resize((size, size)), t, norm]),
        "r256_cc224_float": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), t, norm]),
    }


def main() -> int:
    cfg = load_config("configs/vit.json")
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    test = splits[splits["split"] == "test"].reset_index(drop=True)
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

    rec_pred = recorded["pred"].to_numpy()
    rec_prob = recorded[[f"prob_{c}" for c in CLASSES]].to_numpy()
    rows = []
    for name, tfm in variants(int(cfg["image_size"])).items():
        batch = torch.stack([tfm(im) for im in imgs]).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(batch)
        p = torch.softmax(logits.float(), 1).cpu().numpy()
        pred = np.array([CLASSES[i] for i in p.argmax(1)])
        rows.append({
            "variant": name,
            "mismatches": int((pred != rec_pred).sum()),
            "max_prob_absdiff": float(np.abs(p - rec_prob).max()),
            "mean_prob_absdiff": float(np.abs(p - rec_prob).mean()),
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("mismatches")
    df.to_csv(artifact_dir("logs") / "transform_sweep.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
