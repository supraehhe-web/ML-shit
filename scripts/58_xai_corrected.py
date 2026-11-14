"""Corrected explanation analysis: Grad-CAM (last-block ln_1 and, as a sensitivity run, the patch
grid) and attention rollout, scored against lesion masks carried through the model's own
geometry (`xai.geometry`), with chance baselines and an off-centre robustness pass.

One CSV per (arm, seed, split) in artifacts/logs/xai_v2/, plus a provenance JSON.
Pre-registered in reports/prereg_v5_client_asks.md; metric wording is generated from the
function docstrings (`xai.batched.metric_definitions`).
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.data.dataset import ArmDataset, build_transforms, collate  # noqa: E402
from dm_ahr_vit.data.masks import resolve  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402
from dm_ahr_vit.xai.batched import (  # noqa: E402
    MultiLayerGradCAM,
    attention_rollout_batch,
    localization_scores,
    random_map_baselines,
)
from dm_ahr_vit.xai.geometry import (  # noqa: E402
    ARM_STORED_SIZES,
    FRAME_VERSION,
    mask_to_model_frame,
    shift_zero_fill,
)
from dm_ahr_vit.xai.vit_xai import gaussian_chance_map  # noqa: E402

log = setup_logging()
METHODS = ("rollout", "gradcam_ln1", "gradcam_patch")
MAX_SHIFT_FRAC = 0.30
DRAWS = 100


def upsample(maps: np.ndarray, size: int, device: str) -> np.ndarray:
    t = torch.from_numpy(maps).float().to(device)[:, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()


def scores_row(prefix: str, sal: np.ndarray, mask: np.ndarray, fraction: float) -> dict:
    s = localization_scores(sal, mask, fraction)
    return {f"{prefix}_{k}": v for k, v in s.items()}


def explain(model, x, device, size):
    """Maps for a batch: {method: (B, size, size)} and the predicted classes."""
    with MultiLayerGradCAM(model) as cam:
        grids, pred = cam(x)
    roll = attention_rollout_batch(model, x)
    return {"rollout": upsample(roll, size, device),
            "gradcam_ln1": upsample(grids["last_ln1"], size, device),
            "gradcam_patch": upsample(grids["patch_grid"], size, device)}, pred.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_STORED_SIZES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("configs/vit.json")
    size = int(cfg["image_size"])
    fraction = float(load_config("configs/sampling.json")["xai_top_fraction"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    x1 = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
    x1 = x1[(x1["split"] == args.split) & (x1["status"] == "ok")]
    if args.limit:
        x1 = x1.head(args.limit)
    ids = set(x1["image_id"])
    mask_path = dict(zip(x1["image_id"], x1["mask_path"]))

    ckpt = artifact_dir("checkpoints", "vit", f"{args.arm}_seed{args.seed}") / "best.pt"
    model = build_model(pretrained=True).to(device)
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()
    ds = ArmDataset(args.arm, args.split, splits, isic_images_dir(), data_root() / "processed",
                    build_transforms(cfg, train=False), image_ids=ids)

    # Stored size is per image (hair-absent images pass through at full size in the 128 px arms);
    # read it from the file and check it against the sizes this arm is allowed to have.
    stored_of, allowed = {}, ARM_STORED_SIZES[args.arm]
    for image_id in ids:
        with Image.open(ds._path(image_id)) as fh:
            stored_of[image_id] = fh.size
        if stored_of[image_id] not in allowed:
            raise RuntimeError(f"{args.arm}/{image_id}: stored size {stored_of[image_id]} "
                               f"not in {sorted(allowed)}")
    share_squashed = float(np.mean([s == (128, 128) for s in stored_of.values()]))

    masks = {}
    for image_id in ids:
        with Image.open(resolve(mask_path[image_id])) as fh:
            masks[image_id] = mask_to_model_frame(fh, stored_of[image_id], size)

    gauss = gaussian_chance_map(size)
    rows = []
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate)
    lim = int(round(MAX_SHIFT_FRAC * size))
    yy, xx = np.nonzero(np.ones((size, size)))
    with timed("xai_v2", log) as tmr:
        for batch in loader:
            x = batch["image"].to(device)
            maps, pred = explain(model, x, device, size)
            shifts, x_tr = [], []
            for i, image_id in enumerate(batch["image_id"]):
                rng = np.random.default_rng(zlib.crc32(image_id.encode()))
                dx, dy = (int(v) for v in rng.integers(-lim, lim + 1, size=2))
                shifts.append((dx, dy))
                x_tr.append(shift_zero_fill(x[i], dx, dy))
            maps_tr, pred_tr = explain(model, torch.stack(x_tr), device, size)
            for i, image_id in enumerate(batch["image_id"]):
                m = masks[image_id]
                dx, dy = shifts[i]
                m_tr = shift_zero_fill(m, dx, dy)
                ly, lx = np.nonzero(m)
                row = {
                    "arm": args.arm, "seed": args.seed, "split": args.split, "frame": FRAME_VERSION,
                    "image_id": image_id, "group_id": batch["group_id"][i], "label": batch["dx"][i],
                    "pred": CLASSES[int(pred[i])], "correct": bool(CLASSES[int(pred[i])] == batch["dx"][i]),
                    "mask_area": float(m.mean()),
                    "contains_centre": bool(m[size // 2, size // 2]),
                    "centroid_offset": float(np.hypot(lx.mean() - (size - 1) / 2,
                                                     ly.mean() - (size - 1) / 2) / size) if len(lx) else np.nan,
                    "tr_dx_px": dx, "tr_dy_px": dy, "tr_mask_area": float(m_tr.mean()),
                }
                seed_i = zlib.crc32(image_id.encode())
                for tag, sal_set, mask_i, p in (("", maps, m, pred), ("tr_", maps_tr, m_tr, pred_tr)):
                    for meth in METHODS:
                        row.update(scores_row(f"{tag}{meth}", sal_set[meth][i], mask_i, fraction))
                    row.update(scores_row(f"{tag}gauss", gauss, mask_i, fraction))
                    row.update({f"{tag}{k}": v for k, v in random_map_baselines(
                        mask_i, seed_i, DRAWS, fraction, device=device).items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = artifact_dir("logs", "xai_v2")
    out = out_dir / f"xai_{args.arm}_seed{args.seed}_{args.split}.csv"
    df.to_csv(out, index=False)

    prov = {
        "arm": args.arm, "seed": args.seed, "split": args.split, "n": int(len(df)),
        "frame": FRAME_VERSION, "share_stored_128": share_squashed, "top_fraction": fraction,
        "random_draws": DRAWS, "max_shift_frac": MAX_SHIFT_FRAC,
        "accuracy_on_split": float(df["correct"].mean()),
        "mean_mask_area": float(df["mask_area"].mean()),
        "git_commit": args.commit,
        "sha256": {"checkpoint": sha256_file(ckpt),
                   "configs/vit.json": sha256_file("configs/vit.json"),
                   "configs/sampling.json": sha256_file("configs/sampling.json"),
                   "splits.csv": sha256_file(manifest_dir() / "splits.csv"),
                   "mask_manifest_x1.csv": sha256_file(manifest_dir() / "mask_manifest_x1.csv"),
                   **{p: sha256_file(Path(__file__).resolve().parents[1] / p) for p in (
                       "scripts/58_xai_corrected.py", "src/dm_ahr_vit/xai/geometry.py",
                       "src/dm_ahr_vit/xai/batched.py", "src/dm_ahr_vit/xai/vit_xai.py")}},
        "seconds": tmr["seconds"],
    }
    save_json(prov, out_dir / f"xai_{args.arm}_seed{args.seed}_{args.split}.json")
    record_run("xai_v2", gpu=device, seconds=tmr["seconds"], arm=args.arm, seed=args.seed,
               split=args.split, n=len(df))
    print(json.dumps({k: prov[k] for k in ("arm", "seed", "split", "n", "accuracy_on_split",
                                           "mean_mask_area", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
