"""Grad-CAM and attention-rollout localization against X1 lesion masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json  # noqa: E402
from dm_ahr_vit.data.masks import load_mask_224  # noqa: E402
from dm_ahr_vit.data.dataset import ArmDataset, build_transforms  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402
from dm_ahr_vit.xai.vit_xai import (  # noqa: E402
    GradCAM,
    attention_rollout,
    gaussian_chance_map,
    lesion_overlap,
    pointing_game,
    uniform_chance_overlap,
    upsample_map,
)

log = setup_logging()


def overlay(base_img: np.ndarray, sal: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cm.jet(sal)[..., :3]
    base = base_img.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A", "B", "C", "Astar", "Creleased", "Cv2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tier", default="X1")
    args = ap.parse_args()

    sampling = load_config("configs/sampling.json")
    cfg = load_config("configs/vit.json")
    fraction = float(sampling["xai_top_fraction"])
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    x1 = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
    test_x1 = x1[(x1["split"] == "test") & (x1["status"] == "ok")]
    if args.limit:
        test_x1 = test_x1.head(args.limit)
    ids = set(test_x1["image_id"])
    mask_paths = dict(zip(test_x1["image_id"], test_x1["mask_path"]))
    dx_of = dict(zip(test_x1["image_id"], test_x1["dx"]))
    group_of = dict(zip(test_x1["image_id"], test_x1["group_id"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=True).to(device)
    ckpt = artifact_dir("checkpoints", "vit", f"{args.arm}_seed{args.seed}") / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt}")
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    ds = ArmDataset(args.arm, "test", splits, isic_images_dir(), data_root() / "processed",
                    build_transforms(cfg, train=False), image_ids=ids)
    rows, panels = [], []
    gauss = gaussian_chance_map(int(cfg["image_size"]))
    with timed("xai", log) as t, GradCAM(model) as cam_tool:
        for i in range(len(ds)):
            item = ds[i]
            x = item["image"][None].to(device)
            image_id = item["image_id"]
            mask = load_mask_224(Path(mask_paths[image_id]), int(cfg["image_size"]))
            with torch.no_grad():
                logits = model(x)
            pred = int(logits.argmax(dim=1))
            cam = upsample_map(cam_tool(x, class_idx=pred), int(cfg["image_size"]))
            roll = upsample_map(attention_rollout(model, x), int(cfg["image_size"]))
            row = {
                "arm": args.arm,
                "seed": args.seed,
                "tier": args.tier,
                "image_id": image_id,
                "group_id": group_of[image_id],
                "dx": dx_of[image_id],
                "pred": CLASSES[pred],
                "correct": bool(CLASSES[pred] == dx_of[image_id]),
                "lesion_area_frac": float(mask.mean()),
                "gradcam_overlap20": lesion_overlap(cam, mask, fraction),
                "rollout_overlap20": lesion_overlap(roll, mask, fraction),
                "gradcam_pointing": pointing_game(cam, mask),
                "rollout_pointing": pointing_game(roll, mask),
                "chance_uniform_overlap20": uniform_chance_overlap(mask, fraction, seed=i),
                "chance_gaussian_overlap20": lesion_overlap(gauss, mask, fraction),
                "chance_gaussian_pointing": pointing_game(gauss, mask),
                "gradcam_finite": bool(np.isfinite(cam).all()),
                "rollout_finite": bool(np.isfinite(roll).all()),
                "gradcam_shape": str(cam.shape),
                "rollout_shape": str(roll.shape),
            }
            rows.append(row)
            if len(panels) < 10 and i % max(1, len(ds) // 10) == 0:
                base = (item["image"].permute(1, 2, 0).numpy() * 0.229 + 0.485)
                panels.append((np.clip(base, 0, 1), cam, roll, mask, image_id))
        df = pd.DataFrame(rows)

    out_csv = artifact_dir("logs") / f"xai_localization_{args.arm}_seed{args.seed}_{args.tier}.csv"
    df.to_csv(out_csv, index=False)

    # Overlay panel for human/agent review.
    if panels:
        cols = 5
        th = 150
        canvas = np.zeros((len(panels) * th, cols * th, 3), dtype=np.float32)
        for r, (base, cam, roll, mask, image_id) in enumerate(panels):
            canvas[r * th:(r + 1) * th, 0:th] = np.array(
                Image.fromarray((base * 255).astype(np.uint8)).resize((th, th)))
            canvas[r * th:(r + 1) * th, th:2 * th] = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (th, th), Image.Resampling.NEAREST))[..., None] / 255.0
            canvas[r * th:(r + 1) * th, 2 * th:3 * th] = overlay(
                np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((th, th))),
                np.array(Image.fromarray(cam).resize((th, th))))
            canvas[r * th:(r + 1) * th, 3 * th:4 * th] = overlay(
                np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((th, th))),
                np.array(Image.fromarray(roll).resize((th, th))))
            canvas[r * th:(r + 1) * th, 4 * th:5 * th] = np.zeros((th, th, 3))
        Image.fromarray((canvas * 255).astype(np.uint8)).save(
            artifact_dir("panels") / f"xai_panel_{args.arm}_seed{args.seed}.jpg", quality=90)

    summary = {
        "arm": args.arm,
        "seed": args.seed,
        "tier": args.tier,
        "n": int(len(df)),
        "gradcam_overlap20_mean": float(df["gradcam_overlap20"].mean()),
        "rollout_overlap20_mean": float(df["rollout_overlap20"].mean()),
        "gradcam_pointing_mean": float(df["gradcam_pointing"].mean()),
        "rollout_pointing_mean": float(df["rollout_pointing"].mean()),
        "chance_uniform_overlap20_mean": float(df["chance_uniform_overlap20"].mean()),
        "chance_gaussian_overlap20_mean": float(df["chance_gaussian_overlap20"].mean()),
        "chance_gaussian_pointing_mean": float(df["chance_gaussian_pointing"].mean()),
        "all_maps_finite": bool(df["gradcam_finite"].all() and df["rollout_finite"].all()),
        "map_shape": df["gradcam_shape"].iloc[0] if len(df) else None,
        "seconds": t["seconds"],
    }
    save_json(summary, artifact_dir("logs") / f"xai_summary_{args.arm}_seed{args.seed}_{args.tier}.json")
    record_run("xai", gpu=device, seconds=t["seconds"], arm=args.arm, seed=args.seed,
               n=len(df))
    print(pd.Series(summary).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

