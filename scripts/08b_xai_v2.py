"""XAI v2: Grad-CAM and attention rollout scored against masks aligned to the model's view.

Differences from the first XAI run (kept as the legacy, misaligned measurement):
  * The lesion mask goes through the same geometry as the image the ViT sees
    (`dm_ahr_vit.xai.geometry.model_space`), instead of being stretched to 224x224.
  * Every metric has its own chance level: overlap chance is the salient fraction
    (0.2); pointing chance is the lesion's area share. Both are stored per image.
  * Overlap ceiling min(1, 0.2 / area) and the ceiling-normalised overlap are stored.
  * Argmax locations are stored so border/corner hot spots can be audited, and the
    mean saliency maps per arm/seed are saved as .npy.
  * `--extras` (run for seed 42 only): off-centre robustness (image and mask
    translated by up to +-30%, image reflect-padded, mask zero-padded; lesions that
    keep < 50% of their area are excluded and counted) and insertion/deletion
    faithfulness curves on a fixed subset (every `--faith-stride`-th image).

Outputs: logs/xai_localization_v2_{arm}_seed{seed}_X1.csv, logs/xai_v2_summary_*.json,
         xai_maps/mean_{gradcam,rollout}_{arm}_seed{seed}.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.budget import gate  # noqa: E402
from dm_ahr_vit.config import CLASSES, load_config, save_json  # noqa: E402
from dm_ahr_vit.data.dataset import ArmDataset, build_transforms  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402
from dm_ahr_vit.xai.geometry import model_space  # noqa: E402
from dm_ahr_vit.xai.vit_xai import (  # noqa: E402
    GradCAM,
    attention_rollout,
    ceiling_normalised_overlap,
    gaussian_chance_map,
    insertion_deletion_auc,
    lesion_overlap,
    overlap_ceiling,
    overlap_chance,
    pointing_chance,
    pointing_game,
    seeded_shift,
    translate_array,
    upsample_map,
)

log = setup_logging()
SEC_PER_IMAGE = 126.4 / 1502  # measured, A100-80GB (xai_summary_A_seed42_X1.json)


def argmax_yx(sal: np.ndarray) -> tuple[float, float]:
    y, x = divmod(int(np.argmax(sal)), sal.shape[1])
    return y / sal.shape[0], x / sal.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["A", "B", "C", "Astar", "Creleased", "Cfixed"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--extras", action="store_true", help="shift test + faithfulness (seed 42 only)")
    ap.add_argument("--faith-stride", type=int, default=6)
    ap.add_argument("--gpu", default="A100-40GB")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--tier", default="X1")
    args = ap.parse_args()

    sampling = load_config("configs/sampling.json")
    cfg = load_config("configs/vit.json")
    fraction = float(sampling["xai_top_fraction"])
    size = int(cfg["image_size"])
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    x1 = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
    hair = pd.read_csv(manifest_dir() / "hair_screening.csv").set_index("image_id")
    test_x1 = x1[(x1["split"] == "test") & (x1["status"] == "ok")].sort_values("image_id")
    if args.limit:
        test_x1 = test_x1.head(args.limit)
    ids = set(test_x1["image_id"])
    mask_paths = dict(zip(test_x1["image_id"], test_x1["mask_path"]))
    dx_of = dict(zip(test_x1["image_id"], test_x1["dx"]))
    group_of = dict(zip(test_x1["image_id"], test_x1["group_id"]))

    per_image = SEC_PER_IMAGE * (2.0 if args.extras else 1.0) + (0.02 if args.extras else 0.0)
    gate(f"xai_v2_{args.arm}_{args.seed}", len(ids), per_image, args.gpu, args.budget_usd,
         fixed_seconds=60.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=True).to(device)
    ckpt = artifact_dir("checkpoints", "vit", f"{args.arm}_seed{args.seed}") / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt}")
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    ds = ArmDataset(args.arm, "test", splits, isic_images_dir(), data_root() / "processed",
                    build_transforms(cfg, train=False), image_ids=ids)
    gauss = gaussian_chance_map(size)
    sum_cam = np.zeros((size, size))
    sum_roll = np.zeros((size, size))
    rows = []
    with timed("xai_v2", log) as t, GradCAM(model) as cam_tool:
        for i in range(len(ds)):
            item = ds[i]
            x = item["image"][None].to(device)
            image_id = item["image_id"]
            raw = (np.asarray(Image.open(mask_paths[image_id]).convert("L")) > 127).astype(np.uint8)
            # Geometry is a property of the file the arm feeds the model.
            w, h = (128, 128) if (args.arm in ("Astar",) or
                                  (args.arm.startswith("C") and bool(hair.loc[image_id, "hair_present"]))) \
                else (int(hair.loc[image_id, "width"]), int(hair.loc[image_id, "height"]))
            mask = (model_space(raw, (w, h)) > 0)
            with torch.no_grad():
                logits = model(x)
            pred = int(logits.argmax(dim=1))
            cam = upsample_map(cam_tool(x, class_idx=pred), size)
            roll = upsample_map(attention_rollout(model, x), size)
            sum_cam += cam
            sum_roll += roll
            cy, cx = argmax_yx(cam)
            ry, rx = argmax_yx(roll)
            area = float(mask.mean())
            row = {
                "arm": args.arm, "seed": args.seed, "tier": args.tier, "geometry": "aligned",
                "image_id": image_id, "group_id": group_of[image_id], "dx": dx_of[image_id],
                "pred": CLASSES[pred], "correct": bool(CLASSES[pred] == dx_of[image_id]),
                "hair_present": bool(hair.loc[image_id, "hair_present"]),
                "mask_w_in": w, "mask_h_in": h, "lesion_area_frac": area,
                "overlap_chance": overlap_chance(fraction),
                "pointing_chance": pointing_chance(mask),
                "overlap_ceiling": overlap_ceiling(mask, fraction),
                "gradcam_overlap20": lesion_overlap(cam, mask, fraction),
                "rollout_overlap20": lesion_overlap(roll, mask, fraction),
                "gradcam_pointing": pointing_game(cam, mask),
                "rollout_pointing": pointing_game(roll, mask),
                "gauss_overlap20": lesion_overlap(gauss, mask, fraction),
                "gauss_pointing": pointing_game(gauss, mask),
                "gradcam_argmax_y": cy, "gradcam_argmax_x": cx,
                "rollout_argmax_y": ry, "rollout_argmax_x": rx,
                "gradcam_finite": bool(np.isfinite(cam).all()),
                "rollout_finite": bool(np.isfinite(roll).all()),
            }
            row["gradcam_overlap_norm"] = ceiling_normalised_overlap(row["gradcam_overlap20"], mask, fraction)
            row["rollout_overlap_norm"] = ceiling_normalised_overlap(row["rollout_overlap20"], mask, fraction)
            row["gauss_overlap_norm"] = ceiling_normalised_overlap(row["gauss_overlap20"], mask, fraction)

            if args.extras:
                dy, dxs = seeded_shift(image_id, size, 0.3)
                shifted_mask = translate_array(mask.astype(np.uint8), dy, dxs, "constant") > 0
                kept = float(shifted_mask.sum() / max(mask.sum(), 1))
                row["shift_dy"], row["shift_dx"], row["shift_kept"] = dy, dxs, kept
                if kept >= 0.5:
                    img = x[0].permute(1, 2, 0).cpu().numpy()
                    xs = torch.from_numpy(np.ascontiguousarray(
                        translate_array(img, dy, dxs, "reflect"))).permute(2, 0, 1)[None].to(device)
                    with torch.no_grad():
                        ps = int(model(xs).argmax(dim=1))
                    cam_s = upsample_map(cam_tool(xs, class_idx=ps), size)
                    roll_s = upsample_map(attention_rollout(model, xs), size)
                    gs = translate_array(gauss, dy, dxs, "constant")
                    row.update({
                        "shift_gradcam_overlap20": lesion_overlap(cam_s, shifted_mask, fraction),
                        "shift_rollout_overlap20": lesion_overlap(roll_s, shifted_mask, fraction),
                        "shift_gradcam_pointing": pointing_game(cam_s, shifted_mask),
                        "shift_rollout_pointing": pointing_game(roll_s, shifted_mask),
                        "shift_gauss_overlap20": lesion_overlap(gs, shifted_mask, fraction),
                        "shift_gauss_pointing": pointing_game(gs, shifted_mask),
                        "shift_pointing_chance": pointing_chance(shifted_mask),
                    })
                if i % max(1, args.faith_stride) == 0:
                    ins_c, del_c = insertion_deletion_auc(model, x, cam, pred)
                    ins_r, del_r = insertion_deletion_auc(model, x, roll, pred)
                    rng = np.random.default_rng(i)
                    ins_u, del_u = insertion_deletion_auc(model, x, rng.random(cam.shape), pred)
                    row.update({"faith_gradcam_ins": ins_c, "faith_gradcam_del": del_c,
                                "faith_rollout_ins": ins_r, "faith_rollout_del": del_r,
                                "faith_random_ins": ins_u, "faith_random_del": del_u})
            rows.append(row)
        df = pd.DataFrame(rows)

    stem = f"{args.arm}_seed{args.seed}_{args.tier}"
    df.to_csv(artifact_dir("logs") / f"xai_localization_v2_{stem}.csv", index=False)
    maps = artifact_dir("xai_maps")
    np.save(maps / f"mean_gradcam_{args.arm}_seed{args.seed}.npy", (sum_cam / len(df)).astype(np.float32))
    np.save(maps / f"mean_rollout_{args.arm}_seed{args.seed}.npy", (sum_roll / len(df)).astype(np.float32))
    summary = {"arm": args.arm, "seed": args.seed, "n": int(len(df)), "extras": bool(args.extras),
               "geometry": "aligned", "seconds": t["seconds"],
               "means": {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind == "f"}}
    save_json(summary, artifact_dir("logs") / f"xai_v2_summary_{stem}.json")
    record_run("xai_v2", gpu=device, seconds=t["seconds"], arm=args.arm, seed=args.seed,
               n=len(df), extras=bool(args.extras))
    print({k: round(v, 4) for k, v in summary["means"].items()
           if k.endswith(("overlap20", "pointing", "chance", "ceiling", "norm"))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
