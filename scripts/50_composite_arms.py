"""Stage E V1: DullRazor (Arm B) and Cv2 on validation composites with known clean targets.

Real strands harvested from validation photos are pasted onto other validation photos (same
generator and seeds as 45_composite_eval.py, so the first 48 items are those). Each arm is
scored at 128 px, the diffusion model's native size: PSNR against the clean target, strand
removal, and damage to lesion pixels the hair did not cover (see data/composite_metrics.py).
Validation split only; build_datasets refuses anything else.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, load_frozen, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.data.composite_metrics import bootstrap_mean_ci, item_metrics  # noqa: E402
from dm_ahr_vit.data.hair_composite import build_datasets  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    load_model,
    postprocess,
    preprocess,
    respaced_super_resolution,
)
from dm_ahr_vit.diffusion.train import _resolve_manifest_path  # noqa: E402
from dm_ahr_vit.hair.dullrazor import apply_dullrazor  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()
METRICS = ("psnr", "hair_removal", "lesion_mae", "lesion_changed", "skin_mae", "skin_changed")


def resize(a: np.ndarray, s: int, mode=Image.Resampling.BICUBIC) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((s, s), mode))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--t0", type=int, default=600)
    ap.add_argument("--data-seed", type=int, default=100)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    size = int(cfg["native_resolution"])
    screen = load_frozen("hair_screen", "configs/hair_screen.json")
    dull = load_frozen("dullrazor", "configs/dullrazor.json")
    masks = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv").set_index("image_id")
    _, val = build_datasets(manifest_dir() / "hair_screening.csv", isic_images_dir(),
                            data_root() / "hair_layers", size=size, val_length=args.items,
                            seed=args.data_seed)
    assert val.split == "val"

    items = []
    for i in range(len(val)):
        target, comp, alpha, info = val.compose(i)
        lesion = np.zeros((size, size), bool)
        tid = info["target"]
        if tid in masks.index and bool(masks.loc[tid, "has_mask"]):
            m = np.array(Image.open(_resolve_manifest_path(masks.loc[tid, "mask_path"])).convert("L"))
            lesion = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127
        a128 = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_LINEAR)
        det, st = detect_hair(comp, screen)
        b_full = apply_dullrazor(comp, det, dull) if st["hair_present"] else comp
        items.append({"tid": tid, "t128": resize(target, size), "c128": resize(comp, size),
                      "a128": a128, "lesion": lesion, "b128": resize(b_full, size),
                      "comp_im": Image.fromarray(comp), "b_applied": bool(st["hair_present"])})
    log.info("%d composites, %d with lesion masks, B applied to %d", len(items),
             sum(int(x["lesion"].any()) for x in items), sum(x["b_applied"] for x in items))

    ckpt = artifact_dir("checkpoints") / "dmahr_v2" / "final_gen.pth"
    net = load_model(cfg, ckpt, device="cuda")
    x = preprocess([it["comp_im"] for it in items], size).to("cuda")
    cv2_outs = []
    for k in range(args.seeds):
        out, _ = respaced_super_resolution(net, x, steps=50, seed=42 + k, start_t=args.t0)
        cv2_outs.append([np.asarray(im) for im in postprocess(out)])

    rows = []
    for j, it in enumerate(items):
        for arm, outs in (("hairy_input", [it["c128"]]), ("B_dullrazor", [it["b128"]]),
                          ("Cv2", [c[j] for c in cv2_outs])):
            ms = [item_metrics(o, it["t128"], it["c128"], it["a128"], it["lesion"]) for o in outs]
            row = {"item": j, "target": it["tid"], "arm": arm, "b_applied": it["b_applied"]}
            for k in METRICS:
                row[k] = float(np.nanmean([m[k] for m in ms])) if not all(
                    np.isnan(m[k]) for m in ms) else float("nan")
            row["lesion_pixels_clear_of_hair"] = ms[0]["lesion_pixels_clear_of_hair"]
            rows.append(row)
    df = pd.DataFrame(rows)
    artifact_dir("logs")
    df.to_csv(artifact_dir("logs") / "stageE_composite_arms.csv", index=False)

    summ = {"items": len(items), "lesion_items": int(df[df.arm == "Cv2"]["lesion_mae"].notna().sum()),
            "arms": {}, "paired_Cv2_minus_B": {}}
    for arm, g in df.groupby("arm"):
        summ["arms"][arm] = {k: dict(zip(("mean", "ci_low", "ci_high"),
                                         bootstrap_mean_ci(g[k].to_numpy()))) for k in METRICS}
    piv = {k: df.pivot(index="item", columns="arm", values=k) for k in METRICS}
    for k in METRICS:
        d = (piv[k]["Cv2"] - piv[k]["B_dullrazor"]).to_numpy()
        summ["paired_Cv2_minus_B"][k] = dict(zip(("mean", "ci_low", "ci_high"),
                                                 bootstrap_mean_ci(d)))
    summ["provenance"] = {
        "git_commit": args.commit, "t0": args.t0, "noise_seeds": args.seeds,
        "data_seed": args.data_seed, "checkpoint_sha256": sha256_file(ckpt),
        "target_ids_sha256": hashlib.sha256("\n".join(str(i["tid"]) for i in items).encode()).hexdigest(),
        "screen_config_hash": screen.get("config_hash"), "dullrazor_config_hash": dull.get("config_hash")}
    save_json(summ, artifact_dir("logs") / "stageE_composite_arms.json")
    for arm, d in summ["arms"].items():
        log.info("%s: " + " ".join(f"{k}={d[k]['mean']:.3f}" for k in METRICS), arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
