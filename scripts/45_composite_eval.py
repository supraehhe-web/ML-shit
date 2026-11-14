"""Hair removal against ground truth on held-out real-hair composites (GPU, validation only).

Real strands harvested from validation images are pasted onto other validation images, so the
clean target is known. Reports, per model and start level t0:
  - PSNR of the output (128 px output upsampled to full size, scored like 38_partial_noise_eval)
    against the clean target, with the hairy input as the floor and the 128 px resize-only
    version of the clean target as the ceiling any 128 px model can reach
  - hair removal at native 128 px: 1 - E_out / E_floor, where E is the mean absolute error to the
    clean target weighted by the strand alpha (1.0 = strands fully gone, 0 = unchanged)
The test split is never read: `build_datasets` only accepts train and validation ids.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.data.hair_composite import build_datasets  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    load_model,
    postprocess,
    preprocess,
    respaced_super_resolution,
)
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--t0s", default="600,1000")
    ap.add_argument("--items", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--data-seed", type=int, default=100,
                    help="compositing draws; differs from the training seed")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    size, T = int(cfg["native_resolution"]), int(cfg["num_timesteps"])
    _, val = build_datasets(manifest_dir() / "hair_screening.csv", isic_images_dir(),
                            data_root() / "hair_layers", size=size,
                            val_length=args.items, seed=args.data_seed)
    assert val.split == "val"
    targets, comps, alphas, ids = [], [], [], []
    for i in range(len(val)):
        t, c, a, info = val.compose(i)
        targets.append(t), comps.append(c), alphas.append(a), ids.append(info["target"])
    n = len(targets)
    full = [np.asarray(t, dtype=np.float64) / 255.0 for t in targets]
    comp_im = [Image.fromarray(c) for c in comps]
    t128 = [np.asarray(Image.fromarray(t).resize((size, size), Image.Resampling.BICUBIC),
                       dtype=np.float64) / 255.0 for t in targets]
    c128 = [np.asarray(c.resize((size, size), Image.Resampling.BICUBIC),
                       dtype=np.float64) / 255.0 for c in comp_im]
    a128 = [np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0 for a in alphas]

    def up(arr128: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        im = Image.fromarray((arr128 * 255).round().astype(np.uint8))
        return np.asarray(im.resize((hw[1], hw[0]), Image.Resampling.BICUBIC),
                          dtype=np.float64) / 255.0

    floor = np.array([psnr(f, up(c, f.shape[:2])) for f, c in zip(full, c128)])
    ceiling = np.array([psnr(f, up(t, f.shape[:2])) for f, t in zip(full, t128)])
    w = [a[..., None] for a in a128]
    e_floor = np.array([float((np.abs(c - t) * wi).sum() / (wi.sum() * 3 + 1e-9))
                        for c, t, wi in zip(c128, t128, w)])
    x = preprocess(comp_im, size).to("cuda")

    results = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        net = load_model(cfg, Path(path), device="cuda")
        for t0 in [int(v) for v in args.t0s.split(",")]:
            start = None if t0 >= T - 1 else t0
            P, R = [], []
            for seed in range(args.seeds):
                out, _ = respaced_super_resolution(net, x, steps=args.steps, seed=42 + seed,
                                                   start_t=start)
                imgs = [np.asarray(im, dtype=np.float64) / 255.0 for im in postprocess(out)]
                P.append([psnr(f, up(o, f.shape[:2])) for f, o in zip(full, imgs)])
                e_out = np.array([float((np.abs(o - t) * wi).sum() / (wi.sum() * 3 + 1e-9))
                                  for o, t, wi in zip(imgs, t128, w)])
                R.append(1.0 - e_out / np.maximum(e_floor, 1e-9))
            P, R = np.array(P), np.array(R)
            row = {"model": name, "t0": t0, "items": n, "seeds": args.seeds,
                   "mean_psnr": float(P.mean()), "floor_psnr": float(floor.mean()),
                   "ceiling_psnr": float(ceiling.mean()),
                   "margin_db": float((P - floor[None]).mean()),
                   "fraction_of_gap_closed": float(((P - floor[None]) /
                                                    (ceiling - floor)[None]).mean()),
                   "fraction_runs_below_floor": float((P < floor[None]).mean()),
                   "hair_removal_native": float(R.mean()),
                   "hair_removal_native_median": float(np.median(R)),
                   "checkpoint_sha256": sha256_file(path)}
            results.append(row)
            log.info("%s t0=%d psnr %.2f floor %.2f ceiling %.2f margin %+.2f gap closed %.2f "
                     "hair removal %.2f", name, t0, row["mean_psnr"], row["floor_psnr"],
                     row["ceiling_psnr"], row["margin_db"], row["fraction_of_gap_closed"],
                     row["hair_removal_native"])
            del out
        del net
        torch.cuda.empty_cache()

    save_json({"rows": results, "git_commit": args.commit, "data_seed": args.data_seed,
               "target_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
               "steps": args.steps},
              artifact_dir("logs") / f"{args.tag}_composite_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
