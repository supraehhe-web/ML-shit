"""Run the diffusion model on real-hair images (GPU). Train-split images only.

Writes 128 px PNGs, one folder per start level, for `37_real_hair_check.py measure`
to score with the frozen hair detector. The ids come from the hash-ordered train-split
list built by `37_real_hair_check.py select`; no validation or test image is read here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    load_model,
    postprocess,
    preprocess,
    respaced_super_resolution,
)
from dm_ahr_vit.paths import data_root, isic_images_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated ISIC image ids")
    ap.add_argument("--t0s", default="600,1000,1500")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    size = int(cfg["native_resolution"])
    net = load_model(cfg, Path(args.ckpt), device="cuda")
    ids = [i for i in args.ids.split(",") if i]
    imgs = []
    for iid in ids:
        with Image.open(isic_images_dir() / f"{iid}.jpg") as fh:
            imgs.append(fh.convert("RGB"))
    x = preprocess(imgs, size).to("cuda")
    for t0 in [int(v) for v in args.t0s.split(",")]:
        out, _ = respaced_super_resolution(net, x, steps=args.steps, seed=args.seed,
                                           start_t=None if t0 >= 1999 else t0)
        out_dir = data_root() / "processed" / "diag_realhair" / f"{args.name}_t{t0}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for iid, im in zip(ids, postprocess(out)):
            im.save(out_dir / f"{iid}.png")
        log.info("wrote %d images to %s", len(ids), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
