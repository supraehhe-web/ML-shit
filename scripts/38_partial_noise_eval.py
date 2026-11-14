"""Partial-noise-start DDIM evaluation on validation pairs (GPU).

Does starting the reverse chain from the noised hairy image, at a middling noise level
and with strided steps, give stable hair removal? Scored exactly like 30_gate3_eval.py
(128 px output upsampled to the original size, PSNR/SSIM against the hairless
reference, hairy input as the floor). Validation pairs only; the test pairs are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    load_model,
    postprocess,
    preprocess,
    respaced_super_resolution,
)
from dm_ahr_vit.diffusion.train import _resolve_manifest_path  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--t0s", default="600,1000,1500,1999")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many validation pairs, to evaluate on unseen ones")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    size = int(cfg["native_resolution"])
    T = int(cfg["num_timesteps"])
    ckpt = Path(args.ckpt)
    net = load_model(cfg, ckpt, device="cuda")

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = (pairs[pairs["split"] == "val"].iloc[args.offset:args.offset + args.pairs]
           .reset_index(drop=True))
    refs, conds = [], []
    for r in val.itertuples(index=False):
        with Image.open(_resolve_manifest_path(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(_resolve_manifest_path(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))
    x = preprocess(conds, size).to("cuda")

    def up(img: Image.Image, to: tuple[int, int]) -> np.ndarray:
        return np.asarray(img.resize(to, Image.Resampling.BICUBIC), dtype=np.float64) / 255.0

    ref_np = [np.asarray(r, dtype=np.float64) / 255.0 for r in refs]
    floor = np.array([peak_signal_noise_ratio(rf, up(c, r.size), data_range=1.0)
                      for rf, c, r in zip(ref_np, conds, refs)])

    rows = []
    for t0 in [int(v) for v in args.t0s.split(",")]:
        start = None if t0 >= T - 1 else t0
        per_seed = []
        psnrs, ssims = [], []
        for seed in range(args.seeds):
            out, _ = respaced_super_resolution(net, x, steps=args.steps, seed=42 + seed,
                                               start_t=start)
            imgs = postprocess(out)
            p = np.array([peak_signal_noise_ratio(rf, up(o, r.size), data_range=1.0)
                          for rf, o, r in zip(ref_np, imgs, refs)])
            s = np.array([structural_similarity(rf, up(o, r.size), data_range=1.0,
                                                channel_axis=2, win_size=win)
                          for rf, o, r in zip(ref_np, imgs, refs)])
            psnrs.append(p)
            ssims.append(s)
            per_seed.append(float(p.mean()))
        P = np.stack(psnrs)  # seeds x pairs
        row = {
            "t0": t0, "steps": args.steps, "seeds": args.seeds, "pairs": len(val),
            "mean_psnr": float(P.mean()), "mean_ssim": float(np.stack(ssims).mean()),
            "hairy_floor_psnr": float(floor.mean()),
            "mean_margin_db": float((P - floor[None, :]).mean()),
            "fraction_runs_below_floor": float((P < floor[None, :]).mean()),
            "seed_means": per_seed, "sd_of_seed_means": float(np.std(per_seed)),
        }
        rows.append(row)
        log.info("%s t0=%d  psnr %.2f  margin %+.2f  below-floor %.2f  ssim %.3f",
                 args.name, t0, row["mean_psnr"], row["mean_margin_db"],
                 row["fraction_runs_below_floor"], row["mean_ssim"])

    save_json({
        "name": args.name, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
        "git_commit": args.commit, "val_pair_ids_sha256": hashlib.sha256(
            "\n".join(val["pair_id"]).encode()).hexdigest(),
        "sampler": "ddim_respaced", "rows": rows,
    }, artifact_dir("logs") / f"{args.tag}_partial_noise.json")
    record_run(args.tag, gpu="cuda", seconds=0.0, rows=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
