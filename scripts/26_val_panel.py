"""Visual panel on validation pairs: hairy | reference | released | retrained snapshots.

Validation pairs only. Rows are pairs; columns are inputs and outputs of the same
respaced sampler. PSNR/SSIM per tile are printed in the caption strip so the panel
is self-describing. Agent-reviewed, not expert-reviewed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config  # noqa: E402
from dm_ahr_vit.diffusion import diagnostics as dg  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import CHECKPOINT_NAME, clean_batch, load_model  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--snapshots", default="5000,20000")
    ap.add_argument("--tag", default="arm_c_v2_panel")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].sort_values("pair_id").head(args.n)
    hairy = [Image.open(p).convert("RGB") for p in val["hairy_path"]]
    refs = [Image.open(p).convert("RGB") for p in val["reference_path"]]
    ckpts = [("released", data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME)]
    fixed = artifact_dir("checkpoints", "dmahr_fixed")
    for s in [int(x) for x in args.snapshots.split(",") if x]:
        ckpts.append((f"fixed_{s}", fixed / f"iter_{s}_gen.pth"))
    ckpts.append(("first_trainer", artifact_dir("checkpoints", "dmahr") / "best_gen.pth"))

    cols = [("hairy input", hairy), ("reference", refs)]
    for name, path in ckpts:
        net = load_model(cfg, path, device=device)
        mode = "respaced"
        outs, _ = clean_batch(net, hairy, cfg, device=device, seed=0, num_timesteps=args.steps,
                              use_bf16=True, schedule_mode=mode)
        cols.append((name, outs))
        del net
    th = 128
    canvas = Image.new("RGB", (th * len(cols), (th + 14) * len(val)), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    for ci, (name, imgs) in enumerate(cols):
        for ri, im in enumerate(imgs):
            m = dg.psnr_ssim(im, refs[ri])
            y = ri * (th + 14)
            canvas.paste(im.convert("RGB").resize((th, th), Image.Resampling.BICUBIC), (ci * th, y))
            d.text((ci * th + 2, y + th + 1), f"{name[:11]} {m['psnr']:.1f}/{m['ssim']:.2f}", fill=(255, 255, 255))
    out = artifact_dir("panels") / f"{args.tag}.png"
    canvas.save(out)
    stats = {name: {"mean_psnr": float(np.mean([dg.psnr_ssim(i, r)["psnr"] for i, r in zip(imgs, refs)])),
                    "mean_pixel_mean": float(np.mean([np.asarray(i, dtype=np.float64).mean() / 255 for i in imgs])),
                    "mean_pixel_std": float(np.mean([np.asarray(i, dtype=np.float64).std() / 255 for i in imgs]))}
             for name, imgs in cols}
    import json
    (artifact_dir("logs") / f"{args.tag}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
