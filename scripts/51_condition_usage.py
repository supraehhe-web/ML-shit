"""Stage E V4: does each model use the conditioning (hairy) image?

For each checkpoint and noise level, noise the clean target with a fixed epsilon and measure
epsilon-prediction error with the condition image kept, zeroed, and shuffled (taken from
another pair), split into hair pixels and other pixels. Validation pairs only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import CHECKPOINT_NAME, load_model, preprocess  # noqa: E402
from dm_ahr_vit.diffusion.train import _resolve_manifest_path  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()
T_GRID = (200, 500, 1000, 1500)
EPS_SEED = 20260101
CHUNK = 8
SHIFT = 7


def err_map(net, cond, x0, t, eps) -> torch.Tensor:
    alpha = net.sqrt_alphas_cumprod_prev[t - 1]
    x_t = alpha * x0 + (1.0 - alpha**2) ** 0.5 * eps
    level = torch.full((x0.shape[0], 1), float(alpha), device=x0.device)
    pred = net.denoise_fn(torch.cat([cond, x_t], dim=1), level)
    return (pred - eps).abs().mean(dim=1)          # B x H x W


@torch.no_grad()
def region_errors(net, cond, x0, t, eps, hair) -> dict:
    out = {}
    conds = {"kept": cond, "zeroed": torch.zeros_like(cond), "shuffled": torch.roll(cond, SHIFT, 0)}
    for name, c in conds.items():
        maps = torch.cat([err_map(net, c[i:i + CHUNK], x0[i:i + CHUNK], t, eps[i:i + CHUNK])
                          for i in range(0, x0.shape[0], CHUNK)])
        out[name] = {"hair": float(maps[hair].mean()), "other": float(maps[~hair].mean()),
                     "all": float(maps.mean())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    size = int(cfg["native_resolution"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    pairs = pairs[pairs["split"] == "val"].head(args.pairs).reset_index(drop=True)
    assert set(pairs["split"]) == {"val"}
    refs, conds, hair = [], [], []
    for r in pairs.itertuples(index=False):
        with Image.open(_resolve_manifest_path(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(_resolve_manifest_path(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))
        m = np.array(Image.open(_resolve_manifest_path(r.hair_mask_path)).convert("L"))
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127
        hair.append(cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    x0 = preprocess(refs, size).to(dev)
    cond = preprocess(conds, size).to(dev)
    hair_t = torch.from_numpy(np.stack(hair)).to(dev)
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(EPS_SEED)).to(dev)

    art = artifact_dir("checkpoints")
    ckpts = {"released": data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME,
             "repaired_20k": art / "dmahr_fixed20k" / "final_gen.pth",
             "repaired_50k": art / "dmahr" / "gen_iter050000.pth",
             "stageB_v2": art / "dmahr_v2" / "final_gen.pth"}
    result = {"pairs": len(pairs), "hair_pixel_fraction": float(hair_t.float().mean()),
              "t_grid": list(T_GRID), "eps_seed": EPS_SEED, "git_commit": args.commit,
              "models": {}}
    for name, path in ckpts.items():
        if not path.exists():
            log.warning("skip %s: missing %s", name, path)
            continue
        net = load_model(cfg, path, device=dev)
        rows = []
        for t in T_GRID:
            e = region_errors(net, cond, x0, t, eps, hair_t)
            row = {"t": t, **{f"{c}_{r}": e[c][r] for c in e for r in e[c]}}
            for r in ("hair", "other"):
                row[f"zeroed_over_kept_{r}"] = e["zeroed"][r] / e["kept"][r]
                row[f"shuffled_over_kept_{r}"] = e["shuffled"][r] / e["kept"][r]
            rows.append(row)
            log.info("%s t=%d kept(hair/other)=%.4f/%.4f zeroed x%.2f/%.2f shuffled x%.2f/%.2f",
                     name, t, e["kept"]["hair"], e["kept"]["other"],
                     row["zeroed_over_kept_hair"], row["zeroed_over_kept_other"],
                     row["shuffled_over_kept_hair"], row["shuffled_over_kept_other"])
        result["models"][name] = {"sha256": sha256_file(path), "rows": rows}
        del net
        torch.cuda.empty_cache()
    save_json(result, artifact_dir("logs") / "stageE_condition_usage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
