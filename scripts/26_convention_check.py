"""Does each checkpoint expect the sampler's noise-level encoding or `t/T`?

`artifacts/logs/arm_c_v2_noise_mse.json` claimed our checkpoint was trained under
a different encoding from the one the sampler uses. Its script was never
committed, so the claim could not be checked. This script re-derives it from
source, against the encoder the sampler actually calls.

For each checkpoint and timestep, noise the hairless reference with a fixed
epsilon and measure epsilon-prediction error under both encodings. The encoding
the checkpoint was trained under should show the lower error.
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

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    CHECKPOINT_NAME,
    load_model,
    preprocess,
)
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()

T_GRID = (200, 500, 1000, 1500)
EPS_SEED = 20260101
CHUNK = 8


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def eps_error(net, cond, x0, t, eps, encoding: str, zero_cond: bool = False) -> float:
    """Mean absolute epsilon-prediction error under one encoding."""
    alpha = net.sqrt_alphas_cumprod_prev[t - 1]
    x_t = alpha * x0 + (1.0 - alpha**2) ** 0.5 * eps
    c = torch.zeros_like(cond) if zero_cond else cond
    if encoding == "sampler":       # what p_sample feeds: sqrt(alpha_cumprod_prev)
        level = torch.full((x0.shape[0], 1), float(alpha), device=x0.device)
    elif encoding == "t_over_T":    # what our trainer used to feed
        level = torch.full((x0.shape[0], 1), t / net.num_timesteps, device=x0.device)
    else:
        raise ValueError(encoding)
    pred = net.denoise_fn(torch.cat([c, x_t], dim=1), level)
    return float((pred - eps).abs().mean())


def sweep(net, cond, x0, t, eps, encoding, zero_cond=False) -> float:
    errs = []
    for i in range(0, x0.shape[0], CHUNK):
        sl = slice(i, i + CHUNK)
        errs.append(eps_error(net, cond[sl], x0[sl], t, eps[sl], encoding, zero_cond)
                    * x0[sl].shape[0])
        torch.cuda.empty_cache()
    return sum(errs) / x0.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--variants", default="",
                    help="extra name=path checkpoints, comma separated")
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    size = int(cfg["native_resolution"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    pairs = pairs[pairs["split"] == "val"].head(args.pairs).reset_index(drop=True)
    refs, conds = [], []
    for r in pairs.itertuples(index=False):
        with Image.open(resolve(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(resolve(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))
    x0 = preprocess(refs, size).to(device)
    cond = preprocess(conds, size).to(device)
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(EPS_SEED)).to(device)

    ckpts = {
        "released": data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME,
        "ours": artifact_dir("checkpoints", "dmahr") / "best_gen.pth",
    }
    for item in [v for v in args.variants.split(",") if v]:
        name, _, path = item.partition("=")
        ckpts[name] = Path(path)

    out: dict = {"pairs": int(len(pairs)), "eps_seed": EPS_SEED,
                 "t_grid": list(T_GRID), "checkpoints": {}}
    with timed("convention_check", log) as tmr:
        for name, path in ckpts.items():
            if not path.exists():
                log.warning("skip %s: missing %s", name, path)
                continue
            net = load_model(cfg, path, device=device)
            blk = {"path": str(path), "rows": []}
            for t in T_GRID:
                sampler = sweep(net, cond, x0, t, eps, "sampler")
                linear = sweep(net, cond, x0, t, eps, "t_over_T")
                zeroed = sweep(net, cond, x0, t, eps, "sampler", zero_cond=True)
                row = {"t": t, "eps_err_sampler_encoding": sampler,
                       "eps_err_t_over_T_encoding": linear,
                       "prefers": "sampler" if sampler < linear else "t_over_T",
                       "ratio_linear_over_sampler": linear / sampler if sampler else None,
                       "eps_err_sampler_with_zeroed_condition": zeroed,
                       "condition_effect_ratio": zeroed / sampler if sampler else None}
                blk["rows"].append(row)
                log.info("%s t=%d sampler %.5f linear %.5f (%.1fx) zeroed %.5f (%.2fx)",
                         name, t, sampler, linear,
                         (linear / sampler) if sampler else 0,
                         zeroed, (zeroed / sampler) if sampler else 0)
            out["checkpoints"][name] = blk
            del net
            torch.cuda.empty_cache()

    out["seconds"] = tmr["seconds"]
    save_json(out, artifact_dir("logs") / "noise_convention_check.json")
    record_run("convention_check", gpu=device, seconds=tmr["seconds"],
               pairs=len(pairs))
    for n, b in out["checkpoints"].items():
        print(n, [(r["t"], r["prefers"], round(r["ratio_linear_over_sampler"], 1))
                  for r in b["rows"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
