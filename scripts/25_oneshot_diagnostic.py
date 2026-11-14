"""One-step reconstruction diagnostic: can each checkpoint denoise at all?

This separates "the weights never learned the mapping" from "sampling is
broken". For each checkpoint and timestep it takes a clean hairless reference,
noises it with a fixed epsilon, asks the model to predict that epsilon, and
recovers x0 in a single step. A model that has learned the mapping recovers
something close to the reference even without the full sampling loop.

Validation pairs only; the test set is never read.
"""

from __future__ import annotations

import argparse
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
    CHECKPOINT_NAME,
    load_model,
    preprocess,
)
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()

T_GRID = (50, 200, 500, 1000, 1500, 1900)
ZERO_COND_T = (200, 1000, 1900)
EPS_SEED = 20260101
CHUNK = 8


def resolve(path: str) -> Path:
    """Map a recorded Modal-volume path onto the local data root."""
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def to_unit(t: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor to [0, 1] HWC array for the project's PSNR convention."""
    a = t.detach().float().cpu().numpy()
    a = (a + 1.0) / 2.0
    return np.clip(a, 0.0, 1.0).transpose(0, 2, 3, 1)


def compare(pred: torch.Tensor, ref: torch.Tensor, win: int) -> tuple[float, float]:
    p, r = to_unit(pred), to_unit(ref)
    psnrs, ssims = [], []
    for i in range(p.shape[0]):
        psnrs.append(peak_signal_noise_ratio(r[i], p[i], data_range=1.0))
        ssims.append(structural_similarity(r[i], p[i], data_range=1.0,
                                           channel_axis=2, win_size=win))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def one_step(net, cond: torch.Tensor, x0: torch.Tensor, t: int,
             eps: torch.Tensor) -> torch.Tensor:
    """Reconstruct x0 in a single denoising step at index t.

    Mirrors `GaussianDiffusion.p_mean_variance` exactly: the noise-level
    embedding is `sqrt_alphas_cumprod_prev[t + 1]`, and x0 is recovered with
    `predict_start_from_noise`.
    """
    ab = net.alphas_cumprod[t]
    x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
    # `sqrt_alphas_cumprod_prev` is a numpy array upstream, wrapped in a list
    # before `repeat`; replicate that exactly rather than assuming a tensor.
    level = torch.FloatTensor(
        [net.sqrt_alphas_cumprod_prev[t + 1]]).repeat(x0.shape[0], 1).to(x0.device)
    eps_pred = net.denoise_fn(torch.cat([cond, x_t], dim=1), level)
    x0_pred = (net.sqrt_recip_alphas_cumprod[t] * x_t
               - net.sqrt_recipm1_alphas_cumprod[t] * eps_pred)
    return x0_pred.clamp(-1.0, 1.0)


def sweep(net, cond: torch.Tensor, x0: torch.Tensor, t: int, eps: torch.Tensor,
          win: int, zero_cond: bool = False) -> tuple[float, float, float]:
    """Mean one-step PSNR/SSIM over the set, chunked to bound GPU memory.

    Returns (psnr, ssim, mean_abs_change_vs_original_condition).
    """
    psnrs, ssims, deltas = [], [], []
    for i in range(0, x0.shape[0], CHUNK):
        sl = slice(i, i + CHUNK)
        c = torch.zeros_like(cond[sl]) if zero_cond else cond[sl]
        pred = one_step(net, c, x0[sl], t, eps[sl])
        p, s = compare(pred, x0[sl], win)
        psnrs.append(p * pred.shape[0])
        ssims.append(s * pred.shape[0])
        deltas.append(float((pred - one_step(net, cond[sl], x0[sl], t, eps[sl]))
                            .abs().mean()) * pred.shape[0] if zero_cond else 0.0)
        del pred, c
        torch.cuda.empty_cache()
    n = x0.shape[0]
    return sum(psnrs) / n, sum(ssims) / n, (sum(deltas) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    size = int(cfg["native_resolution"])
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    pairs = pairs[pairs["split"] == "val"].head(args.pairs).reset_index(drop=True)
    if pairs.empty:
        log.error("no validation pairs found")
        return 1

    refs, conds = [], []
    for r in pairs.itertuples(index=False):
        with Image.open(resolve(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(resolve(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))

    x0 = preprocess(refs, size).to(device)
    cond = preprocess(conds, size).to(device)
    gen = torch.Generator(device="cpu").manual_seed(EPS_SEED)
    eps = torch.randn(x0.shape, generator=gen).to(device)

    checkpoints = {
        "released": data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME,
        "ours": artifact_dir("checkpoints", "dmahr") / "best_gen.pth",
    }

    results: dict = {"pairs": int(len(pairs)), "t_grid": list(T_GRID),
                     "eps_seed": EPS_SEED, "device": device, "checkpoints": {}}

    with timed("oneshot_diagnostic", log) as tmr:
        floor_psnr, floor_ssim = compare(cond, x0, win)
        results["hairy_floor"] = {"psnr": floor_psnr, "ssim": floor_ssim}
        log.info("hairy floor: PSNR %.2f SSIM %.3f", floor_psnr, floor_ssim)

        for name, path in checkpoints.items():
            if not path.exists():
                log.error("missing checkpoint %s", path)
                return 1
            net = load_model(cfg, path, device=device)
            blocked: dict = {"path": str(path), "rows": [], "zero_cond": []}
            for t in T_GRID:
                p, s, _ = sweep(net, cond, x0, t, eps, win)
                blocked["rows"].append({"t": t, "psnr": p, "ssim": s})
                log.info("%s one-step t=%d: PSNR %.2f SSIM %.3f", name, t, p, s)
                if t in ZERO_COND_T:
                    zp, zs, delta = sweep(net, cond, x0, t, eps, win, zero_cond=True)
                    blocked["zero_cond"].append(
                        {"t": t, "psnr": zp, "ssim": zs,
                         "mean_abs_change": delta})
                    log.info("%s zero-cond t=%d: PSNR %.2f delta %.5f",
                             name, t, zp, delta)
            results["checkpoints"][name] = blocked
            del net
            torch.cuda.empty_cache()

    results["seconds"] = tmr["seconds"]
    save_json(results, artifact_dir("logs") / "oneshot_diagnostic.json")
    record_run("oneshot_diagnostic", gpu=device, seconds=tmr["seconds"],
               pairs=len(pairs))

    for name, blk in results["checkpoints"].items():
        print(name, [(r["t"], round(r["psnr"], 2)) for r in blk["rows"]])
    print("floor", round(floor_psnr, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
