"""Gate 3: score the retrained Arm C on validation pairs.

Pre-registered thresholds live in `reports/prereg_fixed_trainer_arm_c.md`:
pass at >= floor + 2.0 dB, middle between floor and floor + 2.0, fail at or
below the floor. Includes the one-step and zero-condition diagnostics so the
learning curve and condition usage are recorded alongside the sampling score.
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
    clean_batch,
    load_model,
    preprocess,
)
from dm_ahr_vit.diffusion.train import _resolve_manifest_path  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()

PASS_MARGIN = 2.0
SAMPLE_STEPS = 2000
ONE_STEP_T = (200, 1000, 1500)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=182)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", default="gate3")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="sampler autocast; bf16 was the original default")
    ap.add_argument("--replicate-first", type=int, default=0,
                    help="sample the first pair this many times with different noise "
                         "instead of scoring --pairs distinct pairs")
    ap.add_argument("--no-tf32", action="store_true",
                    help="true fp32: disable TF32 convolutions and matmuls (PyTorch "
                         "enables TF32 convs on Ampere by default)")
    args = ap.parse_args()
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    device = "cuda"
    size = int(cfg["native_resolution"])

    ckpt = Path(args.checkpoint) if args.checkpoint else (
        artifact_dir("checkpoints", "dmahr") / "best_gen.pth")
    if not ckpt.exists():
        log.error("checkpoint missing: %s", ckpt)
        return 1

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].head(args.pairs).reset_index(drop=True)
    if args.replicate_first:
        val = pd.concat([val.head(1)] * args.replicate_first).reset_index(drop=True)
    refs, conds = [], []
    for r in val.itertuples(index=False):
        with Image.open(_resolve_manifest_path(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(_resolve_manifest_path(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))

    out: dict = {"checkpoint": str(ckpt), "pairs": int(len(val)),
                 "sample_steps": SAMPLE_STEPS, "pass_margin_db": PASS_MARGIN,
                 "precision": args.precision,
                 "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)}

    net = load_model(cfg, ckpt, device=device)

    # --- sampling score
    ps, ss, fps, fss = [], [], [], []
    for start in range(0, len(val), 8):
        sl = slice(start, start + 8)
        outs, _ = clean_batch(net, conds[sl], cfg, device=device,
                              seed=42 + (start if args.replicate_first else 0),
                              num_timesteps=SAMPLE_STEPS,
                              use_bf16=(args.precision == "bf16"))
        for o, ref, h in zip(outs, refs[sl], conds[sl]):
            a = np.asarray(o.resize(ref.size, Image.Resampling.BICUBIC),
                           dtype=np.float64) / 255.0
            b = np.asarray(ref, dtype=np.float64) / 255.0
            c = np.asarray(h.resize(ref.size, Image.Resampling.BICUBIC),
                           dtype=np.float64) / 255.0
            ps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
            ss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                            win_size=win))
            fps.append(peak_signal_noise_ratio(b, c, data_range=1.0))
            fss.append(structural_similarity(b, c, data_range=1.0, channel_axis=2,
                                             win_size=win))
        log.info("scored %d/%d", min(start + 8, len(val)), len(val))

    out["sampling"] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                       "hairy_floor_psnr": float(np.mean(fps)),
                       "hairy_floor_ssim": float(np.mean(fss)),
                       "per_pair": [{"pair_id": str(pid), "psnr": float(p), "ssim": float(s),
                                     "hairy_psnr": float(fp)}
                                    for pid, p, s, fp in zip(val["pair_id"], ps, ss, fps)],
                       "n": len(ps)}
    out["sampling"]["margin_db"] = (out["sampling"]["psnr"]
                                    - out["sampling"]["hairy_floor_psnr"])
    m = out["sampling"]["margin_db"]
    out["verdict"] = ("pass" if m >= PASS_MARGIN
                      else "middle" if m > 0 else "fail")

    # --- one-step reconstruction and condition usage
    x0 = preprocess(refs[:24], size).to(device)
    cond = preprocess(conds[:24], size).to(device)
    eps = torch.randn(x0.shape,
                      generator=torch.Generator().manual_seed(20260101)).to(device)
    rows = []
    for t in ONE_STEP_T:
        with torch.no_grad():
            a = float(net.sqrt_alphas_cumprod_prev[t - 1])
            x_t = a * x0 + (1.0 - a**2) ** 0.5 * eps
            lvl = torch.full((x0.shape[0], 1), a, device=device)
            e_pred = net.denoise_fn(torch.cat([cond, x_t], dim=1), lvl)
            x0_pred = (net.sqrt_recip_alphas_cumprod[t] * x_t
                       - net.sqrt_recipm1_alphas_cumprod[t] * e_pred).clamp(-1, 1)
            z_pred = net.denoise_fn(torch.cat([torch.zeros_like(cond), x_t], dim=1), lvl)
            err = float((e_pred - eps).abs().mean())
            err_z = float((z_pred - eps).abs().mean())
        pp = ((x0_pred + 1) / 2).detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        ref_np = ((x0 + 1) / 2).detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        pv = peak_signal_noise_ratio(ref_np, pp, data_range=1.0)
        rows.append({"t": t, "one_step_psnr_first_pair": float(pv),
                     "eps_err": err, "eps_err_zero_cond": err_z,
                     "condition_effect_ratio": err_z / err if err else None})
        log.info("t=%d one-step %.2f dB, cond effect %.2fx", t, pv,
                 (err_z / err) if err else 0)
    out["one_step"] = rows

    save_json(out, artifact_dir("logs") / f"{args.tag}_eval.json")
    record_run(args.tag, gpu=device, seconds=0.0,
               verdict=out["verdict"], margin_db=m)
    print(json_dumps(out))
    return 0


def json_dumps(obj) -> str:
    import json  # noqa: PLC0415
    return json.dumps(obj, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
