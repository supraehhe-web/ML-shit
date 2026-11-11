"""Benchmark DM-AHR sampling precision/compile variants.

The sampler is 2,000 sequential UNet calls, so per-image cost is dominated by
the UNet forward pass. This measures whether bf16 autocast or torch.compile
reduces that cost without changing the output materially.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    postprocess,
    preprocess,
)
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


@torch.no_grad()
def sample(net, x, cfg, use_bf16: bool, seed: int = 42):
    """Run the full 2,000-step trajectory and return the final batch."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        traj = net.super_resolution(x, continous=True)
    return traj[-x.shape[0]:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a100")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    device = "cuda"
    ckpt = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
    ids = screen[(screen["split"] == "test") & (screen["hair_present"])]["image_id"].head(64).tolist()
    images = [Image.open(isic_images_dir() / f"{i}.jpg").convert("RGB") for i in ids]

    net = load_model(cfg, ckpt, device=device)
    results: dict = {"tag": args.tag, "device": torch.cuda.get_device_name(0),
                     "images": len(images)}
    bs = args.batch

    # Baseline: fp32, batch 32 (the current production setting).
    x32 = preprocess(images[:bs], 128).to(device)
    t0 = time.time()
    out_fp32 = sample(net, x32, cfg, use_bf16=False)
    fp32_s = time.time() - t0
    results[f"fp32_b{bs}"] = {"seconds": round(fp32_s, 2),
                              "s_per_image": round(fp32_s / bs, 3)}

    # bf16, batch 32.
    t0 = time.time()
    out_bf16 = sample(net, x32, cfg, use_bf16=True)
    bf16_s = time.time() - t0
    results[f"bf16_b{bs}"] = {"seconds": round(bf16_s, 2),
                              "s_per_image": round(bf16_s / bs, 3),
                              "speedup": round(fp32_s / bf16_s, 2)}

    # How much does bf16 change the pixels?
    a = postprocess(out_fp32)[0]
    b = postprocess(out_bf16)[0]
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    results["bf16_vs_fp32"] = {
        "mean_abs_diff": round(float(np.abs(arr_a - arr_b).mean()), 3),
        "max_abs_diff": round(float(np.abs(arr_a - arr_b).max()), 3),
        "correlation": round(float(np.corrcoef(arr_a.ravel(), arr_b.ravel())[0, 1]), 5),
    }

    bf16_64 = 0.0
    if len(images) >= 64:
        x64 = preprocess(images[:64], 128).to(device)
        t0 = time.time()
        sample(net, x64, cfg, use_bf16=True)
        bf16_64 = time.time() - t0
        results["bf16_b64"] = {"seconds": round(bf16_64, 2),
                               "s_per_image": round(bf16_64 / 64, 3)}

    # torch.compile on the denoiser only (the 2,000x-repeated call).
    try:
        net.denoise_fn = torch.compile(net.denoise_fn, mode="reduce-overhead")
        sample(net, x32, cfg, use_bf16=True)  # warm-up compiles
        t0 = time.time()
        sample(net, x32, cfg, use_bf16=True)
        comp_s = time.time() - t0
        results[f"bf16_compile_b{bs}"] = {"seconds": round(comp_s, 2),
                                          "s_per_image": round(comp_s / bs, 3),
                                          "speedup_vs_fp32": round(fp32_s / comp_s, 2)}
    except Exception as exc:  # noqa: BLE001
        results[f"bf16_compile_b{bs}"] = {"error": repr(exc)[:300]}

    results["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    best = min(
        (v["s_per_image"], k) for k, v in results.items()
        if isinstance(v, dict) and "s_per_image" in v
    )
    results["best"] = {"variant": best[1], "s_per_image": best[0],
                       "projected_gpu_hours_8993": round(best[0] * 8993 / 3600, 2)}
    save_json(results, artifact_dir("logs") / f"dmahr_bench_{args.tag}.json")
    record_run("dmahr_bench", gpu=results["device"], seconds=fp32_s + bf16_s + bf16_64)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

