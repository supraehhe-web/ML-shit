"""Find the throughput-optimal sampling batch size.

The sampler is 2,000 sequential UNet calls at 128x128. At batch 32 it used only
1.5 GB of an 80 GB A100, so the GPU is heavily underutilised and throughput per
image should keep improving with batch size until memory or compute saturates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


@torch.no_grad()
def run(net, x, use_bf16: bool, seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        traj = net.super_resolution(x, continous=True)
    return traj[-x.shape[0]:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="32,64,128,256")
    ap.add_argument("--tag", default="a100")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    device = "cuda"
    ckpt = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
    ids = screen[(screen["split"] == "test") & (screen["hair_present"])]["image_id"].head(256).tolist()
    images = [Image.open(isic_images_dir() / f"{i}.jpg").convert("RGB") for i in ids]
    net = load_model(cfg, ckpt, device=device)

    results = {"tag": args.tag, "device": torch.cuda.get_device_name(0),
               "available_images": len(images), "sweep": {}}
    for bs in [int(b) for b in args.batches.split(",")]:
        if bs > len(images):
            continue
        x = preprocess(images[:bs], 128).to(device)
        for use_bf16 in (False, True):
            key = f"b{bs}_{'bf16' if use_bf16 else 'fp32'}"
            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                run(net, x, use_bf16)
                secs = time.time() - t0
                results["sweep"][key] = {
                    "seconds": round(secs, 2),
                    "s_per_image": round(secs / bs, 3),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                log.info("%s: %.2fs total, %.3f s/img, %.1f GB", key, secs,
                         secs / bs, results["sweep"][key]["peak_mem_gb"])
            except torch.cuda.OutOfMemoryError:
                results["sweep"][key] = {"error": "OOM"}
                log.warning("%s: OOM", key)
                torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()

    valid = {k: v for k, v in results["sweep"].items() if "s_per_image" in v}
    if valid:
        best = min(valid.items(), key=lambda kv: kv[1]["s_per_image"])
        results["best"] = {
            "variant": best[0],
            "s_per_image": best[1]["s_per_image"],
            "gpu_hours_8993": round(best[1]["s_per_image"] * 8993 / 3600, 2),
        }
    save_json(results, artifact_dir("logs") / f"batch_sweep_{args.tag}.json")
    record_run("batch_sweep", gpu=results["device"], seconds=0.0, tag=args.tag)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

