"""Apply the trained DM-AHR checkpoint to hair-present images (sharded, resumable)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import clean_batch, load_model  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, isic_images_dir, manifest_dir, processed_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--pairs-split", default=None,
                    help="process synthetic pairs for this split instead of real hair-present images")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 autocast; ~2x faster, verified against fp32 in the smoke test")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint) if args.checkpoint else artifact_dir("checkpoints", "dmahr") / "best_gen.pth"
    if not ckpt.exists():
        ckpt = artifact_dir("checkpoints", "dmahr") / "final_gen.pth"
    if not ckpt.exists():
        raise FileNotFoundError("no trained DM-AHR checkpoint found")

    if args.pairs_split:
        pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
        pairs = pairs[pairs["split"] == args.pairs_split].sort_values("image_id")
        pairs = pairs[pairs.index % args.n_shards == args.shard] if False else \
            pairs.iloc[args.shard::args.n_shards]
        todo = pd.DataFrame({
            "image_id": pairs["image_id"].values,
            "source_path": pairs["hairy_path"].values,
        })
        out_dir = processed_dir("synth") / args.pairs_split / "cleaned_C"
    else:
        screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
        todo = screen[screen["hair_present"]].sort_values("image_id").reset_index(drop=True)
        todo = todo.iloc[args.shard::args.n_shards]
        todo = pd.DataFrame({
            "image_id": todo["image_id"].values,
            "source_path": [str(isic_images_dir() / f"{i}.jpg") for i in todo["image_id"]],
        })
        out_dir = processed_dir("C")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = todo[~todo["image_id"].map(lambda i: (out_dir / f"{i}.png").exists())]
    log.info("shard %d/%d: %d images to process", args.shard, args.n_shards, len(todo))

    net = load_model(cfg, ckpt, device=device)
    bs = int(cfg["sampling"]["batch_size"])
    rows = []
    t0 = time.time()
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start:start + bs]
        images = [Image.open(p).convert("RGB") for p in chunk["source_path"]]
        seed = int(hashlib.sha256(f"{args.shard}:{start}".encode()).hexdigest()[:8], 16)
        outs, stats = clean_batch(net, images, cfg, device=device, seed=seed,
                                  num_timesteps=args.num_timesteps,
                                  use_bf16=args.bf16)
        for image_id, out in zip(chunk["image_id"], outs):
            path = out_dir / f"{image_id}.png"
            tmp = path.with_suffix(".png.tmp")
            out.save(tmp, format="PNG")
            tmp.replace(path)
            rows.append({"image_id": image_id, "path": str(path), "seed": seed,
                         "num_timesteps": stats["num_timesteps"],
                         "bf16": stats["bf16"],
                         "finite": stats["finite"], "width": out.width, "height": out.height})
        if (start // bs) % 20 == 0:
            log.info("shard %d: %d/%d (%.0fs)", args.shard, start + len(chunk), len(todo),
                     time.time() - t0)
    df = pd.DataFrame(rows)
    tag = f"pairs_{args.pairs_split}" if args.pairs_split else "real"
    out_csv = artifact_dir("logs") / f"dmahr_sample_{tag}_shard{args.shard}.csv"
    df.to_csv(out_csv, index=False)
    summary = {
        "shard": args.shard,
        "n_shards": args.n_shards,
        "source": tag,
        "processed": int(len(df)),
        "seconds": round(time.time() - t0, 1),
        "seconds_per_image": round((time.time() - t0) / max(len(df), 1), 2),
        "checkpoint": str(ckpt),
        "num_timesteps": args.num_timesteps or cfg["num_timesteps"],
    }
    save_json(summary, artifact_dir("logs") / f"dmahr_sample_{tag}_shard{args.shard}.json")
    record_run("dmahr_sample", gpu=device, seconds=summary["seconds"],
               shard=args.shard, images=len(df))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

