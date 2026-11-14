"""Train a leakage-controlled DM-AHR checkpoint on train-split data only.

Default: the original path, simple synthetic-hair pairs from the pair manifest.
`--composite`: pairs made on the fly by pasting REAL hair layers (harvested from
train-split images) onto clean train-split images; validation uses validation-split
images only. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.data.hair_composite import build_datasets  # noqa: E402
from dm_ahr_vit.diffusion.train import train  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--warm-start", default=None,
                    help="checkpoint to initialise from, for sanity checks")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override the configured iteration count")
    ap.add_argument("--out-name", default=None,
                    help="checkpoint subdirectory; set this for sanity runs so the "
                         "recorded Arm C checkpoint is never overwritten")
    ap.add_argument("--composite", action="store_true",
                    help="train on real-hair composites instead of the pair manifest")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient-norm clip; 0 removes clipping (upstream does not clip)")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    log.info("cpu_count=%s workers=%d composite=%s", os.cpu_count(), args.num_workers,
             args.composite)
    default_name = "dmahr_overfit" if args.overfit else "dmahr"
    out_dir = artifact_dir("checkpoints", args.out_name or default_name)
    if args.composite:
        train_ds, val_ds = build_datasets(
            manifest_dir() / "hair_screening.csv", isic_images_dir(),
            data_root() / "hair_layers", size=int(cfg["native_resolution"]))
        log.info("composite: %d train targets, %d train sources, %d val samples",
                 len(train_ds.targets), len(train_ds.sources), len(val_ds))
        result = train(cfg, None, None, out_dir, init_from=(
            Path(args.warm_start) if args.warm_start else None),
            max_iterations=args.iterations, train_ds=train_ds, val_ds=val_ds,
            num_workers=args.num_workers, grad_clip=args.grad_clip)
        tag = "dmahr_train_composite"
    else:
        pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
        train_pairs = pairs[pairs["split"] == "train"]
        val_pairs = pairs[pairs["split"] == "val"]
        if train_pairs.empty or val_pairs.empty:
            raise RuntimeError("synthetic pairs missing; run 02_hair_pipeline.py first")
        if set(train_pairs["group_id"]) & set(val_pairs["group_id"]):
            raise RuntimeError("group leakage between diffusion train and val pairs")
        result = train(cfg, train_pairs, val_pairs, out_dir, overfit=args.overfit,
                       init_from=Path(args.warm_start) if args.warm_start else None,
                       max_iterations=args.iterations, grad_clip=args.grad_clip)
        tag = "dmahr_train_overfit" if args.overfit else "dmahr_train"
    if args.out_name is None or args.composite:
        log_name = f"{args.out_name or 'dmahr'}_train.json" if args.composite else (
            "dmahr_overfit.json" if args.overfit else "dmahr_train.json")
        save_json(result, artifact_dir("logs") / log_name)
    record_run(tag, gpu="A100-80GB", seconds=result["seconds"], iterations=result["iterations"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
