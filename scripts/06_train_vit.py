"""Train one ViT-B/16 arm/seed with the matched recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402
from dm_ahr_vit.vit.train import train_arm  # noqa: E402

log = setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "Astar", "Creleased", "Cv2"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--run-suffix", default=None,
                    help="write to <arm>_seed<seed><suffix> instead of resuming the default dir")
    args = ap.parse_args()

    cfg = load_config("configs/vit.json")
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{args.arm}_seed{args.seed}{args.run_suffix or ''}"
    run_dir = artifact_dir("checkpoints", "vit", name)
    result = train_arm(
        arm=args.arm,
        seed=args.seed,
        cfg=cfg,
        splits_df=splits,
        raw_images_dir=isic_images_dir(),
        processed_root=data_root() / "processed",
        run_dir=run_dir,
        device=device,
        max_epochs=args.max_epochs,
    )
    record_run("vit_train", gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               seconds=result["seconds"], arm=args.arm, seed=args.seed,
               best_val_f1=result["best_val_macro_f1"], epochs=result["epochs_run"])
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

