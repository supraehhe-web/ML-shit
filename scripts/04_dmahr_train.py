"""Train a leakage-controlled DM-AHR checkpoint on train-split pairs only.

The default is the faithful (upstream-conditioned) trainer writing to
`checkpoints/dmahr_fixed`. `--conditioning legacy` reproduces the first trainer
and is only for the diagnosis (it never overwrites `checkpoints/dmahr`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.budget import gate  # noqa: E402
from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.train import train  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()
SECONDS_PER_ITER = 0.055  # measured, reports/compute_plan.md (A100-80GB, batch 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--conditioning", choices=["upstream", "legacy"], default="upstream")
    ap.add_argument("--tag", default=None, help="output directory name under checkpoints/")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start from these weights (released-checkpoint warm-start test)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="upstream does not clip; the legacy trainer clipped at 1.0")
    ap.add_argument("--n-pairs", type=int, default=None, help="overfit: number of pairs")
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--budget-usd", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    if args.n_pairs is not None:
        cfg["train"]["overfit_pairs"] = args.n_pairs
    iterations = args.iterations or int(cfg["train"]["iterations"])
    gate("dmahr_train", iterations, SECONDS_PER_ITER, args.gpu, args.budget_usd,
         fixed_seconds=60.0)

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    train_pairs = pairs[pairs["split"] == "train"]
    val_pairs = pairs[pairs["split"] == "val"]
    if train_pairs.empty or val_pairs.empty:
        raise RuntimeError("synthetic pairs missing; run 02_hair_pipeline.py first")
    if set(train_pairs["group_id"]) & set(val_pairs["group_id"]):
        raise RuntimeError("group leakage between diffusion train and val pairs")

    tag = args.tag or ("dmahr_overfit" if args.overfit else
                       ("dmahr_fixed" if args.conditioning == "upstream" else "dmahr_legacy_repro"))
    out_dir = artifact_dir("checkpoints", tag)
    init = Path(args.init_checkpoint) if args.init_checkpoint else None
    result = train(cfg, train_pairs, val_pairs, out_dir, overfit=args.overfit,
                   conditioning=args.conditioning, iterations=iterations,
                   save_every=args.save_every, init_checkpoint=init,
                   grad_clip=args.grad_clip)
    save_json(result, artifact_dir("logs") / f"{tag}_train.json")
    record_run("dmahr_train", gpu=args.gpu, seconds=result["seconds"],
               iterations=result["iterations"], tag=tag, conditioning=args.conditioning)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
