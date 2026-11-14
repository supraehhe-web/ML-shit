"""Evaluate one frozen ViT checkpoint on the locked test split (once)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.data.dataset import ArmDataset, build_transforms, collate  # noqa: E402
from dm_ahr_vit.paths import (  # noqa: E402
    artifact_dir,
    data_root,
    isic_images_dir,
    manifest_dir,
)
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402
from dm_ahr_vit.vit.metrics import classification_metrics  # noqa: E402
from dm_ahr_vit.vit.model import build_model, load_checkpoint  # noqa: E402

log = setup_logging()


def predict(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            for i, image_id in enumerate(batch["image_id"]):
                row = {
                    "image_id": image_id,
                    "group_id": batch["group_id"][i],
                    "label": batch["dx"][i],
                    "pred": CLASSES[int(pred[i])],
                    "prob_max": float(prob[i].max()),
                }
                for j, c in enumerate(CLASSES):
                    row[f"prob_{c}"] = float(prob[i, j])
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "Astar", "Creleased", "Cv2"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out-dir", default=None,
                    help="override the predictions output directory (reproducibility checks)")
    args = ap.parse_args()

    cfg = load_config("configs/vit.json")
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    splits_hash = sha256_file(manifest_dir() / "splits.csv")
    screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
    hairy_ids = set(screen[screen["hair_present"]]["image_id"])

    run_dir = artifact_dir("checkpoints", "vit", f"{args.arm}_seed{args.seed}")
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=True).to(device)
    ckpt = load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    tfm = build_transforms(cfg, train=False)
    out_dir = Path(args.out_dir) if args.out_dir else artifact_dir("predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_key: dict[tuple, dict] = {}
    for subset in ("full", "hair_present"):
        ids = None if subset == "full" else hairy_ids
        ds = ArmDataset(args.arm, "test", splits, isic_images_dir(),
                        data_root() / "processed", tfm, image_ids=ids)
        if len(ds) == 0:
            log.warning("subset %s empty for %s/%d", subset, args.arm, args.seed)
            continue
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                            num_workers=int(cfg["num_workers"]), collate_fn=collate)
        df = predict(model, loader, device)
        df.insert(0, "subset", subset)
        df.insert(0, "seed", args.seed)
        df.insert(0, "arm", args.arm)
        df.to_csv(out_dir / f"pred_{args.arm}_seed{args.seed}_{subset}.csv", index=False)
        y_true = np.array([CLASSES.index(v) for v in df["label"]])
        y_pred = np.array([CLASSES.index(v) for v in df["pred"]])
        y_prob = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
        metrics_by_key[(args.arm, args.seed, subset)] = classification_metrics(
            y_true, y_pred, y_prob
        )
        log.info("%s/%d %s: macro-F1 %.4f acc %.4f n %d", args.arm, args.seed, subset,
                 metrics_by_key[(args.arm, args.seed, subset)]["macro_f1"],
                 metrics_by_key[(args.arm, args.seed, subset)]["accuracy"], len(df))

    payload = {
        "arm": args.arm,
        "seed": args.seed,
        "splits_sha256": splits_hash,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "best_val_macro_f1": float(ckpt.get("best_val_macro_f1", float("nan"))),
        "metrics": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in metrics_by_key.items()},
    }
    save_json(payload, out_dir / f"metrics_{args.arm}_seed{args.seed}.json")
    record_run("vit_eval", gpu=device, seconds=0.0, arm=args.arm, seed=args.seed,
               n_test=sum(m["n"] for m in metrics_by_key.values()))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_class"}
                      for k, v in payload["metrics"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

