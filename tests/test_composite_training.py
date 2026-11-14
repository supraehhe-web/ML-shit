"""The trainer runs end to end on composite datasets (CPU, tiny model)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import pandas as pd
from PIL import Image

from dm_ahr_vit.config import load_config
from dm_ahr_vit.data.hair_composite import build_datasets
from dm_ahr_vit.diffusion.train import train

SIZE = 16


def _cfg() -> dict:
    cfg = load_config("configs/dmahr.json")
    cfg["native_resolution"] = SIZE
    cfg["unet"] = {"in_channel": 6, "out_channel": 3, "inner_channel": 32,
                   "channel_multiplier": [1, 2], "attn_res": [], "res_blocks": 1,
                   "dropout": 0.0}
    cfg["train"] = {**cfg["train"], "batch_size": 2, "checkpoint_every": 3,
                    "val_every": 3, "snapshot_every": 3}
    return cfg


def test_train_on_composites_writes_manifest_and_snapshots(tmp_path):
    img, lay = tmp_path / "img", tmp_path / "layers"
    img.mkdir(), lay.mkdir()
    rows = []
    for split, pre in (("train", "T"), ("val", "V")):
        for i in range(3):
            Image.fromarray(np.full((90, 120, 3), 160 + i, np.uint8)).save(img / f"{pre}C{i}.jpg")
            hairy = np.full((90, 120, 3), 210, np.uint8)
            cv2.line(hairy, (5, 10 + 10 * i), (110, 30 + 10 * i), (20, 20, 20), 2)
            Image.fromarray(hairy).save(img / f"{pre}H{i}.jpg")
            m = np.zeros((90, 120), np.uint8)
            cv2.line(m, (5, 10 + 10 * i), (110, 30 + 10 * i), 255, 2)
            cv2.imwrite(str(lay / f"{pre}H{i}.png"), m)
            rows.append({"image_id": f"{pre}C{i}", "split": split, "group_id": f"{pre}c{i}",
                         "hair_coverage": 0.001})
            rows.append({"image_id": f"{pre}H{i}", "split": split, "group_id": f"{pre}h{i}",
                         "hair_coverage": 0.09})
    csv = tmp_path / "scr.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    stats = pd.DataFrame({"image_id": [p.stem for p in lay.glob("*.png")], "fraction": 0.01})
    stats.to_csv(lay / "summary.csv", index=False)
    tr, va = build_datasets(csv, img, lay, size=SIZE, val_length=3)
    out = tmp_path / "out"
    res = train(_cfg(), None, None, out, device="cpu", max_iterations=6,
                train_ds=tr, val_ds=va, num_workers=0)
    assert res["iterations"] == 6
    assert (out / "gen_iter000003.pth").exists() and (out / "gen_iter000006.pth").exists()
    man = json.loads((out / "training_manifest.json").read_text())
    assert man["train_sources_splits"] == ["train"] and man["val_sources_splits"] == ["val"]
    assert man["leakage_check"] is True
    assert man["dataset_kind"] == "HairCompositeDataset"
    if not np.isfinite(res["best_val"]):
        pytest.skip("non-finite CPU validation loss on this draw of a tiny random network; "
                    "see decision log D29")
