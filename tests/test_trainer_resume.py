"""CPU check of the resume and snapshot path used by the 20k -> 100k continuation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dm_ahr_vit.config import load_config
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


def _pairs(tmp: Path, n: int, split: str, offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(offset)
    rows = []
    for i in range(n):
        ref, hairy = tmp / f"ref_{split}{i}.png", tmp / f"hairy_{split}{i}.png"
        Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(ref)
        Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(hairy)
        rows.append({"pair_id": f"{split}{i}", "reference_path": str(ref),
                     "hairy_path": str(hairy), "split": split,
                     "group_id": f"g_{split}{i}"})
    return pd.DataFrame(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resume_continues_iteration_history_and_keeps_snapshots(tmp_path):
    cfg = _cfg()
    tr, va = _pairs(tmp_path, 4, "train", 1), _pairs(tmp_path, 2, "val", 2)
    out = tmp_path / "ckpt"

    train(cfg, tr, va, out, device="cpu", max_iterations=6)
    state1 = torch.load(out / "train_state.pt", map_location="cpu", weights_only=False)
    assert state1["iteration"] == 6
    assert (out / "gen_iter000003.pth").exists() and (out / "gen_iter000006.pth").exists()
    first_snapshot = _sha(out / "gen_iter000003.pth")
    saved_best_at_6 = state1["best_val"]

    train(cfg, tr, va, out, device="cpu", max_iterations=12)
    state2 = torch.load(out / "train_state.pt", map_location="cpu", weights_only=False)

    assert state2["iteration"] == 12
    assert [h["iteration"] for h in state2["history"]] == [3, 6, 9, 12]
    for it in (9, 12):
        assert (out / f"gen_iter{it:06d}.pth").exists()
    # Earlier snapshots must survive a continuation untouched.
    assert _sha(out / "gen_iter000003.pth") == first_snapshot
    # The best validation loss carried over and can only improve or stay.
    assert state2["best_val"] <= saved_best_at_6 or not np.isfinite(saved_best_at_6)
