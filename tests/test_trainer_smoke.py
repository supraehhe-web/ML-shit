"""The DM-AHR trainer runs end to end on CPU with a tiny network and real local pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from dm_ahr_vit.config import load_config
from dm_ahr_vit.diffusion.train import PairedDataset, train

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "processed" / "synth" / "test"


def _pairs(n: int) -> pd.DataFrame:
    hairy = sorted((SYNTH / "hairy").glob("*.png"))[:n]
    if len(hairy) < n:
        pytest.skip("local synthetic pairs not available")
    return pd.DataFrame({
        "pair_id": [p.stem for p in hairy], "split": "train",
        "group_id": [f"G{i}" for i in range(n)],
        "hairy_path": [str(p) for p in hairy],
        "reference_path": [str(SYNTH / "reference" / p.name) for p in hairy],
    })


def _tiny_cfg() -> dict:
    cfg = load_config("configs/dmahr.json")
    cfg["num_timesteps"] = 20
    cfg["unet"].update({"inner_channel": 32, "channel_multiplier": [1, 2], "res_blocks": 1,
                        "attn_res": []})
    cfg["train"].update({"batch_size": 2, "checkpoint_every": 3, "val_every": 3, "iterations": 3})
    return cfg


def test_dataset_is_cached_and_scaled():
    ds = PairedDataset(_pairs(3), 128)
    a, b = ds[0], ds[0]
    assert a["HR"].shape == (3, 128, 128) and torch.equal(a["HR"], b["HR"])
    assert -1.0 <= float(a["HR"].min()) and float(a["HR"].max()) <= 1.0
    assert not torch.equal(ds[1]["HR"], ds[1]["HR"].flip(-1))  # odd indices are flipped copies


@pytest.mark.parametrize("mode", ["upstream", "legacy"])
def test_trainer_runs_resumes_and_traces(tmp_path, mode):
    from dm_ahr_vit.diffusion.dmahr_adapter import _ensure_path

    _ensure_path()
    pytest.importorskip("model.networks", reason="upstream not importable")
    cfg = _tiny_cfg()
    pairs = _pairs(4)
    out = train(cfg, pairs, pairs, tmp_path, device="cpu", conditioning=mode, iterations=3,
                save_every=3, seed=0)
    assert out["iterations"] == 3 and out["conditioning"] == mode
    log = json.loads((tmp_path / "train_log.json").read_text())
    assert len(log["trace"]) == 3 and all(torch.isfinite(torch.tensor(t["loss"])) for t in log["trace"])
    assert (tmp_path / "iter_3_gen.pth").exists() and (tmp_path / "final_gen.pth").exists()
    manifest = json.loads((tmp_path / "training_manifest.json").read_text())
    assert manifest["conditioning"] == mode and manifest["leakage_check"] is False  # same groups on purpose
    other = "legacy" if mode == "upstream" else "upstream"
    with pytest.raises(RuntimeError):
        train(cfg, pairs, pairs, tmp_path, device="cpu", conditioning=other, iterations=6)
