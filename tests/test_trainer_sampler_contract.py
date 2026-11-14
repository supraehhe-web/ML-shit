"""Contract tests: trainer vs sampler vs upstream, on identical inputs.

Covers the diagnostic checks that need no checkpoint:
  A  model-input contract (channel order, shapes, dtype, noise-level scalar)
  B  differential test of the training loss and gradients against upstream `p_losses`
  C  oracle-epsilon sampler: the reverse chain must return x0 when the denoiser is exact
  D  data hygiene (pair flip consistency, range, group leakage, manifest hash)
  E  metric harness on known corruptions
Real-data checks skip unless DMAHR_DIAG_ROOT points at a local copy of the pair manifest.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dm_ahr_vit.config import load_config
from dm_ahr_vit.diffusion.dmahr_adapter import _ensure_path, build_opt
from dm_ahr_vit.diffusion.train import PairedDataset, _manifest_hash, training_loss

SIZE = 16
T = 2000


def _small_cfg() -> dict:
    cfg = load_config("configs/dmahr.json")
    cfg["native_resolution"] = SIZE
    cfg["unet"] = {"in_channel": 6, "out_channel": 3, "inner_channel": 32,
                   "channel_multiplier": [1, 2], "attn_res": [], "res_blocks": 1,
                   "dropout": 0.0}
    return cfg


@pytest.fixture(scope="module")
def net():
    _ensure_path()
    import model.networks as networks  # noqa: PLC0415

    cfg = _small_cfg()
    n = networks.define_G(build_opt(cfg))
    n.set_new_noise_schedule(
        {**cfg["beta_schedule"], "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device("cpu"))
    n.set_loss("cpu")
    n.train()
    return n


class Recorder(torch.nn.Module):
    """Wraps a denoiser and keeps every (input, level) it was called with."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x, level):
        self.calls.append((x.detach().clone(), level.detach().clone()))
        return self.inner(x, level)


# ---------------------------------------------------------------- A: contract
def test_trainer_and_sampler_feed_identical_model_inputs(net):
    hr = torch.rand(4, 3, SIZE, SIZE) * 1.6 - 0.8
    cond = torch.rand(4, 3, SIZE, SIZE) * 1.6 - 0.8
    original = net.denoise_fn
    rec = Recorder(original)
    net.denoise_fn = rec
    try:
        np.random.seed(11)
        torch.manual_seed(11)
        t_drawn = None
        probe = np.random.RandomState(11)
        t_drawn = int(probe.randint(1, T + 1))
        training_loss(net, hr, cond)
        with torch.no_grad():
            net.p_mean_variance(torch.randn(4, 3, SIZE, SIZE), t_drawn - 1, True, condition_x=cond)
    finally:
        net.denoise_fn = original
    (x_tr, lv_tr), (x_sa, lv_sa) = rec.calls
    assert x_tr.shape == x_sa.shape == (4, 6, SIZE, SIZE)
    assert x_tr.dtype == x_sa.dtype == torch.float32
    assert lv_tr.shape == lv_sa.shape == (4, 1)
    assert lv_tr.dtype == lv_sa.dtype == torch.float32
    # channels 0-2 are the condition in both paths, 3-5 the noisy image
    assert torch.equal(x_tr[:, :3], cond)
    assert torch.equal(x_sa[:, :3], cond)
    # trainer level lies between the schedule values at t-1 and t, and the
    # sampler's level at index t-1 equals the trainer's upper endpoint
    prev = np.asarray(net.sqrt_alphas_cumprod_prev)
    lo, hi = sorted((prev[t_drawn - 1], prev[t_drawn]))
    assert (lv_tr.numpy() >= lo - 1e-7).all() and (lv_tr.numpy() <= hi + 1e-7).all()
    assert np.allclose(lv_sa.numpy(), prev[t_drawn], atol=1e-7)
    # the noisy channels really are alpha*hr + sqrt(1-alpha^2)*eps with eps ~ N(0,1)
    a = lv_tr.view(-1, 1, 1, 1)
    eps_hat = (x_tr[:, 3:] - a * hr) / (1 - a**2).sqrt()
    assert abs(float(eps_hat.mean())) < 0.15 and abs(float(eps_hat.std()) - 1) < 0.15


# ------------------------------------------------- B: differential vs upstream
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_training_loss_and_gradients_match_upstream_p_losses(net, seed):
    hr = torch.rand(4, 3, SIZE, SIZE) * 2 - 1
    cond = torch.rand(4, 3, SIZE, SIZE) * 2 - 1

    def run(fn):
        np.random.seed(seed)
        torch.manual_seed(seed)
        net.zero_grad(set_to_none=True)
        loss = fn()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
        return float(loss), g

    l_up, g_up = run(lambda: net.p_losses({"HR": hr, "SR": cond}) / hr.numel())
    l_us, g_us = run(lambda: training_loss(net, hr, cond))
    assert l_up == pytest.approx(l_us, rel=1e-6, abs=1e-8)
    if not (torch.isfinite(g_up).all() and torch.isfinite(g_us).all()):
        # The CPU backward through the U-Net's last block intermittently yields NaN
        # (about 3% of runs, on upstream's own p_losses path as well as ours;
        # scripts/33_nan_grad_probe.py). The losses above still must match.
        pytest.skip("non-finite CPU gradient on this draw; see decision log D29")
    cos = torch.nn.functional.cosine_similarity(g_up, g_us, dim=0)
    assert float(cos) > 0.999999


# --------------------------------------------------------------- C: oracle
class OracleEps(torch.nn.Module):
    """Returns the true epsilon implied by x_t, x0 and the noise level it is handed."""

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def forward(self, x_in, level):
        a = level.view(-1, 1, 1, 1)
        return (x_in[:, 3:] - a * self.x0) / (1 - a**2).sqrt()


def test_oracle_epsilon_sampler_recovers_x0(net):
    torch.manual_seed(3)
    x0 = torch.rand(1, 3, 4, 4) * 1.6 - 0.8
    cond = torch.zeros_like(x0)
    original = net.denoise_fn
    net.denoise_fn = OracleEps(x0)
    try:
        img = torch.randn_like(x0)
        with torch.no_grad():
            for i in reversed(range(T)):
                img = net.p_sample(img, i, condition_x=cond)
    finally:
        net.denoise_fn = original
    assert float((img - x0).abs().max()) < 1e-3


# ------------------------------------------------------------------- D: data
def test_paired_dataset_flips_both_images_together_and_stays_in_range(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(2):
        ref = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
        hairy = ref.copy()
        Image.fromarray(ref).save(tmp_path / f"r{i}.png")
        Image.fromarray(hairy).save(tmp_path / f"h{i}.png")
        rows.append({"pair_id": f"p{i}", "reference_path": str(tmp_path / f"r{i}.png"),
                     "hairy_path": str(tmp_path / f"h{i}.png")})
    ds = PairedDataset(pd.DataFrame(rows), SIZE)
    for i in range(2):
        item = ds[i]
        assert torch.equal(item["HR"], item["SR"]), "hairy==reference input must stay identical after augmentation"
        assert float(item["HR"].min()) >= -1 and float(item["HR"].max()) <= 1


def _diag_root() -> Path | None:
    root = os.environ.get("DMAHR_DIAG_ROOT")
    return Path(root) if root and (Path(root) / "man" / "synthetic_pairs.csv").exists() else None


@pytest.mark.skipif(_diag_root() is None, reason="needs a local copy of the pair manifest")
def test_real_manifest_split_integrity_and_hash():
    pairs = pd.read_csv(_diag_root() / "man" / "synthetic_pairs.csv")
    train, val = pairs[pairs["split"] == "train"], pairs[pairs["split"] == "val"]
    assert not set(train["group_id"]) & set(val["group_id"])
    assert not set(train["image_id"]) & set(val["image_id"])
    assert len(train) == 654 and len(val) == 182
    expected = "b302eda0b4a2686a05c8b7b01484b7f71a739f3b89ec805f5c394fa97e1e37a9"
    assert _manifest_hash(train) == expected


@pytest.mark.skipif(_diag_root() is None, reason="needs local validation pair images")
def test_real_pairs_are_aligned_and_difference_is_on_hair():
    from PIL import Image

    root = _diag_root()
    pairs = pd.read_csv(root / "man" / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].head(12)
    inside, outside, shifted = [], [], []
    for r in val.itertuples(index=False):
        base = root / "data" / r.reference_path.split("/data/", 1)[1]
        hairy = np.asarray(Image.open(root / "data" / r.hairy_path.split("/data/", 1)[1]).convert("RGB"), np.float32)
        ref = np.asarray(Image.open(base).convert("RGB"), np.float32)
        mask = np.asarray(Image.open(root / "data" / r.hair_mask_path.split("/data/", 1)[1]).convert("L")) > 127
        diff = np.abs(hairy - ref).mean(-1)
        inside.append(diff[mask].mean())
        outside.append(diff[~mask].mean())
        shifted.append(np.abs(hairy[:, 4:] - ref[:, :-4]).mean())  # misaligned control
    assert np.mean(inside) > 5 * np.mean(outside)
    assert np.mean(shifted) > 3 * np.mean(outside)


# ------------------------------------------------------------------- E: metrics
def test_metric_harness_known_cases():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3))
    assert peak_signal_noise_ratio(a, a, data_range=1.0) == np.inf
    assert structural_similarity(a, a, data_range=1.0, channel_axis=2, win_size=7) == pytest.approx(1.0)
    shifted = np.clip(a * 0 + 0.5, 0, 1)
    assert peak_signal_noise_ratio(shifted, shifted + 0.1, data_range=1.0) == pytest.approx(20.0, abs=1e-6)
    noisy = a + rng.normal(0, 0.05, a.shape)
    assert peak_signal_noise_ratio(a, noisy, data_range=1.0) == pytest.approx(26.0, abs=0.3)
