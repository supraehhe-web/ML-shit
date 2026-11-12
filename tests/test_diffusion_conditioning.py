"""The trainer must condition the denoiser exactly as upstream sampling does."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dm_ahr_vit.diffusion.conditioning import (
    denoiser_level,
    fixed_level_error,
    noise_prediction_loss,
)

SCHEDULE = {"schedule": "linear", "linear_start": 1e-6, "linear_end": 0.01, "n_timestep": 2000}


class _Recorder(torch.nn.Module):
    """Stands in for the UNet and records the noise level it is conditioned on."""

    def __init__(self):
        super().__init__()
        self.levels: list[torch.Tensor] = []

    def forward(self, x, level):
        self.levels.append(level.detach().clone())
        return torch.zeros_like(x[:, 3:])


@pytest.fixture(scope="module")
def upstream_cls():
    from dm_ahr_vit.diffusion.dmahr_adapter import _ensure_path

    _ensure_path()
    try:
        from model.sr3_modules.diffusion import GaussianDiffusion
    except Exception as exc:  # pragma: no cover - environment without upstream deps
        pytest.skip(f"upstream diffusion module not importable: {exc!r}")
    return GaussianDiffusion


@pytest.fixture()
def net(upstream_cls):
    rec = _Recorder()
    n = upstream_cls(rec, image_size=8, channels=3, conditional=True)
    n.set_new_noise_schedule(SCHEDULE, device=torch.device("cpu"))
    n.set_loss(torch.device("cpu"))
    return n


def _batch(b=2):
    return torch.zeros(b, 3, 8, 8), torch.zeros(b, 3, 8, 8)


def test_upstream_mode_matches_upstream_training_levels(net):
    hr, cond = _batch()
    prev = np.asarray(net.sqrt_alphas_cumprod_prev)
    for t in (1, 500, 1900):
        net.denoise_fn.levels.clear()
        noise_prediction_loss(net, hr, cond, mode="upstream", t=t)
        ours = net.denoise_fn.levels[0]
        assert ours.shape == (2, 1)
        assert float(ours.min()) >= min(prev[t - 1], prev[t]) - 1e-6
        assert float(ours.max()) <= max(prev[t - 1], prev[t]) + 1e-6


def test_upstream_training_call_uses_the_same_level_family(net):
    hr, cond = _batch()
    net.denoise_fn.levels.clear()
    np.random.seed(0)
    net({"HR": hr, "SR": cond})  # upstream's own p_losses
    theirs = net.denoise_fn.levels[0]
    assert theirs.shape == (2, 1)
    assert 0.0 < float(theirs.min()) and float(theirs.max()) <= 1.0


def test_sampling_conditions_on_the_same_quantity(net):
    hr, cond = _batch()
    prev = np.asarray(net.sqrt_alphas_cumprod_prev)
    net.denoise_fn.levels.clear()
    net.p_mean_variance(hr, t=1500, clip_denoised=True, condition_x=cond)
    sampled = float(net.denoise_fn.levels[0].flatten()[0])
    assert abs(sampled - prev[1501]) < 1e-6
    net.denoise_fn.levels.clear()
    noise_prediction_loss(net, hr, cond, mode="upstream", t=1501)
    trained = float(net.denoise_fn.levels[0].mean())
    assert abs(trained - prev[1501]) < 0.01


def test_legacy_mode_reproduces_the_bug_and_is_far_from_upstream(net):
    hr, cond = _batch()
    prev = np.asarray(net.sqrt_alphas_cumprod_prev)
    t = 1900
    net.denoise_fn.levels.clear()
    noise_prediction_loss(net, hr, cond, mode="legacy", t=t)
    legacy = float(net.denoise_fn.levels[0].mean())
    assert abs(legacy - t / 2000) < 1e-6
    # At a nearly fully noised step the correct level is ~0 but legacy feeds ~0.95.
    assert prev[t] < 0.05 and legacy > 0.9


def test_fixed_level_error_differs_only_in_the_conditioning_value(net):
    hr, cond = _batch()
    noise = torch.randn_like(hr)
    net.denoise_fn.levels.clear()
    fixed_level_error(net, hr, cond, 800, noise, "upstream")
    fixed_level_error(net, hr, cond, 800, noise, "legacy")
    up, lg = net.denoise_fn.levels
    assert up.shape == (2, 1) and lg.shape == (2,)
    assert abs(float(lg[0]) - 800 / 2000) < 1e-6


def test_bad_mode_is_rejected(net):
    with pytest.raises(ValueError):
        denoiser_level(net, 5, torch.zeros(2), "nope")
