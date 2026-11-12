"""Respaced sampling visits the trained noise levels; re-derived short schedules do not."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dm_ahr_vit.config import load_config
from dm_ahr_vit.diffusion.dmahr_adapter import _ensure_path, respaced_betas, set_schedule


@pytest.fixture(scope="module")
def parts():
    _ensure_path()
    try:
        import model.sr3_modules.diffusion as dmod
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"upstream not importable: {exc!r}")
    cfg = load_config("configs/dmahr.json")
    bs = cfg["beta_schedule"]
    betas = np.asarray(dmod.make_beta_schedule(schedule=bs["schedule"], n_timestep=2000,
                                               linear_start=bs["linear_start"],
                                               linear_end=bs["linear_end"]), dtype=np.float64)
    return cfg, betas, dmod


def _net(dmod, cfg):
    n = dmod.GaussianDiffusion(torch.nn.Identity(), image_size=8, conditional=True)
    set_schedule(n, cfg, 2000, "cpu", "rederive")
    return n


def test_full_respacing_is_the_identity(parts):
    cfg, betas, _ = parts
    assert np.allclose(respaced_betas(cfg, 2000), betas, rtol=1e-9, atol=1e-12)


def test_respaced_chain_follows_the_original_alpha_path(parts):
    cfg, betas, _ = parts
    ac = np.cumprod(1 - betas)
    for steps in (10, 50, 250):
        b = respaced_betas(cfg, steps)
        assert len(b) == steps
        path = np.cumprod(1 - b)
        keep = np.unique(np.round(np.linspace(0, 1999, steps)).astype(int))
        assert np.allclose(path, ac[keep], rtol=1e-9)
        assert path[-1] == pytest.approx(ac[1999], rel=1e-9)


def test_installed_schedule_ends_at_the_trained_final_noise_level(parts):
    cfg, betas, dmod = parts
    trained_final = float(np.sqrt(np.cumprod(1 - betas)[-1]))
    n = _net(dmod, cfg)
    set_schedule(n, cfg, 50, "cpu", "respaced")
    assert n.num_timesteps == 50 and n._sched_mode == "respaced"
    assert float(n.sqrt_alphas_cumprod_prev[-1]) == pytest.approx(trained_final, rel=1e-6)
    m = _net(dmod, cfg)
    set_schedule(m, cfg, 50, "cpu", "rederive")
    assert float(m.sqrt_alphas_cumprod_prev[-1]) > 5 * trained_final  # re-derived chain stops far from noise-free


def test_bad_mode_rejected(parts):
    cfg, _, dmod = parts
    with pytest.raises(ValueError):
        set_schedule(_net(dmod, cfg), cfg, 10, "cpu", "nope")
