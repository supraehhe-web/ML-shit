from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dm_ahr_vit.diffusion import diagnostics as dg


def _smooth(n=128):
    y, x = np.mgrid[0:n, 0:n] / n
    return np.stack([0.3 + 0.4 * x, 0.4 + 0.3 * y, 0.5 + 0.2 * x * y], axis=2)


def _img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def test_noise_sigma_recovers_white_noise_level():
    rng = np.random.default_rng(0)
    base = _smooth()
    for sigma in (0.02, 0.05):
        est = dg.noise_sigma_mad(_img(base + rng.normal(0, sigma, base.shape)))
        expected = sigma / np.sqrt(3)  # grey = mean of three independent channels
        assert abs(est - expected) / expected < 0.25
    assert dg.noise_sigma_mad(_img(base)) < 0.01


def test_blur_lowers_and_noise_raises_high_frequency_energy():
    rng = np.random.default_rng(1)
    base = _smooth() + rng.normal(0, 0.02, (128, 128, 3))
    sharp = _img(base)
    blurred = sharp.resize((32, 32), Image.Resampling.BICUBIC).resize((128, 128), Image.Resampling.BICUBIC)
    noisy = _img(base + rng.normal(0, 0.08, base.shape))
    assert dg.mean_abs_laplacian(blurred) < dg.mean_abs_laplacian(sharp) < dg.mean_abs_laplacian(noisy)
    assert dg.high_freq_power_fraction(blurred) < dg.high_freq_power_fraction(noisy)


def test_radial_power_spectrum_shapes_and_white_noise_is_flat():
    f, p = dg.radial_power_spectrum(_img(np.random.default_rng(2).random((128, 128, 3))), n_bins=8)
    assert f.shape == p.shape == (8,)
    assert p.max() / p.min() < 20
    fs, ps = dg.radial_power_spectrum(_img(_smooth()), n_bins=8)
    assert ps[0] > 1e3 * ps[-1]


def test_psnr_ssim_identity_and_ordering():
    a = _img(_smooth())
    assert dg.psnr_ssim(a, a)["ssim"] == pytest.approx(1.0)
    noisy = _img(_smooth() + np.random.default_rng(3).normal(0, 0.1, (128, 128, 3)))
    assert dg.psnr_ssim(noisy, a)["psnr"] < 30


def test_phase1_rule_boundary():
    assert dg.phase1_verdict(39.04)["adapter_validated"]
    assert not dg.phase1_verdict(39.03)["adapter_validated"]
    assert "not reproduced" in dg.phase1_verdict(26.8)["statement"]


def test_mse_evidence_requires_both_directions():
    bug = {"ours": {"legacy": 0.1, "upstream": 0.5}, "released": {"legacy": 0.6, "upstream": 0.1}}
    assert dg.mse_evidence(bug)["trained_wrongly_evidence"]
    fine = {"ours": {"legacy": 0.3, "upstream": 0.1}, "released": {"legacy": 0.6, "upstream": 0.1}}
    assert not dg.mse_evidence(fine)["trained_wrongly_evidence"]


def test_checkpoint_trend_labels():
    up = dg.checkpoint_trend({5000: 14.0, 10000: 16.0, 15000: 17.5, 20000: 19.0})
    assert up["undertrained_pattern"]
    flat = dg.checkpoint_trend({5000: 14.0, 10000: 14.3, 15000: 13.2, 20000: 14.1})
    assert not flat["undertrained_pattern"] and flat["label"] == "flat or erratic"


def test_warm_start_and_overfit_rules():
    assert dg.warm_start_jump(0.05, 0.09)["loss_jump"]
    assert not dg.warm_start_jump(0.05, 0.06)["loss_jump"]
    assert dg.overfit_pass(26.0, 22.0)["passed"] and not dg.overfit_pass(24.0, 22.0)["passed"]


def test_arm_validity_gate():
    assert dg.arm_is_valid(25.5, 0.6, 22.1, 0.52)["valid"]
    assert not dg.arm_is_valid(16.3, 0.13, 22.1, 0.52)["valid"]


def test_branch_is_not_decided_without_evidence():
    out = dg.phase2_branch(None, None, None, None)
    assert out["branch"] is None and out["status"] == "NOT RUN"
    mse = {"trained_wrongly_evidence": True}
    trend = {"undertrained_pattern": False}
    warm = {"legacy": {"loss_jump": True}, "fixed": {"loss_jump": False}}
    over = {"passed": True}
    assert dg.phase2_branch(mse, trend, warm, over)["branch"] == "c"
    ok = {"trained_wrongly_evidence": False}
    assert dg.phase2_branch(ok, {"undertrained_pattern": True},
                            {"legacy": {"loss_jump": False}, "fixed": {"loss_jump": False}},
                            {"passed": True})["branch"] == "b"


def test_sweep_choice_picks_smallest_valid_step_count():
    rows = [{"steps": 50, "psnr": 15.0, "ssim": 0.2}, {"steps": 100, "psnr": 24.8, "ssim": 0.6},
            {"steps": 250, "psnr": 26.0, "ssim": 0.68}, {"steps": 2000, "psnr": 26.8, "ssim": 0.7}]
    out = dg.sweep_choice(rows, hairy_psnr=22.1, hairy_ssim=0.52)
    assert out["chosen_steps"] == 250
    none = dg.sweep_choice([{"steps": 100, "psnr": 14.0, "ssim": 0.1}, {"steps": 2000, "psnr": 14.5, "ssim": 0.1}], 22.1, 0.52)
    assert none["chosen_steps"] is None


def test_respace_choice_rule():
    rows = [{"steps": 25, "psnr": 20.0, "ssim": 0.4}, {"steps": 50, "psnr": 25.5, "ssim": 0.6},
            {"steps": 100, "psnr": 26.0, "ssim": 0.65}, {"steps": 250, "psnr": 26.2, "ssim": 0.66}]
    assert dg.respace_choice(rows, 22.0, 0.5)["chosen_steps"] == 50
    assert dg.respace_choice([{"steps": 25, "psnr": 15.0, "ssim": 0.1}], 22.0, 0.5)["chosen_steps"] is None
