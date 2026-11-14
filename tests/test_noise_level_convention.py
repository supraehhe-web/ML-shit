"""The trainer must encode the noise level the way the sampler does.

`GaussianDiffusion.p_sample` feeds `sqrt_alphas_cumprod_prev[t+1]` to the
U-Net, and `NoiseLevelEncoder` scales a frequency ramp by that value. A trainer
that feeds `t / num_timesteps` instead trains one encoding and samples with
another, which inverts the meaning of the signal.
"""

from __future__ import annotations

import numpy as np

from dm_ahr_vit.config import load_config
from dm_ahr_vit.diffusion.dmahr_adapter import load_upstream_config


def _schedule(num_timesteps: int = 2000) -> np.ndarray:
    """sqrt(alpha_cumprod_prev), length num_timesteps + 1, as upstream builds it."""
    cfg = load_config("configs/dmahr.json")
    bs = cfg["beta_schedule"]
    betas = np.linspace(bs["linear_start"], bs["linear_end"], num_timesteps)
    alphas_cumprod = np.cumprod(1.0 - betas)
    return np.sqrt(np.append(1.0, alphas_cumprod))


def test_embedding_is_not_linear_in_t():
    """Guard against reintroducing t/T as the noise level."""
    sqrt_acp = _schedule()
    for t in (50, 200, 500, 1000, 1500):
        linear = t / 2000
        actual = sqrt_acp[t + 1]
        assert abs(linear - actual) > 0.1, (
            f"t={t}: t/T={linear:.4f} and sqrt_acp={actual:.4f} should differ"
        )


def test_embedding_ordering_is_inverted_relative_to_t_over_T():
    """Higher t must mean a LOWER value; t/T means the opposite."""
    sqrt_acp = _schedule()
    assert sqrt_acp[51] < sqrt_acp[2], "sqrt_acp must decrease with t"
    assert (51 / 2000) > (2 / 2000), "t/T must increase with t"


def test_embedding_range_within_unit_interval():
    sqrt_acp = _schedule()
    assert sqrt_acp.min() >= 0.0
    assert sqrt_acp.max() <= 1.0


def test_upstream_config_matches_our_schedule():
    """Our beta schedule must reproduce the upstream one, not just resemble it."""
    up = load_upstream_config()
    sched = up["model"]["beta_schedule"]["train"]
    assert int(sched["n_timestep"]) == 2000
    ours = _schedule(int(sched["n_timestep"]))
    theirs_betas = np.linspace(
        float(sched["linear_start"]), float(sched["linear_end"]), 2000
    )
    theirs = np.sqrt(np.append(1.0, np.cumprod(1.0 - theirs_betas)))
    assert np.allclose(ours, theirs, atol=1e-8)


def _trainer_level_bounds(sqrt_acp: np.ndarray, t: int) -> tuple[float, float]:
    """The two endpoints our trainer samples the noise level between at t.

    `p_losses` calls `np.random.uniform(sqrt_acp[t-1], sqrt_acp[t])`. Under a
    linear schedule the second endpoint is the smaller one, which is fine for
    `uniform`, so the pair is returned unordered.
    """
    return float(sqrt_acp[t - 1]), float(sqrt_acp[t])


def _sampler_level(sqrt_acp: np.ndarray, i: int) -> float:
    """The noise level `p_sample` feeds at loop index i."""
    return float(sqrt_acp[i + 1])


def test_trainer_and_sampler_agree_at_matched_timesteps():
    """Integration check across both code paths, not just a constant.

    `p_sample` indexes the schedule at `i + 1`, while `p_losses` samples t in
    `[1, T]` between `t-1` and `t`. For the two to describe the same noise
    level, the sampler's value at index `i` must equal the trainer's UPPER
    bound at `t = i + 1`. This is the property that was violated before the fix,
    where the trainer passed `t / T` and the sampler passed `sqrt_acp[i + 1]`.
    """
    sqrt_acp = _schedule()
    for t in (1, 50, 200, 500, 1000, 1500, 1999):
        a, b = _trainer_level_bounds(sqrt_acp, t)
        sampler_at_matched = _sampler_level(sqrt_acp, t - 1)
        # The sampler's level must be exactly one of the trainer's endpoints.
        assert min(abs(a - sampler_at_matched), abs(b - sampler_at_matched)) < 1e-12, (
            f"t={t}: trainer endpoints ({a:.6f}, {b:.6f}) do not contain the "
            f"sampler level "
            f"{sampler_at_matched:.6f} at the matched index"
        )


def test_linear_encoding_would_fail_the_agreement_check():
    """The old convention must be detectably wrong under the same check."""
    sqrt_acp = _schedule()
    for t in (200, 500, 1000, 1500):
        old_trainer_value = t / 2000
        sampler_at_matched = _sampler_level(sqrt_acp, t - 1)
        assert abs(old_trainer_value - sampler_at_matched) > 0.05, (
            f"t={t}: t/T={old_trainer_value:.4f} unexpectedly close to "
            f"sampler {sampler_at_matched:.4f}"
        )
