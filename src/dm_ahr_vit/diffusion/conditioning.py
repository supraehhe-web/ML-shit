"""Noise-level conditioning for the DM-AHR/SR3 denoiser.

Upstream trains (`GaussianDiffusion.p_losses`) and samples
(`GaussianDiffusion.p_mean_variance`) with the *continuous noise level*
sqrt(alphas_cumprod_prev) as the denoiser's second argument. The first version
of our trainer instead fed `t / T`, a value that rises from 0 to 1 while the
upstream level falls from 1 to ~0, so a checkpoint trained that way is sampled
with a conditioning signal it never saw. `mode="legacy"` reproduces that
behaviour for diagnosis only; `mode="upstream"` is the faithful one.
"""

from __future__ import annotations

import numpy as np
import torch

MODES = ("upstream", "legacy")


def _prev(net) -> np.ndarray:
    return np.asarray(net.sqrt_alphas_cumprod_prev, dtype=np.float64)


def sample_noise_level(net, batch: int, device, t: int | None = None,
                       rng: np.random.Generator | None = None) -> tuple[torch.Tensor, int]:
    """Upstream's training draw: one t per step, level uniform in [prev[t-1], prev[t]]."""
    prev = _prev(net)
    rng = rng if rng is not None else np.random.default_rng()
    if t is None:
        t = int(rng.integers(1, net.num_timesteps + 1))
    lo, hi = float(prev[t - 1]), float(prev[t])
    draw = rng.uniform(min(lo, hi), max(lo, hi), size=batch)
    return torch.as_tensor(draw, dtype=torch.float32, device=device), int(t)


def denoiser_level(net, t: int, level: torch.Tensor, mode: str) -> torch.Tensor:
    """Value passed to `denoise_fn` for a step whose true noise level is `level` (shape (B,))."""
    if mode == "upstream":
        return level.view(-1, 1)
    if mode == "legacy":
        return torch.full((level.shape[0],), float(t) / net.num_timesteps, device=level.device)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def noise_prediction_loss(net, hr: torch.Tensor, cond: torch.Tensor, mode: str = "upstream",
                          t: int | None = None, noise: torch.Tensor | None = None,
                          rng: np.random.Generator | None = None,
                          return_parts: bool = False):
    """Mean absolute noise-prediction error for one batch (the DM-AHR training loss).

    `legacy` keeps the original trainer's draw (level = prev[t-1]) so the old
    behaviour is reproduced exactly; `upstream` uses the continuous draw.
    """
    b = hr.shape[0]
    rng = rng if rng is not None else np.random.default_rng()
    if mode == "legacy":
        if t is None:
            t = int(rng.integers(1, net.num_timesteps + 1))
        level = torch.full((b,), float(_prev(net)[t - 1]), device=hr.device)
    else:
        level, t = sample_noise_level(net, b, hr.device, t=t, rng=rng)
    noise = torch.randn_like(hr) if noise is None else noise
    lv = level.view(-1, 1, 1, 1)
    x_noisy = lv * hr + (1 - lv**2).sqrt() * noise
    pred = net.denoise_fn(torch.cat([cond, x_noisy], dim=1), denoiser_level(net, t, level, mode))
    err = pred - noise
    loss = err.abs().mean()
    if return_parts:
        return loss, {"t": t, "mse": float((err ** 2).mean()), "l1": float(err.abs().mean())}
    return loss


def fixed_level_error(net, hr: torch.Tensor, cond: torch.Tensor, t: int, noise: torch.Tensor,
                      mode: str) -> dict:
    """Noise-prediction error at a fixed step with an identical noisy input for both modes.

    The input is always built at level prev[t-1]; only the conditioning value
    handed to the denoiser differs, so the two modes are compared like for like.
    """
    level = torch.full((hr.shape[0],), float(_prev(net)[t - 1]), device=hr.device)
    lv = level.view(-1, 1, 1, 1)
    x_noisy = lv * hr + (1 - lv**2).sqrt() * noise
    pred = net.denoise_fn(torch.cat([cond, x_noisy], dim=1), denoiser_level(net, t, level, mode))
    err = pred - noise
    return {"l1": float(err.abs().mean()), "mse": float((err ** 2).mean())}
