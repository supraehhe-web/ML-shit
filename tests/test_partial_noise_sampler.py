"""The partial-noise start of the respaced sampler: starts where asked, stays finite."""

import pytest
import torch

from dm_ahr_vit.config import load_config
from dm_ahr_vit.diffusion.dmahr_adapter import (
    _ensure_path,
    build_opt,
    respaced_super_resolution,
)

SIZE = 16


@pytest.fixture(scope="module")
def net():
    _ensure_path()
    import model.networks as networks  # noqa: PLC0415

    cfg = load_config("configs/dmahr.json")
    cfg["native_resolution"] = SIZE
    cfg["unet"] = {"in_channel": 6, "out_channel": 3, "inner_channel": 32,
                   "channel_multiplier": [1, 2], "attn_res": [], "res_blocks": 1,
                   "dropout": 0.0}
    n = networks.define_G(build_opt(cfg))
    n.set_new_noise_schedule(
        {**cfg["beta_schedule"], "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device("cpu"))
    n.eval()
    return n


def test_default_chain_starts_at_the_top_of_the_schedule(net):
    cond = torch.rand(2, 3, SIZE, SIZE) * 2 - 1
    _, stats = respaced_super_resolution(net, cond, steps=4, seed=0)
    assert stats["timesteps_visited"][0] == int(net.num_timesteps) - 1
    assert stats["timesteps_visited"][-1] == 0


@pytest.mark.parametrize("start_t", [300, 1000])
def test_partial_start_visits_only_levels_below_it(net, start_t):
    cond = torch.rand(2, 3, SIZE, SIZE) * 2 - 1
    out, stats = respaced_super_resolution(net, cond, steps=5, seed=0, start_t=start_t)
    assert stats["timesteps_visited"][0] == start_t
    assert max(stats["timesteps_visited"]) == start_t
    assert stats["finite"] and out.shape == cond.shape
    assert float(out.abs().max()) <= 1.0 + 1e-6


def test_partial_start_is_seed_deterministic(net):
    cond = torch.rand(2, 3, SIZE, SIZE) * 2 - 1
    a, _ = respaced_super_resolution(net, cond, steps=4, seed=3, start_t=500)
    b, _ = respaced_super_resolution(net, cond, steps=4, seed=3, start_t=500)
    c, _ = respaced_super_resolution(net, cond, steps=4, seed=4, start_t=500)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_clean_batch_passes_start_t_and_reports_stats(net):
    from PIL import Image

    from dm_ahr_vit.diffusion.dmahr_adapter import clean_batch

    cfg = load_config("configs/dmahr.json")
    cfg["native_resolution"] = SIZE
    imgs = [Image.new("RGB", (40, 30), (120, 90, 80)) for _ in range(2)]
    outs, stats = clean_batch(net, imgs, cfg, device="cpu", seed=1, num_timesteps=4,
                              respaced=True, start_t=400)
    assert stats["start_t"] == 400 and stats["steps"] == 4 and stats["bf16"] is False
    assert stats["timesteps_visited"][0] == 400
    assert len(outs) == 2 and outs[0].size == (SIZE, SIZE)
