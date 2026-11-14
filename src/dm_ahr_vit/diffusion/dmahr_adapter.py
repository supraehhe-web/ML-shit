"""Adapter around the pinned upstream DM-AHR/SR3 implementation.

The upstream tree is vendored in `third_party/dmahr` at a pinned commit so the
audit and the executable code cannot drift apart.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import config_hash
from ..paths import REPO_ROOT

UPSTREAM_DIR = REPO_ROOT / "third_party" / "dmahr"
CHECKPOINT_NAME = "I725000_E1083_gen.pth"


def _ensure_path() -> None:
    p = str(UPSTREAM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _strip_comments(text: str) -> str:
    """Make the upstream config JSON-parseable.

    `config/*.json` in the pinned repo is JSON5: it has `//` and `/* */`
    comments plus trailing commas, so plain `json.load` rejects it.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    escaped = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        out.append(ch)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def load_upstream_config(name: str = "akiec.json") -> dict:
    raw = (UPSTREAM_DIR / "config" / name).read_text()
    return json.loads(_strip_comments(raw))


def build_opt(cfg: dict, upstream: dict | None = None) -> dict:
    upstream = upstream or load_upstream_config()
    keep = upstream.get("model", {})
    opt = {
        "name": "dmahr",
        "phase": "val",
        "gpu_ids": [],
        "path": {"log": "/tmp", "tb_logger": "/tmp", "results": "/tmp",
                 "checkpoint": "/tmp", "resume_state": None},
        "datasets": {},
        "model": {
            "which_model_G": keep.get("which_model_G", "sr3"),
            "finetune_norm": False,
            "unet": dict(cfg["unet"]),
            "beta_schedule": {
                "train": {**cfg["beta_schedule"], "n_timestep": int(cfg["num_timesteps"])},
                "val": {**cfg["beta_schedule"], "n_timestep": int(cfg["num_timesteps"])},
            },
            "diffusion": {
                "image_size": int(cfg["native_resolution"]),
                "channels": 3,
                "conditional": True,
            },
        },
        "train": {
            "n_iter": int(cfg["train"]["iterations"]),
            "val_freq": 1e9,
            "save_checkpoint_freq": 1e9,
            "print_freq": 100,
            "optimizer": {"type": cfg["train"]["optimizer"], "lr": float(cfg["train"]["lr"])},
            "ema_scheduler": {},
        },
        "wandb": {},
        "distributed": False,
    }
    return opt


def opt_hash(opt: dict) -> str:
    return config_hash(opt)


def load_model(cfg: dict, checkpoint: Path, device: str = "cuda") -> torch.nn.Module:
    _ensure_path()
    import model.networks as networks  # noqa: PLC0415

    opt = build_opt(cfg)
    net = networks.define_G(opt)
    # The upstream __init__ deliberately skips buffer registration (the call is
    # commented out), so the diffusion buffers only exist after an explicit
    # schedule call. Register them first, then load so strict=True can match.
    net.set_new_noise_schedule(
        {"schedule": cfg["beta_schedule"]["schedule"],
         "linear_start": cfg["beta_schedule"]["linear_start"],
         "linear_end": cfg["beta_schedule"]["linear_end"],
         "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device("cpu"),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)
    net.num_timesteps = int(cfg["num_timesteps"])
    net.set_new_noise_schedule(
        {"schedule": cfg["beta_schedule"]["schedule"],
         "linear_start": cfg["beta_schedule"]["linear_start"],
         "linear_end": cfg["beta_schedule"]["linear_end"],
         "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device(device),
    )
    print(f"loaded DM-AHR checkpoint {checkpoint} (strict)", flush=True)
    return net


def preprocess(images: list[Image.Image], size: int = 128) -> torch.Tensor:
    arrs = []
    for im in images:
        a = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
                       dtype=np.float32)
        arrs.append((a / 127.5) - 1.0)
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def postprocess(tensor: torch.Tensor) -> list[Image.Image]:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None]
    arr = np.clip((arr + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return [Image.fromarray(a.transpose(1, 2, 0)) for a in arr]


@torch.no_grad()
def clean_batch(
    net: torch.nn.Module,
    images: list[Image.Image],
    cfg: dict,
    device: str = "cuda",
    seed: int | None = None,
    num_timesteps: int | None = None,
    use_bf16: bool = False,
    respaced: bool = False,
    schedule: str | None = None,
    start_t: int | None = None,
) -> tuple[list[Image.Image], dict]:
    """Run DM-AHR on a batch of images. Returns cleaned images and run stats."""
    size = int(cfg["native_resolution"])
    x = preprocess(images, size).to(device)
    if respaced:
        # Few-step DDIM over the trained 2,000-step schedule. Truncating the
        # schedule instead (num_timesteps=25) barely adds noise and produces
        # garbage; striding the trained schedule is what makes 25 steps work.
        out, stats = respaced_super_resolution(
            net, x, steps=int(num_timesteps or 25), seed=seed, use_bf16=use_bf16,
            start_t=start_t,
        )
        stats.update({"respaced": True, "num_timesteps": int(net.num_timesteps),
                      "steps": int(num_timesteps or 25), "start_t": start_t,
                      "bf16": bool(use_bf16)})
        return postprocess(out), stats
    if num_timesteps is not None and int(num_timesteps) != int(net.num_timesteps):
        net.num_timesteps = int(num_timesteps)
        net.set_new_noise_schedule(
            {"schedule": schedule or cfg["beta_schedule"]["schedule"],
             "linear_start": cfg["beta_schedule"]["linear_start"],
             "linear_end": cfg["beta_schedule"]["linear_end"],
             "n_timestep": int(num_timesteps)},
            device=torch.device(device),
        )
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    # Upstream's `continous=False` path returns `ret_img[-1]`, which silently
    # collapses a batch to its last image. Ask for the full trajectory and take
    # the final B rows, which is the last denoised set for the whole batch.
    # bf16 autocast roughly halves the cost of the 2,000 sequential UNet calls
    # on Ampere tensor cores; the sampler is compute bound, not memory bound.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        trajectory = net.super_resolution(x, continous=True)
    out = trajectory[-x.shape[0]:]
    finite = bool(torch.isfinite(out).all())
    stats = {
        "num_timesteps": int(net.num_timesteps),
        "finite": finite,
        "shape": list(out.shape),
        "min": float(out.min()),
        "max": float(out.max()),
        "seed": seed,
        "bf16": use_bf16,
        "batch_returned": int(out.shape[0]),
        "batch_requested": int(x.shape[0]),
    }
    if not finite:
        raise FloatingPointError("DM-AHR produced non-finite values")
    if out.shape[0] != x.shape[0]:
        raise RuntimeError("DM-AHR returned the wrong number of images")
    return postprocess(out), stats


@torch.no_grad()
def respaced_super_resolution(net, cond: torch.Tensor, steps: int = 25,
                              seed: int | None = None,
                              use_bf16: bool = False,
                              start_t: int | None = None) -> tuple[torch.Tensor, dict]:
    """Deterministic DDIM sampling over the trained schedule in `steps` jumps.

    Uses the model's own 2,000-step alphas and visits `steps` of them evenly
    spaced, so every denoiser call is made at a noise level the model was
    trained on. This is what makes 25-step sampling usable.

    `start_t` starts the chain at that schedule index from the noised
    conditioning image instead of from pure noise. Hair is thin, so a chain
    that begins at a middling noise level keeps the global tone of the input and
    never visits the high-noise levels where a briefly trained model is weak.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    T = int(net.num_timesteps)
    top = T - 1 if start_t is None else int(min(max(start_t, 1), T - 1))
    seq = np.linspace(top, 0, steps).round().astype(int)
    seq = np.unique(seq)[::-1]  # descending, no duplicates
    noise = torch.randn(cond.shape, device=cond.device)
    if start_t is None:
        x = noise
    else:
        a = net.alphas_cumprod[top]
        x = a.sqrt() * cond + (1.0 - a).sqrt() * noise
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i, t in enumerate(seq):
            level = torch.FloatTensor(
                [net.sqrt_alphas_cumprod_prev[t + 1]]).repeat(cond.shape[0], 1).to(cond.device)
            eps = net.denoise_fn(torch.cat([cond, x], dim=1), level)
            x0 = (net.sqrt_recip_alphas_cumprod[t] * x
                  - net.sqrt_recipm1_alphas_cumprod[t] * eps).clamp(-1.0, 1.0)
            if i == len(seq) - 1:
                x = x0
                break
            t_next = int(seq[i + 1])
            ac_next = net.alphas_cumprod[t_next]
            # Deterministic DDIM step: re-noise x0 to the next scheduled level.
            x = ac_next.sqrt() * x0 + (1.0 - ac_next).sqrt() * eps
    stats = {"finite": bool(torch.isfinite(x).all()),
             "shape": list(x.shape), "timesteps_visited": seq.tolist()}
    if not stats["finite"]:
        raise FloatingPointError("DM-AHR produced non-finite values")
    return x, stats

