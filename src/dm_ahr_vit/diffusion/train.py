"""Leakage-controlled DM-AHR training on train-split paired sources only."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import save_json
from ..paths import data_root
from ..utils import setup_logging
from .dmahr_adapter import _ensure_path, build_opt, opt_hash

log = setup_logging()


def _resolve_manifest_path(path: str) -> Path:
    """Re-root a recorded modal-volume path onto the current DATA_ROOT.

    Pair manifests were written inside a container, so they hold absolute
    `/__modal/volumes/<id>/data/...` paths that are void on any other volume.
    Splitting on the `/data/` marker keeps them usable.
    """
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def _manifest_hash(pairs: pd.DataFrame) -> str:
    """SHA-256 over the ordered pair ids, so a run records exactly what it saw."""
    ids = sorted(str(i) for i in pairs["pair_id"])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


class PairedDataset(Dataset):
    """(hairy, hairless) pairs at native resolution, in [-1, 1]."""

    def __init__(self, pairs: pd.DataFrame, size: int = 128):
        self.pairs = pairs.reset_index(drop=True)
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        r = self.pairs.iloc[idx]
        hr = np.asarray(
            Image.open(_resolve_manifest_path(r["reference_path"])).convert("RGB").resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        cond = np.asarray(
            Image.open(_resolve_manifest_path(r["hairy_path"])).convert("RGB").resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        if idx % 2 == 1:
            hr, cond = hr[:, ::-1].copy(), cond[:, ::-1].copy()
        return {
            "HR": torch.from_numpy(hr / 127.5 - 1.0).permute(2, 0, 1),
            "SR": torch.from_numpy(cond / 127.5 - 1.0).permute(2, 0, 1),
        }


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def val_loss(net, loader, device: str, timesteps=(200, 800, 1400, 1900)) -> float:
    net.eval()
    losses = []
    for batch in loader:
        hr = batch["HR"].to(device)
        cond = batch["SR"].to(device)
        for t in timesteps:
            # Upstream encodes the noise level as sqrt(alpha_cumprod_prev), not as
            # t/T: `NoiseLevelEncoder` multiplies a frequency ramp by this value,
            # and `p_sample` feeds the same quantity at inference. Using t/T here
            # trains one encoding and samples with another.
            alpha = net.sqrt_alphas_cumprod_prev[int(t) - 1]
            level = torch.full((hr.shape[0], 1), float(alpha), device=device)
            noise = torch.randn_like(hr)
            noisy = alpha * hr + (1 - alpha**2) ** 0.5 * noise
            pred = net.denoise_fn(torch.cat([cond, noisy], dim=1), level)
            losses.append(float((pred - noise).abs().mean()))
    net.train()
    return float(np.mean(losses))


def training_loss(net, hr: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """One training loss, equivalent to upstream `p_losses` / numel.

    RNG draw order (numpy t, numpy levels, torch noise) matches upstream so a
    seeded differential test can compare the two exactly.
    """
    t = np.random.randint(1, net.num_timesteps + 1)
    # Continuous noise level per sample, as upstream: uniform between
    # sqrt(alpha_cumprod_prev) at t-1 and at t.
    lo = float(net.sqrt_alphas_cumprod_prev[t - 1])
    hi = float(net.sqrt_alphas_cumprod_prev[t])
    alpha = torch.tensor(
        np.random.uniform(lo, hi, size=hr.shape[0]), dtype=hr.dtype, device=hr.device
    )
    alpha_4d = alpha.view(-1, 1, 1, 1)
    noise = torch.randn_like(hr)
    x_noisy = alpha_4d * hr + (1 - alpha_4d**2) ** 0.5 * noise
    pred = net.denoise_fn(torch.cat([cond, x_noisy], dim=1), alpha.view(-1, 1))
    return (pred - noise).abs().mean()


def _describe(ds) -> dict:
    """What a dataset contains, for the training manifest (pair table or composite)."""
    if hasattr(ds, "manifest"):
        m = ds.manifest()
        return {"n": m["n"], "ids": m["target_ids"] + m["source_ids"],
                "splits": m["splits"], "groups": set(m["groups"]),
                "hash": hashlib.sha256("\n".join(sorted(m["target_ids"] + m["source_ids"]))
                                       .encode()).hexdigest()}
    p = ds.pairs
    return {"n": int(len(ds)), "ids": p["pair_id"].tolist(),
            "splits": sorted(p["split"].unique().tolist()),
            "groups": set(p["group_id"]), "hash": _manifest_hash(p)}


def train(
    cfg: dict,
    pairs: pd.DataFrame | None,
    val_pairs: pd.DataFrame | None,
    out_dir: Path,
    device: str = "cuda",
    overfit: bool = False,
    init_from: Path | None = None,
    max_iterations: int | None = None,
    train_ds: Dataset | None = None,
    val_ds: Dataset | None = None,
    num_workers: int = 4,
    grad_clip: float | None = 1.0,
) -> dict:
    _ensure_path()
    import model.networks as networks  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    opt = build_opt(cfg)
    net = networks.define_G(opt).to(device)
    # Upstream's define_G/__init__ never registers the diffusion buffers, so the
    # noise schedule must be set explicitly before any q_sample/p_losses use.
    net.set_new_noise_schedule(
        {"schedule": cfg["beta_schedule"]["schedule"],
         "linear_start": cfg["beta_schedule"]["linear_start"],
         "linear_end": cfg["beta_schedule"]["linear_end"],
         "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device(device),
    )
    opt_g = torch.optim.Adam(net.parameters(), lr=float(cfg["train"]["lr"]))
    iterations = int(cfg["train"]["iterations"])
    if overfit and pairs is not None:
        pairs = pairs.head(int(cfg["train"].get("overfit_pairs", 12)))
        iterations = 500
    if max_iterations is not None:
        iterations = int(max_iterations)
    if train_ds is None:
        train_ds = PairedDataset(pairs, int(cfg["native_resolution"]))
    if val_ds is None:
        val_ds = PairedDataset(val_pairs, int(cfg["native_resolution"]))
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                              num_workers=int(num_workers), pin_memory=True,
                              persistent_workers=int(num_workers) > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

    state_path = out_dir / ("overfit_state.pt" if overfit else "train_state.pt")
    start = 0
    best = float("inf")
    history = []
    if init_from is not None:
        # Warm start from an existing checkpoint. Used to check that the fixed
        # trainer does not degrade weights it did not produce.
        init_state = torch.load(init_from, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in init_state):
            init_state = {k.replace("module.", "", 1): v for k, v in init_state.items()}
        net.load_state_dict(init_state, strict=True)
        log.info("warm-started from %s", init_from)
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        opt_g.load_state_dict(state["optimizer"])
        start = int(state["iteration"])
        best = float(state.get("best_val", best))
        history = state.get("history", [])
        log.info("resumed DM-AHR at iteration %d", start)

    d_tr, d_va = _describe(train_ds), _describe(val_ds)
    save_json(
        {
            "config": cfg,
            "opt_hash": opt_hash(opt),
            "train_pairs": d_tr["n"],
            "val_pairs": d_va["n"],
            "train_pair_ids": d_tr["ids"],
            "val_pair_ids": d_va["ids"],
            "train_sources_splits": d_tr["splits"],
            "val_sources_splits": d_va["splits"],
            "leakage_check": bool(not d_tr["groups"] & d_va["groups"]),
            "overfit": overfit,
            # Provenance recorded at run time so a later audit does not have to
            # infer which noise-level convention or which data this run used.
            "noise_level_convention": "sqrt_alphas_cumprod_prev per sample, "
                                      "continuous in [t-1, t] (upstream p_losses)",
            "num_timesteps": int(net.num_timesteps),
            "ema": "none",
            "grad_clip": float(grad_clip) if grad_clip else None,
            "batch_size": int(cfg["train"]["batch_size"]),
            "init_from": str(init_from) if init_from is not None else None,
            "train_manifest_sha256": d_tr["hash"],
            "dataset_kind": type(train_ds).__name__,
        },
        out_dir / "training_manifest.json",
    )

    net.train()
    it = _infinite(train_loader)
    ckpt_every = int(cfg["train"]["checkpoint_every"])
    val_every = int(cfg["train"]["val_every"])
    t0 = time.time()
    for step in range(start + 1, iterations + 1):
        batch = next(it)
        hr = batch["HR"].to(device, non_blocking=True)
        cond = batch["SR"].to(device, non_blocking=True)
        loss = training_loss(net, hr, cond)
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))
        opt_g.step()

        if step % 100 == 0:
            log.info("dmahr iter %d/%d loss %.5f (%.0fs)", step, iterations,
                     float(loss), time.time() - t0)
        if step % val_every == 0 or step == iterations:
            v = val_loss(net, val_loader, device)
            history.append({"iteration": step, "train_l1": float(loss), "val_l1": v})
            log.info("dmahr val iter %d: %.5f", step, v)
            if v < best:
                best = v
                torch.save(net.state_dict(), out_dir / "best_gen.pth")
        if step % ckpt_every == 0 or step == iterations:
            tmp = state_path.with_suffix(".tmp")
            torch.save(
                {"iteration": step, "model": net.state_dict(),
                 "optimizer": opt_g.state_dict(), "best_val": best,
                 "history": history},
                tmp,
            )
            tmp.replace(state_path)
        # Iteration-numbered weights at a coarser interval, so a learning curve
        # can be measured afterwards without retraining.
        snap_every = int(cfg["train"].get("snapshot_every", 5000))
        if snap_every and (step % snap_every == 0 or step == iterations):
            torch.save(net.state_dict(), out_dir / f"gen_iter{step:06d}.pth")

    torch.save(net.state_dict(), out_dir / "final_gen.pth")
    save_json({"history": history, "best_val": best, "iterations": iterations,
               "seconds": round(time.time() - t0, 1)}, out_dir / "train_log.json")
    return {"iterations": iterations, "best_val": best, "seconds": time.time() - t0}

