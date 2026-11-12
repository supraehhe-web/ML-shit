"""Leakage-controlled DM-AHR training on train-split paired sources only."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import save_json
from ..utils import setup_logging
from .conditioning import MODES, fixed_level_error, noise_prediction_loss
from .dmahr_adapter import _ensure_path, build_opt, opt_hash

log = setup_logging()


class PairedDataset(Dataset):
    """(hairy, hairless) pairs at native resolution, in [-1, 1].

    Pairs are decoded and resized once at construction and kept in memory (about
    100 KB per pair at 128 px). Re-decoding two PNGs per sample on a throttled
    container made the first trainer several times slower than the GPU needed.
    """

    def __init__(self, pairs: pd.DataFrame, size: int = 128):
        self.pairs = pairs.reset_index(drop=True)
        self.size = size
        self._hr: list[np.ndarray] = []
        self._cond: list[np.ndarray] = []
        for _, r in self.pairs.iterrows():
            self._hr.append(self._load(r["reference_path"]))
            self._cond.append(self._load(r["hairy_path"]))

    def _load(self, path) -> np.ndarray:
        return np.asarray(
            Image.open(path).convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        hr, cond = self._hr[idx], self._cond[idx]
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
def val_loss(net, loader, device: str, timesteps=None,
             mode: str = "upstream", seed: int = 0) -> float:
    """Mean noise-prediction L1 on validation pairs at fixed steps and fixed noise."""
    net.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for batch in loader:
        hr = batch["HR"].to(device)
        cond = batch["SR"].to(device)
        for t in sorted({max(1, int(round(f * net.num_timesteps))) for f in (0.1, 0.4, 0.7, 0.95)} if timesteps is None else {int(t) for t in timesteps}):
            noise = torch.randn(hr.shape, generator=gen).to(device)
            losses.append(fixed_level_error(net, hr, cond, int(t), noise, mode)["l1"])
    net.train()
    return float(np.mean(losses))


def build_network(cfg: dict, device: str):
    _ensure_path()
    import model.networks as networks  # noqa: PLC0415

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
    return net, opt


def train(
    cfg: dict,
    pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    out_dir: Path,
    device: str = "cuda",
    overfit: bool = False,
    conditioning: str = "upstream",
    iterations: int | None = None,
    save_every: int | None = None,
    init_checkpoint: Path | None = None,
    grad_clip: float | None = None,
    seed: int = 0,
    trace_every: int = 1,
) -> dict:
    """Train DM-AHR with the selected conditioning convention.

    `conditioning="upstream"` is the faithful trainer. `"legacy"` reproduces
    the first trainer (denoiser fed t/T) and exists for diagnosis only.
    `init_checkpoint` warm-starts from existing weights (used for the
    released-checkpoint warm-start test). `save_every` keeps raw-weight
    snapshots `iter_{N}_gen.pth` so quality can be scored along training.
    """
    if conditioning not in MODES:
        raise ValueError(f"conditioning must be one of {MODES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    net, opt = build_network(cfg, device)
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        log.info("warm start from %s", init_checkpoint)
    opt_g = torch.optim.Adam(net.parameters(), lr=float(cfg["train"]["lr"]))
    total = int(iterations if iterations is not None else cfg["train"]["iterations"])
    if overfit:
        pairs = pairs.head(int(cfg["train"].get("overfit_pairs", 8)))
    train_ds = PairedDataset(pairs, int(cfg["native_resolution"]))
    val_ds = PairedDataset(val_pairs, int(cfg["native_resolution"]))
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=min(bs, max(1, len(train_ds))), shuffle=True,
                              drop_last=len(train_ds) >= bs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    state_path = out_dir / ("overfit_state.pt" if overfit else "train_state.pt")
    start = 0
    best = float("inf")
    history: list[dict] = []
    trace: list[dict] = []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("conditioning", "legacy") != conditioning:
            raise RuntimeError(
                f"{state_path} was written with conditioning={state.get('conditioning', 'legacy')}")
        net.load_state_dict(state["model"])
        opt_g.load_state_dict(state["optimizer"])
        start = int(state["iteration"])
        best = float(state.get("best_val", best))
        history = state.get("history", [])
        trace = state.get("trace", [])
        log.info("resumed DM-AHR at iteration %d", start)

    save_json(
        {
            "config": cfg,
            "opt_hash": opt_hash(opt),
            "conditioning": conditioning,
            "grad_clip": grad_clip,
            "iterations": total,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "train_pairs": int(len(train_ds)),
            "val_pairs": int(len(val_ds)),
            "train_pair_ids": train_ds.pairs["pair_id"].tolist(),
            "val_pair_ids": val_ds.pairs["pair_id"].tolist(),
            "train_sources_splits": sorted(train_ds.pairs["split"].unique().tolist()),
            "val_sources_splits": sorted(val_ds.pairs["split"].unique().tolist()),
            "leakage_check": bool(
                not set(train_ds.pairs["group_id"]) & set(val_ds.pairs["group_id"])
            ),
            "overfit": overfit,
        },
        out_dir / "training_manifest.json",
    )

    rng = np.random.default_rng(seed + start)
    net.train()
    it = _infinite(train_loader)
    ckpt_every = int(cfg["train"]["checkpoint_every"])
    val_every = int(cfg["train"]["val_every"])
    t0 = time.time()
    pending: list[tuple[int, torch.Tensor]] = []

    def flush() -> float | None:
        """Move buffered losses to the host in one transfer; returns the latest."""
        if not pending:
            return None
        vals = torch.stack([v for _, v in pending]).float().cpu().tolist()
        for (s, _), v in zip(pending, vals):
            trace.append({"iteration": s, "loss": float(v)})
        pending.clear()
        return float(vals[-1])

    last_loss = float("nan")
    for step in range(start + 1, total + 1):
        batch = next(it)
        hr = batch["HR"].to(device, non_blocking=True)
        cond = batch["SR"].to(device, non_blocking=True)
        loss = noise_prediction_loss(net, hr, cond, mode=conditioning, rng=rng)
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))
        opt_g.step()
        if step % trace_every == 0:
            pending.append((step, loss.detach()))

        if step % 100 == 0:
            last_loss = flush() or last_loss
            log.info("dmahr iter %d/%d loss %.5f (%.0fs)", step, total, last_loss,
                     time.time() - t0)
        if save_every and step % save_every == 0:
            torch.save(net.state_dict(), out_dir / f"iter_{step}_gen.pth")
        if step % val_every == 0 or step == total:
            last_loss = flush() or last_loss
            v = val_loss(net, val_loader, device, mode=conditioning)
            history.append({"iteration": step, "train_l1": last_loss, "val_l1": v})
            log.info("dmahr val iter %d: %.5f", step, v)
            if v < best:
                best = v
                torch.save(net.state_dict(), out_dir / "best_gen.pth")
        if step % ckpt_every == 0 or step == total:
            flush()
            tmp = state_path.with_suffix(".tmp")
            torch.save(
                {"iteration": step, "model": net.state_dict(),
                 "optimizer": opt_g.state_dict(), "best_val": best,
                 "history": history, "trace": trace, "conditioning": conditioning},
                tmp,
            )
            tmp.replace(state_path)

    torch.save(net.state_dict(), out_dir / "final_gen.pth")
    save_json({"history": history, "trace": trace, "best_val": best, "iterations": total,
               "conditioning": conditioning,
               "seconds": round(time.time() - t0, 1)}, out_dir / "train_log.json")
    return {"iterations": total, "best_val": best, "seconds": time.time() - t0,
            "conditioning": conditioning}
