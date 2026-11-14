"""Pre-registered trainer gates: 8-pair overfit regression and copy test.

Gate 1 restores the overfit test the defective trainer failed (margin -7.31 dB
against a required +3.0 dB). Gate 2 sets the conditioning input to the clean
target, an identity mapping that should be learned quickly.

Thresholds are fixed in `reports/prereg_fixed_trainer_arm_c.md`, committed
before this script was run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    _ensure_path,
    build_opt,
    clean_batch,
)
from dm_ahr_vit.diffusion.train import (  # noqa: E402
    PairedDataset,
    _manifest_hash,
    _resolve_manifest_path,
)
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()

GATE1_MARGIN = 3.0      # dB above the hairy floor
GATE2_MIN_PSNR = 35.0
STEPS = 1000
SIZE = 128


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def build_net(cfg, device="cuda"):
    _ensure_path()
    import model.networks as networks  # noqa: PLC0415
    net = networks.define_G(build_opt(cfg))
    net.set_new_noise_schedule(
        {"schedule": cfg["beta_schedule"]["schedule"],
         "linear_start": cfg["beta_schedule"]["linear_start"],
         "linear_end": cfg["beta_schedule"]["linear_end"],
         "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device("cpu"),
    )
    net.to(device)
    net.num_timesteps = int(cfg["num_timesteps"])
    net.set_new_noise_schedule(
        {"schedule": cfg["beta_schedule"]["schedule"],
         "linear_start": cfg["beta_schedule"]["linear_start"],
         "linear_end": cfg["beta_schedule"]["linear_end"],
         "n_timestep": int(cfg["num_timesteps"])},
        device=torch.device(device),
    )
    return net


def train_briefly(net, pairs: pd.DataFrame, iterations: int, cfg: dict,
                  device: str, copy_task: bool) -> list[float]:
    """The fixed training step, identical to `train()` in the library."""
    ds = PairedDataset(pairs, SIZE)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True,
                                         drop_last=False, num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["train"]["lr"]))
    net.train()
    losses = []
    it = iter(loader)
    for step in range(1, iterations + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        hr = batch["HR"].to(device)
        cond = hr if copy_task else batch["SR"].to(device)
        t = np.random.randint(1, net.num_timesteps + 1)
        lo = float(net.sqrt_alphas_cumprod_prev[t - 1])
        hi = float(net.sqrt_alphas_cumprod_prev[t])
        alpha = torch.tensor(np.random.uniform(lo, hi, size=hr.shape[0]),
                             dtype=hr.dtype, device=device)
        a4 = alpha.view(-1, 1, 1, 1)
        noise = torch.randn_like(hr)
        x_noisy = a4 * hr + (1 - a4**2) ** 0.5 * noise
        pred = net.denoise_fn(torch.cat([cond, x_noisy], dim=1), alpha.view(-1, 1))
        loss = (pred - noise).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 250 == 0:
            log.info("  iter %d loss %.5f", step, float(loss))
    net.eval()
    return losses


def score_pairs(net, pairs, cfg, device, condition_is_target: bool) -> dict:
    refs, conds = [], []
    for r in pairs.itertuples(index=False):
        with Image.open(_resolve_manifest_path(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        if condition_is_target:
            conds.append(refs[-1])
        else:
            with Image.open(_resolve_manifest_path(r.hairy_path)) as fh:
                conds.append(fh.convert("RGB"))
    outs, _ = clean_batch(net, conds, cfg, device=device, seed=42,
                          num_timesteps=STEPS, use_bf16=True)
    ps, floor = [], []
    for out, ref, hary in zip(outs, refs, conds):
        a = np.asarray(out.resize((SIZE, SIZE)).convert("RGB"), dtype=np.float64) / 255.0
        b = np.asarray(ref.resize((SIZE, SIZE)).convert("RGB"), dtype=np.float64) / 255.0
        h = np.asarray(hary.resize((SIZE, SIZE)).convert("RGB"), dtype=np.float64) / 255.0
        ps.append(psnr(a, b))
        floor.append(psnr(h, b))
    return {"psnr": float(np.mean(ps)), "floor": float(np.mean(floor)),
            "n": len(ps)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["1", "2"], required=True)
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    device = "cuda"
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    train_pairs = pairs[pairs["split"] == "train"].reset_index(drop=True)

    out: dict = {"gate": args.gate, "steps": STEPS,
                 "train_manifest_sha256": _manifest_hash(train_pairs)}

    if args.gate == "1":
        subset = train_pairs.head(8)
        net = build_net(cfg, device)
        losses = train_briefly(net, subset, 2000, cfg, device, copy_task=False)
        res = score_pairs(net, subset, cfg, device, condition_is_target=False)
        res["margin_db"] = res["psnr"] - res["floor"]
        res["passed"] = bool(res["margin_db"] >= GATE1_MARGIN)
        res["required_margin_db"] = GATE1_MARGIN
        res["final_loss"] = float(np.mean(losses[-100:]))
        out["result"] = res
        torch.save(net.state_dict(),
                   artifact_dir("checkpoints", "gate1_overfit") / "gen.pth")
        log.info("GATE1 psnr %.2f floor %.2f margin %+.2f -> %s",
                 res["psnr"], res["floor"], res["margin_db"],
                 "PASS" if res["passed"] else "FAIL")
    else:
        subset = train_pairs.head(8)
        net = build_net(cfg, device)
        losses = train_briefly(net, subset, 1000, cfg, device, copy_task=True)
        res = score_pairs(net, subset, cfg, device, condition_is_target=True)
        res["passed"] = bool(res["psnr"] >= GATE2_MIN_PSNR)
        res["required_psnr"] = GATE2_MIN_PSNR
        res["final_loss"] = float(np.mean(losses[-100:]))
        out["result"] = res
        torch.save(net.state_dict(),
                   artifact_dir("checkpoints", "gate2_copy") / "gen.pth")
        log.info("GATE2 copy psnr %.2f (need >= %.1f) -> %s",
                 res["psnr"], GATE2_MIN_PSNR,
                 "PASS" if res["passed"] else "FAIL")

    save_json(out, artifact_dir("logs") / f"trainer_gate{args.gate}.json")
    record_run(f"trainer_gate{args.gate}", gpu=device,
               seconds=0.0, passed=out["result"]["passed"])
    print(out["result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
