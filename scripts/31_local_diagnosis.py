"""CPU-only Arm C diagnostics. No Modal, validation pairs only.

Commands
  profile  one-step eps error / x0 PSNR / condition ablations over a t grid, three models
  sample   full or partial reverse chain on one validation pair, saves intermediate states
  analyze  error anatomy of saved samples (tone vs texture, low vs high frequency)

Every JSON written records git commit, config hash, checkpoint sha256 and manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import load_config  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import load_model, preprocess  # noqa: E402

LEVELS = (50, 200, 500, 1000, 1500, 1900)
OUT = ROOT / "artifacts" / "diagnosis"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def provenance(ckpts: dict[str, str], manifest: Path, val_ids: list[str]) -> dict:
    cfg_path = ROOT / "configs" / "dmahr.json"
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_code": bool(git("status", "--porcelain", "--", "scripts", "src", "configs")),
        "config_sha256": sha256(cfg_path),
        "manifest_sha256": sha256(manifest),
        "val_pair_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "checkpoint_sha256": {k: sha256(Path(v)) for k, v in ckpts.items()},
        "torch": torch.__version__,
    }


def load_val(manifest: Path, data_root: Path, n: int, start: int = 0):
    pairs = pd.read_csv(manifest)
    val = pairs[pairs["split"] == "val"].iloc[start:start + n].reset_index(drop=True)

    def fix(p: str) -> Path:
        return data_root / p.split("/data/", 1)[1]

    refs = [Image.open(fix(p)).convert("RGB") for p in val["reference_path"]]
    hairy = [Image.open(fix(p)).convert("RGB") for p in val["hairy_path"]]
    return val, refs, hairy


def to01(x: torch.Tensor) -> np.ndarray:
    return ((x.detach().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    return float(psnr_fn(b, a, data_range=1.0))


def level_for(net, i: int, enc: str) -> float:
    if enc == "sampler":
        return float(net.sqrt_alphas_cumprod_prev[i + 1])
    return (i + 1) / net.num_timesteps


@torch.no_grad()
def cmd_profile(args) -> None:
    cfg = load_config(str(ROOT / "configs" / "dmahr.json"))
    torch.set_num_threads(args.threads)
    manifest = Path(args.manifest)
    val, refs, hairy = load_val(manifest, Path(args.data_root), args.pairs)
    size = int(cfg["native_resolution"])
    x0 = preprocess(refs, size)
    cond = preprocess(hairy, size)
    n = x0.shape[0]
    g = torch.Generator().manual_seed(20260101)
    eps = torch.randn((len(LEVELS),) + tuple(x0.shape), generator=g)
    hairy_floor = float(np.mean([psnr01(a, b) for a, b in zip(to01(cond), to01(x0))]))

    models = [("released", args.released, "sampler"),
              ("new_fixed_20k", args.new, "sampler"),
              ("old_defective_own_encoding_tT", args.old, "tT"),
              ("old_defective_sampler_encoding", args.old, "sampler")]
    if args.only:
        models = [m for m in models if m[0] in args.only]
    result = {"pairs": int(n), "levels": list(LEVELS), "hairy_floor_psnr": hairy_floor, "models": {}}
    for name, ckpt, enc in models:
        net = load_model(cfg, Path(ckpt), device="cpu").eval()
        rows = []
        for k, t in enumerate(LEVELS):
            i = t
            a = float(net.sqrt_alphas_cumprod_prev[i + 1])
            x_t = a * x0 + (1 - a**2) ** 0.5 * eps[k]
            lvl_val = level_for(net, i, enc)
            kept, zero, shuf, xh = [], [], [], []
            for s in range(0, n, 4):
                sl = slice(s, s + 4)
                lvl = torch.full((x_t[sl].shape[0], 1), lvl_val)
                e_k = net.denoise_fn(torch.cat([cond[sl], x_t[sl]], 1), lvl)
                e_z = net.denoise_fn(torch.cat([torch.zeros_like(cond[sl]), x_t[sl]], 1), lvl)
                shifted = torch.roll(cond, 1, dims=0)[sl]
                e_s = net.denoise_fn(torch.cat([shifted, x_t[sl]], 1), lvl)
                kept.append((e_k - eps[k][sl]).abs().mean((1, 2, 3)))
                zero.append((e_z - eps[k][sl]).abs().mean((1, 2, 3)))
                shuf.append((e_s - eps[k][sl]).abs().mean((1, 2, 3)))
                xh.append((net.sqrt_recip_alphas_cumprod[i] * x_t[sl]
                           - net.sqrt_recipm1_alphas_cumprod[i] * e_k).clamp(-1, 1))
            kept, zero, shuf = (torch.cat(v) for v in (kept, zero, shuf))
            x0_hat = torch.cat(xh)
            noisy = (x_t / a).clamp(-1, 1)
            rows.append({
                "t": t,
                "eps_l1_kept": float(kept.mean()),
                "eps_l1_zero_cond": float(zero.mean()),
                "eps_l1_shuffled_cond": float(shuf.mean()),
                "ratio_zero_over_kept": float(zero.mean() / kept.mean()),
                "ratio_shuffled_over_kept": float(shuf.mean() / kept.mean()),
                "x0_psnr_onestep": float(np.mean([psnr01(p, r) for p, r in zip(to01(x0_hat), to01(x0))])),
                "noisy_input_psnr": float(np.mean([psnr01(p, r) for p, r in zip(to01(noisy), to01(x0))])),
            })
            print(name, t, {k2: round(v, 4) for k2, v in rows[-1].items()}, flush=True)
        result["models"][name] = rows
        del net
    ckpts = {"released": args.released, "new_fixed_20k": args.new, "old_defective": args.old}
    result["provenance"] = provenance(ckpts, manifest, val["pair_id"].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out_name).write_text(json.dumps(result, indent=1))


@torch.no_grad()
def cmd_sample(args) -> None:
    cfg = load_config(str(ROOT / "configs" / "dmahr.json"))
    torch.set_num_threads(args.threads)
    manifest = Path(args.manifest)
    val, refs, hairy = load_val(manifest, Path(args.data_root), 1, start=args.pair)
    size = int(cfg["native_resolution"])
    x0 = preprocess(refs, size)
    cond = preprocess(hairy, size)
    net = load_model(cfg, Path(args.ckpt), device="cpu").eval()
    # Hybrid chain: --ckpt-high handles loop indices >= --switch-t, --ckpt the rest.
    net_hi = load_model(cfg, Path(args.ckpt_high), device="cpu").eval() if args.ckpt_high else None
    T = net.num_timesteps
    torch.manual_seed(args.seed)
    if args.start_t is None:
        img = torch.randn(x0.shape)
        i0 = T - 1
    else:
        a = float(net.sqrt_alphas_cumprod_prev[args.start_t + 1])
        img = a * x0 + (1 - a**2) ** 0.5 * torch.randn(x0.shape)
        i0 = args.start_t
    keep = {1500, 1000, 500, 200, 50}
    states = {}
    t0 = time.time()
    for i in range(i0, -1, -1):
        step_net = net_hi if (net_hi is not None and i >= args.switch_t) else net
        img = step_net.p_sample(img, i, condition_x=cond)
        if i in keep:
            states[f"t{i}"] = to01(img)[0]
        if i % 100 == 0:
            print(f"step {i0 - i + 1}/{i0 + 1} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.name}_pair{args.pair}" + (f"_from{args.start_t}" if args.start_t else "")
    if net_hi is not None:
        tag += f"_hi{args.switch_t}"
    np.savez(OUT / f"sample_{tag}.npz", final=to01(img)[0], ref=to01(x0)[0], hairy=to01(cond)[0], **states)
    meta = {"name": args.name, "pair_id": val["pair_id"][0], "pair_index": args.pair,
            "start_t": args.start_t, "seed": args.seed, "seconds": time.time() - t0,
            "switch_t": args.switch_t if net_hi is not None else None,
            "provenance": provenance({args.name: args.ckpt, **({"high": args.ckpt_high} if net_hi is not None else {})},
                                     manifest, val["pair_id"].tolist())}
    (OUT / f"sample_{tag}.json").write_text(json.dumps(meta, indent=1))


def anatomy(o: np.ndarray, r: np.ndarray) -> dict:
    mean_err = (o.reshape(-1, 3).mean(0) - r.reshape(-1, 3).mean(0))
    matched = np.clip(o - o.reshape(-1, 3).mean(0) + r.reshape(-1, 3).mean(0), 0, 1)
    fitted = np.empty_like(o)
    for c in range(3):
        A = np.stack([o[..., c].ravel(), np.ones(o[..., c].size)], 1)
        coef, *_ = np.linalg.lstsq(A, r[..., c].ravel(), rcond=None)
        fitted[..., c] = np.clip(o[..., c] * coef[0] + coef[1], 0, 1)
    lo = lambda x: np.stack([gaussian_filter(x[..., c], 6) for c in range(3)], -1)  # noqa: E731
    low_err = float(np.sqrt(np.mean((lo(o) - lo(r)) ** 2)))
    high_err = float(np.sqrt(np.mean(((o - lo(o)) - (r - lo(r))) ** 2)))
    gray = lambda x: x.mean(-1)  # noqa: E731
    return {
        "psnr": psnr01(o, r),
        "ssim": float(ssim_fn(r, o, data_range=1.0, channel_axis=2, win_size=7)),
        "channel_mean_error_rgb": [float(v) for v in mean_err],
        "psnr_after_mean_match": psnr01(matched, r),
        "psnr_after_gain_bias_fit": psnr01(fitted, r),
        "rmse_low_freq": low_err,
        "rmse_high_freq": high_err,
        "laplacian_energy_ratio_vs_ref": float(np.abs(laplace(gray(o))).mean()
                                              / np.abs(laplace(gray(r))).mean()),
    }


def cmd_analyze(args) -> None:
    out = {}
    for f in sorted(OUT.glob("sample_*.npz")):
        d = np.load(f)
        out[f.stem] = {"output": anatomy(d["final"], d["ref"]), "hairy_input": anatomy(d["hairy"], d["ref"]),
                       "intermediate": {k: anatomy(d[k], d["ref"]) for k in d.files if k.startswith("t")}}
        print(f.stem, json.dumps(out[f.stem]["output"], indent=1), flush=True)
    (OUT / "anatomy.json").write_text(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("profile", "sample"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", default="/root/local/diag/man/synthetic_pairs.csv")
        p.add_argument("--data-root", default="/root/local/diag/data")
        p.add_argument("--threads", type=int, default=1)
    sub.add_parser("analyze")
    sub.choices["profile"].add_argument("--pairs", type=int, default=12)
    sub.choices["profile"].add_argument("--released", default="/root/local/diag/ckpt/released.pth")
    sub.choices["profile"].add_argument("--new", default="/root/local/diag/ckpt/new_best.pth")
    sub.choices["profile"].add_argument("--old", default="/root/local/diag/ckpt/old_defective.pth")
    sub.choices["profile"].add_argument("--only", nargs="*", default=None,
                                        help="restrict to these model names")
    sub.choices["profile"].add_argument("--out-name", default="profile.json")
    s = sub.choices["sample"]
    s.add_argument("--name", required=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--pair", type=int, default=0)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--start-t", type=int, default=None)
    s.add_argument("--ckpt-high", default=None)
    s.add_argument("--switch-t", type=int, default=1500)
    args = ap.parse_args()
    {"profile": cmd_profile, "sample": cmd_sample, "analyze": cmd_analyze}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
