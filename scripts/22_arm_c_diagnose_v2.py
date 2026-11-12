"""Phase 2: separate "undertrained" from "trained wrongly" for our DM-AHR checkpoint.

Validation synthetic pairs only; no test image, label or metric is read. The
rules that turn these measurements into a branch are pre-registered (D22) and
implemented in dm_ahr_vit.diffusion.diagnostics.

Stages (each gated by --budget-usd; `decide` is CPU-only):
  mse          noise-prediction error, released vs ours, under upstream vs legacy
               (t/T) conditioning, identical noisy inputs
  score        sample released, ours (legacy trainer) and every fixed-trainer
               snapshot; PSNR/SSIM/LPIPS plus high-frequency, noise-sigma and
               power-spectrum measurements against the hairy input and reference
  warm_start   fine-tune the released checkpoint with our trainer, legacy and
               fixed; a loss jump flags a trainer bug
  overfit      can the fixed trainer overfit 8 pairs (sampled quality, not loss)
  decide       apply the pre-registered branch rule to whatever has been run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.budget import estimate_usd, gate  # noqa: E402
from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion import diagnostics as dg  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()
SEC_PER_IMG_STEP = 4.35 / 2000.0
SEC_PER_ITER = 0.055
LOGS = lambda name: artifact_dir("logs") / name  # noqa: E731


def paths():
    from dm_ahr_vit.diffusion.dmahr_adapter import CHECKPOINT_NAME
    return {"released": data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME,
            "ours": artifact_dir("checkpoints", "dmahr") / "best_gen.pth",
            "fixed_dir": artifact_dir("checkpoints", "dmahr_fixed")}


def val_pairs(n: int) -> pd.DataFrame:
    p = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    v = p[p["split"] == "val"].sort_values("pair_id").reset_index(drop=True)
    if v.empty:
        raise RuntimeError("no validation synthetic pairs")
    return v.head(n)


def load_images(df: pd.DataFrame):
    return ([Image.open(p).convert("RGB") for p in df["hairy_path"]],
            [Image.open(p).convert("RGB") for p in df["reference_path"]])


def sample(net, imgs, cfg, steps, bf16, device, bs=32):
    from dm_ahr_vit.diffusion.dmahr_adapter import clean_batch
    outs = []
    for s in range(0, len(imgs), bs):
        o, _ = clean_batch(net, imgs[s:s + bs], cfg, device=device, seed=s, num_timesteps=steps,
                           use_bf16=bf16)
        outs.extend(o)
    return outs


# ------------------------------------------------------------------ stage: mse
def stage_mse(args, cfg, device):
    import torch
    from torch.utils.data import DataLoader

    from dm_ahr_vit.diffusion.conditioning import fixed_level_error
    from dm_ahr_vit.diffusion.dmahr_adapter import load_model
    from dm_ahr_vit.diffusion.train import PairedDataset

    P = paths()
    ckpts = {"released": P["released"], "ours": P["ours"]}
    fixed_final = P["fixed_dir"] / "final_gen.pth"
    if fixed_final.exists():
        ckpts["ours_fixed"] = fixed_final
    df = val_pairs(args.pairs_n)
    loader = DataLoader(PairedDataset(df, int(cfg["native_resolution"])), batch_size=25, shuffle=False)
    ts = (200, 800, 1400, 1900)
    table, per_t = {}, []
    for name, path in ckpts.items():
        net = load_model(cfg, path, device=device).eval()
        for mode in ("upstream", "legacy"):
            gen = torch.Generator(device="cpu").manual_seed(0)
            vals = {t: [] for t in ts}
            with torch.no_grad():
                for batch in loader:
                    hr, cond = batch["HR"].to(device), batch["SR"].to(device)
                    for t in ts:
                        noise = torch.randn(hr.shape, generator=gen).to(device)
                        vals[t].append(fixed_level_error(net, hr, cond, t, noise, mode))
            for t in ts:
                per_t.append({"ckpt": name, "mode": mode, "t": t,
                              "mse": float(np.mean([v["mse"] for v in vals[t]])),
                              "l1": float(np.mean([v["l1"] for v in vals[t]]))})
            table.setdefault(name, {})[mode] = float(np.mean([r["mse"] for r in per_t
                                                               if r["ckpt"] == name and r["mode"] == mode]))
        del net
    out = {"n_pairs": len(df), "timesteps": list(ts), "mean_mse": table,
           "evidence": dg.mse_evidence(table), "per_timestep": per_t}
    if "ours_fixed" in table:
        out["fixed_trainer_prefers_upstream"] = bool(table["ours_fixed"]["upstream"] < table["ours_fixed"]["legacy"])
    save_json(out, LOGS("arm_c_v2_noise_mse.json"))
    print(json.dumps({"mean_mse": table, "evidence": out["evidence"]}, indent=1))


# ---------------------------------------------------------------- stage: score
def stage_score(args, cfg, device):
    from dm_ahr_vit.diffusion.dmahr_adapter import load_model

    P = paths()
    conds = [("released", P["released"]), ("ours_legacy_trainer", P["ours"])]
    snaps = sorted(P["fixed_dir"].glob("iter_*_gen.pth"), key=lambda p: int(p.stem.split("_")[1]))
    conds += [(f"fixed_{int(p.stem.split('_')[1])}", p) for p in snaps]
    df = val_pairs(args.score_n)
    hairy, refs = load_images(df)
    floor = [dg.psnr_ssim(h, r) for h, r in zip(hairy, refs)]
    floor_psnr, floor_ssim = float(np.mean([f["psnr"] for f in floor])), float(np.mean([f["ssim"] for f in floor]))
    hf_h = float(np.mean([dg.mean_abs_laplacian(h) for h in hairy]))
    hf_r = float(np.mean([dg.mean_abs_laplacian(r) for r in refs]))
    sig_h = float(np.mean([dg.noise_sigma_mad(h) for h in hairy]))
    sig_r = float(np.mean([dg.noise_sigma_mad(r) for r in refs]))
    spec_h = np.mean([dg.radial_power_spectrum(h)[1] for h in hairy], axis=0)
    spec_r = np.mean([dg.radial_power_spectrum(r)[1] for r in refs], axis=0)
    freqs = dg.radial_power_spectrum(hairy[0])[0]

    rows, spectra = [], {"freq": freqs.tolist(), "hairy": spec_h.tolist(), "reference": spec_r.tolist()}
    for name, path in conds:
        net = load_model(cfg, path, device=device)
        outs = sample(net, hairy, cfg, args.steps, args.bf16, device)
        m = [dg.psnr_ssim(o, r) for o, r in zip(outs, refs)]
        lp = dg.lpips_scores(outs, refs, device=device)
        hf = float(np.mean([dg.mean_abs_laplacian(o) for o in outs]))
        sig = float(np.mean([dg.noise_sigma_mad(o) for o in outs]))
        spec = np.mean([dg.radial_power_spectrum(o)[1] for o in outs], axis=0)
        spectra[name] = spec.tolist()
        rows.append({"condition": name, "steps": args.steps, "n": len(outs),
                     "psnr": float(np.mean([x["psnr"] for x in m])), "ssim": float(np.mean([x["ssim"] for x in m])),
                     "lpips": float(np.mean(lp)), "hf_energy": hf,
                     "hf_ratio_vs_hairy": hf / hf_h, "hf_ratio_vs_reference": hf / hf_r,
                     "noise_sigma_mad": sig, "sigma_ratio_vs_hairy": sig / sig_h,
                     "sigma_ratio_vs_reference": sig / sig_r,
                     "hf_power_fraction": float(np.mean([dg.high_freq_power_fraction(o) for o in outs]))})
        log.info("%s: PSNR %.2f SSIM %.3f hf %.4f (hairy %.4f ref %.4f)", name, rows[-1]["psnr"],
                 rows[-1]["ssim"], hf, hf_h, hf_r)
        del net
    ref_rows = {"hairy_input": {"psnr": floor_psnr, "ssim": floor_ssim, "hf_energy": hf_h, "noise_sigma_mad": sig_h},
                "reference": {"hf_energy": hf_r, "noise_sigma_mad": sig_r}}
    fixed = {int(r["condition"].split("_")[1]): r["psnr"] for r in rows if r["condition"].startswith("fixed_")}
    out = {"n_pairs": len(df), "steps": args.steps, "references": ref_rows, "conditions": rows,
           "spectra": spectra,
           "checkpoint_trend": dg.checkpoint_trend(fixed) if len(fixed) >= 2 else None}
    final = max(fixed) if fixed else None
    if final is not None:
        r = next(x for x in rows if x["condition"] == f"fixed_{final}")
        out["fixed_arm_validity"] = dg.arm_is_valid(r["psnr"], r["ssim"], floor_psnr, floor_ssim)
    save_json(out, LOGS("arm_c_v2_scores.json"))
    pd.DataFrame(rows).to_csv(LOGS("arm_c_v2_scores.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ----------------------------------------------------------- stage: warm_start
def stage_warm_start(args, cfg, device):
    import torch
    from torch.utils.data import DataLoader

    from dm_ahr_vit.diffusion.conditioning import noise_prediction_loss
    from dm_ahr_vit.diffusion.dmahr_adapter import load_model
    from dm_ahr_vit.diffusion.train import PairedDataset, train

    P = paths()
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    tr, va = pairs[pairs["split"] == "train"], pairs[pairs["split"] == "val"]
    res = {}
    for mode in ("legacy", "upstream"):
        net = load_model(cfg, P["released"], device=device).eval()
        loader = DataLoader(PairedDataset(tr, int(cfg["native_resolution"])), batch_size=4, shuffle=True)
        rng = np.random.default_rng(0)
        vals = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.pre_batches:
                    break
                vals.append(float(noise_prediction_loss(net, batch["HR"].to(device), batch["SR"].to(device),
                                                        mode=mode, rng=rng)))
        pre = float(np.mean(vals))
        del net
        out_dir = artifact_dir("checkpoints", f"warmstart_{mode}")
        train(cfg, tr, va, out_dir, device=device, conditioning=mode, iterations=args.warm_iters,
              init_checkpoint=P["released"], save_every=None, grad_clip=1.0 if mode == "legacy" else None)
        trace = json.loads((out_dir / "train_log.json").read_text())["trace"]
        last = float(np.mean([t["loss"] for t in trace[-100:]]))
        res[mode] = dg.warm_start_jump(pre, last)
        res[mode]["first100_mean"] = float(np.mean([t["loss"] for t in trace[:100]]))
    save_json({"iterations": args.warm_iters, "pre_batches": args.pre_batches, **res},
              LOGS("arm_c_v2_warm_start.json"))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------- stage: overfit
def stage_overfit(args, cfg, device):
    from dm_ahr_vit.diffusion.dmahr_adapter import load_model
    from dm_ahr_vit.diffusion.train import train

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    tr = pairs[pairs["split"] == "train"].sort_values("pair_id").head(8).reset_index(drop=True)
    va = pairs[pairs["split"] == "val"].sort_values("pair_id").head(8)
    cfg = json.loads(json.dumps(cfg))
    cfg["train"]["overfit_pairs"] = 8
    out_dir = artifact_dir("checkpoints", "dmahr_overfit_v2")
    train(cfg, tr, va, out_dir, device=device, overfit=True, conditioning="upstream",
          iterations=args.overfit_iters, save_every=None)
    net = load_model(cfg, out_dir / "final_gen.pth", device=device)
    hairy, refs = load_images(tr)
    outs = sample(net, hairy, cfg, args.steps, args.bf16, device)
    m = [dg.psnr_ssim(o, r) for o, r in zip(outs, refs)]
    fl = [dg.psnr_ssim(h, r) for h, r in zip(hairy, refs)]
    res = dg.overfit_pass(float(np.mean([x["psnr"] for x in m])), float(np.mean([x["psnr"] for x in fl])))
    res.update({"iterations": args.overfit_iters, "n_pairs": 8, "steps": args.steps,
                "mean_ssim": float(np.mean([x["ssim"] for x in m])),
                "hairy_ssim": float(np.mean([x["ssim"] for x in fl]))})
    save_json(res, LOGS("arm_c_v2_overfit.json"))
    print(json.dumps(res, indent=1))


# -------------------------------------------------------------- stage: decide
def stage_decide(_args=None, _cfg=None, _device=None):
    def load(name):
        p = LOGS(name)
        return json.loads(p.read_text()) if p.exists() else None

    mse, sc, ws, ov = (load("arm_c_v2_noise_mse.json"), load("arm_c_v2_scores.json"),
                       load("arm_c_v2_warm_start.json"), load("arm_c_v2_overfit.json"))
    branch = dg.phase2_branch(mse["evidence"] if mse else None,
                              sc["checkpoint_trend"] if sc and sc.get("checkpoint_trend") else None,
                              ws, ov)
    out = {"branch": branch, "inputs_present": {"mse": bool(mse), "score": bool(sc),
                                                "warm_start": bool(ws), "overfit": bool(ov)},
           "static_audit": {"ema": "upstream marks EMA 'not used now'; no EMA code upstream or in our trainer",
                            "ema_bug_branch_a": "ruled out by audit (no EMA weights exist)"},
           "validity_of_fixed_arm": sc.get("fixed_arm_validity") if sc else None}
    save_json(out, LOGS("arm_c_diagnosis_v2.json"))
    print(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["mse", "score", "warm_start", "overfit", "decide", "all"])
    ap.add_argument("--pairs-n", type=int, default=100)
    ap.add_argument("--score-n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warm-iters", type=int, default=500)
    ap.add_argument("--pre-batches", type=int, default=200)
    ap.add_argument("--overfit-iters", type=int, default=2000)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.stage == "decide":
        stage_decide()
        return 0

    n_fixed = 4  # snapshots at 5k/10k/15k/20k are expected from the fixed retrain
    f = 0.5 if args.bf16 else 1.0
    est = {
        "mse": 90.0,
        "score": (2 + n_fixed) * args.score_n * args.steps * SEC_PER_IMG_STEP * f + 120.0,
        "warm_start": 2 * args.warm_iters * SEC_PER_ITER + 180.0,
        "overfit": args.overfit_iters * SEC_PER_ITER + 8 * args.steps * SEC_PER_IMG_STEP * f + 60.0,
    }
    stages = list(est) if args.stage == "all" else [args.stage]
    total = sum(est[s] for s in stages)
    if args.dry_run:
        print(json.dumps({"seconds": {s: round(est[s]) for s in stages}, "total_seconds": round(total),
                          "estimated_usd": round(estimate_usd(total, args.gpu), 2), "gpu": args.gpu}, indent=1))
        return 0
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("this job needs a GPU")
    gate("arm_c_diagnose_v2:" + args.stage, 1, total, args.gpu, args.budget_usd)
    cfg = load_config("configs/dmahr.json")
    t0 = time.time()
    for s in stages:
        {"mse": stage_mse, "score": stage_score, "warm_start": stage_warm_start,
         "overfit": stage_overfit}[s](args, cfg, "cuda")
    secs = time.time() - t0
    record_run("arm_c_diagnose_v2", gpu=args.gpu, seconds=secs, stage=args.stage,
               estimated_usd=round(estimate_usd(secs, args.gpu), 2))
    stage_decide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
