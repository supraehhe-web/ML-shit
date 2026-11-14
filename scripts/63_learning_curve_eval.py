"""Learning-curve diagnostics for one or more DM-AHR snapshots. Validation pairs only.

Per snapshot (rules fixed in reports/prereg_v5_client_asks.md section 3):
  (a) one-step epsilon error over t in {200, 500, 1000, 1500, 1900} (48 pairs, fixed epsilon);
  (b) condition-usage ablation (kept / zeroed / shuffled) at t in {200, 500, 1000, 1500};
  (c) partial-noise sampling from t0 = 600, 50 steps, on 50 pairs;
  (d) pure-noise sampling at 2,000 steps on 24 pairs.
The hairy-input floor is computed on the same pairs with the same scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.diffusion import dmahr_adapter as ad  # noqa: E402
from dm_ahr_vit.diffusion.train import _resolve_manifest_path  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
spec = importlib.util.spec_from_file_location("condusage", ROOT / "scripts" / "51_condition_usage.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)

ONE_STEP_T = (200, 500, 1000, 1500, 1900)
USAGE_T = (200, 500, 1000, 1500)
N_GRID, N_PARTIAL, N_PURE = 48, 50, 24
EPS_SEED = 20260101


def load_val(n: int):
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    pairs = pairs[pairs["split"] == "val"].head(n).reset_index(drop=True)
    assert set(pairs["split"]) == {"val"}
    refs, conds, hair = [], [], []
    for r in pairs.itertuples(index=False):
        refs.append(Image.open(_resolve_manifest_path(r.reference_path)).convert("RGB"))
        conds.append(Image.open(_resolve_manifest_path(r.hairy_path)).convert("RGB"))
        m = np.array(Image.open(_resolve_manifest_path(r.hair_mask_path)).convert("L"))
        m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST) > 127
        hair.append(cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return pairs, refs, conds, hair


def score(outs, refs, win: int) -> dict:
    ps, ss = [], []
    for o, ref in zip(outs, refs):
        a = np.asarray(o.convert("RGB").resize(ref.size, Image.Resampling.BICUBIC), np.float64) / 255.0
        b = np.asarray(ref, np.float64) / 255.0
        ps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
        ss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=win))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(ps)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="name=relative/path.pth,... under artifacts/checkpoints")
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    dev = "cuda"
    pairs, refs, conds, hair = load_val(max(N_GRID, N_PARTIAL, N_PURE))
    ids_hash = hashlib.sha256("".join(pairs["pair_id"].astype(str)).encode()).hexdigest() \
        if "pair_id" in pairs else hashlib.sha256(pairs.to_csv().encode()).hexdigest()
    floors = {"partial": score(conds[:N_PARTIAL], refs[:N_PARTIAL], win),
              "pure": score(conds[:N_PURE], refs[:N_PURE], win)}
    x0 = ad.preprocess(refs[:N_GRID], 128).to(dev)
    cond = ad.preprocess(conds[:N_GRID], 128).to(dev)
    hair_t = torch.from_numpy(np.stack(hair[:N_GRID])).to(dev)
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(EPS_SEED)).to(dev)
    out_dir = artifact_dir("logs", "learning_curve")
    ckroot = artifact_dir("checkpoints")

    for item in args.ckpts.split(","):
        name, rel = item.split("=", 1)
        path = ckroot / rel
        net = ad.load_model(cfg, path, device=dev)
        res = {"name": name, "checkpoint": rel, "checkpoint_sha256": sha256_file(path),
               "git_commit": args.commit, "val_pairs_hash": ids_hash, "floors": floors,
               "sha256": {"configs/dmahr.json": sha256_file("configs/dmahr.json"),
                          "scripts/63_learning_curve_eval.py": sha256_file(Path(__file__)),
                          "synthetic_pairs.csv": sha256_file(manifest_dir() / "synthetic_pairs.csv")}}
        with timed(f"lc_{name}", log) as tmr, torch.no_grad():
            res["one_step_eps_mae"] = {}
            for t in ONE_STEP_T:
                maps = torch.cat([cu.err_map(net, cond[i:i + 8], x0[i:i + 8], t, eps[i:i + 8])
                                  for i in range(0, N_GRID, 8)])
                res["one_step_eps_mae"][str(t)] = float(maps.mean())
            res["condition_usage"] = []
            for t in USAGE_T:
                e = cu.region_errors(net, cond, x0, t, eps, hair_t)
                row = {"t": t, **{f"{c}_{r}": e[c][r] for c in e for r in e[c]}}
                for r in ("hair", "other"):
                    row[f"zeroed_over_kept_{r}"] = e["zeroed"][r] / e["kept"][r]
                    row[f"shuffled_over_kept_{r}"] = e["shuffled"][r] / e["kept"][r]
                res["condition_usage"].append(row)
            o, _ = ad.clean_batch(net, conds[:N_PARTIAL], cfg, device=dev, seed=42, num_timesteps=50,
                                  use_bf16=True, respaced=True, start_t=600)
            res["partial_t0_600"] = {**score(o, refs[:N_PARTIAL], win), "floor": floors["partial"]}
            res["partial_t0_600"]["gain_db"] = res["partial_t0_600"]["psnr"] - floors["partial"]["psnr"]
            o, _ = ad.clean_batch(net, conds[:N_PURE], cfg, device=dev, seed=42, num_timesteps=None,
                                  use_bf16=True, respaced=False)
            res["pure_noise_2000"] = {**score(o, refs[:N_PURE], win), "floor": floors["pure"]}
            res["pure_noise_2000"]["gain_db"] = res["pure_noise_2000"]["psnr"] - floors["pure"]["psnr"]
        res["seconds"] = tmr["seconds"]
        save_json(res, out_dir / f"{name}.json")
        record_run("lc_eval", gpu="A100-40GB", seconds=tmr["seconds"], name=name)
        log.info("%s: one-step MAE %s | partial +%.2f dB | pure-noise %.2f dB (floor %.2f, %+.2f)", name,
                 {k: round(v, 4) for k, v in res["one_step_eps_mae"].items()},
                 res["partial_t0_600"]["gain_db"], res["pure_noise_2000"]["psnr"],
                 floors["pure"]["psnr"], res["pure_noise_2000"]["gain_db"])
        del net
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
