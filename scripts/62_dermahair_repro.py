"""Reproduction check against the DM-AHR paper on DERMAHAIR, using the released checkpoint.

Rule fixed in reports/prereg_v5_client_asks.md section 2: reproduced = within 2 dB of the paper's
41.04 dB AND above the hairy-input floor; otherwise "not reproduced". The paper's number is
context, not a target. The paper's "T = 10" is run under both readings that the code allows:
10 respaced (strided) steps over the trained 2,000-step schedule, and a 10-step schedule rebuilt
the way the released config's `n_timestep` would (truncated). T = 2,000 is the released setting.

Metrics at 128 px (upstream convention: ground truth and output both at r_resolution 128), with
the upstream `core.metrics` PSNR/SSIM as primary, plus skimage at 128 and at 256 and LPIPS (alex).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.diffusion import dmahr_adapter as ad  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
PAPER = {"psnr": 41.04, "ssim": 0.9656, "source": "Benjdira et al., Cancers 2024;16(17):2947, DM-AHR(SS)"}
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
BATCH = 64


def load_pairs(root: Path) -> list[dict]:
    pairs = []
    for c in CLASSES:
        for p in sorted((root / "Data_Skin_with_Hair" / c).glob("*.jpg")):
            q = root / "Data_Skin_without_Hair" / c / p.name
            if q.exists():
                pairs.append({"cls": c, "name": p.name, "hairy": p, "clean": q})
    return pairs


def dataset_hash(pairs: list[dict]) -> str:
    h = hashlib.sha256()
    for p in pairs:
        for k in ("hairy", "clean"):
            h.update(p[k].name.encode() + str(p[k].stat().st_size).encode())
    return h.hexdigest()


def metrics_for(outs: list[np.ndarray], gts: list[np.ndarray], gts256: list[np.ndarray],
                lp_net, Metrics, device: str) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for o, g, g256 in zip(outs, gts, gts256):
            o256 = np.asarray(Image.fromarray(o).resize((256, 256), Image.Resampling.BICUBIC))
            t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
            rows.append({
                "psnr_up": Metrics.calculate_psnr(o, g), "ssim_up": Metrics.calculate_ssim(o, g),
                "psnr_sk": peak_signal_noise_ratio(g, o, data_range=255),
                "ssim_sk": structural_similarity(g, o, data_range=255, channel_axis=2, win_size=7),
                "psnr_256": peak_signal_noise_ratio(g256, o256, data_range=255),
                "ssim_256": structural_similarity(g256, o256, data_range=255, channel_axis=2, win_size=7),
                "lpips": float(lp_net(t(o), t(g)).mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp32-check", type=int, default=64)
    args = ap.parse_args()
    import lpips  # noqa: PLC0415

    ad._ensure_path()
    import core.metrics as Metrics  # noqa: PLC0415

    cfg = load_config("configs/dmahr.json")
    device = "cuda"
    root = data_root() / "raw" / "dermahair"
    pairs = load_pairs(root)
    if args.limit:
        pairs = pairs[:args.limit]
    log.info("%d DERMAHAIR pairs", len(pairs))
    hairy = [Image.open(p["hairy"]).convert("RGB") for p in pairs]
    clean = [Image.open(p["clean"]).convert("RGB") for p in pairs]
    gts = [np.asarray(c.resize((128, 128), Image.Resampling.BICUBIC)) for c in clean]
    gts256 = [np.asarray(c) for c in clean]
    floor_in = [np.asarray(h.resize((128, 128), Image.Resampling.BICUBIC)) for h in hairy]
    lp_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ckpt = data_root() / "raw" / "dmahr_weights" / ad.CHECKPOINT_NAME

    results, per_pair = {}, []
    floor = metrics_for(floor_in, gts, gts256, lp_net, Metrics, device)
    # 3 of the 707 pairs have identical hairy and hairless images (PSNR undefined): excluded from every
    # mean below, for the floor and for each setting alike, and counted.
    ok = np.isfinite(floor["psnr_up"]).to_numpy()
    results["n_pairs_excluded_identical"] = int((~ok).sum())
    results["hairy_input_floor"] = {k: float(floor[k][ok].mean()) for k in floor.columns}
    results["hairy_input_floor"]["psnr_up_median"] = float(floor["psnr_up"][ok].median())
    fl = floor.copy()
    fl.insert(0, "setting", "hairy_input_floor")
    fl.insert(1, "cls", [p["cls"] for p in pairs])
    fl.insert(2, "name", [p["name"] for p in pairs])
    per_pair.append(fl)

    settings = [("respaced", 10, True), ("respaced", 25, True), ("respaced", 50, True),
                ("full", 2000, True), ("truncated", 10, True)]
    with timed("dermahair_repro", log) as tmr:
        net = ad.load_model(cfg, ckpt, device=device)
        for mode, steps, bf16 in settings:
            if mode == "truncated":
                net = ad.load_model(cfg, ckpt, device=device)
            outs = []
            for i in range(0, len(hairy), BATCH):
                o, _ = ad.clean_batch(net, hairy[i:i + BATCH], cfg, device=device, seed=42 + i,
                                      num_timesteps=None if mode == "full" else steps,
                                      use_bf16=bf16, respaced=(mode == "respaced"))
                outs += [np.asarray(x.convert("RGB")) for x in o]
            df = metrics_for(outs, gts, gts256, lp_net, Metrics, device)
            key = f"{mode}_{steps}{'_bf16' if bf16 else ''}"
            full_df = df
            df = df[ok]
            results[key] = {k: float(df[k].mean()) for k in df.columns}
            results[key].update({f"{k}_sd": float(df[k].std()) for k in df.columns})
            results[key]["n"] = len(df)
            results[key]["above_floor_psnr"] = bool(results[key]["psnr_up"] > results["hairy_input_floor"]["psnr_up"])
            results[key]["within_2db_of_paper"] = bool(abs(results[key]["psnr_up"] - PAPER["psnr"]) <= 2.0)
            results[key]["verdict"] = ("reproduced" if results[key]["above_floor_psnr"]
                                       and results[key]["within_2db_of_paper"] else "not reproduced")
            d = full_df.copy()
            d.insert(0, "setting", key)
            d.insert(1, "cls", [p["cls"] for p in pairs])
            d.insert(2, "name", [p["name"] for p in pairs])
            per_pair.append(d)
            log.info("%-18s PSNR(up) %.2f SSIM(up) %.4f LPIPS %.4f | floor %.2f | %s", key,
                     results[key]["psnr_up"], results[key]["ssim_up"], results[key]["lpips"],
                     results["hairy_input_floor"]["psnr_up"], results[key]["verdict"])
        if args.fp32_check:
            net = ad.load_model(cfg, ckpt, device=device)
            n = min(args.fp32_check, len(hairy))
            o, _ = ad.clean_batch(net, hairy[:n], cfg, device=device, seed=42, num_timesteps=None,
                                  use_bf16=False, respaced=False)
            df = metrics_for([np.asarray(x.convert("RGB")) for x in o], gts[:n], gts256[:n], lp_net,
                             Metrics, device)
            sub = per_pair[4][per_pair[4]["setting"] == "full_2000_bf16"].head(n)
            results["fp32_check_full_2000"] = {
                "n": n, "psnr_up_fp32": float(df["psnr_up"].mean()),
                "psnr_up_bf16_same_pairs": float(sub["psnr_up"].mean())}
    out_dir = artifact_dir("logs")
    pd.concat(per_pair).to_csv(out_dir / "dermahair_repro_pairs.csv", index=False)
    prov = {"paper": PAPER, "n_pairs": len(pairs), "dataset_sha256_of_names_and_sizes": dataset_hash(pairs),
            "dataset_source": "kaggle riotulab/skin-cancer-hair-removal v1, anonymous public download",
            "checkpoint": ad.CHECKPOINT_NAME, "checkpoint_sha256": sha256_file(ckpt),
            "git_commit": args.commit, "seconds": tmr["seconds"],
            "metric_note": "psnr_up/ssim_up = upstream core.metrics at 128 px; sk = skimage at 128; 256 = output bicubic-upsampled vs the 256 px hairless image",
            "sha256": {"configs/dmahr.json": sha256_file("configs/dmahr.json"),
                       "scripts/62_dermahair_repro.py": sha256_file(Path(__file__))}}
    save_json({"provenance": prov, "results": results}, out_dir / "dermahair_repro.json")
    record_run("dermahair_repro", gpu="A100-40GB", seconds=tmr["seconds"], pairs=len(pairs))
    print(pd.DataFrame(results).T[["psnr_up", "ssim_up", "lpips", "verdict"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
