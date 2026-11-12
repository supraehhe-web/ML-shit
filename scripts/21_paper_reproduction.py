"""Phase 1: reproduce the DM-AHR paper's numbers with the released checkpoint on its own data.

The released checkpoint is run through *our* adapter on DERMAHAIR pairs. For each
setting it reports PSNR / SSIM / LPIPS for three things:
  floor    hairy input vs hairless reference (no model)
  identity hairless reference through the full inference path vs itself
  output   hairy input through the model vs hairless reference

The rule is pre-registered in reports/decision_log.md (D22): mean output PSNR at
128 px / 2,000 steps within 2 dB of 41.04 -> adapter validated, otherwise
"paper numbers not reproduced".

Data: the Kaggle dataset when KAGGLE_USERNAME/KAGGLE_KEY are present, otherwise
the akiec subset of DERMAHAIR vendored with the upstream repo (101 pairs, one
class, 256 px). The source is written into the result so a partial run cannot be
mistaken for the full 700-image test set.

Throughput is re-measured on `--bench-n` images before anything large runs, and
the job refuses to start if its estimate exceeds `--budget-usd`.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.budget import estimate_usd, gate  # noqa: E402
from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.data import dermahair  # noqa: E402
from dm_ahr_vit.diffusion import diagnostics as dg  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()
ROOT = Path(__file__).resolve().parents[1]
VENDORED_ZIP = ROOT / "third_party" / "dmahr" / "dataset" / "akiec_16_128.zip"
SEC_PER_IMG_STEP_128 = 4.35 / 2000.0   # measured, A100-80GB fp32 batch 32 (compute_plan.md)
ASSUMED_RES_SCALING = 4.0              # 256 px has 4x the pixels; the bench replaces this


def plan_units(n_pairs: int, identity_n: int, steps: list[int], resolutions: list[int]) -> list[dict]:
    return [{"res": r, "steps": s, "images": n_pairs + identity_n,
             "image_steps": (n_pairs + identity_n) * s} for r in resolutions for s in steps]


def load_pairs(source: str, n: int) -> tuple[str, list[dict]]:
    """Return (source label, pairs). Pairs are sorted by key and truncated to n."""
    use_kaggle = source == "kaggle" or (source == "auto" and dermahair.kaggle_credentials_present())
    pairs: list[dict] = []
    if use_kaggle:
        dest = data_root() / "raw" / "dermahair_kaggle"
        info = dermahair.attempt_download(dest)
        if info["status"] != "ok":
            raise SystemExit(f"Kaggle download blocked: {info}")
        with_dir = next((p for p in dest.rglob("*") if p.is_dir() and p.name.lower() == "data_skin_with_hair"), None)
        without = next((p for p in dest.rglob("*") if p.is_dir() and p.name.lower() == "data_skin_without_hair"), None)
        if with_dir is None or without is None:
            raise SystemExit(f"expected Data_Skin_with_Hair / Data_Skin_without_Hair under {dest}; "
                             f"found {sorted(x.name for x in dest.iterdir())[:20]}")

        def index(root: Path) -> dict:
            return {str(p.relative_to(root).with_suffix("")): p for p in root.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        a, b = index(with_dir), index(without)
        keys = sorted(set(a) & set(b))
        if len(keys) < 0.5 * min(len(a), len(b)):
            raise SystemExit(f"could only match {len(keys)} of {len(a)}/{len(b)} files by relative path; "
                             "refusing to guess a pairing")
        for k in keys[:n]:
            pairs.append({"key": k, "hairy": Image.open(a[k]).convert("RGB"),
                          "ref": Image.open(b[k]).convert("RGB")})
        return f"kaggle:{dermahair.KAGGLE_SLUG} ({len(keys)} matched pairs)", pairs
    with zipfile.ZipFile(VENDORED_ZIP) as zf:
        names = sorted(n_ for n_ in zf.namelist()
                       if n_.startswith("akiec_16_128/hr_128/") and n_.endswith(".jpg"))
        for nm in names[:n]:
            stem = nm.split("/")[-1]
            pairs.append({
                "key": stem,
                "ref": Image.open(io.BytesIO(zf.read(nm))).convert("RGB"),
                "hairy": Image.open(io.BytesIO(zf.read(f"akiec_16_128/sr_16_128/{stem}"))).convert("RGB"),
            })
    return f"vendored DERMAHAIR akiec subset only ({len(names)} pairs; partial: one class)", pairs


def run_model(net, imgs: list, cfg: dict, res: int, steps: int, bs: int, bf16: bool, device: str):
    from dm_ahr_vit.diffusion.dmahr_adapter import clean_batch

    c = dict(cfg, native_resolution=res, condition_resolution=res)
    outs = []
    for s in range(0, len(imgs), bs):
        o, _ = clean_batch(net, imgs[s:s + bs], c, device=device, seed=s,
                           num_timesteps=steps, use_bf16=bf16)
        outs.extend(o)
    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "kaggle", "vendored_akiec"], default="auto")
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--identity-n", type=int, default=50)
    ap.add_argument("--steps", default="2000,1000")
    ap.add_argument("--resolutions", default="128", help="128 is the rule; add 256 as the native-size diagnostic")
    ap.add_argument("--bench-n", type=int, default=20)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the cost plan and exit (no GPU)")
    ap.add_argument("--tag", default="paper_reproduction")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    resolutions = [int(r) for r in args.resolutions.split(",")]
    units = plan_units(args.n_pairs, args.identity_n, steps, resolutions)
    scale = {128: 1.0, 256: ASSUMED_RES_SCALING}
    est_sec = sum(u["image_steps"] * SEC_PER_IMG_STEP_128 * scale.get(u["res"], 1.0) for u in units)
    if args.bf16:
        est_sec *= 0.5
    if args.dry_run:
        print(json.dumps({"units": units, "estimated_seconds_before_bench": round(est_sec),
                          "estimated_usd": round(estimate_usd(est_sec, args.gpu), 2), "gpu": args.gpu,
                          "note": "256 px scaling is an assumption until the on-GPU bench replaces it"}, indent=1))
        return 0

    import torch

    from dm_ahr_vit.diffusion.dmahr_adapter import CHECKPOINT_NAME, load_model

    cfg = load_config("configs/dmahr.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("this job needs a GPU")
    ckpt = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    label, pairs = load_pairs(args.source, args.n_pairs)
    log.info("pairs: %s -> using %d", label, len(pairs))
    net = load_model(cfg, ckpt, device=device)

    # --- bench: measured seconds per image-step at each resolution, 20 images, 100 steps
    bench_imgs = [p["hairy"] for p in pairs[:args.bench_n]]
    per_step: dict[int, float] = {}
    for res in resolutions:
        run_model(net, bench_imgs[:2], cfg, res, 5, 2, args.bf16, device)  # warm-up
        t0 = time.time()
        run_model(net, bench_imgs, cfg, res, 100, 32 if res <= 128 else 8, args.bf16, device)
        torch.cuda.synchronize()
        per_step[res] = (time.time() - t0) / (len(bench_imgs) * 100)
        log.info("bench res %d: %.5f s per image-step (%d images)", res, per_step[res], len(bench_imgs))
    total_sec = sum(u["image_steps"] * per_step[u["res"]] for u in units)
    gate("paper_reproduction", 1, total_sec, args.gpu, args.budget_usd)

    rows, summary = [], {}
    started = time.time()
    n_id = min(args.identity_n, len(pairs))
    for res in resolutions:
        hairy = [p["hairy"] for p in pairs]
        refs = [p["ref"] for p in pairs]
        floor = [dg.psnr_ssim(h, r, size=res) for h, r in zip(hairy, refs)]
        floor_lp = dg.lpips_scores(hairy, refs, size=res, device=device)
        for st in steps:
            bs = 32 if res <= 128 else 8
            outs = run_model(net, hairy, cfg, res, st, bs, args.bf16, device)
            ids = run_model(net, refs[:n_id], cfg, res, st, bs, args.bf16, device)
            out_m = [dg.psnr_ssim(o, r, size=res) for o, r in zip(outs, refs)]
            id_m = [dg.psnr_ssim(o, r, size=res) for o, r in zip(ids, refs[:n_id])]
            out_lp = dg.lpips_scores(outs, refs, size=res, device=device)
            id_lp = dg.lpips_scores(ids, refs[:n_id], size=res, device=device)
            for i, p in enumerate(pairs):
                rows.append({"res": res, "steps": st, "key": p["key"], "kind": "output",
                             **out_m[i], "lpips": out_lp[i], "floor_psnr": floor[i]["psnr"],
                             "floor_ssim": floor[i]["ssim"], "floor_lpips": floor_lp[i]})
            for i in range(n_id):
                rows.append({"res": res, "steps": st, "key": pairs[i]["key"], "kind": "identity",
                             **id_m[i], "lpips": id_lp[i]})
            mean = lambda xs, k: float(np.mean([x[k] for x in xs]))  # noqa: E731
            summary[f"res{res}_steps{st}"] = {
                "res": res, "steps": st, "n_output": len(out_m), "n_identity": len(id_m),
                "output": {"psnr": mean(out_m, "psnr"), "ssim": mean(out_m, "ssim"), "lpips": float(np.mean(out_lp))},
                "identity": {"psnr": mean(id_m, "psnr"), "ssim": mean(id_m, "ssim"), "lpips": float(np.mean(id_lp))},
                "floor": {"psnr": mean(floor, "psnr"), "ssim": mean(floor, "ssim"), "lpips": float(np.mean(floor_lp))},
            }
            log.info("res %d steps %d: output PSNR %.2f SSIM %.3f (floor %.2f)", res, st,
                     summary[f"res{res}_steps{st}"]["output"]["psnr"],
                     summary[f"res{res}_steps{st}"]["output"]["ssim"],
                     summary[f"res{res}_steps{st}"]["floor"]["psnr"])

    primary = summary.get("res128_steps2000")
    result = {
        "source": label, "n_pairs": len(pairs), "identity_n": n_id, "bf16": args.bf16,
        "checkpoint": ckpt.name, "paper_reference": {"psnr": dg.PAPER_PSNR, "ssim": 0.966, "lpips": 0.0128,
                                                     "citation": "DM-AHR paper, Cancers 2024, 16, 2947"},
        "settings": summary,
        "verdict": dg.phase1_verdict(primary["output"]["psnr"]) if primary else
                   {"statement": "NOT EVALUATED: res128_steps2000 was not part of this run"},
        "measured_seconds_per_image_step": per_step,
        "seconds": round(time.time() - started, 1),
        "estimated_usd": round(estimate_usd(time.time() - started, args.gpu), 2),
        "note": "hairless references may contain natural hair; they were not filtered",
    }
    pd.DataFrame(rows).to_csv(artifact_dir("logs") / f"{args.tag}.csv", index=False)
    save_json(result, artifact_dir("logs") / f"{args.tag}.json")
    record_run("paper_reproduction", gpu=args.gpu, seconds=result["seconds"],
               estimated_usd=result["estimated_usd"], source=label[:40])
    print(json.dumps(result["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
