"""Diagnose Arm C: is the poor output undertraining or an inference-path defect?

Runs on validation-source synthetic pairs only. The test pairs are never read
here, so no test-set choice depends on this diagnosis.

Produces:
  artifacts/logs/arm_c_diagnosis.json      summary and decision
  artifacts/logs/arm_c_identity.csv        identity test per image
  artifacts/logs/arm_c_headtohead.csv      per-pair metrics for every condition
  artifacts/reports/arm_c_diag_*.png       visual panels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    CHECKPOINT_NAME, clean_batch, load_model,
)
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir, processed_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()
SSIM_CFG = {"ssim_win_size": 7}


def score(pred: Image.Image, ref: Image.Image, size: int = 128) -> dict:
    a = np.asarray(pred.convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
                   dtype=np.float64) / 255.0
    b = np.asarray(ref.convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
                   dtype=np.float64) / 255.0
    return {
        "psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
        "ssim": float(structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                            win_size=SSIM_CFG["ssim_win_size"])),
    }


def channel_stats(im: Image.Image) -> dict:
    a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return {
        "mean_r": float(a[..., 0].mean()),
        "mean_g": float(a[..., 1].mean()),
        "mean_b": float(a[..., 2].mean()),
        "std": float(a.std()),
    }


def high_freq_energy(im: Image.Image) -> float:
    """Mean absolute Laplacian: a proxy for retained texture."""
    a = np.asarray(im.convert("L").resize((128, 128), Image.Resampling.BICUBIC),
                   dtype=np.float64) / 255.0
    lap = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    return float(np.abs(lap).mean())


def alignment_offset(pred: Image.Image, ref: Image.Image) -> dict:
    """Best integer shift of pred vs ref by normalised cross-correlation."""
    a = np.asarray(pred.convert("L").resize((128, 128), Image.Resampling.BICUBIC),
                   dtype=np.float64)
    b = np.asarray(ref.convert("L").resize((128, 128), Image.Resampling.BICUBIC),
                   dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best = (-2.0, 0, 0)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            aa = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
            c = float((aa * b).mean())
            if c > best[0]:
                best = (c, dy, dx)
    flip = float((np.flipud(a) * b).mean())
    fliplr = float((np.fliplr(a) * b).mean())
    return {"best_ncc": best[0], "shift_dy": best[1], "shift_dx": best[2],
            "ncc_flipud": flip, "ncc_fliplr": fliplr}


def panel(rows: list[tuple[Image.Image, str]], out: Path) -> None:
    th = 160
    canvas = Image.new("RGB", (th * len(rows), th + 16), (0, 0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(rows):
        canvas.paste(im.convert("RGB").resize((th, th), Image.Resampling.BICUBIC),
                     (i * th, 0))
        d.text((i * th + 3, th + 2), label[:26], fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-n", type=int, default=50)
    ap.add_argument("--pairs-n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config("configs/dmahr.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = int(cfg["native_resolution"])

    released = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    ours = artifact_dir("checkpoints", "dmahr") / "best_gen.pth"
    for p in (released, ours):
        if not p.exists():
            raise FileNotFoundError(f"missing checkpoint {p}")

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].sort_values("pair_id").reset_index(drop=True)
    if val.empty:
        raise RuntimeError("no validation synthetic pairs")
    subs = val.head(args.pairs_n)
    log.info("validation pairs available=%d using=%d", len(val), len(subs))

    refs = [Image.open(p).convert("RGB") for p in subs["reference_path"]]
    hairy = [Image.open(p).convert("RGB") for p in subs["hairy_path"]]

    identity_rows: list[dict] = []
    head_rows: list[dict] = []
    panels: dict[str, list] = {}

    # ---------------------------------------------------------------- identity
    log.info("identity test on %d validation hairless references", args.identity_n)
    net = load_model(cfg, released, device=device)
    id_refs = refs[:args.identity_n]
    outs, _ = clean_batch(net, id_refs, cfg, device=device, seed=0,
                          num_timesteps=1000, use_bf16=True)
    for i, (o, r) in enumerate(zip(outs, id_refs)):
        s = score(o, r)
        identity_rows.append({
            "i": i, "psnr": s["psnr"], "ssim": s["ssim"],
            "hf_in": high_freq_energy(r), "hf_out": high_freq_energy(o),
            **{f"in_{k}": v for k, v in channel_stats(r).items()},
            **{f"out_{k}": v for k, v in channel_stats(o).items()},
        })
    id_df = pd.DataFrame(identity_rows)
    identity = {
        "n": len(id_df),
        "psnr_mean": float(id_df["psnr"].mean()),
        "psnr_sd": float(id_df["psnr"].std()),
        "ssim_mean": float(id_df["ssim"].mean()),
        "ssim_sd": float(id_df["ssim"].std()),
        "hf_ratio": float(id_df["hf_out"].mean() / max(id_df["hf_in"].mean(), 1e-9)),
    }
    log.info("identity: PSNR %.2f SSIM %.3f hf_ratio %.3f",
             identity["psnr_mean"], identity["ssim_mean"], identity["hf_ratio"])
    panels["identity"] = [(id_refs[0], "reference"), (outs[0], "round-trip"),
                          (id_refs[1], "reference"), (outs[1], "round-trip")]

    # --------------------------------------------------------------- head to head
    conditions = [
        ("released_2000", released, 2000),
        ("released_1000", released, 1000),
        ("ours_2000", ours, 2000),
        ("ours_1000", ours, 1000),
    ]
    for tag, ckpt, steps in conditions:
        log.info("running %s", tag)
        net = load_model(cfg, ckpt, device=device)
        outs_all: list[Image.Image] = []
        for start in range(0, len(hairy), args.batch_size):
            chunk = hairy[start:start + args.batch_size]
            o, _ = clean_batch(net, chunk, cfg, device=device, seed=start,
                               num_timesteps=steps, use_bf16=True)
            outs_all.extend(o)
        for (pair_id, ref, hair, out) in zip(subs["pair_id"], refs, hairy, outs_all):
            s = score(out, ref)
            sh = score(hair, ref)
            head_rows.append({
                "condition": tag, "pair_id": pair_id,
                "ckpt": ckpt.name, "steps": steps,
                "psnr": s["psnr"], "ssim": s["ssim"],
                "hairy_psnr": sh["psnr"], "hairy_ssim": sh["ssim"],
                "hf_out": high_freq_energy(out), "hf_ref": high_freq_energy(ref),
                **{f"out_{k}": v for k, v in channel_stats(out).items()},
                **{f"ref_{k}": v for k, v in channel_stats(ref).items()},
                "out_min_scale": float(np.asarray(out).min()),
                "out_max_scale": float(np.asarray(out).max()),
            })
        if tag not in panels:
            panels[tag] = [(hair[0], "hairy input"), (refs[0], "reference"),
                           (outs_all[0], tag)]

    head = pd.DataFrame(head_rows)
    artifact_dir("logs")
    head.to_csv(artifact_dir("logs") / "arm_c_headtohead.csv", index=False)
    id_df.to_csv(artifact_dir("logs") / "arm_c_identity.csv", index=False)

    summary = (head.groupby("condition")
               .agg(n=("psnr", "size"), psnr=("psnr", "mean"), psnr_sd=("psnr", "std"),
                    ssim=("ssim", "mean"), ssim_sd=("ssim", "std"),
                    hairy_psnr=("hairy_psnr", "mean"),
                    hf_out=("hf_out", "mean"), hf_ref=("hf_ref", "mean"))
               .round(4))
    log.info("head-to-head\n%s", summary.to_string())

    # ---------------------------------------------------------------- alignment
    net = load_model(cfg, ours, device=device)
    probe, _ = clean_batch(net, hairy[:2], cfg, device=device, seed=0,
                           num_timesteps=1000, use_bf16=True)
    align = [alignment_offset(o, r) for o, r in zip(probe, refs[:2])]

    # ---------------------------------------------------------------- decision
    id_ok = identity["psnr_mean"] >= 30.0 and identity["hf_ratio"] >= 0.7
    rel = summary.loc["released_2000"] if "released_2000" in summary.index else None
    our = summary.loc["ours_1000"] if "ours_1000" in summary.index else None
    released_ok = bool(rel is not None and rel["psnr"] >= rel["hairy_psnr"]
                       and rel["psnr"] >= 25.0)
    ours_ok = bool(our is not None and our["psnr"] >= our["hairy_psnr"])

    if not id_ok:
        outcome = "a"
        verdict = ("Adapter defect: a hairless reference does not round-trip through "
                   "the inference path. Fix the path before interpreting Arm C.")
    elif not released_ok:
        outcome = "a"
        verdict = ("Adapter or checkpoint defect: the released checkpoint does not "
                   "produce sensible output through this adapter at reference settings.")
    elif not ours_ok:
        outcome = "b"
        verdict = ("Undertrained: the released checkpoint behaves sensibly through the "
                   "same adapter; our 20,000-iteration checkpoint does not.")
    else:
        outcome = "c"
        verdict = ("Both behave as expected: the reported Arm C failure does not "
                   "reproduce on the validation pairs.")

    out = {
        "outcome": outcome, "verdict": verdict,
        "identity": identity, "identity_ok": id_ok,
        "released_ok": released_ok, "ours_ok": ours_ok,
        "summary": summary.reset_index().to_dict(orient="records"),
        "alignment": align,
        "split": "validation synthetic pairs only; test pairs never read",
        "n_pairs": int(len(subs)), "identity_n": int(len(id_df)),
        "batch_size": args.batch_size,
    }
    save_json(out, artifact_dir("logs") / "arm_c_diagnosis.json")

    rep = artifact_dir("reports")
    for tag, rows in panels.items():
        panel(rows, rep / f"arm_c_diag_{tag}.jpg")
    record_run("arm_c_diagnose", gpu=device, pairs=len(subs),
               identity=len(id_df), outcome=outcome)
    print(json.dumps({k: out[k] for k in
                      ("outcome", "verdict", "identity", "identity_ok",
                       "released_ok", "ours_ok")}, indent=2))
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
