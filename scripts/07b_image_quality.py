"""Paired PSNR/SSIM against held-out hairless references.

Scores the hairy input against its reference (the degradation to be removed),
plus Arm B and Arm C outputs on the same hairy inputs. Only synthetic pairs,
where a genuine hairless reference exists, are used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.hair.dullrazor import apply_dullrazor  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir, processed_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402
from dm_ahr_vit.config import load_frozen  # noqa: E402

log = setup_logging()


def resolve(path: str) -> Path:
    """Map a recorded Modal-volume path onto the local data root.

    Manifests are written inside the container, so they hold `/__modal/...`
    paths. Re-rooting on the `/data/` segment keeps them usable off-Modal.
    """
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def score(pred: np.ndarray, ref: np.ndarray, size: int = 128, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    a = np.asarray(Image.fromarray(pred).convert("RGB").resize((size, size),
                                                               Image.Resampling.BICUBIC),
                   dtype=np.float64) / 255.0
    b = np.asarray(Image.fromarray(ref).convert("RGB").resize((size, size),
                                                              Image.Resampling.BICUBIC),
                   dtype=np.float64) / 255.0
    psnr = peak_signal_noise_ratio(b, a, data_range=1.0)
    ssim = structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                 win_size=int(cfg.get("ssim_win_size", 7)))
    return {"psnr": float(psnr), "ssim": float(ssim)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--c-arm", default="C", help="diffusion arm to score; default is the recorded Arm C")
    args = ap.parse_args()

    sampling_cfg = load_config("configs/sampling.json")
    screen_cfg = load_frozen("hair_screen", "configs/hair_screen.json")
    dull_cfg = load_frozen("dullrazor", "configs/dullrazor.json")
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    pairs = pairs[pairs["split"] == args.split]
    if args.limit:
        pairs = pairs.head(args.limit)
    ca = args.c_arm
    c_dir = processed_dir("synth") / args.split / f"cleaned_{ca}"
    suffix = "" if ca == "C" else f"_{ca}"
    released_dir = processed_dir("synth") / args.split / "cleaned_Creleased"

    rows = []
    with timed("image_quality", log) as t:
        for r in pairs.itertuples(index=False):
            ref = np.array(Image.open(resolve(r.reference_path)).convert("RGB"))
            hairy = np.array(Image.open(resolve(r.hairy_path)).convert("RGB"))
            mask, _ = detect_hair(hairy, screen_cfg)
            b_out = apply_dullrazor(hairy, mask, dull_cfg)
            row = {
                "pair_id": r.pair_id,
                "image_id": r.image_id,
                "split": r.split,
                "dx": r.dx,
                "group_id": r.group_id,
                "source": r.source,
            }
            for arm, arr in (("hairy_input", hairy), ("B", b_out)):
                m = score(arr, ref, cfg=sampling_cfg)
                row[f"{arm}_psnr"], row[f"{arm}_ssim"] = m["psnr"], m["ssim"]
            c_path = c_dir / f"{r.image_id}.png"
            if c_path.exists():
                c_out = np.array(Image.open(c_path).convert("RGB"))
                m = score(c_out, ref, cfg=sampling_cfg)
                row[f"{ca}_psnr"], row[f"{ca}_ssim"] = m["psnr"], m["ssim"]
                row[f"{ca}_status"] = "ok"
            else:
                row[f"{ca}_psnr"] = row[f"{ca}_ssim"] = float("nan")
                row[f"{ca}_status"] = "missing"
            rel_path = released_dir / f"{r.image_id}.png"
            if rel_path.exists():
                rel_out = np.array(Image.open(rel_path).convert("RGB"))
                m = score(rel_out, ref, cfg=sampling_cfg)
                row["Creleased_psnr"], row["Creleased_ssim"] = m["psnr"], m["ssim"]
                row["Creleased_status"] = "ok"
            else:
                row["Creleased_psnr"] = row["Creleased_ssim"] = float("nan")
                row["Creleased_status"] = "missing"
            rows.append(row)

        df = pd.DataFrame(rows)
        out = artifact_dir("logs") / f"paired_psnr_ssim_{args.split}{suffix}.csv"
        df.to_csv(out, index=False)

    summary = {
        "split": args.split,
        "pairs": int(len(df)),
        f"{ca}_scored": int((df[f"{ca}_status"] == "ok").sum()),
        "Creleased_scored": int((df["Creleased_status"] == "ok").sum()),
        "means": {
            col: float(df[col].mean()) for col in
            ("hairy_input_psnr", "B_psnr", f"{ca}_psnr", "Creleased_psnr",
             "hairy_input_ssim", "B_ssim", f"{ca}_ssim", "Creleased_ssim")
            if col in df and df[col].notna().any()
        },
        "seconds": t["seconds"],
    }
    save_json(summary, artifact_dir("logs") / f"image_quality_{args.split}{suffix}.json")
    record_run("image_quality", gpu=None, seconds=t["seconds"], split=args.split,
               pairs=len(df))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

