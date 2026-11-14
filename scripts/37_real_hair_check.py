"""Does the released DM-AHR model remove REAL hair? CPU check, train-split images only.

Every image-quality number so far is on synthetic hair overlays, which is what DM-AHR
was trained to remove. This measures the effect on real hair, using the frozen hair
detector as an independent instrument and DullRazor (Arm B) as the positive control.

For each selected real-hair image the original, the Arm B output and the released-DM-AHR
output are brought to 128 px (the diffusion model's native size) and compared:
  1. detector hair coverage before and after (fraction remaining)
  2. mean absolute change inside the detected hair mask versus outside it

Commands
  select   write a deterministic list of train-split, clearly hairy image ids
  measure  compute the metrics and a contact sheet from downloaded images

Artifacts record git commit, detector config hash and the id-list hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.hair.screening import detect_hair  # noqa: E402

SIZE = 128
OUT = ROOT / "artifacts" / "diagnosis"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cmd_select(args) -> None:
    scr = pd.read_csv(args.screening)
    spl = pd.read_csv(args.splits)[["image_id", "split"]].rename(columns={"split": "split2"})
    df = scr.merge(spl, on="image_id")
    df = df[(df["split2"] == args.split) & (df["hair_present"]) & (df["hair_coverage"] >= args.min_coverage)]
    df = df.assign(key=df["image_id"].map(_sha)).sort_values("key").head(args.n)
    Path(args.out).write_text("\n".join(df["image_id"]) + "\n")
    print(f"selected {len(df)} {args.split}-split images, coverage >= {args.min_coverage}, "
          f"median coverage {df['hair_coverage'].median():.3f}")


def _load(path: Path, to128: bool = True) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if to128 and img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(img)


def cmd_measure(args) -> None:
    cfg = json.loads(Path(args.config).read_text())["screen"]
    ids = [i for i in Path(args.ids).read_text().split() if i]
    if args.arm:
        arm_dirs = {k: Path(v) for k, v in (a.split("=", 1) for a in args.arm)}
    else:
        arm_dirs = {"B": Path(args.b_dir), "Creleased": Path(args.c_dir)}
    rows = []
    sheet = []
    for iid in ids:
        o = _load(Path(args.orig_dir) / f"{iid}.jpg")
        outs = {name: _load(d / f"{iid}.png") for name, d in arm_dirs.items()}
        mask, st0 = detect_hair(o, cfg)
        m = mask.astype(bool)
        row = {"image_id": iid, "cov_orig": st0["coverage"]}
        for name, arr in outs.items():
            _, st = detect_hair(arr, cfg)
            diff = np.abs(arr.astype(np.float64) - o.astype(np.float64)).mean(axis=2)
            row[f"cov_{name}"] = st["coverage"]
            row[f"remaining_{name}"] = st["coverage"] / st0["coverage"] if st0["coverage"] > 0 else np.nan
            row[f"absdiff_in_{name}"] = float(diff[m].mean()) if m.any() else np.nan
            row[f"absdiff_out_{name}"] = float(diff[~m].mean())
        rows.append(row)
        if len(sheet) < 8:
            sheet.append(np.concatenate([o, *outs.values()], axis=1))
    df = pd.DataFrame(rows)
    summ = {}
    for name in arm_dirs:
        summ[name] = {
            "median_fraction_of_hair_coverage_remaining": float(df[f"remaining_{name}"].median()),
            "mean_fraction_remaining": float(df[f"remaining_{name}"].mean()),
            "images_with_less_than_half_remaining": int((df[f"remaining_{name}"] < 0.5).sum()),
            "mean_absdiff_inside_hair_mask": float(df[f"absdiff_in_{name}"].mean()),
            "mean_absdiff_outside_hair_mask": float(df[f"absdiff_out_{name}"].mean()),
            "inside_over_outside_ratio": float(df[f"absdiff_in_{name}"].mean() / df[f"absdiff_out_{name}"].mean()),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(sheet, axis=0)).save(OUT / f"real_hair_contact_sheet{args.out_suffix}.png")
    df.to_csv(OUT / f"real_hair_check{args.out_suffix}.csv", index=False)
    result = {
        "n_images": len(df), "median_orig_coverage": float(df["cov_orig"].median()),
        "summary": summ,
        "provenance": {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=ROOT).stdout.strip(),
            "detector_config_hash": cfg.get("config_hash"),
            "id_list_sha256": _sha(Path(args.ids).read_text()),
            "resolution": SIZE,
        },
    }
    (OUT / f"real_hair_check{args.out_suffix}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--screening", default="/root/local/gate_data/manifests/hair_screening.csv")
    s.add_argument("--splits", default="/root/local/gate_data/manifests/splits.csv")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--split", default="train", choices=["train", "val"])  # never test
    s.add_argument("--min-coverage", type=float, default=0.05)
    s.add_argument("--out", default="/root/local/hair_check/ids.txt")
    m = sub.add_parser("measure")
    m.add_argument("--ids", default="/root/local/hair_check/ids.txt")
    m.add_argument("--orig-dir", default="/root/local/hair_check/orig")
    m.add_argument("--b-dir", default="/root/local/hair_check/B")
    m.add_argument("--c-dir", default="/root/local/hair_check/Creleased")
    m.add_argument("--arm", action="append", default=None,
                   help="NAME=DIR, repeatable; replaces the default B and Creleased arms")
    m.add_argument("--out-suffix", default="")
    m.add_argument("--config", default=str(ROOT / "artifacts" / "logs" / "hair_frozen_config.json"))
    args = ap.parse_args()
    {"select": cmd_select, "measure": cmd_measure}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
