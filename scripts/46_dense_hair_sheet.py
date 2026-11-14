"""Dense real-hair validation photos: pick them and build a side-by-side sheet (local, CPU).

select: the N validation-split images with the largest harvested strand area (a fixed rule,
        chosen before looking at any output). Reads the harvest summary pulled from the volume.
sheet:  original | Arm B | one column per --arm NAME=DIR, each tile enlarged 2x from 128 px.
Validation images only; the test split is never listed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SIZE = 128


def cmd_select(a) -> None:
    df = pd.read_csv(a.summary)
    assert set(df["split"]) <= {"train", "val"}
    df = df[df["split"] == "val"].sort_values(["fraction", "image_id"], ascending=[False, True])
    Path(a.out).write_text("\n".join(df["image_id"].head(a.n)) + "\n")
    print(f"selected {min(a.n, len(df))} validation images; fraction range "
          f"{df['fraction'].head(a.n).min():.4f} to {df['fraction'].head(a.n).max():.4f}")


def _tile(path: Path, label: str | None, scale: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((SIZE * scale, SIZE * scale), Image.Resampling.BICUBIC)
    if label:
        ImageDraw.Draw(im).text((3, 3), label, fill=(255, 255, 0))
    return np.asarray(im)


def cmd_sheet(a) -> None:
    ids = [i for i in Path(a.ids).read_text().split() if i]
    cols = {"orig": Path(a.orig_dir), "B": Path(a.b_dir)}
    for spec in a.arm:
        k, v = spec.split("=", 1)
        cols[k] = Path(v)
    rows = []
    for iid in ids[:a.rows]:
        tiles = []
        for j, (name, d) in enumerate(cols.items()):
            p = d / f"{iid}.jpg" if name == "orig" else d / f"{iid}.png"
            tiles.append(_tile(p, name if iid == ids[0] else None, a.scale))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(a.out)
    print("columns:", list(cols), "->", a.out)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--summary", required=True)
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--out", required=True)
    h = sub.add_parser("sheet")
    h.add_argument("--ids", required=True)
    h.add_argument("--orig-dir", required=True)
    h.add_argument("--b-dir", required=True)
    h.add_argument("--arm", action="append", default=[])
    h.add_argument("--rows", type=int, default=12)
    h.add_argument("--scale", type=int, default=2)
    h.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "sheet": cmd_sheet}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
