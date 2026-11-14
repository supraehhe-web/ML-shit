"""Blinded review kit for real-photo hair removal, and the analysis of returned ratings.

build:   50 random validation hair-present photos (fixed seed). For each, one PNG with three
         candidates in random order, labelled X, Y, Z: DullRazor (B), Cv2, and the plain 128 px
         original (hair-remains control). All are shown at 128 px upscaled 3x so sharpness does
         not give the arm away. Writes ratings_template.csv and key.csv (the key is only read by
         `analyze`; do not open it before rating).
analyze: reads a completed ratings CSV and the key; per-arm means of hair_remaining (0 none to
         3 heavy) and lesion_preserved (0 destroyed to 3 intact) and paired B-versus-Cv2
         differences with bootstrap CIs over photos. Validation photos only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, isic_images_dir, manifest_dir, processed_dir  # noqa: E402

SEED, N, SIZE, SCALE = 20261113, 50, 128, 3
ARMS = ("B", "Cv2", "hairy_128")
LETTERS = ("X", "Y", "Z")


def load128(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)


def build(out: Path) -> None:
    scr = pd.read_csv(manifest_dir() / "hair_screening.csv")
    pool = scr[(scr["split"] == "val") & scr["hair_present"].astype(bool)]
    ids = sorted(pool["image_id"])
    rng = np.random.default_rng(SEED)
    pick = [ids[i] for i in rng.choice(len(ids), N, replace=False)]
    out.mkdir(parents=True, exist_ok=True)
    key, rate = [], []
    for k, iid in enumerate(pick, 1):
        cands = {"B": load128(processed_dir("B") / f"{iid}.png"),
                 "Cv2": load128(processed_dir("Cv2") / f"{iid}.png"),
                 "hairy_128": load128(isic_images_dir() / f"{iid}.jpg")}
        order = list(rng.permutation(len(ARMS)))
        canvas = Image.new("RGB", (3 * SIZE * SCALE + 40, SIZE * SCALE + 36), (30, 30, 30))
        d = ImageDraw.Draw(canvas)
        row = {"photo": f"review_{k:03d}", "image_id": iid}
        for slot, ai in enumerate(order):
            arm, letter = ARMS[ai], LETTERS[slot]
            x0 = 10 + slot * (SIZE * SCALE + 10)
            canvas.paste(cands[arm].resize((SIZE * SCALE, SIZE * SCALE), Image.Resampling.BICUBIC), (x0, 26))
            d.text((x0 + SIZE * SCALE // 2 - 4, 6), letter, fill=(255, 255, 0))
            row[letter] = arm
            rate.append({"photo": row["photo"], "candidate": letter, "hair_remaining_0to3": "",
                         "lesion_preserved_0to3": ""})
        canvas.save(out / f"review_{k:03d}.png")
        key.append(row)
    pd.DataFrame(rate).to_csv(out / "ratings_template.csv", index=False)
    pd.DataFrame(key).to_csv(out / "key_DO_NOT_OPEN_BEFORE_RATING.csv", index=False)
    (out / "README.txt").write_text(
        "Blinded review of hair removal on 50 random validation dermoscopy photos (research only).\n"
        "Each review_NNN.png shows three candidates X, Y, Z in random order. For every candidate fill\n"
        "ratings_template.csv:\n"
        "  hair_remaining_0to3   0 = no hair visible, 1 = faint, 2 = clear, 3 = heavy\n"
        "  lesion_preserved_0to3 3 = outline, colour and texture look intact, 0 = lesion destroyed or altered\n"
        "One candidate in each photo is the unprocessed image at the same resolution; it shows what hair\n"
        "there was. Rate from the images alone and do not open the key file until finished. Note: DullRazor leaves\n"
        "blocky inpainting patches that may be recognisable, so blinding is imperfect; rate what you see.\n"
        "Then run: python scripts/54_review_kit.py analyze --dir <this folder> --ratings <your csv>\n")
    print(f"wrote {N} photos to {out}")


def boot_ci(x: np.ndarray, reps: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (reps, x.size))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def analyze(folder: Path, ratings: Path) -> dict:
    key = pd.read_csv(folder / "key_DO_NOT_OPEN_BEFORE_RATING.csv")
    r = pd.read_csv(ratings)
    cols = ["hair_remaining_0to3", "lesion_preserved_0to3"]
    r[cols] = r[cols].apply(pd.to_numeric, errors="raise")
    assert r[cols].notna().all().all(), "ratings incomplete"
    assert r[cols].min().min() >= 0 and r[cols].max().max() <= 3, "ratings must be 0 to 3"
    long = key.melt(id_vars=["photo", "image_id"], value_vars=list(LETTERS), var_name="candidate",
                    value_name="arm").merge(r, on=["photo", "candidate"])
    assert len(long) == len(r) == 3 * len(key)
    res = {"photos": int(len(key)), "arms": {}, "paired_Cv2_minus_B": {}}
    for arm, g in long.groupby("arm"):
        res["arms"][arm] = {c: dict(zip(("mean", "ci_low", "ci_high"), boot_ci(g[c].to_numpy(float))))
                            for c in cols}
    piv = {c: long.pivot(index="photo", columns="arm", values=c) for c in cols}
    for c in cols:
        d = (piv[c]["Cv2"] - piv[c]["B"]).to_numpy(float)
        res["paired_Cv2_minus_B"][c] = dict(zip(("mean", "ci_low", "ci_high"), boot_ci(d)))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--dir", required=True)
    a.add_argument("--ratings", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(Path(args.out) if args.out else artifact_dir("review_kit"))
    else:
        res = analyze(Path(args.dir), Path(args.ratings))
        save_json(res, Path(args.dir) / "review_result.json")
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
