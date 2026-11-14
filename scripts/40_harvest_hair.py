"""Harvest long thin dark hair strands from train and validation images (CPU, parallel).

Writes `<DATA_ROOT>/hair_layers/<image_id>.png` (0 or 255) for every image that yields at
least one accepted strand, plus `summary.csv`. The test split is never read. Hair-present
images only (detector coverage of at least 0.03); the rule for what counts as a strand lives
in `dm_ahr_vit/hair/harvest.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_hair_screen_config  # noqa: E402
from dm_ahr_vit.data.hair_composite import HAIR_MIN_COVERAGE  # noqa: E402
from dm_ahr_vit.hair.harvest import thin_hair_mask  # noqa: E402
from dm_ahr_vit.paths import data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
CFG: dict = {}
OUT: Path = Path()


def work(image_id: str) -> dict:
    bgr = cv2.imread(str(isic_images_dir() / f"{image_id}.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask, info = thin_hair_mask(rgb, CFG)
    if info["hair_pixels"] > 0:
        cv2.imwrite(str(OUT / f"{image_id}.png"), (mask * 255).astype(np.uint8))
    return {"image_id": image_id, **info}


def init(cfg: dict, out: str) -> None:
    global CFG, OUT
    CFG, OUT = cfg, Path(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process only this many (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_hair_screen_config()
    log.info("hair screen config hash %s", cfg["config_hash"])
    scr = pd.read_csv(manifest_dir() / "hair_screening.csv")
    scr = scr[scr["split"].isin(["train", "val"]) & (scr["hair_coverage"] >= HAIR_MIN_COVERAGE)]
    assert not (scr["split"] == "test").any()
    ids = sorted(scr["image_id"])
    if args.limit:
        ids = ids[:args.limit]
    out = Path(args.out) if args.out else data_root() / "hair_layers"
    out.mkdir(parents=True, exist_ok=True)
    log.info("harvesting %d images with %d workers -> %s", len(ids), args.workers, out)
    with timed("harvest_hair", log) as t:
        with Pool(args.workers, initializer=init, initargs=(cfg, str(out))) as pool:
            rows = pool.map(work, ids, chunksize=16)
    df = pd.DataFrame(rows).merge(scr[["image_id", "split", "hair_coverage"]], on="image_id")
    df.to_csv(out / "summary.csv", index=False)
    kept = df[df["hair_pixels"] > 0]
    summ = {"images": int(len(df)), "with_strands": int(len(kept)),
            "by_split": kept.groupby("split").size().to_dict(),
            "median_fraction": float(kept["fraction"].median()) if len(kept) else 0.0,
            "median_components": float(kept["components_kept"].median()) if len(kept) else 0.0,
            "seconds": t["seconds"]}
    record_run("harvest_hair", gpu=None, **{k: v for k, v in summ.items() if k != "by_split"})
    print(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
