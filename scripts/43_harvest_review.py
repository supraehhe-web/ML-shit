"""Contact sheets for reviewing harvested hair layers (train/val images only).

Two sheets: 24 images drawn at random with a fixed seed, and the 12 with the largest
harvested fraction (where a wrong rule would show up first). Each tile is the original next
to the same image with the harvested pixels painted magenta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()


def tile(image_id: str, layers: Path) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.imread(str(isic_images_dir() / f"{image_id}.jpg")), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(layers / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    ov = rgb.copy()
    ov[m > 127] = (255, 0, 255)
    both = np.concatenate([rgb, ov], axis=1)
    both = cv2.resize(both, (640, 240), interpolation=cv2.INTER_AREA)
    cv2.putText(both, image_id[-7:], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return both


def sheet(ids: list[str], layers: Path, cols: int = 2) -> np.ndarray:
    tiles = [tile(i, layers) for i in ids]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[i:i + cols], axis=1)
                           for i in range(0, len(tiles), cols)], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=24)
    ap.add_argument("--n-top", type=int, default=12)
    args = ap.parse_args()
    layers = data_root() / "hair_layers"
    df = pd.read_csv(layers / "summary.csv")
    kept = df[df["hair_pixels"] > 0].sort_values("image_id").reset_index(drop=True)
    assert set(kept["split"]) <= {"train", "val"}
    rng = np.random.default_rng(args.seed)
    rand_ids = list(kept["image_id"].iloc[rng.choice(len(kept), args.n_random, replace=False)])
    top_ids = list(kept.sort_values("fraction", ascending=False)["image_id"].head(args.n_top))
    out = artifact_dir("diagnosis")
    for name, ids in (("random", rand_ids), ("top_fraction", top_ids)):
        for k in range(0, len(ids), 12):
            cv2.imwrite(str(out / f"harvest_review_{name}_{k // 12}.jpg"),
                        cv2.cvtColor(sheet(ids[k:k + 12], layers), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    q = kept["fraction"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(5).to_dict()
    summ = {"harvested_images": int(len(kept)), "fraction_quantiles": {str(k): v for k, v in q.items()},
            "random_ids": rand_ids, "top_ids": top_ids, "seed": args.seed}
    (out / "harvest_review_stageB.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("harvested_images", "fraction_quantiles")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
