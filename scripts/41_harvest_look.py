"""Local look at the thin-hair filter on a few real-hair train images (CPU)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dm_ahr_vit.hair.harvest import thin_hair_mask  # noqa: E402

cfg = json.loads((ROOT / "artifacts/logs/hair_frozen_config.json").read_text())["screen"]
ids = Path("/root/local/hair_check/ids.txt").read_text().split()[:12]
tiles, stats = [], []
for iid in ids:
    rgb = cv2.cvtColor(cv2.imread(f"/root/local/hair_check/orig/{iid}.jpg"), cv2.COLOR_BGR2RGB)
    mask, info = thin_hair_mask(rgb, cfg)
    ov = rgb.copy()
    ov[mask > 0] = (255, 0, 255)
    both = np.concatenate([rgb, ov], axis=1)
    tiles.append(cv2.resize(both, (600, 225)))
    stats.append((iid, info["components_total"], info["components_kept"], info["rejected"],
                  round(info["fraction"] * 100, 2)))
rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)]
cv2.imwrite("/root/local/hair_check/harvest_check.jpg",
            cv2.cvtColor(np.concatenate(rows[:4], axis=0), cv2.COLOR_RGB2BGR))
for s in stats:
    print(s)
