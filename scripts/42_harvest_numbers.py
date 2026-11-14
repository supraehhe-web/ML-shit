"""Print per-component statistics of the strand finder on a few real-hair train images."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dm_ahr_vit.hair.screening import hair_response  # noqa: E402

cfg = json.loads((ROOT / "artifacts/logs/hair_frozen_config.json").read_text())["screen"]
ids = Path("/root/local/hair_check/ids.txt").read_text().split()[:12]
# what I saw by eye: dense hair = 0032694, 0033344, 0030085(index 3?), 0029509 ; lesion texture = 0026593, 0029744, 0030965, 0029486
label = {"ISIC_0032694": "HAIR", "ISIC_0033344": "HAIR", "ISIC_0024813": "HAIR", "ISIC_0030626": "HAIR",
         "ISIC_0026593": "LESION", "ISIC_0029744": "LESION", "ISIC_0030965": "LESION",
         "ISIC_0029486": "LESION", "ISIC_0030085": "PEN+LESION"}
for iid in ids:
    rgb = cv2.cvtColor(cv2.imread(f"/root/local/hair_check/orig/{iid}.jpg"), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark, _ = hair_response(gray, {**cfg, "light_hair": False})
    mask = (dark > 30).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    skel = skeletonize(mask > 0)
    su = skel.astype(np.uint8)
    sl = np.bincount(lab[skel], minlength=n)
    nb = cv2.filter2D(su, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    jn = np.bincount(lab[skel & ((nb - su) >= 3)], minlength=n)
    area = st[:, cv2.CC_STAT_AREA]
    order = [i for i in np.argsort(-sl) if i > 0][:6]
    print(iid, label.get(iid, "?"))
    for i in order:
        print(f"   area {area[i]:6d}  skel {sl[i]:5d}  width {area[i]/max(sl[i],1):5.2f}  "
              f"junc/len {jn[i]/max(sl[i],1):.3f}  maxdark {int(dark[lab==i].max())}")
