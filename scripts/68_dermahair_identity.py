"""How many DERMAHAIR hairy/hairless pairs are pixel-identical, and the pixel statistics of the rest."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
same, diffs, total = 0, [], 0
for p in sorted((root / "Data_Skin_with_Hair").glob("*/*.jpg")):
    q = root / "Data_Skin_without_Hair" / p.parent.name / p.name
    a = np.asarray(Image.open(p).convert("RGB"), np.float64)
    b = np.asarray(Image.open(q).convert("RGB"), np.float64)
    total += 1
    mse = float(((a - b) ** 2).mean())
    same += mse == 0
    diffs.append(mse)
d = np.array(diffs)
print(f"pairs={total} identical={same} min_mse_nonzero={d[d > 0].min():.2f} median_psnr_nonidentical="
      f"{np.median(10 * np.log10(255 ** 2 / d[d > 0])):.2f}")
