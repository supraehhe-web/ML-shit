"""Copy originals into Arm C for hair-absent images.

The protocol says hair-absent images are identical in every arm, so Arm C must
contain the original for those images rather than a diffusion output. The
sampler only processes hair-present images, so this fills the gap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.paths import isic_images_dir, manifest_dir, processed_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", default="C")
    args = ap.parse_args()
    screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
    out_dir = processed_dir(args.arm_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    absent = screen[~screen["hair_present"].astype(bool)]
    written = skipped = failed = 0
    with timed("fill_c_passthrough", log) as t:
        for image_id in absent["image_id"]:
            dest = out_dir / f"{image_id}.png"
            if dest.exists():
                skipped += 1
                continue
            try:
                with Image.open(isic_images_dir() / f"{image_id}.jpg") as im:
                    im.convert("RGB").save(dest, format="PNG")
                written += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.warning("failed %s: %r", image_id, exc)
    summary = {
        "hair_absent": int(len(absent)),
        "written": written,
        "skipped_existing": skipped,
        "failed": failed,
        "seconds": t["seconds"],
    }
    record_run("fill_c_passthrough", gpu=None, **summary)
    print(summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

