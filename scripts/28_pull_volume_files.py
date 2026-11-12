"""Pull specific files from the Modal volume with the client API (no compute).

`modal volume get` on a directory writes a single file in this CLI version, so
this uses `Volume.read_file` directly. Used to bring the C-released test images
local for a CPU evaluation while Modal compute is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

VOLUME = "dmahr-vit-vol"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-dir", required=True)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--ids-file", default=None, help="text file with one image id per line")
    ap.add_argument("--suffix", default=".png")
    args = ap.parse_args()
    vol = modal.Volume.from_name(VOLUME)
    out = Path(args.local_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.ids_file:
        ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
    else:
        ids = [Path(p.path).stem for p in vol.iterdir(args.remote_dir)
               if p.path.endswith(args.suffix)]
    done = failed = 0
    for i, image_id in enumerate(ids, 1):
        dest = out / f"{image_id}{args.suffix}"
        if dest.exists():
            done += 1
            continue
        try:
            data = b"".join(vol.read_file(f"{args.remote_dir}/{image_id}{args.suffix}"))
            dest.write_bytes(data)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAILED {image_id}: {exc!r}", file=sys.stderr)
        if i % 200 == 0:
            print(f"{i}/{len(ids)} ({done} ok, {failed} failed)", flush=True)
    print(f"done={done} failed={failed} total={len(ids)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
