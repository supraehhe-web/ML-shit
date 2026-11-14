"""Equal-weight average of saved weight snapshots (CPU).

A stand-in for an exponential moving average: if high-noise error is limited by
jitter in the weights, the average should beat every individual snapshot. Writes the
averaged state dict and a JSON record of the input hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    total = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        if total is None:
            total = {k: v.clone().double() if v.is_floating_point() else v.clone()
                     for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    total[k] += v.double()
    avg = {k: (v / len(paths)).float() if v.is_floating_point() else v
           for k, v in total.items()}
    out = Path(args.out)
    torch.save(avg, out)
    out.with_suffix(".json").write_text(json.dumps({
        "inputs": {str(p): sha256(p) for p in paths}, "n": len(paths),
        "output_sha256": sha256(out)}, indent=1))
    print("averaged", len(paths), "snapshots ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
