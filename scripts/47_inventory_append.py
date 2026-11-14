"""Append rows to reports/artifact_inventory.csv for files added after the last finalize run.

Hashes repository-resident files; large volume-only files (checkpoints) come from
`modal volume ls --json` listings and are recorded with sha256 = skipped_large, the
convention the inventory already uses. Rows whose path is already listed are left alone.

usage: 47_inventory_append.py --glob 'artifacts/logs/*Cv2*' ... [--listing FILE.json ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import sha256_file  # noqa: E402

VOLUME_PREFIX = "/__modal/volumes/vo-TbNx5IITKUKx9iVQZHZpNF/"


def kind_of(rel: str) -> str:
    return "_".join(rel.split("/")[:2]).replace("artifacts_", "stage_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=str(ROOT / "reports" / "artifact_inventory.csv"))
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--listing", action="append", default=[])
    args = ap.parse_args()
    inv = Path(args.inventory)
    with inv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    have = {r["path"] for r in rows}
    new = []
    for pattern in args.glob:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                if rel not in have:
                    new.append({"kind": kind_of(rel), "path": rel, "bytes": p.stat().st_size,
                                "sha256": sha256_file(p)})
    for listing in args.listing:
        for e in json.loads(Path(listing).read_text()):
            if e.get("type") != "file":
                continue
            path = VOLUME_PREFIX + e["filename"]
            if path not in have:
                new.append({"kind": "checkpoints_stage_b_c", "path": path, "bytes": e["size"],
                            "sha256": "skipped_large"})
    for r in new:
        have.add(r["path"])
    with inv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows + new)
    print(f"appended {len(new)} rows; inventory now {len(rows) + len(new)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
