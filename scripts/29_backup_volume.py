"""Recursively copy a Modal volume to local disk, in parallel.

Used to preserve the previous account's volume before switching accounts.
Writes a manifest with sizes and SHA-256 so the copy can be verified and later
restored into a new volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal


def walk(vol, root: str) -> list[tuple[str, int]]:
    """`Volume.iterdir` already returns the whole subtree, so one call is enough."""
    return [(e.path, int(e.size)) for e in vol.iterdir(root)
            if e.type != modal.volume.FileEntryType.DIRECTORY]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="dmahr-vit-vol")
    ap.add_argument("--remote-root", default="/")
    ap.add_argument("--local-root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-prefix", action="append", default=[])
    args = ap.parse_args()
    vol = modal.Volume.from_name(args.volume)
    local = Path(args.local_root)
    local.mkdir(parents=True, exist_ok=True)
    files = [(p, s) for p, s in walk(vol, args.remote_root)
             if not any(p.lstrip("/").startswith(x) for x in args.skip_prefix)]
    total = sum(s for _, s in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB", flush=True)

    def fetch(item):
        remote, size = item
        dest = local / remote.lstrip("/")
        if dest.exists() and dest.stat().st_size == size:
            return remote, size, "cached"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = b"".join(vol.read_file(remote))
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return remote, len(data), "ok"
        except Exception as exc:  # noqa: BLE001
            return remote, 0, f"FAILED {exc!r}"

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (remote, size, status) in enumerate(ex.map(fetch, files), 1):
            if status.startswith("FAILED"):
                failed += 1
                print(f"{status} {remote}", file=sys.stderr, flush=True)
            else:
                done += 1
            if i % 200 == 0:
                print(f"{i}/{len(files)} ok={done} failed={failed}", flush=True)

    manifest = []
    for remote, size in files:
        dest = local / remote.lstrip("/")
        if dest.exists():
            h = hashlib.sha256()
            with dest.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            manifest.append({"path": remote, "size": dest.stat().st_size, "sha256": h.hexdigest()})
    (local / "_backup_manifest.json").write_text(json.dumps(
        {"volume": args.volume, "files": manifest, "n": len(manifest),
         "bytes": sum(m["size"] for m in manifest)}, indent=1))
    print(f"done={done} failed={failed} manifest={len(manifest)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
