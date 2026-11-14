"""ISIC Archive collection download."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def download_collection(collection_id: int, images_dir: Path, records_path: Path,
                        workers: int = 16, limit: int | None = None) -> dict:
    """Download one ISIC collection, recording per-file status to records.jsonl."""
    images_dir.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    api = f"https://api.isic-archive.com/api/v2/images?collection_id={collection_id}"
    images, page = [], 0
    while True:
        r = requests.get(api, params={"limit": 100, "offset": page * 100}, timeout=60)
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        images.extend(batch)
        page += 1
        if limit and len(images) >= limit:
            images = images[:limit]
            break

    def fetch(img):
        iid = img["isic_id"]
        out = images_dir / f"{iid}.jpg"
        url = f"https://isic-archive.s3.amazonaws.com/images/{iid}.jpg"
        rec = {"isic_id": iid, "url": url, "bytes": None, "sha256": None,
               "status": "failed", "attribution": img.get("attribution"),
               "license": img.get("license"),
               "expected_size": img.get("metadata", {}).get("acquisition", {}).get("pixels")}
        try:
            if out.exists():
                blob = out.read_bytes()
            else:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                blob = resp.content
                out.write_bytes(blob)
            rec.update({"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                        "status": "downloaded"})
        except Exception as exc:
            rec["status"] = f"failed: {exc}"
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(fetch, images))
    with records_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return {"collection": collection_id, "n": len(records),
            "downloaded": sum(r["status"] == "downloaded" for r in records)}
