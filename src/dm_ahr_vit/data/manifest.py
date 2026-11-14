"""File manifests: decode audit, dHash near-duplicate candidates, confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _describe(path: Path, image_id: str) -> dict:
    row = {
        "image_id": image_id, "decode_status": "fail_missing",
        "width": None, "height": None, "mode": None, "format": None,
        "sha256": None, "bytes": None,
    }
    if not path.exists():
        return row
    row["bytes"] = path.stat().st_size
    try:
        with Image.open(path) as fh:
            fh.load()
            w, h = fh.size
            row.update({"decode_status": "ok", "width": w, "height": h,
                        "mode": fh.mode, "format": fh.format})
    except Exception:
        row["decode_status"] = "fail_decode"
        return row
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash on a (size+1)x(size) grayscale grid."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR),
                   dtype=np.int16)
    bits = g[:, 1:] > g[:, :-1]
    v = 0
    for b in bits.ravel():
        v = (v << 1) | int(b)
    return v


def hamming_pairs(hashes: np.ndarray, threshold: int, ids: list) -> list:
    n = len(hashes)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = bin(int(hashes[i]) ^ int(hashes[j])).count("1")
            if d <= threshold:
                out.append({"id_a": ids[i], "id_b": ids[j], "distance": d})
    return out


def confirm_near_duplicates(candidates: pd.DataFrame, man: pd.DataFrame,
                            max_mean_abs_diff: float = 4.0,
                            min_correlation: float = 0.985) -> pd.DataFrame:
    """Pixel-level confirmation of dHash candidates: mean abs diff and correlation."""
    import cv2

    paths = dict(zip(man["image_id"], man["path"]))
    rows = []
    for r in candidates.itertuples(index=False):
        try:
            a = cv2.imread(str(Path(paths[r.id_a])), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(str(Path(paths[r.id_b])), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or a.shape != b.shape:
                rows.append({"id_a": r.id_a, "id_b": r.id_b, "dhash_distance": r.distance,
                             "mean_abs_diff": float("inf"), "correlation": 0.0,
                             "confirmed": False})
                continue
            mad = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())
            va, vb = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
            corr = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else 0.0
            rows.append({"id_a": r.id_a, "id_b": r.id_b, "dhash_distance": r.distance,
                         "mean_abs_diff": mad, "correlation": corr,
                         "confirmed": bool(mad <= max_mean_abs_diff and corr >= min_correlation)})
        except Exception:
            rows.append({"id_a": r.id_a, "id_b": r.id_b, "dhash_distance": r.distance,
                         "mean_abs_diff": float("inf"), "correlation": 0.0, "confirmed": False})
    return pd.DataFrame(rows)


def build_manifest(records_path: Path, images_dir: Path, metadata: pd.DataFrame,
                   out_dir: Path):
    """Per-image manifest from the download records + HAM metadata."""
    records = [json.loads(l) for l in Path(records_path).read_text().splitlines() if l.strip()]
    meta = metadata.set_index("image_id")
    rows, excl = [], []
    for rec in records:
        iid = rec["isic_id"]
        path = images_dir / f"{iid}.jpg"
        d = _describe(path, iid)
        d.update({
            "path": str(path),
            "in_c0": iid in meta.index,
            "cohort": "C0" if iid in meta.index else "C0-extra",
            "license": rec.get("license"),
            "attribution": rec.get("attribution"),
            "expected_bytes": rec.get("expected_size"),
            "download_status": rec.get("status"),
        })
        if iid in meta.index:
            d["dx"] = str(meta.loc[iid, "dx"])
            d["lesion_id"] = str(meta.loc[iid, "lesion_id"])
        else:
            d["dx"] = None
            d["lesion_id"] = None
            excl.append({"image_id": iid, "cohort": "C0-extra",
                         "reason": "not in HAM10000 metadata", "path": str(path)})
        if d["decode_status"] != "ok":
            excl.append({"image_id": iid, "cohort": d["cohort"],
                         "reason": d["decode_status"], "path": str(path)})
        if d["decode_status"] == "ok":
            with Image.open(path) as fh:
                d["dhash"] = dhash(fh)
        else:
            d["dhash"] = None
        rows.append(d)
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "data_manifest.csv", index=False)
    pd.DataFrame(excl).to_csv(out_dir / "exclusions.csv", index=False)
    return df, pd.DataFrame(excl), {"n": len(df)}
