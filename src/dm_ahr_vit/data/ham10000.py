"""HAM10000 metadata (Harvard Dataverse) and cohort helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from ..config import CLASSES, EXPECTED_C0_COUNTS, EXPECTED_C0_TOTAL

DATAVERSE_DOI = "10.7910/DVN/DBW86T"
DATAVERSE_ACCESS = "https://dataverse.harvard.edu/api/access/datafile/{fid}"
METADATA_FILE_ID = 3282766
MASKS_FILE_ID = 3282767


def read_metadata(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    return df


def class_counts(df: pd.DataFrame) -> dict:
    counts = df["dx"].value_counts().to_dict()
    return {c: int(counts.get(c, 0)) for c in CLASSES}


def expected_comparison(metadata: pd.DataFrame) -> dict:
    counts = class_counts(metadata)
    return {
        "recomputed": counts,
        "expected": EXPECTED_C0_COUNTS,
        "total": int(sum(counts.values())),
        "expected_total": EXPECTED_C0_TOTAL,
        "matches": all(counts[c] == EXPECTED_C0_COUNTS[c] for c in CLASSES),
    }


def extract_masks(zip_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = out_dir / Path(name).name
            if not target.exists():
                target.write_bytes(zf.read(name))
            n += 1
    return {"extracted": n, "out_dir": str(out_dir)}
