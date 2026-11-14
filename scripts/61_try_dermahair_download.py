"""Try an anonymous download of the public DERMAHAIR Kaggle dataset (no credentials supplied)."""

import sys

import kagglehub

handle = sys.argv[1] if len(sys.argv) > 1 else "riotulab/skin-cancer-hair-removal"
try:
    path = kagglehub.dataset_download(handle)
    print("DOWNLOADED", path)
except Exception as exc:  # noqa: BLE001
    print("FAILED", type(exc).__name__, str(exc)[:400])
