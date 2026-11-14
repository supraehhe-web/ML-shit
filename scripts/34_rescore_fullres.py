"""Rescore a saved CPU sample with the Gate-3 scoring (CPU, validation pair only).

`31_local_diagnosis.py` scores at 128 px against a 128 px reference. `30_gate3_eval.py`
upsamples the 128 px output to the original image size and scores against the
full-resolution reference. This applies the second scoring to a saved sample so the
two numbers can be compared directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diag31", ROOT / "scripts" / "31_local_diagnosis.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(a * 255), 0, 255).astype(np.uint8))


def score(pred: np.ndarray, ref: np.ndarray) -> dict:
    return {"psnr": float(psnr_fn(ref, pred, data_range=1.0)),
            "ssim": float(ssim_fn(ref, pred, data_range=1.0, channel_axis=2, win_size=7))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--manifest", default="/root/local/diag/man/synthetic_pairs.csv")
    ap.add_argument("--data-root", default="/root/local/diag/data")
    args = ap.parse_args()

    val, refs, hairy = diag.load_val(Path(args.manifest), Path(args.data_root), 1, start=args.pair)
    d = np.load(args.npz)
    ref_full = np.asarray(refs[0], dtype=np.float64) / 255.0
    size = refs[0].size

    def up(a: np.ndarray) -> np.ndarray:
        return np.asarray(to_pil(a).resize(size, Image.Resampling.BICUBIC), dtype=np.float64) / 255.0

    res = {
        "pair_id": val["pair_id"][0],
        "reference_size": list(size),
        "scored_at_128": {"output": score(d["final"], d["ref"]), "hairy": score(d["hairy"], d["ref"])},
        "scored_full_res_gate3_style": {
            "output": score(up(d["final"]), ref_full),
            "hairy": score(np.asarray(hairy[0].resize(size, Image.Resampling.BICUBIC),
                                      dtype=np.float64) / 255.0, ref_full),
        },
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip(),
        "npz": args.npz,
    }
    Path(args.npz).with_suffix(".rescore.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
