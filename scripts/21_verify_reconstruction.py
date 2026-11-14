"""Verify the reconstructed data package reproduces the recorded predictions.

Compares a re-evaluation written under a scratch artifact root against the
recorded CSV, so data-loading fidelity is proven rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import save_json  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()

FLOAT_TOL = 1e-6


def macro_f1(df: pd.DataFrame) -> float:
    return float(f1_score(df["label"], df["pred"], average="macro"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recorded-root", default="/vol/artifacts")
    ap.add_argument("--repro-root", default="/vol/artifacts/repro_check")
    args = ap.parse_args()

    name = f"pred_{args.arm}_seed{args.seed}_full.csv"
    recorded = pd.read_csv(Path(args.recorded_root) / "predictions" / name)
    repro = pd.read_csv(Path(args.repro_root) / "predictions" / name)

    m = recorded.merge(repro, on="image_id", suffixes=("_rec", "_rep"))
    same_ids = len(m) == len(recorded) == len(repro)
    label_diff = int((m["label_rec"] != m["label_rep"]).sum())
    pred_diff = int((m["pred_rec"] != m["pred_rep"]).sum())
    prob_abs = float((m["prob_max_rec"] - m["prob_max_rep"]).abs().max())
    f1_rec, f1_rep = macro_f1(recorded), macro_f1(repro)

    ok = (same_ids and label_diff == 0 and pred_diff == 0
          and prob_abs <= FLOAT_TOL and abs(f1_rec - f1_rep) <= FLOAT_TOL)
    result = {
        "arm": args.arm,
        "seed": args.seed,
        "rows": int(len(recorded)),
        "same_image_ids": bool(same_ids),
        "label_mismatches": label_diff,
        "prediction_mismatches": pred_diff,
        "max_prob_absdiff": prob_abs,
        "macro_f1_recorded": f1_rec,
        "macro_f1_reproduced": f1_rep,
        "verified": bool(ok),
    }
    save_json(result, Path(args.recorded_root) / "logs" / "repro_reconstruction.json")
    print(result)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
