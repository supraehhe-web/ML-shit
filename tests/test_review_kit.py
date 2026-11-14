import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "54_review_kit.py"


def _mod():
    spec = importlib.util.spec_from_file_location("review_kit", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _kit(tmp_path, n=12):
    rng = np.random.default_rng(0)
    key, rate = [], []
    for k in range(n):
        order = rng.permutation(3)
        row = {"photo": f"review_{k:03d}", "image_id": f"ISIC_{k}"}
        for slot, ai in enumerate(order):
            row["XYZ"[slot]] = ("B", "Cv2", "hairy_128")[ai]
            rate.append({"photo": row["photo"], "candidate": "XYZ"[slot]})
        key.append(row)
    pd.DataFrame(key).to_csv(tmp_path / "key_DO_NOT_OPEN_BEFORE_RATING.csv", index=False)
    return pd.DataFrame(key), pd.DataFrame(rate)


def test_analyze_recovers_arm_effects_from_blinded_letters(tmp_path):
    m = _mod()
    key, rate = _kit(tmp_path)
    arm_of = {(r.photo, l): getattr(r, l) for r in key.itertuples() for l in "XYZ"}
    hair = {"B": 0, "Cv2": 1, "hairy_128": 3}
    les = {"B": 1, "Cv2": 3, "hairy_128": 3}
    rate["hair_remaining_0to3"] = [hair[arm_of[(p, c)]] for p, c in zip(rate.photo, rate.candidate)]
    rate["lesion_preserved_0to3"] = [les[arm_of[(p, c)]] for p, c in zip(rate.photo, rate.candidate)]
    path = tmp_path / "ratings.csv"
    rate.to_csv(path, index=False)
    res = m.analyze(tmp_path, path)
    assert res["arms"]["B"]["hair_remaining_0to3"]["mean"] == 0
    assert res["arms"]["hairy_128"]["hair_remaining_0to3"]["mean"] == 3
    d = res["paired_Cv2_minus_B"]["lesion_preserved_0to3"]
    assert d["mean"] == 2 and d["ci_low"] > 0


def test_analyze_rejects_incomplete_or_out_of_range_ratings(tmp_path):
    m = _mod()
    _, rate = _kit(tmp_path)
    rate["hair_remaining_0to3"] = 1
    rate["lesion_preserved_0to3"] = np.nan
    p = tmp_path / "r.csv"
    rate.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        m.analyze(tmp_path, p)
    rate["lesion_preserved_0to3"] = 5
    rate.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        m.analyze(tmp_path, p)
