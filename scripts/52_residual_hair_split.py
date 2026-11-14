"""Stage E V2 and V3 (CPU): lesion change on real validation photos; residual hair by split.

V2: for the validation real-hair photos passed in (gate set and dense set), the fraction of
    lesion pixels changed by more than 10 gray levels versus the original, for B and Cv2,
    at 128 px. Hair that crosses the lesion also counts as change, equally for both arms.
V3: detector coverage at 128 px after over before, for Cv2 and B, on hair-present photos from
    three groups: train photos whose strands were harvested, other train photos, validation
    photos. Tests the train/test asymmetry worry: the diffusion model was fine-tuned on
    composites made from the first group and ViT training then sees that group's outputs.
No test photo is read; ids are checked against the split table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_frozen, save_json  # noqa: E402
from dm_ahr_vit.data.hair_composite import HAIR_MIN_COVERAGE  # noqa: E402
from dm_ahr_vit.diffusion.train import _resolve_manifest_path  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir, processed_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging  # noqa: E402

log = setup_logging()
SIZE = 128
MIN_LAYER_FRACTION = 0.004
MIN_COV128 = 0.005


def load128(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC))


def boot_median(x: np.ndarray, reps: int = 2000, seed: int = 0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, (reps, x.size))], axis=1)
    return float(np.median(x)), meds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-gate", required=True)
    ap.add_argument("--ids-dense", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    screen = load_frozen("hair_screen", "configs/hair_screen.json")
    splits = pd.read_csv(manifest_dir() / "splits.csv").set_index("image_id")["split"]
    masks = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv").set_index("image_id")

    # V2 -------------------------------------------------------------------------------
    ids = [i for i in (args.ids_gate + "," + args.ids_dense).split(",") if i]
    bad = [i for i in ids if splits.get(i) != "val"]
    assert not bad, f"non-validation ids: {bad[:3]}"
    rows = []
    for iid in ids:
        if iid not in masks.index or not bool(masks.loc[iid, "has_mask"]):
            continue
        lm = np.array(Image.open(_resolve_manifest_path(masks.loc[iid, "mask_path"])).convert("L"))
        lm = cv2.resize(lm, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127
        o = load128(isic_images_dir() / f"{iid}.jpg").astype(np.int16)
        row = {"image_id": iid, "set": "gate" if iid in args.ids_gate else "dense"}
        for arm, p in (("B", processed_dir("B") / f"{iid}.png"), ("Cv2", processed_dir("Cv2") / f"{iid}.png")):
            a = load128(p).astype(np.int16)
            row[f"lesion_changed_{arm}"] = float((np.abs(a - o).max(axis=2) > 10)[lm].mean())
            row[f"lesion_mae_{arm}"] = float(np.abs(a - o).mean(axis=2)[lm].mean())
        rows.append(row)
    v2 = pd.DataFrame(rows)
    v2.to_csv(artifact_dir("logs") / "stageE_lesion_change_real.csv", index=False)

    # V3 -------------------------------------------------------------------------------
    scr = pd.read_csv(manifest_dir() / "hair_screening.csv")
    layers = pd.read_csv(data_root() / "hair_layers" / "summary.csv")
    harvested = set(layers.loc[(layers["hair_pixels"] > 0) & (layers["fraction"] >= MIN_LAYER_FRACTION)
                               & (layers["split"] == "train"), "image_id"])
    hp = scr[scr["hair_present"].astype(bool)]
    harvested_val = set(layers.loc[(layers["hair_pixels"] > 0) & (layers["fraction"] >= MIN_LAYER_FRACTION)
                                   & (layers["split"] == "val"), "image_id"])
    groups = {"train_harvested": hp[(hp["split"] == "train") & hp["image_id"].isin(harvested)],
              "train_other": hp[(hp["split"] == "train") & ~hp["image_id"].isin(harvested)],
              "val": hp[hp["split"] == "val"],
              # post-hoc (see prereg addendum): same harvest criterion as train_harvested, never trained on
              "val_harvested": hp[(hp["split"] == "val") & hp["image_id"].isin(harvested_val)]}
    rng = np.random.default_rng(0)
    res, meds = [], {}
    for name, g in groups.items():
        pick = sorted(g["image_id"])
        pick = [pick[i] for i in rng.choice(len(pick), min(args.n, len(pick)), replace=False)]
        for iid in pick:
            o = load128(isic_images_dir() / f"{iid}.jpg")
            c0 = detect_hair(o, screen)[1]["coverage"]
            if c0 < MIN_COV128:
                continue
            row = {"group": name, "image_id": iid, "cov_orig": c0}
            for arm, p in (("Cv2", processed_dir("Cv2") / f"{iid}.png"), ("B", processed_dir("B") / f"{iid}.png")):
                row[f"remaining_{arm}"] = detect_hair(load128(p), screen)[1]["coverage"] / c0
            res.append(row)
    v3 = pd.DataFrame(res)
    v3.to_csv(artifact_dir("logs") / "stageE_residual_hair_by_split.csv", index=False)

    summ = {"git_commit": args.commit, "detector_config_hash": screen.get("config_hash"),
            "v2": {"n": int(len(v2)), "gate_photos": int((v2["set"] == "gate").sum()),
                   "dense_photos": int((v2["set"] == "dense").sum())},
            "v3": {"min_coverage_at_128": MIN_COV128}}
    for arm in ("B", "Cv2"):
        summ["v2"][arm] = {"mean_lesion_changed": float(v2[f"lesion_changed_{arm}"].mean()),
                           "mean_lesion_mae": float(v2[f"lesion_mae_{arm}"].mean())}
    d = v2["lesion_changed_Cv2"] - v2["lesion_changed_B"]
    summ["v2"]["paired_Cv2_minus_B_changed"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1))}
    for arm in ("Cv2", "B"):
        summ["v3"][arm] = {}
        for name in groups:
            x = v3.loc[v3["group"] == name, f"remaining_{arm}"].to_numpy()
            med, _ = boot_median(x)
            summ["v3"][arm][name] = {"n": int(x.size), "median_remaining": med}
    for arm in ("Cv2", "B"):
        a = v3.loc[v3["group"] == "train_harvested", f"remaining_{arm}"].to_numpy()
        b = v3.loc[v3["group"] == "val", f"remaining_{arm}"].to_numpy()
        _, ma = boot_median(a, seed=1)
        _, mb = boot_median(b, seed=2)
        diff = ma - mb
        lo, hi = np.percentile(diff, [2.5, 97.5])
        obs = float(np.median(a) - np.median(b))
        summ["v3"][arm]["harvested_minus_val"] = {
            "median_diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "asymmetry_by_rule": bool(obs < -0.05 and hi < 0)}
        c = v3.loc[v3["group"] == "val_harvested", f"remaining_{arm}"].to_numpy()
        _, mc = boot_median(c, seed=3)
        d2 = ma - mc
        lo2, hi2 = np.percentile(d2, [2.5, 97.5])
        summ["v3"][arm]["posthoc_train_harvested_minus_val_harvested"] = {
            "median_diff": float(np.median(a) - np.median(c)), "ci_low": float(lo2),
            "ci_high": float(hi2), "n_val_harvested": int(c.size)}
    save_json(summ, artifact_dir("logs") / "stageE_residual_hair_and_lesion_change.json")
    log.info("V2 %s", summ["v2"])
    log.info("V3 %s", summ["v3"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
