"""Saliency geometry audit and baseline re-analysis (CPU only, test masks).

Finding behind this script: the original XAI run scored saliency maps (in model-input
space: shorter side resized to 255, centre-cropped to 224) against lesion masks that
were simply stretched to 224x224. Those are different geometries, and the mismatch
differs by arm (Arms A/B see a non-square centre crop; hair-present A*/C images are
128x128 stretched copies). This script quantifies the mismatch from the masks alone and
recomputes every baseline (centred Gaussian, random-argmax pointing chance, overlap
ceiling) under the *aligned* mask for each geometry.

Needs the HAM10000 lesion masks (Tschandl et al., Harvard Dataverse doi:10.7910/DVN/DBW86T);
pass --mask-dir to a local copy. It never reads a model or a saliency map.

Writes artifacts/logs/saliency_geometry.csv and saliency_geometry_summary.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import cluster_mean_ci  # noqa: E402
from dm_ahr_vit.xai.geometry import model_space, retained_fraction, stretched_224  # noqa: E402
from dm_ahr_vit.xai.vit_xai import (  # noqa: E402
    gaussian_chance_map,
    lesion_overlap,
    mask_centroid_offset,
    mask_contains_centre,
    overlap_ceiling,
    pointing_chance,
    pointing_game,
    random_argmax_pointing,
    seeded_shift,
    translate_array,
)

SIZE = 224


def score_gaussian(gauss: np.ndarray, m: np.ndarray, fraction: float) -> tuple[float, float]:
    mb = m > 0
    return lesion_overlap(gauss, mb, fraction), pointing_game(gauss, mb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    mask_dir = Path(args.mask_dir)
    fraction = float(load_config("configs/sampling.json")["xai_top_fraction"])

    x1 = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
    x1 = x1[(x1["split"] == "test") & (x1["status"] == "ok")].sort_values("image_id")
    hair = pd.read_csv(manifest_dir() / "hair_screening.csv").set_index("image_id")
    if args.limit:
        x1 = x1.head(args.limit)
    v1 = {}
    for arm in ("A", "A*", "B", "C"):
        src = "astar" if arm == "A*" else arm
        p = artifact_dir("logs") / f"xai_localization_{'Astar' if arm == 'A*' else arm}_seed42_X1.csv"
        if p.exists():
            v1[arm] = pd.read_csv(p).set_index("image_id")

    gauss = gaussian_chance_map(SIZE)
    rows = []
    for _, r in x1.iterrows():
        iid = r["image_id"]
        f = mask_dir / f"{iid}_segmentation.png"
        if not f.exists():
            raise FileNotFoundError(f)
        mask = (np.asarray(Image.open(f).convert("L")) > 127).astype(np.uint8)
        w, h = int(hair.loc[iid, "width"]), int(hair.loc[iid, "height"])
        present = bool(hair.loc[iid, "hair_present"])
        geoms = {"stretched_v1": stretched_224(mask), "native": model_space(mask, (w, h)),
                 "squashed128": model_space(mask, (128, 128))}
        row = {"image_id": iid, "group_id": r["group_id"], "width": w, "height": h, "hair_present": present}
        dy, dx = seeded_shift(iid, SIZE, 0.3)
        row["shift_dy"], row["shift_dx"] = dy, dx
        for name, m in geoms.items():
            mb = m > 0
            row[f"area_{name}"] = float(mb.mean())
            row[f"centroid_dy_{name}"], row[f"centroid_dx_{name}"] = mask_centroid_offset(mb)
            row[f"contains_centre_{name}"] = mask_contains_centre(mb)
            row[f"ceiling_{name}"] = overlap_ceiling(mb, fraction)
            row[f"pointing_chance_{name}"] = pointing_chance(mb)
            row[f"mc_pointing_{name}"] = random_argmax_pointing(mb, n=2000, seed=len(rows))
            row[f"gauss_overlap_{name}"], row[f"gauss_pointing_{name}"] = score_gaussian(gauss, m, fraction)
            shifted = translate_array(m, dy, dx)
            kept = float((shifted > 0).sum() / max((m > 0).sum(), 1))
            row[f"shift_kept_{name}"] = kept
            if kept >= 0.5:
                row[f"gauss_overlap_shift_{name}"], row[f"gauss_pointing_shift_{name}"] = \
                    score_gaussian(gauss, shifted, fraction)
            else:
                row[f"gauss_overlap_shift_{name}"] = row[f"gauss_pointing_shift_{name}"] = np.nan
        row["retained_native"] = retained_fraction(mask, (w, h))
        row["retained_squashed128"] = retained_fraction(mask, (128, 128))
        if "A" in v1 and iid in v1["A"].index:
            row["csv_area_v1"] = float(v1["A"].loc[iid, "lesion_area_frac"])
            row["csv_gauss_overlap_v1"] = float(v1["A"].loc[iid, "chance_gaussian_overlap20"])
            row["csv_gauss_pointing_v1"] = float(v1["A"].loc[iid, "chance_gaussian_pointing"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(artifact_dir("logs") / "saliency_geometry.csv", index=False)

    def agg(col):
        c = cluster_mean_ci(df[col], df["group_id"], reps=2000)
        return {"mean": c["mean"], "ci_low": c["ci_low"], "ci_high": c["ci_high"]}

    summ: dict = {"n_masks": int(len(df)), "mask_source": "HAM10000_segmentations_lesion_tschandl.zip, "
                  "Harvard Dataverse doi:10.7910/DVN/DBW86T", "fraction": fraction}
    if "csv_area_v1" in df:
        summ["cross_check_against_modal_run"] = {
            "max_abs_diff_area_v1": float((df["area_stretched_v1"] - df["csv_area_v1"]).abs().max()),
            "max_abs_diff_gauss_overlap_v1": float((df["gauss_overlap_stretched_v1"] - df["csv_gauss_overlap_v1"]).abs().max()),
            "max_abs_diff_gauss_pointing_v1": float((df["gauss_pointing_stretched_v1"] - df["csv_gauss_pointing_v1"]).abs().max()),
            "note": "agreement shows the locally downloaded masks equal the masks used on Modal"}
    for g in ("stretched_v1", "native", "squashed128"):
        summ[g] = {
            "mean_area": float(df[f"area_{g}"].mean()), "sd_area": float(df[f"area_{g}"].std(ddof=1)),
            "centroid_offset_dy_mean": float(df[f"centroid_dy_{g}"].mean()),
            "centroid_offset_dx_mean": float(df[f"centroid_dx_{g}"].mean()),
            "centroid_offset_dy_sd": float(df[f"centroid_dy_{g}"].std(ddof=1)),
            "centroid_offset_dx_sd": float(df[f"centroid_dx_{g}"].std(ddof=1)),
            "mean_abs_centroid_offset": float(np.nanmean(np.hypot(df[f"centroid_dy_{g}"], df[f"centroid_dx_{g}"]))),
            "fraction_masks_containing_centre": float(df[f"contains_centre_{g}"].mean()),
            "mean_ceiling": float(df[f"ceiling_{g}"].mean()),
            "mean_pointing_chance_analytic": float(df[f"pointing_chance_{g}"].mean()),
            "mean_pointing_chance_monte_carlo": float(df[f"mc_pointing_{g}"].mean()),
            "gaussian_overlap": agg(f"gauss_overlap_{g}"), "gaussian_pointing": agg(f"gauss_pointing_{g}"),
            "gaussian_overlap_shifted": {"mean": float(df[f"gauss_overlap_shift_{g}"].mean()),
                                         "n_kept": int(df[f"gauss_overlap_shift_{g}"].notna().sum())},
            "gaussian_pointing_shifted": {"mean": float(df[f"gauss_pointing_shift_{g}"].mean())},
        }
    summ["lesion_retained_by_centre_crop"] = {
        "native_mean": float(df["retained_native"].mean()),
        "native_min": float(df["retained_native"].min()),
        "squashed128_mean": float(df["retained_squashed128"].mean())}
    summ["hair_present_fraction_in_test_x1"] = float(df["hair_present"].mean())
    save_json(summ, artifact_dir("logs") / "saliency_geometry_summary.json")
    import json as _json
    print(_json.dumps(summ, indent=1)[:4500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
