"""Statistics for the corrected explanation analysis (rules: reports/prereg_v5_client_asks.md s.1).

Reads artifacts/logs/xai_v2/xai_{arm}_seed{seed}_{split}.csv; writes tables to
artifacts/logs/xai_v2_analysis/ and the generated report reports/xai_corrected_results.md.
Two-level bootstrap (lesion groups shared across arms; seeds resampled within each arm), 10,000
replicates, seed 20261113, per-seed differences shown next to it, Holm within family, and the
pre-registered strike rule for rows whose seed spread exceeds the between-arm difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import save_json  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.stats.pooled import compare, draw_weights, two_level  # noqa: E402
from dm_ahr_vit.xai.batched import metric_definitions  # noqa: E402

REPS, BSEED = 10000, 20261113
METHODS = {"rollout": "attention rollout", "gradcam_ln1": "Grad-CAM, last-block ln_1 (primary)",
           "gradcam_patch": "Grad-CAM, patch grid (sensitivity)"}
METRICS = ("recall", "precision", "pointing", "centroid")
FRAMES = {"": "as is", "tr_": "off-centre (shifted up to 30%)"}
ARMS = ["A", "B", "Astar", "Cv2", "C", "Creleased"]
PAIRS = [("A", "B", "planned"), ("A", "Astar", "exploratory"), ("A", "Cv2", "exploratory"),
         ("B", "Cv2", "exploratory"), ("Cv2", "Astar", "exploratory")]
ARM_LABEL = {"A": "A original", "B": "B DullRazor", "Astar": "A* 128 px control",
             "Cv2": "Cv2 rebuilt diffusion (exploratory)", "C": "C first diffusion (defective trainer)",
             "Creleased": "Creleased released checkpoint (exploratory)"}


def load(split: str, base: Path):
    data, prov = {}, {}
    for arm in ARMS:
        for f in sorted(base.glob(f"xai_{arm}_seed*_{split}.csv")):
            seed = int(f.stem.split("_seed")[1].split("_")[0])
            data[(arm, seed)] = pd.read_csv(f).sort_values("image_id").reset_index(drop=True)
            prov[(arm, seed)] = json.loads(f.with_suffix(".json").read_text())
            prov[(arm, seed)]["csv_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return data, prov


def group_index(df: pd.DataFrame):
    groups = sorted(df["group_id"].unique())
    gi = df["group_id"].map({g: i for i, g in enumerate(groups)}).to_numpy()
    return groups, gi


def sums(v: np.ndarray, gi: np.ndarray, g: int):
    ok = ~np.isnan(v)
    s = np.bincount(gi[ok], weights=v[ok], minlength=g)
    n = np.bincount(gi[ok], minlength=g).astype(float)
    return s, n


def boot_scores(vecs: list[np.ndarray], gi: np.ndarray, g: int, w: np.ndarray) -> np.ndarray:
    sn = [sums(v, gi, g) for v in vecs]
    S = np.stack([a for a, _ in sn])
    N = np.stack([b for _, b in sn])
    return ((w @ S.T) / np.maximum(w @ N.T, 1.0)).T            # (seeds, reps)


def pooled(vecs_a, vecs_b, gi, g, w, rng) -> dict:
    sa, sb = boot_scores(vecs_a, gi, g, w), boot_scores(vecs_b, gi, g, w)
    obs_a = float(np.mean([np.nanmean(v) for v in vecs_a]))
    obs_b = float(np.mean([np.nanmean(v) for v in vecs_b]))
    return compare(two_level(sa, rng), two_level(sb, rng), obs_a, obs_b)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) and not np.isnan(v)
                                     else ("" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--src", default=str(ROOT / "artifacts" / "logs" / "xai_v2"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "logs" / "xai_v2_analysis"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, prov = load(args.split, Path(args.src))
    arms = [a for a in ARMS if any(k[0] == a for k in data)]
    seeds_of = {a: sorted(s for (x, s) in data if x == a) for a in arms}
    first = next(iter(data.values()))
    assert all((d["image_id"].to_numpy() == first["image_id"].to_numpy()).all() for d in data.values())
    groups, gi = group_index(first)
    g = len(groups)
    w = draw_weights(g, REPS, BSEED)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()

    def vec(arm, seed, col):
        return data[(arm, seed)][col].to_numpy(float)

    # geometry: lesion size and position in the model's frame, per arm
    geo = []
    for a in arms:
        d = data[(a, seeds_of[a][0])]
        area = d["mask_area"].to_numpy()
        geo.append({"arm": a, "n": len(d), "share_images_stored_128": prov[(a, seeds_of[a][0])]["share_stored_128"],
                    "mean_mask_area": area.mean(), "sd_mask_area": area.std(ddof=1),
                    "recall_ceiling": np.minimum(1, 0.2 / np.maximum(area, 1e-9)).mean(),
                    "precision_ceiling": np.minimum(1, area / 0.2).mean(),
                    "centroid_offset_mean": d["centroid_offset"].mean(),
                    "centroid_offset_sd": d["centroid_offset"].std(ddof=1),
                    "share_masks_containing_centre": d["contains_centre"].mean(),
                    "tr_mean_mask_area": d["tr_mask_area"].mean()})
    geo = pd.DataFrame(geo)
    geo.to_csv(out / f"xai_geometry_by_arm_{args.split}.csv", index=False)

    # per arm: means over seeds, spreads, baselines, model-minus-baseline
    rows = []
    for a in arms:
        for fr, frl in FRAMES.items():
            for m, ml in METHODS.items():
                for q in METRICS:
                    col = f"{fr}{m}_{q}"
                    vecs = [vec(a, s, col) for s in seeds_of[a]]
                    per_seed = [np.nanmean(v) for v in vecs]
                    row = {"arm": a, "frame": frl, "method": ml, "metric": q, "n_seeds": len(vecs),
                           "mean": float(np.mean(per_seed)),
                           "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                           "seed_min": float(np.min(per_seed)), "seed_max": float(np.max(per_seed)),
                           "per_seed": ";".join(f"{s}:{x:.3f}" for s, x in zip(seeds_of[a], per_seed))}
                    for bname, bcol in (("gauss", f"{fr}gauss_{q}"), ("rand_pixel", f"{fr}rand_pixel_{q}"),
                                        ("rand_patch", f"{fr}rand_patch_{q}")):
                        if bcol not in data[(a, seeds_of[a][0])]:
                            continue
                        bv = vec(a, seeds_of[a][0], bcol)
                        row[f"baseline_{bname}"] = float(np.nanmean(bv))
                        if bname in ("gauss", "rand_patch") and q != "centroid":
                            diff = [v - bv for v in vecs]
                            sc = boot_scores(diff, gi, g, w)
                            dist = two_level(sc, np.random.default_rng(BSEED))
                            lo, hi = np.percentile(dist, [2.5, 97.5])
                            row[f"minus_{bname}"] = float(np.mean([np.nanmean(d) for d in diff]))
                            row[f"minus_{bname}_ci_low"], row[f"minus_{bname}_ci_high"] = float(lo), float(hi)
                    rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"xai_arm_summary_{args.split}.csv", index=False)

    # comparisons with strike rule
    comp = []
    for a, b, fam in PAIRS:
        if a not in arms or b not in arms:
            continue
        common = sorted(set(seeds_of[a]) & set(seeds_of[b]))
        for fr, frl in FRAMES.items():
            for m, ml in METHODS.items():
                for q in METRICS:
                    col = f"{fr}{m}_{q}"
                    va = [vec(a, s, col) for s in seeds_of[a]]
                    vb = [vec(b, s, col) for s in seeds_of[b]]
                    res = pooled(va, vb, gi, g, w, np.random.default_rng(BSEED))
                    ma = [np.nanmean(v) for v in va]
                    mb = [np.nanmean(v) for v in vb]
                    per_seed = [np.nanmean(vec(a, s, col)) - np.nanmean(vec(b, s, col)) for s in common]
                    rng_a, rng_b = max(ma) - min(ma), max(mb) - min(mb)
                    struck = max(rng_a, rng_b) > abs(res["difference"])
                    comp.append({"family": fam, "a": a, "b": b, "frame": frl, "method": ml, "metric": q,
                                 "difference": res["difference"], "ci_low": res["ci_low"],
                                 "ci_high": res["ci_high"], "p_raw": res["p_raw"],
                                 "per_seed_differences": ";".join(f"{x:+.3f}" for x in per_seed),
                                 "seeds_agree_in_sign": bool(len(per_seed) > 0 and (all(x > 0 for x in per_seed)
                                                                                    or all(x < 0 for x in per_seed))),
                                 "seed_range_a": rng_a, "seed_range_b": rng_b, "struck": bool(struck)})
    comp = pd.DataFrame(comp)
    comp["p_holm"] = np.nan
    for _, idx in comp.groupby(["family", "frame", "method", "metric"]).groups.items():
        comp.loc[idx, "p_holm"] = holm(comp.loc[idx, "p_raw"].tolist())
    comp.to_csv(out / f"xai_comparisons_{args.split}.csv", index=False)

    # cross-seed agreement of per-image scores (recall), per arm and map
    agree = []
    for a in arms:
        for m, ml in METHODS.items():
            col = f"{m}_recall"
            rs = [spearmanr(vec(a, s1, col), vec(a, s2, col), nan_policy="omit")[0]
                  for s1, s2 in combinations(seeds_of[a], 2)]
            means = [np.nanmean(vec(a, s, col)) for s in seeds_of[a]]
            agree.append({"arm": a, "method": ml, "n_seeds": len(means),
                          "mean_pairwise_spearman": float(np.mean(rs)) if rs else np.nan,
                          "seed_range_of_mean_recall": float(max(means) - min(means))})
    agree = pd.DataFrame(agree)
    agree.to_csv(out / f"xai_seed_agreement_{args.split}.csv", index=False)

    save_json({"split": args.split, "git_commit_of_analysis": commit, "reps": REPS, "seed": BSEED,
               "n_images": int(len(first)), "n_lesion_groups": g,
               "runs": {f"{a}/{s}": {"git_commit": p["git_commit"], "csv_sha256": p["csv_sha256"],
                                     "checkpoint_sha256": p["sha256"]["checkpoint"],
                                     "mask_manifest_sha256": p["sha256"]["mask_manifest_x1.csv"],
                                     "accuracy": p["accuracy_on_split"]}
                        for (a, s), p in sorted(prov.items())}},
              out / f"xai_provenance_{args.split}.json")

    if args.split == "test":
        write_report(geo, summary, comp, agree, commit, g, len(first))
    print(geo.round(3).to_string())
    print(agree.round(3).to_string())
    return 0


def write_report(geo, summary, comp, agree, commit, g, n) -> None:
    def pick(df, **kw):
        for k, v in kw.items():
            df = df[df[k] == v]
        return df

    L = ["# Corrected explanation results (generated; do not edit by hand)", "",
         "Research prototype only. Not clinically validated and not intended for diagnosis or patient care.", "",
         f"Generated by `scripts/59_xai_stats.py` at commit `{commit[:10]}` from `artifacts/logs/xai_v2/*_test.csv` "
         f"({n:,} test images with lesion masks, {g:,} lesion groups). Rules: `reports/prereg_v5_client_asks.md` section 1. "
         "Supersedes the explanation rows of `reports/stageE_results.md` section 6 (kept as `legacy_frame`).", "",
         "## Metric definitions (generated from the function docstrings)", "", metric_definitions(0.2), "",
         "## Lesion size and position in the model's frame, per arm", "",
         md_table(geo.rename(columns={"share_images_stored_128": "share stored 128", "mean_mask_area": "mask area",
                                      "sd_mask_area": "area SD", "recall_ceiling": "recall ceiling",
                                      "precision_ceiling": "precision ceiling", "centroid_offset_mean": "centroid offset",
                                      "centroid_offset_sd": "offset SD", "share_masks_containing_centre": "contains centre"})
                  [["arm", "share stored 128", "mask area", "area SD", "recall ceiling", "precision ceiling",
                    "centroid offset", "offset SD", "contains centre"]]), "",
         "Offsets are in image widths. The recall ceiling is the mean of min(1, 0.2 / area); the precision ceiling the mean of "
         "min(1, area / 0.2). Pointing chance for random maps equals the mean mask area; for the centre-fixed Gaussian it is the "
         "share of masks containing the image centre.", ""]
    for fr in FRAMES.values():
        L += [f"## Per-arm results, {fr} (mean over seeds; baselines on the same masks)", ""]
        for q in ("recall", "pointing", "precision"):
            t = pick(summary, frame=fr, metric=q)[["arm", "method", "n_seeds", "mean", "seed_min", "seed_max",
                                                    "baseline_gauss", "baseline_rand_pixel", "baseline_rand_patch",
                                                    "minus_gauss", "minus_gauss_ci_low", "minus_gauss_ci_high"]]
            L += [f"### {q}", "", md_table(t), ""]
    L += ["## Comparisons (two-level bootstrap with per-seed differences; struck rows are not interpretable)", ""]
    for fr in FRAMES.values():
        for q in ("recall", "pointing"):
            t = pick(comp, frame=fr, metric=q)[["family", "a", "b", "method", "difference", "ci_low", "ci_high",
                                                 "p_holm", "per_seed_differences", "struck"]]
            L += [f"### {q}, {fr}", "", md_table(t), ""]
    L += ["## Cross-seed agreement of per-image recall", "", md_table(agree), ""]
    Path("reports/xai_corrected_results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
