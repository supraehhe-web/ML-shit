"""Generated wording for the corrected explanation results (single source for every report).

Reads the tables written by scripts/59_xai_stats.py and returns sentences and table rows, so
no explanation number is typed by hand in any generator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..xai.batched import metric_definitions

ARM_NAME = {"A": "A (original)", "B": "B (DullRazor)", "Astar": "A* (128 px control)",
            "Cv2": "Cv2 (rebuilt diffusion, exploratory)", "C": "C (first diffusion run, defective trainer)",
            "Creleased": "Creleased (released checkpoint, exploratory)"}
PRIMARY = "Grad-CAM, last-block ln_1 (primary)"
SENS = "Grad-CAM, patch grid (sensitivity)"
ROLL = "attention rollout"
AS_IS, SHIFTED = "as is", "off-centre (shifted up to 30%)"


def load(logs: Path, split: str = "test") -> dict:
    d = Path(logs) / "xai_v2_analysis"
    return {"geo": pd.read_csv(d / f"xai_geometry_by_arm_{split}.csv"),
            "summary": pd.read_csv(d / f"xai_arm_summary_{split}.csv"),
            "comp": pd.read_csv(d / f"xai_comparisons_{split}.csv"),
            "agree": pd.read_csv(d / f"xai_seed_agreement_{split}.csv")}


def _s(x, metric=None):
    return f"{x:.3f}"


def geometry_sentence(x: dict) -> str:
    g = x["geo"].set_index("arm")
    full = g.loc["A"]
    dif = g.loc["Cv2"] if "Cv2" in g.index else None
    s = (f"Carried into the 224-pixel frame the classifier sees, the lesion covers on average "
         f"{full['mean_mask_area']:.3f} of the image for the arms stored at 600 x 450 (A, B)")
    if dif is not None:
        s += (f" and {dif['mean_mask_area']:.3f} for Cv2 and A*, where {dif['share_images_stored_128']:.1%} of "
              f"images are stored at 128 x 128")
    s += ". With the top 20% of pixels, the best possible lesion coverage (recall ceiling) is " \
         f"{full['recall_ceiling']:.3f} for A and B"
    if dif is not None:
        s += f" and {dif['recall_ceiling']:.3f} for Cv2 and A*"
    s += (", and the best possible share of salient pixels inside the lesion (precision ceiling) is "
          f"{full['precision_ceiling']:.3f}")
    if dif is not None:
        s += f" and {dif['precision_ceiling']:.3f} respectively"
    return s + "."


def baseline_verdicts(x: dict, metric: str, frame: str = AS_IS) -> list[str]:
    """For each map: in how many arms the model is above, below, or not distinguishable from the
    centred Gaussian (95% two-level bootstrap interval of the model-minus-Gaussian difference)."""
    out = []
    t = x["summary"]
    for method in (ROLL, PRIMARY, SENS):
        q = t[(t["frame"] == frame) & (t["metric"] == metric) & (t["method"] == method)]
        if q.empty or "minus_gauss" not in q:
            continue
        above = [r.arm for r in q.itertuples() if r.minus_gauss_ci_low > 0]
        below = [r.arm for r in q.itertuples() if r.minus_gauss_ci_high < 0]
        mixed = [r.arm for r in q.itertuples() if r.minus_gauss_ci_low <= 0 <= r.minus_gauss_ci_high]
        out.append(f"{method}: above the centred Gaussian in {len(above)} of {len(q)} arms"
                   f"{' (' + ', '.join(above) + ')' if above else ''}, below in {len(below)}"
                   f"{' (' + ', '.join(below) + ')' if below else ''}, not distinguishable in {len(mixed)}"
                   f"{' (' + ', '.join(mixed) + ')' if mixed else ''}.")
    return out


def comparison_rows(x: dict, metric: str, frame: str = AS_IS, methods=(ROLL, PRIMARY, SENS)) -> list[list[str]]:
    c = x["comp"]
    rows = []
    for r in c[(c["frame"] == frame) & (c["metric"] == metric) & c["method"].isin(methods)].itertuples():
        rows.append([r.family, f"{r.a} - {r.b}", r.method, f"{r.difference:+.3f}",
                     f"{r.ci_low:+.3f} to {r.ci_high:+.3f}", f"{r.p_holm:.3f}", r.per_seed_differences,
                     "struck: seed spread exceeds the difference" if r.struck else
                     ("seeds agree in sign" if r.seeds_agree_in_sign else "seeds disagree in sign")])
    return rows


def comparison_header() -> list[str]:
    return ["Family", "Comparison", "Map", "Difference", "95% CI (lesions and seeds)", "p (Holm)",
            "Per-seed differences", "Reading"]


def arm_table(x: dict, metric: str, frame: str = AS_IS) -> tuple[list[str], list[list[str]]]:
    t = x["summary"]
    head = ["Arm", "Map", "Seeds", "Mean (seed range)", "Gaussian", "Random pixels", "Random patches",
            "Model - Gaussian (95% CI)"]
    rows = []
    for r in t[(t["frame"] == frame) & (t["metric"] == metric)].itertuples():
        rows.append([ARM_NAME.get(r.arm, r.arm), r.method, str(r.n_seeds),
                     f"{r.mean:.3f} ({r.seed_min:.3f} to {r.seed_max:.3f})",
                     f"{r.baseline_gauss:.3f}", f"{getattr(r, 'baseline_rand_pixel', float('nan')):.3f}",
                     f"{getattr(r, 'baseline_rand_patch', float('nan')):.3f}",
                     f"{r.minus_gauss:+.3f} ({r.minus_gauss_ci_low:+.3f} to {r.minus_gauss_ci_high:+.3f})"])
    return head, rows


def agreement_sentence(x: dict) -> str:
    a = x["agree"]
    parts = []
    for m in (ROLL, PRIMARY, SENS):
        q = a[a["method"] == m]
        parts.append(f"{m}: {q['mean_pairwise_spearman'].min():.2f} to {q['mean_pairwise_spearman'].max():.2f}")
    return ("Per-image lesion-coverage scores agree across independently trained seeds (mean pairwise Spearman "
            "correlation, range over arms) as follows: " + "; ".join(parts) + ".")


def definitions() -> str:
    return metric_definitions(0.2)


def strike_summary(x: dict) -> str:
    c = x["comp"]
    parts = []
    for m in (ROLL, PRIMARY, SENS):
        q = c[(c["method"] == m) & (c["frame"] == AS_IS) & (c["metric"].isin(["recall", "pointing"]))]
        parts.append(f"{m}: {int(q['struck'].sum())} of {len(q)}")
    return ("Rows struck by the pre-registered rule (seed spread larger than the between-arm difference), "
            "recall and pointing, unshifted: " + "; ".join(parts) + ".")
