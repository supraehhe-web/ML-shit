"""Figures for the paper draft and the reports, generated from artifacts only.

Writes PNGs to artifacts/figures/. A figure whose inputs are missing is skipped and listed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / "logs"
OUT = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"A": "#1f77b4", "B": "#d62728", "Astar": "#2ca02c", "Cv2": "#9467bd", "C": "#7f7f7f", "Creleased": "#8c564b"}
NAME = {"A": "A original", "B": "B DullRazor", "Astar": "A* 128 px", "Cv2": "Cv2 diffusion", "C": "C first run",
        "Creleased": "Creleased"}
made, skipped = [], []


def save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    made.append(str(p.relative_to(ROOT)))


def fig_macro_f1() -> None:
    ps = pd.read_csv(LOGS / "pooled_per_seed.csv")
    ps = ps[ps["subset"] == "full"]
    arms = [a for a in ("A", "Astar", "B", "Cv2") if a in set(ps["arm"])]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, a in enumerate(arms):
        v = ps[ps["arm"] == a]["macro_f1"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[a], s=18, alpha=0.8)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1.5)
    ax.set_xticks(range(len(arms)), [NAME[a] + ("\n(exploratory)" if a == "Cv2" else "") for a in arms], fontsize=8)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Eight seeds per arm; bar = mean", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_macro_f1_by_seed.png")


def fig_image_quality() -> None:
    q = json.loads((LOGS / "image_quality_test_Cv2.json").read_text())["means"]
    items = [("Hairy input", q["hairy_input_psnr"], q["hairy_input_ssim"], "#999999"),
             ("DullRazor (B)", q["B_psnr"], q["B_ssim"], COL["B"]),
             ("Cv2 diffusion", q["Cv2_psnr"], q["Cv2_ssim"], COL["Cv2"]),
             ("Released ckpt", q["Creleased_psnr"], q["Creleased_ssim"], COL["Creleased"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, k, lab in ((axs[0], 1, "PSNR (dB)"), (axs[1], 2, "SSIM")):
        ax.bar([i[0] for i in items], [i[k] for i in items], color=[i[3] for i in items])
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("186 synthetic test pairs (synthetic hair on hair-free photographs)", fontsize=9)
    save(fig, "fig_image_quality_synthetic.png")


def fig_xai() -> None:
    s = pd.read_csv(LOGS / "xai_v2_analysis" / "xai_arm_summary_test.csv")
    g = pd.read_csv(LOGS / "xai_v2_analysis" / "xai_geometry_by_arm_test.csv").set_index("arm")
    maps = ["attention rollout", "Grad-CAM, last-block ln_1 (primary)", "Grad-CAM, patch grid (sensitivity)"]
    arms = [a for a in ("A", "Astar", "B", "Cv2") if a in set(s["arm"])]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.4), sharey="row")
    for r, metric in enumerate(("recall", "pointing")):
        for c, m in enumerate(maps):
            ax = axs[r, c]
            for frame, mk in (("as is", "o"), ("off-centre (shifted up to 30%)", "s")):
                q = s[(s["metric"] == metric) & (s["method"] == m) & (s["frame"] == frame)].set_index("arm")
                x = np.arange(len(arms)) + (0.12 if mk == "s" else -0.12)
                ax.errorbar(x, [q.loc[a, "mean"] for a in arms],
                            yerr=[[q.loc[a, "mean"] - q.loc[a, "seed_min"] for a in arms],
                                  [q.loc[a, "seed_max"] - q.loc[a, "mean"] for a in arms]],
                            fmt=mk, color="black" if mk == "o" else "#c05a33", ms=5, capsize=2,
                            label="as is" if mk == "o" else "off-centre")
                ax.scatter(x, [q.loc[a, "baseline_gauss"] for a in arms], marker="_", s=120, color="#1f77b4",
                           label="centred Gaussian" if (r, c, mk) == (0, 0, "o") else None)
            if metric == "recall":
                ax.scatter(np.arange(len(arms)), [g.loc[a, "recall_ceiling"] for a in arms], marker="^", s=25,
                           color="#2ca02c", label="ceiling" if c == 0 else None)
            ax.set_xticks(range(len(arms)), [NAME[a] for a in arms], fontsize=7, rotation=20)
            ax.set_title(m.replace(" (sensitivity)", "\n(sensitivity)").replace(" (primary)", "\n(primary)"),
                         fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        axs[r, 0].set_ylabel({"recall": "Lesion coverage (recall)", "pointing": "Pointing"}[metric])
    axs[0, 0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Explanation scores in the classifier's frame; bars = seed range; blue dash = centred Gaussian", fontsize=9)
    save(fig, "fig_xai_by_arm.png")


def fig_learning_curve() -> None:
    d = LOGS / "learning_curve"
    rows = []
    for f in sorted(d.glob("it*.json")):
        j = json.loads(f.read_text())
        rows.append({"iteration": int(f.stem[2:]), **{f"mae_{k}": v for k, v in j["one_step_eps_mae"].items()},
                     "pure_gain": j["pure_noise_2000"]["gain_db"], "partial_gain": j["partial_t0_600"]["gain_db"]})
    for tag, ls in (("lc", "-"), ("lcnc", "--")):
        pass
    df = pd.DataFrame(rows).sort_values("iteration")
    if df.empty:
        raise FileNotFoundError("no learning-curve JSON")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    for k in ("500", "1000", "1500", "1900"):
        axs[0].plot(df["iteration"], df[f"mae_{k}"], marker="o", label=f"t={k}")
    axs[0].set_title("One-step epsilon error (validation)", fontsize=9)
    axs[0].legend(fontsize=7)
    axs[1].plot(df["iteration"], df["partial_gain"], marker="o", color="#2ca02c")
    axs[1].set_title("Partial-noise sampling, gain over hairy input (dB)", fontsize=9)
    axs[2].plot(df["iteration"], df["pure_gain"], marker="o", color="#d62728")
    axs[2].axhline(0, color="black", lw=0.8)
    axs[2].axhline(2, color="grey", ls=":", lw=0.8)
    axs[2].set_title("Pure-noise 2,000-step sampling, gain (dB)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("training iterations")
        ax.grid(alpha=0.3)
    save(fig, "fig_learning_curve.png")


def fig_dermahair() -> None:
    j = json.loads((LOGS / "dermahair_repro.json").read_text())
    r = j["results"]
    keys = [k for k in r if k.startswith(("respaced_", "full_", "truncated_"))]
    vals = [r[k]["psnr_up"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(keys, vals, color="#8c564b")
    ax.axhline(r["hairy_input_floor"]["psnr_up"], color="black", ls="--", lw=1, label="hairy-input floor")
    ax.axhline(j["provenance"]["paper"]["psnr"], color="#2ca02c", ls=":", lw=1.5, label="paper: 41.04 dB")
    ax.set_ylabel("PSNR (dB), upstream metric at 128 px")
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Released checkpoint on DERMAHAIR", fontsize=9)
    save(fig, "fig_dermahair.png")


def main() -> int:
    for fn in (fig_macro_f1, fig_image_quality, fig_xai, fig_learning_curve, fig_dermahair):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    print("made:", *made, sep="\n  ")
    print("skipped:", *skipped, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
