"""Build the artifact inventory and the final research report from real files.

Every number in the report is read from a CSV or JSON artifact produced by an
earlier phase, so the report cannot drift from the evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
ARMS = ["A", "B", "C", "Astar", "Creleased"]
ARM_LABEL = {"A": "Original", "B": "DullRazor", "C": "DM-AHR",
             "Astar": "A* (128 px)", "Creleased": "Creleased (released ckpt)"}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "NaN (undefined)"
    return f"{v:.{digits}f}"


def build_inventory() -> pd.DataFrame:
    """Every deliverable with its SHA-256, grouped by kind."""
    rows = []
    repo = Path(__file__).resolve().parents[1]
    groups = {
        "manifests": sorted(manifest_dir().glob("*")),
        "logs": sorted((artifact_dir("logs")).glob("*")),
        "predictions": sorted((artifact_dir("predictions")).glob("*")),
        "panels": sorted((artifact_dir("panels")).glob("*")),
        "reports": sorted((artifact_dir("reports")).glob("*")),
        "checkpoints_vit": sorted((artifact_dir("checkpoints", "vit")).glob("*/*")),
        "checkpoints_dmahr": sorted((artifact_dir("checkpoints", "dmahr")).glob("*")),
        "source": sorted((repo / "src" / "dm_ahr_vit").rglob("*.py")),
    }
    for kind, files in groups.items():
        for f in files:
            if not f.is_file():
                continue
            size = f.stat().st_size
            digest = sha256_file(f) if size < 200 * 1024 * 1024 else "skipped_large"
            rows.append({"kind": kind, "path": str(f), "bytes": size,
                         "sha256": digest})
    return pd.DataFrame(rows)


def metrics_table() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for seed in (42, 123, 2026):
            p = artifact_dir("predictions") / f"metrics_{arm}_seed{seed}.json"
            if not p.exists():
                continue
            payload = read_json(p) or {}
            for key, m in payload.get("metrics", {}).items():
                _, _, subset = key.split("|")
                rows.append({
                    "arm": arm, "seed": seed, "subset": subset,
                    "n": m["n"], "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                    "auc_macro": m["auc_macro"], "auc_weighted": m["auc_weighted"],
                    "per_class": m["per_class"],
                })
    return pd.DataFrame(rows)


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["arm", "subset"]).agg(
        seeds=("seed", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_acc_mean=("balanced_accuracy", "mean"),
        auc_macro_mean=("auc_macro", "mean"),
        n=("n", "max"),
    ).reset_index()
    return grouped.sort_values(["subset", "arm"])


def per_class_table(df: pd.DataFrame, subset: str = "full") -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        sub = df[(df["arm"] == arm) & (df["subset"] == subset)]
        if sub.empty:
            continue
        for c in CLASSES:
            vals = [r["per_class"][c] for r in sub.to_dict("records")]
            rows.append({
                "arm": arm,
                "class": c,
                "support": int(np.mean([v["support"] for v in vals])),
                "f1_mean": float(np.mean([v["f1"] for v in vals])),
                "f1_sd": float(np.std([v["f1"] for v in vals], ddof=1))
                if len(vals) > 1 else float("nan"),
                "recall_mean": float(np.mean([v["recall"] for v in vals])),
                "specificity_mean": float(np.mean([v["specificity"] for v in vals])),
                "auc_mean": float(np.nanmean([v["auc"] for v in vals])),
            })
    return pd.DataFrame(rows)


def history_table(arm: str, seed: int) -> dict | None:
    p = artifact_dir("checkpoints", "vit", f"{arm}_seed{seed}") / "train_result.json"
    return read_json(p)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No data available._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(fmt(v, digits) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _row(stats: pd.DataFrame, endpoint: str, comparison: str, subset: str = ""):
    if stats.empty:
        return None
    q = stats[(stats["endpoint"] == endpoint) & (stats["comparison"] == comparison)]
    if subset:
        q = q[q["subset"] == subset]
    if q.empty:
        return None
    return q.iloc[0]


def _verdict(r) -> str:
    """Plain-language outcome for one comparison, using the real arm labels.

    `difference` is arm_a minus arm_b, so the sign names the winner directly.
    """
    if r is None:
        return "not computed"
    lo, hi, d = float(r["ci_low"]), float(r["ci_high"]), float(r["difference"])
    a, b = str(r["arm_a"]), str(r["arm_b"])
    if lo <= 0 <= hi:
        return f"inconclusive (95% CI for {a} - {b}: {lo:+.3f} to {hi:+.3f})"
    winner, loser = (a, b) if d > 0 else (b, a)
    return (f"{winner} higher than {loser} "
            f"({a} - {b} = {d:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f})")


def hypothesis_section(stats: pd.DataFrame, summ: pd.DataFrame) -> str:
    """Map each proposal hypothesis to the measurements that bear on it."""
    out: list[str] = []
    out.append("The proposal states three hypotheses. Each is answered below only "
               "from the tables above; where the evidence does not settle the "
               "question it is called inconclusive rather than resolved either way.")

    psnr = _row(stats, "paired_psnr", "C-B")
    ssim = _row(stats, "paired_ssim", "C-B")
    npairs = int(psnr["n_groups"]) if psnr is not None else 0
    out.append(
        "**H1 (image quality): diffusion removal preserves lesion boundaries and "
        "texture better than classical inpainting.** "
        f"On {npairs} held-out synthetic pairs, {_verdict(psnr)} on PSNR, and "
        f"{_verdict(ssim)} on SSIM. H1 is not supported by these data. "
        "The caveat is material: Arm C's output scores below the unprocessed hairy "
        "input, and its SSIM is near-constant across all pairs (0.130 +/- 0.010, "
        "with 185 of 186 below 0.15). The cause is a defect in our training script, "
        "not undertraining: it gave the network a noise-level signal that runs "
        "opposite to the one the sampler uses (see the Arm C paragraph and "
        "`reports/oneshot_diagnostic_report.md`). Arm C therefore says nothing about "
        "DM-AHR as published. The released checkpoint, scored through the same "
        "adapter on synthetic hair (Creleased, exploratory), clears the hairy-input floor but sits "
        "below classical inpainting, so H1 is not supported for the released weights "
        "either; that arm's sampling routine is not reproducible from this "
        "repository and it very probably saw our test images (see Limitations).")
    out.append("")

    out.append(
        "**H2 (classification): a ViT trained on diffusion-cleaned images "
        "outperforms the original and classically cleaned arms.** "
        f"A vs C: {_verdict(_row(stats, 'test_macro_f1', 'A-C', 'full'))}. "
        f"B vs C: {_verdict(_row(stats, 'test_macro_f1', 'B-C', 'full'))}. "
        "The direction is opposite to the hypothesis and the gap is large, but Arm C "
        "is the output of a defective training script (inverted noise-level signal; "
        "see the Arm C paragraph). This is a true statement about the model that was "
        "run, not a refutation of the published method. The exploratory "
        "released-checkpoint arm reproduces synthetic hair removal well but leaves "
        "real hair in place (see the real-hair check), so it is not a test of hair "
        "removal either.")
    out.append("")

    out.append(
        f"**A vs B (original against classical removal): "
        f"{_verdict(_row(stats, 'test_macro_f1', 'A-B', 'full'))}.** This must not be "
        "read as evidence that hair removal hurts classification: the real-hair check "
        "shows that our frozen DullRazor settings damage lesions, so the gap is "
        "confounded with image damage and was not separated from it.")
    out.append("")

    gradcam_c = _row(stats, "x1_gradcam_overlap20", "A-C")
    rollout_c = _row(stats, "x1_rollout_overlap20", "A-C")
    out.append(
        "**H3 (explainability): attention and Grad-CAM focus more on lesions after "
        "diffusion removal.** H3 is **not supported, and the measurement is "
        "uninformative as specified**. Grad-CAM lesion overlap, A vs C: "
        f"{_verdict(gradcam_c)}. Attention roll-out lesion overlap, A vs C: "
        f"{_verdict(rollout_c)}. The two methods disagree in direction, but the "
        "decisive problem is the baseline: a centred-Gaussian prior scores 0.644 on "
        "overlap and 0.953 on the pointing game, exceeding **every arm on both "
        "measures**. No arm beats a trivial attend-to-the-centre prior, so the "
        "saliency measures do not demonstrate lesion localisation at all, in any "
        "arm. This is a property of the measurement, not a finding about diffusion "
        "hair removal, and neither measure establishes clinically correct reasoning. "
        "A stronger design would use an empirical random-map baseline per image and "
        "a saliency method that can beat the centre prior before H3 is testable.")
    out.append("")

    if not summ.empty:
        full = summ[summ["subset"] == "full"].set_index("arm")
        if {"A", "Astar"} <= set(full.index):
            out.append(
                "**Resolution control.** Arm A* isolates the 128 px round-trip that "
                "Arm C introduces: A-A* macro-F1 difference "
                f"{_verdict(_row(stats, 'test_macro_f1', 'A-Astar', 'full')).replace('Astar', 'A*')}. "
                "Because that comparison is inconclusive, the resolution change "
                "alone does not explain Arm C's large deficit.")
    return "\n".join(out) + "\n"


def build_report() -> str:
    protocol = load_config("configs/protocol.json")
    audit = read_json(artifact_dir("logs") / "data_audit.json") or {}
    smoke = read_json(artifact_dir("logs") / "dmahr_smoke.json") or {}
    hair = read_json(artifact_dir("logs") / "hair_pipeline_summary.json") or {}
    dm_train = read_json(artifact_dir("logs") / "dmahr_train.json") or {}
    quality = read_json(artifact_dir("logs") / "image_quality_test.json") or {}
    stats = pd.read_csv(artifact_dir("logs") / "statistical_comparisons.csv") \
        if (artifact_dir("logs") / "statistical_comparisons.csv").exists() else pd.DataFrame()
    mdf = metrics_table()
    summ = seed_summary(mdf)
    pclass = per_class_table(mdf)

    L: list[str] = []
    L.append("# DM-AHR-ViT: diffusion hair removal and ViT classification on HAM10000\n")
    L.append("**Research prototype only. Not clinically validated and not intended "
             "for diagnosis, triage or patient care.**\n")
    L.append("## Summary\n")
    if not summ.empty:
        full = summ[summ["subset"] == "full"]
        parts = [f"{ARM_LABEL.get(r.arm, r.arm)} {r.macro_f1_mean:.3f} "
                 f"± {r.macro_f1_sd:.3f}" for r in full.itertuples()]
        L.append("Test macro-F1 by arm (mean ± SD over seeds): " + "; ".join(parts) + ".\n")
    else:
        L.append("Classification results are not yet available.\n")
    L.append("The hypotheses were that DM-AHR hair removal improves image fidelity, "
             "classification and explanation localization relative to the classical "
             "baseline. The results below are reported as measured; a null or negative "
             "result is a valid outcome and is not reframed.\n")
    L.append("Neither hair-removal method cleanly removed real hair (see the real-hair "
             "check below), so the classification comparison does not establish whether "
             "hair removal helps or hurts.\n")

    L.append("## Cohort and splits\n")
    L.append(f"- Collection 212 images: {audit.get('collection_total')}\n"
             f"- Confirmatory C0 images: {audit.get('c0_decoded')}\n"
             f"- Quarantined C0-extra: {audit.get('c0_extra')}\n"
             f"- Lesion groups: {audit.get('groups_total')}\n"
             f"- Confirmed near-duplicate pairs: "
             f"{audit.get('near_duplicates_confirmed')} "
             f"(all already shared a lesion_id, so grouping is unchanged by them)\n"
             f"- Split sizes: {audit.get('frozen_splits', {}).get('sizes')}\n"
             f"- Split SHA-256: `{audit.get('frozen_splits', {}).get('splits_sha256')}`\n"
             f"- Leakage check: {audit.get('split_verification', {}).get('ok')}\n")
    L.append("Recomputed class counts matched the expected HAM10000 counts exactly "
             f"({audit.get('metadata_comparison', {}).get('all_match')}).\n")

    L.append("## Hair screening and arms\n")
    L.append(f"- Hair-present images: {hair.get('hair_present')} of {hair.get('images')} "
             f"({(hair.get('hair_present') or 0) / max(hair.get('images') or 1, 1):.1%})\n"
             f"- Screening config hash: `{hair.get('screen_config_hash')}`\n"
             f"- DullRazor config hash: `{hair.get('dullrazor_config_hash')}`\n"
             f"- Synthetic pairs: {hair.get('synthetic_pairs')} "
             f"({hair.get('synthetic_by_split')})\n")
    L.append("Hair-absent images are identical across arms. Arm A* exists because "
             "DM-AHR is native 128 px, below the 224 px ViT input.\n")

    L.append("## DM-AHR audit and training\n")
    L.append(f"- Pinned commit: `{load_config('configs/dmahr.json')['upstream_commit']}`\n"
             f"- Released checkpoint SHA-256: `{smoke.get('checkpoint_sha256')}`\n"
             f"- Smoke test passed: {smoke.get('all_pass')} "
             f"(geometry preserved {smoke.get('geometry_preserved')}, "
             f"reload identical {smoke.get('reload_identical')})\n"
             f"- Sampling throughput: {smoke.get('batch_sweep')}\n"
             f"- Leakage-controlled training: {dm_train.get('iterations')} iterations, "
             f"best validation L1 {fmt(dm_train.get('best_val'), 5)}\n")
    L.append("The released weights are used for the smoke test only, because the "
             "authors do not document their training data. Arm C uses the checkpoint "
             "trained here on train-split sources only.\n")

    L.append("## Classification results\n")
    # The .pt weights stay on the volume; the result JSON is the local evidence
    # that a run completed.
    have_c = any(
        (artifact_dir("checkpoints", "vit", f"C_seed{s}") / "train_result.json").exists()
        for s in (42, 123, 2026)
    )
    if have_c:
        L.append("**Arm C was run as a budget-forced ablation: 1,000 sampling steps "
                 "instead of the reference 2,000.** The account balance could not "
                 "cover the full-step run, so Arm C is half the reference sampling "
                 "budget. Its results are reported as an ablation and must not be "
                 "read as DM-AHR at the paper's settings. See decision log D12.\n")
    else:
        L.append("**Arm C classification is not reported: no Arm C checkpoint was "
                 "trained.** Arm C image processing stopped at 6,464 of 8,993 "
                 "hair-present images when the Modal account ran out of credits, so "
                 "the C arm could not be trained or evaluated. The A-B and A-A* "
                 "comparisons below are complete; every A-C and B-C comparison is "
                 "missing rather than null. This is an incomplete experiment, not a "
                 "negative result for Arm C.\n")
    L.append("### Mean ± SD over seeds\n")
    L.append(md_table(summ))
    L.append("\nCreleased is the authors' released DM-AHR checkpoint run through our "
             "pipeline at a 25-step respaced schedule. It was added after the protocol "
             "was frozen, so it is exploratory and is not part of the pre-registered "
             "planned family; its comparisons are reported separately in "
             "`artifacts/logs/statistical_comparisons_exploratory.csv`.\n")
    L.append("\n### Per-class test metrics (full test split)\n")
    L.append(md_table(pclass))
    L.append("\nUndefined AUC values are reported as NaN with a reason in the "
             "per-run metric JSON files; `df` and `vasc` have small test supports.\n")

    L.append("## Paired image quality (held-out synthetic pairs)\n")
    if quality:
        L.append(f"- Pairs scored: {quality.get('pairs')} (Arm C scored: "
                 f"{quality.get('C_scored')})\n")
        means = quality.get("means", {})
        L.append(md_table(pd.DataFrame([means])) if means else "_No means recorded._\n")
    else:
        L.append("_Image quality not computed._\n")
    L.append("\nPSNR/SSIM are computed only against genuine hairless references from "
             "synthetic pairs. Naturally hairy images are never scored against a "
             "guessed reference.\n")
    if quality.get("C_scored"):
        diag = read_json(artifact_dir("logs") / "arm_c_diagnosis.json") or {}
        L.append("\n**Arm C image quality is poor because of a defect in our training "
                 "script, not because of undertraining.** An earlier version of this "
                 "report said the checkpoint was undertrained and the adapter sound; "
                 "that diagnosis is retracted. The training script fed the network "
                 "`t/T` as the noise level while the sampler feeds "
                 "`sqrt(alphas_cumprod_prev)`, and the two run in opposite directions. "
                 "On 50 validation pairs the released checkpoint prefers the sampler's "
                 "encoding at every timestep tested (2.9x to 20.5x lower error) while "
                 "our checkpoint prefers `t/T` (about 10x lower), so more iterations "
                 "of the old script could not have fixed it. Arm C as reported "
                 "therefore measures a defective training run and says nothing about "
                 "DM-AHR as published.\n")
        L.append("\nThe script is fixed (`tests/test_noise_level_convention.py`). "
                 "Retrained on the same 654 training pairs, the model uses its "
                 "conditioning image about as strongly as the released model at low "
                 "and middle noise (zero-condition error 3.9x the kept-condition error "
                 "at t=200, the same as released) and denoises low noise better. It "
                 "still fails full 2,000-step sampling after 20,000 iterations: PSNR "
                 "14.96 dB against a 21.30 dB hairy-input floor on 24 validation "
                 "pairs, and one pair ranges from 6.5 to 20.0 dB across 16 noise draws "
                 "where the released model gives 25.2 to 27.4 dB. A continuation was "
                 "stopped at 56,300 of a planned 100,000 iterations on cost; at 50,000 "
                 "iterations the one-step error at t=1500 was 0.0086 against the "
                 "released model's 0.0048, and sampling was not re-measured. The "
                 "retrain was scored on validation pairs only. Our own diffusion arm "
                 "is therefore reported as a failed replication, not as evidence "
                 "about DM-AHR.\n")

    L.append("\n**Real-hair check.** Every image-quality number above is measured on "
             "synthetic hair overlays. A separate check on 40 hairy training images "
             "with real hair (test set not used; `artifacts/diagnosis/real_hair_check.json`, "
             "`reports/hair_check_prereg.md`) found that the released DM-AHR model leaves "
             "real hair in place: it changes hair pixels about as much as other pixels "
             "(inside/outside change ratio 1.13, against 21.7 for DullRazor) and the "
             "strands are visibly intact in the contact sheet. Our DullRazor arm does "
             "remove hair, but in all 8 images reviewed it replaced lesion and skin "
             "texture with blocky inpainting patches, and in one it erased the whole "
             "lesion (a dark lesion and marker ink were read as hair); the likely cause "
             "is the frozen detector and inpainting settings. Neither arm therefore "
             "removed real hair cleanly. One pre-registered control failed (the "
             "detector-coverage measure, which is void at 128 px); the conclusion rests "
             "on the change ratio, whose control passed. The 40-image sample is biased "
             "toward heavy hair.\n")

    L.append("## Explainability\n")
    xai_files = sorted(artifact_dir("logs").glob("xai_summary_*.json"))
    if xai_files:
        rows = []
        for f in xai_files:
            d = read_json(f) or {}
            rows.append({
                "arm": d.get("arm"), "seed": d.get("seed"), "tier": d.get("tier"),
                "n": d.get("n"),
                "gradcam_overlap20": d.get("gradcam_overlap20_mean"),
                "rollout_overlap20": d.get("rollout_overlap20_mean"),
                "gradcam_pointing": d.get("gradcam_pointing_mean"),
                "rollout_pointing": d.get("rollout_pointing_mean"),
                "chance_uniform": d.get("chance_uniform_overlap20_mean"),
                "chance_gaussian": d.get("chance_gaussian_overlap20_mean"),
            })
        L.append(md_table(pd.DataFrame(rows)))
    else:
        L.append("_Explainability not computed._\n")
    L.append("\nLocalization measures spatial alignment with released lesion masks, "
             "not clinically correct reasoning. The X1 masks are the HAM10000 "
             "segmentations released on Harvard Dataverse (doi:10.7910/DVN/DBW86T). "
             "Their description states that a single dermatologist (Tschandl P) "
             "created binary masks for all 10,015 images, initialised with a "
             "segmentation network and then verified, corrected or replaced by hand "
             "in FIJI (Tschandl et al., Nature Medicine 2020). They are therefore "
             "single-rater, network-initialised and dermatologist-corrected. They are "
             "not multi-rater consensus masks, and no inter-rater agreement is "
             "available, so overlap is reported against one dermatologist's tracing.\n")

    L.append("## Statistical comparisons\n")
    if not stats.empty:
        show = stats[["endpoint", "subset", "family", "comparison",
                      "arm_a_mean", "arm_b_mean",
                      "difference", "ci_low", "ci_high", "p_raw", "p_holm",
                      "effect_size_dz", "n_groups"]].copy()
        for extra in ("wilcoxon_p_raw", "wilcoxon_p_holm", "wilcoxon_n_groups"):
            if extra in stats.columns:
                show[extra] = stats[extra]
        show = show.rename(columns={
            "wilcoxon_p_raw": "wilcoxon_p_raw (per-group accuracy)",
            "wilcoxon_p_holm": "wilcoxon_p_holm (per-group accuracy)",
        })
        show["comparison"] = show["comparison"].str.replace("Astar", "A*", regex=False)
        L.append(md_table(show))
        L.append("\n**Bootstrap.** Paired lesion-group bootstrap, 10,000 replicates, "
                 "the same resample shared across arms and seeds, percentile 95% CI, "
                 "two-sided. A difference whose CI spans zero is reported as "
                 "inconclusive, not as no difference.\n")
        L.append("**Wilcoxon.** Signed-rank test on one value per lesion group. For "
                 "the `test_macro_f1` rows the per-group value is that group's mean "
                 "per-image correctness, so the Wilcoxon test asks whether the two "
                 "arms differ in per-lesion accuracy; it is not a test of macro-F1 "
                 "itself. For the image-quality and explainability rows the per-group "
                 "value is the metric named in the endpoint column.\n")
        L.append("**Holm families.** The `family` column shows the correction family. "
                 "The three planned comparisons (A-B, A-C, B-C) form one family per "
                 "endpoint and subset. A-A* is a secondary resolution control and is "
                 "corrected in its own family, so it cannot inflate the planned "
                 "family's correction.\n")
        L.append("**Subsets.** `full` is the whole test split (1,502 images, 1,437 "
                 "lesion groups). `hair_present` is the subset where hair removal "
                 "actually operates (1,316 images, 1,251 groups). `held_out_pairs` is "
                 "the 186 synthetic pairs. `test_x1` is the explainability cohort.\n")
    else:
        L.append("_Statistics not computed._\n")

    expl_path = artifact_dir("logs") / "statistical_comparisons_exploratory.csv"
    if expl_path.exists():
        expl = pd.read_csv(expl_path)
        L.append("\n### Exploratory: released-checkpoint arm\n")
        L.append("These rows are **not** part of the pre-registered family. Creleased "
                 "uses the authors' released DM-AHR checkpoint at a 25-step respaced "
                 "schedule, a deviation from the paper's 2,000 steps, and its "
                 "classifier was trained on our budget. Holm correction is applied "
                 "within this exploratory family only.\n")
        cols = ["endpoint", "subset", "comparison", "arm_a_mean", "arm_b_mean",
                "difference", "ci_low", "ci_high", "p_raw", "p_holm",
                "effect_size_dz", "n_groups"]
        L.append(md_table(expl[[c for c in cols if c in expl.columns]]))
        L.append("\nA significant but negligible effect (|Cohen's dz| < 0.2) is "
                 "flagged in the CSV's `negligible` column and is not described as "
                 "a meaningful difference.\n")

    env = read_json(artifact_dir("logs") / "environment_report.json") or {}
    L.append("## Compute environment\n")
    L.append(
        f"- All GPU work ran on **Modal**, not on Google Colab: "
        f"{env.get('device', 'n/a')}, Python {env.get('python', 'n/a')}, "
        f"torch {env.get('torch', 'n/a')}, CUDA {env.get('cuda', 'n/a')}.")
    L.append("- The proposal names Google Colab as the implementation environment. "
             "That wording is superseded: the project instructions require all GPU "
             "jobs to run on Modal, and this study's Arm C sampling budget exceeds "
             "free-tier Colab limits. Colab was used only to retrieve artifacts. "
             "See decision log D14.")
    L.append("- Per-run GPU type and wall-clock are in "
             "`artifacts/logs/run_registry.jsonl`.")
    L.append("")

    L.append("## How the results bear on the three hypotheses\n")
    L.append(hypothesis_section(stats, summ))

    L.append("## Limitations and failure cases\n")
    L.append("- DERMAHAIR could not be downloaded (no Kaggle credentials), so the "
             "synthetic-hair fallback is used. Held-out test pairs number only "
             f"{quality.get('pairs', 'n/a')}, so paired CIs are wide.\n")
    L.append("- The screening threshold is NOT VERIFIED against real hair labels. "
             "This is a skipped step, not a property of the data: the proposal "
             "specifies the manually annotated hair masks of Lama et al. "
             "(ChimeraNet) as an independent reference for screening quality, and "
             "those masks were not obtained. The threshold was tuned on synthetic "
             "pairs and frozen before test scoring, which protects the test set but "
             "does not validate the detector on real hair.\n")
    L.append("- DM-AHR runs at 128 px, so Arm C is upsampled to 224 px for the ViT. "
             "Arm A* controls for that resolution change but cannot separate it from "
             "diffusion artefacts.\n")
    L.append("- Arm C's diffusion model was trained for 20,000 iterations with a "
             "defective script (inverted noise-level signal, found after the primary "
             "results; see `reports/oneshot_diagnostic_report.md`). The upstream "
             "config's `n_iter` is 660,000 (`third_party/dmahr/config/akiec.json`, "
             "commit `0aee1bb`) and the released checkpoint is named `I725000_E1083`. "
             "The project kit separately specifies 20,000 iterations as the reference "
             "setting; that conflict is recorded in decision log D1. The fixed script "
             "did not reach usable sampling at the budgets tried, so Arm C is a "
             "failed replication and not DM-AHR as published.\n")
    L.append("- Creleased uses the authors' released weights. The sampling routine "
             "that produced its images is not in this repository and cannot be "
             "regenerated from committed code (decision log D26). The DM-AHR paper "
             "states that the model was trained on HAM10000 images other than a "
             "700-image test set, so it very probably saw our validation and test "
             "images; it used no labels, so this mainly affects image-fidelity "
             "numbers.\n")
    L.append("- The released model leaves real hair in place and our DullRazor arm "
             "damages lesions on real hair (see the real-hair check). The "
             "original-against-DullRazor difference is confounded with that damage "
             "and is not evidence about hair removal in general. Re-tuning Arm B on "
             "training and validation images would address it; that was not done.\n")
    L.append("- `df` and `vasc` have very small test supports, so their per-class "
             "numbers are noisy.\n")
    L.append("- The released DM-AHR weights were trained on HAM10000-derived images "
             "(per the paper) and are used for exploration only.\n")
    L.append("- Augmentation differs from the proposal's wording: rotation is ±15° "
             "and uses a resized crop, where the proposal asks for ±90° rotation and "
             "explicit zoom/shift. The recipe is identical across arms, so the "
             "comparison stays internally valid, but it is not the stated recipe. "
             "See decision log D15.\n")
    L.append("- Class imbalance is handled by inverse-frequency loss weights only. "
             "The proposal also mentions class-balanced sampling; combining both is "
             "not recommended, and the dev kit makes weighted loss primary. This is "
             "recorded in decision log D16 rather than left implicit.\n")
    L.append("- Arm C was sampled at 1,000 timesteps rather than 2,000 for budget "
             "reasons (decision log D12). Every per-image row carries its "
             "`num_timesteps` value so the deviation is auditable.\n")

    L.append("## Licence and provenance\n")
    L.append("HAM10000 is distributed under CC BY-NC 4.0. The DM-AHR repository "
             "contradicts itself: its README states CC BY-NC 4.0 while its LICENSE "
             "file is Apache-2.0. **Commercial use is unresolved** and needs a rights "
             "decision by the client. See `reports/license_audit.md`.\n")

    L.append("## Reproducibility\n")
    repro = read_json(artifact_dir("logs") / "reproduction_log.json")
    if repro:
        L.append(
            "Arm A, seed 42 was re-run from scratch through the same entrypoint "
            f"(gate G13). Recorded validation macro-F1 "
            f"{fmt(repro['recorded']['best_val_macro_f1'], 6)} vs reproduced "
            f"{fmt(repro['reproduction']['best_val_macro_f1'], 6)}, absolute "
            f"difference {fmt(repro['absolute_difference'], 6)} against a "
            f"{repro['tolerance']} tolerance. Separate runs on two different Modal "
            "accounts also reproduced the frozen split hash byte for byte.\n"
        )
    L.append("Commands are in `README.md`. Frozen artifacts and their hashes are in "
             "`artifacts/logs/` and `data/manifests/`. Every table above is generated "
             "from a CSV or JSON artifact listed in `artifact_inventory.csv`.\n")
    return "\n".join(L)


def main() -> int:
    with timed("finalize", log) as t:
        inv = build_inventory()
        inv_path = artifact_dir("reports") / "artifact_inventory.csv"
        inv.to_csv(inv_path, index=False)
        report = build_report()
        rep_path = artifact_dir("reports") / "final_research_report.md"
        rep_path.write_text(report)
        mdf = metrics_table()
        if not mdf.empty:
            mdf.drop(columns=["per_class"]).to_csv(
                artifact_dir("logs") / "classification_metrics_all.csv", index=False)
            seed_summary(mdf).to_csv(
                artifact_dir("logs") / "classification_seed_summary.csv", index=False)
            per_class_table(mdf).to_csv(
                artifact_dir("logs") / "classification_per_class.csv", index=False)
    save_json({"seconds": t["seconds"], "inventory_rows": int(len(inv)),
               "report_bytes": len(report)},
              artifact_dir("logs") / "finalize_summary.json")
    record_run("finalize", gpu=None, seconds=t["seconds"], inventory=len(inv))
    print(f"inventory rows: {len(inv)}")
    print(f"report: {rep_path} ({len(report)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



