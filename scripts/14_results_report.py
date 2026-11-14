"""Build the client-facing results report (Word).

Covers training setup, classification results, image quality, explainability,
statistics and hypothesis verdicts. Numbers are recomputed from committed
artifacts. Shares document helpers with 13_preprocessing_report.py.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dm_ahr_vit.paths import artifact_dir  # noqa: E402
from dm_ahr_vit.reporting import stats_text as st  # noqa: E402
from dm_ahr_vit.reporting import xai_text as xt  # noqa: E402

pre = import_module("13_preprocessing_report")
h1, h2, body, bullet, table, callout, figure = (
    pre.h1, pre.h2, pre.body, pre.bullet, pre.table, pre.callout, pre.figure)
add_page_number = pre.add_page_number
ARM_LABEL = {"A": "A — original", "Astar": "A* — 128 px control",
             "B": "B — DullRazor classical",
             "C": "C — first diffusion run (defective trainer)",
             "Creleased": "Creleased — released ckpt",
             "Cv2": "Cv2 — rebuilt diffusion"}
REPO = Path(__file__).resolve().parents[1]


def cv2_seed_metrics(subset: str = "full") -> dict:
    """Mean/SD over seeds of the Cv2 test metrics, read from the per-seed metric files."""
    import numpy as np

    vals: dict[str, list] = {"macro_f1": [], "accuracy": [], "balanced_accuracy": [],
                             "auc_macro": []}
    pred = artifact_dir("predictions")
    for s in (42, 123, 2026):
        m = json.loads((pred / f"metrics_Cv2_seed{s}.json").read_text())["metrics"]
        for k in vals:
            vals[k].append(m[f"Cv2|{s}|{subset}"][k])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)), v) for k, v in vals.items()}

OUT = artifact_dir("reports") / "Results_Report.docx"


def verdict(r) -> str:
    if r is None:
        return "not computed"
    d, lo, hi = float(r["difference"]), float(r["ci_low"]), float(r["ci_high"])
    a = str(r["arm_a"]).replace("Astar", "A*")
    b = str(r["arm_b"]).replace("Astar", "A*")
    if lo <= 0 <= hi:
        return f"inconclusive ({a} - {b} 95% CI {lo:+.3f} to {hi:+.3f})"
    w = a if d > 0 else b
    l = b if d > 0 else a
    return f"{w} higher than {l} ({a} - {b} = {d:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f})"


def row(stats, endpoint, comparison, subset=None):
    q = stats[(stats["endpoint"] == endpoint) & (stats["comparison"] == comparison)]
    if subset:
        q = q[q["subset"] == subset]
    return q.iloc[0] if len(q) else None


def build() -> Path:
    logs = artifact_dir("logs")
    seed = pd.read_csv(logs / "classification_seed_summary.csv")
    per_class = pd.read_csv(logs / "classification_per_class.csv")
    stats = pd.read_csv(logs / "statistical_comparisons.csv")
    quality = pd.read_csv(logs / "paired_psnr_ssim_test.csv")
    # Exclude the `_v2_` diagnostic generation: it uses different columns and
    # would otherwise be averaged together with the reported measurements.
    xai = pd.concat([pd.read_csv(f) for f in sorted(logs.glob("xai_localization_*.csv"))
                     if "_v2_" not in f.name], ignore_index=True)
    env = json.loads((logs / "environment_report.json").read_text())
    repro = json.loads((logs / "reproduction_log.json").read_text()) \
        if (logs / "reproduction_log.json").exists() else {}

    full = seed[seed["subset"] == "full"].set_index("arm").loc[
        ["A", "Astar", "B", "C", "Creleased"]].reset_index()
    expl_path = logs / "statistical_comparisons_exploratory.csv"
    expl = pd.read_csv(expl_path) if expl_path.exists() else pd.DataFrame()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(0.95)
    sec.top_margin, sec.bottom_margin = Inches(0.8), Inches(0.75)
    add_page_number(sec)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    doc.core_properties.title = "DM-AHR-ViT — Results Report"

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("RESULTS REPORT")
    r.font.size = Pt(9); r.font.bold = True; r.font.color.rgb = pre.ACCENT
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Model training,\nevaluation and findings")
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = pre.NAVY
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(18)
    r = p.add_run("What was trained, how it was measured, and what it showed")
    r.font.size = Pt(13); r.font.color.rgb = pre.SLATE
    pre.para_border(p, "C05A33", 12, "bottom")
    body(doc, "DM-AHR-ViT · Explainable Vision Transformer for Skin Lesion "
              "Classification Using Dermoscopy Images", size=10.5)
    body(doc, "Research prototype only. Not clinically validated and not intended "
              "for diagnosis, triage or patient care.", size=9, italic=True)

    best = full.loc[full["macro_f1_mean"].idxmax()]
    worst = full.loc[full["macro_f1_mean"].idxmin()]
    pooled = pd.read_csv(logs / "pooled_per_seed.csv")

    def pf1(arm, subset="full"):
        v = pooled[(pooled["arm"] == arm) & (pooled["subset"] == subset)]["macro_f1"]
        return float(v.mean()), float(v.std(ddof=1)), int(v.size)

    callout(doc, "In one sentence",
            f"Across 38 trained models (Original, the 128 px control, DullRazor and the rebuilt "
            f"diffusion arm Cv2 at eight seeds each; the released checkpoint and our first, failed "
            f"diffusion run at three) evaluated on a locked test set of 1,502 images, Original "
            f"scored highest at macro-F1 {pf1('A')[0]:.3f}. Our first diffusion run "
            f"({worst['macro_f1_mean']:.3f}) failed because of a training-script defect. After "
            f"repairing it, retraining and teaching the model with real hair strands, the rebuilt "
            f"arm Cv2 scored {pf1('Cv2')[0]:.3f} ± {pf1('Cv2')[1]:.3f}, statistically "
            f"indistinguishable from Original, the 128 px control and classical DullRazor "
            f"(section 8); only Original over DullRazor is established. DullRazor recovered clean "
            f"images better than the diffusion arm on synthetic hair and on pasted real strands. "
            f"The pre-registered check that the rebuilt model removes real hair from photographs "
            f"was not met, so this study does not show that hair removal helps or hurts "
            f"classification, and removal on real photographs is supported only by a 12-image "
            f"visual look and a pasted-strand test.")
    doc.add_page_break()

    h1(doc, "1. What was trained and how")
    body(doc,
         "The first fifteen Vision Transformers (sections 2 to 7) are five input conditions (arms) at three random "
         "seeds each; Stage C added three Cv2 runs and Stage E twenty more (section 8). Everything except the arm is identical — same frozen split, same manifests, "
         "same architecture, hyperparameters, augmentation and evaluation code — so a difference "
         "between arms is attributable to the hair-removal condition alone. Creleased is the "
         "authors' released DM-AHR checkpoint run through our pipeline at a 25-step respaced "
         "schedule; it was added after the protocol was frozen and is reported as exploratory.")
    table(doc, ["Arm", "Input the classifier sees", "Macro-F1 (mean ± SD)"],
          [[ARM_LABEL[a], desc,
            f"{full[full['arm'] == a]['macro_f1_mean'].iloc[0]:.3f} ± "
            f"{full[full['arm'] == a]['macro_f1_sd'].iloc[0]:.3f}"]
           for a, desc in [("A", "untouched dermoscopy image"),
                           ("B", "classical hair removal (DullRazor-style)"),
                           ("C", "diffusion-generated hairless image"),
                           ("Astar", "original through the same 128 px round-trip"),
                           ("Creleased", "released DM-AHR checkpoint, 25-step respaced "
                                         "(exploratory)")]],
          widths=[1.7, 3.0, 1.6], align_right_from=2)
    h2(doc, "Training recipe (identical for every run)")
    table(doc, ["Component", "Setting"],
          [["Backbone", "ViT-B/16, ImageNet-1k pretrained, 224 px input, 196 patches"],
           ["Optimiser", "AdamW, lr 3e-5, weight decay 0.05, warm-up 5 epochs then cosine decay"],
           ["Batch / precision", "32, bf16 autocast, gradient clipping 1.0"],
           ["Class imbalance", "inverse-frequency class weights from the training split only"],
           ["Early stopping", "validation macro-F1, patience 10, max 100 epochs"],
           ["Seeds", "42, 123, 2026"],
           ["Test protocol", "the locked test set is scored once per frozen checkpoint"]],
          widths=[1.55, 4.75])
    h2(doc, "Compute environment")
    body(doc,
         f"All GPU work ran on Modal: {env.get('device')}, Python {env.get('python')}, "
         f"torch {env.get('torch')}, CUDA {env.get('cuda')}. A typical training run took "
         f"about 6.5 minutes (25 epochs to early stopping). Per-run GPU type and wall-clock "
         f"are logged in artifacts/logs/run_registry.jsonl.")

    h1(doc, "2. Classification results")
    body(doc,
         "All metrics are computed on the held-out test set of 1,502 images. Macro-F1 is the "
         "primary endpoint because classes are severely imbalanced: nv is two thirds of the "
         "data, so plain accuracy rewards ignoring the rare classes.")
    table(doc, ["Arm", "Macro-F1", "Accuracy", "Balanced acc.", "Macro AUC"],
          [[ARM_LABEL[r.arm], f"{r.macro_f1_mean:.3f} ± {r.macro_f1_sd:.3f}",
            f"{r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f}",
            f"{r.balanced_acc_mean:.3f}", f"{r.auc_macro_mean:.3f}"]
           for r in full.itertuples()],
          widths=[1.85, 1.35, 1.15, 1.05, 0.95], align_right_from=1)
    h2(doc, "Hair-present subset")
    hp = seed[seed["subset"] == "hair_present"].set_index("arm")
    table(doc, ["Arm", "Macro-F1", "n"],
          [[ARM_LABEL[a], f"{hp.loc[a, 'macro_f1_mean']:.3f} ± {hp.loc[a, 'macro_f1_sd']:.3f}",
            int(hp.loc[a, 'n'])] for a in ["A", "Astar", "B", "C", "Creleased"]],
          widths=[2.2, 1.9, 0.8], align_right_from=1)
    body(doc,
         "This is the subset where hair removal actually operates — the images it could help. "
         "The ordering is unchanged from the full test set.", size=9.5, italic=True)

    h2(doc, "Per-class test metrics")
    rows = []
    for cls in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        for arm in ["A", "B", "C", "Astar", "Creleased"]:
            q = per_class[(per_class["arm"] == arm) & (per_class["class"] == cls)]
            if len(q):
                r = q.iloc[0]
                rows.append([cls if arm == "A" else "", arm,
                             f"{int(r['support'])}", f"{r['f1_mean']:.3f}",
                             f"{r['recall_mean']:.3f}", f"{r['specificity_mean']:.3f}",
                             f"{r['auc_mean']:.3f}"])
    table(doc, ["Class", "Arm", "Support", "F1", "Recall (sensitivity)", "Specificity", "AUC"],
          rows, widths=[0.72, 0.5, 0.72, 0.62, 1.25, 0.95, 0.62], align_right_from=2)
    body(doc,
         "Recall is the proposal's sensitivity. Supports for df (17) and vasc (21) are small, so "
         "their per-class numbers are noisy and are not interpreted individually.",
         size=9.5, italic=True)

    h1(doc, "3. Hair-removal image quality")
    body(doc,
         f"Measured on the {len(quality)} held-out synthetic pairs — hair composited onto genuinely "
         f"hair-absent images, so a true hairless reference exists. The unprocessed hairy input is "
         f"scored as a floor: falling below it means the method made the image worse than doing "
         f"nothing.")
    qrows = [["Hairy input (floor)",
              f"{quality['hairy_input_psnr'].mean():.2f} ± {quality['hairy_input_psnr'].std():.2f}",
              f"{quality['hairy_input_ssim'].mean():.3f} ± {quality['hairy_input_ssim'].std():.3f}"],
             ["Arm B — classical",
              f"{quality['B_psnr'].mean():.2f} ± {quality['B_psnr'].std():.2f}",
              f"{quality['B_ssim'].mean():.3f} ± {quality['B_ssim'].std():.3f}"]]
    if "Creleased_psnr" in quality.columns and quality["Creleased_psnr"].notna().any():
        qrows.append(["Creleased — released ckpt",
                      f"{quality['Creleased_psnr'].mean():.2f} ± "
                      f"{quality['Creleased_psnr'].std():.2f}",
                      f"{quality['Creleased_ssim'].mean():.3f} ± "
                      f"{quality['Creleased_ssim'].std():.3f}"])
    qrows.append(["Arm C — diffusion",
                  f"{quality['C_psnr'].mean():.2f} ± {quality['C_psnr'].std():.2f}",
                  f"{quality['C_ssim'].mean():.3f} ± {quality['C_ssim'].std():.3f}"])
    table(doc, ["Condition", "PSNR (dB) mean ± SD", "SSIM mean ± SD"],
          qrows, widths=[1.9, 2.1, 2.1], align_right_from=1)
    callout(doc, "Why Arm C scores below the untouched input — a training-script defect",
            "An earlier version of this report called the checkpoint undertrained; that "
            "diagnosis is retracted. Our training script fed the network t/T as the noise "
            "level while the sampler feeds sqrt(alphas_cumprod_prev), and the two run in "
            "opposite directions. On 50 validation pairs the released checkpoint prefers the "
            "sampler's encoding at every timestep tested (2.9× to 20.5× lower error) and ours "
            "prefers t/T (about 10× lower), so more iterations of the old script could not have "
            "fixed it. Arm C as reported measures a defective training run and says nothing "
            "about DM-AHR as published. The script is now fixed; a retrain on validation pairs "
            "still failed full 2,000-step sampling (14.96 dB against a 21.30 dB hairy-input "
            "floor, 24 pairs) and was stopped on cost at 56,300 of a planned 100,000 iterations. "
            "See reports/oneshot_diagnostic_report.md.",
            fill="FDF3ED", bar="C05A33")

    callout(doc, "Does any arm remove real hair? Not cleanly",
            "Every image-quality number above is measured on synthetic hair overlays. A separate "
            "check on 40 hairy training images with real hair (test set not used; "
            "artifacts/diagnosis/real_hair_check.json) found that the released DM-AHR model leaves "
            "real hair in place: it changes hair pixels about as much as other pixels (ratio 1.13, "
            "against 21.7 for DullRazor), and the strands are visibly intact in the contact sheet. "
            "Our DullRazor arm does remove hair, but in all 8 images reviewed it replaced lesion and "
            "skin texture with blocky patches, and in one it erased the whole lesion; the likely "
            "cause is our frozen detector and inpainting settings. So neither arm cleanly removed "
            "real hair, and the classification comparison does not test whether hair removal helps. "
            "One control in that check failed (the hair-coverage measure, now void); the conclusion "
            "rests on the change ratio, whose control passed. See reports/hair_check_prereg.md.",
            fill="FDF3ED", bar="C05A33")

    h1(doc, "4. Explainability")
    x2 = xt.load(logs)
    body(doc,
         "For every test image with a lesion mask (1,502 images), attention rollout and Grad-CAM were "
         "compared with the released HAM10000 lesion mask after carrying the mask through the same "
         "resize and centre crop the classifier applies to the image. Grad-CAM uses the output of ln_1 "
         "in the last encoder block as its primary target; the patch-embedding grid used in the first "
         "analysis is kept as a labelled sensitivity run. The first analysis compared maps with masks "
         "that were not in the classifier's frame; it is retained in the repository as the legacy "
         "analysis and is not used here. Measures:")
    for line in xt.definitions().split("\n"):
        bullet(doc, line[2:])
    body(doc, xt.geometry_sentence(x2) + " " + xt.agreement_sentence(x2), size=9.5)
    for metric, title in (("recall", "Lesion coverage (recall)"), ("pointing", "Pointing")):
        head, rows = xt.arm_table(x2, metric)
        body(doc, f"{title}: mean over seeds, with the seed range, against the chance baselines on the "
                  "same masks (centred Gaussian, random pixel maps, random patch maps; 100 draws per image) "
                  "and the model-minus-Gaussian difference with a 95% interval over lesions and seeds.",
             size=9.5)
        table(doc, head, rows, widths=[1.35, 1.35, 0.4, 1.05, 0.55, 0.6, 0.6, 1.1], font=7,
              align_right_from=2)
        for line in xt.baseline_verdicts(x2, metric):
            bullet(doc, line)
    body(doc, xt.strike_summary(x2) + " Overlap measures spatial alignment with released masks, not "
              "clinically correct reasoning. The masks are single-rater: a dermatologist (Tschandl P) "
              "created binary masks for all 10,015 images, network-initialised and then verified, "
              "corrected or replaced by hand in FIJI (Tschandl et al., Nature Medicine 2020), so overlap "
              "is measured against one tracing, not consensus.", size=9.5, italic=True)
    fig_dir = artifact_dir("reports") / "preprocessing_assets"
    figure(doc, fig_dir / "dmahr.jpg",
           "Figure 1 — Arm C diffusion removal, from the preprocessing report, reproduced here "
           "because the same failed hair-removal output drives both its quality scores and its "
           "classification deficit.")

    h1(doc, "5. Statistical comparisons")
    body(doc,
         "Every endpoint is tested twice, as the protocol requires: a lesion-group bootstrap "
         "(10,000 replicates, the same resample shared across arms and seeds, percentile 95% CI) "
         "and a Wilcoxon signed-rank test on values collapsed to one per lesion group. "
         "For the classification endpoint the per-group value is that group's mean "
         "per-image correctness, so the Wilcoxon test asks whether the arms differ in "
         "per-lesion accuracy — it is not a test of macro-F1 itself, and the column "
         "header says so. Holm correction is applied across the three planned "
         "comparisons within each endpoint, separately for each test; A–A* is a "
         "secondary control in its own family and cannot inflate the planned family. "
         "A CI spanning zero is called inconclusive, not 'no difference'.")
    rows = []
    for r in stats.itertuples():
        rows.append([r.endpoint, r.comparison, f"{r.difference:+.3f}",
                     f"{r.ci_low:+.3f} to {r.ci_high:+.3f}",
                     f"{r.p_holm:.4f}" if r.p_holm >= 0.0001 else "<0.0001",
                     f"{r.wilcoxon_p_holm:.4f}" if r.wilcoxon_p_holm >= 0.0001 else "<0.0001",
                     f"{r.n_groups}"])
    table(doc, ["Endpoint", "Comparison", "Diff.", "95% CI (bootstrap)",
                 "p (Holm)", "Wilcoxon p (per-group acc.)", "Groups"],
          rows, widths=[1.35, 0.95, 0.6, 1.25, 0.72, 0.9, 0.55], font=8, align_right_from=2)
    if not expl.empty:
        h2(doc, "Exploratory: released-checkpoint arm")
        body(doc,
             "These rows are not part of the pre-registered family. Creleased uses the authors' "
             "released DM-AHR checkpoint at a 25-step respaced schedule, a deviation from the "
             "paper's 2,000 steps, and its classifier was trained on our budget. Holm correction "
             "is applied within this exploratory family only.", size=9.5, italic=True)
        erows = []
        for r in expl.itertuples():
            comp_ = str(r.comparison)
            ll = ""
            if r.endpoint == "test_macro_f1" and "-" in comp_:
                a_, b_ = comp_.split("-", 1)
                d_ = st.logloss_diff(artifact_dir("predictions"), a_, b_)
                if d_["per_seed"]:
                    ll = (f"{d_['difference']:+.3f} ({d_['ci_low']:+.3f} to {d_['ci_high']:+.3f}); per-seed "
                          + ", ".join(f"{v:+.3f}" for v in d_["per_seed"].values()))
            erows.append([r.endpoint, r.comparison, f"{r.difference:+.3f}",
                          f"{r.ci_low:+.3f} to {r.ci_high:+.3f}",
                          f"{r.p_holm:.4f}" if r.p_holm >= 0.0001 else "<0.0001",
                          ll, f"{r.n_groups}"])
        table(doc, ["Endpoint", "Comparison", "Diff.", "95% CI (bootstrap)",
                    "p (Holm)", "Per-image log-loss difference (a - b; negative favours a)", "Groups"],
              erows, widths=[1.15, 1.0, 0.55, 1.1, 0.6, 1.9, 0.5], font=7.5,
              align_right_from=2)

    h1(doc, "6. What this says about the three hypotheses")
    psnr = row(stats, "paired_psnr", "C-B")
    ssim = row(stats, "paired_ssim", "C-B")
    gc = row(stats, "x1_gradcam_overlap20", "A-C")
    ro = row(stats, "x1_rollout_overlap20", "A-C")
    body(doc, "H1 — image quality (diffusion preserves texture better than classical): ",
         space_after=2)
    c_ssim = quality["C_ssim"].dropna()
    bullet(doc, f"PSNR: {verdict(psnr)}. SSIM: {verdict(ssim)}. H1 is not supported by these "
                f"data. Arm C's SSIM is near-constant across all {len(c_ssim)} pairs "
                f"({c_ssim.mean():.3f} ± {c_ssim.std():.3f}, with "
                f"{int((c_ssim < 0.15).sum())} below 0.15), which is what the inverted "
                "noise-level signal in our training script produces. "
                "The released checkpoint was then scored through the same adapter (Creleased): "
                "it clears the hairy-input floor by 4.95 dB but sits 8.66 dB below classical "
                "inpainting, so H1 is now tested for DM-AHR and is not supported.")
    body(doc, "H2 — classification (diffusion-cleaned images improve the ViT): ", space_after=2)
    bullet(doc, f"A vs C: {verdict(row(stats, 'test_macro_f1', 'A-C', 'full'))}. "
                f"B vs C: {verdict(row(stats, 'test_macro_f1', 'B-C', 'full'))}. The direction is "
                "opposite to the hypothesis and the gap is large. The released checkpoint does "
                "not rescue it: A vs Creleased is +0.038 (95% CI +0.005 to +0.071, Holm p=0.048, "
                "negligible effect) and B vs Creleased is inconclusive (-0.015, CI -0.049 to "
                "+0.019, Holm p=0.400).")
    bullet(doc, f"A vs B: {verdict(row(stats, 'test_macro_f1', 'A-B', 'full'))}. This must not be "
                "read as evidence that hair removal hurts classification: the real-hair check shows "
                "our frozen DullRazor settings damage lesions, so the gap is confounded with image "
                "damage and was not separated from it.")
    body(doc, "H3 — explainability (attention focuses more on lesions after diffusion): ",
         space_after=2)
    pl = x2["comp"]
    pl = pl[(pl["family"] == "planned") & (pl["frame"] == xt.AS_IS) & (pl["metric"] == "recall")]
    for r in pl.itertuples():
        bullet(doc, f"{r.a} - {r.b}, {r.method}, lesion coverage: {r.difference:+.3f} (95% CI "
                    f"{r.ci_low:+.3f} to {r.ci_high:+.3f}; per-seed {r.per_seed_differences}); "
                    f"{'struck: seed spread exceeds the difference' if r.struck else 'not struck'}.")
    bullet(doc, "Diffusion versus the others (Cv2 against A, B and A*) is exploratory; see the tables in "
                "section 4. " + xt.strike_summary(x2))
    bullet(doc, "H3 compares attention after diffusion removal with attention on the original photos. "
                "With the masks in the classifier's frame, the first reading (a centred Gaussian beats every "
                "arm) must be re-read from the baseline rows in section 4: " +
                " ".join(xt.baseline_verdicts(x2, "recall")) + " Whether H3 is supported is read from the "
                "planned comparison above and the Cv2 rows, not from this sentence.")
    body(doc, "Resolution control: ", space_after=2)
    bullet(doc, f"A vs A*: {verdict(row(stats, 'test_macro_f1', 'A-Astar', 'full'))}. Because "
                "this is inconclusive, the 128 px round-trip alone does not explain Arm C's "
                "deficit.")

    h1(doc, "7. Limitations")
    for lead, txt in [
        ("Arm C is a failed replication, caused by a training-script defect — ",
         "Our script gave the network an inverted noise-level signal (t/T instead of the "
         "sampler's sqrt(alphas_cumprod_prev)), so the model ran 20,000 iterations learning the "
         "wrong mapping; it was also sampled at 1,000 steps against 2,000 and upsampled from "
         "128 px to 224. The upstream config's n_iter is 660,000 and the released checkpoint is "
         "named I725000_E1083. The fixed script, sampled from pure noise at the budgets "
         "tried, did not give usable output; the partial-noise sampler used for Cv2 did. "
         "Arm C's results describe that defective run, not the published method. The "
         "rebuilt arm Cv2 (section 8) is the fair test of the method."),
        ("Arm B damages lesions on real hair — ",
         "On 8 real-hair training images reviewed, our frozen DullRazor settings replaced lesion "
         "and skin texture with blocky inpainting patches, and in one image erased the lesion (a "
         "dark lesion and marker ink were read as hair). The A-versus-B difference is therefore "
         "confounded and is not evidence about hair removal in general. Re-tuning the arm on "
         "training and validation images would address it; that was not done."),
        ("The released diffusion model leaves real hair — ",
         "It reproduces synthetic hair removal well, which is what it was trained on, but on 40 "
         "real-hair training images it left the hair in place. Creleased is therefore close to "
         "'the original at 128 px' and is not a test of hair removal."),
        ("Creleased is exploratory, not pre-registered — ",
         "It uses the authors' released checkpoint at a 25-step respaced schedule, a deviation "
         "from the paper's 2,000 steps, and the routine that produced those images is not "
         "reproducible from committed code. Its classifier was trained on our budget rather than "
         "the paper's. The paper states the model was trained on HAM10000 images other than a "
         "700-image test set, so it very probably saw our validation and test images (no labels "
         "were used). Its comparisons are reported in their own Holm family and must not be "
         "read as pre-registered results."),
        ("Seeds — ",
         "the intervals in sections 2 to 7 resample ~1,437 lesion groups with the three seeds held "
         "fixed, so they leave out training-seed variance, and the three-seed SDs turned out to be "
         "misleading (section 8). Section 8 adds eight seeds and a two-level bootstrap for the four "
         "arms that matter; even so the design resolves only differences of about 0.04 to 0.05, "
         "limited by the test sample."),
        ("Screening unvalidated — ",
         "the proposal specifies the manually annotated hair masks of Lama et al. (ChimeraNet) "
         "as an independent reference for screening quality, and those masks were not obtained. "
         "This is a skipped step, not a gap in the data. The threshold was calibrated on "
         "synthetic pairs and frozen, which protects the test set but does not validate the "
         "detector on real hair. The 89.8% screening prevalence is reported as a limitation."),
        ("Masks are single-rater — ",
         "the HAM10000 segmentations were created by one dermatologist (Tschandl P), "
         "network-initialised and then verified, corrected or replaced by hand in FIJI "
         "(Tschandl et al., Nature Medicine 2020). No inter-rater agreement is available, so "
         "explanation overlap is measured against one tracing."),
        ("Single corpus — ",
         "one public dataset; nothing here establishes behaviour on other scanners or "
         "populations."),
        ("Licensing unresolved — ",
         "HAM10000 is CC BY-NC 4.0; the DM-AHR repository's README (CC BY-NC 4.0) and LICENSE "
         "(Apache-2.0) contradict each other. Commercial use needs a rights decision."),
    ]:
        bullet(doc, txt, lead)

    h1(doc, "8. Stage C and Stage E: the rebuilt diffusion arm (Cv2) and the review response")
    body(doc,
         "The first diffusion run (arm C) failed because of the training defect described above. "
         "The trainer was repaired and the model retrained. A partial-noise sampler (start from "
         "the noised input at level 600 of 2,000, 50 strided DDIM steps) replaced sampling from "
         "pure noise, and the model was fine-tuned for 20,000 iterations on real hair strands "
         "harvested from train-split photos and pasted onto clean train-split photos. The "
         "procedure, checkpoint hash, sampler and endpoints were written down before any "
         "test-split use of the new model (reports/protocol_addendum_stageC.md); each of the "
         "three Cv2 classifiers was evaluated once on the test split. After an independent review, Stage E (reports/prereg_v4_review_response.md) added five seeds for four arms, a corrected statistical analysis that includes training-seed variance, and several validation-only checks.")
    stats_cv2 = pd.read_csv(logs / "statistical_comparisons_Cv2.csv")
    q2 = pd.read_csv(logs / "paired_psnr_ssim_test_Cv2.csv")
    pooled_an = pd.read_csv(logs / "pooled_seed_analysis.csv")
    h2(doc, "Classification (eight seeds per arm; test macro-F1, mean ± SD over seeds)")

    def seedcell(a, col):
        r = seed[(seed["arm"] == a) & (seed["subset"] == col)].iloc[0]
        return f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f} (3 seeds)"

    def pcell(a, sub):
        m, sd, _ = pf1(a, sub)
        return f"{m:.3f} ± {sd:.3f}"
    table(doc, ["Arm", "Full test (n=1,502)", "Hair-present (n=1,316)"],
          [[ARM_LABEL[a], pcell(a, "full"), pcell(a, "hair_present")]
           for a in ("A", "Astar", "B", "Cv2")] +
          [[ARM_LABEL["Creleased"], seedcell("Creleased", "full"), seedcell("Creleased", "hair_present")],
           [ARM_LABEL["C"], seedcell("C", "full"), seedcell("C", "hair_present")]],
          widths=[2.5, 2.0, 2.0], align_right_from=1)
    rows8 = pooled_an[(pooled_an["endpoint"] == "macro_f1") & (pooled_an["subset"] == "full")]
    for comp, label in (("A-B", "A vs B"), ("A-Astar", "A vs A* (resolution control)"),
                        ("A-Cv2", "A vs Cv2"), ("B-Cv2", "B vs Cv2"),
                        ("Cv2-Astar", "Cv2 vs A* (resolution control)")):
        r = rows8[rows8["comparison"] == comp].iloc[0]
        bullet(doc, f"{r['difference']:+.3f} (95% CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}, "
                    f"{r['family']}); Holm p {r['p_holm']:.3f}; with seeds held fixed the interval "
                    f"would have been {r['ci_low_seeds_fixed']:+.3f} to "
                    f"{r['ci_high_seeds_fixed']:+.3f}.", label + ": ")
    body(doc, "Only the original-over-DullRazor difference is established, and DullRazor damages "
              "lesions on real photos, so it says little about hair removal in general. The "
              "intervals resample lesion groups and training seeds. The minimum detectable "
              "difference is 0.04 to 0.05 (80% power), set mainly by the size of the test "
              "sample, so this design cannot resolve a 0.02 difference whatever the seed count. "
              "All Cv2 rows are exploratory because Cv2 was designed after the Stage B gate "
              "failed. Standardised effect sizes for macro-F1 are not reported because the "
              "earlier ones were not interpretable; per-image log-loss shows no established "
              "difference either (pooled_seed_analysis.csv).", size=9.5)
    h2(doc, "Per-seed differences behind the pooled comparisons (macro-F1)")
    ps_ = pd.read_csv(logs / "pooled_per_seed.csv")
    ps_ = ps_[ps_["subset"] == "full"]
    pooled_ = pd.read_csv(logs / "pooled_seed_analysis.csv")
    pooled_ = pooled_[pooled_["subset"] == "full"]
    prow = []
    for name_ in ("A-B", "A-Astar", "A-Cv2", "B-Cv2", "Cv2-Astar"):
        a_, b_ = name_.split("-")
        pa_ = ps_[ps_["arm"] == a_].set_index("seed")["macro_f1"]
        pb_ = ps_[ps_["arm"] == b_].set_index("seed")["macro_f1"]
        common_ = sorted(set(pa_.index) & set(pb_.index))
        dd_ = [float(pa_[s] - pb_[s]) for s in common_]
        f1_ = pooled_[(pooled_["endpoint"] == "macro_f1") & (pooled_["comparison"] == name_)].iloc[0]
        ll_ = pooled_[(pooled_["endpoint"] == "log_loss") & (pooled_["comparison"] == name_)].iloc[0]
        prow.append([name_, str(f1_["family"]), f"{f1_['difference']:+.3f} ({f1_['ci_low']:+.3f} to {f1_['ci_high']:+.3f})",
                     f"{f1_['p_holm']:.3f}", ", ".join(f"{v:+.3f}" for v in dd_),
                     f"{sum(v > 0 for v in dd_)} of {len(dd_)}",
                     f"{ll_['difference']:+.3f} ({ll_['ci_low']:+.3f} to {ll_['ci_high']:+.3f})"])
    table(doc, ["Comparison", "Family", "Pooled macro-F1 diff. (95% CI, lesions and seeds)", "p (Holm)",
                "Per-seed differences (seeds " + ", ".join(str(s) for s in common_) + ")",
                "Seeds a higher", "Per-image log-loss diff. (95% CI)"],
          prow, widths=[0.75, 0.85, 1.35, 0.5, 1.9, 0.5, 1.25], font=7, align_right_from=2)
    body(doc, "The pooled difference replaces the standardised effect size used earlier, which divided by the "
              "spread of per-lesion differences and could take the wrong sign. Rows involving Cv2 are "
              "exploratory. Negative log-loss differences favour the first arm.", size=9.5, italic=True)
    h2(doc, "Image quality on the 186 synthetic test pairs")
    mq = {c: q2[c].mean() for c in ("hairy_input_psnr", "B_psnr", "Cv2_psnr", "Creleased_psnr",
                                    "hairy_input_ssim", "B_ssim", "Cv2_ssim", "Creleased_ssim")}
    table(doc, ["Input", "PSNR (dB)", "SSIM"],
          [["Hairy input", f"{mq['hairy_input_psnr']:.2f}", f"{mq['hairy_input_ssim']:.3f}"],
           ["B — DullRazor", f"{mq['B_psnr']:.2f}", f"{mq['B_ssim']:.3f}"],
           ["Cv2 — rebuilt diffusion", f"{mq['Cv2_psnr']:.2f}", f"{mq['Cv2_ssim']:.3f}"],
           ["Creleased (its own sampler)", f"{mq['Creleased_psnr']:.2f}",
            f"{mq['Creleased_ssim']:.3f}"]],
          widths=[3.0, 1.5, 1.5], align_right_from=1)
    r_b = row(stats_cv2, "paired_psnr", "Cv2-B")
    r_h = row(stats_cv2, "paired_psnr", "Cv2-hairy input")
    body(doc, f"Cv2 improves on the hairy input by {float(r_h['difference']):+.2f} dB (95% CI "
              f"{float(r_h['ci_low']):+.2f} to {float(r_h['ci_high']):+.2f}) but is "
              f"{float(r_b['difference']):+.2f} dB against DullRazor (CI "
              f"{float(r_b['ci_low']):+.2f} to {float(r_b['ci_high']):+.2f}). The fidelity "
              "hypothesis is not supported on these pairs. The synthetic hair is simple dark "
              "curves that the detector finds exactly, DullRazor repairs only those pixels, and "
              "the diffusion model regenerates the whole 128 px image and loses fine detail. "
              "The pairs say nothing about real hair.", size=9.5)
    h2(doc, "Explanation results for the rebuilt diffusion arm (exploratory)")
    cmp_ = x2["comp"]
    for metric in ("recall", "pointing"):
        sub_ = cmp_[(cmp_["a"] == "Cv2") & (cmp_["b"] == "Astar") & (cmp_["frame"] == xt.AS_IS)
                    & (cmp_["metric"] == metric)]
        for r in sub_.itertuples():
            bullet(doc, f"Cv2 - A*, {r.method}, {metric}: {r.difference:+.3f} (95% CI {r.ci_low:+.3f} to "
                        f"{r.ci_high:+.3f}; per-seed {r.per_seed_differences}); "
                        f"{'struck: seed spread exceeds the difference' if r.struck else 'not struck'}.")
    body(doc, "A* is the resolution-matched control, added to the statistics after the planned rows had "
              "been seen, so every Cv2 row is exploratory. The full tables, with the shifted-lesion "
              "robustness test, are in section 4 and in reports/xai_corrected_results.md.", size=9.5)
    h2(doc, "Checks added after the independent review (validation data only)")
    comp = json.loads((logs / "stageE_composite_arms.json").read_text())
    ca = comp["arms"]
    table(doc, ["Pasted real strands, 96 validation composites", "PSNR (dB)", "Strand removal",
                "Lesion pixels changed", "Skin pixels changed"],
          [[{"hairy_input": "Hairy input", "B_dullrazor": "B - DullRazor", "Cv2": "Cv2 - rebuilt diffusion"}[k],
            f"{ca[k]['psnr']['mean']:.2f}", f"{ca[k]['hair_removal']['mean']:.3f}",
            f"{100 * ca[k]['lesion_changed']['mean']:.1f}%", f"{100 * ca[k]['skin_changed']['mean']:.1f}%"]
           for k in ("hairy_input", "B_dullrazor", "Cv2")],
          widths=[2.5, 0.9, 1.0, 1.2, 1.1], align_right_from=1)
    pdif = comp["paired_Cv2_minus_B"]
    body(doc, f"Lesion and skin pixels changed are measured only where the hair did not cover them, "
              f"so any change there is damage. Cv2 minus DullRazor, paired: PSNR "
              f"{pdif['psnr']['mean']:+.2f} dB (95% CI {pdif['psnr']['ci_low']:+.2f} to "
              f"{pdif['psnr']['ci_high']:+.2f}), strand removal {pdif['hair_removal']['mean']:+.3f}, "
              f"lesion pixels changed {100 * pdif['lesion_changed']['mean']:+.1f} points. DullRazor "
              "is better on every measure here. The composite targets are the cleanest photos, so "
              "this does not show how DullRazor behaves on photos with ink or dense hair.", size=9.5)
    er = json.loads((logs / "stageE_residual_hair_and_lesion_change.json").read_text())
    bullet(doc, f"on 52 real validation photos DullRazor changed {100 * er['v2']['B']['mean_lesion_changed']:.1f}% "
                f"of lesion pixels and Cv2 {100 * er['v2']['Cv2']['mean_lesion_changed']:.1f}% "
                "(hair crossing the lesion counts as change for both).", "Lesion change on real photos — ")
    bullet(doc, "the pre-registered rule for a train/test asymmetry fired, but untrained DullRazor "
                "shows the same gap, so the rule is confounded by which photos are harvested. A "
                "post-hoc comparison with validation photos meeting the same harvest criterion "
                "shows the train photos keep slightly more hair, not less.", "Residual hair — ")
    bullet(doc, "all four models use the hairy input at noise levels up to 1000; at 1500 the released "
                "model still does and the repaired models barely do (error ratio with the input "
                "zeroed: 2.6 to 4.2 against 1.1 to 1.45).", "Use of the conditioning image — ")
    h2(doc, "What is and is not validated")
    bullet(doc, "the Stage B gate for removing hair from real photographs was not met "
                "(inside/outside ratio 1.95 and 1.57 against 2.0; follow-up composite rule 0.14 "
                "and 0.15 against 0.20). Stage C ran anyway as a recorded deviation (decision "
                "D34).", "Gate not met — ")
    bullet(doc, "an agent-reviewed look at 12 dense-hair validation photos shows nearly all hair "
                "removed with lesion outline and colour kept and fine texture lost; the "
                "ground-truth test on pasted real strands removes 77% of strand error at "
                "t0=600. This is supporting evidence, not a passed pre-registered test. A blinded "
                "human review kit (50 photos) is built and awaits a person's ratings.",
           "What supports removal on photographs — ")
    figure(doc, artifact_dir("diagnosis") / "dense_hair_sheet_stageB_val_a.png",
           "Validation photos with dense hair. Columns: original, DullRazor, first fixed model, "
           "rebuilt model at t0=600, rebuilt model at t0=1000 (128 px, agent-reviewed).",
           width=5.6)

    h1(doc, "9. Reproducibility")
    if repro:
        bullet(doc, f"Arm A seed 42 re-run from scratch reproduced validation macro-F1 exactly "
                    f"({repro['recorded']['best_val_macro_f1']:.6f} vs "
                    f"{repro['reproduction']['best_val_macro_f1']:.6f}).", "Determinism — ")
        bullet(doc, "the frozen split hash was reproduced byte for byte across two Modal "
                    "accounts, and the full data-preparation chain was re-run natively with no "
                    "Modal present, giving the identical hash.", "Portability — ")
    bullet(doc, "notebooks/01_results_walkthrough.ipynb reads the committed artifacts and "
                "explains these tables interactively; notebooks/02 and 03 reproduce the pipeline "
                "on Modal or a local GPU.", "How to explore — ")
    body(doc,
         "Every table here is generated from artifacts/logs/: classification_seed_summary.csv, "
         "classification_per_class.csv, paired_psnr_ssim_test.csv, xai_localization_*.csv and "
         "statistical_comparisons.csv. This document is built by scripts/14_results_report.py "
         "and can be regenerated after any rerun.", size=9, italic=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main() -> int:
    out = build()
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
