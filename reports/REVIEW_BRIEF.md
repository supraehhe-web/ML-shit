> **SUPERSEDED (2026-11-13).** This document concludes that Arm C's failure is
> undertraining and that the inference path is sound. That conclusion is
> retracted: the training script gave the network an inverted noise-level signal
> (see `reports/oneshot_diagnostic_report.md` and `reports/hypotheses.md`). Cost
> figures and any "repair" plan below assumed the old script and no longer apply.
> The text is kept unchanged below for the record.

# Review brief: DM-AHR-ViT

Written for an adversarial reviewer. It states what I claim, the evidence behind each claim, and
the places I think I am most likely to be wrong. Please attack the weak points in section 5 first;
that is where the value is.

---

## 1. The study

Does generative (diffusion) hair removal help a skin-lesion classifier more than classical hair
removal, or than leaving the image alone?

Four input conditions on HAM10000, ImageNet-pretrained ViT-B/16, 7 classes, 3 seeds:

| Arm | Input | Purpose |
|---|---|---|
| A | original image | baseline |
| B | DullRazor-style morphology + Telea inpainting | classical removal |
| C | DM-AHR conditional diffusion inpainting | generative removal |
| A\* | original through a 128 px round-trip | controls Arm C's resolution change |

Split is 70/15/15 by **lesion group**, frozen and hashed before training. Test set scored once per
frozen checkpoint.

## 2. Headline results

Test macro-F1, mean ± SD over 3 seeds, 1,502 test images:

| Arm | Macro-F1 | Accuracy | Balanced acc. | Macro AUC |
|---|---|---|---|---|
| A | 0.750 ± 0.005 | 0.858 | 0.735 | 0.969 |
| A\* | 0.732 ± 0.008 | 0.851 | 0.726 | 0.969 |
| B | 0.698 ± 0.009 | 0.842 | 0.687 | 0.965 |
| C | 0.426 ± 0.013 | 0.685 | 0.452 | 0.878 |

Image quality on 186 held-out synthetic pairs (PSNR dB / SSIM):

| Condition | PSNR | SSIM |
|---|---|---|
| Hairy input (floor) | 22.13 | 0.520 |
| Arm B | 35.73 | 0.918 |
| Arm C | 16.31 | 0.130 |

## 3. The Arm C diagnosis (the main new work)

Arm C's output scored *below* the unprocessed input, with SSIM near-constant across all pairs
(0.130 ± 0.010, 185 of 186 below 0.15). Two candidate causes: undertrained model, or a defect in
the inference path. I ran a diagnosis on **validation pairs only** (test pairs never read), 100
pairs, A100-40GB, 29 minutes, $1.02.

**Identity test** — pass a genuinely hairless reference through the exact Arm C path and compare it
with itself, using the *released* checkpoint:

| Steps | n | PSNR | SSIM | high-freq ratio (out/in) |
|---|---|---|---|---|
| 1,000 | 50 | 34.62 ± 6.18 | 0.990 ± 0.004 | 0.926 |
| 2,000 | 50 | 41.72 ± 4.89 | 0.992 ± 0.001 | 0.909 |

**Head to head** — same adapter, same 100 pairs, both checkpoints, both step counts:

| Condition | PSNR | SSIM | Beats 22.11 floor? |
|---|---|---|---|
| Released, 2,000 | 26.80 ± 2.78 | 0.684 ± 0.098 | yes |
| Released, 1,000 | 24.66 ± 2.67 | 0.641 ± 0.101 | yes |
| Ours, 2,000 | 14.99 ± 1.23 | 0.079 ± 0.008 | no |
| Ours, 1,000 | 14.12 ± 1.22 | 0.073 ± 0.006 | no |

**Defect checks:** no value-range error; per-channel means track the reference; best spatial shift
0 to −1 px; flip correlations ≈ 0.

**Conclusion: outcome (b), undertrained.** The adapter is sound; the checkpoint never learned the
mapping.

**Consequence:** I withdrew the earlier description of Arm C as "over-smoothed". Our output has
mean absolute Laplacian 0.269 against the reference's 0.018 — about 15× *more* high-frequency
energy — while global pixel std falls only from 0.166 to 0.127. I read that as noise around a
compressed range, not blur.

## 4. Other claims

**H3 is uninformative.** A centred-Gaussian prior scores 0.644 on lesion overlap and 0.953 on the
pointing game, exceeding every arm on both. No arm beats a trivial attend-to-the-centre prior.

**Uniform chance of 0.200 is correct.** The metric is |top-20% ∩ mask| / |mask| — mask recall, not
saliency precision. For a random map the expectation is exactly 0.2 regardless of mask size. I
measured 0.2000 ± 0.0038 with correlation −0.00 against mask area (mean 0.293). A reviewer
previously claimed this should equal the mask area fraction; I believe that reviewer was wrong.

**A−B is statistically clear but negligible.** Difference 0.052, CI +0.019 to +0.087, Holm
p = 0.0012, Cohen's dz = 0.056. I flag it as negligible.

**C vs A\* (resolution-matched):** −0.306, CI −0.348 to −0.265, dz −0.46. Resolution cannot explain
Arm C's deficit.

**Hair-coverage deciles (exploratory):** Arm B is *better* than A in the lowest-coverage decile
(0.555 vs 0.507), and no B−A gap is significant after Holm. Arm C's deficit appears in every decile
with hair (−0.15 to −0.25) and is inconclusive in the hair-absent decile (−0.020, CI −0.051 to
+0.009).

**Duplicate merge changed nothing.** All 21 pixel-confirmed pairs already shared a `lesion_id`, so
7,470 groups equals the unique `lesion_id` count. I describe the check as verifying that no
*unlinked* duplicates exist, not as extending the grouping.

**Masks are single-rater.** The Dataverse record says one dermatologist created binary masks for
all 10,015 images, network-initialised then verified/corrected/replaced by hand in FIJI (Tschandl
et al., Nature Medicine 2020).

## 5. Where I am most likely to be wrong — please attack these

**5.1 The identity test uses the released checkpoint, not ours.** It proves the *code path* is
sound for weights that work. It does not prove our checkpoint's inference is identical, because our
checkpoint could in principle have been saved with a different config.

*Partially resolved after writing this:* `artifacts/checkpoints/dmahr/training_manifest.json`
records the config our checkpoint was trained with, and it matches `configs/dmahr.json` on every
structural field — `native_resolution`, `condition_resolution`, `num_timesteps`, `beta_schedule`
and the full `unet` block. So the architecture is the same. What this still does not prove is that
the *weights* are the only difference; a training-time bug that corrupted the weights while leaving
the config intact would look identical. The identity test cannot distinguish "undertrained" from
"trained wrongly", and I have not separated those.

**5.2 I never measured the hairy input's high-frequency energy.** My "15× the reference" claim
compares our output to the *hairless reference*. If the hairy input also has high Laplacian energy,
then "more high-frequency energy than the reference" is a much weaker statement than it sounds, and
the "noise not blur" conclusion could be overstated. I should have measured `hf_hairy`.

**5.3 I do not know what PSNR the DM-AHR paper reports.** I treat 26.80 as "sensible" because it
clears the hairy-input floor. If the paper reports 35 dB, then even the released checkpoint is
performing poorly through my adapter and my "adapter is sound" conclusion is too generous.

**5.4 The released checkpoint may have seen my validation pairs.** Its training data is
undocumented. If it was trained on HAM10000 broadly, the 26.80 figure could be optimistic. This does
not affect the identity test (which only asks whether the path works) but it does affect how much
weight the 26.80 deserves.

**5.5 Arm C's classification result may be better described as invalid than as a result.** I report
C's macro-F1 of 0.426 and label it an under-resourced ablation. An alternative reading is that a
classifier trained on images with SSIM 0.079 to their targets is not measuring anything about
diffusion hair removal and should be marked invalid rather than reported. I chose to report it with
a caveat; that is a judgment call I would defend but not strongly.

**5.6 The centre-prior finding may indict the task, not just my measurement.** HAM10000 lesions are
often centred because images are cropped around them. If a fixed centre Gaussian beats every
saliency method, that may say the dataset makes localisation trivially easy, which is a stronger
claim than "my H3 measurement is uninformative". I have not checked lesion centroid distribution.

**5.7 The 660,000 iteration figure is from the upstream config, not the paper.** `n_iter: 660000`
in `third_party/dmahr/config/akiec.json`, and the released checkpoint is named `I725000_E1083`. The
project kit separately says 20,000. I recorded the conflict but never read the paper to see what it
actually states. If the paper says 20,000, then my Arm C was trained at the paper's budget and the
entire "under-resourced" framing collapses.

**5.8 The hair-present subset is 87.6% of the test set** (1,316 of 1,502), so the subset analysis
adds little beyond the full set. I am not sure I stated this prominently enough.

**5.9 PSNR/SSIM are measured against synthetic hair only.** The synthetic hair is a proxy for real
terminal hair. H1 is therefore tested on synthetic hair, and the 186-pair sample is small.

**5.10 The decile analysis uses per-group mean coverage** to assign a lesion to a decile, which
averages over that lesion's images. A lesion with one hairless and one hairy photograph lands in a
middle decile. I have not tested sensitivity to that choice.

## 6. What I would ask a reviewer to check first

1. Is the identity test actually sufficient to exonerate the adapter, given 5.1?
2. Does 5.2 undermine the "noise not blur" claim?
3. Is reporting C's macro-F1 defensible, or should it be marked invalid (5.5)?
4. Is the uniform-chance reasoning in section 4 correct, or was the earlier reviewer right?
5. Does the centre-prior result mean H3 is uninformative, or that the dataset is trivially easy
   (5.6)?

## 7. Evidence index

| Artifact | Contents |
|---|---|
| `reports/arm_c_diagnosis.md` | full diagnosis write-up |
| `artifacts/logs/arm_c_diagnosis.json` | outcome, verdict, identity and head-to-head summaries |
| `artifacts/logs/arm_c_identity.csv` | per-image identity test, both step counts |
| `artifacts/logs/arm_c_headtohead.csv` | per-pair metrics, all four conditions |
| `artifacts/logs/statistical_comparisons.csv` | 22 comparisons, both tests, Holm |
| `artifacts/logs/hair_coverage_by_decile.csv` | macro-F1 per arm per coverage decile |
| `artifacts/logs/hair_coverage_gaps.csv` | per-decile gaps vs Arm A with cluster CIs |
| `artifacts/logs/classification_seed_summary.csv` | headline metrics |
| `artifacts/logs/paired_psnr_ssim_test.csv` | per-pair PSNR/SSIM |
| `artifacts/logs/xai_localization_*.csv` | per-image saliency metrics and baselines |
| `reports/decision_log.md` | every deviation and its reason (D1–D21) |
| `scripts/16_verify_report_claims.py` | recomputes 42 report claims; all pass |
| `scripts/20_deliverable_gates.py` | V1–V8 gates; 8/8 pass |
