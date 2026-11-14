# Stage E pre-registration: response to the independent review (committed before any of it is run)

Written after an independent reviewer read `reports/SECOND_OPINION_BRIEF.md` and raised points that bear on how far the Stage C results can be trusted. The owner approved doing the reviewer's plan without the hybrid. Everything below is fixed at the commit that adds this file. Nothing here may be changed after a result is seen; changes go in a dated addendum.

## 0. What this stage is and is not

- It repairs the **statistics and measurement** of what already exists and adds seeds. It does **not** change any image, model or classifier recipe, and builds no new arm.
- The hybrid (diffusion as detector plus classical fill) and the 256 px retrain are **not** done. Stage D stays as a recorded failed attempt.
- No metric will be tuned toward a target. The reference ranges in section 7 are context supplied by the reviewer, not bars, and not requirements of the client (`docs/ACCEPTANCE_CRITERIA.md`: no numeric target is promised). Every number is reported where it falls.

## 1. Statistics corrections (frozen)

| # | Problem | Rule from now on |
|---|---|---|
| S1 | Cv2 rows were labelled `planned`, but Cv2 was designed after the Stage B gate failed (D34) | Every row involving Cv2 or Creleased is family **exploratory**. The planned family is A vs B only; A vs A\* is the secondary control. Both stay in the original files, unchanged |
| S2 | `effect_size_dz` for macro-F1 divides by the SD of per-lesion macro-F1 differences, which is not additive; it can have the wrong sign (A minus Cv2: difference +0.031, dz -0.001) | No standardised effect size is reported for macro-F1. The effect size is the difference in macro-F1 with its CI. A new endpoint, **per-image log-loss** (lower is better), is added; its dz is valid because per-image losses are additive |
| S3 | Confidence intervals resample lesions but hold the three seeds fixed, so they omit training-seed variance | The pooled analysis (section 3) uses a **two-level bootstrap**: resample lesion groups, and resample seeds within each arm |
| S4 | Grad-CAM overlap swings by seed (arm A: 0.223, 0.439, 0.426), as large as between-arm differences | Explanation overlap is reported **per seed**. No between-arm claim is made from Grad-CAM; rollout is reported with the same caveat. Both are compared with the centred-Gaussian baseline stored in the same files (0.644) |
| S5 | `dz` flag "negligible" depended on S2's dz | The flag is applied only to the log-loss rows |

## 2. Extra seeds (this is the only new test-split use)

- Arms: **A, A\*, B, Cv2**. Seeds: **7, 11, 17, 23, 31**, added regardless of how the first three came out. That gives 8 seeds per arm (42, 123, 2026 and these five), 20 new ViT runs.
- Recipe, split, images, evaluation transform and code are unchanged. Cv2 images are the frozen ones from Stage C. Each new checkpoint is evaluated **once** on the 1,502 test images. Explanation maps are **not** computed for the new seeds.
- Not run for arms C (retired failed run) and Creleased (exploratory, its images come from an unreproducible routine); they keep three seeds.
- Why these seeds and this number: the reviewer estimated about 12 seeds for a 0.02 difference; 8 seeds roughly halves the seed-induced standard error compared with 3 at about 3.5 GPU-hours of the estimated 5 for 12 (price unverified). The achieved minimum detectable difference is computed from the data (section 3) and reported; if it is above 0.02 the report says the comparison is underpowered for 0.02.
- Stop rule: no further seeds are added after these 20, whatever the pooled result.

## 3. Pooled analysis (`scripts/53_pooled_analysis.py`, frozen design)

- Statistic per arm and seed: test macro-F1 (full and hair-present subsets) and mean per-image log-loss; the arm score is the mean over its seeds.
- Two-level bootstrap, 10,000 replicates, seed 20261113: each replicate resamples lesion groups (shared across arms) and, independently within each arm, resamples its seeds with replacement. Percentile 95% CI; two-sided p as for the existing analysis; Holm within each (endpoint, subset, family).
- Comparisons: A−B (planned), A−A\* (secondary control), A−Cv2, B−Cv2, Cv2−A\* (exploratory).
- Reported with each comparison: difference with CI, raw p, Holm p, the bootstrap standard error of the difference, and the minimum detectable difference (2.8 times that standard error, about 80% power at 5% two-sided). For log-loss also dz from per-lesion paired differences.
- Reading: a difference counts as established only if the CI excludes zero and Holm p is below 0.05. With exploratory family labels, even then it is described as exploratory.

## 4. Validation-only checks (descriptive, no pass/fail bars; none reads a test image)

| Check | What | Script |
|---|---|---|
| V1 | Ground-truth composites (96 validation composites, seed 100): DullRazor (B) and Cv2 scored for PSNR at 128 px, strand removal, and **lesion damage on lesion pixels the hair did not cover**: mean absolute error against the clean target there, and fraction of those pixels changed by more than 10 gray levels. Bootstrap CI over items | `scripts/50_composite_arms.py` |
| V2 | Real validation photos (the 40 gate photos and the 12 dense-hair photos): fraction of lesion pixels changed by more than 10 gray levels, B and Cv2, at 128 px. Hair crossing the lesion also counts as change, equally for both arms | `scripts/52_residual_hair_split.py` |
| V3 | Residual hair after Cv2 on train photos whose strands were harvested for training, other train photos, and validation photos (300 hair-present photos each, detector coverage at 128 px after over before). Asymmetry is declared if the median remaining fraction on harvested train photos is lower than on validation photos by more than 0.05 absolute with a bootstrap CI excluding zero | `scripts/52_residual_hair_split.py` |
| V4 | Condition usage: epsilon-prediction error with the conditioning image kept, zeroed, and shuffled (taken from another pair), at t of 200, 500, 1000, 1500, split into hair pixels and other pixels, on 48 validation pairs, for the released checkpoint, the repaired 20k model, the 50k model and the Stage B model. "Uses the condition" is read as error ratio zeroed over kept above 1.5 at t of 200 | `scripts/51_condition_usage.py` |

## 5. Human review kit (needs a person; I cannot supply independent ratings)

50 random validation hair-present photos (fixed seed). Per photo: the original and three candidate outputs in random order (DullRazor, Cv2, and the plain 128 px original as a hair-remains control), all at 128 px, upscaled 3x so sharpness does not give the arm away. A ratings sheet asks for hair remaining (0 to 3) and lesion preserved (0 to 3) per candidate. The key to which letter is which arm is stored separately and is read only by the analysis script. Until a person returns ratings, **real-photo hair removal remains unvalidated** and every document keeps saying so.

## 6. Recorded deviations and facts found while preparing this stage

- **Gradient clipping.** Upstream has no gradient clipping anywhere in the vendored code; my trainer clips the gradient norm at 1.0 (`train.py`, "clip 1.0, as before"). That is an undisclosed deviation from upstream, and the paper does not mention clipping either. It engaged on 5 of 8 draws in the 20k-model diagnostic. Nothing is retrained here; it stays listed as a deviation and as one possible cause of the high-noise weakness.
- **EMA.** The upstream config contains an `ema_scheduler` block commented "not used now", and no vendored Python file mentions EMA. The paper's training details do not mention EMA. My trainer uses none. EMA is therefore not a deviation, and not a promising lever.
- **Paper reference numbers.** The paper (Cancers 2024;16(17):2947) reports average PSNR 41.04 dB and SSIM 0.9656 for DM-AHR(SS) on its 700-image synthetic-hair test set (DM-AHR without self-supervision 39.13 dB / 0.9575, SwinIR 39.76 / 0.9588, SRGAN 31.91 / 0.8803, SDM 34.38 / 0.8779). Different hair and pairs, no hairy-input floor: context only. My earlier statement that I could not confirm this was wrong; my audit had not read the table.
- The synthetic pairs' sources equal the hair-absent images exactly (1,022 images; train 654, val 182, test 186), verified.

## 7. Reference ranges (reviewer's rough estimates, not requirements; reported against, not tuned toward)

Accuracy 0.82 to 0.90; macro-F1 0.65 to 0.80; balanced accuracy 0.65 to 0.80; macro AUC 0.94 to 0.98; melanoma recall 0.55 to 0.80; seed SD of macro-F1 up to 0.01 to 0.02; synthetic-pair fidelity: floor 22.1 dB / 0.52, "credible" about 27 dB and SSIM 0.70, "good" about 30 dB and 0.90 (these two are the reviewer's heuristics, and the report will not use them as pass/fail lines). Explanation overlap: uniform chance 0.20, centred-Gaussian baseline 0.644.

Metrics already known to fall outside these ranges, so nobody is surprised: arm C (retired failed run) everywhere; B melanoma recall 0.529; Cv2 seed SD 0.021; Cv2 fidelity below the heuristic "good" lines; every arm's explanation overlap below the centred-Gaussian baseline.

## 8. Cost

20 ViT runs of about 9 minutes training plus about 1 minute evaluation on A100-80GB, about 3.3 GPU-hours; checks V1 to V4 well under 0.3 GPU-hours; analysis on CPU. Estimated $8 to $12 at an assumed $2.5 to $3.5 per hour; Modal prices not verified.

## 9. Addendum after the results (dated 2026-11-13; results are in `reports/stageE_results.md`)

- Everything in sections 1 to 6 was run as written. Measured cost: 3.0 GPU-hours for the 20 runs (no preemption), about 0.1 for the checks.
- **V3 rule fired and is confounded.** Harvested train photos against all validation photos gave an asymmetry verdict of "yes" for Cv2 (median remaining coverage 0.489 against 0.957), and also for DullRazor (0.331 against 0.731), which never trained on anything: the harvest criterion selects photos that are easy to clean. The verdict as pre-registered is recorded. **Post hoc**, a matched comparison against validation photos meeting the same harvest criterion (n=204) was added: harvested train photos keep slightly more hair than harvested validation photos (Cv2 +0.081, B +0.062), so no sign of the feared asymmetry. The post-hoc group was chosen after seeing that B showed the same gap.
- **The "underpowered for 0.02" flag applies to every macro-F1 comparison:** minimum detectable difference 0.039 to 0.052 with eight seeds. The cause is the test-set sample (about 1,400 lesion groups), not seed count; section 2's reasoning about seeds alone was incomplete.
- **V1 outcome not anticipated in this plan:** DullRazor beats Cv2 on pasted real strands, including on lesion pixels the hair did not cover. Reported as measured, with the caveat that composite targets are the cleanest photos.
- Not done: the human ratings (needs a person), any re-training. Gradient clipping stays a listed deviation.

