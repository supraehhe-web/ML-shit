# Stage E results: response to the independent review

Research prototype only. Not clinically validated and not intended for diagnosis, triage or patient care.

Plan and rules: `reports/prereg_v4_review_response.md` (committed before anything below was run). One analysis was added after seeing a result and is marked **post hoc**. All numbers trace to files in `artifacts/logs/` named in each section. Nothing here changes an image, a model or the classifier recipe; the only new test-split use is the 20 extra-seed checkpoints, each evaluated once.

## 1. Headline: eight seeds change the picture of the spread, not the conclusion

Test macro-F1 over eight seeds (42, 123, 2026, 7, 11, 17, 23, 31), full test n=1,502 (`pooled_per_seed.csv`, `headline_metrics_pooled.csv`):

| Arm | Mean | SD over seeds | Range | Earlier 3-seed mean ± SD |
|---|---:|---:|---|---|
| A original | 0.7411 | 0.0236 | 0.690 to 0.767 | 0.7500 ± 0.0049 |
| A\* 128 px control | 0.7278 | 0.0204 | 0.701 to 0.765 | 0.7317 ± 0.0084 |
| Cv2 rebuilt diffusion | 0.7218 | 0.0119 | 0.699 to 0.741 | 0.7196 ± 0.0210 |
| B DullRazor | 0.6934 | 0.0166 | 0.661 to 0.714 | 0.6983 ± 0.0086 |

The three-seed SDs were misleading in both directions: arm A looked far steadier than it is (0.005 against 0.024) and Cv2 looked far noisier (0.021 against 0.012). Every difference of interest is of the size of one arm's seed spread.

## 2. Pooled comparisons (`pooled_seed_analysis.csv`; two-level bootstrap over lesion groups and seeds, 10,000 replicates)

Macro-F1, full test. "Seeds fixed" is the earlier style of interval, shown to make the effect of the correction visible.

| Comparison | Family | Difference | 95% CI (two-level) | 95% CI (seeds fixed) | Holm p | Minimum detectable (80% power) |
|---|---|---:|---|---|---:|---:|
| A − B | planned | +0.0477 | +0.0121 to +0.0846 | +0.0222 to +0.0751 | 0.0084 | 0.052 |
| A − A\* | secondary control | +0.0132 | −0.0192 to +0.0472 | −0.0084 to +0.0350 | 0.411 | 0.047 |
| A − Cv2 | exploratory | +0.0193 | −0.0134 to +0.0544 | −0.0042 to +0.0442 | 0.498 | 0.048 |
| B − Cv2 | exploratory | −0.0285 | −0.0592 to +0.0016 | **−0.0517 to −0.0059** | 0.194 | 0.043 |
| Cv2 − A\* | exploratory | −0.0060 | −0.0340 to +0.0207 | −0.0241 to +0.0113 | 0.671 | 0.039 |

Hair-present subset: A − B +0.0542 (CI +0.0186 to +0.0936, Holm p 0.0032); every other comparison inconclusive. Per-image log-loss (lower is better): A 0.547, A\* 0.587, B 0.551, Cv2 0.562; every comparison inconclusive, intervals about ±0.1 (dz over lesion groups is in the CSV).

- **Only original over DullRazor is established**, and DullRazor damages lesions on real photos, so it is not evidence about hair removal in general.
- **B − Cv2 shows why the correction mattered.** With the seeds held fixed the interval excluded zero (Cv2 better than B); with seed variance included it does not.
- **The comparison is underpowered for 0.02 whatever the seed count.** The minimum detectable difference is 0.04 to 0.05, and most of it comes from sampling 1,502 images in about 1,400 lesion groups, which more seeds cannot shrink. The reviewer's estimate of about 12 seeds considered seed variance only. Reaching 0.02 would need a larger test set, not more training runs.
- The family labels follow S1: Cv2 rows are exploratory; they were designed after the Stage B gate failed (D34).

## 3. Does diffusion removal beat DullRazor on real strands? Not on these composites (V1; `stageE_composite_arms.json`)

96 validation composites (real strands pasted on clean validation photos, so the target is known), scored at 128 px, mean over items, bootstrap CI over items. Lesion damage is measured only on lesion pixels the hair did not cover, where any change from the input is damage.

| Input | PSNR (dB) | Strand removal | Lesion pixels changed (>10 gray levels) | Lesion error (gray levels) | Skin pixels changed |
|---|---:|---:|---:|---:|---:|
| hairy input | 25.59 | 0 | 0 | 0.20 | 0 |
| B DullRazor | **40.31** | **0.905** | **5.9%** | **1.36** | **1.0%** |
| Cv2 | 33.86 | 0.787 | 18.3% | 4.90 | 17.8% |

Cv2 minus B, paired: PSNR −6.45 dB (95% CI −7.34 to −5.62); strand removal −0.118 (−0.138 to −0.099); lesion pixels changed +12.4 points (+7.0 to +18.4); lesion error +3.53 (+2.94 to +4.16). **Cv2 is worse than DullRazor on every measure here, including lesion damage.** Caveats: composite targets are the cleanest fifth of photos, where DullRazor's detector finds the pasted dark strands easily and has little else to mistake for hair. On real photos the damage shows up through false positives on ink, texture and dense hair, which composites do not exercise. The diffusion model regenerates every pixel, which is where its 18% change on untouched skin comes from.

## 4. Real photos: lesion change and residual hair (V2, V3; `stageE_residual_hair_and_lesion_change.json`)

**V2, 52 real validation photos (40 gate + 12 dense), at 128 px.** Fraction of lesion pixels changed by more than 10 gray levels versus the original: B 25.1%, Cv2 10.9%; paired Cv2 − B −14.2 points (SD 20.2 over 52 photos). Hair that crosses the lesion also counts as change, equally for both arms, so this shows B alters the lesion region more, not that the alteration is all damage. Taken with section 3: on clean targets B changes little, on real photos it changes a lot, which fits false positives, though this check cannot prove the cause.

**V3, residual hair after the arm, median share of detector coverage left, 128 px.**

| Group (hair-present, n) | Cv2 | B |
|---|---:|---:|
| train photos whose strands were harvested (300) | 0.489 | 0.331 |
| other train photos (300) | 0.976 | 0.781 |
| validation photos (298) | 0.957 | 0.731 |
| validation photos meeting the same harvest criterion (204), **post hoc** | 0.407 | 0.270 |

- **Pre-registered rule fired, and it is confounded.** Harvested train photos against all validation photos: Cv2 −0.468 (CI −0.511 to −0.424), so the asymmetry rule says "yes". But DullRazor, which never trained on anything, shows the same gap (−0.400, CI −0.445 to −0.360). The harvest criterion selects photos with clear thin dark strands, which both arms clean more easily. I record the rule's verdict as written.
- **Post hoc matched comparison** (added after seeing this): harvested train against harvested validation photos, the same selection and never trained on. Cv2 +0.081 (CI +0.010 to +0.161) and B +0.062 (CI +0.014 to +0.112): the train photos keep slightly **more** hair, not less. There is no sign that the classifier trains on cleaner images than it is tested on. The detector is the project's own instrument at 128 px, with the limits noted before.

## 5. Condition usage (V4; `stageE_condition_usage.json`; 48 validation pairs)

Ratio of epsilon-prediction error with the hairy conditioning image zeroed over kept (hair pixels / other pixels). A ratio above 1.5 at t=200 was pre-registered as "uses the condition".

| Model | t=200 | t=500 | t=1000 | t=1500 | Error, condition kept, t=1500 (hair / other) |
|---|---|---|---|---|---|
| released | 2.99 / 6.49 | 2.86 / 6.49 | 3.22 / 8.20 | 2.64 / 4.23 | 0.0064 / 0.0039 |
| repaired 20k | 3.56 / 4.88 | 4.03 / 4.63 | 2.91 / 2.88 | 1.13 / 1.11 | 0.0122 / 0.0114 |
| repaired 50k | 5.43 / 8.77 | 6.33 / 8.04 | 3.94 / 3.89 | 1.30 / 1.20 | 0.0099 / 0.0093 |
| Stage B v2 | 3.01 / 7.10 | 5.18 / 9.36 | 3.70 / 4.55 | 1.36 / 1.45 | 0.0092 / 0.0084 |

All four use the condition at t=200 (shuffled gives the same picture). At t=1500 the released model still leans on it by 2.6 to 4.2 times while mine barely do (1.1 to 1.45 times) and their error with the condition kept is 1.4 to 2.9 times higher. Cv2 never samples above t=600, so its behaviour is unaffected, but this is consistent with why pure-noise sampling, which must start where the condition matters little to my models, failed. It does not prove the cause.

## 6. Explanation overlap: no arm beats a centre prior (`xai_localization_*_X1.csv`, per seed)

> **Post-review correction (2026-11-13).** Three problems were found after this section was written. The numbers below are reported as recorded; the conclusions drawn from them are **not established** until re-measured (plan: `reports/client_plan_and_timeline.md`, Milestone 3).
>
> 1. **Mask frame.** The lesion mask was squashed from the whole photo to 224 x 224, but the model sees the photo resized to 255 px on the short side and centre-cropped to 224. Stored images differ by arm (A and B 600 x 450; A*, Cv2, Creleased 128 x 128), so the mismatch differs by arm. On the 1,502 validation masks (`artifacts/logs/xai_registration_val.json`; no model, no test data) the lesion occupies 0.457 of the model's frame for A and B and 0.366 for the diffusion arms, against 0.295 for the mask used here (IoU with it 0.61 and 0.79). The centred-Gaussian baseline scores 0.642 on the squashed mask and 0.466 (A, B) or 0.555 (diffusion arms) on the registered one. The between-arm comparison is confounded and "no arm beats a centre prior" is not established.
> 2. **Metric definition.** The CSV column is the share of the *lesion* covered by the top 20% of the map (`lesion_overlap`, which is why the uniform baseline is 0.200). The text here, the proposal and the kit describe the share of the top 20% that falls *inside* the lesion. Both will be reported in the re-measurement.
> 3. **Seed stability.** Per-image Grad-CAM overlap correlates only 0.19 to 0.54 (Spearman) between independently trained seeds; rollout correlates 0.96 to 0.99, so rollout depends little on the trained weights. Neither can be read as a property of an arm without this caveat.

Share of the top 20% attention inside the lesion, X1 mask subset, n=1,502 images per arm and seed. Baselines stored in the same files: uniform 0.200, centred Gaussian **0.644**.

| Arm | Grad-CAM, mean (range over 3 seeds) | Rollout, mean (range) |
|---|---|---|
| A | 0.363 (0.223 to 0.439) | 0.555 (0.514 to 0.601) |
| A\* | 0.404 (0.357 to 0.477) | 0.577 (0.540 to 0.607) |
| B | 0.388 (0.363 to 0.411) | 0.586 (0.547 to 0.607) |
| Cv2 | 0.406 (0.389 to 0.424) | 0.601 (0.575 to 0.622) |

Every arm, every seed, is below the centred-Gaussian baseline for both methods. Per S4, no between-arm claim is made from these rows: arm A's Grad-CAM swings by 0.22 across seeds, as much as any gap between arms, and the earlier intervals did not include that. The extra seeds have no explanation maps (not run, by design), so these stay three-seed.

## 7. Reference ranges (reviewer's rough estimates, not requirements; nothing was tuned toward them)

Eight-seed means, full test (`headline_metrics_pooled.csv`):

| Metric | Range | A | A\* | B | Cv2 |
|---|---|---:|---:|---:|---:|
| Accuracy | 0.82 to 0.90 | 0.858 | 0.853 | 0.837 | 0.859 |
| Macro-F1 | 0.65 to 0.80 | 0.741 | 0.728 | 0.693 | 0.722 |
| Balanced accuracy | 0.65 to 0.80 | 0.732 | 0.716 | 0.688 | 0.699 |
| Macro AUC | 0.94 to 0.98 | 0.970 | 0.967 | 0.965 | 0.966 |
| Melanoma recall (167 images) | 0.55 to 0.80 | 0.590 | 0.582 | 0.565 | 0.592 |
| Seed SD of macro-F1 | up to 0.02 | 0.024 | 0.020 | 0.017 | 0.012 |

All four live arms are inside every range; A and A\* are slightly above the 0.02 SD line. Melanoma recall swings by seed (SD 0.05 to 0.13), so it cannot rank arms. The retired first diffusion run (arm C, defective trainer) is below every range (macro-F1 0.426) and stays documented as a failure. Creleased (3 seeds, exploratory) is inside all of them. Synthetic-pair fidelity: Cv2 29.2 dB / SSIM 0.861 against the reviewer's heuristic "good" lines of about 30 dB / 0.90; I do not use those lines as pass or fail. Explanation overlap is below the centre-prior baseline (section 6), which is a limit of the metric as specified.

## 8. Corrections to earlier statements

- Three-seed SDs and the standard errors built on them (sections 1, 2).
- Cv2 was labelled planned; it is exploratory. Macro-F1 `dz` is withdrawn for the Cv2 file; the original file's macro-F1 `dz` values are not interpretable and are not used anywhere.
- I said I could not confirm the paper's PSNR. The paper reports 41.04 dB / 0.9656 for DM-AHR(SS) on its own 700 synthetic-hair test images. Different pairs and no hairy-input floor; context only.
- Upstream applies no gradient clipping; my trainer clips at 1.0. EMA is "not used now" in the upstream config. Both recorded in the pre-registration, section 6.
- The earlier statement that the high-noise error was "flat" was too strong (D37).

## 9. What is still not validated

Hair removal on real photographs. The only evidence is my own visual review, the pasted-strand test (where DullRazor does better), and the V2/V3 numbers above. The blinded human review kit is built (`artifacts/review_kit/`, 50 validation photos, ratings sheet, key, analysis script) and needs a person to fill in `ratings_template.csv`; until it is, real-photo removal stays unvalidated in every document.

## 10. Bottom line for the deliverable

Hair removal, as built here, made no measurable difference to seven-class classification: the rebuilt diffusion arm is statistically indistinguishable from the originals, from the 128 px control and from DullRazor. On pixel fidelity the diffusion arm is clearly behind DullRazor on both synthetic hair and pasted real strands. Explanation overlap is below a centre prior for every arm. Those are the findings; none was tuned toward a target.

## 11. Cost

20 ViT runs: 3.0 GPU-hours (40 jobs, 10,835 s of container time, no preemption), checks V1 to V4 about 0.1 GPU-hours, analysis on CPU. At an assumed $2.5 to $3.5 per A100-hour that is roughly $8 to $11; Modal prices were not verified.
