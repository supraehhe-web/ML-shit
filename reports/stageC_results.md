# Stage C results: the rebuilt diffusion arm (Cv2)

Research prototype only. Not clinically validated and not intended for diagnosis, triage or patient care.

> **Partly superseded (2026-11-13, Stage E).** The three-seed means, SDs and intervals below omit training-seed variance, the Cv2 rows were labelled "planned" (they are exploratory), and macro-F1 effect sizes (dz) are not interpretable. The pooled eight-seed analysis with a two-level bootstrap is in `reports/stageE_results.md` and replaces the Cv2 classification comparisons here. The image-quality and explanation tables below are unchanged; explanation overlap should not be used for between-arm claims (arm A swings by 0.22 across seeds, and every arm is below the centred-Gaussian baseline 0.644).

This document reports the one run of the rebuilt diffusion arm against the locked test split, under `reports/protocol_addendum_stageC.md` (committed before any test use of the new model). Every number traces to a file in `artifacts/`.

## What was run

- **Diffusion model:** the 50,000-iteration model from the repaired trainer, fine-tuned for 20,000 iterations on real hair strands pasted onto clean images (`dmahr_v2/final_gen.pth`, sha256 `dd06c216…7130`). Only train-split images supplied training strands and targets; the leakage check is recorded in `artifacts/diagnosis/dmahr_v2_training_manifest.json`.
- **Sampler:** 50 strided DDIM steps starting from the noised input at level 600 of 2,000, fp32, 128 px output.
- **Arm Cv2:** the 8,993 hair-present images replaced by the diffusion output, the 1,022 hair-absent images unchanged. 8,993 of 8,993 outputs finite; ids equal the frozen hair-present list.
- **Classifier:** the same ViT-B/16 recipe as arms A, B and A*, seeds 42, 123 and 2026. Each checkpoint was evaluated once on the 1,502 test images. The 186 synthetic test pairs were processed once.
- **Before the test split was touched,** the production path was checked on the 182 validation pairs: PSNR 29.02 dB against 22.08 dB for the hairy input (+6.94 dB, bar +5 dB).
- Earlier arm `C` (trainer with the inverted noise level) and `Creleased` stay in the record unchanged.

## Classification (test macro-F1, mean ± SD over three seeds)

| Input | Full test (n=1,502) | Hair-present (n=1,316) |
|---|---:|---:|
| A: original | **0.750 ± 0.005** | 0.749 ± 0.002 |
| A*: original at 128 px (resolution control) | 0.732 ± 0.008 | 0.735 ± 0.013 |
| B: DullRazor | 0.698 ± 0.009 | 0.691 ± 0.009 |
| **Cv2: rebuilt diffusion** | **0.720 ± 0.021** (0.699, 0.718, 0.741) | 0.714 ± 0.021 |
| C: earlier diffusion run, defective trainer | 0.426 ± 0.013 | 0.411 ± 0.017 |

Cv2 accuracy 0.860 ± 0.001, balanced accuracy 0.697 ± 0.028, macro AUC 0.966 ± 0.003 (full test).

### Planned comparisons (lesion-grouped paired bootstrap, 10,000 replicates, Holm within the planned family)

| Comparison (full test) | Difference in macro-F1 | 95% CI | raw p | Holm p | effect size d_z |
|---|---:|---|---:|---:|---:|
| A − B | +0.052 | +0.019 to +0.087 | 0.0012 | 0.0036 | 0.06 |
| A − Cv2 | +0.031 | −0.006 to +0.066 | 0.099 | 0.198 | −0.001 |
| B − Cv2 | −0.022 | −0.052 to +0.009 | 0.160 | 0.198 | −0.06 |
| Cv2 − A* (resolution control) | −0.012 | −0.040 to +0.016 | 0.386 | 0.386 | 0.05 |

Hair-present subset: A − Cv2 +0.035 (−0.003 to +0.073, Holm p 0.146); B − Cv2 −0.023 (−0.057 to +0.010, Holm p 0.167). The secondary Wilcoxon test on per-lesion correctness gives Holm p of 0.027 for B against Cv2 and 0.034 for Cv2 against A*, but every one of those effect sizes is below 0.06 and the bootstrap intervals include zero, so I do not read them as findings.

**Reading.** Original photos scored highest, but the rebuilt diffusion arm is statistically indistinguishable from the originals, from DullRazor and from the resolution-matched control A*. Of the planned comparisons only A over B is clear. No arm shows that removing hair improves classification, and none shows that it hurts. With three seeds and seed-to-seed spread of 0.02 for Cv2, differences of a few hundredths in macro-F1 cannot be resolved.

## Image quality on the 186 synthetic test pairs

| Input | PSNR (dB) | SSIM |
|---|---:|---:|
| Hairy input | 22.13 | 0.520 |
| B: DullRazor | **35.73** | **0.918** |
| **Cv2: rebuilt diffusion** | 29.18 | 0.861 |
| Creleased (released checkpoint, its own sampler) | 27.08 | 0.703 |

Cv2 − hairy input: +7.05 dB (95% CI +6.89 to +7.21) and +0.341 SSIM. Cv2 − B: **−6.55 dB** (−6.91 to −6.19, d_z −2.63) and −0.057 SSIM (−0.065 to −0.049). All Holm p below 0.0001.

**Reading.** The diffusion arm removes most of the damage from synthetic hair but scores clearly lower than DullRazor on this test. That is expected for this data: the synthetic hair is simple dark curves that the hair detector finds exactly, so DullRazor repairs only the hair pixels and leaves everything else byte-identical, whereas the diffusion model regenerates the whole 128 px image and loses fine detail everywhere. The hypothesis that diffusion gives better image fidelity than the classical method is **not supported** on these pairs. These pairs say nothing about real hair, where DullRazor was shown to damage lesions (`reports/hair_check_prereg.md`).

## Explanation overlap (X1 lesion-mask subset; share of the top 20% attention inside the lesion; higher is better)

| Arm | Grad-CAM | Attention rollout |
|---|---:|---:|
| A | 0.364 | 0.558 |
| A* | 0.406 | 0.580 |
| B | 0.392 | 0.589 |
| Cv2 | 0.410 | 0.604 |

- Cv2 over A: Grad-CAM +0.046 and rollout +0.046 (both Holm p below 0.0001). Cv2 over B: +0.018 and +0.014 (Holm p 0.002 and below 0.0001).
- **Against the resolution control A*:** Grad-CAM +0.004 (−0.006 to +0.014, p 0.43), no difference, so the Grad-CAM gain over A comes from the 128 px resolution, not from hair removal. Rollout +0.024 (0.022 to 0.026, Holm p below 0.0001), a small gain that remains after the control.
- Caution on the rollout gain: these intervals resample lesions after averaging the three seeds, so they leave out seed-to-seed variation. By seed, Cv2 exceeds A* in two of three seeds (0.622 against 0.585, 0.575 against 0.540) and ties in the third (0.606 against 0.607), and A* alone moves between 0.54 and 0.61 across seeds. I read the rollout gain as suggestive, not established. The A* comparison rows were added to the statistics script after the planned rows had been seen, because the addendum names A* as the control; they are reported exactly as computed.

## What this does and does not show

- **Does show:** a diffusion model trained with the repaired trainer and real strands removes hair from synthetic pairs (+7 dB) and, on 12 dense-hair validation photos, visibly removes nearly all hair while keeping lesion outline and colour (`artifacts/diagnosis/dense_hair_sheet_stageB_val_a.png` and `_b.png`, agent-reviewed). Classification with these images is on par with the other arms within the noise of three seeds.
- **Does not show:** that hair removal helps or hurts classification; that the diffusion method beats DullRazor on image quality (it does not, on synthetic pairs); that hair removal on photographs is validated. The Stage B gate that was meant to establish the last point was **not met** (ratio 1.95 and 1.57 against 2.0), and Stage C was run anyway as a recorded deviation (decision D34). Only a 12-image agent-reviewed look and a composite ground-truth test support removal on real photographs.
- **Known limits:** the model works at 128 px, so lesions lose fine texture (visible in the sheets); the classification comparison is on three seeds; the test pairs use our own synthetic hair; the released checkpoint very probably saw our validation and test photos during its own training.

## Files

`artifacts/logs/statistical_comparisons_Cv2.csv`, `artifacts/logs/paired_psnr_ssim_test_Cv2.csv`, `artifacts/logs/image_quality_test_Cv2.json`, `artifacts/predictions/*Cv2*`, `artifacts/logs/xai_*Cv2*`, `artifacts/logs/dmahr_sample_*Cv2*`, `artifacts/checkpoints_meta/vit_Cv2_seed*_train_result.json`. Checkpoints and images stay on the Modal volume `dmahr-vit-vol`.
