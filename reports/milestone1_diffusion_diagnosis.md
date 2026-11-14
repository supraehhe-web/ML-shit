# Milestone 1: diffusion diagnosis and reproduction status

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

Status: **for review, not final.** Nothing was retrained for this milestone and the held-out test set was not re-scored. Every figure traces to a file listed in section 6. Decisions: `reports/decision_log.md` D25, D39.

## 1. Bottom line

1. **The old diffusion arm failed because of a trainer defect, not because the images were over-smoothed.** The original training script gave the network a noise-level signal that was inverted relative to what the model's sampler expects. "Over-smoothed" is withdrawn (section 2).
2. **Training length is not shown to be the cause, and the premise is partly out of date.** The paper's own text says 20,000 iterations; the released config says 660,000 and the released weights are at 725,000. The model behind the current diffusion arm (Cv2) was trained for 50,000 iterations and then fine-tuned for 20,000, which is 70,000 in total. Whether more iterations help has not been measured; a learning-curve run is proposed (Milestone 2).
3. **Sampling steps cost accuracy but did not cause the failure.** On the released checkpoint, going from 2,000 to 1,000 steps cost about 2.1 dB; ten steps, the paper's quoted inference setting if read as a step count, is barely above the hairy input.
4. **The paper's numbers are not reproduced.** The paper reports 41.04 dB / SSIM 0.9656 on its own 700-image DERMAHAIR test set. The released checkpoint scores about 27 dB on our synthetic pairs. The dataset itself could not be obtained: it needs a Kaggle login that this workspace does not have.
5. **A correction to earlier explainability reports** was found while verifying this milestone (section 5). It does not change the diffusion findings.

## 2. What was wrong with the original diffusion arm

| Question | Finding | Evidence |
|---|---|---|
| Over-smoothed? | No. The first model's output carried about 15 times the high-frequency energy of the clean reference (0.269 against 0.018) while its pixel spread was only modestly reduced, and its structure was unrelated to the target (SSIM 0.07). That is noise, not blur. Caveat: the hairy input's own high-frequency energy was never measured, so "noise" is the better reading but is not proven. | `reports/arm_c_diagnosis.md` (section 4, marked superseded for its cause) |
| Cause | The trainer's noise-level encoding was inverted relative to the sampler. After the repair the model uses its conditioning image at low noise levels, and a sampler that starts from the noised hairy image removes synthetic hair on validation pairs by about 9 dB over the hairy input. | `reports/oneshot_diagnostic_report.md`, `reports/stageE_results.md` section 5 |
| Inference path sound? | Yes. A genuinely hairless image passed through the same path comes back at 34.6 dB (1,000 steps) and 41.7 dB (2,000 steps), keeping 91 to 93% of its high-frequency energy. | `reports/arm_c_diagnosis.md` section 1 |

## 3. Iterations and sampling steps

**Iterations.** Paper (Europe PMC full text, read for this milestone): "2000 time steps ... and 20,000 iterations". Released config: `n_iter` 660,000, 2,000 timesteps. Released weights: `I725000`. The paper's text therefore does not match its own release, and the iteration count in the paper is not evidence that 20,000 is too few. Measured training speed is about 0.055 s per iteration on one A100-80GB (`reports/compute_plan.md`), so 200,000 iterations is about 3.1 GPU-hours and 660,000 about 10.1 GPU-hours. Modal prices were not verified.

**Sampling steps.** Released checkpoint, validation synthetic pairs, 50 pairs (`artifacts/logs/paper_t10_check.json`). The paper also says "the inference number is 10 (T = 10)"; the released test config uses 2,000. I tested the reading "ten sampling steps":

| Setting | PSNR (dB) | SSIM |
|---|---:|---:|
| Hairy input (floor) | 21.50 | 0.739 |
| 10 steps | 21.76 | 0.419 |
| 25 steps | 23.52 | 0.599 |
| 50 steps | 24.55 | 0.750 |
| 2,000 steps | **27.08** | **0.807** |

Earlier, on a different set of 50 validation pairs: 2,000 steps 26.80 / 0.684 and 1,000 steps 24.66 / 0.641 (`reports/arm_c_diagnosis.md`). The two 50-pair sets have different hairy floors (21.50 and 22.11), so the 27.08 and 26.80 figures are not directly comparable. Other readings of "T = 10" were not tested.

## 4. Reproduction against the paper, and image quality

| Measure | PSNR (dB) | SSIM | Data |
|---|---:|---:|---|
| Paper, DM-AHR(SS) | 41.04 | 0.9656 | its own 700 DERMAHAIR test images (context, not a target) |
| Released checkpoint, its own sampler (Creleased) | 27.08 | 0.703 | our 186 synthetic test pairs |
| Rebuilt diffusion arm (Cv2) | 29.18 | 0.861 | our 186 synthetic test pairs |
| DullRazor (B) | 35.73 | 0.918 | our 186 synthetic test pairs |
| Hairy input | 22.13 | 0.520 | our 186 synthetic test pairs |
| First diffusion arm (C, defective trainer) | 14.29 | 0.074 | our 186 synthetic test pairs |

Cv2 minus DullRazor is -6.55 dB (95% CI -6.91 to -6.19) and -0.057 SSIM; Cv2 minus hairy input is +7.05 dB. Our synthetic hair is simple dark curves that a classical detector finds exactly, so this favours DullRazor and says little about real photographs. Real-photo hair removal is **not validated**: a blinded human review kit (`artifacts/review_kit/`, 50 photos) is built and waiting for a person's ratings.

**Why the paper gap is open.** Candidate causes, none tested: a different synthetic-hair generator and image scale in DERMAHAIR, differences in how it was cropped or resized, or training on data closer to its test distribution. Retrying DERMAHAIR (decision D9) today: no Kaggle credentials are available here, and the unauthenticated download link returns HTTP 404. **Needed from you or the client:** a Kaggle API login, or a copy of the dataset placed in the shared folder. A separate paired benchmark exists (618 pairs with and without artificial hair, University of the Balearic Islands), but it is a different dataset and would not reproduce the paper's number; I will use it only if you ask.

## 5. Correction found while verifying this milestone (explainability)

The lesion masks used to score Grad-CAM and attention roll-out were not lined up with the 224-pixel centre crop the model sees, and the mismatch differs by arm (the diffusion images are stored at 128 x 128, the original and DullRazor images at 600 x 450). On 1,502 validation masks (no model, no test data) the lesion covers 0.457 of the model's frame for the original and DullRazor arms and 0.366 for the diffusion arms, against 0.295 for the mask used. The reports also described the overlap score as the share of salient pixels inside the lesion, but the code computes the share of the lesion covered. Grad-CAM per-image scores correlate only 0.19 to 0.54 between seeds. Consequence: "the Gaussian baseline beats every model" and the H3 reading are not established as recorded. Correction notices were added to the three reports; re-measurement is Milestone 3. Numbers: `artifacts/logs/xai_registration_val.json`.

## 6. Files

`reports/arm_c_diagnosis.md`, `reports/oneshot_diagnostic_report.md`, `reports/stageC_results.md`, `reports/stageE_results.md`, `reports/compute_plan.md`, `reports/decision_log.md` (D1, D9, D25, D39), `artifacts/logs/paper_t10_check.json`, `artifacts/logs/control_released_eval.json`, `artifacts/logs/image_quality_test.json`, `artifacts/logs/image_quality_test_Cv2.json`, `artifacts/logs/xai_registration_val.json`, `scripts/56_xai_registration_check.py`, `scripts/57_paper_t10_check.py`.
