# DM-AHR-ViT: diffusion hair removal and ViT classification on HAM10000

**Research prototype only. Not clinically validated and not intended for diagnosis, triage or patient care.**

> **Post-review correction (2026-11-13), explanation results.** Every statement below about explanation overlap, the centred-Gaussian baseline (0.644) and H3 is subject to three problems found after it was written: the lesion masks were not registered to the cropped frame the model sees (and differently so per arm), the reports described the overlap metric differently from how the code computes it (lesion coverage, not share of salient pixels inside the lesion), and per-image Grad-CAM overlap is unstable across seeds (Spearman 0.19 to 0.54). Evidence and numbers: `reports/stageE_results.md` section 6 and `artifacts/logs/xai_registration_val.json`. The conclusions are not established until re-measured (`reports/client_plan_and_timeline.md`, Milestone 3).

## Summary

Test macro-F1 by arm (mean ± SD over seeds): Original 0.741 ± 0.024; A* (128 px) 0.728 ± 0.020; **rebuilt diffusion arm Cv2 0.722 ± 0.012**; DullRazor 0.693 ± 0.017 (these four over eight seeds, `reports/stageE_results.md`); Creleased (released ckpt, exploratory, three seeds) 0.713 ± 0.028; earlier diffusion run C (defective trainer, three seeds, kept as a recorded failure) 0.426 ± 0.013. The first three-seed means were Original 0.750, A* 0.732, DullRazor 0.698, Cv2 0.720; the three-seed SDs were misleading (Original 0.005 against 0.024 with eight seeds), and the Stage C paragraph below uses those first three-seed numbers.

The hypotheses were that DM-AHR hair removal improves image fidelity, classification and explanation localization relative to the classical baseline. The results below are reported as measured; a null or negative result is a valid outcome and is not reframed.

### Stage C update (rebuilt diffusion arm; full detail in `reports/stageC_results.md`)

The earlier diffusion arm failed because the trainer gave the network an inverted noise-level signal (D25). After repairing the trainer, retraining, adding a partial-noise sampler and fine-tuning on real hair strands (D34, `reports/prereg_v2_diffusion.md`), a rebuilt arm Cv2 was run once against the locked test split under `reports/protocol_addendum_stageC.md`.

- **Classification:** Cv2 0.720 ± 0.021 against Original 0.750, A* 0.732 and DullRazor 0.698. Of the planned comparisons only Original over DullRazor is clear (difference +0.052, Holm p 0.004). Cv2 against Original −0.031 (95% CI −0.066 to +0.006, Holm p 0.198); Cv2 against DullRazor +0.022 (−0.009 to +0.052, Holm p 0.198); Cv2 against A* −0.012 (Holm p 0.386). No arm shows that hair removal helps or hurts classification.
- **Image quality (186 synthetic test pairs):** Cv2 29.18 dB / SSIM 0.861, DullRazor 35.73 dB / 0.918, hairy input 22.13 dB / 0.520. The diffusion arm repairs synthetic hair (+7.05 dB) but is 6.55 dB below DullRazor (CI −6.91 to −6.19). The fidelity hypothesis is not supported on these pairs.
- **Explanation overlap:** against the resolution-matched control A*, Grad-CAM overlap does not differ (+0.004, p 0.43); attention-rollout overlap is higher by +0.024 (Holm p below 0.0001), but that interval leaves out seed-to-seed variation and Cv2 exceeds A* in only two of three seeds, so it is suggestive.
- **What is not validated:** the Stage B gate for removing hair from real photographs was **not met** (inside/outside ratio 1.95 and 1.57 against 2.0; follow-up composite rule 0.14 and 0.15 against 0.20). Stage C ran anyway as a recorded deviation (D34). The only support for removal on photographs is an agent-reviewed look at 12 dense-hair validation images (nearly all hair gone, lesions intact, fine texture lost at 128 px) and a ground-truth test on pasted real strands.

### Stage E update (response to an independent review; `reports/stageE_results.md`, decision D38)

- **Pooled over eight seeds** with a two-level bootstrap (lesion groups and seeds): only Original over DullRazor is established (macro-F1 difference +0.048, 95% CI +0.012 to +0.085, Holm p 0.008). Original against Cv2 +0.019 (CI −0.013 to +0.054, Holm p 0.50), DullRazor against Cv2 −0.029 (CI −0.059 to +0.002, Holm p 0.19), Cv2 against A* −0.006 (CI −0.034 to +0.021, Holm p 0.67). All Cv2 rows are labelled exploratory, because Cv2 was designed after the Stage B gate failed. Per-image log-loss: no comparison is established. These pooled numbers supersede the Cv2 rows of the Stage C paragraph above.
- **The study cannot resolve a 0.02 macro-F1 difference.** The minimum detectable difference is 0.04 to 0.05, set by the test sample (about 1,400 lesion groups), not by the seed count.
- **Standardised effect sizes (`dz`) for macro-F1 in the earlier tables are not interpretable and are not used**; differences with intervals, and a per-image log-loss endpoint, replace them.
- **On pasted real hair strands with known clean targets, DullRazor beats the diffusion arm on every measure**: PSNR 40.3 against 33.9 dB, strand removal 0.905 against 0.787, and lesion pixels changed where the hair did not cover them 5.9% against 18.3% (96 validation composites). The targets are the cleanest photos, so this does not show how DullRazor behaves on real photos with ink or dense hair, where it changes more of the lesion region than the diffusion arm does (25.1% against 10.9% on 52 validation photos, hair crossing the lesion included).
- **Condition usage:** every model uses the hairy input at noise levels up to 1000; at 1500 the released model still does and the repaired models barely do. No arm's explanation overlap exceeds the centred-Gaussian baseline (0.644) for either method in any seed.
- **Real-photo hair removal is still not validated.** A blinded human-review kit exists (`artifacts/review_kit/`) and awaits ratings from a person.

Earlier real-hair finding, unchanged: the released checkpoint and the original DullRazor arm did not cleanly remove real hair (released model left it in place, DullRazor smeared lesions), so the Original-versus-DullRazor gap is confounded by damaged Arm B images.

**The sections below were written before Stage C and describe arm C, the earlier diffusion run with the defective trainer, as "DM-AHR". Read every "C" and "DM-AHR" result below as that failed run; Cv2 is the rebuilt arm.**

## Cohort and splits

- Collection 212 images: 11720
- Confirmatory C0 images: 10015
- Quarantined C0-extra: 1705
- Lesion groups: 7470
- Confirmed near-duplicate pairs: 21 (all already shared a lesion_id, so grouping is unchanged by them)
- Split sizes: {'test': 1502, 'train': 7011, 'val': 1502}
- Split SHA-256: `fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b`
- Leakage check: True

Recomputed class counts matched the expected HAM10000 counts exactly (True).

## Hair screening and arms

- Hair-present images: 8993 of 10015 (89.8%)
- Screening config hash: `e232efe4705133a175ec50d5147954a6545456d469b6353fb42d348387704273`
- DullRazor config hash: `0af616f50f77e1c57c4fff1e1f6bb8c2ab99100052c703d937bccbbf109a9d70`
- Synthetic pairs: 1022 ({'test': 186, 'train': 654, 'val': 182})

Hair-absent images are identical across arms. Arm A* exists because DM-AHR is native 128 px, below the 224 px ViT input.

## DM-AHR audit and training

- Pinned commit: `0aee1bbabf788a59be5e1eecddc20f533b5006d8`
- Released checkpoint SHA-256: `c99e406050d3aaf36473f76477bcfb8ba05d80a420d50d57533eac75b8c8d2fe`
- Smoke test passed: True (geometry preserved True, reload identical True)
- Sampling throughput: {'8': {'returned': 8, 'seconds': 48.8, 'seconds_per_image': 6.1}, '16': {'returned': 16, 'seconds': 75.28, 'seconds_per_image': 4.705}, '32': {'returned': 32, 'seconds': 138.96, 'seconds_per_image': 4.343}, '64': {'returned': 64, 'seconds': 261.65, 'seconds_per_image': 4.088}}
- Leakage-controlled training: 20000 iterations, best validation L1 0.01766

The released weights are used for the smoke test only, because the authors do not document their training data. Arm C uses the checkpoint trained here on train-split sources only.

## Classification results

**Arm C was run as a budget-forced ablation: 1,000 sampling steps instead of the reference 2,000.** The account balance could not cover the full-step run, so Arm C is half the reference sampling budget. Its results are reported as an ablation and must not be read as DM-AHR at the paper's settings. See decision log D12.

### Mean ± SD over seeds

| arm | subset | seeds | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd | balanced_acc_mean | auc_macro_mean | n |
|---|---|---|---|---|---|---|---|---|---|
| A | full | 3 | 0.750 | 0.005 | 0.858 | 0.012 | 0.735 | 0.969 | 1502 |
| Astar | full | 3 | 0.732 | 0.008 | 0.851 | 0.006 | 0.726 | 0.969 | 1502 |
| B | full | 3 | 0.698 | 0.009 | 0.842 | 0.005 | 0.687 | 0.965 | 1502 |
| C | full | 3 | 0.426 | 0.013 | 0.685 | 0.019 | 0.452 | 0.878 | 1502 |
| Creleased | full | 3 | 0.713 | 0.028 | 0.848 | 0.009 | 0.696 | 0.967 | 1502 |
| A | hair_present | 3 | 0.749 | 0.002 | 0.843 | 0.013 | 0.736 | 0.966 | 1316 |
| Astar | hair_present | 3 | 0.735 | 0.013 | 0.836 | 0.008 | 0.735 | 0.966 | 1316 |
| B | hair_present | 3 | 0.691 | 0.009 | 0.824 | 0.006 | 0.680 | 0.960 | 1316 |
| C | hair_present | 3 | 0.411 | 0.017 | 0.651 | 0.018 | 0.440 | 0.864 | 1316 |
| Creleased | hair_present | 3 | 0.712 | 0.029 | 0.833 | 0.010 | 0.695 | 0.962 | 1316 |


Creleased is the authors' released DM-AHR checkpoint run through our pipeline at a 25-step respaced schedule. It was added after the protocol was frozen, so it is exploratory and is not part of the pre-registered planned family; its comparisons are reported separately in `artifacts/logs/statistical_comparisons_exploratory.csv`.


### Per-class test metrics (full test split)

| arm | class | support | f1_mean | f1_sd | recall_mean | specificity_mean | auc_mean |
|---|---|---|---|---|---|---|---|
| A | akiec | 49 | 0.625 | 0.034 | 0.694 | 0.982 | 0.956 |
| A | bcc | 77 | 0.750 | 0.035 | 0.706 | 0.991 | 0.975 |
| A | bkl | 165 | 0.676 | 0.036 | 0.644 | 0.967 | 0.955 |
| A | df | 17 | 0.733 | 0.078 | 0.706 | 0.998 | 0.987 |
| A | mel | 167 | 0.634 | 0.019 | 0.615 | 0.959 | 0.939 |
| A | nv | 1006 | 0.943 | 0.006 | 0.956 | 0.855 | 0.981 |
| A | vasc | 21 | 0.889 | 0.015 | 0.825 | 1.000 | 0.991 |
| B | akiec | 49 | 0.589 | 0.056 | 0.639 | 0.982 | 0.947 |
| B | bcc | 77 | 0.724 | 0.021 | 0.740 | 0.984 | 0.979 |
| B | bkl | 165 | 0.628 | 0.009 | 0.590 | 0.965 | 0.947 |
| B | df | 17 | 0.591 | 0.066 | 0.529 | 0.997 | 0.978 |
| B | mel | 167 | 0.561 | 0.039 | 0.529 | 0.957 | 0.935 |
| B | nv | 1006 | 0.942 | 0.002 | 0.958 | 0.844 | 0.980 |
| B | vasc | 21 | 0.853 | 0.014 | 0.825 | 0.998 | 0.991 |
| C | akiec | 49 | 0.286 | 0.019 | 0.408 | 0.952 | 0.911 |
| C | bcc | 77 | 0.315 | 0.043 | 0.290 | 0.971 | 0.881 |
| C | bkl | 165 | 0.356 | 0.024 | 0.461 | 0.863 | 0.791 |
| C | df | 17 | 0.140 | 0.073 | 0.137 | 0.990 | 0.863 |
| C | mel | 167 | 0.400 | 0.024 | 0.373 | 0.939 | 0.820 |
| C | nv | 1006 | 0.871 | 0.018 | 0.827 | 0.855 | 0.922 |
| C | vasc | 21 | 0.615 | 0.090 | 0.667 | 0.993 | 0.955 |
| Astar | akiec | 49 | 0.556 | 0.042 | 0.673 | 0.974 | 0.961 |
| Astar | bcc | 77 | 0.747 | 0.051 | 0.719 | 0.989 | 0.982 |
| Astar | bkl | 165 | 0.677 | 0.015 | 0.630 | 0.971 | 0.949 |
| Astar | df | 17 | 0.719 | 0.046 | 0.627 | 0.999 | 0.985 |
| Astar | mel | 167 | 0.613 | 0.011 | 0.611 | 0.952 | 0.925 |
| Astar | nv | 1006 | 0.943 | 0.002 | 0.949 | 0.872 | 0.982 |
| Astar | vasc | 21 | 0.868 | 0.024 | 0.873 | 0.998 | 0.999 |
| Creleased | akiec | 49 | 0.562 | 0.048 | 0.633 | 0.979 | 0.954 |
| Creleased | bcc | 77 | 0.726 | 0.012 | 0.736 | 0.984 | 0.976 |
| Creleased | bkl | 165 | 0.692 | 0.021 | 0.679 | 0.965 | 0.946 |
| Creleased | df | 17 | 0.620 | 0.104 | 0.471 | 1.000 | 0.980 |
| Creleased | mel | 167 | 0.609 | 0.028 | 0.617 | 0.949 | 0.936 |
| Creleased | nv | 1006 | 0.942 | 0.004 | 0.941 | 0.884 | 0.980 |
| Creleased | vasc | 21 | 0.840 | 0.053 | 0.794 | 0.999 | 0.994 |


Undefined AUC values are reported as NaN with a reason in the per-run metric JSON files; `df` and `vasc` have small test supports.

## Paired image quality (held-out synthetic pairs)

- Pairs scored: 186 (Arm C scored: 186)

| B_psnr | B_ssim | C_psnr | C_ssim | Creleased_psnr | Creleased_ssim | hairy_input_psnr | hairy_input_ssim |
|---|---|---|---|---|---|---|---|
| 35.732 | 0.918 | 14.292 | 0.074 | 27.076 | 0.703 | 22.127 | 0.520 |


PSNR/SSIM are computed only against genuine hairless references from synthetic pairs. Naturally hairy images are never scored against a guessed reference.


**Arm C image quality is poor because of a defect in our training script, not because of undertraining.** An earlier version of this report said the checkpoint was undertrained and the adapter sound; that diagnosis is retracted. The training script fed the network `t/T` as the noise level while the sampler feeds `sqrt(alphas_cumprod_prev)`, and the two run in opposite directions. On 50 validation pairs the released checkpoint prefers the sampler's encoding at every timestep tested (2.9x to 20.5x lower error) while our checkpoint prefers `t/T` (about 10x lower), so more iterations of the old script could not have fixed it. Arm C as reported therefore measures a defective training run and says nothing about DM-AHR as published.


The script is fixed (`tests/test_noise_level_convention.py`). Retrained on the same 654 training pairs, the model uses its conditioning image about as strongly as the released model at low and middle noise (zero-condition error 3.9x the kept-condition error at t=200, the same as released) and denoises low noise better. It still fails full 2,000-step sampling after 20,000 iterations: PSNR 14.96 dB against a 21.30 dB hairy-input floor on 24 validation pairs, and one pair ranges from 6.5 to 20.0 dB across 16 noise draws where the released model gives 25.2 to 27.4 dB. A continuation was stopped at 56,300 of a planned 100,000 iterations on cost; at 50,000 iterations the one-step error at t=1500 was 0.0086 against the released model's 0.0048, and sampling was not re-measured. The retrain was scored on validation pairs only. Our own diffusion arm is therefore reported as a failed replication, not as evidence about DM-AHR.


**Real-hair check.** Every image-quality number above is measured on synthetic hair overlays. A separate check on 40 hairy training images with real hair (test set not used; `artifacts/diagnosis/real_hair_check.json`, `reports/hair_check_prereg.md`) found that the released DM-AHR model leaves real hair in place: it changes hair pixels about as much as other pixels (inside/outside change ratio 1.13, against 21.7 for DullRazor) and the strands are visibly intact in the contact sheet. Our DullRazor arm does remove hair, but in all 8 images reviewed it replaced lesion and skin texture with blocky inpainting patches, and in one it erased the whole lesion (a dark lesion and marker ink were read as hair); the likely cause is the frozen detector and inpainting settings. Neither arm therefore removed real hair cleanly. One pre-registered control failed (the detector-coverage measure, which is void at 128 px); the conclusion rests on the change ratio, whose control passed. The 40-image sample is biased toward heavy hair.

## Explainability

| arm | seed | tier | n | gradcam_overlap20 | rollout_overlap20 | gradcam_pointing | rollout_pointing | chance_uniform | chance_gaussian |
|---|---|---|---|---|---|---|---|---|---|
| A | 123 | X1 | 1502 | 0.439 | 0.514 | 0.565 | 0.754 | 0.200 | 0.644 |
| A | 2026 | X1 | 1502 | 0.426 | 0.549 | 0.535 | 0.870 | 0.200 | 0.644 |
| A | 42 | X1 | 1502 | 0.223 | 0.601 | 0.303 | 0.901 | 0.200 | 0.644 |
| Astar | 123 | X1 | 1502 | 0.477 | 0.540 | 0.637 | 0.784 | 0.200 | 0.644 |
| Astar | 2026 | X1 | 1502 | 0.357 | 0.607 | 0.446 | 0.926 | 0.200 | 0.644 |
| Astar | 42 | X1 | 1502 | 0.378 | 0.585 | 0.524 | 0.864 | 0.200 | 0.644 |
| B | 123 | X1 | 1502 | 0.411 | 0.547 | 0.463 | 0.851 | 0.200 | 0.644 |
| B | 2026 | X1 | 1502 | 0.363 | 0.607 | 0.499 | 0.906 | 0.200 | 0.644 |
| B | 42 | X1 | 1502 | 0.390 | 0.605 | 0.519 | 0.871 | 0.200 | 0.644 |
| C | 123 | X1 | 1502 | 0.381 | 0.322 | 0.572 | 0.212 | 0.200 | 0.644 |
| C | 2026 | X1 | 1502 | 0.406 | 0.342 | 0.608 | 0.260 | 0.200 | 0.644 |
| C | 42 | X1 | 1502 | 0.448 | 0.347 | 0.670 | 0.255 | 0.200 | 0.644 |
| Creleased | 123 | X1 | 1502 | 0.411 | 0.547 | 0.575 | 0.838 | 0.200 | 0.644 |
| Creleased | 2026 | X1 | 1502 | 0.352 | 0.564 | 0.467 | 0.917 | 0.200 | 0.644 |
| Creleased | 42 | X1 | 1502 | 0.382 | 0.592 | 0.531 | 0.878 | 0.200 | 0.644 |


Localization measures spatial alignment with released lesion masks, not clinically correct reasoning. The X1 masks are the HAM10000 segmentations released on Harvard Dataverse (doi:10.7910/DVN/DBW86T). Their description states that a single dermatologist (Tschandl P) created binary masks for all 10,015 images, initialised with a segmentation network and then verified, corrected or replaced by hand in FIJI (Tschandl et al., Nature Medicine 2020). They are therefore single-rater, network-initialised and dermatologist-corrected. They are not multi-rater consensus masks, and no inter-rater agreement is available, so overlap is reported against one dermatologist's tracing.

## Statistical comparisons

| endpoint | subset | family | comparison | arm_a_mean | arm_b_mean | difference | ci_low | ci_high | p_raw | p_holm | effect_size_dz | n_groups | wilcoxon_p_raw (per-group accuracy) | wilcoxon_p_holm (per-group accuracy) | wilcoxon_n_groups |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paired_psnr | held_out_pairs | planned | B-hairy input | 35.732 | 22.127 | 13.604 | 13.261 | 13.943 | 0.000 | 0.000 | 5.759 | 186 | 0.000 | 0.000 | 186 |
| paired_psnr | held_out_pairs | planned | C-B | 14.292 | 35.732 | -21.441 | -21.768 | -21.106 | 0.000 | 0.000 | -9.391 | 186 | 0.000 | 0.000 | 186 |
| paired_psnr | held_out_pairs | planned | C-hairy input | 14.292 | 22.127 | -7.838 | -8.088 | -7.592 | 0.000 | 0.000 | -4.584 | 186 | 0.000 | 0.000 | 186 |
| paired_ssim | held_out_pairs | planned | B-hairy input | 0.918 | 0.520 | 0.398 | 0.385 | 0.410 | 0.000 | 0.000 | 4.666 | 186 | 0.000 | 0.000 | 186 |
| paired_ssim | held_out_pairs | planned | C-B | 0.074 | 0.918 | -0.843 | -0.847 | -0.840 | 0.000 | 0.000 | -35.655 | 186 | 0.000 | 0.000 | 186 |
| paired_ssim | held_out_pairs | planned | C-hairy input | 0.074 | 0.520 | -0.446 | -0.459 | -0.433 | 0.000 | 0.000 | -4.851 | 186 | 0.000 | 0.000 | 186 |
| test_macro_f1 | full | secondary_control | A-A* | 0.750 | 0.732 | 0.018 | -0.008 | 0.045 | 0.167 | 0.167 | 0.048 | 1437 | 0.074 | 0.074 | 1437 |
| test_macro_f1 | full | planned | A-B | 0.750 | 0.698 | 0.052 | 0.019 | 0.087 | 0.001 | 0.001 | 0.056 | 1437 | 0.011 | 0.011 | 1437 |
| test_macro_f1 | full | planned | A-C | 0.750 | 0.426 | 0.324 | 0.283 | 0.366 | 0.000 | 0.000 | 0.469 | 1437 | 0.000 | 0.000 | 1437 |
| test_macro_f1 | full | planned | B-C | 0.698 | 0.426 | 0.272 | 0.225 | 0.320 | 0.000 | 0.000 | 0.430 | 1437 | 0.000 | 0.000 | 1437 |
| test_macro_f1 | full | secondary_control | C-A* | 0.426 | 0.732 | -0.306 | -0.348 | -0.265 | 0.000 | 0.000 | -0.456 | 1437 | 0.000 | 0.000 | 1437 |
| test_macro_f1 | hair_present | secondary_control | A-A* | 0.749 | 0.735 | 0.014 | -0.014 | 0.041 | 0.322 | 0.322 | 0.049 | 1251 | 0.080 | 0.080 | 1251 |
| test_macro_f1 | hair_present | planned | A-B | 0.749 | 0.691 | 0.058 | 0.022 | 0.095 | 0.001 | 0.001 | 0.066 | 1251 | 0.006 | 0.006 | 1251 |
| test_macro_f1 | hair_present | planned | A-C | 0.749 | 0.411 | 0.337 | 0.291 | 0.383 | 0.000 | 0.000 | 0.505 | 1251 | 0.000 | 0.000 | 1251 |
| test_macro_f1 | hair_present | planned | B-C | 0.691 | 0.411 | 0.279 | 0.224 | 0.333 | 0.000 | 0.000 | 0.456 | 1251 | 0.000 | 0.000 | 1251 |
| test_macro_f1 | hair_present | secondary_control | C-A* | 0.411 | 0.735 | -0.324 | -0.370 | -0.279 | 0.000 | 0.000 | -0.490 | 1251 | 0.000 | 0.000 | 1251 |
| x1_gradcam_overlap20 | test_x1 | planned | A-B | 0.364 | 0.391 | -0.027 | -0.039 | -0.017 | 0.000 | 0.000 | NaN (undefined) | 1437 | 0.000 | 0.000 | 1437 |
| x1_gradcam_overlap20 | test_x1 | planned | A-C | 0.364 | 0.417 | -0.053 | -0.065 | -0.040 | 0.000 | 0.000 | NaN (undefined) | 1437 | 0.000 | 0.000 | 1437 |
| x1_gradcam_overlap20 | test_x1 | planned | B-C | 0.391 | 0.417 | -0.025 | -0.037 | -0.013 | 0.000 | 0.000 | NaN (undefined) | 1437 | 0.000 | 0.000 | 1437 |
| x1_rollout_overlap20 | test_x1 | planned | A-B | 0.558 | 0.589 | -0.032 | -0.034 | -0.029 | 0.000 | 0.000 | NaN (undefined) | 1437 | 0.000 | 0.000 | 1437 |
| x1_rollout_overlap20 | test_x1 | planned | A-C | 0.558 | 0.341 | 0.217 | 0.208 | 0.226 | 0.000 | 0.000 | NaN (undefined) | 1437 | 0.000 | 0.000 | 1437 |
| x1_rollout_overlap20 | test_x1 | planned | B-C | 0.589 | 0.341 | 0.249 | 0.239 | 0.258 | 0.000 | 0.000 | NaN (undefined) | 1437 | 0.000 | 0.000 | 1437 |


**Bootstrap.** Paired lesion-group bootstrap, 10,000 replicates, the same resample shared across arms and seeds, percentile 95% CI, two-sided. A difference whose CI spans zero is reported as inconclusive, not as no difference.

**Wilcoxon.** Signed-rank test on one value per lesion group. For the `test_macro_f1` rows the per-group value is that group's mean per-image correctness, so the Wilcoxon test asks whether the two arms differ in per-lesion accuracy; it is not a test of macro-F1 itself. For the image-quality and explainability rows the per-group value is the metric named in the endpoint column.

**Holm families.** The `family` column shows the correction family. The three planned comparisons (A-B, A-C, B-C) form one family per endpoint and subset. A-A* is a secondary resolution control and is corrected in its own family, so it cannot inflate the planned family's correction.

**Subsets.** `full` is the whole test split (1,502 images, 1,437 lesion groups). `hair_present` is the subset where hair removal actually operates (1,316 images, 1,251 groups). `held_out_pairs` is the 186 synthetic pairs. `test_x1` is the explainability cohort.


### Exploratory: released-checkpoint arm

These rows are **not** part of the pre-registered family. Creleased uses the authors' released DM-AHR checkpoint at a 25-step respaced schedule, a deviation from the paper's 2,000 steps, and its classifier was trained on our budget. Holm correction is applied within this exploratory family only.

| endpoint | subset | comparison | arm_a_mean | arm_b_mean | difference | ci_low | ci_high | p_raw | p_holm | effect_size_dz | n_groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paired_psnr | held_out_pairs | Creleased-B | 27.076 | 35.732 | -8.657 | -9.063 | -8.254 | 0.000 | 0.000 | -3.073 | 186 |
| paired_psnr | held_out_pairs | Creleased-hairy input | 27.076 | 22.127 | 4.946 | 4.503 | 5.388 | 0.000 | 0.000 | 1.639 | 186 |
| paired_ssim | held_out_pairs | Creleased-B | 0.703 | 0.918 | -0.215 | -0.228 | -0.204 | 0.000 | 0.000 | -2.558 | 186 |
| paired_ssim | held_out_pairs | Creleased-hairy input | 0.703 | 0.520 | 0.182 | 0.166 | 0.199 | 0.000 | 0.000 | 1.595 | 186 |
| test_macro_f1 | full | A-Creleased | 0.750 | 0.713 | 0.038 | 0.005 | 0.071 | 0.024 | 0.048 | 0.053 | 1437 |
| test_macro_f1 | full | B-Creleased | 0.698 | 0.713 | -0.015 | -0.049 | 0.019 | 0.400 | 0.400 | -0.011 | 1437 |
| test_macro_f1 | hair_present | A-Creleased | 0.749 | 0.712 | 0.037 | 0.002 | 0.073 | 0.038 | 0.077 | 0.060 | 1251 |
| test_macro_f1 | hair_present | B-Creleased | 0.691 | 0.712 | -0.021 | -0.057 | 0.016 | 0.261 | 0.261 | -0.016 | 1251 |


A significant but negligible effect (|Cohen's dz| < 0.2) is flagged in the CSV's `negligible` column and is not described as a meaningful difference.

## Compute environment

- All GPU work ran on **Modal**, not on Google Colab: NVIDIA A100 80GB PCIe, Python 3.11.12, torch 2.5.1+cu124, CUDA 12.4.
- The proposal names Google Colab as the implementation environment. That wording is superseded: the project instructions require all GPU jobs to run on Modal, and this study's Arm C sampling budget exceeds free-tier Colab limits. Colab was used only to retrieve artifacts. See decision log D14.
- Per-run GPU type and wall-clock are in `artifacts/logs/run_registry.jsonl`.

## How the results bear on the three hypotheses

The proposal states three hypotheses. Each is answered below only from the tables above; where the evidence does not settle the question it is called inconclusive rather than resolved either way.
**H1 (image quality): diffusion removal preserves lesion boundaries and texture better than classical inpainting.** On 186 held-out synthetic pairs, B higher than C (C - B = -21.441, 95% CI -21.768 to -21.106) on PSNR, and B higher than C (C - B = -0.843, 95% CI -0.847 to -0.840) on SSIM. H1 is not supported by these data. The caveat is material: Arm C's output scores below the unprocessed hairy input, and its SSIM is near-constant across all pairs (0.130 +/- 0.010, with 185 of 186 below 0.15). The cause is a defect in our training script, not undertraining: it gave the network a noise-level signal that runs opposite to the one the sampler uses (see the Arm C paragraph and `reports/oneshot_diagnostic_report.md`). Arm C therefore says nothing about DM-AHR as published. The released checkpoint, scored through the same adapter on synthetic hair (Creleased, exploratory), clears the hairy-input floor but sits below classical inpainting, so H1 is not supported for the released weights either; that arm's sampling routine is not reproducible from this repository and it very probably saw our test images (see Limitations).

**H2 (classification): a ViT trained on diffusion-cleaned images outperforms the original and classically cleaned arms.** A vs C: A higher than C (A - C = +0.324, 95% CI +0.283 to +0.366). B vs C: B higher than C (B - C = +0.272, 95% CI +0.225 to +0.320). The direction is opposite to the hypothesis and the gap is large, but Arm C is the output of a defective training script (inverted noise-level signal; see the Arm C paragraph). This is a true statement about the model that was run, not a refutation of the published method. The exploratory released-checkpoint arm reproduces synthetic hair removal well but leaves real hair in place (see the real-hair check), so it is not a test of hair removal either.

**A vs B (original against classical removal): A higher than B (A - B = +0.052, 95% CI +0.019 to +0.087).** This must not be read as evidence that hair removal hurts classification: the real-hair check shows that our frozen DullRazor settings damage lesions, so the gap is confounded with image damage and was not separated from it.

**H3 (explainability): attention and Grad-CAM focus more on lesions after diffusion removal.** H3 is **not supported, and the measurement is uninformative as specified**. Grad-CAM lesion overlap, A vs C: C higher than A (A - C = -0.053, 95% CI -0.065 to -0.040). Attention roll-out lesion overlap, A vs C: A higher than C (A - C = +0.217, 95% CI +0.208 to +0.226). The two methods disagree in direction, but the decisive problem is the baseline: a centred-Gaussian prior scores 0.644 on overlap and 0.953 on the pointing game, exceeding **every arm on both measures**. No arm beats a trivial attend-to-the-centre prior, so the saliency measures do not demonstrate lesion localisation at all, in any arm. This is a property of the measurement, not a finding about diffusion hair removal, and neither measure establishes clinically correct reasoning. A stronger design would use an empirical random-map baseline per image and a saliency method that can beat the centre prior before H3 is testable.

**Resolution control.** Arm A* isolates the 128 px round-trip that Arm C introduces: A-A* macro-F1 difference inconclusive (95% CI for A - A*: -0.008 to +0.045). Because that comparison is inconclusive, the resolution change alone does not explain Arm C's large deficit.

## Limitations and failure cases

- DERMAHAIR could not be downloaded (no Kaggle credentials), so the synthetic-hair fallback is used. Held-out test pairs number only 186, so paired CIs are wide.

- The screening threshold is NOT VERIFIED against real hair labels. This is a skipped step, not a property of the data: the proposal specifies the manually annotated hair masks of Lama et al. (ChimeraNet) as an independent reference for screening quality, and those masks were not obtained. The threshold was tuned on synthetic pairs and frozen before test scoring, which protects the test set but does not validate the detector on real hair.

- DM-AHR runs at 128 px, so Arm C is upsampled to 224 px for the ViT. Arm A* controls for that resolution change but cannot separate it from diffusion artefacts.

- Arm C's diffusion model was trained for 20,000 iterations with a defective script (inverted noise-level signal, found after the primary results; see `reports/oneshot_diagnostic_report.md`). The upstream config's `n_iter` is 660,000 (`third_party/dmahr/config/akiec.json`, commit `0aee1bb`) and the released checkpoint is named `I725000_E1083`. The project kit separately specifies 20,000 iterations as the reference setting; that conflict is recorded in decision log D1. The fixed script did not reach usable sampling at the budgets tried, so Arm C is a failed replication and not DM-AHR as published.

- Creleased uses the authors' released weights. The sampling routine that produced its images is not in this repository and cannot be regenerated from committed code (decision log D26). The DM-AHR paper states that the model was trained on HAM10000 images other than a 700-image test set, so it very probably saw our validation and test images; it used no labels, so this mainly affects image-fidelity numbers.

- The released model leaves real hair in place and our DullRazor arm damages lesions on real hair (see the real-hair check). The original-against-DullRazor difference is confounded with that damage and is not evidence about hair removal in general. Re-tuning Arm B on training and validation images would address it; that was not done.

- `df` and `vasc` have very small test supports, so their per-class numbers are noisy.

- The released DM-AHR weights were trained on HAM10000-derived images (per the paper) and are used for exploration only.

- Augmentation differs from the proposal's wording: rotation is ±15° and uses a resized crop, where the proposal asks for ±90° rotation and explicit zoom/shift. The recipe is identical across arms, so the comparison stays internally valid, but it is not the stated recipe. See decision log D15.

- Class imbalance is handled by inverse-frequency loss weights only. The proposal also mentions class-balanced sampling; combining both is not recommended, and the dev kit makes weighted loss primary. This is recorded in decision log D16 rather than left implicit.

- Arm C was sampled at 1,000 timesteps rather than 2,000 for budget reasons (decision log D12). Every per-image row carries its `num_timesteps` value so the deviation is auditable.

## Licence and provenance

HAM10000 is distributed under CC BY-NC 4.0. The DM-AHR repository contradicts itself: its README states CC BY-NC 4.0 while its LICENSE file is Apache-2.0. **Commercial use is unresolved** and needs a rights decision by the client. See `reports/license_audit.md`.

## Reproducibility

Arm A, seed 42 was re-run from scratch through the same entrypoint (gate G13). Recorded validation macro-F1 0.778336 vs reproduced 0.778336, absolute difference 0.000000 against a 0.02 tolerance. Separate runs on two different Modal accounts also reproduced the frozen split hash byte for byte.

Commands are in `README.md`. Frozen artifacts and their hashes are in `artifacts/logs/` and `data/manifests/`. Every table above is generated from a CSV or JSON artifact listed in `artifact_inventory.csv`.
