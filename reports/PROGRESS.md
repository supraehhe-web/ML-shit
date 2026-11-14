# Progress ledger

Updated after every phase. A fresh session should be able to resume from this
file alone.

## Status

| Phase | Gate | Status | Evidence |
|---|---|---|---|
| 0 Environment | G1 | PASS | `artifacts/logs/environment_report.json` (A100-80GB, torch 2.5.1+cu124) |
| 1 Data audit | G2 | PASS | 11,720 collection images, 10,015 C0, 1,705 quarantined, class counts match expected exactly |
| 2 Splits | G3 | PASS | 7,011/1,502/1,502, zero group/lesion/hash overlap, hash frozen |
| 3 Hair screening + Arm B | G6 | PASS | 8,993 hair-present, 10,015 B and A* outputs written |
| 4 DM-AHR smoke | G5 | PASS | batch 16 returns 16 images, geometry preserved, reload identical |
| 5 Diffusion training | G4 | PASS | 20,000 iters, train split only, val split only for selection |
| 6 Arm C sampling | - | PASS | 8,993/8,993 hair-present images at 1,000 steps (labelled ablation) |
| 7 ViT training | G7 | PASS | 12 of 12 runs (A, B, A*, C x 3 seeds) |
| 8 Classification eval | G8 | PASS | 12 of 12 checkpoints evaluated on the locked test split |
| 9 Image quality | G9 | PASS | 186 held-out pairs, all three conditions scored |
| 10 Explainability | G10 | PASS | 9 runs x 1,502 test images, X1 tier |
| 11 Statistics | G11 | PASS | 10 comparisons, Holm-corrected, 10,000 replicates |
| 12 Report + handover | G12-G14 | PASS | report, inventory, acceptance 14 PASS / 0 FAIL / 0 BLOCKED |

## Final acceptance state

`artifacts/logs/acceptance_report.json`: **14 PASS, 0 FAIL, 0 BLOCKED**.

**G13 closed.** Arm A, seed 42 was re-run from scratch into a fresh directory
through the same entrypoint. Recorded and reproduced validation macro-F1 are
identical to 16 significant figures (0.7783361, absolute difference 0.0 against
a 0.02 tolerance). The log is `artifacts/logs/reproduction_log.json`.

One functional item remains open, and it is a scope blocker rather than a defect:
**F04 (DERMAHAIR provenance)**. No Kaggle credentials were configured, so the
kit's required synthetic-hair fallback was used. It is recorded as blocked with
evidence; the DERMAHAIR route was never silently substituted.

## Arm C is a labelled ablation, not the paper's method

Arm C ran at **1,000 sampling steps instead of the reference 2,000**, with bf16
autocast, because the account balance could not cover the full-step run. This is
recorded in decision log D12, in `configs/dmahr.json`, in every per-image CSV
(`num_timesteps` column) and in the report. The A-C and B-C comparisons answer
"does a half-budget DM-AHR help?", not "does DM-AHR help?".

## Headline results (real, from artifacts)

Test macro-F1, mean ± SD over 3 seeds, 1,502 test images:

| Arm | macro-F1 | accuracy | macro AUC |
|---|---|---|---|
| A original | 0.750 ± 0.005 | 0.858 | 0.969 |
| A* 128 px | 0.732 ± 0.008 | 0.851 | 0.969 |
| B DullRazor | 0.698 ± 0.009 | 0.842 | 0.965 |
| C DM-AHR (1,000-step ablation) | 0.426 | — | — |

- **A vs B**: +0.052 macro-F1, 95% CI [0.019, 0.087], p_holm = 0.0012.
  Do not read this as evidence that hair removal hurts: our DullRazor settings
  damage lesions on real hair (decision log D31), so the gap is confounded.
- **A vs A***: +0.018, CI [-0.008, 0.045], p_holm = 0.167. The 128 px
  resolution change alone is not a significant effect.
- **A vs C**: +0.324, CI [0.283, 0.366], p_holm < 0.0001. Arm C is far worse
  because its images come from a defective training run (inverted noise-level
  signal), not because DM-AHR cannot help.
- **B vs C**: +0.272, CI [0.225, 0.320], p_holm < 0.0001.
- **Image quality** (186 held-out pairs): B improves PSNR 22.13 -> 35.73 and
  SSIM 0.520 -> 0.918 versus the hairy input. Arm C scores 16.31 / 0.130,
  *worse than doing nothing*, because the 20,000-iteration checkpoint was trained
  with an inverted noise-level signal (diagnosis of "undertrained" retracted;
  see `reports/oneshot_diagnostic_report.md`).
- **Explainability** (X1, 1,502 images): Grad-CAM lesion-overlap@20% 0.38,
  attention rollout 0.55-0.60, against a uniform chance baseline of 0.20 and a
  centred-Gaussian baseline of 0.64. Rollout does not beat the Gaussian
  baseline, so its apparent alignment is largely a centre bias.

## Measured compute

See `reports/compute_plan.md`. Headline measured units:

- DM-AHR sampling: 4.35 s/image at batch 32 on A100-80GB.
- DM-AHR training: 20,000 iterations in 706 s.
- ViT-B/16: ~17 s/epoch, full early-stopped run ~420 s.

## Failed attempts and fixes (do not repeat)

1. **Serial file I/O over the Modal Volume.** Reading 11,720 images one at a
   time took >15 min with ~12 s of CPU time. Fixed by 32-way parallel reads in
   `data/manifest.py` and `data/masks.py`; the audit now runs in ~30 s.
2. **dHash alone as a near-duplicate test.** 70,062 candidate pairs from 10,015
   images. Fixed by pixel confirmation (MAD <= 4/255 and r >= 0.985), leaving
   21 real pairs.
3. **Size-first split assignment.** Overall 70/15/15 but `vasc` split 114/6/22.
   Fixed with class-major apportionment.
4. **`modal.Image.add_local_*` before `.env()`.** Modal rejects build steps
   after local-file mounts; all `add_local_*` calls must come last.
5. **Upstream config is JSON5.** `json.load` fails on comments and trailing
   commas; a tolerant reader is required.
6. **Upstream checkpoint loading.** `GaussianDiffusion.__init__` omits
   `set_new_noise_schedule`, so diffusion buffers are absent and strict
   `load_state_dict` fails. Call the schedule setter first.
7. **Upstream sampler collapses batches.** `continous=False` returns only the
   last image; take the final B rows of the `continous=True` trajectory.
8. **PIL atomic write.** Saving to `name.png.tmp` needs an explicit
   `format="PNG"`.
9. **Volume commits.** `vol.commit()` only at job end loses a long run on
   preemption; `modal_app._run` now commits every 60 s and streams logs.
10. **Over-broad app cleanup.** Stopping every `ephemeral` app also killed an
    unrelated ViT sweep. Stop apps by explicit id only.

## Blockers (documented, with evidence)

- **DERMAHAIR**: Kaggle credentials absent, so F04 is blocked and the synthetic
  fallback is used. Consequence: 186 held-out test pairs, wide CIs.
- **Hair annotations**: none independent, so the screening threshold is
  NOT VERIFIED against real hair labels.
- **Licences**: HAM10000 is CC BY-NC 4.0; DM-AHR's README (CC BY-NC 4.0) and
  LICENSE (Apache-2.0) contradict each other. Commercial use unresolved.

## Post-delivery alignment pass

The client supplied the proposal (`DM-AHR-ViT_Skin_Lesion_Classification_(2).docx`)
and asked for the results to be checked against it, plus presentation material for
a supervisor discussion.

Found and fixed two gaps where a required deliverable was silently absent:

- **Wilcoxon tests were never computed.** `wilcoxon_group_paired` existed and was
  unit-tested, but `scripts/09_stats.py` hard-coded `wilcoxon_p_raw` to NaN. All
  endpoints now carry group-aggregated Wilcoxon p-values with Holm correction
  (decision log D17).
- **Explainability endpoints never reached the statistics.** The XAI CSVs had no
  `group_id`, so `xai_rows()` returned nothing. Added at generation and backfilled
  from `mask_manifest_x1.csv`. The stats table grew from 14 rows / 3 endpoints to
  20 rows / 5 endpoints.

Also added: a "Compute environment" section (Modal, real versions) and a
"How the results bear on the three hypotheses" section, both generated from
artifacts; decision log entries D14-D16 for the Colab, augmentation and
class-balancing divergences; `docs/CODE_GUIDE.md`; and two notebooks.

Verified: 28 CPU tests pass, notebook 1 executes with zero errors, 14/14 gates
still pass (G11 now rows=20, endpoints=5).

## Next step

Nothing outstanding. A faithful Arm C requires the full training and sampling
budget (decision log D1, D12); everything else is reproducible from
`notebooks/02_pipeline_reproduction.ipynb`.

## 2026-11-12 - Released-checkpoint arm evaluated; reconstruction proven

The sandbox was wiped mid-session, so this phase started by re-establishing the
new Modal account rather than from the old one.

- **Uploads were already complete** on `dmahr-vit-vol` in the new account
  (`motog5g987`): manifests, `processed/Creleased`, `synth`, ViT checkpoints,
  predictions and logs. `data/raw/isic_212`, `processed/B` and `processed/Astar`
  were missing and have been copied across from the old account.
- **Reconstruction verified exactly.** The reconstructed data package squashed
  images to a square at eval. Sweeping resize sizes against the recorded A/42
  run found the original pipeline: resize 255 (`int(224 * 1.14)`) then centre
  crop 224. With it, A/42 reproduces 1,502/1,502 predictions, a maximum
  probability difference of 0.0 and identical macro-F1 (0.7510451247030895).
  `scripts/21_verify_reconstruction.py` automates the check (decision D23).
- **Creleased evaluated** on all three seeds: macro-F1 0.708 / 0.688 / 0.743,
  mean 0.713 ± 0.028 full and 0.712 ± 0.029 hair-present. XAI ran for all three
  seeds. The first evaluation used the wrong transform and was discarded and
  re-run.
- **Image quality** now scores all four conditions on the 186 held-out pairs:
  B 35.732 / 0.918, Creleased 27.076 / 0.703, hairy input 22.127 / 0.520,
  C 14.292 / 0.074. Creleased clears the >=hairy+3 dB rule by 4.95 dB but sits
  8.66 dB behind classical inpainting.
- **Exploratory statistics** (`statistical_comparisons_exploratory.csv`):
  A-Creleased +0.038 (95% CI +0.005 to +0.071, Holm p=0.048, dz 0.05,
  negligible); B-Creleased -0.015 (CI -0.049 to +0.019, Holm p=0.400). The
  released checkpoint does not beat classical removal. The pre-registered
  `09_stats.py` arm list is untouched.
- **Gates.** Acceptance: 13 pass, G4 blocked (no leakage-controlled diffusion
  training run at a realistic budget - unchanged, decision D1/D12). Deliverable
  gates: 7 of 8 pass; V7 fails only because its PDF evidence lived in a local
  `/tmp/verify` scratch directory that the sandbox wipe removed, and no
  LibreOffice/pymupdf is installed to re-render. The three .docx deliverables
  themselves are tracked in git and copied to the volume.

## Next step

Nothing outstanding on the research question. Remaining known gap: our own
Arm C is a failed replication (inverted noise-level signal fixed; the fixed run
still fails sampling at the budgets tried), and V7's PDF rendering evidence needs
a machine with LibreOffice.

## 2026-11-12 - Delivery package complete

- **Deliverable gates: 8 of 8 pass.** V7 now checks PDFs stored beside the
  .docx on the volume instead of a `/tmp` scratch directory, so the evidence
  survives a sandbox wipe. The three reports were rendered with LibreOffice:
  Preprocessing 10 pages, Results 8 pages, Code Guide 14 pages.
- **Acceptance: 13 of 14 pass, G4 blocked.** G4 is the only non-pass and it is
  a documented blocker, which the criteria allow: no leakage-controlled
  diffusion training run exists at a realistic budget (D1, D12). The released
  checkpoint arm answers the DM-AHR question that G4 would have answered.
- **Inventory refreshed** to 295 entries including the Creleased artifacts.
- **Handover package (criteria section D)** is present: source and configs,
  `requirements.lock`, README with Modal commands and `modal volume get`
  retrieval lines, manifests and frozen splits, licence audit, DullRazor and
  DM-AHR scripts, 15 ViT checkpoints plus the diffusion checkpoint, run
  registry, per-image predictions, classification tables, PSNR/SSIM pair CSV,
  Grad-CAM and rollout outputs, statistical CSVs, final report in Markdown,
  DOCX and PDF, artifact inventory with SHA-256, and the research-only
  statement.

## Delivery caveats to state to the client

1. Arm C (our diffusion run) is a failed replication: its training script gave
   the network an inverted noise-level signal. The released-checkpoint arm
   (Creleased) is the only diffusion arm with usable hair removal; it does not
   beat classical removal or original images, its sampling routine is not
   reproducible from this repository, and its training data is unverified.
2. DERMAHAIR was unavailable, so image quality uses 186 synthetic pairs and the
   hair detector is unvalidated on real hair.
3. Licensing is unresolved: HAM10000 is CC BY-NC 4.0 and the DM-AHR repository
   contradicts itself between README and LICENSE. Commercial use needs a rights
   decision.
4. Single public dataset, research prototype only, not clinically validated.

## 2026-11-13 - One-step reconstruction diagnostic

Ran `scripts/25_oneshot_diagnostic.py` on 100 validation pairs to separate
"weights never learned" from "sampler is broken". Released and our checkpoints
were both scored on one-step x0 reconstruction at t = 50, 200, 500, 1000, 1500,
1900, with a zeroed-condition ablation.

- Ours reaches 31.54 dB at t=50 against a 22.11 dB do-nothing floor, so it has
  learned to denoise and the "never learned" explanation is ruled out.
- Ours is behind the released checkpoint at every timestep, by 7.0 to 9.3 dB in
  the low-to-mid range, which reads as incomplete convergence.
- Zeroing the conditioning barely changes our output at t=200 (delta 0.0234
  versus 0.1076), so our model uses the hair mask far less than the released one.
- Conclusion: more training should improve Arm C, but the released checkpoint's
  0.713 caps the realistic ceiling below the 0.750 baseline.

Report: `reports/oneshot_diagnostic_report.md`. Data:
`artifacts/logs/oneshot_diagnostic.json`.

### Correction: the diagnostic found a trainer defect, so the answer is no

An independent review challenged the "yes" conclusion. Following it up found a
real defect, and the answer is now **no**.

- The trainer fed the U-Net `t / num_timesteps` as the noise level. The sampler
  feeds `sqrt_alphas_cumprod_prev[t+1]`. The first increases with `t`, the second
  decreases: the encodings are inverted, not merely different.
- Measured epsilon-error under both encodings: the released checkpoint is 2.9x
  to 20.5x better under the sampler encoding at every timestep; ours is about
  10x better under `t/T` at every timestep. `artifacts/logs/noise_convention_check.json`.
- Zeroing the conditioning image changes the released model's error by 3.3x to
  4.8x, ours by 0.97x to 1.8x. Ours barely uses the hair mask.
- Fixed `src/dm_ahr_vit/diffusion/train.py` to mirror upstream `p_losses`, added
  `tests/test_noise_level_convention.py` (32 tests pass), and fixed
  `PairedDataset` to re-root manifest paths so training runs on any volume.
- Verified the fixed trainer runs: overfit job completed 500 iterations, loss
  0.185 to 0.082, checkpoint saved.
- Earlier report errors corrected: the 22.11 dB floor was applied at every
  timestep when the correct per-t baseline at t=50 is 28.1 dB, and the
  "loop amplification" comparison has been withdrawn.

**Consequence: training to 660,000 on the old trainer would have reproduced the
defect. Any retraining must start from scratch with the fixed trainer.**


## 2026-11-13 - Arm C retrain stopped; reports corrected

See decision log D30. The fixed trainer was tested on validation pairs only; the 100,000-iteration continuation was stopped at 56,300. The retracted "undertrained" diagnosis was removed from the final report, Word reports, PDFs, code guide and README. Deliverable gates 8 of 8 and acceptance 14 of 14 pass on the Modal volume. The 186 test pairs were not touched.

## 2026-11-13 - Rebuilt diffusion arm (Cv2): Stages A, B, C complete

- Stage A: partial-noise sampler passes on validation pairs (+9 dB at t0 600 and 1000); the positive control missed its bar twice and is recorded as such.
- Stage B: real strands harvested from train/validation photos (1,893 usable layers), fine-tune of 20,000 iterations. The pre-registered gate was **not met** (ratio 1.95 and 1.57 against 2.0; composite rule +0.14 and +0.15 against +0.20). Dense-hair validation sheets show removal (agent-reviewed). See D33, D34.
- Stage C: Cv2 sampled once (8,993 images + 186 test pairs), three ViT seeds trained and evaluated once, statistics with the A* control. Macro-F1 0.720 ± 0.021, inconclusive against A, B and A*; PSNR 6.55 dB below DullRazor on synthetic pairs. See D35 and `reports/stageC_results.md`.
- Reports updated: final report, client summary, Results_Report and Code_Guide_Report (docx and pdf), README, CODE_GUIDE. Inventory extended with `scripts/47_inventory_append.py`.
- Open items: none required by the protocol. Optional next step (client decision): higher-resolution model or paste-back of original pixels outside the hair, and more than three seeds.

## 2026-11-13 - Stage D (full-resolution paste-back) failed its validation gates; second-opinion brief written

- Stage D (`reports/prereg_v3_pasteback.md`, D36): variant `coarse` 31.76 dB / SSIM 0.831 (fails D1, D2); variant `thin` 33.33 dB / 0.846 (passes D1, fails D2 at 0.90). Stop rule applied; no Cv2p image built; no test image or pair read.
- Pending owner decision: (1) hybrid with classical full-resolution inpainting as the fill, labelled a hybrid; or (2) 256 px diffusion training. See section 11 of the brief.
- `reports/SECOND_OPINION_BRIEF.md` written for an independent reviewer (D37). The Word/PDF reports, README and final report describe Stages A to C and do not yet mention Stage D.

## 2026-11-13 - Stage E: response to the independent review

- Pre-registered (e3d3ef6), then run: 20 extra-seed ViT evaluations (A, A*, B, Cv2 at seeds 7, 11, 17, 23, 31), pooled two-level-bootstrap analysis, composite/real-photo/condition-usage/residual-hair checks, human-review kit. Results `reports/stageE_results.md`, decision D38.
- Eight-seed macro-F1: A 0.741, A* 0.728, Cv2 0.722, B 0.693; only A over B established; minimum detectable difference 0.04 to 0.05.
- Open: human ratings of `artifacts/review_kit/` (needs a person). Real-photo hair removal remains unvalidated.

## 2026-11-13 - Client review points: verification pass, Milestone 1 package (D39)

- Verified the reviewer's claims against files; corrected premises are in `reports/milestone1_diffusion_diagnosis.md`. DERMAHAIR still blocked (no Kaggle login; unauthenticated download 404).
- New validation-only checks: `scripts/57_paper_t10_check.py` (released checkpoint at the paper's quoted T = 10: no reproduction) and `scripts/56_xai_registration_check.py` (lesion masks are not in the model's frame, differently per arm).
- Correction notices added to `reports/stageE_results.md` (section 6), `reports/final_research_report.md`, `reports/client_summary.md`. The Word/PDF reports and their generator scripts were NOT updated and still carry the superseded explanation reading: do not send them.
- Plan and timeline: `reports/client_plan_and_timeline.md`. Nothing trained, nothing re-scored on the test set.
- Open: Kaggle login or dataset copy; approval for the learning-curve run; checkpoint destination; human ratings of `artifacts/review_kit/`.
