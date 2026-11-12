# Decision log

Authority order applied: `AGENTS.md` > `docs/ACCEPTANCE_CRITERIA.md` > `docs/DEVELOPMENT_KIT.md` > the proposal. Every source disagreement is recorded below with the decision taken.

## D1 - DM-AHR reference iteration count

- Disagreement: the kit states reference settings "2,000 timesteps, 20,000 iterations, Adam lr 1e-4". The shipped upstream `config/akiec.json` has `"n_iter": 660000`, and the released checkpoint is `I725000_E1083` (725,000 iterations).
- Decision: keep 2,000 timesteps, Adam 1e-4 and 128x128 exactly as verified in the config, but train for 20,000 iterations as the kit specifies, and report this divergence.
- Reason: AGENTS.md outranks the kit; the kit's number is the requirement, the config's number is the evidence of what upstream actually shipped. 20,000 iterations is a compute-bounded deviation and is not presented as reproducing the released checkpoint's training budget.

## D2 - Upstream config is not valid JSON

- Finding: `config/akiec.json` contains `//` and `/* */` comments and trailing commas, so `json.load` rejects it.
- Decision: parse it with a small JSON5-tolerant reader that strips comments outside string literals and removes trailing commas. Parsed values are written to `artifacts/logs/dmahr_smoke.json` for audit.

## D3 - Upstream sampler and schedule bugs

- Finding: `GaussianDiffusion.super_resolution(x, continous=False)` returns `ret_img[-1]`, so a batch of N images yields only the last one. Separately, `GaussianDiffusion.__init__` never calls `set_new_noise_schedule` (the call is commented out), so the diffusion buffers do not exist until an explicit schedule call.
- Decision: call `set_new_noise_schedule` explicitly before loading weights, so strict `load_state_dict` validates parameters and buffers together; then take the final B rows of the `continous=True` trajectory to recover the whole batch.
- Verification: the smoke test confirms batch 16 returns 16 images, geometry is preserved, output is finite, and reloading the checkpoint with the same seed reproduces identical pixels.
- Reason: AGENTS.md rule 11 - document the exact failure, then work around it without substituting a different model.

## D4 - Near-duplicate merging

- Finding: dHash at Hamming distance <= 6 produced 70,062 candidate pairs among 10,015 images, far too many to be real duplicates; similar dermoscopic backgrounds collide.
- Decision: re-check every candidate on a common 96x96 grayscale grid and keep it only if mean absolute difference <= 4/255 and Pearson r >= 0.985. 21 pairs survive. Groups merge by `lesion_id` plus these pixel-confirmed matches, giving 7,470 groups for 10,015 images, with 0 groups spanning two diagnoses.
- Reason: the kit requires confirmed near-duplicates; an unconfirmed hash match must not merge lesions.

## D5 - Split construction

- Finding: a size-first greedy splitter hit 70/15/15 overall but put 80% of `vasc` in train and only 3% in val.
- Decision: apportion each class independently under a global size cap. Result: 7,011/1,502/1,502 images with every class at 0.148-0.150 of its total in each split.
- Reason: the kit requires class-stratified splits, and the rare classes have only tens of test images.

## D6 - C0-extra quarantine

- Finding: collection 212 holds 11,720 images; Dataverse metadata lists 10,015. The 1,705 remainder are quarantined as `C0-extra`.
- Decision: all confirmatory work uses C0 only, proven by `test_extras_never_enter_a_split`.

## D7 - Arm A* is required

- Finding: DM-AHR is native 128x128, below the 224px ViT input.
- Decision: include Arm A* (originals round-tripped through the same 128px bicubic down/up-sample) for all three seeds, as the kit mandates when Arm C is below 224px.

## D8 - Hair-screening threshold frozen before test scoring

- Finding: the black-hat/top-hat detector separates synthetic hairy from synthetic clean images perfectly (balanced accuracy 1.0) at coverage >= 0.01, so synthetic pairs do not constrain the threshold sharply. Real-world prevalence at that threshold is 89.8%.
- Decision: freeze 0.01, tuned on train/val sources only. No independent hair annotations exist, so this is recorded as NOT VERIFIED: no independent hair annotations. The threshold is not changed after any test-set metric was computed.
- Reason: changing a frozen parameter after inspecting test-set coverage statistics would be test-set tuning. The high prevalence is reported as a limitation and a cost driver, not hidden.

## D9 - DERMAHAIR availability

- Finding: DERMAHAIR is distributed on Kaggle and needs Kaggle API credentials, which are not configured here (`KAGGLE_USERNAME`/`KAGGLE_KEY` absent).
- Decision: use the kit's required synthetic-hair fallback, generating pairs only from hair-absent C0 images with a recorded seed, and mark the DERMAHAIR route BLOCKED: Kaggle credentials unavailable. F04 is reported blocked-with-evidence, not passed.
- Consequence: test pairs number 186 (every hair-absent C0 test image), so paired image-quality confidence intervals will be wide.

## D10 - Spend cap

- Finding: `AGENTS.md` requires `MAX_SPEND_USD` to be set by the user; no value was supplied.
- Decision: record measured GPU-hours only and set a provisional `max_spend_usd = 250` in `configs/protocol.json`. Modal prices are not guessed.

## D11 - Sharded Arm C sampling

- Finding: Arm C must process 8,993 hair-present images at ~4.35 s/image on A100-80GB, about 10.9 GPU-hours.
- Decision: shard by image across containers with per-shard manifests, skip already-written outputs on restart, and commit the volume periodically. Reduced-step sampling is not used (it would need a separately labelled ablation).

## D12 - Arm C run as a 1,000-step ablation (budget forced)

- Finding: the account held $23. Full 2,000-step sampling of 8,993 images costs
  about $21 on A100-40GB, leaving nothing for the three ViT C trainings,
  evaluations and XAI passes. Measured throughput was 4.38 s/image at 2,000
  steps and 2.18 s/image at 1,000 steps.
- Decision: run Arm C at **1,000 sampling steps with bf16 autocast**, which
  halves the cost and fits the balance. This is recorded as a labelled ablation
  in the report, the config and every per-image CSV (`num_timesteps` column).
- Reason: the kit permits reduced-step runs only as a labelled, justified
  ablation. A complete four-arm comparison at reduced steps is more informative
  than a missing arm, provided the deviation is stated plainly and never
  presented as the paper's method.
- Consequence: Arm C image quality and classification are expected to be worse
  than a full-step run, and the measured results confirm that. The A-C and B-C
  comparisons therefore answer "does a half-budget DM-AHR help?" and not "does
  DM-AHR help?".

## D13 - Arm C passthrough for hair-absent images

- Finding: the sampler only processes hair-present images, but the ViT dataset
  for Arm C reads every split image from `processed/C`. Hair-absent images were
  therefore missing and training failed with `FileNotFoundError`.
- Decision: copy the original image into `processed/C` for every hair-absent
  image, matching the protocol rule that hair-absent images are identical in
  every arm. Implemented in `scripts/05b_fill_c_passthrough.py`; all 1,022
  images filled, giving 10,015 Arm C inputs.

## D14 - Implementation environment: Modal, not Google Colab

- Disagreement: the proposal states four times that the pipeline is
  "implemented in Python with PyTorch in Google Colab with GPU acceleration",
  and section 5.2 makes reporting that environment a deliverable. `AGENTS.md`
  mandates that all GPU work runs on Modal.
- Decision: run every GPU job on Modal, and state the divergence here and in
  the report's "Compute environment" section. Record the real versions:
  Python 3.11.12, torch 2.5.1+cu124, CUDA 12.4, NVIDIA A100-80GB PCIe.
- Reason: `AGENTS.md` outranks the proposal. The proposal's own workload also
  does not fit the described environment: Arm C needs 8,993 images at about
  2.18 s each, roughly 5.4 A100-hours, which exceeds a free Colab session and
  cannot be relied on with Colab's availability-dependent GPU allocation.
- Consequence: Colab is used only to retrieve artifacts. It is not the
  execution environment and the report must not claim otherwise.

## D15 - Augmentation differs from the plus/minus 90 degree rotation proposed

- Finding: the proposal and dev kit specify rotation up to +/-90 degrees and
  zoom/shift +/-15%. `configs/vit.json` applies `RandomRotation(15)` and
  `RandomResizedCrop` (scale 0.75-1.0, ratio 0.9-1.11) with no explicit shift.
- Decision: keep the implemented recipe, which is applied identically to all
  four arms, and disclose the difference in the report's limitations.
- Reason: matched arms preserve the internal validity of the comparison, which
  is what the study measures. Re-running twelve ViT trainings to change
  augmentation would spend the remaining balance without changing the
  conclusion about the hair-removal condition. Disclosed rather than hidden.

## D16 - Class balancing uses loss weights only

- Finding: the proposal mentions "class-balanced sampling with loss weighting".
  The dev kit makes inverse-frequency loss weights primary and warns against
  combining sampling with weights unless reported as a separate ablation.
- Decision: use inverse-frequency class weights only, computed from the train
  split, identical across arms.
- Reason: the dev kit's guidance replaces the proposal's looser wording.
  Recorded here because the proposal implies both were used.

## D17 - Wilcoxon tests added; explainability endpoints now enter the table

- Finding: the proposal and dev kit require a Wilcoxon signed-rank test reported
  beside the bootstrap CI. `wilcoxon_group_paired` existed with unit tests but
  was never called; `scripts/09_stats.py` hard-coded `wilcoxon_p_raw` to NaN.
  Separately, the XAI CSVs carried no `group_id`, so `xai_rows()` returned
  nothing and the two primary explainability endpoints never reached the table.
- Decision: compute group-aggregated Wilcoxon signed-rank tests for every
  endpoint, Holm-corrected within the same family as the bootstrap, and write
  `wilcoxon_p_raw`, `wilcoxon_p_holm`, `wilcoxon_statistic` and
  `wilcoxon_n_groups`. Values are collapsed to one per lesion group before
  testing. Add `group_id` to the `scripts/08_xai.py` output, and backfill it in
  `09_stats.py` from `mask_manifest_x1.csv` for XAI CSVs written before the fix.
- Verification: the table grew from 14 to 20 rows; all 20 carry a non-NaN
  Wilcoxon p-value and the two XAI endpoints are present.
- Reason: both were required deliverables that were silently absent. Computed
  from existing artifacts on CPU, so no GPU spend is involved.

## D18 - External review corrections to the reports

- Finding: two independent reviews of the results documents surfaced four
  claims that the artifacts do not support. Each was checked against the
  source before being changed.
- Corrections:
  1. **Mask provenance.** The report said the source "does not state they were
     expert-traced". The Harvard Dataverse record for doi:10.7910/DVN/DBW86T
     states that a single dermatologist (Tschandl P) created binary masks for
     all 10,015 images, network-initialised and then verified, corrected or
     replaced by hand in FIJI (Tschandl et al., Nature Medicine 2020). The
     report now describes them as single-rater and dermatologist-corrected,
     and notes that no inter-rater agreement exists.
  2. **Duplicate merge was credited as leakage control.** All 21
     pixel-confirmed pairs already shared a `lesion_id`, so the merge changed
     no grouping: 7,470 groups equals the unique `lesion_id` count. The step is
     now described as verifying that no unlinked duplicates exist, not as
     extending the grouping.
  3. **Centre prior beats every arm.** A centred-Gaussian baseline scores
     0.644 on overlap and 0.953 on the pointing game, exceeding every arm on
     both. The report previously said it explained "part of that margin"; it
     now states that no arm beats a trivial attend-to-the-centre prior, so the
     explainability numbers do not establish lesion localisation.
  4. **Wilcoxon mislabelled.** The test uses per-group mean correctness for the
     classification endpoint, so it measures per-lesion accuracy, not
     macro-F1. Columns are relabelled and the test is described explicitly.
     The Holm family column and the four analysis subsets are now visible in
     the table, and A* is rendered as `A*` rather than `Astar`.
- Also corrected: the 660,000-iteration figure now cites its source
  (`third_party/dmahr/config/akiec.json` `n_iter`, commit `0aee1bb`, consistent
  with the released checkpoint name `I725000_E1083`), and the Lama et al. hair
  masks are described as a skipped validation step rather than as annotations
  that do not exist.
- Not changed: Arm C's verdict. A reviewer proposed "failed reproduction,
  H1-H3 untested" as an alternative to "did not help". The reports now say the
  two possible causes (training budget, inference path) have not been
  separated and that the released checkpoint was never scored through this
  adapter, which is a stronger statement than either label.
- Verification: `scripts/16_verify_report_claims.py` recomputes all 42
  report claims from the artifacts; all pass. The three Word documents were
  regenerated and re-rendered to PDF.

## D19 - Arm C diagnosis: undertrained, not an adapter defect

- Question: Arm C scored PSNR 16.3 against a hairy-input floor of 22.1, with
  SSIM 0.130 +/- 0.010 and 185 of 186 pairs below 0.15. A near-constant
  failure of that kind is consistent with either an undertrained model or a
  defect in the inference path, and the two had not been separated.
- Method: `scripts/17_arm_c_diagnose.py` runs on validation synthetic pairs
  only, so no test-set choice depends on it. Three checks: an identity test
  (hairless reference through the full inference path), a head-to-head between
  the released checkpoint and ours at both step counts, and defect checks
  (value range, channel means, spatial alignment, high-frequency energy).
- Result: **outcome (b), undertrained.** The identity test returns PSNR 34.62 /
  SSIM 0.990 at 1,000 steps and 41.72 / 0.992 at 2,000. The released checkpoint
  reaches PSNR 26.80 / SSIM 0.684 (2,000) and 24.66 / 0.641 (1,000), both above
  the 22.11 floor. Our checkpoint reaches 14.99 / 0.079 and 14.12 / 0.073, both
  below it. No flip, no offset beyond 1 px, no range or channel error.
- Consequence 1: the reports' description of Arm C as "over-smoothed" is
  withdrawn. Our output carries 0.269 mean absolute Laplacian against the
  reference's 0.018, with global pixel std only falling from 0.166 to 0.127.
  That is a model that has not learned the mapping, not one that blurs.
- Consequence 2: Arm C stays a labelled under-resourced ablation. The repair
  needs $12.85 (ablation budget) to $24.38 (reference settings); the remaining
  balance is $7.35 after a $1.02 diagnosis, so it does not fit.
- Consequence 3: H1 and H2 are reported as untested for DM-AHR at a realistic
  training budget rather than as evidence against the method, because the
  released weights show this adapter can produce sensible output.
- Cost and evidence: 29.1 min on A100-40GB, $1.02. See
  `reports/arm_c_diagnosis.md` and `artifacts/logs/arm_c_*.csv`.

## D20 - H3 reported as not supported and uninformative

- Finding: a centred-Gaussian prior scores 0.644 on lesion overlap and 0.953 on
  the pointing game, exceeding every arm on both measures. No arm beats a
  trivial attend-to-the-centre prior.
- Decision: H3 is reported as **not supported, and the measurement is
  uninformative as specified**, rather than "partly supported". The distinction
  matters: "partly supported" would imply the saliency measures carry
  information about lesion localisation, and they do not.
- Reason: reporting a direction-of-difference between two methods that both
  lose to a centre prior would overstate what the data show.

## D21 - Test-set reuse disclosed

- Finding: regenerating Arm C would require scoring the 186 test pairs a second
  time. The first scoring is void if the inference path is ever repaired.
- Decision: no repair is affordable, so the test pairs were scored once and the
  diagnosis used validation pairs only. If a future run repairs Arm C, the
  first test scoring must be recorded as void in the report rather than
  presented as an independent second look.


## D22 - Pre-registration for the post-review revision (written before any new result)

Written and committed before any of the new analyses or Modal jobs below were
run or looked at. Frozen split hash at the start of this revision:
`fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b`
(`data/manifests/splits.sha256`, re-verified against `splits.csv`).
No test image, label, mask or metric is used to choose any threshold, iteration
count or sampling setting below.

Source facts supplied by the project owner (cited as the DM-AHR paper, Cancers
2024, 16, 2947; not independently re-verified in this repository): PSNR ~41.04
dB, SSIM ~0.966, LPIPS ~0.0128 (with self-supervision); ~39.13 dB / ~0.958
without; no hairy-input floor reported; 2,000 time steps, 20,000 iterations,
Adam lr 1e-4; test set = 700 HAM10000 images (100 per class) = DERMAHAIR, the
rest of HAM10000 used for training. The repository config says `n_iter` 660,000
and the released checkpoint is `I725000_E1083`, so the paper's text and its
own release disagree.

### Phase 1 - paper reproduction (released checkpoint, Kaggle DERMAHAIR pairs)
- Primary comparison: our adapter, native resolution 128, 2,000 steps, mean PSNR
  over the evaluated pairs against the hairless reference.
- **Rule:** mean PSNR >= 39.04 dB (within 2 dB of 41.04) -> "adapter validated";
  the gap to our 26.80 dB on our own synthetic pairs is then attributed to a
  harder hair domain. Anything lower -> "paper numbers not reproduced" and the
  adapter and checkpoint loading are investigated (resolution, conditioning
  image, value range, noise schedule, weight loading). "Adapter sound" is not
  concluded without passing this rule.
- Secondary diagnostic (not part of the rule): the same pairs at the data's
  native 256 px, because the vendored DERMAHAIR sample is 256 px while the
  config says 128. Pairs: first 100 after sorting file names, 50 for the
  identity test; steps 2,000 and 1,000. Metrics: PSNR, SSIM, LPIPS (AlexNet).
- Hairless references may contain natural hair; this is reported, not removed.

### Phase 2 - undertrained versus trained wrongly (our checkpoint)
- EMA: audited statically (upstream config marks it "not used now"; no EMA code
  exists upstream or in our trainer). A sampling-weights/EMA bug is ruled out by
  that audit; the raw-versus-EMA sampling comparison is recorded as not
  applicable rather than run.
- Static audit result recorded before any experiment: our trainer fed the
  denoiser `t / T`, whereas upstream trains and samples with the continuous
  noise level `sqrt(alphas_cumprod_prev)`. This is a candidate trainer bug and
  is tested, not assumed, by the experiments below.
- Noise-prediction MSE on validation pairs, 2x2 (released/ours x upstream
  conditioning / legacy `t/T` conditioning), timesteps {200, 800, 1400, 1900},
  100 validation pairs. **Trained-wrongly evidence:** ours has lower MSE under
  legacy conditioning than under upstream conditioning by a factor >= 1.5 while
  the released checkpoint shows the reverse.
- Checkpoints at 5k/10k/15k/20k (fixed trainer, cheap retrain), validation
  PSNR at 1,000 steps on 50 pairs. **Undertrained:** PSNR(20k) - PSNR(5k) >=
  1.0 dB and no successive drop > 0.5 dB. Otherwise "flat or erratic".
- Warm start: fine-tune the released checkpoint with our trainer for 500
  iterations (batch 4, lr 1e-4), legacy and fixed variants. **Loss jump:** mean
  L1 over the last 100 iterations > 1.5x the pre-fine-tune validation L1 under
  the matching conditioning. A jump in the legacy variant but not the fixed one
  confirms a trainer bug.
- Overfit: 8 training pairs, 2,000 iterations, 1,000-step sampling of those same
  pairs. **Pass:** mean PSNR to reference >= hairy-input PSNR + 3 dB.
- Branch taken is written to `reports/arm_c_diagnosis_v2.md` from the artifact,
  with branches (a) EMA bug, (b) undertrained, (c) trainer bug exactly as
  specified. Retrain scale if (c): 20,000 iterations (the paper's stated
  count), batch 4, lr 1e-4, saves every 5,000.
- Noise versus blur: high-frequency energy (mean absolute Laplacian), noise
  sigma by median-absolute-deviation in flat patches, and radial power spectra
  for hairy input, hairless reference and output; ratios stated against the
  hairy input as well as the reference.

### Phase 3 / 4 - arms and validity
- **Valid DM-AHR arm:** mean PSNR >= hairy-input PSNR + 3 dB and SSIM above the
  hairy-input SSIM on the *validation* synthetic pairs. This gate is applied
  before any test result for that arm is read.
- `C-released`: released checkpoint, hair-present gating rule identical to Arm
  C, hair-absent images pass through unchanged (as in every arm). Exploratory
  only; the checkpoint very likely saw our validation and test clean images.
- `C-fixed`: our checkpoint retrained with the fixed trainer; the confirmatory
  DM-AHR arm only if it passes the validity gate.
- Holm family stays m = 3 (A-B, A-C*, B-C*, with C* = C-fixed). If C-fixed is
  absent or invalid, the two C* comparisons are reported as untested and A-B
  keeps its m = 3 correction (conservative, and the family is not changed after
  seeing results). The legacy Arm C is moved to an appendix family.
- Sampling steps for any regenerated arm are set by the budget gate, not by the
  result: every Modal job re-measures throughput on 20 images, estimates cost,
  and refuses to start above the supplied `--budget-usd`.

### Phase 5 - saliency
- Overlap = mask recall |top-20% intersect mask| / |mask|; chance = 0.2 exactly.
  Pointing chance = probability a random argmax lands on the lesion = mean
  mask area (measured, not assumed 0.2). Ceiling = min(1, 0.2 / area).
- Off-centre robustness: translate image and mask by U(-30%, +30%) per axis
  (seeded by image id, reflect-padded image, zero-padded mask); images whose
  lesion loses more than 50% of its area are excluded and counted.
- **H3 uninformative** if the centred-Gaussian prior scores >= every arm on a
  metric on the unshifted test set AND at least 75% of lesion masks contain
  the image centre. The shifted analysis is reported either way.

### Phase 6 / 7 - coverage reanalysis and wording
- Pooled model: per-image correctness (mean over seeds) ~ arm x coverage,
  fractional-response logit, lesion-clustered sandwich standard errors,
  image-level coverage; sensitivity with group-mean coverage. Per-decile
  balanced accuracy with lesion-cluster bootstrap CIs replaces per-decile
  macro-F1.
- Hair-absent images (coverage below the frozen screening threshold) are
  pass-through in Arms B, C and A*, so that decile is an internal control for
  training-set composition rather than for hair removal.
- A-B macro-F1 is worded as "clear; magnitude +0.019 to +0.087", not
  "negligible". Cohen's dz is not defined for macro-F1 (non-decomposable over
  lesions) and is dropped there; it is kept only for per-pair quantities.
- Duplicates: "no unlinked near-duplicates found at the stated pixel
  thresholds".

### Budget
`BUDGET_USD` was left unfilled by the owner. The last recorded balance
estimate was ~$7.35 (decision D19). No Modal job in this revision has been run.
Every new Modal entry point requires an explicit `--budget-usd` and aborts if
its own estimate exceeds it.
