> **SUPERSEDED (2026-11-13).** This document concludes that Arm C's failure is
> undertraining and that the inference path is sound. That conclusion is
> retracted: the training script gave the network an inverted noise-level signal
> (see `reports/oneshot_diagnostic_report.md` and `reports/hypotheses.md`). Cost
> figures and any "repair" plan below assumed the old script and no longer apply.
> The text is kept unchanged below for the record.

# Final changelog

What changed in this pass, why, and which numbers moved. Written against the state at commit
`578ba97`.

## 1. Arm C diagnosis — the main new result

**Outcome (b): undertrained.** The inference path is sound; the checkpoint trained for this study
is not.

| Measurement | Result |
|---|---|
| Identity test: hairless reference through the same path | PSNR 34.62 / SSIM 0.990 (1,000 steps); 41.72 / 0.992 (2,000) |
| Released DM-AHR checkpoint on validation pairs | PSNR 26.80 / SSIM 0.684 (2,000); 24.66 / 0.641 (1,000) |
| Our checkpoint on the same pairs | PSNR 14.99 / SSIM 0.079 (2,000); 14.12 / 0.073 (1,000) |
| Hairy-input floor | PSNR 22.11 |

The released weights clear the floor; ours sit 7–8 dB below it. Same adapter, same pairs, same step
counts — only the weights differ. Evidence: `reports/arm_c_diagnosis.md`,
`artifacts/logs/arm_c_diagnosis.json`, `arm_c_identity.csv`, `arm_c_headtohead.csv`.

**Cost:** 29.1 minutes on A100-40GB, **$1.02**. Validation pairs only; the test pairs were not read.

## 2. The "over-smoothed" description was wrong and is withdrawn

Every report said Arm C's output was over-smoothed. The diagnosis contradicts it: our output carries
**0.269 mean absolute Laplacian against the reference's 0.018** — roughly 15× *more* high-frequency
energy — while global pixel standard deviation falls only modestly (0.127 against 0.166). Low
contrast plus high pixel-to-pixel variation is a model that has not learned the mapping, not one
that has learned to blur.

Replaced in `10_finalize.py`, `13_preprocessing_report.py`, `14_results_report.py` and
`docs/CODE_GUIDE.md`. The remaining occurrence of the word is an explicit negation.

## 3. H3 downgraded from "partly supported" to "not supported, uninformative"

A centred-Gaussian prior scores **0.644** on lesion overlap and **0.953** on the pointing game,
exceeding every arm on both measures. No arm beats a trivial attend-to-the-centre prior, so the
saliency numbers do not demonstrate lesion localisation at all. This is a defect in the measurement,
not a finding about diffusion hair removal.

## 4. Statistics

| Change | Reason |
|---|---|
| `C vs A*` added as a secondary control | resolution-matched comparison; C − A* = −0.306 (CI −0.348 to −0.265), so resolution cannot explain the deficit |
| `negligible` flag added | A−B has p = 0.0012 but Cohen's dz = 0.056 — statistically clear, practically negligible |
| `family` column surfaced in the report table | shows A−A* is corrected in its own family, so the planned family really is three comparisons |
| Four analysis subsets documented | `full` 1,437 groups, `hair_present` 1,251, `held_out_pairs` 186, `test_x1` 1,437 |
| Wilcoxon labelled by what it tests | per-group mean correctness, not macro-F1 |
| `Astar` rendered as `A*` | readability |

The A−B p-value was spot-checked and is correct; it is small because n is large, not because of an
error.

## 5. Hair-coverage analysis (new, exploratory, no retraining)

Macro-F1 by hair-coverage decile on the existing predictions:

- **Arm B is not worse where hair is scarce.** In the lowest-coverage decile B scores 0.555 against
  A's 0.507. The B−A gap is not significant in any decile after Holm correction, so the evidence
  does not support "false-positive damage" as the mechanism behind B's overall 0.052 drop.
- **Arm C's deficit appears only where hair was present.** C−A is −0.15 to −0.25 in every decile
  with hair and statistically indistinguishable from zero in the hair-absent decile (−0.020, CI
  −0.051 to +0.009). That localises the damage to the processed images.

Evidence: `artifacts/logs/hair_coverage_by_decile.csv`, `hair_coverage_gaps.csv`.
Label: exploratory, no confirmatory claims.

## 6. Claims corrected (carried over from the previous pass, re-verified)

| Claim | Correction |
|---|---|
| Masks "not stated to be expert-traced" | Dataverse says a single dermatologist created them, network-initialised then hand-corrected — single-rater, not undocumented |
| Duplicate merge as leakage control | all 21 confirmed pairs already shared a `lesion_id`, so grouping is unchanged |
| "Part of that margin from centre bias" | the centre prior beats every arm outright |
| "No independent hair annotations exist" | Lama et al. published them; the step was skipped |

## 7. Documentation

- `docs/CODE_GUIDE.md` restructured to **data → methodology → ML pipeline → notebook cell guide**,
  with the pipeline section now covering every phase in order.
- All **64 code cells** across the three notebooks now carry an explanatory markdown cell above
  them (17 / 19 / 28).
- `scripts/20_deliverable_gates.py` generates `reports/deliverable_checklist.md` from measurement.

## 8. Numbers that moved

Only these, and none of them are model results:

| Value | Before | After | Why |
|---|---|---|---|
| Statistical comparison rows | 14 | 22 | C−A* control added to two subsets |
| Report length | 11.5 KB | 19.9 KB | diagnosis, hypothesis verdicts, environment, subset documentation |
| Artifact inventory rows | 173 | 178 | diagnosis artifacts |
| Gates passing | 14/14 project gates | 14/14 + 8/8 deliverable gates | new checklist |

**No classification, image-quality or explainability number changed.** Arm A 0.750, A* 0.732,
B 0.698, C 0.426; PSNR 35.732 / 16.312 / 22.127; all untouched.
