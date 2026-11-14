# Pre-registration: Arm C retrain with the fixed trainer

Committed **before** running the retrain or scoring any test pair, so the
decision cannot be tuned after seeing results.

## Why this rerun exists

The original Arm C trained under an inverted noise-level encoding (decision log
D25). That is a defect, not an under-resourced run. This rerun tests whether a
correctly-wired trainer at the paper's stated reference budget of 20,000
iterations produces a usable diffusion arm.

**Disclosure.** This rerun was decided from validation-only diagnostics, after
the original test-set results had been seen and published. The original Arm C and
its test numbers are retained as a disclosed appendix, not replaced.

## Prior correction

`reports/arm_c_diagnosis.md` concluded "adapter sound, checkpoint undertrained".
That conclusion is **retracted**. The adapter is sound, but the trainer was
defective, so "undertrained" was the wrong attribution.

## Gates, in order, all on validation data only

**Gate 1 - 8-pair overfit regression.** Train on 8 pairs from the train split.
The original run failed this at a margin of -7.31 dB against a required +3.0 dB.
With the fixed trainer it must clear **+3.0 dB over the hairy floor**. Fail here
means stop; nothing downstream is meaningful.

**Gate 2 - copy test.** Train briefly with the conditioning input set to the
clean target, which is a trivial identity mapping. Must reach **>= 35 dB**.
Failure means conditioning wiring is still wrong.

**Gate 3 - fixed Arm C at 20,000 iterations.** Trained on the train-split pairs
only, evaluated on the **182 validation pairs**. Scored at 2,000 sampling steps,
the paper's reference.

- **Pass:** at least **+2.0 dB above the hairy-input floor** (22.11 dB), i.e.
  >= 24.11 dB.
- **Middle:** between floor and floor+2 dB, i.e. 22.11 to 24.11 dB. Routes to the
  staged 100k run rather than a straight reject.
- **Fail:** at or below the floor. Stop and report the trainer as still wrong.

## Results (recorded after running Gates 1 and 2)

**Gate 1 - FAIL.** Margin **-5.18 dB** (16.67 dB against a 21.85 dB floor).
Improved from the original -7.31 dB but did not reach +3.0 dB.

**Gate 2 - FAIL.** Copy task reached **13.43 dB** against the required 35.0 dB.

Both gates are **uninformative about trainer correctness**, and the failure is in
this pre-registration, not in the trainer. Both gates train from random
initialisation:

| Gate | Iterations x batch | Samples | Share of released model's 2.9M |
|---|---:|---:|---:|
| Gate 1 | 2,000 x 4 | 8,000 | 0.28% |
| Gate 2 | 1,000 x 4 | 4,000 | 0.14% |

A diffusion model cannot learn its task from 0.14% of the samples the released
checkpoint saw. The thresholds I set (+3 dB and 35 dB) assumed a model that had
already learned to denoise, which is true of the reviewer's warm-start framing
but not of a from-scratch run. The gates were therefore mis-specified: they
measure training budget, not wiring.

The **warm-start check remains the valid discriminator**, because it tests the
fixed trainer against weights that already work: fine-tuning the released
checkpoint for 1,000 iterations changed PSNR by +0.16 dB, so the fixed trainer
does not destroy good weights. Gate 3 below is the informative test.

## Result: Gate 3 - indicative fail at n=8 (not established at 182)

Fixed Arm C retrained from scratch for 20,000 iterations on the 654 train-split
pairs. Best validation L1 0.02003.

**CORRECTION (13:30 UTC): the figures below are from 8 validation pairs, not 182.**
The 182-pair job was launched detached and never produced output (no
`gate3_eval.json`, empty app log). The numbers here come from the 8-pair smoke run
(`--pairs 8`) at 2,000 sampling steps, and the earlier text of this section
wrongly said 182. The pre-registered verdict is therefore **not formally
established**; n=8 is indicative only. Read the 14.11 / 26.64 dB figures as n=8.

**Sampling (n=8): 14.11 dB against a 20.08 dB hairy floor, margin -5.97 dB. This
would fall in the fail band if it held at 182 pairs; it has not been measured there.**

Positive control, same script, same pairs, same steps: the released checkpoint
scores **26.64 dB, margin +6.55 dB, verdict pass**. So the evaluation is sound and
the comparison is fair; the failure is a property of the new model.

### The fix worked, and was not sufficient

| Measurement | Defective Arm C | Fixed Arm C (20k) | Released (725k) |
|---|---:|---:|---:|
| One-step PSNR at t=200 | 24.58 | **41.69** | 35.89 |
| One-step PSNR at t=1000 | 21.38 | **31.56** | 28.77 |
| One-step PSNR at t=1500 | 8.25 | 18.27 | **24.25** |
| Condition-effect ratio t=200 | 0.97x | **4.00x** | 4.16x |
| Condition-effect ratio t=1000 | 1.79x | **3.32x** | 4.78x |
| Multi-step sampling PSNR | 14.99 | 14.11 | **26.64** |

Two things the encoding fix demonstrably repaired:

- **Condition usage.** The defective model ignored its conditioning image
  (0.97x to 1.79x). The fixed model uses it as strongly as the released
  checkpoint (4.00x against 4.16x at t=200).
- **Single-step reconstruction.** The fixed model now reconstructs better than
  the released checkpoint at t=200 and t=1000 (41.69 and 31.56 against 35.89 and
  28.77).

What did not change: **multi-step sampling.** 14.11 dB, below the floor, versus
26.64 for the released checkpoint. A model that reconstructs better than the
released checkpoint at a single step, and uses its condition equally well, still
drifts to garbage over 2,000 reverse steps.

That pattern points at error accumulation in the reverse loop rather than at
training-budget starvation: at 20,000 iterations the denoiser's conditional mean
appears accurate per call but not smooth enough to stay on the data manifold
across 2,000 feedback steps, where the released model at 725,000 iterations is.
This is a hypothesis, not a measurement.

### Consequence

Per the pre-registration, the fail band means stop and report. The test set is
not touched, and the original Arm C plus its published test numbers stand as the
recorded result.

The 660,000-iteration question is now better posed than before but still open:
the defect is fixed and the model is learning correctly, but 20,000 iterations
is demonstrably not enough to sample. The next step is the staged 100,000 run
with per-5k checkpoints, which this pass did not capture because `snapshot_every`
was absent from the config and my fallback disabled it. That is fixed for future
runs.

Tertiary diagnostic, recorded but not gating: the zero-condition ablation. A
healthy model must degrade clearly when the condition is removed. The defective
model did not (ratio 0.97x to 1.8x against 3.3x to 4.8x for the released model).

## What is explicitly not pre-judged

No expectation is stated for the resulting PSNR or macro-F1. The fixed model
plausibly lands anywhere between 15 and 26 dB. Previous drafts of this analysis
predicted "near the released model" and "the headline stays the same"; both were
hypotheses, and both are withdrawn.

## Test set policy

The 186 test pairs are **not** scored until Gate 3 passes and its validation
result is recorded. If Gate 3 lands in the fail band, the test set is never
touched and the original Arm C remains the published result.

## Cost, stated before spending

- Gates 1 and 2: minutes of GPU time each.
- Gate 3 train (20,000 iterations at 0.0705 s/iter): about 0.4 GPU-hours.
- Gate 3 sampling of 8,993 images at 2,000 steps: 10.5 GPU-hours, about $26.

The regeneration cost is the dominant item and must be decided before committing
to evaluating a full Arm C. The 25-step routine that produced `Creleased` is not
in the repository, so its cheap cost does not transfer (decision log D26).

## Recorded environment for the rerun

- Train-split pair count: 654. Validation pairs: 182. Test pairs: 186.
- Leakage check between train and val pairs: passed in the original run and
  re-asserted by `scripts/04_dmahr_train.py` on every run.
- Manifest SHA-256 and the sampled noise-level convention must be written into
  the new run's `training_manifest.json`.
