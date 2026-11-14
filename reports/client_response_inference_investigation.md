> **SUPERSEDED (2026-11-13).** This document concludes that Arm C's failure is
> undertraining and that the inference path is sound. That conclusion is
> retracted: the training script gave the network an inverted noise-level signal
> (see `reports/oneshot_diagnostic_report.md` and `reports/hypotheses.md`). Cost
> figures and any "repair" plan below assumed the old script and no longer apply.
> The text is kept unchanged below for the record.

# Response: inference-pipeline investigation before further GPU spend

**Status: the investigation you asked for is complete.** It was run on validation
pairs only, cost $1.02, and did not touch the test set. The short answer is that
the inference pipeline is correct, the released checkpoint works through it, and
our 20,000-iteration checkpoint is undertrained. Details and costs below.

## 1. Released checkpoint under the correct inference configuration

The released DM-AHR checkpoint was run through our adapter at four step counts on
50 validation pairs. The paper's reference is 2,000 steps; we also tested the
respaced schedule.

| Steps | PSNR (dB) | SSIM | LPIPS | s/image |
|---:|---:|---:|---:|---:|
| 25 | **26.91** | **0.698** | 0.483 | 0.090 |
| 50 | 26.73 | 0.695 | 0.486 | 0.111 |
| 100 | 26.63 | 0.690 | 0.487 | 0.215 |
| 250 | 26.59 | 0.688 | 0.487 | 0.531 |

The hairy-input floor on the same pairs is 22.34 dB / 0.531 SSIM. The released
checkpoint clears it by 4.3-4.6 dB at every step count, so it is genuinely
removing hair.

**Finding: more sampling steps did not improve PSNR.** Within this sweep the
score declines slightly from 25 to 250 steps. A separate 100-pair head-to-head
gave 26.80 dB at 2,000 steps and 24.66 dB at 1,000 steps, so the step-count
effect is small and pair-set dependent. We therefore use 25 steps, which is
within noise of the best and roughly 20x cheaper to run.

## 2. Inference-pipeline review

**Identity test.** A genuinely hairless reference was passed through the exact
Arm C path (same adapter, resize, value range, sampler) and compared with itself.
A broken path cannot pass this.

| Steps | n | PSNR (dB) | SSIM | high-frequency retained |
|---:|---:|---:|---:|---:|
| 1,000 | 50 | 34.62 | 0.990 | 92.6% |
| 2,000 | 50 | 41.72 | 0.992 | 90.9% |

**Defect checks.**

| Check | Result | Reading |
|---|---|---|
| Value range | output within 0-255 after postprocess | no range error |
| Channel order | per-channel means track reference | no RGB/BGR swap |
| Spatial alignment | best shift 0 to -1 px, flip correlation ~0 | no flip or offset |
| High-frequency energy | ours 0.269 vs reference 0.018 | excess noise, not smoothing |
| Global pixel std | ours 0.127 vs reference 0.166 | mild collapse |

**Two genuine upstream bugs were found and worked around** (decision log D3):
`GaussianDiffusion.super_resolution(continous=False)` returns only the last image
of a batch, and `__init__` never calls `set_new_noise_schedule`, so the diffusion
buffers do not exist until an explicit call. Both are in the authors' code, not
ours. Our adapter calls the schedule explicitly and takes the final B rows of the
`continous=True` trajectory, which the smoke test verifies returns the whole
batch with geometry preserved and identical pixels on reload.

**Conclusion: the inference path is not the defect.**

## 3. Released checkpoint versus our 20,000-iteration model

100 validation pairs, both checkpoints, both step counts, identical adapter.

| Condition | PSNR (dB) | SSIM | Beats its own input? |
|---|---:|---:|---|
| Released, 2,000 steps | **26.80** | **0.684** | yes |
| Released, 1,000 steps | 24.66 | 0.641 | yes |
| Ours, 2,000 steps | 14.99 | 0.079 | no |
| Ours, 1,000 steps | 14.12 | 0.073 | no |

Same adapter, same pairs, same step counts. The only difference is the weights.
Our checkpoint sits 7-8 dB *below* the do-nothing floor with SSIM near 0.07,
which is essentially no structural correspondence.

On the locked test set, the released checkpoint (25-step respaced) scores
**27.08 dB / 0.703 SSIM** on the 186 held-out pairs and classifies at
**macro-F1 0.713 ± 0.028** over three seeds. Our 20,000-iteration arm scores
14.29 dB / 0.074 and 0.426.

One correction to the earlier report: our output was described as
"over-smoothed". The measurements do not support that word. It carries 15x the
reference's high-frequency energy, which is the signature of a model producing
noise around a compressed range, not a blur. The accurate description is that
the checkpoint has not learned the hair-removal mapping.

## 4. Cost and time by training configuration

Measured throughput: 0.0705 s/iteration (A100-80GB, batch 4, 20,000 iterations
in 1,409.7 s). Modal rates: A100-80GB $2.50/h, A100-40GB $2.10/h.

| Iterations | GPU-hours | A100-80GB | A100-40GB |
|---:|---:|---:|---:|
| 20,000 (done) | 0.4 | $1 | $1 |
| 100,000 | 2.0 | $5 | $4 |
| 250,000 | 4.9 | $12 | $10 |
| 400,000 | 7.8 | $20 | $16 |
| 660,000 (upstream config) | 12.9 | $32 | $27 |
| 725,000 (released checkpoint) | 14.2 | $35 | $30 |

Regenerating all 8,993 hair-present images, at measured per-image cost:

| Steps | s/image | GPU-hours | A100-80GB |
|---:|---:|---:|---:|
| 25 | 0.090 | 0.2 | $1 |
| 100 | 0.215 | 0.5 | $1 |
| 1,000 | 2.10 | 5.2 | $13 |
| 2,000 | 4.20 | 10.5 | $26 |

Retraining the three ViT seeds, re-evaluating and redoing XAI adds about $1.

**Because 25 steps is sufficient, the full repair is about $34** (660,000
iterations of training, $32, plus $1 regeneration and $1 downstream), not the
$40-50 implied by the paper's 2,000-step sampling. At 725,000 iterations it is
about $37.

## Recommendation

The investigation you asked for is done and it rules out the inference pipeline.
The remaining question is only whether a fair test at published budget is worth
$34.

Two things to weigh before spending it:

1. **The ceiling is already known.** The released checkpoint is the authors' own
   fully-trained model, and it scores 0.713 - statistically tied with classical
   DullRazor (difference -0.015, 95% CI -0.049 to +0.019, Holm p=0.40) and below
   original images (+0.038, CI +0.005 to +0.071, Holm p=0.048, negligible effect
   size). Training our own model to the same budget should move Arm C from 0.426
   to roughly 0.70. It is very unlikely to beat the 0.750 baseline.
2. **What the $34 buys is a clean claim, not a better number.** It removes the
   "undertrained" caveat and makes the diffusion arm a fair test of the published
   method. That is a defensible result for a thesis or paper; it is not a path to
   a higher headline figure.

If the deliverable requires the diffusion arm at published budget, the $34 is
justified and we can start immediately. If the goal is a better result than
baseline, no amount of training has demonstrated that, and we should say so
rather than spend.
