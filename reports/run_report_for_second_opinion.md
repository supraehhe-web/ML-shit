> **SUPERSEDED (2026-11-13).** This document concludes that Arm C's failure is
> undertraining and that the inference path is sound. That conclusion is
> retracted: the training script gave the network an inverted noise-level signal
> (see `reports/oneshot_diagnostic_report.md` and `reports/hypotheses.md`). Cost
> figures and any "repair" plan below assumed the old script and no longer apply.
> The text is kept unchanged below for the record.

# DM-AHR-ViT: complete run report and an open question

This document reports every measurement we have, so an independent reviewer can
judge one question: **if we train the diffusion model for 660,000 iterations
instead of 20,000, will image quality and downstream classification improve?**

All numbers below trace to files in `artifacts/logs/` and `artifacts/predictions/`.

## 1. Study design

Seven-class skin lesion classification on HAM10000 (10,015 confirmatory images,
11,720 in the client collection with 1,705 quarantined). Frozen, lesion-grouped
70/15/15 split: 7,011 / 1,502 / 1,502 images, 7,470 groups, zero group overlap.
Backbone ViT-B/16, ImageNet-1k pretrained, 224 px, seeds 42 / 123 / 2026.

Five input conditions:

| Arm | Input |
|---|---|
| A | original image |
| B | DullRazor-style morphology plus Telea inpainting |
| C | our DM-AHR diffusion run, 20,000 iterations |
| A* | original at 128 px, controlling the resolution change |
| Creleased | the authors' released DM-AHR checkpoint, 25-step respaced sampling |

## 2. Classification results (locked test set, 1,502 images, 1,437 groups)

Macro-F1, mean ± SD over three seeds:

| Arm | Macro-F1 | Accuracy | Balanced acc. | Macro AUC |
|---|---|---|---|---|
| A | 0.750 ± 0.005 | 0.858 ± 0.012 | 0.735 | 0.969 |
| A* | 0.732 ± 0.008 | 0.851 ± 0.006 | 0.726 | 0.969 |
| B | 0.698 ± 0.009 | 0.842 ± 0.005 | 0.687 | 0.965 |
| C | 0.426 ± 0.013 | 0.685 ± 0.019 | 0.452 | 0.878 |
| Creleased | 0.713 ± 0.028 | 0.848 ± 0.009 | 0.696 | 0.967 |

Per-seed values: C = 0.4226 / 0.4152 / 0.4405. Creleased = 0.7082 / 0.6875 / 0.7429.

## 3. Image quality on 186 held-out synthetic pairs

PSNR / SSIM against genuinely hairless references. The hairy input is the
do-nothing floor.

| Condition | PSNR (dB) | SSIM |
|---|---|---|
| Arm B (classical) | 35.732 | 0.918 |
| Creleased (released ckpt) | 27.076 | 0.703 |
| Hairy input (floor) | 22.127 | 0.520 |
| Arm C (ours, 20k) | 14.292 | 0.074 |

Arm C sits 7.8 dB **below** the floor. Creleased clears the floor by 4.95 dB.

## 4. Paired statistics

Lesion-group paired bootstrap, 10,000 replicates, percentile 95% CI, two-sided,
Holm correction within family.

Planned family (frozen before evaluation):

| Comparison | Subset | Difference | 95% CI | Holm p |
|---|---|---|---|---|
| A − B | full | +0.052 | +0.019 to +0.087 | 0.0012 |
| A − C | full | +0.324 | +0.283 to +0.366 | <0.0001 |
| B − C | full | +0.272 | +0.225 to +0.320 | <0.0001 |
| A − A* | full | +0.018 | −0.008 to +0.045 | 0.167 |

Exploratory family (Creleased added after the protocol was frozen):

| Comparison | Subset | Difference | 95% CI | Holm p | Cohen's dz |
|---|---|---|---|---|---|
| A − Creleased | full | +0.038 | +0.005 to +0.071 | 0.048 | 0.05 |
| B − Creleased | full | −0.015 | −0.049 to +0.019 | 0.400 | −0.01 |
| Creleased − B (PSNR) | pairs | −8.657 | −9.063 to −8.254 | <0.0001 | −3.07 |
| Creleased − hairy (PSNR) | pairs | +4.946 | +4.503 to +5.388 | <0.0001 | +1.64 |

## 5. Diffusion training as executed

20,000 iterations, batch 4, Adam lr 1e-4, 128x128 native, linear beta schedule
with 2,000 timesteps, U-Net inner channel 64, channel multipliers
[1,2,4,8,8], dropout 0.2. Wall clock 1,409.7 s on A100-80GB, which is
0.0705 s/iteration. Best validation loss 0.0170.

Upstream reference: `config/akiec.json` in the authors' repository sets
`n_iter = 660000`. The released checkpoint is named `I725000_E1083`, i.e.
725,000 iterations.

**Correction to an earlier version of this report.** It claimed we trained at
about 3% of the published budget. That framing was wrong. The paper's stated
reference settings, quoted in `docs/DEVELOPMENT_KIT.md` line 178, are 2,000
timesteps and **20,000 training iterations**. That is exactly what we ran. The
660,000 figure comes from the shipped config file, not from the paper's stated
reference. So our run matched the paper's stated budget and still failed, which
points at implementation rather than budget.

Two further differences between our run and the released checkpoint that the
budget framing obscured:

| Setting | Ours | Upstream |
|---|---|---|
| Iterations | 20,000 | 660,000 config, 725,000 checkpoint |
| EMA | **none** | decay 0.9999, started at step 5,000 |
| Batch size | 4 | 4 |
| Noise-prediction convention | legacy (see 7c) | upstream |

Our trainer has no EMA at all, while upstream applies EMA from step 5,000
onwards. If the released weights are EMA weights, part of the observed gap is an
EMA difference, not an iteration-count difference.

## 6. The evidence that more iterations would help

**6a. Released versus ours, identical everything except weights.** 100 validation
synthetic pairs, same adapter, same resize, same value range, same sampler:

| Checkpoint | Steps | PSNR | SSIM | Beats its own input? |
|---|---|---|---|---|
| Released | 2,000 | 26.80 | 0.684 | yes |
| Released | 1,000 | 24.66 | 0.641 | yes |
| Ours | 2,000 | 14.99 | 0.079 | no |
| Ours | 1,000 | 14.12 | 0.073 | no |

Same code path, same data, same step counts. The only difference is the trained
weights. The gap is 11.8 dB PSNR and 0.61 SSIM.

**6b. The inference path is sound.** A genuinely hairless reference image passed
through the exact Arm C path returns:

| Steps | n | PSNR | SSIM | High-frequency retained |
|---|---|---|---|---|
| 1,000 | 50 | 34.62 | 0.990 | 92.6% |
| 2,000 | 50 | 41.72 | 0.992 | 90.9% |

Value range, channel order and spatial alignment also check clean (best shift 0
to −1 px, flip correlation near zero). So the pipeline can produce near-identity
output, and the failure is localised to the weights.

**6c. Budget ratio.** 20,000 of 660,000 iterations is 3.0%. The released
checkpoint demonstrates that this architecture and adapter produce good hair
removal at the full budget.

## 7. The evidence that cuts against it

This section is the reason I am asking for a second opinion rather than
recommending the spend outright.

**7a. There is no learning curve.** We have two endpoints, 20k and 725k. We have
no checkpoint at 100k, 250k or 400k, so we have never measured the trajectory.
The inference that quality rises monotonically with iterations is an assumption,
not an observation we made.

**7b. The overfit test failed.** Training 2,000 iterations on 8 fixed pairs to
see whether the model can memorise them gave PSNR 14.54 and SSIM 0.287, a margin
of −7.31 dB against the +3 dB required. A model that cannot overfit eight pairs
is not obviously limited only by iteration count.

**7c. Our run and the released run differ in more than iteration count.** Noise
prediction error, measured at four timesteps on 100 validation pairs, in the
convention the released weights expect:

| Checkpoint | Noise MSE |
|---|---|
| Released | 0.004469 |
| Ours (as trained) | 0.042166 |
| Ours, retrained variant | 0.000772 |

Our original run was trained under a different noise-prediction convention from
the one the released weights expect. So "more iterations" is not the only
difference between the two endpoints. A retrained variant under the expected
convention reaches a lower noise MSE than the released checkpoint (0.00077
versus 0.00447) yet still produces only 15.70 dB at 250 respaced steps. Better
noise fit did not translate into better images.

**7d. Sampling steps, for completeness.** With naive truncation of the released
model, quality needs the full schedule: 12.70 dB at 50 steps rising to 26.41 at
2,000. With a properly respaced schedule the same model reaches 26.91 dB at 25
steps, 26.73 at 50, 26.63 at 100 and 26.59 at 250. Respacing is what makes short
sampling viable, which is why image regeneration is cheap.

## 8. Cost of the experiment

Measured throughput 0.0705 s/iteration on A100-80GB ($2.50/h). A100-40GB is
$2.10/h.

| Iterations | GPU-hours | A100-80GB | A100-40GB |
|---|---|---|---|
| 100,000 | 2.0 | $5 | $4 |
| 250,000 | 4.9 | $12 | $10 |
| 400,000 | 7.8 | $20 | $16 |
| 660,000 | 12.9 | $32 | $27 |
| 725,000 | 14.2 | $35 | $30 |

Regenerating 8,993 hair-present images at 25 respaced steps (0.090 s/image) is
0.2 GPU-hours, about $1. At 2,000 steps it is 10.5 GPU-hours, about $26.
Retraining three ViT seeds and redoing evaluation and XAI adds about $1.

**Full repair at 660,000 iterations: about $34 and roughly one day of wall clock.**

## 9. The question

An independent review of this report concluded that budget is unlikely to be the
defect and recommended staged diagnostics first. Its reasoning, which we accept:

- Our 20,000 iterations matched the paper's stated reference, so "undertrained
  relative to the paper" was not a valid premise.
- The 6a contrast compares weights that differ in training convention, EMA and
  iteration count at once, so it cannot attribute the gap to iterations.
- Low noise MSE with bad samples, as in the retrained variant, places the
  defect downstream of average noise accuracy: sampler, respacing, noise-level
  embedding, conditioning at sampling time, x0 clipping, or raw versus EMA.
- The released model already scores 0.713, tied with DullRazor and below
  original images, so beating 0.750 is unlikely regardless of budget.

The cheapest discriminating test is one-step x0 PSNR across a full timestep
grid for released versus our model, plus a zeroed-condition ablation. If
one-step reconstruction matches the released model but multi-step sampling does
not, the sampler or schedule is at fault and no amount of training will help.

Given all of the above, and especially the three counter-indications in section
7, please judge:

1. **Will training to 660,000 iterations measurably improve Arm C's image
   quality (PSNR, SSIM) and downstream macro-F1, or is the failure mode something
   else?**
2. **How much weight should the failed 8-pair overfit test carry?** Does failing
   to overfit 8 pairs in 2,000 iterations indicate a defect that more training
   will not fix, or is 2,000 iterations simply too few to tell?
3. **Is the noise-prediction-convention difference material?** The retrained
   variant fits the noise better than the released checkpoint (0.00077 versus
   0.00447) but still produces poor images. What does that imply?
4. **Is there a cheaper diagnostic than the $34 full run** that would settle
   this, for example a 100,000-iteration checkpoint evaluated on validation pairs
   at about $5?
5. **Ceiling check.** The released checkpoint, which is the authors' own fully
   trained model, scores 0.713 macro-F1, statistically tied with classical
   DullRazor (Holm p = 0.40) and below original images (Holm p = 0.048, negligible
   effect size). Should we expect our own full-budget model to beat 0.750, or is
   0.70 the realistic ceiling for this approach on this dataset?
