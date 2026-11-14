# Arm C: does more training explain the failure?

**Answer: no. Do not train longer until the trainer is fixed. The trainer feeds
the U-Net a different noise-level encoding from the one the sampler uses, and the
two are inverted.**

This supersedes an earlier version of this report that answered "yes for image
quality". That answer was wrong, for the reasons in section 6.

Data: `artifacts/logs/oneshot_diagnostic.json` and
`artifacts/logs/noise_convention_check.json`. Scripts:
`scripts/25_oneshot_diagnostic.py`, `scripts/26_convention_check.py`. Validation
pairs only; the test set was not read.

## 1. The defect

The U-Net receives a scalar describing how noisy the input is.
`NoiseLevelEncoder` (`third_party/dmahr/model/sr3_modules/unet.py`, line 23)
scales a frequency ramp by that scalar, so its meaning is fixed by what the
caller passes.

- **Sampler** (`GaussianDiffusion.p_sample`, line 154) passes
  `sqrt_alphas_cumprod_prev[t+1]`. It *decreases* with `t`: 0.997 at t=50,
  0.059 at t=1500. High value means nearly clean.
- **Our trainer**, before this fix, passed `t / num_timesteps`. It *increases*
  with `t`: 0.025 at t=50, 0.750 at t=1500. High value meant maximum noise.

The two signals are not merely different, they are inverted. The model was
trained to read a small number as "almost no noise" and is asked at inference to
read a small number as "almost pure noise".

| t | Our trainer sent | Sampler sent |
|---:|---:|---:|
| 50 | 0.0250 | 0.9968 |
| 200 | 0.1000 | 0.9509 |
| 500 | 0.2500 | 0.7307 |
| 1000 | 0.5000 | 0.2853 |
| 1500 | 0.7500 | 0.0594 |
| 1900 | 0.9500 | 0.0108 |

Upstream `p_losses` (line 221) passes `sqrt_alphas_cumprod_prev` sampled
continuously between `t-1` and `t`, per sample in the batch. Our trainer now does
the same, and `tests/test_noise_level_convention.py` pins it.

## 2. Empirical confirmation

For each checkpoint and timestep we measured epsilon-prediction error under both
encodings, on 50 validation pairs with a fixed noise draw. A checkpoint should
show lower error under the encoding it was trained with. Ratio is
`t_over_T error / sampler error`; above 1 means the sampler encoding wins.

| Checkpoint | t=200 | t=500 | t=1000 | t=1500 | Prefers |
|---|---:|---:|---:|---:|---|
| Released | 4.0x | 2.9x | 3.9x | 20.5x | sampler at every t |
| Ours | 0.1x | 0.1x | 0.2x | 0.1x | **`t/T` at every t** |

Our checkpoint is roughly 10x better under the encoding the sampler never uses.
That is not noise, and it is not a training-budget effect.

This reproduces, from committed source, what
`artifacts/logs/arm_c_v2_noise_mse.json` claimed and what an earlier version of
this report wrongly set aside because its generating script was missing.

## 3. One-step reconstruction, with a correct baseline

One-step x0 reconstruction PSNR on 100 validation pairs, fixed noise seed, same
conditioning arrangement and noise-level embedding as the sampler. The
`noisy input` column is the exact analytic baseline `10*log10(4/(1-alpha^2))`:
what you get by returning the noisy image untouched. It is the floor that matters
at each timestep.

| t | Noisy input | Ours | Released | Ours vs floor |
|---:|---:|---:|---:|---:|
| 50 | 28.1 | 31.54 | 38.56 | +3.4 |
| 200 | 16.2 | 24.58 | 33.90 | +8.3 |
| 500 | 9.3 | 23.44 | 31.00 | +14.1 |
| 1000 | 6.4 | 21.38 | 29.51 | +15.0 |
| 1500 | 6.0 | 8.25 | 25.38 | +2.2 |
| 1900 | 6.0 | 6.36 | 14.61 | +0.3 |

**An earlier version used 22.11 dB as the floor at every t. That was wrong.** At
low `t` the hairless reference is barely perturbed, so returning it untouched
already scores 28.1 dB. Our model is only 3.4 dB above that at t=50, not 9.4 dB.

At high `t` the error runs the other way: noise dominates, so the only usable
information is the conditioning image. Copying the hairy input scores 22.11 dB
against the reference. The released model beats that at t=1500 (25.38). Ours
scores 8.25, far below copying. **That is a conditioning failure, not a budget
effect.**

## 4. Condition usage

Zeroing the conditioning image, epsilon-error ratio (zeroed / with condition).
Above 1 means the model relies on the condition.

| Checkpoint | t=200 | t=500 | t=1000 | t=1500 |
|---|---:|---:|---:|---:|
| Released | 4.28x | 4.13x | 4.83x | 3.34x |
| Ours | 0.97x | 1.05x | 1.79x | 1.23x |

Our model barely uses the hair image, and at t=200 removing it slightly *helps*. A
model that denoises but ignores its condition produces output unrelated to the
input, which matches the measured SSIM of 0.074 and the SSIM 0.287 on the failed
8-pair overfit test.

## 5. What this means

The failure is consistent with the encoding defect, and the encoding defect
explains all four observations that a budget story could not: output uncorrelated
with the input, near-zero condition usage, failure to overfit 8 pairs, and
epsilon-error that is far lower under the wrong encoding.

**More iterations on the old trainer would have reproduced the defect**, at any
budget. That is why the answer is no.

## 6. Corrections to the earlier version of this report

- It claimed "uniform deficit across noise levels is the signature of incomplete
  convergence". Invalid: a wiring bug touching the whole network is also
  uniform, and the gap was not uniform anyway (17 dB at t=1500).
- It applied a single 22.11 dB floor at every timestep. Wrong at both ends, as
  section 3 shows.
- It argued the sampling loop "amplifies the weakness" by comparing 11.8 dB and
  16.5 dB loop losses. That compares one-step reconstruction of a lightly noised
  image against full generation from pure noise, which are different tasks. The
  argument is withdrawn.
- It set aside the retrained-variant evidence (lower epsilon-MSE than the
  released model, yet only 15.7 dB samples) because the script was missing. That
  evidence is now reproduced by a committed script and is central, not
  peripheral.
- Its headline "yes for image quality" was not supported. The correct answer is
  no until the trainer is fixed.

## 7. What is still true

The sampling loop is not broken: the released checkpoint works through it at
26.80 dB and a hairless round-trip returns 34.62 dB. Our adapter is fine.

The result ceiling is unchanged. The released checkpoint is the authors' own
fully-trained model and scores macro-F1 0.713, statistically tied with classical
DullRazor (difference −0.015, 95% CI −0.049 to +0.019, Holm p = 0.40) and below
original images (difference +0.038, 95% CI +0.005 to +0.071, Holm p = 0.048,
Cohen's dz 0.05). That 0.713 carries a seed SD of 0.028, against 0.005 to 0.009
for the other arms, so treat 0.70 as an estimate, not a tight prediction.

## 8. A second reproducibility gap, found while checking the fix

The `Creleased` images were sampled at 25 steps and score 26.91 dB on validation
pairs. The committed code cannot regenerate that. `scripts/28_respacing_check.py`
tests the candidates on 25 validation pairs, released checkpoint, fixed seed:

| Mode | 25 steps | 50 steps | 250 steps |
|---|---:|---:|---:|
| Schedule truncation | 11.44 | 13.16 | 18.20 |
| My DDIM respacing | 23.23 | 24.23 | 20.87 |
| `jsd` schedule | non-finite | non-finite | non-finite |
| **Recorded Creleased** | **26.91** | 26.73 | 26.59 |

None reproduces it. The recorded values also *decrease* with more steps, while
mine increase then decrease, so it is a different sampling routine, not a
parameter difference. **The 25-step respacing implementation that produced the
Creleased images is not in the repository.**

What this does and does not affect:

- The Creleased images on the volume are real artefacts, and the measurements
  taken from them stand. The 0.713 macro-F1 and 27.08 dB are properties of those
  images.
- What it affects is **regeneration**. The earlier "$1 to regenerate 8,993
  images" figure assumed working 25-step sampling. Until that routine is
  recovered, regenerating costs the 2,000-step figure, about $26.
- My DDIM implementation is a genuine improvement over truncation (23.23 versus
  11.44 dB) and is kept, but it is not the original method and should not be
  described as reproducing it.

## 9. Next step

The trainer is fixed and pinned by tests. Before spending on a long run, do the
two cheap checks that confirm the correction:

1. Fine-tune the released weights with the fixed trainer for about 1,000
   iterations. If PSNR collapses from 26.8, the trainer is still wrong.
2. Train 8 pairs briefly with the condition set to the clean target. This is a
   copy task and should be learned within a few hundred iterations.

Only after those pass is a 100k run meaningful, checkpointed every 10k, with the
rule stated in advance: continue to 660k only if the sampler-encoding error drops
below the `t/T` error, PSNR gains at least 3 dB, and the result clears the
noisy-input baseline by at least 3 dB.

Separately, recover or rebuild the 25-step sampling routine that produced the
Creleased images. Until then, treat the Creleased arm as recorded evidence that
cannot be regenerated, and price any regeneration at 2,000 steps.
