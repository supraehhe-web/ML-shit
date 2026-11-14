# Arm C root-cause hypotheses (written before the local diagnostics run)

Status of this file: predictions below were committed **before** `scripts/31_local_diagnosis.py`
was executed. Verdicts are appended in a later commit under "Results", never edited in place.
Only validation pairs are used. The 186 test pairs are not read.

## What is already observed (inputs to the predictions)

| Fact | Source | Tag |
|---|---|---|
| Released checkpoint, 2,000 steps, 100 val pairs: 26.80 dB / SSIM 0.684 (hairy floor 22.11) | `reports/arm_c_diagnosis.md` | Observed |
| Old Arm C (inverted noise-level encoding), same set: 14.99 dB / SSIM 0.079, 15x reference high-frequency energy | same | Observed |
| Fixed-encoding Arm C, 20k its, **8** val pairs, 2,000 steps: 14.11 dB / SSIM 0.594 (hairy floor 20.08 / 0.712) | `gate3_smoke_eval.json` | Observed (n=8) |
| Released on the same 8 pairs: 26.64 dB / SSIM 0.815 | `control_released_eval.json` | Observed (n=8) |
| Upstream: Adam 1e-4, batch 4, L1 mean, **no gradient clipping, no EMA**, 725,000 its at epoch 1083 (about 669 it/epoch, so about 2,700 pairs) | `third_party/dmahr/config/*.json`, `model/model.py` | Observed |
| Ours: Adam 1e-4, batch 4, L1 mean, **grad clip 1.0**, no EMA, 20,000 its, 654 train pairs | `src/dm_ahr_vit/diffusion/train.py` | Observed |

Observation O1: fixing the encoding moved SSIM from 0.08 to 0.59 but PSNR stayed near 14 dB.
That is the signature of right-looking structure with a large global intensity error, not of noise.
It is a prediction target below, not yet a finding.

## Register

| ID | Hypothesis | Prediction if true | Test | Verdict |
|---|---|---|---|---|
| H1 | Encoding mismatch (t/T vs sqrt alpha-bar) explains the full sampling gap | Fixed model reaches released-level sampling PSNR | already run: Gate 3 | pending write-up |
| H2 | Conditioning wiring (order, channels, zeroed/dropped) | Zeroing or shuffling the condition changes eps error by the same factor at every t | condition ablation, `profile` | pending |
| H3 | Data pipeline (misaligned/swapped pair, range, split) | Hair-mask region carries the pair difference; range [-1,1]; no train/val group overlap | data checks in `tests/test_trainer_sampler_contract.py` | pending |
| H4 | Loss/target parametrisation | Our loss differs from upstream `p_losses` at fixed seed | differential test | pending |
| H5 | EMA | Upstream uses EMA and we do not | read upstream config: `ema_scheduler` is marked "not used now" | pending |
| H6 | Optimiser: clipping, samples seen, data size | Gap concentrates where the learning signal is weakest (high noise) | `profile` eps-error vs t | pending |
| H7 | Sampler algebra / schedule indexing | Oracle-epsilon sampler fails to reconstruct x0 | oracle test | pending |
| H8 | Undertraining (measured remainder only) | Released beats ours by a margin that grows with t | `profile` | pending |
| H9 | Evaluation harness | Metric function disagrees with known corruptions | metric tests | pending |

## Specific predictions

- **P-H7**: with the true epsilon fed to the posterior step, the final sample equals x0 to within 1e-3.
- **P-H4**: our per-step loss equals upstream `p_losses` / numel at identical seeds to float precision.
- **P-H2/H8 (profile)**: released shows a condition effect (eps error with condition zeroed over eps
  error with condition kept) of about 3x or more at t = 1000 and 1500. If the new model's ratio is
  near 1 at t = 1500 only, the cause is a weak learning signal at high noise, not wiring, since wiring
  does not depend on t.
- **P-colour (sampling, 1 pair)**: if the sampling failure is a global tone error, the new model's
  output scores at least 5 dB higher after per-channel mean matching, and its low-frequency error
  dominates its high-frequency error. If it is texture noise, mean matching changes PSNR by under
  1 dB and high-frequency error dominates.
- Falsifier for "undertraining": the new model's eps error at high t is already within 10 percent
  of the released model's. Then the gap is not in the one-step quantity this trainer optimises and
  the cause lies elsewhere (sampler interaction, data domain).

## H10 (added 13:45 UTC, before the test below is run): sampler precision

New observation O2. The fixed-20k model, sampled on CPU in fp32 for one validation
pair (`synth_val_ISIC_0024337`, seed 42, 2,000 steps), scores **28.17 dB / SSIM 0.919
against a 19.54 dB hairy floor** under the Gate-3 scoring (`scripts/34_rescore_fullres.py`,
`artifacts/diagnosis/sample_new_fixed_20k_pair0.rescore.json`). The earlier GPU run on 8
pairs scored 14.11 dB / SSIM 0.594 with **bf16 autocast, batch 8**. Tag: Observed, n=1
against n=8; the two runs differ in precision, device, batch size and pairs.

H10: bf16 autocast over 2,000 sampler steps degrades this model's output (the released
model tolerates it: 26.64 dB in bf16, n=8).

- Test: `scripts/30_gate3_eval.py --pairs 24 --precision fp32` and `--precision bf16`, same
  checkpoint (`best_gen.pth`, fixed 20k), same 24 validation pairs, seed 42, 2,000 steps.
- Prediction if H10 is true: fp32 margin is at least 5 dB above bf16 margin, and fp32
  margin is at least +2.0 dB.
- Prediction if H10 is false: fp32 and bf16 agree within 1 dB and both are near the
  earlier 14 dB, meaning pair 0 was an easy case and the continuation in
  `reports/prereg_continuation_arm_c.md` stays justified.
- Anything in between is reported as it is and read as partial.
- Rule: if fp32 passes, the continuation is **not** launched.

### H10 results so far (13:55 UTC), and a correction to how H10 was tested

All GPU runs below: A100, fixed-20k `best_gen.pth` (sha256 prefix 4269a678, identical to the
local CPU copy), seed 42, 2,000 steps, validation pairs only.

| Run | Result | Tag |
|---|---|---|
| 24 pairs, "fp32" vs bf16 | 14.96 dB vs 15.06 dB mean PSNR; floor 21.30 | Observed |
| pair 0 alone, "fp32" | 9.64 dB (floor 19.54) | Observed |
| pair 0, 16 noise draws, "fp32": fixed-20k | 6.4 to 20.0 dB, mean 11.84 | Observed |
| pair 0, 16 noise draws, "fp32": released | 25.2 to 27.4 dB, mean 26.26 | Observed |
| pair 0, CPU true fp32, one draw: fixed-20k | 28.17 dB, SSIM 0.919 | Observed (n=1) |

**Correction.** The two GPU "fp32" runs were not true fp32. PyTorch enables TF32
convolutions on Ampere by default, so "fp32" and bf16 were both reduced-precision
paths, and agreeing with each other says nothing about true fp32. The statement above
that H10 is "falsified" is withdrawn; H10 is **untested for true fp32 on the GPU**.
The only true-fp32 run is the CPU one, a single draw.

What the data do show: sampling is unstable for the fixed model (6 to 20 dB across
draws of the same pair) and tight for the released model, and the one CPU draw is
outside the GPU range. Next test, committed before running: `--no-tf32` (true fp32
on GPU), 16 draws of pair 0. Prediction if numerical precision drives the instability:
tight spread near 28 dB. Prediction if not: spread like the TF32 runs, in which case the
CPU draw was a tail event and the continuation in
`reports/prereg_continuation_arm_c.md` stays justified.

## Results (14:50 UTC, while the 20k to 100k continuation is still running)

All CPU, validation pairs only, scripts committed before running (`31_local_diagnosis.py`
at 554d4c5, `32_gradnorm.py` at 66d10d4). One-step quantities, 12 pairs, one noise draw.

**Learning curve of the fixed model, eps L1 error (lower is better); released for reference.**

| t | fixed 20k | fixed 40k | released | Reading |
|---:|---:|---:|---:|---|
| 50 | 0.1097 | 0.0851 | 0.2008 | improving, already better than released |
| 200 | 0.0368 | 0.0268 | 0.0799 | improving |
| 500 | 0.0183 | 0.0152 | 0.0384 | improving |
| 1000 | 0.0102 | 0.0099 | 0.0118 | flat, at released level |
| 1500 | 0.0098 | **0.0098** | 0.0048 | **flat over 20k iterations; 2x released** |
| 1900 | 0.0098 | **0.0095** | 0.0038 | **flat; 2.5x released** |

Condition effect at t=1500 stays 1.23 (released 3.43); at t=1900 0.94 (released 1.35).
Artifacts: `artifacts/diagnosis/profile_20k.json`, `profile_40k.json` (40k snapshot sha256
prefix 109027bf). Observed.

At t=1500 the sampler's x0 estimate multiplies eps error by about 17 (and about 93 at
t=1900), so an eps error of 0.0098 costs about 20 dB of one-step PSNR where the released
model's 0.0048 costs 25.6 dB. Inferred from the schedule, consistent with the table.

**Prediction check.** Undertraining predicted that high-t error would fall across
snapshots. Over 20k to 40k it did not (0.0098 to 0.0098); low-t error did fall. The
pre-registered rule at the 50k snapshot (stop if t=1500 error is still at or above 0.0095)
is on track to fire. Verdict on H8 so far: **not supported at this budget**; not rejected
for a larger budget.

**H5 (EMA): cleared.** Upstream's `ema_scheduler` is marked "not used now" in every
vendored config. Observed in `third_party/dmahr/config/*.json`.

**H6 (optimiser), gradient norms** (`artifacts/diagnosis/gradnorm.json`, train mode, 2
draws per level, 4 levels, CPU; one draw non-finite from the CPU NaN issue in D29):

| Model | Gradient norm range | Draws above the clip of 1.0 |
|---|---|---:|
| fixed 20k | 0.72 to 1.88 | 62.5% |
| released (725k) | 0.18 to 0.40 | 0% |

Our trainer clips at 1.0 and upstream does not, and the clip engages on most draws of our
model. The two models are at different training stages, so the norm gap alone does not
show that clipping slows learning. Observed (n=8 per model); the effect of clipping is
**Unknown**. This makes clipping the next single change in the continuation plan.

### Update (15:05 UTC): 50k snapshot and snapshot averaging

Same procedure and 12 pairs as above (`profile_50k.json`, `profile_avg_30k_50k.json`;
`scripts/35_average_snapshots.py` and the `--only` filter committed at 7f76b08 before use).
eps L1 error by t (50 / 200 / 500 / 1000 / 1500 / 1900):

| Model | 50 | 200 | 500 | 1000 | 1500 | 1900 |
|---|---:|---:|---:|---:|---:|---:|
| fixed 20k | 0.1097 | 0.0368 | 0.0183 | 0.0102 | 0.0098 | 0.0098 |
| fixed 40k | 0.0851 | 0.0268 | 0.0152 | 0.0099 | 0.0098 | 0.0095 |
| fixed 50k | 0.0981 | 0.0278 | 0.0134 | 0.0090 | **0.0086** | 0.0101 |
| average of 30k, 35k, 40k, 45k, 50k | 0.0912 | 0.0255 | 0.0126 | 0.0090 | 0.0089 | 0.0090 |
| released | 0.2008 | 0.0799 | 0.0384 | 0.0118 | 0.0048 | 0.0038 |

- **Early-stop rule (t=1500 at or above 0.0095 at the 50k snapshot): did not fire**, 0.0086.
  The job continues. Observed. One noise draw on 12 pairs, so a 12% change between
  snapshots is within what measurement noise could produce; the trend is not established.
- **Snapshot averaging** lowers t=1900 from 0.0095 to 0.0090 and leaves t=1500 near 0.0089.
  The floor stays at about 0.009, about twice the released model at t=1500 and 2.4 times
  at t=1900. Weight jitter is therefore at most a minor part of the floor. Observed.
- The floor is nearly the same at t=1000, 1500 and 1900 (0.009 to 0.010) while the
  released model's error falls with t. Inferred: at high noise the network must reproduce
  the input noise almost exactly (eps close to x_t), and that precision, not the
  noise-level signal, is what is missing.

## Rules in force

1. Every script is committed before it runs; artifacts record git commit, config hash, checkpoint
   hash and manifest hash.
2. One change at a time. A fix is confirmed only if the failing metric recovers, a regression test
   passes, and reverting the fix restores the failure.
3. Conflicting evidence is reproduced or explained, never dropped.
4. Attribution is quantified: dB closed out of the dB gap; the rest is reported as unexplained.
5. One-step floors are the noisy input's own PSNR, not the hairy-input PSNR.
