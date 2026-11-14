# Pre-registration: continue the fixed Arm C trainer from 20,000 to 100,000 iterations

Written and committed **before** the job is launched. Validation pairs only. The 186
test pairs are not read, and stay unread until a checkpoint passes here and is frozen.

## Question

After the noise-level encoding fix, the diffusion model still fails multi-step
sampling (14.11 dB on n=8, not yet confirmed at larger n) while its one-step
denoising beats the released checkpoint at low noise and trails it at high noise.
Does more training of the *unchanged* fixed trainer close that gap?

## What is held fixed (one change only: iteration count)

Same trainer commit lineage as `b11c70d` plus the resume path tested in
`tests/test_trainer_resume.py`. Same config except `train.iterations` is overridden
to 100,000 on the command line. Adam 1e-4, batch 4, L1 mean, gradient clip 1.0, no
EMA, 654 train pairs. Resumes optimiser and weights from the 20,000-iteration
state. The 20,000-iteration files are copied to
`artifacts/checkpoints/dmahr_fixed20k/` first, because the continuation overwrites
`best_gen.pth` and `train_state.pt`.

## Starting evidence (Observed, `artifacts/diagnosis/profile.json`, 12 val pairs, one noise draw, commit 554d4c5)

| t | released eps L1 | fixed-20k eps L1 | released one-step PSNR | fixed-20k one-step PSNR |
|---:|---:|---:|---:|---:|
| 1000 | 0.012 | 0.010 | 29.5 | 33.1 |
| 1500 | 0.005 | 0.010 | 25.6 | 20.2 |
| 1900 | 0.004 | 0.010 | 14.5 | 9.3 |

## Prediction if undertraining is the cause

Epsilon error at t=1500 and t=1900 falls steadily across the 5,000-iteration
snapshots, and full sampling follows. If the cause is something else (clipping,
data size, a defect not yet found), epsilon error at high t stays near 0.010 while
low-t error keeps improving.

## Decision rules, fixed in advance

Learning curve, measured on CPU from downloaded snapshots with
`scripts/31_local_diagnosis.py profile` on the same 12 pairs.

- **Early stop at the 50,000 snapshot.** If epsilon error at t=1500 is still at or
  above 0.0095 (less than 5% below 0.0100), stop the job. Undertraining at this
  budget is then rejected as the explanation.

Final verdict at 100,000 iterations, 100 validation pairs, 2,000 sampling steps,
raw weights, `scripts/30_gate3_eval.py --pairs 100`. The margin is PSNR over the
hairy input's PSNR on the same pairs.

| Outcome | Condition |
|---|---|
| **Pass** | margin at least +2.0 dB **and** epsilon error at t=1500 at most 0.0075 (1.5x the released model's 0.005) |
| **Middle** | margin above 0 and below +2.0 dB, or margin passes but t=1500 error does not |
| **Fail** | margin at or below 0 dB |

## Added after the H10 results, before launching (13:58 UTC)

Observed on GPU, validation pair 0 (hairy floor 19.54 dB), 16 noise draws, true fp32
(TF32 off, confirmed in the output file), 2,000 steps, `best_gen.pth` sha256 prefix 4269a678:

| Model | PSNR range over 16 draws | Mean |
|---|---|---:|
| fixed-20k | 6.5 to 20.0 dB | 11.99 |
| released | 25.2 to 27.4 dB | 26.26 |

TF32 and bf16 runs of the fixed model gave the same spread (6.4 to 20.0). Precision is
therefore ruled out. The CPU loop in `scripts/31_local_diagnosis.py` is the same
algorithm as the production sampler (read line by line), and its single draw reached
28.17 dB, so the sampler code is also not at fault. The symptom is instability across
noise draws of the same pair, which a well-trained denoiser does not show.

**Added pass condition (stability):** the 16-draw replicate on pair 0, true fp32, at
the final checkpoint must have every draw at or above the hairy floor (19.54 dB) and a
standard deviation of at most 2 dB. The released model's standard deviation is about
0.7 dB. This condition was added after seeing the data above and is stated before the
continuation starts; it tightens the pass rule and does not loosen it.

- Pass: freeze the checkpoint, record its hash, then and only then score the 186
  test pairs once and regenerate Arm C.
- Middle: continue only if the t=1500 error was still falling by more than 10%
  over the last three snapshots; otherwise treat as fail.
- Fail: stop training. Report Arm C as a failed replication, with the released
  checkpoint as the valid diffusion arm. Next single changes, one at a time and
  each pre-registered: remove gradient clipping; then enlarge the pair set.

## Cost (estimate, not verified)

20,000 iterations took 1,220 s on one A100-80GB, so 80,000 more take about 4,900 s,
about 1.4 GPU-hours. At the roughly $2.5 per GPU-hour implied by earlier estimates
this is about $3-4. Modal's price has not been verified; GPU-hours are the firm
number. Scoring the final checkpoint on 100 pairs adds about 10 GPU-minutes.

## Not claimed

Nothing here shows the continuation will pass. A pass would show the gap closes with
training; it would not show which part of the gap was budget and which was something
else, because the 20,000-iteration profile shows a deficit that is not uniform in t.

## Outcome: stopped by the project owner at 56,300 of 100,000 iterations (14:58 UTC)

The continuation was stopped on cost before reaching the final evaluation, so the pass, middle and fail rules above were **not applied**. No verdict is claimed for the 100,000-iteration endpoint.

What was measured before stopping (validation pairs only, CPU, 12 pairs, one noise draw; `artifacts/diagnosis/profile_*.json`):

| Snapshot | eps L1 at t=1500 | eps L1 at t=1900 |
|---|---:|---:|
| 20k | 0.0098 | 0.0098 |
| 40k | 0.0098 | 0.0095 |
| 50k | 0.0086 | 0.0101 |
| average of 30k to 50k | 0.0089 | 0.0090 |
| released | 0.0048 | 0.0038 |

The 50,000-iteration early-stop rule (stop if t=1500 error is at or above 0.0095) did not fire. Sampling quality was not measured at any snapshot after 20,000 iterations. Training time spent on the continuation: about 3,590 s on one A100-80GB. Snapshots up to 55,000 iterations remain on the volume (`artifacts/checkpoints/dmahr/`); the 20,000-iteration originals are in `dmahr_fixed20k/`.
