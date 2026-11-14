# Compute plan and measured usage

Modal prices read from `https://modal.com/pricing` on 2026-11-11 (per second):

| Resource | $/second | $/hour |
|---|---|---|
| A100 80 GB | 0.000694 | 2.4984 |
| A100 40 GB | 0.000583 | 2.0988 |
| A10 | 0.000306 | 1.1016 |
| L4 | 0.000222 | 0.7992 |
| T4 | 0.000164 | 0.5904 |

Budget constraint: the account holds **$23**.

## Why a cheaper GPU is the main lever

DM-AHR sampling peaks at **1.52 GB** of GPU memory and consists of 2,000
sequential calls to a small UNet (inner channel 64) at 128x128. That profile is
launch-overhead bound, not compute bound, so a mid-range GPU should be close to
A100 throughput at a third of the price. This is a hypothesis that still needs
measuring; GPU allocation on this account has been failing, so it is
**NOT VERIFIED**.

## Cost of the remaining work

8,993 hair-present images at the measured 4.34 s/image (A100, fp32, batch 32)
= 10.8 GPU-hours of sampling, plus ~0.5 GPU-hour for 3 ViT C trainings, 3
evaluations and 3 XAI passes.

| GPU | Sampling | ViT + eval + XAI | Total | Fits $23? |
|---|---|---|---|---|
| A100 80 GB | $27.0 | $1.3 | **$28.3** | No |
| A100 40 GB | $22.7 | $1.3 | **$24.0** | No (marginal) |
| A10 | $11.9 | $1.5 | **$13.4** | Yes |
| L4 | $8.6 | $1.6 | **$10.2** | Yes |
| L4 at 2x slower | $17.3 | $3.2 | **$20.5** | Yes, tight |

**Recommendation: run Arm C on L4.** Even if L4 is twice as slow as A100, the
total stays under $23. A10 is the fallback if L4 capacity is unavailable.

bf16 autocast is a second lever that may cut sampling time further; it is
implemented in `scripts/03b_dmahr_bench.py` but unmeasured for the same reason.

Provisional cap recorded in `configs/protocol.json`: `max_spend_usd = 250`.
The user did not supply `MAX_SPEND_USD` at project start, so this is a
documented placeholder (see decision log D10). The real constraint is the $23
account balance.

## Measured unit costs

| Unit | Measured | Source |
|---|---|---|
| DM-AHR sampling, batch 1 | 35-38 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 8 | 6.04 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 16 | 4.69 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 32 | 4.35 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 64 | 4.10 s/image | `dmahr_smoke.json` |
| DM-AHR training | ~0.055 s/iteration steady state | `dmahr_train.json` |
| ViT-B/16 epoch (7,011 train images, batch 32) | ~17 s | `train_result.json` |
| ViT-B/16 full run (early stop, 25 epochs observed) | ~420 s | `A_seed42/train_result.json` |
| Peak GPU memory, DM-AHR sampling | 1.52 GB | `dmahr_smoke.json` |

## Projected units

| Job | Units | GPU-hours |
|---|---|---|
| DM-AHR training, 20,000 iterations + validation | 1 run | ~0.35 |
| Arm C, hair-present images, batch 32 | 8,993 x 4.35 s | ~10.9 |
| Arm C, held-out synthetic test pairs | 186 x 4.35 s | ~0.22 |
| ViT training, 4 arms x 3 seeds | 12 runs x ~7 min | ~1.4 |
| ViT locked-test evaluation | 12 runs | ~0.2 |
| XAI, 4 arms x 3 seeds on 1,502 test images | 12 runs | ~1.0 |
| **Projected total** | | **~14.1 GPU-hours** |

Batch 64 is only 6% faster per image than batch 32 while using more memory, so
batch 32 is used for Arm C.

## Recorded usage

Every job appends wall-clock seconds and GPU type to
`artifacts/logs/run_registry.jsonl`. The authoritative spend figure is the sum
over that file, not this projection.


## Addendum 2026-11-13: retrain and checks (A100-80GB, Modal account motog5g987)

The run registry under-records this work: the evaluation jobs wrote 0 seconds and the continuation was stopped before it could record its time. The figures below come from job logs and are estimates. The Modal price per GPU-hour was not verified; $2.5 is the rate used in earlier estimates.

| Job | Wall-clock | Source |
|---|---:|---|
| Fixed-trainer retrain, 20,000 iterations | 1,220 s | training log |
| Continuation to 56,300 iterations (stopped by the owner) | about 3,590 s | training log |
| About 15 evaluation and replicate jobs (24 pairs or fewer, 2,000 sampling steps) | roughly 30 to 45 min in total | job logs, rough |
| Total GPU time | about 2 hours | sum |
| Estimated cost | about $5 | at $2.5 per GPU-hour, price unverified |

CPU-only diagnostics ran on the development sandbox and are not billed to Modal. The 20,000 to 100,000 plan was cut at 56,300 iterations, so the planned 1.9 GPU-hours were not fully spent.

## Addendum: Stages B and C (rebuilt diffusion arm)

Measured on one A100-80GB unless noted: strand harvest 125 s on CPU; fine-tune 682 s (20,000 iterations, one preemption resumed from the 9,000 snapshot); validation evaluations about 5 min in total; Cv2 sampling in four containers of 356 to 601 s plus 67 s for the 186 test pairs; three ViT trainings of 452 to 553 s each plus 47 to 165 s of evaluation and explanation. About 1.3 GPU-hours in total. Modal prices were not verified; at an assumed $2.5 to $3.5 per hour that is roughly $3 to $5. No spend cap was set by the owner (the provisional `max_spend_usd` is in `configs/protocol.json`).


## Addendum: Stage E (extra seeds and checks)

Measured: 20 ViT runs (train and evaluate) took 10,835 s of container time in total (3.0 GPU-hours, 40 jobs, no preemption); composite arms 76 s, condition usage 38 s, residual hair 69 s and 125 s (CPU). About 3.1 GPU-hours. Modal prices were not verified; at an assumed $2.5 to $3.5 per hour that is roughly $8 to $11.
