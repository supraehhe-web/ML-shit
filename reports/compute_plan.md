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

