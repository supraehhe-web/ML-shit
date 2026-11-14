> **SUPERSEDED (2026-11-13).** This document concludes that Arm C's failure is
> undertraining and that the inference path is sound. That conclusion is
> retracted: the training script gave the network an inverted noise-level signal
> (see `reports/oneshot_diagnostic_report.md` and `reports/hypotheses.md`). Cost
> figures and any "repair" plan below assumed the old script and no longer apply.
> The text is kept unchanged below for the record.

# Arm C diagnosis

**Outcome: (b) undertrained.** The inference path is sound and the released DM-AHR checkpoint
behaves sensibly through it. The checkpoint trained for this study does not. Arm C's poor result is
a property of that checkpoint, not of the adapter, the value range, the channel order or the
alignment.

All measurements below use **validation-source synthetic pairs only**. The 186 test pairs were not
read, so no test-set choice depends on this diagnosis.

## 1. Identity test — does a hairless image survive the inference path?

The strongest single check: pass a genuinely hairless reference through the exact Arm C path (same
adapter, resize, value range, sampler) and compare it with itself. A working path returns something
close to the input.

| Steps | n | PSNR (dB) | SSIM | high-frequency ratio (out/in) |
|---|---|---|---|---|
| 1,000 | 50 | **34.62** ± 6.18 | **0.990** ± 0.004 | 0.926 |
| 2,000 | 50 | **41.72** ± 4.89 | **0.992** ± 0.001 | 0.909 |

The path preserves the image almost exactly, and retains 91–93% of its high-frequency energy. A
value-range error, a channel swap or a broken sampler would all fail this test. **The inference path
is not the defect.**

## 2. Head to head — released checkpoint versus ours

100 validation pairs, both checkpoints, both step counts, identical adapter.

| Condition | PSNR (dB) | SSIM | Hairy input PSNR | Beats its own input? |
|---|---|---|---|---|
| Released, 2,000 steps | **26.80** ± 2.78 | **0.684** ± 0.098 | 22.11 | **yes** |
| Released, 1,000 steps | **24.66** ± 2.67 | **0.641** ± 0.101 | 22.11 | **yes** |
| Ours, 2,000 steps | 14.99 ± 1.23 | 0.079 ± 0.008 | 22.11 | no |
| Ours, 1,000 steps | 14.12 ± 1.22 | 0.073 ± 0.006 | 22.11 | no |

The released checkpoint clears the hairy-input floor by 2.6–4.7 dB and reaches SSIM 0.64–0.68. Our
checkpoint sits 7–8 dB **below** the floor with SSIM near 0.07, which is essentially no structural
correspondence at all.

The same adapter, the same pairs, the same step counts. The only difference is the weights.

## 3. Defect checks

| Check | Result | Reading |
|---|---|---|
| Value range | output min/max within 0–255 after postprocess | no range error |
| Channel order | per-channel means track the reference within noise | no RGB/BGR swap |
| Spatial alignment | best shift 0 to −1 px; flip correlations ≈ 0 | no flip or offset |
| High-frequency energy | ours 0.269 vs reference 0.018 | **excess** high-frequency energy, not smoothing |
| Global pixel std | ours 0.127 vs reference 0.166 | mild collapse, not the dominant effect |

## 4. The "over-smoothed" description was wrong

The earlier reports described Arm C's output as over-smoothed. The measurements do not support that
word. Our output carries **15× the reference's high-frequency energy** (0.269 against 0.018) while
its global standard deviation is only modestly reduced (0.127 against 0.166). That combination —
low contrast with high pixel-to-pixel variation — is the signature of a model producing noise around
a compressed range, not of a model producing a blur.

The correct description is: **the checkpoint has not learned the hair-removal mapping.** Its output
is noisy and structurally unrelated to the target (SSIM 0.07). "Over-smoothed" implied a model that
had learned to remove detail; this model has not learned the task.

## 5. What this means for the study

**H1 (image quality) — not supported, and the test is valid.** The released checkpoint demonstrates
that this adapter can produce sensible hair removal (PSNR 26.8, SSIM 0.68). Our checkpoint cannot.
The comparison against Arm B therefore measures an undertrained model, and the honest statement is
that H1 is untested for DM-AHR at a realistic training budget.

**H2 (classification) — not supported, same caveat.** Arm C's macro-F1 of 0.426 follows from its
image quality. It is a statement about this checkpoint.

**H3 (explainability) — not supported, and the measurement is uninformative.** Independently of
Arm C, a centred-Gaussian prior beats every arm on both overlap (0.644) and pointing (0.953), so no
arm demonstrates lesion localisation beyond a centre bias. Explaining a classifier at 0.43 macro-F1
would not be a fair test of H3 in any case.

## 6. What would fix it, and what it costs

The diagnosis rules out the cheap repairs. A value-range, channel-order or alignment fix would have
been correctable on disk without regenerating anything; none of those is the problem. The defect is
in the weights, so the fix is a longer training run followed by full regeneration.

| Action | Cost at measured throughput |
|---|---|
| Train to the upstream 660,000 iterations | ~$21.15 |
| Regenerate Arm C, 8,993 images at 1,000 steps | ~$11.43 |
| Regenerate Arm C at 2,000 steps | ~$22.96 |
| Retrain C × 3 seeds, re-evaluate, redo XAI | ~$0.80 |

The cheapest complete repair is roughly **$12.85** at the ablation budget and **$24.38** at
reference settings. Neither fits the remaining balance of $8.37, so Arm C stays as a labelled
under-resourced ablation.

## 7. Evidence

| Artifact | Contents |
|---|---|
| `artifacts/logs/arm_c_diagnosis.json` | outcome, verdict, identity and head-to-head summaries, alignment |
| `artifacts/logs/arm_c_identity.csv` | per-image identity test at both step counts |
| `artifacts/logs/arm_c_headtohead.csv` | per-pair metrics for all four conditions |
| `artifacts/reports/arm_c_diag_*.jpg` | visual panels: reference vs round-trip, and each condition |
| `scripts/17_arm_c_diagnose.py` | the script that produced all of the above |

Reproduce with:

```bash
modal run modal_app.py::arm_c_diagnose --identity-n 50 --pairs-n 100
```
