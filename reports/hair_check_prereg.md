## Pre-registration for scripts/37_real_hair_check.py (written before the script is run)

Question: does the released DM-AHR checkpoint remove REAL hair, or only the synthetic
overlays it was trained on? Data: 40 train-split images with detector hair coverage of at
least 0.05, chosen by a hash order, so no test image is read. Instrument: the frozen hair
detector at 128 px, with DullRazor (Arm B) as the positive control.

Predictions, fixed in advance:
- Control: DullRazor leaves a median of 30% or less of the detector's hair coverage and
  edits hair-mask pixels much more than the rest (inside/outside ratio above 3). If it does
  not, the detector at 128 px is not a usable instrument and the experiment is void.
- If the released model removes real hair: median fraction remaining 50% or less and an
  inside/outside ratio above 2.
- If it mostly leaves real hair: median fraction remaining 80% or more and a ratio near 1.
- Between 50% and 80%: reported as partial removal.

## Result

n = 40 train-split images (full-resolution detector coverage at least 0.05, median 0.152), chosen by hash order; no test image read. Script committed at 1d4896c before running; id-list sha256 b3992e39. Artifacts: `artifacts/diagnosis/real_hair_check.json`, `real_hair_check.csv`, `real_hair_contact_sheet.png`. The sample is biased toward heavy hair and is not representative of the cohort.

| At 128 px | DullRazor (Arm B) | Released DM-AHR |
|---|---:|---:|
| Median fraction of detector coverage remaining | 63% | 95% |
| Images with under half the coverage remaining | 9 of 40 | 0 of 40 |
| Mean absolute change inside / outside the hair mask (grey levels) | 5.78 / 0.27 (ratio 21.7) | 2.79 / 2.48 (ratio 1.13) |

**Control failed for the coverage measure.** DullRazor left 63% where the rule required 30% or less. The detector reports a median coverage of 0.448 at 128 px against 0.152 at full resolution on these images, so at 128 px it fires on texture and not only on hair. The coverage measure is void. The inside/outside ratio passed its control (21.7 against the threshold of 3), and the released model scores 1.13, which is the pre-registered signature of leaving real hair in place.

Post hoc, not pre-registered: fraction of pixels changed by more than 25 grey levels at 128 px. DullRazor median 2.5% (mean 3.8%, max 16.0%); released model median 0.0% (mean 0.4%, max 11.8%).

Visual review (agent-reviewed, 8 images; left original, middle DullRazor, right released model). In the 4 images with clearly visible real hair, the released model leaves the hair strands in place. DullRazor removes hair but replaces lesion and skin texture with polygonal patches in all 8 images, and in one image replaces the whole lesion with a pale blob (the dark lesion and surgical-marker ink were read as hair).
