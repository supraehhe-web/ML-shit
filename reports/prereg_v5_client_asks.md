# Pre-registration v5: the client's five asks (committed before the runs below)

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

Scope: the client's five asks only (diffusion root cause and reproduction; regenerate, retrain, re-evaluate; explainability; paper; deliverables). No new stage. The owner set an autonomous spend limit of **$40** for this task (inside the protocol cap in `configs/protocol.json`). Spend is logged as GPU-seconds times the per-second price listed on modal.com/pricing (fetched 2026-11-13; section 6) and labelled an estimate.

## 1. Explanation re-measurement (corrects D39)

**Why.** `reports/stageE_results.md` section 6 documents that masks were not in the model's frame. This is a measurement correction, not a new hypothesis. It is a second look at the same 1,502 test images with the same frozen classifiers, authorised by the owner and recorded as a deviation from "test set touched once per frozen checkpoint" (decision D40). The old CSVs stay in the repository unchanged and labelled `legacy_frame`.

**Geometry.** One function, `src/dm_ahr_vit/xai/geometry.py`: the mask goes through the arm's stored size (A and B 600 x 450; A\*, C, Creleased, Cv2 128 x 128), then the classifier's own resize to `int(224 * 1.14)` and 224 centre crop. The classifier's evaluation transform calls the same operations. Tests check alignment per arm (`tests/test_xai_registration.py`).

**Metric definitions** (generated from the function docstrings, `xai.batched.metric_definitions`): overlap = mask recall (share of the lesion covered by the top 20% most salient pixels); precision = share of the top 20% that is inside the lesion; pointing = the single most salient pixel is inside the lesion; centroid distance. Recall is the primary overlap endpoint; precision is reported and labelled.

**Maps.** Attention rollout (unchanged algorithm, batched; equality with the single-image function is tested). Grad-CAM primary target: the output of `ln_1` in the last encoder block, class token dropped, tokens reshaped to 14 x 14. The patch-embedding grid used in the recorded results is rerun as a labelled **sensitivity** run. The primary target was set by the owner; before the test run the cross-seed agreement of both targets is measured on **validation** images (arms A, A\*, B, Cv2; seeds 42, 123, 2026) and reported whatever it shows.

**Models and seeds.** Frozen checkpoints only. A, A\*, B, Cv2: all eight seeds (42, 123, 2026, 7, 11, 17, 23, 31); the extra seeds had no maps before. C and Creleased: three seeds. No new training.

**Baselines, per metric, on the same masks in the same frame:** (i) random pixel maps and random 14 x 14 patch maps, 100 draws per image; (ii) the centred Gaussian (sigma = 0.25 of the width); (iii) pointing chance = mean mask area for random maps, and for the centre-fixed Gaussian the share of masks containing the image centre; (iv) the overlap ceiling per arm: mean of min(1, 0.2 / mask area) for recall, and mean of min(1, mask area / 0.2) for precision. Overlap and pointing baselines are reported separately.

**Lesion position.** Mean and SD of lesion-centroid offset from the image centre, and the share of masks containing the centre, per arm geometry. **Off-centre robustness:** image and mask are translated together by a random integer shift of up to +/-30% of the width and height (zero fill, shifts fixed per image id so every arm sees the same shifts) and the maps are recomputed on the shifted input.

**Statistics.** Per arm and seed, the mean of each metric over the 1,502 images. Between-arm comparisons: A-B (planned), and A-A\*, A-Cv2, B-Cv2, Cv2-A\* (exploratory). For each: per-seed differences (seeds paired by index) shown next to a two-level bootstrap (10,000 replicates, seed 20261113: lesion groups resampled jointly, seeds resampled within each arm), raw p, Holm p within family. **Strike rule (fixed now):** a (map, metric) row of a comparison is struck as not interpretable when the larger of the two arms' seed ranges (max minus min of the seed means) exceeds the absolute difference of the arm means. Struck rows stay in the tables, marked.

## 2. Reproduction against the paper (DERMAHAIR)

Search order before calling it blocked: Hugging Face (`AnasHXH/SR_SDM`, `AnasHXH/SR`, dataset and model searches), the repository README and paper links, other mirrors. Kaggle only if a token is supplied. **If obtained:** the released checkpoint through the adapter at T = 10 and T = 2000 on the dataset's hairy/hairless pairs; report hairy-input floor, PSNR, SSIM and LPIPS. **Rule:** reproduced = within 2 dB of the paper's 41.04 dB and above the hairy-input floor; otherwise "not reproduced", with the evidence. **If not obtained:** state exactly what was tried, and build a proxy from the paper's described synthetic-hair method, labelled a proxy. The paper's PSNR is context, not a target.

## 3. Learning curve (the only open root-cause question: does more training help?)

- Start: the repaired-trainer run, iteration 56,000 state (`checkpoints/dmahr/train_state.pt`), copied to a new directory so no recorded checkpoint is overwritten. Same data (train-split synthetic pairs), same loss, same sampler. Snapshots every 5,000 iterations; the curve is read at 25k, 50k, 75k, 100k and, if the stop rule allows, 125k to 200k.
- At each snapshot, **validation pairs only**: (a) one-step epsilon error over t in {200, 500, 1000, 1500, 1900}; (b) condition-usage ablation (kept, zeroed, shuffled) at t in {200, 500, 1000, 1500}, 48 pairs; (c) partial-noise sampling at t0 = 600 on 50 pairs; (d) pure-noise sampling at 2,000 steps on 24 pairs. The hairy-input floor is computed on the same pairs in the same run.
- **Stop rule (fixed now):** continue past 100,000 only if both hold at the 100k snapshot: (1) the high-noise one-step error (mean of t = 1500 and t = 1900) is at least 10% lower than at 50k, and no higher than at 75k; (2) pure-noise sampling at 2,000 steps beats the hairy-input floor by at least 2 dB on the 24 pairs. Otherwise stop at 100k and report. Reaching 200k does not by itself count as a result.
- **Clip comparison:** one run from the same 56k state to 100k with gradient clipping removed (upstream does not clip), same data order not guaranteed, single run: a difference smaller than the snapshot-to-snapshot movement of the clipped run is reported as not distinguishable from run-to-run noise.
- **Contract tests first:** `tests/test_trainer_sampler_contract.py`, `test_noise_level_convention.py`, `test_trainer_resume.py` must pass (locally and, for the two that need the manifest, on Modal).
- If a snapshot shows a gain, regenerating Cv2-style images and retraining the ViT is a separate step with the same split and seeds; it is not pre-committed here.
- Throughput is re-measured on the A100-80GB before any job expected to cost more than $5.

## 4. Statistics and reports

Cv2 and Creleased rows are labelled exploratory. The standardised dz is replaced by the pooled difference plus the per-image log-loss difference. Seed-aware intervals show the per-seed differences. Generated reports carry a stale-phrase check as a gate. No number in a report is edited by hand: the generator is fixed and the report regenerated.

## 5. Not decided by results

No hybrid arm, no tuning toward the reviewer's ranges, no change to the frozen ViT recipe, split or seeds.

## 6. Spend log

`reports/spend_log.csv`: job, GPU, seconds from the run registry or the job log, estimated price per GPU-hour (A100-80GB 2.4984, A100-40GB 2.0988, L4 0.7992, A10 1.1016 per GPU-hour, from modal.com/pricing fetched 2026-11-13; A10G is not listed separately, so the A10 price is used), estimated cost, running total against the $40 limit.
