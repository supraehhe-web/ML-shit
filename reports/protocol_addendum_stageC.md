# Stage C protocol addendum (committed before any test-split use of the new diffusion model)

Written after Stage B (`reports/prereg_v2_diffusion.md`, decision D34). Everything below is frozen at the commit that adds this file. Nothing here was chosen after seeing a test-split result for the new model.

## What is new and what is not

- **New arm `Cv2`.** Hair-present images (frozen screening, coverage of at least 1%, 8,993 of 10,015) are replaced by the 128 px output of the Stage B fine-tuned diffusion model. Hair-absent images are the original, as in every arm.
- **Unchanged.** Cohort, frozen 70/15/15 split, ViT-B/16 recipe, augmentation, seeds 42/123/2026, evaluation code, endpoints, bootstrap settings. Arms A, B and A* keep their recorded checkpoints and predictions. The earlier arm `C` (trainer with the inverted noise level, 2,000-step sampling from pure noise) and `Creleased` stay in the record untouched and are labelled as such in every report.

## Frozen components

| Item | Value |
|---|---|
| Diffusion checkpoint | `artifacts/checkpoints/dmahr_v2/final_gen.pth`, sha256 `dd06c2163e93c5348515ad9ac69f8858b2dd2e5154380471014bcde1b33e7130` (fine-tuned 20,000 iterations from `gen_iter050000.pth`, sha256 `77031da6dbea73ad751adf9aaddf6c7cfb2cd762582eb1c8b2bb3956d188b845`) |
| Sampler | strided DDIM over the trained 2,000-step schedule, 50 steps, start from the input noised to schedule index **600**, fp32, 128 px |
| Noise seeds | derived per batch from `sha256("<shard>:<batch start>")`, as in `scripts/05_dmahr_sample.py` |
| Input set | hair-present images of all splits, produced by `modal_app.py::cv2_sample_driver` |
| Synthetic test pairs | the 186 held-out pairs, processed with the same settings (`--pairs-split test`) |
| Training recipe | `scripts/06_train_vit.py --arm Cv2 --seed {42,123,2026}`, identical to the other arms |

Method deviations from the proposal, all recorded: the sampler starts from the noised hairy input rather than pure noise (Stage A); the training hair comes from harvested real strands rather than an external hair dataset (Stage B); the Stage B gate was not met (D34).

## Checks before any test-split use (validation data only)

1. Run the production sampler on the 182 validation pairs (`cv2_sample` with `--pairs-split val`) and score with `scripts/07b_image_quality.py --split val --c-arm Cv2`. Pass: mean PSNR margin over the hairy input of at least +5 dB and no non-finite output. Fail: stop, find the cause, do not touch test data.
2. Count the Cv2 PNGs: one per image of the 10,015 (hair-present from the sampler, hair-absent copied), no missing files, no non-finite flags in the shard logs.

## Test-split touches, each once per frozen checkpoint

- 186 synthetic test pairs: scored once for the Cv2 outputs (PSNR and SSIM against the hairless reference, against B and against the hairy input).
- 1,502 test images: each of the three Cv2 ViT checkpoints is evaluated once (`07_eval_vit.py`), plus the explanation analysis on the X1 mask subset (`08_xai.py`).
- No retraining, resampling or checkpoint swap after a test result is seen. If a bug is found afterwards the run is repeated in full and both are reported.

## Endpoints and comparisons (same definitions as `experiment_protocol.md`)

- Planned family with Holm correction per endpoint and subset: **A vs B, A vs Cv2, B vs Cv2**, for test macro-F1 (full and hair-present subsets), paired PSNR and SSIM, Grad-CAM and attention-rollout lesion overlap.
- Secondary controls, own Holm family: A vs A*, **Cv2 vs A*** (holds the 128 px round trip constant), Cv2 vs B and Cv2 vs hairy input on paired image quality.
- Run with `scripts/09_stats.py --c-arm Cv2`; outputs carry the suffix `_Cv2` so the earlier statistics files are not overwritten.

## Reading the result

- A difference counts as real only with raw p, Holm p, effect size and CI reported, Holm p below 0.05 and the CI excluding zero; a clear p-value with |d_z| below 0.2 is flagged negligible.
- A null or negative result is a finding and is reported as one. Nothing in the reports may say hair removal was validated; the permitted wording is in D34.
- Image-quality numbers are on the project's synthetic hair only. The classification result is on real images.
- A* is the main control for any Cv2 effect: a Cv2 gain or loss that A* shows too is a resolution effect, not a hair effect.

## Cost and time

Sampling about 9,000 images on four A100-80GB containers, three ViT trainings of 5 to 10 minutes each in parallel (from `run_registry.jsonl`), statistics on CPU. Estimated total about $3 at an unverified $2.5 to $3 per GPU-hour. No spend cap was set by the owner.
