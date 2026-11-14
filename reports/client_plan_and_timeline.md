# Plan and timeline for the remaining work

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

Durations are my planning estimates in working days from approval of each step, not commitments; GPU time is from measured speeds (`reports/compute_plan.md`), with Modal prices unverified. Findings are hypotheses to test: a null result, honestly reported, is an acceptable outcome of every step below.

## Where each of your five asks stands

| Ask | Status | What remains |
|---|---|---|
| 1. Diffusion diagnosis and reproduction | Cause of the first failure found; sampling steps checked; released checkpoint run on our pairs (about 27 dB). **Paper numbers not reproduced.** | DERMAHAIR (blocked on a Kaggle login); learning curve for iteration count (Milestone 2) |
| 2. Regenerate, retrain, re-evaluate | Done once for the rebuilt arm (Cv2), same split and seeds. Cv2 is exploratory; intervals include seed variation. | Repeat only if Milestone 2 shows a measured gain |
| 3. Explainability | Reported, but the scoring needs correction (`milestone1_diffusion_diagnosis.md` section 5) | Milestone 3 |
| 4. Paper | **Not started.** The uploaded document is still the Part 1 draft. | Milestone 4 |
| 5. Deliverables | Results report done, with the explainability caveat. | Paper, Colab notebooks, exported checkpoints, figures (Milestone 5). The Word/PDF reports still carry the superseded explainability reading and are not to be sent yet. |

## Milestones

**Milestone 1 (this package, ready for review).** `reports/milestone1_diffusion_diagnosis.md`.

**Milestone 2: diffusion (3 to 5 days).**
- *2a DERMAHAIR reproduction (0.5 day after access).* Run the released checkpoint at the released settings (2,000 steps) on its 700 test images, about 1 GPU-hour. Needs the Kaggle login or dataset copy.
- *2b Learning curve (about 2 days, about 3.1 GPU-hours).* Continue the repaired trainer from its 50,000-iteration checkpoint to 200,000, with snapshots at 100,000, 150,000 and 200,000, and score each on the validation synthetic pairs with the same sampler. Written down before running: iterations count as a lever only if the 200,000 snapshot beats the 50,000 one on validation PSNR with a paired 95% interval that excludes zero. Test data stay locked. I will not state that more training helps unless this shows it.
- *2c Conditional (about 2 days).* Only if 2b shows a gain: refine on real strands, regenerate the cleaned images, retrain the ViT on the same split and seeds, and re-evaluate, about 1 GPU-hour. If not, the null result goes in the paper and no retraining happens.

**Milestone 3: explainability (2 to 3 days).**
- Re-register the lesion masks to the model's frame, with tests (code already added).
- Report both overlap definitions: share of the lesion covered, and share of salient pixels inside the lesion (the proposal's wording).
- Add the centroid-distance and off-centre analyses, and compare against the centred-Gaussian baseline on the corrected masks.
- Grad-CAM layer: the paper and code use the patch-embedding grid, which is seed-unstable. Compare it with a late-block layer on validation images only, choosing by cross-seed agreement; the layer chosen is the one named in the paper.
- Written addendum before any test-set re-scoring: this is a second look at the same test images, justified as a measurement correction and disclosed as a deviation. Original numbers stay in the report.

**Milestone 4: paper (4 to 5 days).** Rewrite list below. Every table and figure regenerated from `artifacts/`; every number traced to a file.

**Milestone 5: handover (2 to 3 days).** Exported checkpoints, notebooks, final reports, inventory, acceptance checks. The human review of real-photo hair removal still needs a person to rate 50 images, and the paper says so until it is done.

Total: about 12 to 17 working days from approval if Kaggle access arrives early. Without it, 2a is documented as not reproduced and the rest proceeds.

## What I need from you

1. A Kaggle API login or a copy of DERMAHAIR.
2. Approval for the 2b run (about 3.1 GPU-hours; the provisional spend cap recorded in decision D10 is $250).
3. Where the checkpoints and notebooks should be delivered; the ViT checkpoints are about 1 GB each.
4. A person to rate the 50 blinded photos.
5. Whether the paper should name Modal rather than Google Colab (decision D14).

## Paper rewrite list (checked against the uploaded draft)

| Place in draft | Problem | Change |
|---|---|---|
| Abstract | The "Expected Results" paragraph states hoped-for results; "improves not only classification performance but also interpretability" is not supported. | Replace with measured results; remove the claim. |
| Introduction, contributions | Three input conditions; the study has A, A* (128 px control), B, C (first run, defective trainer), Creleased and Cv2. | State the arms actually run. |
| Introduction, XAI sentence | Reference [19] (Abnar and Zuidema, attention flow, ACL 2020) is cited for "regulatory acceptability". It is not about regulation. | Cite a source on regulation or drop the claim. |
| Section 2.6 | "its visual faithfulness has been validated on diagnosis and attention-verification tasks" has no citation. "no prior study has quantified ..." is an absolute claim. | Cite or remove; soften to what the search covered. |
| Section 3.3 | Resolution and sampler absent. | State 128 px training and the partial-noise sampler; state the repaired noise-level encoding. |
| Section 4.3 | Draft says rotation up to ±90°, zoom and shift ±15%, class-balanced sampling. `configs/vit.json` has rotation 15°, random resized crop scale 0.75 to 1.0, flips, colour jitter, and inverse-frequency loss weights only. | Describe what was run; check the sampler claim. |
| Sections 3.4 to 3.5, 5 | Grad-CAM described as on the patch-embedding grid "or the selected transformer block". | Name the layer used after Milestone 3. |
| Section 5 (explainability protocol) | Lesion mask "obtained by a standard segmentation network". Masks used are the HAM10000 lesion segmentations (Harvard Dataverse). | Correct the provenance. |
| Section 3.6 | Hypotheses written as expected outcomes. | H1 not supported, H2 not supported (no gain shown), H3 uninformative as measured and under re-measurement. |
| Section 5 and the closing paragraph | "Google Colab" appears in three places; "so that the client can run and present the experiments independently". The phrase "to be agreed by the client" is not in the uploaded text. | Align to the environment actually used and the delivered notebooks. |
| Results, discussion, conclusion | Not written. | Write from the final tables, with intervals that include seed variation. |

## Notes on the notebooks and checkpoints

The pipeline ran on Modal, so the existing notebooks in `notebooks/` were not run in Colab. I can check them on a Modal GPU container, but only you can confirm they run in Colab. The ViT and diffusion checkpoints are on the Modal volume and have not been exported (the inventory lists them as skipped for size).
