# Literature and code audit (2026-11-13)

Question: what is wrong in our project, checked against the DM-AHR paper and repository, the SR3/Palette
codebase it forks, and the hair-removal literature. Tags: Observed (our artifact), Literature (source, see
reliability), Inferred, Unknown.

## What could and could not be read

| Source | Status |
|---|---|
| DM-AHR paper (Cancers 2024;16(17):2947) | Publisher and PMC blocked. Read through Europe PMC full-text XML, which is cut off before the classification results, SSIM table and limitations. Facts below are from that partial text as summarised by the fetch tool. |
| DM-AHR README | Read. No iterations, batch size, resolution or step count. |
| Janspiry SR3 issues #78 and #154, Palette README | Read. |
| Quishpe-Usca 2024, Jutte 2024 | Full text via Europe PMC. |
| Wang 2024 (hair presence, Comput Biol Med) | Search-result highlights only; the page returned 403. |
| SR3 paper hyperparameters (EMA decay, steps) | Not readable. Not verified. |

## Findings

1. **DM-AHR was trained on HAM10000.** Literature: DERMAHAIR is built from HAM10000 lesion images plus hair
   patterns from the Digital Hair Dataset; 100 images per class (700) were held out for testing and "the
   remaining images were augmented with synthetic hair" for training, about 9,300 images. Hair is added at
   dataset preparation, which reads as fixed per image and not resampled per iteration. Consequence: the
   released checkpoint has very probably seen images in our validation and test splits. It used no labels,
   so this affects image fidelity numbers more than classification labels, but Creleased is not
   leakage-free and its PSNR on our "held-out" pairs is optimistic. The report says overlap "cannot be ruled
   out"; the evidence supports "likely".
2. **The paper, the config and the checkpoint disagree on training length.** Literature: paper says 20,000
   iterations, Adam 1e-4, 2,000 timesteps, MSE loss; repo config says 660,000; checkpoint is named
   I725000_E1083. They cannot all describe one model. Our kit's "20,000 iterations" came from the paper.
3. **The noise-level convention matches Palette's.** Literature: "We encode the gamma rather than t".
   Independent confirmation of the bug we fixed.
4. **Others see noisy samples at 50k iterations with this codebase.** Literature (issue #78, anecdotal): noisy
   outputs at n_timestep=2000 after 50k iterations; more ResBlocks removed the noise. Not evidence of cause.
5. **Hair removal is not established to help classification.** Literature: Quishpe-Usca 2024, AlexNet on 10,605
   images: original 91.10% against 90.0% to 91.3% for four filter-plus-removal variants (best +0.20 points).
   Wang 2024 highlights: sparse hair can improve accuracy, dense hair hinders it. Jutte 2024: DullRazor can
   blur lesion details and remove parts of lesions on real hair; no downstream test.

## Code against sources

| Item | Ours | Source | Verdict |
|---|---|---|---|
| Noise-level encoding | was t/T | gamma | Bug, fixed, tested |
| Loss, per-sample continuous level, one t per batch, schedule, Adam 1e-4, batch 4, dropout 0.2, no EMA | matched | upstream code and config | No difference |
| Gradient clipping | 1.0 | none upstream | Unknown effect; norms exceeded the clip on 62.5% of draws (n=8) |
| Training images | 654 hair-free pairs | about 9,300 images | 14x fewer distinct images; Observed and Literature |
| Samples seen | 56,300 iterations x 4 = 225k, about 344 epochs of 654 | paper: 20,000 x ? | If the paper's number is right we saw more samples than they did and still lag, which favours data diversity over iteration count. Inferred, and unresolved by finding 2 |
| Validation loss | flat at 0.0140 to 0.0158 from about 39k to 56k | | Consistent with a data limit. Generalisation gap not measured (Unknown) |

## New experiment: does anything remove REAL hair? (`reports/hair_check_prereg.md`)

40 train-split hairy images, committed script, pre-registered. Observed:

- **The released DM-AHR model leaves real hair in place.** Inside/outside change ratio 1.13 (DullRazor 21.7);
  median 0.0% of pixels changed by more than 25 grey levels; strands visibly intact in the contact sheet.
  Our image-quality numbers are all on synthetic overlays, which is what it was trained on.
- **Our DullRazor (Arm B) damages lesions.** Hair is removed, but lesion and skin texture are replaced by
  polygonal inpainting patches in all 8 reviewed images, and in one the whole lesion is erased (dark lesion
  plus marker ink read as hair). The cause is in `configs/hair_screen.json` (black-hat threshold 18, mask
  dilated 2 iterations of 5x5) and `configs/dullrazor.json` (Telea radius 3): a mask several times wider than
  a hair, filled by a method that cannot reconstruct texture.
- Pre-registered coverage control failed (DullRazor left 63%, rule 30%); that measure is void and is
  disclosed there.

## What this means for the results

- **A beats B (0.750 against 0.698) is confounded.** It is partly or wholly an artefact of our broken
  classical removal, not evidence that hair removal costs accuracy. (Inferred; not yet separated.)
- **A against Creleased is not a hair-removal test.** The released model barely edits real hair, so that arm is
  close to "the original at 128 px"; its F1 of 0.713 sits next to A* at 0.732.
- The honest claim today is that neither arm cleanly removed real hair, so the hypothesis was not tested.
  My earlier statement that "nothing I can honestly do will change that" was too strong: it rested on the
  A-against-B gap, which this check undermines.

## Options, priced (estimates; Modal price not verified)

1. Fix Arm B: mask only thin elongated components, drop the dilation inflation, exclude dark blobs, tune on
   train and validation only, check visually, reprocess 10,015 images (CPU) and retrain 3 ViT seeds. About
   $2 to $3. Reported as a post-hoc exploratory arm, since the original B was frozen before test.
2. Diffusion that removes real hair needs hair synthesis that resembles real hair and far more than 654
   targets. Roughly $35 to $40 with an uncertain outcome. Not recommended before option 1.
