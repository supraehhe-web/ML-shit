# DM-AHR-ViT: plain-English summary

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

> **Correction (2026-11-13).** The "where the model looks" result below should not be relied on. After review we found the lesion outlines used for scoring were not lined up with the part of the photo the model actually sees, and the report described the score differently from how it was computed. The numbers are being re-measured; until then no claim about where the model looks, for or against hair removal, is supported. Details: `reports/stageE_results.md`, section 6.

## The question

Does removing hair from skin-lesion photos help a computer program classify the lesion? We compared
kinds of input on the 10,015 HAM10000 photos: the original photos, photos cleaned by a classical
hair-removal tool (DullRazor), and photos cleaned by an AI hair-removal model (DM-AHR).

## The scores

How well each version classified the held-out test photos (macro-F1, higher is better). Each score is
the average of eight separate training runs, except the last two rows (three runs). The spread between
runs of the same input is about 0.01 to 0.02.

| Input | Score |
|---|---:|
| Original photos | **0.741** |
| Original photos shrunk to 128 px (a fairness control) | 0.728 |
| AI hair removal, rebuilt and retrained by us | 0.722 |
| Classical hair removal | 0.693 |
| AI hair removal, authors' released model (exploratory, 3 runs) | 0.713 |
| AI hair removal, our first training run (had a bug, 3 runs) | 0.426 (failed) |

All the working inputs land in the range we would expect for this kind of classifier (accuracy 0.84 to 0.86,
macro-F1 0.69 to 0.74, AUC 0.965 to 0.970). The first run is far outside it and is kept only as a
documented failure.

## What this shows

- **Hair removal did not change the classification score in a way we can measure.** The rebuilt AI input
  is statistically indistinguishable from the originals, from the shrunk-photo control and from the
  classical tool. Only "originals beat the classical tool" is clear (0.048 higher). Even with eight runs
  per input, this test set can only resolve differences of about 0.04 to 0.05; resolving 0.02 would need
  a larger test set, not more runs.
- **On picture quality the classical tool wins.** On 186 test photos where we drew hair ourselves it
  recovered the clean picture better (35.7 dB against 29.2 dB for the AI model; the hairy photo itself is
  22.1 dB). We then pasted real hair strands cut from other photos onto clean photos, where the true clean
  picture is known: the classical tool again wins (40.3 dB against 33.9 dB), and it also changes fewer
  pixels in the lesion where there was no hair (6% against 18%). The AI model redraws the whole 128-pixel
  picture, so it loses fine detail everywhere.
- **The classical tool still behaves worse on real photos.** On 52 real photos it altered 25% of the
  lesion area against 11% for the AI model, which fits it mistaking ink, texture or dense hair for hair.
  This check cannot prove that, because hair lying across a lesion also counts as change.
- **Where the model looks.** For every input, the area the model attends to overlaps the lesion less than
  a plain "look at the centre" guess would (0.64). So this measure cannot say any input helps the model
  look at the lesion.

## What is not proven

- **That the AI model removes real hair from real photographs.** Our planned test (a ratio of change
  inside versus outside the hair) came out at 1.95 and 1.57 against a bar of 2.0, and we went ahead anyway,
  recording the deviation. The evidence for real hair is a visual look at 12 photos by us. A blinded
  review kit of 50 photos is ready and needs a person to rate it; until then this stays unproven.
- **That hair removal helps or hurts diagnosis.** The study found no measurable difference.

## Why we could not do better

- The dataset of real hairy and hairless photo pairs (DERMAHAIR) was not available to us. We used real
  hair strands cut from other photos and artificial hair, which give a true reference picture but are not
  the same as real hairy-versus-clean pairs.
- Our AI model works at 128 pixels, so lesions lose fine texture.
- The authors' released model very probably saw our test photos during its own training (their paper
  says it trained on HAM10000). It used no labels, so this mainly affects image-quality numbers.
- Our training differs from the original code in one way we found late: we clip the training gradient,
  which the original does not.

## What is solid

- No test photo was used to choose settings or to tune any model. Photos were split by lesion, so the
  same lesion never appears in both training and test. Every choice was written down before its test
  photos were processed, and each trained model was tested once.
- All planned checks pass, earlier results reproduce, and every number in the reports traces to a saved
  file. Earlier claims that turned out wrong (an "undertrained" diagnosis, a failed validation gate that
  was really an 8-pair smoke run, three-run spreads that were too small) were retracted and corrected.
- Cost of the whole rebuild and checks was roughly $11 to $16 of cloud compute (prices not verified).

## Optional next step

Have a person rate the 50-photo review kit (about an hour). If the client wants to resolve differences of
0.02, the test set would need to be larger; more training runs alone cannot do it.

## Licensing

HAM10000 is CC BY-NC 4.0. The DM-AHR repository's README and LICENSE contradict each other (CC BY-NC
4.0 against Apache-2.0), so commercial use is unresolved and needs a rights decision.
