# Data audit (recomputed from files)

- Collection 212 records: **11720** (expected 11720)
- C0 decoded images: **10015** (expected 10015)
- C0-extra quarantined: **1705**
- Exclusions: 1705 ({'c0_extra_quarantined': 1705})
- dHash candidates (distance <= 6): 70062; pixel-confirmed near-duplicates: **21**
- Groups after merging lesion + confirmed near-duplicates: 7470 (0 groups span more than one class)
- Split sizes: {'train': 7011, 'val': 1502, 'test': 1502} (target {'train': 0.7, 'val': 0.15, 'test': 0.15})
- Split hash: `fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b`
- Split verification: PASS

## Recovered class counts vs expected

| class | count | expected | delta |
|---|---|---|---|
| akiec | 327 | 327 | 0 |
| bcc | 514 | 514 | 0 |
| bkl | 1099 | 1099 | 0 |
| df | 115 | 115 | 0 |
| mel | 1113 | 1113 | 0 |
| nv | 6705 | 6705 | 0 |
| vasc | 142 | 142 | 0 |

## Per-split class counts (fraction of class in each split)

| class | train n (frac) | val n (frac) | test n (frac) | total | target frac |
|---|---|---|---|---|---|
| akiec | 229 (0.700) | 49 (0.150) | 49 (0.150) | 327 | 0.70/0.15/0.15 |
| bcc | 360 (0.700) | 77 (0.150) | 77 (0.150) | 514 | 0.70/0.15/0.15 |
| bkl | 769 (0.700) | 165 (0.150) | 165 (0.150) | 1099 | 0.70/0.15/0.15 |
| df | 81 (0.704) | 17 (0.148) | 17 (0.148) | 115 | 0.70/0.15/0.15 |
| mel | 779 (0.700) | 167 (0.150) | 167 (0.150) | 1113 | 0.70/0.15/0.15 |
| nv | 4693 (0.700) | 1006 (0.150) | 1006 (0.150) | 6705 | 0.70/0.15/0.15 |
| vasc | 100 (0.704) | 21 (0.148) | 21 (0.148) | 142 | 0.70/0.15/0.15 |

## X1 lesion masks (released HAM10000 segmentations)

- Masks available: 10015; QA-ok: 10015
- Test images: 1502, with mask: 1502, QA-ok: 1502
- Mask/image size mismatches: 0 (masks are resized with nearest-neighbour when scoring)
- Provenance: Harvard Dataverse doi:10.7910/DVN/DBW86T file `HAM10000_segmentations_lesion_tschandl.zip`. The source does not state that the masks were expert-drawn, so they are labelled X1 released masks, not "expert-traced".

