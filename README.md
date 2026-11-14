# DM-AHR-ViT

Does diffusion-based hair removal (DM-AHR) change dermoscopic classification or
explanation quality compared with a classical DullRazor-style baseline?

Six input arms on HAM10000 with a frozen, lesion-grouped split:

| Arm | Input |
|---|---|
| A | original image |
| B | DullRazor-style morphology + Telea inpainting |
| C | first diffusion run (defective trainer, recorded failure) |
| Cv2 | rebuilt diffusion hair removal (repaired trainer, partial-noise sampler, real-strand fine-tune) |
| A* | original at 128px, controlling DM-AHR's resolution change |
| Creleased | the authors' released DM-AHR checkpoint, 25-step respaced sampling (exploratory) |

Arm C is our first diffusion run at 20,000 iterations. It was trained with a
script that gave the network an inverted noise-level signal, so it is a failed
replication and says nothing about DM-AHR as published (see
`reports/oneshot_diagnostic_report.md`). Arm Cv2 is the rebuilt arm: repaired
trainer, sampler that starts from the noised input, fine-tune on real hair
strands pasted onto clean photos. Its pre-registered real-photo gate was **not
met**, and it was run on the test split anyway as a recorded deviation
(`reports/stageC_results.md`, decision D34). Creleased runs the released weights
through the same pipeline; it was added after the protocol was frozen and is
reported as exploratory, and its sampling routine is not reproducible from this
repository. See `reports/decision_log.md` (D1, D12, D22, D26, D34).

ImageNet-pretrained ViT-B/16, seven classes, seeds 42/123/2026, plus paired
PSNR/SSIM on held-out synthetic pairs and Grad-CAM / attention-rollout
localization against released lesion masks.

Start with `AGENTS.md` (project rules), `docs/CODE_GUIDE.md` (how the code
fits together), `reports/final_research_report.md` (results) and
`reports/decision_log.md` (every deviation and why). The client's proposal is
at `docs/proposal.docx`.

**Research prototype only. Not clinically validated. Not for diagnosis or
patient care.**

Licence status is unresolved; see `reports/license_audit.md`.

## Requirements

- Python 3.11
- A Modal account with a token: `export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...`
  (the CLI reads `MODAL_API_TOKEN` if it contains both, see below)
- No local GPU is needed; all GPU work runs on Modal.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Modal setup

The app is `modal_app.py`; the volume is `dmahr-vit-vol` with
`DATA_ROOT=/vol/data` and `ARTIFACT_ROOT=/vol/artifacts`.

```bash
modal profile current          # confirms credentials
modal volume ls dmahr-vit-vol  # confirms the volume exists
```

## Step-by-step pipeline

```bash
# 1. Environment probe (writes artifacts/logs/environment_report.json)
modal run modal_app.py::gpu_probe

# 2. Download ISIC collection 212, HAM10000 metadata and released masks
modal run modal_app.py::download_data

# 3. Audit + confirm near-duplicates + freeze splits + mask QA
modal run modal_app.py::build_manifests

# 4. Hair screening (frozen), Arm B, Arm A*, synthetic pairs
modal run modal_app.py::hair_pipeline

# 5. DM-AHR smoke test on the released weights (gate G5)
modal run modal_app.py::dmahr_smoke

# 6. Diffusion sanity check, then the leakage-controlled training run
modal run modal_app.py::dmahr_overfit
modal run --detach modal_app.py::dmahr_train

# 7. Arm C: 4 shards over hair-present images, then the held-out test pairs
modal run --detach modal_app.py::sample_all --n-shards 4

# 8. ViT training, 4 arms x 3 seeds (sequential in one container)
modal run --detach modal_app.py::vit_sweep --runs-json \
  '[["A",42],["A",123],["A",2026],["B",42],["B",123],["B",2026],["Astar",42],["Astar",123],["Astar",2026],["C",42],["C",123],["C",2026]]'

# 9. Locked test evaluation, one call per frozen checkpoint
modal run modal_app.py::vit_eval --arm A --seed 42

# 10. Paired image quality on held-out synthetic pairs
modal run modal_app.py::image_quality

# 11. Explainability against X1 masks
modal run modal_app.py::xai_run --arm A --seed 42

# 12. Statistics, acceptance report, inventory and final report
modal run modal_app.py::stats_run
modal run modal_app.py::acceptance
modal run modal_app.py::finalize
```

### Rebuilt diffusion arm (Cv2): Stage B and Stage C

Arm C above is the first diffusion run, made with the defective trainer; it is kept as a recorded
failure. The rebuilt arm is described in `reports/prereg_v2_diffusion.md` (Stage A, B) and
`reports/protocol_addendum_stageC.md` (Stage C); results are in `reports/stageC_results.md`.

```bash
# Harvest real hair strands from train/validation images, then fine-tune on composites
modal run modal_app.py::harvest_hair
modal run --detach modal_app.py::dmahr_train_v2 --iterations 20000 --out-name dmahr_v2

# Validation-only checks (never the test split)
modal run modal_app.py::partial_noise_eval --ckpt <path> --name <n> --tag <t>
modal run modal_app.py::composite_eval --ckpts "name=<path>" --tag <t>

# Stage C: sample Cv2 (partial-noise start 600, 50 DDIM steps), train/evaluate ViTs, statistics
modal run modal_app.py::fill_cv2_passthrough
modal run --detach modal_app.py::cv2_sample_driver --n-shards 4 --pairs-split test
modal run --detach modal_app.py::vit_arm_driver --arm Cv2 --seeds 42,123,2026
modal run modal_app.py::image_quality_arm --split test --c-arm Cv2
modal run modal_app.py::stats_arm --c-arm Cv2
```

Stage E (response to an independent review, `reports/prereg_v4_review_response.md`, results in
`reports/stageE_results.md`): extra seeds, pooled statistics, validation-only checks, review kit.

```bash
modal run --detach modal_app.py::vit_extra_seeds_driver --arms A,Astar,B,Cv2 --seeds 7,11,17,23,31
python scripts/53_pooled_analysis.py            # two-level bootstrap over lesion groups and seeds
python scripts/55_headline_table.py             # pooled headline metrics against reference ranges
modal run modal_app.py::composite_arms          # DullRazor and Cv2 on pasted real strands, lesion damage
modal run modal_app.py::condition_usage         # does each model use the hairy input?
modal run modal_app.py::residual_hair_split --ids-gate <ids> --ids-dense <ids>
modal run modal_app.py::review_kit_build        # blinded 50-photo kit; rate it, then
python scripts/54_review_kit.py analyze --dir artifacts/review_kit --ratings <completed csv>
```

## Local tests (no GPU, no Modal)

```bash
PYTHONPATH=src pytest tests -q
```

## Notebooks and documentation

Three notebooks, all runnable from the repository root:

- `notebooks/01_results_walkthrough.ipynb` — reads the committed artifacts and explains what was
  done, what the numbers mean and where each one comes from. Needs no GPU, no Modal and no image
  download. Use it to present the study.
- `notebooks/02_pipeline_reproduction.ipynb` — the full pipeline on Modal, phase by phase, with cost
  warnings. Modeled on the commands below.
- `notebooks/03_local_gpu_pipeline.ipynb` — the same pipeline run **natively** on a local GPU or a
  Colab GPU, calling `scripts/` directly with `python`. No Modal account, no token. This is the one
  to use if you want to watch the pipeline execute on your own hardware.

The `scripts/` directory is Modal-free by design: every phase is a plain CLI that reads
`DATA_ROOT` and `ARTIFACT_ROOT` from the environment. Modal is one way to run them, not a
requirement.

`docs/CODE_GUIDE.md` is the written companion: repository map, phase-by-phase explanation, a
traceability table from every reported number to its artifact, the reasoning behind each design
decision, the list of deviations from the proposal, a glossary and likely supervisor questions.

## Repository layout

```
src/dm_ahr_vit/       library code, importable without Modal
  data/               download, manifest, splits, masks, synthetic hair
  hair/               screening detector and DullRazor-style removal
  diffusion/          pinned DM-AHR adapter and leakage-controlled training
  vit/                ViT-B/16 model, matched training, metrics
  xai/                Grad-CAM and attention rollout
  stats/              paired lesion-cluster bootstrap, Holm correction
scripts/              numbered phase scripts
configs/              JSON configs, hashed and frozen
third_party/dmahr/    pinned upstream DM-AHR checkout (unmodified)
reports/              protocol, decision log, licence audit, compute plan
tests/                pytest suite
```

## Data and artifacts

Large data, checkpoints and outputs live on the Modal volume, never in Git.
To pull anything locally:

```bash
modal volume get dmahr-vit-vol /artifacts/logs f"local_logs/"
modal volume get dmahr-vit-vol /artifacts/checkpoints/vit local_vit/
modal volume get dmahr-vit-vol /data/manifests local_manifests/
```

## Data acquisition notes

- ISIC collection 212 has 11,720 images; the confirmatory cohort is the 10,015
  listed in the Harvard Dataverse metadata. The remaining 1,705 are quarantined.
- DERMAHAIR (Kaggle) needs `KAGGLE_USERNAME`/`KAGGLE_KEY`. Those are not
  configured here, so the required synthetic-hair fallback is used and the
  DERMAHAIR route is recorded as blocked.

## Moving artifacts to Google Drive (cloud to cloud)

`modal_app.py::upload_to_drive` streams the volume straight to Drive with
rclone, so nothing transits your machine. Egress is free within Modal's 1 TiB
monthly allowance; the job itself is a few cents of CPU.

### One-time credentials

Pick **one** of these.

**OAuth as yourself (recommended for personal Gmail).** Service accounts often
fail on personal accounts with "Service Accounts do not have storage quota".

```bash
# On a machine with a browser:
rclone config create gdrive drive scope=drive
# then read the token rclone printed and store it as a secret
modal secret create gdrive-sa GDRIVE_TOKEN='{"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}'
```

**Service account (use with a Workspace Shared Drive).**

1. Create a service account in Google Cloud Console and download its JSON key.
2. Create the destination folder in Drive and share it with the service
   account's `client_email`, granting Editor.
3. Store the key:

```bash
modal secret create gdrive-sa GDRIVE_SA_JSON="$(cat service-account.json)"
```

### Upload

```bash
# always verify first: lists what would move, transfers nothing
modal run modal_app.py::upload_to_drive --tier results --dry-run

modal run --detach modal_app.py::upload_to_drive --tier results     # ~9 MB
modal run --detach modal_app.py::upload_to_drive --tier checkpoints # ~43 GB
modal run --detach modal_app.py::upload_to_drive --tier all
```

`--dest` sets the Drive folder (default `dmahr-vit`). Each run writes
`artifacts/logs/drive_upload_<tier>.json` with a per-path status record.

### Free alternative: Colab as the bridge

Needs no Google Cloud project and no Modal credits:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q modal
import os
os.environ['MODAL_TOKEN_ID'] = 'ak-...'
os.environ['MODAL_TOKEN_SECRET'] = 'as-...'

!modal volume get dmahr-vit-vol /artifacts/logs /content/drive/MyDrive/dmahr/
```

Note the destination-directory quirk: pass an existing **parent** directory and
Modal appends the remote's last path segment.

