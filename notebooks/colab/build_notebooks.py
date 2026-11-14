"""Generate the three Colab notebooks (outputs cleared) so their shared setup cell stays identical.

Run from anywhere:  python notebooks/colab/build_notebooks.py
The committed .ipynb files are the product; this script is the single place that edits them.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

DISCLAIMER = """> **Research prototype only. Not clinically validated.** Nothing in this notebook is a diagnosis, a
> medical device, or advice for patients or clinicians.
>
> **Licences.** HAM10000 images and lesion masks are CC BY-NC 4.0 (non-commercial use). The DM-AHR
> hair-removal repository has a licence conflict (its README says CC BY-NC 4.0, its LICENSE file says
> Apache-2.0); commercial use is **unresolved** (`reports/license_audit.md`). Keep third-party notices intact.
>
> Tested on CPU and on a Modal GPU. **A real Google Colab session has not been run** for these notebooks."""

# ----------------------------------------------------------------------------------------------
# Shared setup cell (identical in all three notebooks)
# ----------------------------------------------------------------------------------------------
SETUP = r'''# Setup: runs unchanged in Google Colab and on a local machine or server.
import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def _param(name, default):
    # Notebook parameter: the value in the cell below, unless the environment variable DMAHR_<NAME> is set.
    raw = os.environ.get(f"DMAHR_{name}")
    if raw is None:
        return default
    raw = raw.strip()
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None:  # optional integer: "", "none" or "all" mean None
        return None if raw.lower() in ("", "none", "all") else int(raw)
    return raw


try:
    from google.colab import drive  # type: ignore

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")
    DRIVE_BASE = Path(_param("DRIVE_BASE", "/content/drive/MyDrive/dm-ahr-vit"))
    WORK_DIR = Path(_param("WORK_DIR", "/content"))
else:  # local stand-in for the Drive folder
    DRIVE_BASE = Path(_param("DRIVE_BASE", str(Path.home() / "dm-ahr-vit")))
    WORK_DIR = Path(_param("WORK_DIR", str(Path.home() / "dm-ahr-vit-work")))

# --- Repository: use the one on disk if present, else unzip REPO_ZIP, else git clone REPO_URL ---
REPO_ZIP = Path(_param("REPO_ZIP", str(DRIVE_BASE / "dm-ahr-vit-repo.zip")))
REPO_URL = _param("REPO_URL", "")


def _is_repo(p):
    return (p / "src" / "dm_ahr_vit").is_dir() and (p / "configs" / "vit.json").is_file()


def _find_repo():
    cands = []
    if os.environ.get("DMAHR_REPO_DIR"):
        cands.append(Path(os.environ["DMAHR_REPO_DIR"]))
    cwd = Path.cwd().resolve()
    cands += [cwd, *cwd.parents, DRIVE_BASE / "repo", WORK_DIR / "dm-ahr-vit"]
    return next((c for c in cands if _is_repo(c)), None)


REPO = _find_repo()
if REPO is None:
    target = WORK_DIR / "dm-ahr-vit"
    if REPO_ZIP.is_file():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(REPO_ZIP) as zf:
            zf.extractall(target)
        REPO = next((p.parent.parent for p in sorted(target.rglob("vit.json")) if _is_repo(p.parent.parent)), None)
    elif REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True)
        REPO = target
if REPO is None or not _is_repo(REPO):
    raise RuntimeError(
        "Repository not found. Put it on disk, set DMAHR_REPO_DIR, copy dm-ahr-vit-repo.zip into "
        f"{DRIVE_BASE}, or set REPO_URL. See notebooks/colab/README.md."
    )

# --- Dependencies: full = pip install -r requirements.txt; missing = only what cannot be imported ---
PIP_MODE = _param("PIP_MODE", "missing" if IN_COLAB else "full")  # full | missing | skip
if PIP_MODE == "full":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")], check=True)
elif PIP_MODE == "missing":
    # Colab ships torch/numpy/pandas...; replacing them with the pinned versions needs a runtime restart.
    _import_name = {"opencv-python-headless": "cv2", "scikit-image": "skimage", "scikit-learn": "sklearn",
                    "pillow": "PIL", "pyyaml": "yaml"}
    _unused = {"modal", "pytest", "gdown"}
    _wanted = []
    for _line in (REPO / "requirements.txt").read_text().splitlines():
        _line = _line.split("#")[0].strip()
        if not _line:
            continue
        _name = re.split(r"[=<>!~\[ ]", _line, maxsplit=1)[0].lower()
        if _name in _unused or importlib.util.find_spec(_import_name.get(_name, _name.replace("-", "_"))) is not None:
            continue
        _wanted.append(_line)
    if _wanted:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_wanted], check=True)

# --- Locations (environment variables win over the Drive defaults) ---
os.environ.setdefault("DATA_ROOT", str(DRIVE_BASE / "data"))
os.environ.setdefault("ARTIFACT_ROOT", str(DRIVE_BASE / "artifacts"))
os.environ.setdefault("EXPORT_DIR", str(DRIVE_BASE / "exports"))
DATA_ROOT, ARTIFACT_ROOT, EXPORT_DIR = (Path(os.environ[k]) for k in ("DATA_ROOT", "ARTIFACT_ROOT", "EXPORT_DIR"))
RUNS_ROOT = Path(_param("RUNS_ROOT", str(ARTIFACT_ROOT / "notebook_runs")))  # outputs of these notebooks

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
GIT_COMMIT = _git.stdout.strip() if _git.returncode == 0 else "unknown (not a git checkout)"

print("in Colab      :", IN_COLAB)
print("repo          :", REPO, "| commit", GIT_COMMIT)
print("DATA_ROOT     :", DATA_ROOT)
print("ARTIFACT_ROOT :", ARTIFACT_ROOT)
print("EXPORT_DIR    :", EXPORT_DIR)
print("torch         :", torch.__version__)
print("accelerator   :", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU only")
'''

# ----------------------------------------------------------------------------------------------
# Notebook 1: evaluate a ViT checkpoint and explain it
# ----------------------------------------------------------------------------------------------
NB1_MD = (
    "# DM-AHR-ViT 01: evaluate a ViT checkpoint and explain it\n\n" + DISCLAIMER + """

**What this notebook does**

1. Loads one exported ViT-B/16 checkpoint (`<EXPORT_DIR>/vit/vit_b16_<arm>_seed<seed>.pt`).
2. Evaluates it on the cached images of one arm with the repository's evaluation transform
   (shorter side to 255 px, centre crop 224): macro-F1, accuracy, per-class recall, confusion matrix.
3. Computes Grad-CAM (primary: last encoder block `ln_1`; sensitivity: patch-embedding grid) and attention
   rollout, carries each lesion mask through the **same** resize and crop as the image using the image's own
   stored size, and reports lesion recall, lesion precision and pointing against a centred-Gaussian map and
   random maps, plus overlay figures.

`N_IMAGES` (default 32) runs a class-balanced subset so a free Colab GPU finishes in minutes. **Subset numbers are
indicative only** and are not comparable with the full-split results in the report; set `N_IMAGES=None` for
the full split. The default split is `test`: use it only to reproduce reported numbers, never to choose models
or settings. Checkpoints are loaded with `torch.load`, which unpickles: load only files you trust.

Parameters are plain assignments in the *Parameters* cell; the same names can be set as `DMAHR_<NAME>`
environment variables."""
)

NB1_PARAMS = r'''# Parameters
ARM = _param("ARM", "C")                  # A, B, Astar, Cv2, C or Creleased
SEED = _param("SEED", 42)                 # the reported runs use 42, 123 and 2026
SPLIT = _param("SPLIT", "test")           # train | val | test
N_IMAGES = _param("N_IMAGES", 32)         # None (or DMAHR_N_IMAGES=all) = every image that has a mask
SUBSET = _param("SUBSET", "stratified")   # stratified (class round-robin) | random
# export: <EXPORT_DIR>/vit/vit_b16_<arm>_seed<seed>.pt   run: <ARTIFACT_ROOT>/checkpoints/vit/<arm>_seed<seed>/best.pt
EXPORT_FORMAT = _param("EXPORT_FORMAT", "export")
CKPT_PATH = _param("CKPT_PATH", "")       # explicit checkpoint path; overrides EXPORT_FORMAT
BATCH_SIZE = _param("BATCH_SIZE", 16)
NUM_WORKERS = _param("NUM_WORKERS", 2)
N_OVERLAYS = _param("N_OVERLAYS", 6)      # overlay rows in the figure
RANDOM_DRAWS = _param("RANDOM_DRAWS", 100)  # random maps per image for the chance baselines
'''

NB1_LOAD = r'''# Imports, configuration and checkpoint
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader

from dm_ahr_vit.config import CLASSES, config_hash_file, load_config, save_json, sha256_file
from dm_ahr_vit.data.dataset import ArmDataset, build_transforms, collate
from dm_ahr_vit.data.masks import resolve
from dm_ahr_vit.paths import data_root, ham_masks_dir, isic_images_dir, manifest_dir
from dm_ahr_vit.vit.metrics import classification_metrics
from dm_ahr_vit.vit.model import build_model, load_checkpoint
from dm_ahr_vit.xai.batched import (
    MultiLayerGradCAM,
    attention_rollout_batch,
    localization_scores,
    metric_definitions,
    random_map_baselines,
)
from dm_ahr_vit.xai.geometry import ARM_STORED_SIZES, FRAME_VERSION, mask_to_model_frame, to_model_frame
from dm_ahr_vit.xai.vit_xai import gaussian_chance_map, overlay_heatmap

cfg = load_config("configs/vit.json")
SIZE = int(cfg["image_size"])
TOP_FRACTION = float(load_config("configs/sampling.json")["xai_top_fraction"])
if ARM not in ARM_STORED_SIZES:
    raise ValueError(f"ARM must be one of {sorted(ARM_STORED_SIZES)}, got {ARM!r}")

if CKPT_PATH:
    ckpt_path = Path(CKPT_PATH)
elif EXPORT_FORMAT == "export":
    ckpt_path = EXPORT_DIR / "vit" / f"vit_b16_{ARM}_seed{SEED}.pt"
elif EXPORT_FORMAT == "run":
    ckpt_path = ARTIFACT_ROOT / "checkpoints" / "vit" / f"{ARM}_seed{SEED}" / "best.pt"
else:
    raise ValueError("EXPORT_FORMAT must be 'export' or 'run'")
if not ckpt_path.is_file():
    raise FileNotFoundError(f"checkpoint not found: {ckpt_path} (see notebooks/colab/README.md for the Drive layout)")

model = build_model(pretrained=False).to(DEVICE)  # every weight comes from the checkpoint
load_checkpoint(ckpt_path, model, map_location=DEVICE)
model.eval()
CKPT_SHA = sha256_file(ckpt_path)
OUT_DIR = RUNS_ROOT / "01_evaluate_vit_and_xai" / f"{ARM}_seed{SEED}_{SPLIT}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("checkpoint:", ckpt_path, "\nsha256    :", CKPT_SHA)
'''

NB1_SELECT = r'''# Choose the images: split rows that have a lesion mask and whose cached image file exists
splits = pd.read_csv(manifest_dir() / "splits.csv")
x1 = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv")
x1 = x1[(x1["split"] == SPLIT) & (x1["status"] == "ok")]
mask_path_of = dict(zip(x1["image_id"], x1["mask_path"]))
label_of = dict(zip(splits["image_id"], splits["dx"]))


def mask_file(image_id):
    p = resolve(mask_path_of[image_id])  # manifests written in a container hold /__modal/... paths
    return p if p.exists() else ham_masks_dir() / f"{image_id}_segmentation.png"


path_probe = ArmDataset(ARM, SPLIT, splits, isic_images_dir(), data_root() / "processed", None)
candidates = sorted(mask_path_of)
present = [i for i in candidates if path_probe._path(i).is_file() and mask_file(i).is_file()]
n_missing = len(candidates) - len(present)
if n_missing:
    pd.DataFrame({"image_id": sorted(set(candidates) - set(present))}).to_csv(OUT_DIR / "missing_files.csv", index=False)
    print(f"WARNING: {n_missing} of {len(candidates)} {SPLIT} images lack an image or mask file (listed in missing_files.csv)")
    if N_IMAGES is None:
        raise FileNotFoundError("full-split run requested but files are missing; copy them to Drive or set N_IMAGES")


def pick(ids, n, mode, seed=0):
    rng = np.random.default_rng(seed)
    if n is None or n >= len(ids):
        return sorted(ids)
    if mode == "random":
        return sorted(str(v) for v in rng.choice(ids, n, replace=False))
    by_class = {c: [str(v) for v in rng.permutation([i for i in ids if label_of[i] == c])] for c in CLASSES}
    chosen = []
    while len(chosen) < n:
        for c in CLASSES:
            if by_class[c] and len(chosen) < n:
                chosen.append(by_class[c].pop())
    return sorted(chosen)


ids = pick(present, N_IMAGES, SUBSET)
if not ids:
    raise RuntimeError(f"no usable {SPLIT} images found under {data_root()}")
print(f"{ARM} seed {SEED} {SPLIT}: {len(ids)} images ({SUBSET} subset of {len(present)} usable)")
print(pd.Series([label_of[i] for i in ids]).value_counts().to_dict())
'''

NB1_EVAL = r'''# Classification: macro-F1, accuracy, per-class recall, confusion matrix (repository evaluation transform)
ds = ArmDataset(ARM, SPLIT, splits, isic_images_dir(), data_root() / "processed",
                build_transforms(cfg, train=False), image_ids=set(ids))
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)

pred_rows = []
with torch.no_grad():
    for b in loader:
        x = b["image"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(x)
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        for i, image_id in enumerate(b["image_id"]):
            row = {"image_id": image_id, "group_id": b["group_id"][i], "label": b["dx"][i],
                   "pred": CLASSES[int(prob[i].argmax())]}
            row.update({f"prob_{c}": float(prob[i, j]) for j, c in enumerate(CLASSES)})
            pred_rows.append(row)
preds = pd.DataFrame(pred_rows).sort_values("image_id").reset_index(drop=True)
preds.to_csv(OUT_DIR / "predictions.csv", index=False)

y_true = np.array([CLASSES.index(v) for v in preds["label"]])
y_pred = np.array([CLASSES.index(v) for v in preds["pred"]])
y_prob = preds[[f"prob_{c}" for c in CLASSES]].to_numpy()
metrics = classification_metrics(y_true, y_pred, y_prob)
print(f"n = {metrics['n']}   macro-F1 = {metrics['macro_f1']:.4f}   accuracy = {metrics['accuracy']:.4f}"
      f"   balanced accuracy = {metrics['balanced_accuracy']:.4f}")
per_class = pd.DataFrame(metrics["per_class"]).T[["support", "recall", "precision", "f1"]]
per_class[["recall", "precision", "f1"]] = per_class[["recall", "precision", "f1"]].astype(float).round(3)
display(per_class)

cm = np.array(metrics["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(np.array(metrics["confusion_matrix_normalized"]), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
ax.set_yticks(range(len(CLASSES)), CLASSES)
for r in range(len(CLASSES)):
    for c in range(len(CLASSES)):
        ax.text(c, r, int(cm[r, c]), ha="center", va="center", fontsize=8)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"{ARM} seed {SEED} {SPLIT} (n={metrics['n']}; row-normalised colour, counts in cells)")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
plt.show()
'''

NB1_XAI_MD = """## Explanations and lesion localisation

Metric definitions (generated from the repository's function docstrings):"""

NB1_XAI_DEFS = r'''print(metric_definitions(TOP_FRACTION))
'''

NB1_XAI = r'''# Masks go through the identical geometry as the image, using each image's ACTUAL stored size.
allowed = ARM_STORED_SIZES[ARM]
stored, masks = {}, {}
for image_id in ids:
    with Image.open(path_probe._path(image_id)) as fh:
        stored[image_id] = fh.size  # (width, height) of the file the model sees
    if stored[image_id] not in allowed:
        raise RuntimeError(f"{ARM}/{image_id}: stored size {stored[image_id]} is not one of {sorted(allowed)}")
    with Image.open(mask_file(image_id)) as fh:
        masks[image_id] = mask_to_model_frame(fh, stored[image_id], SIZE)
print(f"images stored at 128x128: {np.mean([s == (128, 128) for s in stored.values()]):.0%}; "
      f"mean lesion share of the model frame: {np.mean([m.mean() for m in masks.values()]):.3f}")

METHODS = {"rollout": "attention rollout",
           "gradcam_ln1": "Grad-CAM, last block ln_1 (primary)",
           "gradcam_patch": "Grad-CAM, patch grid (sensitivity)"}


def upsample(maps, size):
    t = torch.from_numpy(maps).float()[:, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()


gauss = gaussian_chance_map(SIZE)
rows, keep = [], {}
for b in loader:
    x = b["image"].to(DEVICE)
    with MultiLayerGradCAM(model) as cam:  # one backward pass gives both Grad-CAM layers
        grids, pred = cam(x)
    maps = {"rollout": upsample(attention_rollout_batch(model, x), SIZE),
            "gradcam_ln1": upsample(grids["last_ln1"], SIZE),
            "gradcam_patch": upsample(grids["patch_grid"], SIZE)}
    for i, image_id in enumerate(b["image_id"]):
        m = masks[image_id]
        row = {"image_id": image_id, "group_id": b["group_id"][i], "label": b["dx"][i],
               "pred": CLASSES[int(pred[i])], "stored_w": stored[image_id][0], "stored_h": stored[image_id][1],
               "mask_area": float(m.mean())}
        for key in METHODS:
            row.update({f"{key}_{k}": v for k, v in localization_scores(maps[key][i], m, TOP_FRACTION).items()})
        row.update({f"gauss_{k}": v for k, v in localization_scores(gauss, m, TOP_FRACTION).items()})
        row.update(random_map_baselines(m, zlib.crc32(image_id.encode()), RANDOM_DRAWS, TOP_FRACTION, device=DEVICE))
        rows.append(row)
        if len(keep) < N_OVERLAYS:
            keep[image_id] = {k: v[i] for k, v in maps.items()}
xai = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
xai.to_csv(OUT_DIR / "xai_per_image.csv", index=False)
print("explained", len(xai), "images")
'''

NB1_SUMMARY = r'''# Summary table: mean over images with a bootstrap 95% interval over images (indicative on a subset)
def boot_ci(v, reps=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    means = v[np.random.default_rng(seed).integers(0, v.size, (reps, v.size))].mean(axis=1)
    return (v.mean(), *np.percentile(means, [2.5, 97.5]))


def fmt(col):
    mean, lo, hi = boot_ci(xai[col])
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


entries = [(label, key) for key, label in METHODS.items()] + [
    ("centred Gaussian map (baseline)", "gauss"),
    ("random pixel map (baseline)", "rand_pixel"),
    ("random patch map (baseline)", "rand_patch"),
]
summary = pd.DataFrame([{"map": label, "lesion recall": fmt(f"{k}_recall"), "lesion precision": fmt(f"{k}_precision"),
                         "pointing": fmt(f"{k}_pointing")} for label, k in entries])
display(summary)
print(f"pointing chance for a uniformly random point = mean lesion area in the model frame = {xai['mask_area'].mean():.3f}")
print(f"image accuracy on these images: {(xai['pred'] == xai['label']).mean():.3f}")
summary.to_csv(OUT_DIR / "xai_summary.csv", index=False)
'''

NB1_FIG = r'''# Overlays: model-frame image with the lesion outline, then each saliency map
n_show = min(N_OVERLAYS, len(keep))
if n_show:
    titles = ["image + lesion outline", *METHODS.values()]
    fig, axes = plt.subplots(n_show, 4, figsize=(13, 3.2 * n_show), squeeze=False)
    for r, image_id in enumerate(list(keep)[:n_show]):
        with Image.open(path_probe._path(image_id)) as fh:
            base = np.asarray(to_model_frame(fh.convert("RGB"), SIZE))
        m = masks[image_id]
        panels = [base] + [overlay_heatmap(base, keep[image_id][k]) for k in METHODS]
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel)
            if 0 < m.sum() < m.size:
                axes[r, c].contour(m, levels=[0.5], colors="white", linewidths=1.2)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
        row = xai[xai["image_id"] == image_id].iloc[0]
        axes[r, 0].text(2, 12, f"{image_id}\ntrue {row['label']} / pred {row['pred']}", color="yellow", fontsize=7,
                        va="top")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "xai_overlays.png", dpi=130)
    plt.show()
'''

NB1_SAVE = r'''# Provenance: what produced these numbers
save_json({
    "arm": ARM, "seed": SEED, "split": SPLIT, "n_images": len(ids), "subset": SUBSET, "n_total_usable": len(present),
    "checkpoint": str(ckpt_path), "checkpoint_sha256": CKPT_SHA, "git_commit": GIT_COMMIT, "frame": FRAME_VERSION,
    "configs_vit_json_sha256": config_hash_file("configs/vit.json"), "top_fraction": TOP_FRACTION,
    "random_draws": RANDOM_DRAWS, "device": DEVICE, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
}, OUT_DIR / "run_info.json")
print("outputs written to", OUT_DIR)
'''

# ----------------------------------------------------------------------------------------------
# Notebook 2: train the ViT
# ----------------------------------------------------------------------------------------------
NB2_MD = (
    "# DM-AHR-ViT 02: train a ViT-B/16 with the matched recipe\n\n" + DISCLAIMER + """

**What this notebook does.** Trains one ViT-B/16 for one arm and seed with the repository's training code and
`configs/vit.json` (AdamW, lr 3e-5, weight decay 0.05, cosine schedule with 5 warm-up epochs, up to 100 epochs,
early stopping on validation macro-F1 with patience 10, class-weighted loss, bf16 on CUDA, gradient clipping 1.0).
Every arm uses the same recipe; only the input images differ.

* **Resumable.** The run folder `<ARTIFACT_ROOT>/checkpoints/vit/<arm>_seed<seed>` lives on Drive. `last.pt` is written
  after every epoch; if the Colab session dies, run the notebook again and training resumes at the next epoch.
  A finished run is recognised by `train_result.json` and is not repeated.
* **`SMOKE = True` (the default) is a quick check**: one epoch on a few hundred images in a separate run folder
  (`..._smoke`). Set `SMOKE = False` for the real run.
* A full run is long (100 epochs over ~7,000 images): use an A100/L4-class GPU. A free T4 will be slow, and every
  epoch writes about 1 GB of checkpoints to Drive.
* ImageNet weights are downloaded by torchvision on first use (internet needed).
* On success (not in smoke mode) the best checkpoint is exported to `<EXPORT_DIR>/vit/vit_b16_<arm>_seed<seed>.pt`,
  the file notebook 01 loads."""
)

NB2_PARAMS = r'''# Parameters
ARM = _param("ARM", "A")                   # A, B, Astar, Cv2, C or Creleased
SEED = _param("SEED", 42)                  # the reported runs use 42, 123 and 2026
SMOKE = _param("SMOKE", True)              # True: 1 epoch on a few hundred images; False: the real run
EPOCHS = _param("EPOCHS", None)            # None = configs/vit.json (100); smoke = 1
SMOKE_TRAIN_IMAGES = _param("SMOKE_TRAIN_IMAGES", 256)
SMOKE_VAL_IMAGES = _param("SMOKE_VAL_IMAGES", 64)
NUM_WORKERS = _param("NUM_WORKERS", 2)     # data-loader workers (Colab has 2 CPU cores)
BATCH_SIZE = _param("BATCH_SIZE", None)    # None = recipe value; set only if the GPU runs out of memory
EXPORT_AFTER_TRAIN = _param("EXPORT_AFTER_TRAIN", True)
OVERWRITE_EXPORT = _param("OVERWRITE_EXPORT", False)
ALLOW_CPU_FULL = _param("ALLOW_CPU_FULL", False)
'''

NB2_PREP = r'''# Imports, recipe, data check
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from dm_ahr_vit.config import CLASSES, config_hash_file, load_config, save_json, sha256_file
from dm_ahr_vit.data.dataset import ArmDataset, build_transforms, collate
from dm_ahr_vit.paths import data_root, isic_images_dir, manifest_dir
from dm_ahr_vit.vit.model import build_model, load_checkpoint
from dm_ahr_vit.vit.train import train_arm
from dm_ahr_vit.xai.geometry import ARM_STORED_SIZES

if ARM not in ARM_STORED_SIZES:
    raise ValueError(f"ARM must be one of {sorted(ARM_STORED_SIZES)}, got {ARM!r}")
if DEVICE == "cpu" and not SMOKE and not ALLOW_CPU_FULL:
    raise RuntimeError("A full ViT-B/16 run needs a GPU. Use SMOKE=True on CPU, or set ALLOW_CPU_FULL=True.")

cfg = load_config("configs/vit.json")
cfg["num_workers"] = NUM_WORKERS
if BATCH_SIZE:
    cfg["batch_size"] = int(BATCH_SIZE)
    print(f"WARNING: batch size {BATCH_SIZE} differs from the recipe ({load_config('configs/vit.json')['batch_size']});"
          " this run is not the matched recipe.")
total_epochs = int(EPOCHS) if EPOCHS else (1 if SMOKE else int(cfg["epochs"]))
print("recipe:", {k: cfg[k] for k in ("image_size", "batch_size", "lr", "weight_decay", "warmup_epochs",
                                      "early_stopping_patience", "grad_clip", "precision")}, "| epochs", total_epochs)

splits = pd.read_csv(manifest_dir() / "splits.csv")
probe = ArmDataset(ARM, "train", splits, isic_images_dir(), data_root() / "processed", None)
folder = isic_images_dir() if ARM == "A" else data_root() / "processed" / ARM
on_disk = {p.stem for p in folder.iterdir()} if folder.is_dir() else set()
need = splits[splits["split"].isin(["train", "val"])]
missing = need[~need["image_id"].isin(on_disk)]
print(f"{ARM}: {len(need) - len(missing)} of {len(need)} train+val images found in {folder}")
if len(missing) and not SMOKE:
    missing[["image_id", "split"]].to_csv(RUNS_ROOT / f"missing_{ARM}.csv", index=False) if RUNS_ROOT.exists() else None
    raise FileNotFoundError(f"{len(missing)} train/val images missing for arm {ARM} (first: {missing['image_id'].iloc[0]})")


def stratified(df, n, seed=0):
    rng = np.random.default_rng(seed)
    by_class = {c: list(rng.permutation(df[df["dx"] == c].index.to_numpy())) for c in CLASSES}
    chosen = []
    while len(chosen) < min(n, len(df)):
        for c in CLASSES:
            if by_class[c] and len(chosen) < n:
                chosen.append(by_class[c].pop())
    return df.loc[chosen]


if SMOKE:
    have = splits[splits["image_id"].isin(on_disk)]
    train_df = stratified(have[have["split"] == "train"], SMOKE_TRAIN_IMAGES)
    val_df = stratified(have[have["split"] == "val"], SMOKE_VAL_IMAGES)
    splits_run = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"SMOKE: {len(train_df)} train and {len(val_df)} val images, {total_epochs} epoch(s)")
else:
    splits_run = splits
run_dir = ARTIFACT_ROOT / "checkpoints" / "vit" / f"{ARM}_seed{SEED}{'_smoke' if SMOKE else ''}"
print("run folder:", run_dir)
'''

NB2_TRAIN = r'''# Train (resumes automatically from last.pt)
import time

result_path = run_dir / "train_result.json"
t0 = time.time()
if result_path.exists():
    result = json.loads(result_path.read_text())
    print(f"already complete: {result['epochs_run']} epochs, best val macro-F1 {result['best_val_macro_f1']:.4f}"
          f" (delete {result_path.name} and final.pt to retrain from last.pt)")
else:
    last = run_dir / "last.pt"
    if last.exists():
        print(f"RESUMING from {last} (written {time.ctime(last.stat().st_mtime)})")
    else:
        print("starting a new run")
    result = train_arm(arm=ARM, seed=SEED, cfg=cfg, splits_df=splits_run, raw_images_dir=isic_images_dir(),
                       processed_root=data_root() / "processed", run_dir=run_dir, device=DEVICE,
                       max_epochs=total_epochs)
print(f"epochs run {result['epochs_run']}, best val macro-F1 {result['best_val_macro_f1']:.4f}, "
      f"this session {time.time() - t0:.0f}s")
'''

NB2_CURVE = r'''# Learning curve and checkpoint round trip
hist = pd.DataFrame(result["history"])
display(hist[["epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "lr", "seconds"]].round(4))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(hist["epoch"], hist["train_loss"], marker="o", label="train")
ax[0].plot(hist["epoch"], hist["val_loss"], marker="o", label="val")
ax[0].set_xlabel("epoch")
ax[0].set_title("loss")
ax[0].legend()
ax[1].plot(hist["epoch"], hist["val_macro_f1"], marker="o")
ax[1].set_xlabel("epoch")
ax[1].set_title("validation macro-F1")
fig.tight_layout()
RUNS_ROOT.joinpath("02_train_vit").mkdir(parents=True, exist_ok=True)
fig.savefig(RUNS_ROOT / "02_train_vit" / f"{run_dir.name}_curve.png", dpi=130)
plt.show()

check = build_model(pretrained=False)
best_ckpt = load_checkpoint(run_dir / "best.pt", check, map_location="cpu")
print(f"best.pt loads into a fresh vit_b_16 (epoch {best_ckpt['epoch']}, val macro-F1 {best_ckpt['best_val_macro_f1']:.4f})")
'''

NB2_EXPORT = r'''# Export the best checkpoint for notebook 01 (never from a smoke run, never over an existing file by accident)
export_path = EXPORT_DIR / "vit" / f"vit_b16_{ARM}_seed{SEED}.pt"
if SMOKE:
    print("smoke run: nothing exported")
elif not EXPORT_AFTER_TRAIN:
    print("EXPORT_AFTER_TRAIN is False: nothing exported")
elif export_path.exists() and not OVERWRITE_EXPORT:
    print(f"{export_path} already exists; set OVERWRITE_EXPORT=True to replace it")
else:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = export_path.with_suffix(".pt.tmp")
    torch.save({"model": best_ckpt["model"], "arm": ARM, "seed": SEED,
                "best_val_macro_f1": float(best_ckpt["best_val_macro_f1"]), "epoch": int(best_ckpt["epoch"])}, tmp)
    tmp.replace(export_path)
    print("exported", export_path, sha256_file(export_path))
RUNS_ROOT.joinpath("02_train_vit").mkdir(parents=True, exist_ok=True)
save_json({"arm": ARM, "seed": SEED, "smoke": SMOKE, "epochs_run": result["epochs_run"],
           "best_val_macro_f1": result["best_val_macro_f1"], "git_commit": GIT_COMMIT,
           "configs_vit_json_sha256": config_hash_file("configs/vit.json"),
           "splits_csv_sha256": sha256_file(manifest_dir() / "splits.csv"),
           "best_pt_sha256": sha256_file(run_dir / "best.pt"), "device": DEVICE},
          RUNS_ROOT / "02_train_vit" / f"{run_dir.name}_run_info.json")
'''

# ----------------------------------------------------------------------------------------------
# Notebook 3: diffusion sampling
# ----------------------------------------------------------------------------------------------
NB3_MD = (
    "# DM-AHR-ViT 03: diffusion hair removal (batched, resumable, cached to Drive)\n\n" + DISCLAIMER + """

**What this notebook does.** Runs the DM-AHR diffusion model through the repository adapter
(`dmahr_adapter.load_model` and `clean_batch`) on hair-present photos and writes one 128 x 128 PNG per image.

* **Sampler.** `SAMPLER="partial"` (default) is the recorded Cv2 recipe: deterministic DDIM from the input
  noised to schedule index `START_T = 600`, in `STEPS = 50` strided steps (`respaced=True, start_t=600,
  num_timesteps=50`). `SAMPLER="full"` is the 2,000-step sampler from pure noise: minutes per batch.
* **Resume.** Outputs that already exist are skipped, files are written atomically, and a `_sampler.json` record
  stops outputs made with different settings from being mixed in one folder.
* **Drive cache.** Outputs go to `<OUTPUT_ROOT>/<OUT_ARM>/<image_id>.png` (default `<DATA_ROOT>/processed/Cv2_colab`).
  Set `OUT_ARM="Cv2"` only to extend the project's own Cv2 cache with the same settings.
* **Check.** When `synthetic_pairs.csv` exists, the notebook also scores validation pairs (synthetic hair on a clean
  reference): hairy-input floor PSNR, output PSNR and SSIM. These pairs are synthetic, so this is a sanity check, not a
  validation of removal on real hair.
* The output is 128 x 128: the model works at that size, which also lowers resolution against the original arm.
* Weights: `<EXPORT_DIR>/diffusion/*.pth` (U-Net state dict). If more than one file is there, set `DIFFUSION_WEIGHTS`."""
)

NB3_PARAMS = r'''# Parameters
OUT_ARM = _param("OUT_ARM", "Cv2_colab")      # output folder name
OUTPUT_ROOT = Path(_param("OUTPUT_ROOT", str(DATA_ROOT / "processed")))
SOURCE = _param("SOURCE", "screened")         # screened: hair-present photos | pairs: synthetic hairy images | folder
SPLITS = _param("SPLITS", "val")              # screened: comma-separated splits; pairs: one split (train or val)
INPUT_DIR = _param("INPUT_DIR", "")           # SOURCE=folder: directory of .jpg/.png photos
N_IMAGES = _param("N_IMAGES", 16)             # None (or DMAHR_N_IMAGES=all) = everything selected
SAMPLER = _param("SAMPLER", "partial")        # partial | full
START_T = _param("START_T", 600)              # partial sampler: schedule index to start from
STEPS = _param("STEPS", 50)                   # partial sampler: number of strided steps
BATCH_SIZE = _param("BATCH_SIZE", 16)
BF16 = _param("BF16", False)                  # bf16 autocast on CUDA; the recorded Cv2 images were fp32
SEED = _param("SEED", 42)                     # base seed; each batch seed is derived from it and the first image id
DIFFUSION_WEIGHTS = _param("DIFFUSION_WEIGHTS", "")  # file name inside <EXPORT_DIR>/diffusion, or a full path
N_PSNR_PAIRS = _param("N_PSNR_PAIRS", 24)     # validation pairs for the PSNR/SSIM check (0 = skip)
FORCE_MIXED = _param("FORCE_MIXED", False)    # allow writing next to outputs made with other settings
CFG_OVERRIDE = _param("CFG_OVERRIDE", "")     # JSON (or path to JSON) merged into configs/dmahr.json; for tests only
'''

NB3_LOAD = r'''# Imports, configuration, weights
import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm

from dm_ahr_vit.config import load_config, save_json, sha256_file
from dm_ahr_vit.diffusion.dmahr_adapter import clean_batch, load_model
from dm_ahr_vit.diffusion.train import _resolve_manifest_path
from dm_ahr_vit.paths import isic_images_dir, manifest_dir, processed_dir


def deep_update(base, new):
    for k, v in new.items():
        base[k] = deep_update(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


cfg = load_config("configs/dmahr.json")
if CFG_OVERRIDE:
    raw = Path(CFG_OVERRIDE).read_text() if Path(CFG_OVERRIDE).is_file() else CFG_OVERRIDE
    cfg = deep_update(cfg, json.loads(raw))
    print("WARNING: configs/dmahr.json overridden:", CFG_OVERRIDE[:80])
if SAMPLER not in ("partial", "full"):
    raise ValueError("SAMPLER must be 'partial' or 'full'")

wdir = EXPORT_DIR / "diffusion"
if DIFFUSION_WEIGHTS:
    weights = Path(DIFFUSION_WEIGHTS) if Path(DIFFUSION_WEIGHTS).is_absolute() else wdir / DIFFUSION_WEIGHTS
else:
    found = sorted(wdir.glob("*.pth"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one .pth in {wdir}, found {[p.name for p in found]}; "
                                "set DIFFUSION_WEIGHTS")
    weights = found[0]
if not weights.is_file():
    raise FileNotFoundError(f"diffusion weights not found: {weights}")
WEIGHTS_SHA = sha256_file(weights)
net = load_model(cfg, weights, device=DEVICE)
sampler_kwargs = (dict(respaced=True, start_t=int(START_T), num_timesteps=int(STEPS)) if SAMPLER == "partial"
                  else dict(respaced=False, num_timesteps=None))
sampler_record = {"sampler": SAMPLER, "start_t": int(START_T) if SAMPLER == "partial" else None,
                  "steps": int(STEPS) if SAMPLER == "partial" else int(cfg["num_timesteps"]), "bf16": bool(BF16),
                  "weights_sha256": WEIGHTS_SHA, "native_resolution": int(cfg["native_resolution"]),
                  "num_timesteps": int(cfg["num_timesteps"])}
print("weights:", weights, "\nsha256 :", WEIGHTS_SHA, "\nsampler:", sampler_record)
if SAMPLER == "full":
    print("NOTE: the full sampler makes", cfg["num_timesteps"], "network calls per batch (slow).")
'''

NB3_SAMPLE_FN = r'''# Batched sampling with resume and atomic writes
def claim_folder(out_dir):
    # Refuse to mix outputs made with different settings in one folder.
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = out_dir / "_sampler.json"
    if rec.exists():
        old = json.loads(rec.read_text())
        if old != sampler_record and not FORCE_MIXED:
            raise RuntimeError(f"{out_dir} holds outputs made with {old}; this run uses {sampler_record}. "
                               "Use another OUT_ARM or set FORCE_MIXED=True.")
    else:
        if any(out_dir.glob("*.png")):
            print(f"NOTE: {out_dir} has images but no _sampler.json; assuming the same settings.")
        save_json(sampler_record, rec)


def run_sampling(items, out_dir):
    # items: list of (image_id, source_path). Skips ids whose output exists; returns the number written.
    claim_folder(out_dir)
    todo = [(i, p) for i, p in items if not (out_dir / f"{i}.png").exists()]
    print(f"{out_dir.name}: to process {len(todo)}, already done {len(items) - len(todo)}")
    t0, written = time.time(), 0
    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="batches"):
        chunk = todo[start:start + BATCH_SIZE]
        images = []
        for _, p in chunk:
            with Image.open(p) as fh:
                images.append(fh.convert("RGB"))
        seed = int(hashlib.sha256(f"{SEED}:{chunk[0][0]}".encode()).hexdigest()[:8], 16)
        outs, stats = clean_batch(net, images, cfg, device=DEVICE, seed=seed, use_bf16=bool(BF16), **sampler_kwargs)
        assert stats["finite"] and len(outs) == len(chunk)
        for (image_id, _), out in zip(chunk, outs):
            final = out_dir / f"{image_id}.png"
            tmp = final.with_suffix(".png.tmp")
            out.save(tmp, format="PNG")
            tmp.replace(final)  # a killed session never leaves a half-written PNG under the final name
            written += 1
    if written:
        print(f"wrote {written} images in {time.time() - t0:.0f}s ({(time.time() - t0) / written:.2f} s/image)")
    return written
'''

NB3_RUN = r'''# Select the inputs and run
if SOURCE == "screened":
    screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
    wanted = [s.strip() for s in SPLITS.split(",")]
    rows = screen[screen["hair_present"].astype(str).str.lower().eq("true") & screen["split"].isin(wanted)]
    items = [(i, isic_images_dir() / f"{i}.jpg") for i in sorted(rows["image_id"])]
    out_dir = OUTPUT_ROOT / OUT_ARM
elif SOURCE == "pairs":
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    if SPLITS not in ("train", "val"):
        raise ValueError("SOURCE='pairs' accepts SPLITS='train' or 'val' only")
    rows = pairs[pairs["split"] == SPLITS].sort_values("image_id")
    items = [(r.image_id, _resolve_manifest_path(r.hairy_path)) for r in rows.itertuples()]
    out_dir = OUTPUT_ROOT / "synth" / SPLITS / f"cleaned_{OUT_ARM}"
elif SOURCE == "folder":
    src = Path(INPUT_DIR)
    items = [(p.stem, p) for p in sorted([*src.glob("*.jpg"), *src.glob("*.png")])]
    out_dir = ARTIFACT_ROOT / "diffusion_outputs" / OUT_ARM
else:
    raise ValueError("SOURCE must be screened, pairs or folder")
if not items:
    raise RuntimeError(f"no inputs found for SOURCE={SOURCE}")
if N_IMAGES is not None:
    items = items[:N_IMAGES]
missing = [(i, p) for i, p in items if not Path(p).is_file()]
if missing:
    raise FileNotFoundError(f"{len(missing)} input files missing, first: {missing[0][1]}")
print(f"{len(items)} inputs ({SOURCE}) -> {out_dir}")
run_sampling(items, out_dir)
'''

NB3_PSNR = r'''# Check on synthetic pairs (validation split only): hairy floor, output PSNR, SSIM
pairs_csv = manifest_dir() / "synthetic_pairs.csv"
psnr_summary = None
if N_PSNR_PAIRS and pairs_csv.is_file():
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    pairs = pd.read_csv(pairs_csv)
    pairs = pairs[pairs["split"] == "val"].sort_values("image_id").head(N_PSNR_PAIRS)
    pair_items = [(r.image_id, _resolve_manifest_path(r.hairy_path)) for r in pairs.itertuples()]
    pair_dir = ARTIFACT_ROOT / "diffusion_outputs" / f"psnr_pairs_{OUT_ARM}"
    run_sampling(pair_items, pair_dir)

    def arr(img, size=None):
        if size is not None and img.size != size:
            img = img.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0

    recs = []
    for r in pairs.itertuples():
        with Image.open(_resolve_manifest_path(r.reference_path)) as fh:
            ref = fh.convert("RGB")
        with Image.open(_resolve_manifest_path(r.hairy_path)) as fh:
            hairy = fh.convert("RGB")
        with Image.open(pair_dir / f"{r.image_id}.png") as fh:
            out = fh.convert("RGB")
        a_ref, a_hairy, a_out = arr(ref), arr(hairy, ref.size), arr(out, ref.size)
        recs.append({"image_id": r.image_id,
                     "floor_psnr": peak_signal_noise_ratio(a_ref, a_hairy, data_range=1.0),
                     "psnr": peak_signal_noise_ratio(a_ref, a_out, data_range=1.0),
                     "ssim": structural_similarity(a_ref, a_out, data_range=1.0, channel_axis=2, win_size=win)})
    pdf = pd.DataFrame(recs)
    pdf["margin_db"] = pdf["psnr"] - pdf["floor_psnr"]
    RUNS_ROOT.joinpath("03_diffusion_sampling").mkdir(parents=True, exist_ok=True)
    pdf.to_csv(RUNS_ROOT / "03_diffusion_sampling" / f"psnr_pairs_{OUT_ARM}.csv", index=False)
    psnr_summary = {"n_pairs": len(pdf), "hairy_floor_psnr": float(pdf["floor_psnr"].mean()),
                    "output_psnr": float(pdf["psnr"].mean()), "margin_db": float(pdf["margin_db"].mean()),
                    "pairs_above_floor": float((pdf["margin_db"] > 0).mean()), "output_ssim": float(pdf["ssim"].mean())}
    print(psnr_summary)
    display(pdf.round(3).head(10))
else:
    print("PSNR/SSIM check skipped (no synthetic_pairs.csv or N_PSNR_PAIRS = 0)")
'''

NB3_FIG = r'''# Look at a few results (and compare with the project cache when it has the same image)
shown = [(i, p) for i, p in items if (out_dir / f"{i}.png").exists()][:4]
ref_dir = processed_dir("Cv2")
if shown:
    fig, axes = plt.subplots(len(shown), 3, figsize=(9, 3 * len(shown)), squeeze=False)
    for r, (image_id, src_path) in enumerate(shown):
        with Image.open(src_path) as fh:
            axes[r, 0].imshow(fh.convert("RGB"))
        with Image.open(out_dir / f"{image_id}.png") as fh:
            axes[r, 1].imshow(fh.convert("RGB"))
        cache = ref_dir / f"{image_id}.png"
        if SOURCE == "screened" and cache.is_file() and out_dir != ref_dir:
            with Image.open(cache) as fh:
                axes[r, 2].imshow(fh.convert("RGB"))
            mad = np.abs(arr(Image.open(cache)) - arr(Image.open(out_dir / f"{image_id}.png"))).mean() * 255 \
                if "arr" in globals() else float("nan")
            axes[r, 2].set_title(f"project Cv2 cache (mean abs diff {mad:.1f}/255)", fontsize=8)
        for c, t in enumerate(["input", f"notebook output ({SAMPLER})"]):
            axes[r, c].set_title(t if r == 0 else "", fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    RUNS_ROOT.joinpath("03_diffusion_sampling").mkdir(parents=True, exist_ok=True)
    fig.savefig(RUNS_ROOT / "03_diffusion_sampling" / f"examples_{OUT_ARM}.png", dpi=120)
    plt.show()
save_json({"out_dir": str(out_dir), "n_inputs": len(items), "sampler": sampler_record, "git_commit": GIT_COMMIT,
           "configs_dmahr_json_sha256": sha256_file(REPO / "configs" / "dmahr.json"), "device": DEVICE,
           "psnr_check": psnr_summary},
          RUNS_ROOT / "03_diffusion_sampling" / f"run_info_{OUT_ARM}.json")
print("Colab: Drive writes are asynchronous; wait for the Drive sync indicator before closing the session.")
'''


def make(path, cells, accelerator=True):
    nb = new_notebook()
    nb["nbformat"], nb["nbformat_minor"] = 4, 5
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    if accelerator:
        nb["metadata"]["accelerator"] = "GPU"
        nb["metadata"]["colab"] = {"gpuType": "T4", "provenance": []}
    stem = Path(path).stem
    for k, (kind, text) in enumerate(cells):
        cell = new_markdown_cell(text) if kind == "md" else new_code_cell(text)
        cell["id"] = f"{stem[:2]}-{k:02d}"
        nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, HERE / path)
    print("wrote", HERE / path, len(nb.cells), "cells")


def main() -> None:
    SETUP_MD = "## 1. Setup\nWorks in Colab (mounts Drive) and outside it. Edit nothing here; change the parameters in the next cell."
    make("01_evaluate_vit_and_xai.ipynb", [
        ("md", NB1_MD), ("md", SETUP_MD), ("code", SETUP),
        ("md", "## 2. Parameters"), ("code", NB1_PARAMS),
        ("md", "## 3. Load the checkpoint"), ("code", NB1_LOAD),
        ("md", "## 4. Choose images"), ("code", NB1_SELECT),
        ("md", "## 5. Classification"), ("code", NB1_EVAL),
        ("md", NB1_XAI_MD), ("code", NB1_XAI_DEFS), ("code", NB1_XAI), ("code", NB1_SUMMARY),
        ("md", "## Overlays\nWhite line: lesion outline in the model's frame."), ("code", NB1_FIG),
        ("md", "## Provenance"), ("code", NB1_SAVE),
    ])
    make("02_train_vit.ipynb", [
        ("md", NB2_MD), ("md", SETUP_MD), ("code", SETUP),
        ("md", "## 2. Parameters"), ("code", NB2_PARAMS),
        ("md", "## 3. Recipe and data check"), ("code", NB2_PREP),
        ("md", "## 4. Train (re-run this cell after a disconnect to resume)"), ("code", NB2_TRAIN),
        ("md", "## 5. Learning curve"), ("code", NB2_CURVE),
        ("md", "## 6. Export and provenance"), ("code", NB2_EXPORT),
    ])
    make("03_diffusion_sampling.ipynb", [
        ("md", NB3_MD), ("md", SETUP_MD), ("code", SETUP),
        ("md", "## 2. Parameters"), ("code", NB3_PARAMS),
        ("md", "## 3. Load the model"), ("code", NB3_LOAD),
        ("md", "## 4. Sampler"), ("code", NB3_SAMPLE_FN),
        ("md", "## 5. Run (re-run after a disconnect: finished images are skipped)"), ("code", NB3_RUN),
        ("md", "## 6. PSNR / SSIM on synthetic validation pairs"), ("code", NB3_PSNR),
        ("md", "## 7. Examples and provenance"), ("code", NB3_FIG),
    ])


if __name__ == "__main__":
    main()
