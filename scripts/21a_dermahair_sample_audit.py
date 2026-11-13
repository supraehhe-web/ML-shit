"""Audit the released DM-AHR configs and the vendored DERMAHAIR (akiec) sample. CPU only.

Answers, from files in the repository and without any Modal job:
  * Do the seven class configs imply class-specific checkpoints?
  * What do the configs say about training data, iterations and EMA?
  * What does the paper's own released sample look like (size, conditioning
    image, hairy-input floor), and how does our synthetic hair differ from it?

Writes artifacts/logs/dermahair_sample_audit.json.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import _strip_comments  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root  # noqa: E402

UP = ROOT / "third_party" / "dmahr"
ZIP = UP / "dataset" / "akiec_16_128.zip"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DIFF_THRESHOLD = 30  # 8-bit levels; the same definition is used for both domains


def _load_cfg(name: str) -> dict:
    obj = json.loads(_strip_comments((UP / "config" / f"{name}.json").read_text()))
    return obj[0] if isinstance(obj, list) else obj


def config_audit() -> dict:
    cfgs = {c: _load_cfg(c) for c in CLASSES}
    resume = {c: Path(cfgs[c]["path"]["resume_state"]).name for c in CLASSES}
    dataroots = {c: cfgs[c]["datasets"]["train"]["dataroot"] for c in CLASSES}
    base = cfgs["akiec"]
    same_model = all(cfgs[c]["model"] == base["model"] for c in CLASSES)
    same_train = all(cfgs[c]["train"] == base["train"] for c in CLASSES)
    ckpt_name = resume["akiec"]
    ckpt_iter = int(ckpt_name.split("_")[0].lstrip("I"))
    src = "\n".join(p.read_text(errors="ignore") for p in UP.rglob("*.py"))
    return {
        "classes": CLASSES,
        "resume_state_basename_per_class": resume,
        "single_shared_checkpoint_name": len(set(resume.values())) == 1,
        "dataroot_per_class": dataroots,
        "model_block_identical_across_classes": same_model,
        "train_block_identical_across_classes": same_train,
        "implies_class_specific_checkpoint": False,
        "interpretation": (
            "All seven configs resume from the same experiment/checkpoint and differ only in "
            "the dataset folder they point at, so they select the evaluation class folder, not "
            "a class-specific model. The training data path named in each config is that class "
            "folder; the configs do not document the data the released weights were trained on."),
        "n_iter_in_config": int(base["train"]["n_iter"]),
        "released_checkpoint_iteration": ckpt_iter,
        "checkpoint_iteration_exceeds_config_n_iter": ckpt_iter > int(base["train"]["n_iter"]),
        "paper_stated_iterations": 20000,
        "ema_scheduler_in_config": base["train"].get("ema_scheduler"),
        "ema_marked_not_used_in_upstream_config": "not used now" in (UP / "config" / "akiec.json").read_text(),
        "ema_referenced_in_any_upstream_python": any(
            tok in src for tok in ("ema_decay", "EMA(", "ema_scheduler", "ema_model")),
        "l_resolution_r_resolution": [base["datasets"]["train"]["l_resolution"],
                                      base["datasets"]["train"]["r_resolution"]],
        "dataset_mode_train_val": [base["datasets"]["train"]["mode"], base["datasets"]["val"]["mode"]],
        "batch_size": base["datasets"]["train"]["batch_size"],
        "lr": base["train"]["optimizer"]["lr"],
        "grad_clip_in_upstream_trainer": "clip_grad" in src,
    }


def _read(zf: zipfile.ZipFile, name: str) -> bytes:
    return zf.read(name)


def _hair_fraction(hairy: np.ndarray, ref: np.ndarray) -> float:
    d = np.abs(hairy.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
    return float((d > DIFF_THRESHOLD).mean())


def _floor(hairy: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    a, b = hairy.astype(np.float64) / 255.0, ref.astype(np.float64) / 255.0
    return (float(peak_signal_noise_ratio(b, a, data_range=1.0)),
            float(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=7)))


def _rs(arr: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.Resampling.BICUBIC))


def _summ(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def dermahair_sample() -> dict:
    screen_cfg = load_config("configs/hair_screen.json")
    out: dict = {"zip": str(ZIP.relative_to(ROOT)), "class": "akiec"}
    rows = {"floor_psnr_256": [], "floor_ssim_256": [], "floor_psnr_128": [], "floor_ssim_128": [],
            "hair_frac_128": [], "ref_detector_coverage_at_256px": [], "sr_equals_lr_bytes": [], "sizes": set()}
    with zipfile.ZipFile(ZIP) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith("akiec_16_128/hr_128/") and n.endswith(".jpg"))
        for n in names:
            stem = n.split("/")[-1]
            ref = np.asarray(Image.open(io.BytesIO(_read(zf, n))).convert("RGB"))
            sr_b = _read(zf, f"akiec_16_128/sr_16_128/{stem}")
            lr_b = _read(zf, f"akiec_16_128/lr_16/{stem}")
            hairy = np.asarray(Image.open(io.BytesIO(sr_b)).convert("RGB"))
            rows["sr_equals_lr_bytes"].append(sr_b == lr_b)
            rows["sizes"].add(tuple(ref.shape[:2]))
            p, s = _floor(hairy, ref)
            rows["floor_psnr_256"].append(p), rows["floor_ssim_256"].append(s)
            h128, r128 = _rs(hairy, 128), _rs(ref, 128)
            p, s = _floor(h128, r128)
            rows["floor_psnr_128"].append(p), rows["floor_ssim_128"].append(s)
            rows["hair_frac_128"].append(_hair_fraction(h128, r128))
            rows["ref_detector_coverage_at_256px"].append(detect_hair(ref, screen_cfg)[1]["coverage"])
    out["n_pairs"] = len(rows["floor_psnr_256"])
    out["image_sizes"] = sorted(map(list, rows["sizes"]))
    out["sr_and_lr_folders_byte_identical_fraction"] = float(np.mean(rows["sr_equals_lr_bytes"]))
    for k in ("floor_psnr_256", "floor_ssim_256", "floor_psnr_128", "floor_ssim_128",
              "hair_frac_128", "ref_detector_coverage_at_256px"):
        out[k] = _summ(rows[k])
    out["ref_detector_flagged_fraction_UNINTERPRETABLE_outside_design_resolution"] = float(np.mean(
        np.asarray(rows["ref_detector_coverage_at_256px"]) >= screen_cfg["coverage_threshold"]))
    return out


def our_synthetic_test() -> dict:
    root = data_root() / "processed" / "synth" / "test"
    hairy_dir, ref_dir = root / "hairy", root / "reference"
    if not hairy_dir.exists():
        return {"status": "not available locally"}
    fl_p, fl_s, hf = [], [], []
    for p in sorted(hairy_dir.glob("*.png")):
        r = ref_dir / p.name
        if not r.exists():
            continue
        h = _rs(np.asarray(Image.open(p).convert("RGB")), 128)
        g = _rs(np.asarray(Image.open(r).convert("RGB")), 128)
        a, b = _floor(h, g)
        fl_p.append(a), fl_s.append(b), hf.append(_hair_fraction(h, g))
    return {"split": "test", "n_pairs": len(fl_p), "floor_psnr_128": _summ(fl_p),
            "floor_ssim_128": _summ(fl_s), "hair_frac_128": _summ(hf),
            "note": "same diff-threshold hair-fraction definition as the DERMAHAIR sample"}


def main() -> int:
    result = {"definition": f"hair_frac = share of pixels whose max-channel |hairy - reference| exceeds "
                            f"{DIFF_THRESHOLD}/255, measured at 128 px",
              "config_audit": config_audit(),
              "dermahair_akiec_sample": dermahair_sample(),
              "our_synthetic_test_pairs": our_synthetic_test()}
    save_json(result, artifact_dir("logs") / "dermahair_sample_audit.json")
    print(json.dumps({k: result[k] for k in ("config_audit",)}, indent=1)[:1500])
    print(json.dumps(result["dermahair_akiec_sample"], indent=1)[:1800])
    print(json.dumps(result["our_synthetic_test_pairs"], indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
