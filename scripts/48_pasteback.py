"""Stage D: tune, check and build the full-resolution paste-back arm (Cv2p).

tune   grid search on the 182 VALIDATION synthetic pairs only (protocol scorer); writes
       artifacts/logs/stageD_val_tuning.json with every setting, baselines and the choice
real   frozen setting on real VALIDATION photos: full-resolution detector coverage before and
       after (original, DullRazor, Cv2p) and contact sheets; ids are passed in, never test
build  write the Cv2p arm: real images (all splits), or the val / test synthetic pairs

The test split is only read by `build --target pairs-test|real` after the gates in
reports/prereg_v3_pasteback.md have passed and the parameters are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from importlib import import_module
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm_ahr_vit.config import load_config, load_frozen, save_json  # noqa: E402
from dm_ahr_vit.hair.dullrazor import apply_dullrazor  # noqa: E402
from dm_ahr_vit.hair.pasteback import pasteback  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair, hair_response  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, isic_images_dir, manifest_dir, processed_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

iq = import_module("07b_image_quality")
log = setup_logging()

GRIDS = {
    # (tau, dilate, refine, thin_T, grow)
    "coarse": list(itertools.product((8, 16, 24, 32), (0, 1, 2), (False, True), (0,), (2,))),
    "thin": list(itertools.product((2, 4, 8), (1, 2), (True,), (18, 30), (1, 2, 3))),
}
OUT_NAME = {"coarse": "stageD_val_tuning", "thin": "stageD_val_tuning_thin"}


def _pairs(split: str) -> pd.DataFrame:
    p = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    return p[p["split"] == split]


def _load(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def thin_mask(img: np.ndarray, screen: dict, thr: float) -> np.ndarray:
    """Thin dark-hair pixels: black-hat response above thr, no component filter or dilation."""
    dark, _ = hair_response(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), screen)
    return (dark > thr).astype(np.uint8)


def _mask_for(img, screen, thin_t, refine, cache):
    """Full-resolution restriction mask for a setting; cache holds per-image precomputation."""
    if thin_t:
        key = ("thin", thin_t)
        if key not in cache:
            cache[key] = thin_mask(img, screen, thin_t)
        return cache[key]
    if refine:
        if "det" not in cache:
            cache["det"] = detect_hair(img, screen)[0]
        return cache["det"]
    return None


def _assert_val(ids) -> None:
    spl = pd.read_csv(manifest_dir() / "splits.csv").set_index("image_id")["split"]
    bad = [i for i in ids if spl.get(i) != "val"]
    if bad:
        raise SystemExit(f"refusing non-validation ids: {bad[:3]}")


def cmd_tune(a) -> None:
    variant = a.variant
    samp = load_config("configs/sampling.json")
    screen = load_frozen("hair_screen", "configs/hair_screen.json")
    dull = load_frozen("dullrazor", "configs/dullrazor.json")
    pairs = _pairs("val")
    assert set(pairs["split"]) == {"val"}
    items = []
    for r in pairs.itertuples(index=False):
        ref, hairy = _load(iq.resolve(r.reference_path)), _load(iq.resolve(r.hairy_path))
        fill = _load(processed_dir("synth") / "val" / "cleaned_Cv2" / f"{r.image_id}.png")
        det, _ = detect_hair(hairy, screen)
        items.append((ref, hairy, fill, det, {}))
    base = {"hairy_input": [], "Cv2_alone": [], "B_dullrazor": []}
    for ref, hairy, fill, det, _c in items:
        base["hairy_input"].append(iq.score(hairy, ref, cfg=samp))
        base["Cv2_alone"].append(iq.score(fill, ref, cfg=samp))
        base["B_dullrazor"].append(iq.score(apply_dullrazor(hairy, det, dull), ref, cfg=samp))
    baselines = {k: {"psnr": float(np.mean([m["psnr"] for m in v])),
                     "ssim": float(np.mean([m["ssim"] for m in v]))} for k, v in base.items()}
    rows = []
    for tau, dil, ref_on, thin_t, grow in GRIDS[variant]:
        sc, full, frac = [], [], []
        for ref, hairy, fill, det, cache in items:
            cache.setdefault("det", det)
            out, info = pasteback(hairy, fill, tau=tau, dilate=dil, refine_dilate=grow,
                                  det_mask=_mask_for(hairy, screen, thin_t, ref_on, cache))
            sc.append(iq.score(out, ref, cfg=samp))
            full.append(peak_signal_noise_ratio(ref, out, data_range=255))
            frac.append(info["mask_fraction"])
        rows.append({"tau": tau, "dilate": dil, "refine": ref_on, "thin_T": thin_t,
                     "grow": grow,
                     "psnr128": float(np.mean([m["psnr"] for m in sc])),
                     "ssim128": float(np.mean([m["ssim"] for m in sc])),
                     "psnr_full": float(np.mean(full)), "mask_fraction": float(np.mean(frac))})
        log.info("tau=%s dilate=%s refine=%s T=%s grow=%s psnr128=%.2f ssim128=%.3f full=%.2f "
                 "mask=%.4f", tau, dil, ref_on, thin_t, grow, rows[-1]["psnr128"], rows[-1]["ssim128"],
                 rows[-1]["psnr_full"], rows[-1]["mask_fraction"])
    top = max(r["psnr128"] for r in rows)
    near = [r for r in rows if top - r["psnr128"] <= 0.05]
    best = min(near, key=lambda r: r["mask_fraction"])
    gates = {"D1_psnr_at_least_cv2_plus_3": best["psnr128"] >= baselines["Cv2_alone"]["psnr"] + 3.0,
             "D2_ssim_at_least_0.90": best["ssim128"] >= 0.90}
    save_json({"variant": variant, "split": "val", "pairs": len(items),
               "pair_ids_sha256": hashlib.sha256("\n".join(pairs["pair_id"]).encode()).hexdigest(),
               "baselines": baselines, "grid": rows, "chosen": best, "gates": gates,
               "screen_config_hash": screen.get("config_hash")},
              artifact_dir("logs") / f"{OUT_NAME[variant]}.json")
    record_run("stageD_tune", gpu=None, seconds=0.0, settings=len(rows))
    print(json.dumps({"baselines": baselines, "chosen": best, "gates": gates}, indent=1))


def _coverage(img: np.ndarray, screen: dict) -> float:
    return detect_hair(img, screen)[1]["coverage"]


def cmd_real(a) -> None:
    screen = load_frozen("hair_screen", "configs/hair_screen.json")
    ids = [i for i in a.ids.split(",") if i]
    _assert_val(ids)
    masks = pd.read_csv(manifest_dir() / "mask_manifest_x1.csv").set_index("image_id")
    out_dir = processed_dir("diag_pasteback") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for iid in ids:
        orig = _load(isic_images_dir() / f"{iid}.jpg")
        fill = _load(processed_dir("Cv2") / f"{iid}.png")
        b = _load(processed_dir("B") / f"{iid}.png")
        det, st0 = detect_hair(orig, screen)
        cache = {"det": det}
        out, info = pasteback(orig, fill, tau=a.tau, dilate=a.dilate, refine_dilate=a.grow,
                              det_mask=_mask_for(orig, screen, a.thin_T, a.refine, cache))
        les_b = les_c = np.nan
        if iid in masks.index and bool(masks.loc[iid, "has_mask"]):
            lm = _load(iq.resolve(masks.loc[iid, "mask_path"]))[..., 0] > 127
            if lm.any():
                chg = lambda x: (np.abs(x.astype(np.int16) - orig.astype(np.int16)).max(axis=2) > 10)  # noqa: E731
                les_b, les_c = float(chg(b)[lm].mean()), float(chg(out)[lm].mean())
        Image.fromarray(out).save(out_dir / f"{iid}.png")
        c0 = st0["coverage"]
        rows.append({"image_id": iid, "cov_orig": c0, "cov_B": _coverage(b, screen),
                     "cov_Cv2p": _coverage(out, screen),
                     "remaining_B": _coverage(b, screen) / c0 if c0 > 0 else np.nan,
                     "remaining_Cv2p": _coverage(out, screen) / c0 if c0 > 0 else np.nan,
                     "mask_fraction": info["mask_fraction"],
                     "lesion_changed_B": les_b, "lesion_changed_Cv2p": les_c})
        tiles.append((iid, orig, b, out))
    df = pd.DataFrame(rows)
    summ = {"name": a.name, "n": len(df), "tau": a.tau, "dilate": a.dilate, "refine": a.refine,
            "median_cov_orig": float(df["cov_orig"].median()),
            "median_remaining_B": float(df["remaining_B"].median()),
            "median_remaining_Cv2p": float(df["remaining_Cv2p"].median()),
            "mean_mask_fraction": float(df["mask_fraction"].mean()),
            "lesion_masks_used": int(df["lesion_changed_B"].notna().sum()),
            "mean_lesion_changed_B": float(df["lesion_changed_B"].mean()),
            "mean_lesion_changed_Cv2p": float(df["lesion_changed_Cv2p"].mean()),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    summ["gate_D3_median_remaining_at_most_0.5"] = summ["median_remaining_Cv2p"] <= 0.5
    summ["gate_D6_lesion_change_not_above_B"] = (
        summ["mean_lesion_changed_Cv2p"] <= summ["mean_lesion_changed_B"])
    summ["thin_T"], summ["grow"] = a.thin_T, a.grow
    df.to_csv(artifact_dir("diagnosis") / f"pasteback_real_{a.name}.csv", index=False)
    save_json(summ, artifact_dir("diagnosis") / f"pasteback_real_{a.name}.json")
    for k in range(0, len(tiles), 4):
        rws = []
        for iid, o, bb, c in tiles[k:k + 4]:
            row = [cv2.resize(x, (400, 300), interpolation=cv2.INTER_AREA) for x in (o, bb, c)]
            im = Image.fromarray(np.concatenate(row, axis=1))
            ImageDraw.Draw(im).text((4, 4), iid[-7:], fill=(255, 255, 0))
            rws.append(np.asarray(im))
        Image.fromarray(np.concatenate(rws, axis=0)).save(
            artifact_dir("diagnosis") / f"pasteback_sheet_{a.name}_{k // 4}.png")
    print(json.dumps(summ, indent=1))


def _build_one(job):
    kind, iid, src, fill_path, dst = job
    dst = Path(dst)
    if dst.exists():
        return 0
    orig = _load(src)
    if fill_path is None:
        out = orig
    else:
        det = _mask_for(orig, P["screen"], P["thin_T"], P["refine"], {})
        out, _ = pasteback(orig, _load(fill_path), tau=P["tau"], dilate=P["dilate"],
                           refine_dilate=P["grow"], det_mask=det)
    Image.fromarray(out).save(dst)
    return 1


P: dict = {}


def _init(p):
    P.update(p)


def cmd_build(a) -> None:
    screen = load_frozen("hair_screen", "configs/hair_screen.json")
    params = {"tau": a.tau, "dilate": a.dilate, "refine": bool(a.refine),
              "thin_T": a.thin_T, "grow": a.grow, "screen_config_hash": screen.get("config_hash")}
    jobs = []
    if a.target == "real":
        scr = pd.read_csv(manifest_dir() / "hair_screening.csv")
        out_dir = processed_dir("Cv2p")
        for r in scr.itertuples(index=False):
            hair = bool(r.hair_present)
            jobs.append(("real", r.image_id, isic_images_dir() / f"{r.image_id}.jpg",
                         processed_dir("Cv2") / f"{r.image_id}.png" if hair else None,
                         out_dir / f"{r.image_id}.png"))
    else:
        split = a.target.split("-")[1]
        out_dir = processed_dir("synth") / split / "cleaned_Cv2p"
        for r in _pairs(split).itertuples(index=False):
            jobs.append(("pair", r.image_id, iq.resolve(r.hairy_path),
                         processed_dir("synth") / split / "cleaned_Cv2" / f"{r.image_id}.png",
                         out_dir / f"{r.image_id}.png"))
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "params.json"
    if marker.exists() and json.loads(marker.read_text()) != params:
        raise SystemExit(f"{out_dir} was built with different parameters; refusing to mix")
    marker.write_text(json.dumps(params, indent=1))
    P.update({**params, "screen": screen})
    with Pool(a.workers, initializer=_init, initargs=({**params, "screen": screen},)) as pool:
        written = sum(pool.map(_build_one, jobs, chunksize=16))
    n_files = len(list(out_dir.glob("*.png")))
    save_json({"target": a.target, "params": params, "jobs": len(jobs), "written_now": written,
               "files": n_files}, artifact_dir("logs") / f"stageD_build_{a.target}.json")
    save_json(params, artifact_dir("logs") / "stageD_params.json")
    print(json.dumps({"target": a.target, "jobs": len(jobs), "written_now": written,
                      "files": n_files}))
    assert n_files == len(jobs), "missing outputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--variant", choices=list(GRIDS), default="coarse")
    for name in ("real", "build"):
        s = sub.add_parser(name)
        s.add_argument("--tau", type=float, required=True)
        s.add_argument("--dilate", type=int, required=True)
        s.add_argument("--refine", action="store_true")
        s.add_argument("--thin-T", type=float, default=0.0, help="0 = use the detector mask")
        s.add_argument("--grow", type=int, default=2)
        if name == "real":
            s.add_argument("--ids", required=True)
            s.add_argument("--name", required=True)
        else:
            s.add_argument("--target", required=True, choices=["real", "pairs-val", "pairs-test"])
            s.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"tune": cmd_tune, "real": cmd_real, "build": cmd_build}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
