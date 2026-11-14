"""Does the fixed trainer preserve weights it did not produce?

Fine-tunes the authors' released checkpoint for a short run, then scores the
result through the normal sampling path on validation pairs. If the trainer is
correct the released quality should hold or improve; a collapse means the
trainer still disagrees with the sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    CHECKPOINT_NAME,
    clean_batch,
    load_model,
)
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()

STEPS = 25
MAX_PAIRS = 30


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def score(net, pairs, cfg, device, win) -> dict:
    refs, conds = [], []
    for r in pairs.itertuples(index=False):
        with Image.open(resolve(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(resolve(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))
    outs, _ = clean_batch(net, conds, cfg, device=device, seed=42,
                          num_timesteps=STEPS, use_bf16=True)
    ps, ss = [], []
    for out, ref in zip(outs, refs):
        a = np.asarray(out.convert("RGB").resize(ref.size, Image.Resampling.BICUBIC),
                       dtype=np.float64) / 255.0
        b = np.asarray(ref, dtype=np.float64) / 255.0
        ps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
        ss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                        win_size=win))
    # Hairy floor on the same pairs.
    fps, fss = [], []
    for noisy, ref in zip(conds, refs):
        a = np.asarray(noisy.resize(ref.size, Image.Resampling.BICUBIC),
                       dtype=np.float64) / 255.0
        b = np.asarray(ref, dtype=np.float64) / 255.0
        fps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
        fss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                         win_size=win))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "hairy_psnr": float(np.mean(fps)), "hairy_ssim": float(np.mean(fss)),
            "n": len(ps)}


def main() -> int:
    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    device = "cuda"

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].head(MAX_PAIRS).reset_index(drop=True)

    released = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    fine = artifact_dir("checkpoints", "dmahr_warmstart_check") / "best_gen.pth"
    if not fine.exists():
        fine = artifact_dir("checkpoints", "dmahr_warmstart_check") / "final_gen.pth"
    if not fine.exists():
        log.error("warm-start checkpoint missing; run the 1000-iteration warm start first")
        return 1

    out: dict = {"steps": STEPS, "pairs": int(len(val))}
    with timed("warmstart_check", log) as tmr:
        for name, path in (("released", released), ("warmstart_1k", fine)):
            net = load_model(cfg, path, device=device)
            out[name] = score(net, val, cfg, device, win)
            log.info("%s: PSNR %.2f SSIM %.3f (floor %.2f / %.3f)",
                     name, out[name]["psnr"], out[name]["ssim"],
                     out[name]["hairy_psnr"], out[name]["hairy_ssim"])
            del net

    delta = out["warmstart_1k"]["psnr"] - out["released"]["psnr"]
    out["psnr_change"] = delta
    out["verdict"] = ("trainer consistent" if delta > -2.0 else "trainer still wrong")
    out["seconds"] = tmr["seconds"]
    save_json(out, artifact_dir("logs") / "warmstart_check.json")
    record_run("warmstart_check", gpu=device, seconds=tmr["seconds"],
               psnr_change=delta)
    print(f"released {out['released']['psnr']:.2f} -> warmstart {out['warmstart_1k']['psnr']:.2f} "
          f"(change {delta:+.2f} dB) => {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
