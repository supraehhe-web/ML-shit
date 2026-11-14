"""Can the committed code reproduce the recorded 25-step released result?

The `Creleased` images were sampled at 25 steps and scored 26.91 dB on
validation pairs. This checks whether the repository can regenerate that, which
it could not while only schedule truncation was available.

Compares schedule truncation against DDIM respacing at several step counts.
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
PAIRS = 25


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    marker = "/data/"
    if marker in str(p):
        return data_root() / str(p).split(marker, 1)[1]
    return p


def main() -> int:
    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    device = "cuda"

    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].head(PAIRS).reset_index(drop=True)
    refs, conds = [], []
    for r in val.itertuples(index=False):
        with Image.open(resolve(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(resolve(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))

    def score(outs):
        ps, ss = [], []
        for out, ref in zip(outs, refs):
            a = np.asarray(out.convert("RGB").resize(ref.size, Image.Resampling.BICUBIC),
                           dtype=np.float64) / 255.0
            b = np.asarray(ref, dtype=np.float64) / 255.0
            ps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
            ss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                            win_size=win))
        return float(np.mean(ps)), float(np.mean(ss))

    fps, fss = [], []
    for noisy, ref in zip(conds, refs):
        a = np.asarray(noisy.resize(ref.size, Image.Resampling.BICUBIC),
                       dtype=np.float64) / 255.0
        b = np.asarray(ref, dtype=np.float64) / 255.0
        fps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
        fss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=win))

    ckpt = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    out: dict = {"pairs": int(len(val)),
                 "hairy_floor": {"psnr": float(np.mean(fps)), "ssim": float(np.mean(fss))},
                 "recorded_25_step_psnr": 26.9148, "rows": []}

    with timed("respacing_check", log) as tmr:
        for mode in ("truncated", "respaced", "jsd25"):
            net = load_model(cfg, ckpt, device=device)
            for steps in (25, 50, 250):
                try:
                    outs, _ = clean_batch(net, conds, cfg, device=device, seed=42,
                                          num_timesteps=steps, use_bf16=True,
                                          respaced=(mode == "respaced"),
                                          schedule=("jsd" if mode == "jsd25" else None))
                    p, s = score(outs)
                    out["rows"].append({"mode": mode, "steps": steps, "psnr": p,
                                        "ssim": s, "status": "ok"})
                    log.info("%-9s steps=%-4d PSNR %.2f SSIM %.3f", mode, steps, p, s)
                except Exception as exc:  # noqa: BLE001
                    out["rows"].append({"mode": mode, "steps": steps, "psnr": None,
                                        "ssim": None, "status": f"failed: {type(exc).__name__}"})
                    log.warning("%-9s steps=%-4d failed: %s", mode, steps, exc)
            del net
        torch_empty = None  # noqa: F841
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    out["seconds"] = tmr["seconds"]
    best = max((r for r in out["rows"] if r["mode"] == "respaced"),
               key=lambda r: r["psnr"], default=None)
    out["respaced_matches_recorded"] = bool(
        best is not None and abs(best["psnr"] - out["recorded_25_step_psnr"]) <= 1.0
    )
    save_json(out, artifact_dir("logs") / "respacing_check.json")
    record_run("respacing_check", gpu=device, seconds=tmr["seconds"],
               pairs=len(val))
    for r in out["rows"]:
        print(r)
    print("floor", round(out["hairy_floor"]["psnr"], 2),
          "| matches recorded 25-step:", out["respaced_matches_recorded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
