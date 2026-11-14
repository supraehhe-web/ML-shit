"""Released DM-AHR checkpoint at the paper's quoted inference setting (T = 10) versus 2,000 steps.

The paper says "the inference number is 10 (T = 10)"; the released configs use 2,000 timesteps for
both training and testing. Validation-source synthetic pairs only; no test data.
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
from dm_ahr_vit.data.masks import resolve  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import CHECKPOINT_NAME, clean_batch, load_model  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
PAIRS = 50


def main() -> int:
    cfg = load_config("configs/dmahr.json")
    win = int(load_config("configs/sampling.json").get("ssim_win_size", 7))
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    val = pairs[pairs["split"] == "val"].head(PAIRS)
    refs, conds = [], []
    for r in val.itertuples(index=False):
        with Image.open(resolve(r.reference_path)) as fh:
            refs.append(fh.convert("RGB"))
        with Image.open(resolve(r.hairy_path)) as fh:
            conds.append(fh.convert("RGB"))

    def score(outs):
        ps, ss = [], []
        for o, ref in zip(outs, refs):
            a = np.asarray(o.convert("RGB").resize(ref.size, Image.Resampling.BICUBIC),
                           dtype=np.float64) / 255.0
            b = np.asarray(ref, dtype=np.float64) / 255.0
            ps.append(peak_signal_noise_ratio(b, a, data_range=1.0))
            ss.append(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=win))
        return float(np.mean(ps)), float(np.std(ps)), float(np.mean(ss))

    floor = score(conds)
    out = {"pairs": len(refs), "hairy_floor": {"psnr": floor[0], "ssim": floor[2]}, "rows": []}
    net = load_model(cfg, data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME, device="cuda")
    with timed("paper_t10_check", log) as tmr:
        for label, steps, respaced in (("respaced", 10, True), ("respaced", 25, True),
                                       ("respaced", 50, True), ("full", 2000, False)):
            outs, _ = clean_batch(net, conds, cfg, device="cuda", seed=42,
                                  num_timesteps=None if not respaced else steps,
                                  use_bf16=True, respaced=respaced)
            p, sd, s = score(outs)
            out["rows"].append({"mode": label, "steps": steps, "psnr": p, "psnr_sd": sd, "ssim": s})
            log.info("%-8s steps=%-5d PSNR %.2f SSIM %.3f", label, steps, p, s)
    out["seconds"] = tmr["seconds"]
    save_json(out, artifact_dir("logs") / "paper_t10_check.json")
    record_run("paper_t10_check", gpu="cuda", seconds=tmr["seconds"], pairs=len(refs))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
