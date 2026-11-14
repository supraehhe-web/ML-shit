"""CPU check: how large are training gradients, i.e. does clip_grad_norm_(1.0) engage?

Upstream does not clip; our trainer clips at 1.0. Fixed-t gradient norms at the new
checkpoint and the released one, train mode (dropout on) as during training.
Validation pairs only. Writes artifacts/diagnosis/gradnorm.json with provenance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm_ahr_vit.config import load_config  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import load_model, preprocess  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("diag31", ROOT / "scripts" / "31_local_diagnosis.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

LEVELS = (200, 800, 1400, 1900)


def main() -> int:
    torch.set_num_threads(1)
    cfg = load_config(str(ROOT / "configs" / "dmahr.json"))
    manifest = Path("/root/local/diag/man/synthetic_pairs.csv")
    val, refs, hairy = diag.load_val(manifest, Path("/root/local/diag/data"), 8)
    x0, cond = preprocess(refs, 128), preprocess(hairy, 128)
    ckpts = {"new_fixed_20k": "/root/local/diag/ckpt/new_best.pth",
             "released": "/root/local/diag/ckpt/released.pth"}
    out = {"clip_threshold": 1.0, "levels": list(LEVELS), "models": {}}
    for name, path in ckpts.items():
        net = load_model(cfg, Path(path), device="cpu")
        net.train()
        norms = {t: [] for t in LEVELS}
        torch.manual_seed(0)
        for t in LEVELS:
            a = float(net.sqrt_alphas_cumprod_prev[t + 1])
            for s in range(0, 8, 4):
                net.zero_grad(set_to_none=True)
                eps = torch.randn_like(x0[s:s + 4])
                x_t = a * x0[s:s + 4] + (1 - a**2) ** 0.5 * eps
                lvl = torch.full((x_t.shape[0], 1), a)
                loss = (net.denoise_fn(torch.cat([cond[s:s + 4], x_t], 1), lvl) - eps).abs().mean()
                loss.backward()
                g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in net.parameters() if p.grad is not None))
                norms[t].append(float(g))
            print(name, t, [round(v, 3) for v in norms[t]], flush=True)
        out["models"][name] = {str(t): v for t, v in norms.items()}
        out["models"][name]["fraction_above_clip"] = float(np.mean([v > 1.0 for t in LEVELS for v in norms[t]]))
        del net
    out["provenance"] = diag.provenance(ckpts, manifest, val["pair_id"].tolist())
    (ROOT / "artifacts" / "diagnosis" / "gradnorm.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
