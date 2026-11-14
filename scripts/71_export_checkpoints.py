"""Export deliverable checkpoints with a hash manifest.

ViT checkpoints are re-saved weights-only ({"model": state_dict, "meta": {...}}), which
`dm_ahr_vit.vit.model.load_checkpoint` reads directly; the training files also hold optimiser
state and are about three times larger. Diffusion checkpoints are copied as they are.
The released third-party DM-AHR weights are NOT copied (licence caveat in reports/license_audit.md);
their hash is listed so the file can be matched to the upstream download.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import sys
from pathlib import Path as _P

import torch

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import sha256_file
from dm_ahr_vit.paths import artifact_dir, data_root

ARMS = ("A", "B", "Astar", "Cv2", "C", "Creleased")
EIGHT = (42, 123, 2026, 7, 11, 17, 23, 31)
THREE = (42, 123, 2026)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = artifact_dir("checkpoints")
    out = Path(args.out) if args.out else artifact_dir("exports")
    (out / "vit").mkdir(parents=True, exist_ok=True)
    (out / "diffusion").mkdir(parents=True, exist_ok=True)
    rows = []

    for arm in ARMS:
        for seed in (EIGHT if arm in ("A", "B", "Astar", "Cv2") else THREE):
            src = ck / "vit" / f"{arm}_seed{seed}" / "best.pt"
            if not src.exists():
                print("MISSING", src)
                continue
            dst = out / "vit" / f"vit_b16_{arm}_seed{seed}.pt"
            state = torch.load(src, map_location="cpu", weights_only=False)
            meta = {k: state[k] for k in state if k not in ("model", "optimizer", "scheduler")
                    and isinstance(state[k], (int, float, str, bool))}
            meta.update({"arm": arm, "seed": seed, "source_sha256": sha256_file(src),
                         "git_commit": args.commit, "architecture": "torchvision vit_b_16, 7 classes"})
            torch.save({"model": state["model"], "meta": meta}, dst)
            rows.append({"kind": "vit", "arm": arm, "seed": seed, "file": str(dst.relative_to(out)),
                         "bytes": dst.stat().st_size, "sha256": sha256_file(dst),
                         "source": str(src.relative_to(ck)), "source_sha256": meta["source_sha256"]})
            print("exported", dst.name, flush=True)

    diffusion = {"diffusion/cv2_sampler_dmahr_v2_final_gen.pth": ck / "dmahr_v2" / "final_gen.pth",
                 "diffusion/repaired_trainer_iter050000_gen.pth": ck / "dmahr" / "gen_iter050000.pth"}
    for rel, src in diffusion.items():
        if src.exists():
            dst = out / rel
            shutil.copy2(src, dst)
            rows.append({"kind": "diffusion", "arm": "Cv2" if "cv2" in rel else "repaired", "seed": "",
                         "file": rel, "bytes": dst.stat().st_size, "sha256": sha256_file(dst),
                         "source": str(src.relative_to(ck)), "source_sha256": sha256_file(src)})
    released = data_root() / "raw" / "dmahr_weights" / "I725000_E1083_gen.pth"
    if released.exists():
        rows.append({"kind": "reference_not_copied", "arm": "Creleased", "seed": "",
                     "file": "(not copied: third-party weights, see reports/license_audit.md)",
                     "bytes": released.stat().st_size, "sha256": sha256_file(released),
                     "source": "raw/dmahr_weights/" + released.name, "source_sha256": sha256_file(released)})

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "README.txt").write_text(
        "DM-AHR-ViT exported checkpoints (research prototype only; not for clinical use).\n"
        "vit/*.pt: torch.load(path)['model'] is a torchvision vit_b_16 state_dict (7 classes).\n"
        "diffusion/*.pth: DM-AHR U-Net state dicts; load with dm_ahr_vit.diffusion.dmahr_adapter.load_model.\n"
        "manifest.csv lists bytes and SHA-256 for every file; verify with sha256sum before use.\n")
    print(f"{len(rows)} manifest rows, {sum(int(r['bytes']) for r in rows) / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
