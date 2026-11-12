"""Image-quality, noise-versus-blur and decision helpers for the DM-AHR diagnosis.

Everything here is pure numpy (LPIPS is imported lazily) so it runs in CPU unit
tests. The decision functions encode the thresholds pre-registered in
`reports/decision_log.md` (D22) so the reports cannot drift from them.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---- pre-registered thresholds (D22) -------------------------------------
PAPER_PSNR = 41.04
PAPER_TOLERANCE_DB = 2.0
MSE_RATIO_THRESHOLD = 1.5
UNDERTRAINED_GAIN_DB = 1.0
UNDERTRAINED_MAX_DROP_DB = 0.5
WARM_START_JUMP_FACTOR = 1.5
OVERFIT_MARGIN_DB = 3.0
VALID_ARM_MARGIN_DB = 3.0


def _arr(im: Image.Image | np.ndarray, size: int | None) -> np.ndarray:
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im = im.convert("RGB")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255.0


def psnr_ssim(pred, ref, size: int | None = 128, win_size: int = 7) -> dict:
    a, b = _arr(pred, size), _arr(ref, size)
    return {"psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
            "ssim": float(structural_similarity(b, a, data_range=1.0, channel_axis=2,
                                                win_size=win_size))}


def lpips_scores(preds: list, refs: list, size: int | None = 128, device: str = "cpu",
                 net: str = "alex") -> list[float]:
    """LPIPS distances; requires the `lpips` package (installed in the Modal image)."""
    import lpips  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    def prep(im):
        t = torch.from_numpy(_arr(im, size)).permute(2, 0, 1).float() * 2 - 1
        return t[None].to(device)

    out = []
    with torch.no_grad():
        for p, r in zip(preds, refs):
            out.append(float(model(prep(p), prep(r)).item()))
    return out


# ---- noise versus blur ---------------------------------------------------
def _gray(im, size: int | None) -> np.ndarray:
    return _arr(im, size).mean(axis=2)


def mean_abs_laplacian(im, size: int | None = 128) -> float:
    """High-frequency energy: mean absolute 4-neighbour Laplacian of the grey image."""
    a = _gray(im, size)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(np.abs(lap).mean())


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def noise_sigma_mad(im, size: int | None = 128, patch: int = 16, flat_quantile: float = 0.2) -> float:
    """Noise sigma by median absolute deviation of an Immerkaer residual in the flattest patches.

    The residual kernel suppresses smooth structure and amplifies white noise by
    exactly 6; the estimate is the median over the `flat_quantile` of
    non-overlapping patches with the lowest mean gradient magnitude.
    """
    a = _gray(im, size)
    resid = convolve(a, _IMMERKAER, mode="nearest")
    grad = np.hypot(sobel(a, axis=0), sobel(a, axis=1))
    h, w = a.shape
    sig, flat = [], []
    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            r = resid[y:y + patch, x:x + patch].ravel()
            sig.append(1.4826 * np.median(np.abs(r - np.median(r))) / 6.0)
            flat.append(grad[y:y + patch, x:x + patch].mean())
    sig, flat = np.asarray(sig), np.asarray(flat)
    keep = flat <= np.quantile(flat, flat_quantile)
    return float(np.median(sig[keep]))


def radial_power_spectrum(im, size: int | None = 128, n_bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of the Hann-windowed grey image.

    Returns (frequency in cycles/pixel at bin centres, mean power per bin).
    """
    a = _gray(im, size)
    a = a - a.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
    cy, cx = (np.array(power.shape) - 1) / 2.0
    yy, xx = np.indices(power.shape)
    r = np.hypot((yy - cy) / power.shape[0], (xx - cx) / power.shape[1])
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins)
    return (edges[:-1] + edges[1:]) / 2, sums / np.maximum(cnts, 1)


def high_freq_power_fraction(im, size: int | None = 128, cutoff: float = 0.25) -> float:
    f, p = radial_power_spectrum(im, size)
    return float(p[f >= cutoff].sum() / max(p.sum(), 1e-30))


# ---- pre-registered decision rules ----------------------------------------
def phase1_verdict(mean_psnr: float) -> dict:
    ok = mean_psnr >= PAPER_PSNR - PAPER_TOLERANCE_DB
    return {"rule": f"mean PSNR >= {PAPER_PSNR - PAPER_TOLERANCE_DB:.2f} dB (within {PAPER_TOLERANCE_DB} dB of {PAPER_PSNR})",
            "mean_psnr": float(mean_psnr), "adapter_validated": bool(ok),
            "statement": ("adapter validated on the paper's own data" if ok
                          else "paper numbers not reproduced; adapter and checkpoint loading under investigation")}


def mse_evidence(table: dict) -> dict:
    """table[ckpt][mode] = mean MSE. Trained-wrongly evidence per the D22 rule."""
    ours_l, ours_u = table["ours"]["legacy"], table["ours"]["upstream"]
    rel_l, rel_u = table["released"]["legacy"], table["released"]["upstream"]
    ours_ratio = ours_u / max(ours_l, 1e-12)
    rel_ratio = rel_l / max(rel_u, 1e-12)
    wrong = ours_ratio >= MSE_RATIO_THRESHOLD and rel_ratio >= MSE_RATIO_THRESHOLD
    return {"ours_upstream_over_legacy": float(ours_ratio),
            "released_legacy_over_upstream": float(rel_ratio),
            "threshold": MSE_RATIO_THRESHOLD, "trained_wrongly_evidence": bool(wrong)}


def checkpoint_trend(psnr_by_iter: dict[int, float]) -> dict:
    its = sorted(psnr_by_iter)
    vals = [psnr_by_iter[i] for i in its]
    gain = vals[-1] - vals[0]
    drops = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
    improving = gain >= UNDERTRAINED_GAIN_DB and max(drops, default=0.0) <= UNDERTRAINED_MAX_DROP_DB
    return {"iterations": its, "psnr": vals, "gain_db": float(gain),
            "max_successive_drop_db": float(max(drops, default=0.0)),
            "undertrained_pattern": bool(improving),
            "label": "monotonic improvement (undertrained pattern)" if improving
                     else "flat or erratic"}


def warm_start_jump(pre_l1: float, last100_mean_l1: float) -> dict:
    ratio = last100_mean_l1 / max(pre_l1, 1e-12)
    return {"pre_l1": float(pre_l1), "last100_mean_l1": float(last100_mean_l1),
            "ratio": float(ratio), "threshold": WARM_START_JUMP_FACTOR,
            "loss_jump": bool(ratio > WARM_START_JUMP_FACTOR)}


def overfit_pass(mean_psnr: float, hairy_psnr: float) -> dict:
    return {"mean_psnr": float(mean_psnr), "hairy_psnr": float(hairy_psnr),
            "margin_db": float(mean_psnr - hairy_psnr), "required_margin_db": OVERFIT_MARGIN_DB,
            "passed": bool(mean_psnr - hairy_psnr >= OVERFIT_MARGIN_DB)}


def arm_is_valid(psnr: float, ssim: float, hairy_psnr: float, hairy_ssim: float) -> dict:
    ok = psnr >= hairy_psnr + VALID_ARM_MARGIN_DB and ssim > hairy_ssim
    return {"psnr": float(psnr), "ssim": float(ssim), "hairy_psnr": float(hairy_psnr),
            "hairy_ssim": float(hairy_ssim), "valid": bool(ok),
            "rule": f"PSNR >= hairy + {VALID_ARM_MARGIN_DB} dB and SSIM > hairy SSIM (validation pairs)"}


def phase2_branch(mse: dict | None, trend: dict | None, warm: dict | None,
                  overfit: dict | None) -> dict:
    """Combine the experiment outcomes into the (a)/(b)/(c) branch; None = not run."""
    missing = [n for n, v in (("noise-prediction MSE", mse), ("checkpoint trend", trend),
                              ("warm start", warm), ("overfit", overfit)) if v is None]
    if missing:
        return {"branch": None, "status": "NOT RUN", "missing": missing}
    bug = (mse["trained_wrongly_evidence"] or
           (warm["legacy"]["loss_jump"] and not warm["fixed"]["loss_jump"]) or
           (not overfit["passed"] and not trend["undertrained_pattern"]))
    if bug:
        return {"branch": "c", "status": "trainer bug",
                "action": "fix trainer, retrain diffusion within budget, regenerate Arm C, retrain 3 ViT seeds"}
    if trend["undertrained_pattern"]:
        return {"branch": "b", "status": "undertrained",
                "action": "keep Arm C as under-resourced (paper-stated 20,000 iterations vs 725,000 released)"}
    return {"branch": None, "status": "inconclusive", "action": "report as inconclusive"}
