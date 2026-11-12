"""Explainability maps for torchvision ViT-B/16.

Grad-CAM is taken on `conv_proj`, whose output is the (B, 768, 14, 14) patch
grid. Attention rollout multiplies identity-blended attention matrices captured
from the real self-attention modules.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F


def _patch_grid(model) -> int:
    size = model.image_size if isinstance(model.image_size, int) else model.image_size[0]
    return size // model.patch_size


class GradCAM:
    """Grad-CAM on the patch-embedding grid (conv_proj)."""

    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer if layer is not None else model.conv_proj
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = None

    def __enter__(self):
        def fwd_hook(_module, _inp, out):
            self.activations = out
            if out.requires_grad:
                out.register_hook(self._save_grad)

        self._handle = self.layer.register_forward_hook(fwd_hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _save_grad(self, grad):
        self.gradients = grad

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a normalized (grid, grid) CAM for one image tensor (1,3,H,W)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        # Grad-CAM needs a gradient path back to the patch grid, so the input
        # must require grad even though the model is in eval mode.
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            score = logits[0, class_idx]
            score.backward(retain_graph=False)
        if self.gradients is None:
            raise RuntimeError("Grad-CAM captured no gradients on conv_proj")
        act = self.activations.detach()[0]           # (768, g, g)
        grad = self.gradients.detach()[0]            # (768, g, g)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0))
        cam = cam - cam.min()
        denom = cam.max()
        if float(denom) > 0:
            cam = cam / denom
        return cam.cpu().numpy()


@contextlib.contextmanager
def capture_attention(model):
    """Capture averaged attention weights from every block's self_attn module."""
    captured: list[torch.Tensor] = []
    originals = []

    for block in model.encoder.layers:
        # torchvision names this module `self_attention`; timm-style code uses
        # `self_attn`. Accept either so the hook does not depend on the fork.
        attn = getattr(block, "self_attention", None) or getattr(block, "self_attn", None)
        if attn is None:
            raise AttributeError("encoder block exposes no self-attention module")
        originals.append((attn, attn.forward))

        def make(original_forward, store):
            def forward(query, key, value, *args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                out, weights = original_forward(query, key, value, *args, **kwargs)
                store.append(weights.detach())
                # torchvision unpacks `x, _ = self_attention(...)`; keep that shape.
                return out, None

            return forward

        attn.forward = make(attn.forward, captured)

    try:
        yield captured
    finally:
        for module, original in originals:
            module.forward = original


@torch.no_grad()
def attention_rollout(model, x: torch.Tensor, discard_ratio: float = 0.9) -> np.ndarray:
    """Identity-blended attention rollout, returned as a (grid, grid) map."""
    model.eval()
    with capture_attention(model) as captured:
        model(x)
    if not captured:
        raise RuntimeError("no attention weights captured")

    tokens = captured[0].shape[-1]
    result = torch.eye(tokens)
    for attn in captured:
        a = attn[0].float().cpu()
        if discard_ratio > 0:
            flat = a.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                idx = flat.topk(k, dim=1, largest=False).indices
                flat.scatter_(1, idx, 0.0)
                a = flat.view_as(a)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        blended = 0.5 * a + 0.5 * torch.eye(tokens)
        result = blended @ result

    cls_attention = result[0, 1:]
    grid = _patch_grid(model)
    mask = cls_attention.reshape(grid, grid).numpy()
    mask = mask - mask.min()
    denom = mask.max()
    if denom > 0:
        mask = mask / denom
    return mask


def upsample_map(cam: np.ndarray, size: int = 224) -> np.ndarray:
    t = torch.from_numpy(cam).float()[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def top_fraction_mask(saliency: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    flat = saliency.ravel()
    k = max(1, int(round(flat.size * fraction)))
    idx = np.argpartition(-flat, k - 1)[:k]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(saliency.shape)


def lesion_overlap(saliency: np.ndarray, mask: np.ndarray, fraction: float = 0.2) -> float:
    """Mask recall: the share of lesion pixels inside the top-`fraction` most salient pixels, |top ∩ mask| / |mask|. This is recall of the lesion, not precision of the salient region."""
    top = top_fraction_mask(saliency, fraction)
    lesion_area = float(mask.sum())
    if lesion_area == 0:
        return float("nan")
    return float((top & mask.astype(bool)).sum() / lesion_area)


def pointing_game(saliency: np.ndarray, mask: np.ndarray, tolerance: int = 0) -> float:
    """Pointing game: 1.0 if the single most salient pixel lies on the lesion mask (optionally dilated by `tolerance`), else 0.0."""
    m = mask.astype(bool)
    if tolerance > 0:
        k = 2 * tolerance + 1
        m = (
            F.max_pool2d(
                torch.from_numpy(m.astype(np.float32))[None, None], k, stride=1, padding=tolerance
            )[0, 0].numpy()
            > 0
        )
    if m.sum() == 0:
        return float("nan")
    idx = int(np.argmax(saliency))
    r, c = divmod(idx, saliency.shape[1])
    return float(bool(m[r, c]))


def uniform_chance_overlap(mask: np.ndarray, fraction: float = 0.2, seed: int = 0,
                           n: int = 1) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sal = rng.random(mask.shape)
        vals.append(lesion_overlap(sal, mask, fraction))
    return float(np.nanmean(vals))


def gaussian_chance_map(size: int = 224, sigma_frac: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    sigma = size * sigma_frac
    g = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2)))
    return g / g.max()


def overlap_chance(fraction: float = 0.2) -> float:
    """Chance level of the overlap metric: a random salient region of area `fraction` recovers that same share of any lesion, so the expectation equals `fraction` for every mask size."""
    return float(fraction)


def pointing_chance(mask: np.ndarray) -> float:
    """Chance level of the pointing game: the probability that a uniformly random argmax lands on the lesion, which is the lesion's share of the image area (mask.mean())."""
    return float(np.asarray(mask, dtype=bool).mean())


def random_argmax_pointing(mask: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    """Monte-Carlo estimate of the pointing chance level from random salience maps, used to check `pointing_chance` empirically."""
    rng = np.random.default_rng(seed)
    flat = np.asarray(mask, dtype=bool).ravel()
    hits = flat[rng.integers(0, flat.size, size=n)]
    return float(hits.mean())


def overlap_ceiling(mask: np.ndarray, fraction: float = 0.2) -> float:
    """Largest attainable overlap for this mask, min(1, fraction / area share): a salient region of area `fraction` cannot cover more than that share of a larger lesion."""
    area = float(np.asarray(mask, dtype=bool).mean())
    if area <= 0:
        return float("nan")
    return float(min(1.0, fraction / area))


def ceiling_normalised_overlap(overlap: float, mask: np.ndarray, fraction: float = 0.2) -> float:
    """Overlap divided by its ceiling for the same mask, so large lesions are not penalised for the fixed salient-region budget."""
    ceiling = overlap_ceiling(mask, fraction)
    if not np.isfinite(ceiling) or ceiling <= 0:
        return float("nan")
    return float(overlap / ceiling)


def mask_centroid_offset(mask: np.ndarray) -> tuple[float, float]:
    """Lesion centroid offset from the image centre as (dy, dx), in fractions of the image height and width (0 = centred, +-0.5 = at the border)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan"), float("nan")
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return (float(ys.mean() + 0.5) / h - 0.5, float(xs.mean() + 0.5) / w - 0.5)


def mask_contains_centre(mask: np.ndarray) -> bool:
    """True when the pixel at the image centre belongs to the lesion mask."""
    m = np.asarray(mask, dtype=bool)
    return bool(m[m.shape[0] // 2, m.shape[1] // 2])


def translate_array(arr: np.ndarray, dy: int, dx: int, mode: str = "constant") -> np.ndarray:
    """Translate an (H, W) or (H, W, C) array by (dy, dx) pixels. `constant` pads with zeros (masks); `reflect` mirrors the border (images)."""
    h, w = arr.shape[:2]
    pad = [(max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))] + [(0, 0)] * (arr.ndim - 2)
    padded = np.pad(arr, pad, mode=mode)
    top, left = max(-dy, 0), max(-dx, 0)
    return padded[top:top + h, left:left + w]


def seeded_shift(key: str, size: int = 224, max_frac: float = 0.3) -> tuple[int, int]:
    """Deterministic (dy, dx) shift, uniform in +-`max_frac` of the image size, derived from a string key such as the image id."""
    import hashlib

    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    lim = int(round(size * max_frac))
    return int(rng.integers(-lim, lim + 1)), int(rng.integers(-lim, lim + 1))


@torch.no_grad()
def insertion_deletion_auc(model, x: torch.Tensor, saliency: np.ndarray, class_idx: int,
                           steps: int = 20, batch: int = 21) -> tuple[float, float]:
    """Faithfulness curves: class probability as the most salient pixels are removed (deletion, lower is better) or revealed from an all-mean image (insertion, higher is better). Returns (insertion AUC, deletion AUC) over pixel fractions 0..1."""
    model.eval()
    device = x.device
    h, w = saliency.shape
    order = torch.from_numpy(np.argsort(-saliency.ravel(), kind="stable")).to(device)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=device)
    rank = rank.view(1, 1, h, w).float()
    fracs = torch.linspace(0.0, 1.0, steps + 1, device=device)
    baseline = torch.zeros_like(x)  # zero in normalised space = dataset mean colour
    ins, dele = [], []
    for start in range(0, len(fracs), batch):
        fr = fracs[start:start + batch]
        keep = (rank < (fr.view(-1, 1, 1, 1) * h * w)).float()
        ins_in = baseline * (1 - keep) + x * keep
        del_in = x * (1 - keep) + baseline * keep
        pi = torch.softmax(model(ins_in), dim=1)[:, class_idx]
        pd_ = torch.softmax(model(del_in), dim=1)[:, class_idx]
        ins.append(pi.float().cpu())
        dele.append(pd_.float().cpu())
    ins_c, del_c = torch.cat(ins).numpy(), torch.cat(dele).numpy()
    grid = fracs.cpu().numpy()
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(ins_c, grid)), float(trap(del_c, grid))


METRIC_FUNCTIONS = {
    "overlap": lesion_overlap,
    "pointing": pointing_game,
    "overlap_chance": overlap_chance,
    "pointing_chance": pointing_chance,
    "overlap_ceiling": overlap_ceiling,
    "ceiling_normalised_overlap": ceiling_normalised_overlap,
}


def metric_definition(name: str) -> str:
    """Definition sentence for a saliency metric, taken from the function docstring so report text cannot drift from the code."""
    import inspect

    return " ".join(inspect.getdoc(METRIC_FUNCTIONS[name]).split())


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heat = cm.jet(cam)[..., :3]
    base = image.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1).astype(np.float32)

