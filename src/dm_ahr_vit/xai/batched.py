"""Batched explanation maps and scoring used by the corrected localization analysis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .vit_xai import (
    _patch_grid,
    capture_attention,
    lesion_overlap,
    pointing_game,
    salient_in_lesion,
    top_fraction_mask,
    top_region_centroid_distance,
)


def metric_definitions(fraction: float = 0.2) -> str:
    """The metric wording used in reports, generated from the function docstrings."""
    pct = f"{int(round(fraction * 100))}%"
    fns = (lesion_overlap, salient_in_lesion, pointing_game, top_region_centroid_distance)
    return "\n".join("- " + (fn.__doc__ or "").strip().replace("20%", pct) for fn in fns)


class MultiLayerGradCAM:
    """Batched Grad-CAM at several layers from one backward pass.

    ``last_ln1``: tokens after `ln_1` of the last encoder block, class token dropped, reshaped to
    the patch grid. ``patch_grid``: output of the patch-embedding convolution (the layer used
    by the recorded results, kept as a labelled sensitivity run).
    """

    def __init__(self, model, names=("last_ln1", "patch_grid")):
        self.model = model
        self.grid = _patch_grid(model)
        avail = {"last_ln1": (model.encoder.layers[-1].ln_1, "tokens"),
                 "patch_grid": (model.conv_proj, "grid")}
        self.layers = {n: avail[n] for n in names}
        self._handles: list = []
        self._act: dict[str, torch.Tensor] = {}
        self._grad: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for name, (module, _kind) in self.layers.items():
            def hook(_m, _i, out, name=name):
                self._act[name] = out
                if out.requires_grad:
                    out.register_hook(lambda g, name=name: self._grad.__setitem__(name, g))
            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _as_grid(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "tokens":
            b, _, c = t.shape
            return t[:, 1:, :].transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return t

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | None = None):
        """Return ({layer: (B, g, g) maps in [0, 1]}, predicted class index per image)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._act, self._grad = {}, {}
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            idx = logits.argmax(dim=1) if class_idx is None else class_idx
            logits.gather(1, idx[:, None]).sum().backward()
        maps = {}
        for name, (_m, kind) in self.layers.items():
            if name not in self._grad:
                raise RuntimeError(f"Grad-CAM captured no gradient at {name}")
            act = self._as_grid(self._act[name].detach(), kind)
            grad = self._as_grid(self._grad[name].detach(), kind)
            cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1))
            flat = cam.flatten(1)
            flat = flat - flat.min(dim=1, keepdim=True).values
            mx = flat.max(dim=1, keepdim=True).values
            flat = torch.where(mx > 0, flat / mx.clamp_min(1e-12), flat)
            maps[name] = flat.view_as(cam).cpu().numpy()
        return maps, idx


@torch.no_grad()
def attention_rollout_batch(model, x: torch.Tensor, discard_ratio: float = 0.9) -> np.ndarray:
    """`attention_rollout` for a batch: (B, g, g) maps."""
    model.eval()
    with capture_attention(model) as captured:
        model(x)
    if not captured:
        raise RuntimeError("no attention weights captured")
    b, tokens = captured[0].shape[0], captured[0].shape[-1]
    eye = torch.eye(tokens, device=captured[0].device)
    result = eye.expand(b, tokens, tokens).clone()
    for attn in captured:
        a = attn.float().clone()
        if discard_ratio > 0:
            k = int(tokens * discard_ratio)
            if k > 0:
                idx = a.topk(k, dim=-1, largest=False).indices
                a.scatter_(-1, idx, 0.0)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        result = (0.5 * a + 0.5 * eye) @ result
    grid = _patch_grid(model)
    m = result[:, 0, 1:].reshape(b, grid * grid)
    m = m - m.min(dim=1, keepdim=True).values
    mx = m.max(dim=1, keepdim=True).values
    m = torch.where(mx > 0, m / mx.clamp_min(1e-12), m)
    return m.view(b, grid, grid).cpu().numpy()


def localization_scores(saliency: np.ndarray, mask: np.ndarray, fraction: float = 0.2) -> dict:
    """recall, precision, pointing and centroid distance for one map, from one top-region pass."""
    m = mask.astype(bool)
    if not m.any():
        return {"recall": float("nan"), "precision": float("nan"),
                "pointing": float("nan"), "centroid": float("nan")}
    top = top_fraction_mask(saliency, fraction)
    inter = float((top & m).sum())
    ty, tx = np.nonzero(top)
    ly, lx = np.nonzero(m)
    r, c = divmod(int(np.argmax(saliency)), saliency.shape[1])
    return {"recall": inter / float(m.sum()), "precision": inter / float(top.sum()),
            "pointing": float(m[r, c]),
            "centroid": float(np.hypot(tx.mean() - lx.mean(), ty.mean() - ly.mean())
                              / saliency.shape[1])}


def random_map_baselines(mask: np.ndarray, seed: int, draws: int = 100, fraction: float = 0.2,
                         grid: int = 14, device: str = "cpu") -> dict:
    """Mean recall, precision and pointing of random maps over `draws` draws: white-noise pixel
    maps, and random patch-grid maps bilinearly upsampled the way the real maps are."""
    kinds = ("pixel", "patch")
    keys = ("recall", "precision", "pointing")
    m = torch.from_numpy(mask.astype(bool)).to(device).flatten()
    if not bool(m.any()):
        return {f"rand_{k}_{q}": float("nan") for k in kinds for q in keys}
    n, size = m.numel(), mask.shape[0]
    k_top = max(1, int(round(n * fraction)))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = {}
    for kind in kinds:
        if kind == "pixel":
            sal = torch.rand(draws, n, generator=gen).to(device)
        else:
            g = torch.rand(draws, 1, grid, grid, generator=gen).to(device)
            sal = F.interpolate(g, size=(size, size), mode="bilinear",
                                align_corners=False).reshape(draws, n)
        top = torch.zeros(draws, n, dtype=torch.bool, device=device)
        top.scatter_(1, sal.topk(k_top, dim=1).indices, True)
        inter = (top & m).sum(dim=1).float()
        out[f"rand_{kind}_recall"] = float((inter / m.sum()).mean())
        out[f"rand_{kind}_precision"] = float((inter / k_top).mean())
        out[f"rand_{kind}_pointing"] = float(m[sal.argmax(dim=1)].float().mean())
    return out
