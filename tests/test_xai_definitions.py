"""Saliency metric definitions: the code must match the sentence the reports quote."""

from __future__ import annotations

import numpy as np
import torch

from dm_ahr_vit.xai.vit_xai import (
    ceiling_normalised_overlap,
    gaussian_chance_map,
    insertion_deletion_auc,
    lesion_overlap,
    mask_centroid_offset,
    mask_contains_centre,
    metric_definition,
    overlap_ceiling,
    overlap_chance,
    pointing_chance,
    pointing_game,
    random_argmax_pointing,
    seeded_shift,
    translate_array,
)

SIZE = 100


def _square_mask(frac: float) -> np.ndarray:
    side = int(round(SIZE * np.sqrt(frac)))
    m = np.zeros((SIZE, SIZE), dtype=bool)
    lo = (SIZE - side) // 2
    m[lo:lo + side, lo:lo + side] = True
    return m


def test_overlap_is_recall_not_precision():
    mask = _square_mask(0.4)
    sal = np.zeros((SIZE, SIZE))
    sal[mask] = np.random.default_rng(0).random(mask.sum())  # salient region sits inside the lesion
    from dm_ahr_vit.xai.vit_xai import top_fraction_mask

    top = top_fraction_mask(sal, 0.2)
    precision = (top & mask).sum() / top.sum()
    recall = (top & mask).sum() / mask.sum()
    assert precision == 1.0
    assert abs(lesion_overlap(sal, mask, 0.2) - recall) < 1e-12
    assert lesion_overlap(sal, mask, 0.2) < 1.0
    text = metric_definition("overlap")
    assert "recall" in text and "not precision" in text


def test_overlap_chance_is_the_fraction_for_every_mask_size():
    rng = np.random.default_rng(1)
    for area in (0.1, 0.3, 0.6):
        mask = _square_mask(area)
        vals = [lesion_overlap(rng.random((SIZE, SIZE)), mask, 0.2) for _ in range(300)]
        assert abs(np.mean(vals) - overlap_chance(0.2)) < 0.01


def test_pointing_chance_is_mask_area_not_a_constant():
    rng = np.random.default_rng(2)
    for area in (0.1, 0.3, 0.6):
        mask = _square_mask(area)
        hits = [pointing_game(rng.random((SIZE, SIZE)), mask) for _ in range(3000)]
        assert abs(np.mean(hits) - pointing_chance(mask)) < 0.03
        assert abs(random_argmax_pointing(mask, n=20000) - mask.mean()) < 0.02
    assert abs(pointing_chance(_square_mask(0.6)) - 0.2) > 0.3


def test_overlap_ceiling_and_normalisation():
    small, large = _square_mask(0.1), _square_mask(0.5)
    assert overlap_ceiling(small, 0.2) == 1.0
    assert abs(overlap_ceiling(large, 0.2) - 0.2 / large.mean()) < 1e-12
    sal = large.astype(float) + np.random.default_rng(3).random((SIZE, SIZE)) * 1e-3
    best = lesion_overlap(sal, large, 0.2)
    assert abs(best - overlap_ceiling(large, 0.2)) < 1e-9
    assert abs(ceiling_normalised_overlap(best, large, 0.2) - 1.0) < 1e-9


def test_definition_text_is_generated_from_docstrings():
    for name in ("overlap", "pointing", "overlap_chance", "pointing_chance",
                 "overlap_ceiling", "ceiling_normalised_overlap"):
        assert len(metric_definition(name)) > 20
    assert "area" in metric_definition("pointing_chance")
    assert "0.2" not in metric_definition("pointing_chance")


def test_gaussian_prior_is_centred_and_scores_high_on_centred_masks():
    g = gaussian_chance_map(SIZE)
    centred = _square_mask(0.3)
    assert pointing_game(g, centred) == 1.0
    assert lesion_overlap(g, centred, 0.2) > 0.5


def test_centroid_and_centre_containment():
    assert mask_centroid_offset(_square_mask(0.3)) == (0.0, 0.0) or \
        max(abs(v) for v in mask_centroid_offset(_square_mask(0.3))) < 0.01
    m = np.zeros((SIZE, SIZE), dtype=bool)
    m[0:20, 0:20] = True
    dy, dx = mask_centroid_offset(m)
    assert dy < -0.35 and dx < -0.35
    assert mask_contains_centre(_square_mask(0.3)) and not mask_contains_centre(m)


def test_translate_array_and_seeded_shift():
    arr = np.arange(16, dtype=float).reshape(4, 4)
    out = translate_array(arr, 1, 2)
    assert out.shape == arr.shape and out[1, 2] == arr[0, 0] and (out[0] == 0).all()
    refl = translate_array(np.ones((4, 4, 3)), 1, 1, mode="reflect")
    assert refl.shape == (4, 4, 3)
    a, b = seeded_shift("ISIC_0000001", 224, 0.3), seeded_shift("ISIC_0000001", 224, 0.3)
    assert a == b and all(abs(v) <= 67 for v in a)
    assert seeded_shift("ISIC_0000001") != seeded_shift("ISIC_0000002")


class _Toy(torch.nn.Module):
    """Class-0 score is the mean of the top-left quadrant; no other pixels matter."""

    def forward(self, x):
        s = x[:, :, :8, :8].mean(dim=(1, 2, 3))
        return torch.stack([s * 20, torch.zeros_like(s)], dim=1)


def test_insertion_deletion_prefers_a_faithful_map():
    x = torch.ones(1, 3, 16, 16)
    good = np.zeros((16, 16))
    good[:8, :8] = 1.0
    bad = np.zeros((16, 16))
    bad[8:, 8:] = 1.0
    ins_g, del_g = insertion_deletion_auc(_Toy(), x, good, 0, steps=16)
    ins_b, del_b = insertion_deletion_auc(_Toy(), x, bad, 0, steps=16)
    assert ins_g > ins_b and del_g < del_b
