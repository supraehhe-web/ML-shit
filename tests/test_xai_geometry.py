from __future__ import annotations

import numpy as np

from dm_ahr_vit.xai.geometry import model_space, retained_fraction, stretched_224


def _full_field_mask(w, h):
    return np.ones((h, w), dtype=np.uint8)


def test_stretched_mask_is_whole_field_but_model_space_is_a_crop():
    # Lesion occupies only the left 10% of a 600x450 photo: outside the centre crop.
    m = np.zeros((450, 600), dtype=np.uint8)
    m[:, :60] = 1
    assert stretched_224(m).mean() > 0.05          # the first XAI run still scored this lesion
    assert model_space(m, (600, 450)).sum() == 0     # the ViT never sees it
    assert retained_fraction(m, (600, 450)) == 0.0


def test_whole_field_mask_fills_model_space_for_both_geometries():
    for wh in ((600, 450), (128, 128)):
        out = model_space(_full_field_mask(600, 450), wh)
        assert out.shape == (224, 224) and out.mean() == 1.0


def test_centre_lesion_scale_differs_between_native_and_stretched():
    m = np.zeros((450, 600), dtype=np.uint8)
    m[125:325, 200:400] = 1                           # 200x200 centred lesion
    native = model_space(m, (600, 450)).mean()
    stretched = stretched_224(m).mean()
    assert native > stretched * 1.3                   # same lesion looks ~1.5x larger to the ViT
    assert model_space(m, (128, 128)).shape == (224, 224)


def test_retained_fraction_is_one_for_central_lesion():
    m = np.zeros((450, 600), dtype=np.uint8)
    m[150:300, 225:375] = 1
    assert retained_fraction(m, (600, 450)) > 0.99
