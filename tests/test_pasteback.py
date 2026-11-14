import cv2
import numpy as np
from PIL import Image

from dm_ahr_vit.hair.pasteback import pasteback


def _world(h=450, w=600):
    clean = np.full((h, w, 3), 170, np.uint8)
    hairy = clean.copy()
    cv2.line(hairy, (40, 100), (500, 160), (30, 25, 20), 3)
    fill = np.asarray(Image.fromarray(clean).resize((128, 128), Image.Resampling.BICUBIC))
    return clean, hairy, fill


def test_hair_pixels_are_replaced_and_everything_else_is_identical():
    clean, hairy, fill = _world()
    out, info = pasteback(hairy, fill, tau=16, dilate=1)
    assert out.dtype == np.uint8 and out.shape == hairy.shape
    assert abs(int(out[130, 270, 0]) - 170) < 20            # strand centre now skin-like
    far = np.zeros(hairy.shape[:2], bool)
    far[300:, :] = True                                      # well away from the strand
    assert np.array_equal(out[far], hairy[far])
    assert 0 < info["changed_fraction"] < 0.2


def test_high_threshold_changes_nothing():
    _, hairy, fill = _world()
    out, info = pasteback(hairy, fill, tau=250, dilate=2)
    assert np.array_equal(out, hairy) and info["mask_fraction"] == 0.0


def test_detector_mask_limits_the_replaced_region_exactly():
    _, hairy, fill = _world()
    det = np.zeros(hairy.shape[:2], np.uint8)
    det[:, :250] = 1                                         # detector only sees the left half
    out, _ = pasteback(hairy, fill, tau=16, dilate=1, det_mask=det, refine_dilate=0)
    assert np.array_equal(out[:, 260:], hairy[:, 260:])      # right half untouched
    assert not np.array_equal(out[:, :250], hairy[:, :250])  # left half repaired
