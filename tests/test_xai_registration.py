"""The lesion mask must be carried through the same geometry as the model input, for every arm."""

import numpy as np
import pytest
import torch
from PIL import Image

from dm_ahr_vit.config import load_config
from dm_ahr_vit.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from dm_ahr_vit.data.masks import load_mask_224, load_mask_registered
from dm_ahr_vit.xai.batched import (
    MultiLayerGradCAM,
    attention_rollout_batch,
    localization_scores,
    metric_definitions,
    random_map_baselines,
)
from dm_ahr_vit.xai.geometry import ARM_STORED_SIZES, mask_to_model_frame, shift_zero_fill
from dm_ahr_vit.xai.vit_xai import (
    GradCAM,
    attention_rollout,
    lesion_overlap,
    pointing_game,
    salient_in_lesion,
    top_region_centroid_distance,
)

CFG = load_config("configs/vit.json")
FULL_WH = (600, 450)
BOX = (0.30, 0.35, 0.70, 0.80)          # a lesion-like box, off-centre vertically


def _box_image(size_wh, box=BOX):
    w, h = size_wh
    img = np.zeros((h, w, 3), np.uint8)
    img[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)] = 255
    return Image.fromarray(img)


def _model_input_bright(img):
    t = build_transforms(CFG, train=False)(img)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (t * std + mean).mean(0).numpy() > 0.5


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


STORED_CASES = sorted({(a, wh) for a, s in ARM_STORED_SIZES.items() for wh in s})


@pytest.mark.parametrize("arm,stored_wh", STORED_CASES)
def test_mask_stays_aligned_with_the_image_the_model_sees(arm, stored_wh):
    full = _box_image(FULL_WH)
    stored = full.resize(stored_wh, Image.Resampling.BICUBIC)      # how the arm stores this image
    mask = mask_to_model_frame(full.convert("L"), stored_wh, 224)
    seen = _model_input_bright(stored)
    assert _iou(mask, seen) > 0.9, (arm, stored_wh, _iou(mask, seen))


def test_mask_fraction_in_model_frame_differs_by_stored_size():
    fr = {wh: float(mask_to_model_frame(_box_image(FULL_WH).convert("L"), wh, 224).mean())
          for wh in {w for s in ARM_STORED_SIZES.values() for w in s}}
    # 600x450 loses the sides to the crop (box covers ~0.31 of the frame); 128x128 loses less (~0.23)
    assert fr[(600, 450)] == pytest.approx(0.311, abs=0.01)
    assert fr[(128, 128)] == pytest.approx(0.233, abs=0.01)


def test_legacy_squash_is_misregistered_for_full_size_images(tmp_path):
    full = _box_image(FULL_WH).convert("L")
    p = tmp_path / "m.png"
    full.save(p)
    seen = _model_input_bright(_box_image(FULL_WH))
    assert _iou(load_mask_224(p, 224), seen) < 0.6
    assert _iou(load_mask_registered(p, None, 224), seen) > 0.93


def test_metric_definitions_come_from_the_docstrings():
    text = metric_definitions(0.2)
    for word in ("recall", "Precision", "Pointing", "Centroid"):
        assert word in text


def test_recall_precision_pointing_centroid_on_known_maps():
    sal = np.zeros((10, 10))
    sal[:2, :] = 1.0
    mask = np.zeros((10, 10), np.uint8)
    mask[:4, :] = 1
    assert salient_in_lesion(sal, mask, 0.2) == 1.0 and lesion_overlap(sal, mask, 0.2) == 0.5
    s = localization_scores(sal, mask, 0.2)
    assert (s["recall"], s["precision"]) == (0.5, 1.0)
    assert s["pointing"] == pointing_game(sal, mask)
    assert s["centroid"] == pytest.approx(top_region_centroid_distance(sal, mask, 0.2))
    sal2 = np.zeros((20, 20))
    sal2[5:10, 5:10] = 1.0
    m2 = np.zeros((20, 20), np.uint8)
    m2[5:10, 10:15] = 1
    assert top_region_centroid_distance(sal2, m2, 25 / 400) == pytest.approx(5 / 20)


def test_random_baselines_match_their_expectations():
    mask = np.zeros((224, 224), np.uint8)
    mask[60:160, 50:170] = 1
    b = random_map_baselines(mask, seed=1, draws=100)
    area = float(mask.mean())
    assert b["rand_pixel_recall"] == pytest.approx(0.2, abs=0.01)
    assert b["rand_pixel_precision"] == pytest.approx(area, abs=0.01)
    assert b["rand_pixel_pointing"] == pytest.approx(area, abs=0.03)
    assert abs(b["rand_patch_recall"] - 0.2) < 0.03


def test_shift_zero_fill_moves_image_and_mask_together():
    a = np.arange(36, dtype=float).reshape(6, 6)
    s = shift_zero_fill(a, 2, -1)
    assert s[0, 2] == a[1, 0] and s[:, :2].sum() == 0 and s[-1].sum() == 0
    assert shift_zero_fill(torch.ones(3, 6, 6), 9, 0).sum() == 0


def _tiny_vit():
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(0)
    return VisionTransformer(image_size=32, patch_size=8, num_layers=3, num_heads=2,
                             hidden_dim=16, mlp_dim=32, num_classes=7).eval()


def test_batched_rollout_matches_single_image_rollout():
    model = _tiny_vit()
    x = torch.randn(3, 3, 32, 32)
    batch = attention_rollout_batch(model, x)
    for i in range(3):
        assert np.allclose(batch[i], attention_rollout(model, x[i:i + 1]), atol=1e-5)


def test_batched_gradcam_matches_single_image_gradcam_on_the_patch_grid():
    model = _tiny_vit()
    x = torch.randn(3, 3, 32, 32)
    with MultiLayerGradCAM(model) as cam:
        maps, idx = cam(x)
    for i in range(3):
        with GradCAM(model) as g:
            single = g(x[i:i + 1], class_idx=int(idx[i]))
        assert np.allclose(maps["patch_grid"][i], single, atol=1e-5)
    assert maps["last_ln1"].shape == (3, 4, 4) and np.isfinite(maps["last_ln1"]).all()
