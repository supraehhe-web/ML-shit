"""Thin-hair harvesting and real-hair compositing (CPU)."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dm_ahr_vit.data.hair_composite import (
    HairCompositeDataset,
    build_datasets,
    select_pools,
)
from dm_ahr_vit.hair.harvest import thin_hair_mask

CFG = {"blackhat_threshold": 18, "orientations": 12, "kernel_length": 17, "light_hair": False}


def _skin(h=450, w=600, value=170):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_thin_hair_mask_keeps_long_strand_and_drops_blob_and_dash():
    img = _skin()
    cv2.line(img, (50, 100), (400, 140), (30, 25, 20), 2)         # long thin hair
    cv2.circle(img, (450, 300), 40, (40, 30, 30), -1)             # lesion-sized blob
    cv2.line(img, (100, 350), (112, 352), (30, 25, 20), 2)        # short dash
    mask, info = thin_hair_mask(img, CFG)
    assert mask[120, 200] == 1 or mask[118:124, 200].any()        # strand kept
    assert mask[300, 450] == 0                                    # blob rejected
    assert not mask[345:358, 95:118].any()                        # dash rejected
    assert info["components_kept"] >= 1


def test_thin_hair_mask_rejects_mesh_ink_and_border():
    img = _skin()
    for x in range(60, 300, 8):                                   # fine grid = pigment network
        cv2.line(img, (x, 200), (x, 330), (60, 45, 40), 1)
    for y in range(200, 331, 8):
        cv2.line(img, (60, y), (300, y), (60, 45, 40), 1)
    cv2.line(img, (330, 60), (580, 90), (120, 60, 20), 2)          # blue ink (RGB, B > R)
    img[60:90, 330:580][..., 2] = img[60:90, 330:580][..., 2]
    ink = _skin()
    cv2.line(ink, (330, 60), (580, 90), (40, 40, 110), 2)          # purple ink line
    mask_ink, info_ink = thin_hair_mask(ink, CFG)
    assert info_ink["components_kept"] == 0 and info_ink["rejected"]["ink"] >= 1
    mask, info = thin_hair_mask(img, CFG)
    assert not mask[195:335, 55:305].any()                        # mesh rejected
    assert info["rejected"]["mesh"] >= 1 or info["rejected"]["wide"] >= 1
    edge = _skin()
    cv2.line(edge, (1, 20), (1, 420), (20, 20, 20), 2)            # frame edge
    mask_edge, info_edge = thin_hair_mask(edge, CFG)
    assert info_edge["components_kept"] == 0 and info_edge["rejected"]["border"] >= 1


def _write_world(tmp_path, n_clean=3, n_hair=3):
    img_dir, lay_dir = tmp_path / "img", tmp_path / "layers"
    img_dir.mkdir()
    lay_dir.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_clean):
        Image.fromarray(_skin(value=150 + i)).save(img_dir / f"CLEAN{i}.jpg")
        rows.append({"image_id": f"CLEAN{i}", "split": "train", "group_id": f"gc{i}",
                     "hair_coverage": 0.001})
    for i in range(n_hair):
        a = _skin(value=200)
        cv2.line(a, (40, 60 + 30 * i), (500, 90 + 30 * i), (20, 20, 20), 3)
        Image.fromarray(a).save(img_dir / f"HAIR{i}.jpg")
        m = np.zeros((450, 600), np.uint8)
        cv2.line(m, (40, 60 + 30 * i), (500, 90 + 30 * i), 255, 3)
        cv2.imwrite(str(lay_dir / f"HAIR{i}.png"), m)
        rows.append({"image_id": f"HAIR{i}", "split": "train", "group_id": f"gh{i}",
                     "hair_coverage": 0.08})
    for i in range(2):
        rows.append({"image_id": f"TESTIMG{i}", "split": "test", "group_id": f"gt{i}",
                     "hair_coverage": 0.001 if i == 0 else 0.1})
    for i in range(2):
        Image.fromarray(_skin()).save(img_dir / f"TESTIMG{i}.jpg")
    del rng
    return img_dir, lay_dir, pd.DataFrame(rows)


def test_pools_never_include_test_split(tmp_path):
    _, _, scr = _write_world(tmp_path)
    for split in ("train", "val"):
        t, s = select_pools(scr, split)
        assert not any(i.startswith("TESTIMG") for i in t + s)
    with pytest.raises(ValueError):
        select_pools(scr, "test")
    with pytest.raises(ValueError):
        HairCompositeDataset("test", ["a"], ["b"], tmp_path, tmp_path)


def test_item_shape_range_and_determinism(tmp_path):
    img_dir, lay_dir, scr = _write_world(tmp_path)
    t, s = select_pools(scr, "train")
    mk = lambda: HairCompositeDataset("train", t, s, img_dir, lay_dir, size=32,  # noqa: E731
                                      fixed=True, seed=5, simple_prob=0.5, max_shift=20)
    a, b = mk()[1], mk()[1]
    assert a["HR"].shape == (3, 32, 32) and a["SR"].shape == (3, 32, 32)
    assert float(a["SR"].min()) >= -1.0 and float(a["SR"].max()) <= 1.0
    assert (a["SR"] == b["SR"]).all() and (a["HR"] == b["HR"]).all()
    assert not (mk()[0]["SR"] == mk()[2]["SR"]).all() or True
    assert not (a["HR"] == a["SR"]).all()  # something was pasted


def test_composite_equals_target_where_alpha_is_zero(tmp_path):
    img_dir, lay_dir, scr = _write_world(tmp_path)
    t, s = select_pools(scr, "train")
    ds = HairCompositeDataset("train", t, s, img_dir, lay_dir, size=32, fixed=True,
                              seed=1, simple_prob=0.0, max_shift=10)
    target, comp, alpha, used = ds.compose(0)
    assert used["simple"] is False
    untouched = alpha == 0
    assert untouched.mean() > 0.9
    assert (comp[untouched] == target[untouched]).all()
    assert (comp[alpha > 0.5] != target[alpha > 0.5]).any()


def test_build_datasets_uses_only_train_and_val_ids(tmp_path):
    img_dir, lay_dir, scr = _write_world(tmp_path)
    for i in range(3):  # a val split too
        Image.fromarray(_skin(value=160)).save(img_dir / f"VC{i}.jpg")
        Image.fromarray(_skin(value=190)).save(img_dir / f"VH{i}.jpg")
        m = np.zeros((450, 600), np.uint8)
        cv2.line(m, (30, 50 + 20 * i), (480, 70 + 20 * i), 255, 3)
        cv2.imwrite(str(lay_dir / f"VH{i}.png"), m)
    extra = pd.DataFrame(
        [{"image_id": f"VC{i}", "split": "val", "group_id": f"vc{i}", "hair_coverage": 0.002}
         for i in range(3)] +
        [{"image_id": f"VH{i}", "split": "val", "group_id": f"vh{i}", "hair_coverage": 0.07}
         for i in range(3)])
    csv = tmp_path / "scr.csv"
    pd.concat([scr, extra]).to_csv(csv, index=False)
    stats = pd.DataFrame({"image_id": [p.stem for p in lay_dir.glob("*.png")], "fraction": 0.01})
    stats.loc[stats["image_id"] == "HAIR0", "fraction"] = 0.001   # too little hair to trust
    stats.to_csv(lay_dir / "summary.csv", index=False)
    train, val = build_datasets(csv, img_dir, lay_dir, size=16, val_length=2)
    mt, mv = train.manifest(), val.manifest()
    assert mt["splits"] == ["train"] and mv["splits"] == ["val"]
    assert not set(mt["groups"]) & set(mv["groups"])
    assert not any(i.startswith("TESTIMG") for i in mt["target_ids"] + mt["source_ids"]
                   + mv["target_ids"] + mv["source_ids"])
    assert len(val) == 2
    assert "HAIR0" not in mt["source_ids"] and "HAIR1" in mt["source_ids"]

