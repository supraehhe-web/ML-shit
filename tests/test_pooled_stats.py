import numpy as np
import pandas as pd

from dm_ahr_vit.config import CLASSES
from dm_ahr_vit.stats.pooled import (
    compare,
    draw_weights,
    f1_scores,
    group_counts,
    group_loss,
    loss_scores,
    macro_f1,
    one_level,
    two_level,
)


def _preds(n_groups=40, per=3, acc=0.8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for _ in range(per):
            y = int(rng.integers(0, len(CLASSES)))
            p = y if rng.random() < acc else int(rng.integers(0, len(CLASSES)))
            prob = np.full(len(CLASSES), 0.02)
            prob[p] = 1 - 0.02 * (len(CLASSES) - 1)
            rows.append({"group_id": f"g{g}", "label": CLASSES[y], "pred": CLASSES[p],
                         **{f"prob_{c}": prob[i] for i, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows)


def test_group_counts_reproduce_sklearn_style_macro_f1():
    df = _preds()
    groups = sorted(df["group_id"].unique())
    c = group_counts(df, groups)
    got = macro_f1(c["tp"].sum(0)[None], c["fp"].sum(0)[None], c["fn"].sum(0)[None])[0]
    from sklearn.metrics import f1_score
    ref = f1_score(df["label"], df["pred"], labels=list(CLASSES), average="macro", zero_division=0)
    assert abs(got - ref) < 1e-9


def test_group_loss_matches_direct_mean():
    df = _preds()
    groups = sorted(df["group_id"].unique())
    ls, n = group_loss(df, groups)
    direct = -np.log(df[[f"prob_{c}" for c in CLASSES]].to_numpy()[
        np.arange(len(df)), df["label"].map(CLASSES.index).to_numpy()]).mean()
    assert abs(ls.sum() / n.sum() - direct) < 1e-9


def test_two_level_ci_is_wider_than_seed_fixed_ci_when_seeds_differ():
    groups = [f"g{i}" for i in range(40)]
    counts = [group_counts(_preds(acc=a, seed=s), groups) for s, a in enumerate((0.6, 0.8, 0.7, 0.9, 0.65))]
    w = draw_weights(len(groups), 2000, 1)
    sc = f1_scores(counts, w)
    rng = np.random.default_rng(2)
    wide = two_level(sc, rng)
    narrow = one_level(sc)
    assert wide.std() > narrow.std() * 1.3


def test_compare_detects_a_real_gap_and_not_a_null():
    rng = np.random.default_rng(0)
    base = rng.normal(0.7, 0.01, 4000)
    real = compare(base + 0.05, base, 0.75, 0.70)
    assert real["ci_low"] > 0 and real["p_raw"] < 0.01 and abs(real["mdd_80pct"] - 2.8 * real["se_diff"]) < 1e-12
    null = compare(base, rng.permutation(base), 0.7, 0.7)
    assert null["ci_low"] < 0 < null["ci_high"] and null["p_raw"] > 0.05


def test_loss_scores_shape_and_all_ones_weights_equal_pooled_mean():
    df = _preds()
    groups = sorted(df["group_id"].unique())
    ls_n = [group_loss(df, groups)]
    w = np.ones((1, len(groups)), dtype=np.float32)
    s = loss_scores(ls_n, w)
    assert s.shape == (1, 1) and abs(s[0, 0] - ls_n[0][0].sum() / ls_n[0][1].sum()) < 1e-6
