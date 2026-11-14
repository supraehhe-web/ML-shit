import numpy as np

from dm_ahr_vit.data.composite_metrics import bootstrap_mean_ci, item_metrics


def _world(s=64):
    target = np.full((s, s, 3), 150.0)
    lesion = np.zeros((s, s), bool)
    lesion[16:48, 16:48] = True
    target[lesion] = 90.0
    alpha = np.zeros((s, s))
    alpha[30:34, :] = 1.0                      # one horizontal strand across the lesion
    comp = target.copy()
    comp[alpha > 0.5] = 20.0
    return target, comp, alpha, lesion


def test_perfect_removal_scores_one_and_no_damage():
    target, comp, alpha, lesion = _world()
    m = item_metrics(target.copy(), target, comp, alpha, lesion)
    assert m["hair_removal"] == 1.0 and m["lesion_mae"] == 0.0 and m["lesion_changed"] == 0.0


def test_doing_nothing_scores_zero_removal():
    target, comp, alpha, lesion = _world()
    m = item_metrics(comp.copy(), target, comp, alpha, lesion)
    assert abs(m["hair_removal"]) < 1e-9 and m["lesion_changed"] == 0.0


def test_damage_to_clear_lesion_pixels_is_counted_but_strand_pixels_are_not():
    target, comp, alpha, lesion = _world()
    out = target.copy()
    out[20:24, 20:44] += 40.0                   # change lesion pixels away from the strand
    m = item_metrics(out, target, comp, alpha, lesion)
    assert m["lesion_changed"] > 0 and m["lesion_mae"] > 0
    out2 = comp.copy()
    out2[alpha > 0.5] = 255.0                   # only strand pixels differ from the input
    m2 = item_metrics(out2, target, comp, alpha, lesion)
    assert m2["lesion_changed"] == 0.0


def test_tiny_lesion_gives_nan_not_a_number():
    target, comp, alpha, _ = _world()
    tiny = np.zeros(alpha.shape, bool)
    tiny[0:3, 0:3] = True
    m = item_metrics(comp.copy(), target, comp, alpha, tiny)
    assert np.isnan(m["lesion_mae"])


def test_bootstrap_ci_brackets_the_mean_and_ignores_nan():
    x = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
    mean, lo, hi = bootstrap_mean_ci(x)
    assert mean == 2.5 and lo <= mean <= hi
