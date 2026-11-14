import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_stats.py"


def _load():
    spec = importlib.util.spec_from_file_location("stats09", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_as_c_arm_replaces_only_c():
    mod = _load()
    base = [("A", "B"), ("A", "C"), ("B", "C")]
    assert mod.as_c_arm(base) == base  # default role is unchanged
    mod.C_ARM = "Cv2"
    assert mod.as_c_arm(base) == [("A", "B"), ("A", "Cv2"), ("B", "Cv2")]
    assert mod.as_c_arm([("C", "Astar")]) == [("Cv2", "Astar")]
