"""Spend guard for Modal jobs.

Every GPU entry point calls `gate()` with its own measured seconds-per-unit
before starting a large job. The job refuses to run without an explicit budget
or when its estimate exceeds it. Prices are the per-hour values recorded in
`reports/compute_plan.md` (read from modal.com/pricing on 2026-11-11); they are
a record, not a live quote, and should be re-checked before a large run.
"""

from __future__ import annotations

import json
import sys

USD_PER_HOUR = {
    "A100-80GB": 2.4984,
    "A100-40GB": 2.0988,
    "A10": 1.1016,
    "A10G": 1.1016,
    "L4": 0.7992,
    "T4": 0.5904,
}
PRICE_SOURCE = "reports/compute_plan.md (modal.com/pricing, read 2026-11-11)"


def estimate_usd(seconds: float, gpu: str) -> float:
    if gpu not in USD_PER_HOUR:
        raise KeyError(f"no recorded price for GPU {gpu!r}")
    return float(seconds) / 3600.0 * USD_PER_HOUR[gpu]


def gate(label: str, units: int, seconds_per_unit: float, gpu: str,
         budget_usd: float | None, fixed_seconds: float = 0.0) -> dict:
    """Return the cost estimate, or exit if there is no budget or it is exceeded."""
    seconds = units * float(seconds_per_unit) + fixed_seconds
    cost = estimate_usd(seconds, gpu)
    report = {"label": label, "units": int(units), "seconds_per_unit": float(seconds_per_unit),
              "gpu": gpu, "estimated_seconds": round(seconds, 1),
              "estimated_usd": round(cost, 2), "budget_usd": budget_usd,
              "price_source": PRICE_SOURCE}
    print(json.dumps(report), flush=True)
    if budget_usd is None:
        sys.exit(f"[budget] {label}: refusing to run without --budget-usd "
                 f"(estimate ${cost:.2f})")
    if cost > float(budget_usd):
        sys.exit(f"[budget] {label}: estimate ${cost:.2f} exceeds budget ${budget_usd:.2f}; stopping")
    return report
