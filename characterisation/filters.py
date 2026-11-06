"""The lock-in output filter as n cascaded first-order (RC) stages with time constant TC; n = roll-off / 6 dB/oct.

Used to place analysis windows (settling), to predict the delay of a filtered step, and to turn a noise
amplitude into a density (ENBW). Standard lock-in relations, pure math.
"""
from __future__ import annotations

import math


def stages(rolloff_db: int) -> int:
    """6 / 12 / 18 / 24 dB/oct -> 1 / 2 / 3 / 4 stages."""
    n = int(round(rolloff_db / 6))
    if n not in (1, 2, 3, 4):
        raise ValueError(f'roll-off {rolloff_db} dB/oct is not 6, 12, 18 or 24')
    return n


def enbw(tau: float, rolloff_db: int) -> float:
    """Equivalent noise bandwidth: 1/(4 TC), 1/(8 TC), 3/(32 TC), 5/(64 TC) for 1..4 stages."""
    return {1: 1 / 4, 2: 1 / 8, 3: 3 / 32, 4: 5 / 64}[stages(rolloff_db)] / tau


def step_response(t: float, tau: float, rolloff_db: int) -> float:
    """Fraction of a step reached after ``t``: 1 - exp(-t/TC) * sum_{k<n} (t/TC)^k / k!."""
    n, x = stages(rolloff_db), t / tau
    return 1 - math.exp(-x) * sum(x ** k / math.factorial(k) for k in range(n))


def settle_time(tau: float, rolloff_db: int, accuracy: float = 0.01) -> float:
    """Time to come within ``accuracy`` of a step (1 %: 4.6, 6.6, 8.4, 10.0 TC for 1..4 stages)."""
    lo, hi = 0.0, 100 * tau
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if step_response(mid, tau, rolloff_db) < 1 - accuracy else (lo, mid)
    return hi


def group_delay(tau: float, rolloff_db: int) -> float:
    """Delay of a slow signal through the filter: n TC."""
    return stages(rolloff_db) * tau
