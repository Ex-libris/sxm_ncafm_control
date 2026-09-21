"""
Data types shared by every trial backend (simulator now, DDE + IOCTL hardware later).

A *trial* applies one candidate (Kp, Ki) and records the loop's response to a
train of frequency steps. Backends return a :class:`TrialResult`; the planner
never needs to know where it came from.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class StepProtocol:
    """
    A train of alternating +-step_hz offsets of the apparent resonance.

    This is the manual's PLL test: toggle ``DNC use`` by +-1 Hz around f_res.
    The offset sits at ``-step_hz`` for the first hold, then alternates, so the
    steps are ``+2*step_hz, -2*step_hz, ...`` (peak to peak).
    """

    step_hz: float = 1.0
    hold_s: float = 3.0
    n_holds: int = 7

    @property
    def levels(self) -> List[float]:
        return [(-1.0) ** (k + 1) * self.step_hz for k in range(self.n_holds)]

    @property
    def step_times(self) -> List[float]:
        return [self.hold_s * k for k in range(1, self.n_holds)]

    @property
    def step_sizes(self) -> List[float]:
        """Commanded step of each transition (signed, Hz)."""
        lv = self.levels
        return [lv[k] - lv[k - 1] for k in range(1, self.n_holds)]

    @property
    def duration(self) -> float:
        return self.hold_s * self.n_holds


@dataclass
class TrialResult:
    """Everything recorded while one candidate gain set was tested."""

    kp: float                       # raw SXM value that was applied
    ki: float
    t: np.ndarray                   # seconds since the start of the protocol
    df: np.ndarray                  # PLL frequency output [Hz]
    phase: np.ndarray               # PLL phase error [deg]
    protocol: StepProtocol
    locked: bool = True             # False if the loop lost lock / was aborted
    meta: dict = field(default_factory=dict)
