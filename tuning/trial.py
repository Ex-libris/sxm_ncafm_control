"""
Data types shared by every trial backend (simulator now, DDE + IOCTL hardware later).

A *trial* applies one candidate (Kp, Ki) and records the loop's response to a
train of frequency steps. Backends return a :class:`TrialResult`; the planner
never needs to know where it came from.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

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


@dataclass(frozen=True)
class StepTrain:
    """
    A commanded piecewise-constant input: a train of steps with arbitrary timing.

    ``levels[0]`` holds until ``step_times[0]``, ``levels[1]`` until
    ``step_times[1]``, and so on, so ``len(levels) == len(step_times) + 1``.
    Times are seconds from the start of the recording. Unlike
    :class:`StepProtocol` the holds may differ (0.2 s ... 2 s is typical) and the
    levels may be anything, so real captures with events typed into the Step
    Test tab fit here.
    """

    step_times: Tuple[float, ...]
    levels: Tuple[float, ...]

    def __post_init__(self):
        if len(self.levels) != len(self.step_times) + 1:
            raise ValueError("need exactly one more level than step times")
        if any(b <= a for a, b in zip(self.step_times, self.step_times[1:])):
            raise ValueError("step times must be strictly increasing")

    @classmethod
    def from_steps(cls, step_times: Sequence[float], step_sizes: Sequence[float], initial_level: float = 0.0):
        levels = [float(initial_level)]
        for s in step_sizes:
            levels.append(levels[-1] + float(s))
        return cls(tuple(float(t) for t in step_times), tuple(levels))

    @classmethod
    def from_holds(cls, holds: Sequence[float], levels: Sequence[float]):
        """``holds[k]`` is how long ``levels[k]`` is held (the last hold's length is not needed for the times)."""
        times = np.cumsum(holds)[:-1]
        return cls(tuple(float(t) for t in times), tuple(float(v) for v in levels))

    @classmethod
    def from_protocol(cls, p: StepProtocol):
        return cls.from_holds([p.hold_s] * p.n_holds, p.levels)

    @property
    def step_sizes(self) -> List[float]:
        return [b - a for a, b in zip(self.levels, self.levels[1:])]

    def holds(self, t_end: float) -> List[float]:
        """Duration of the hold after each step (the last one runs to ``t_end``)."""
        ends = list(self.step_times[1:]) + [t_end]
        return [e - s for s, e in zip(self.step_times, ends)]

    def value_at(self, t, delay: float = 0.0):
        """The input at times ``t`` if every step reaches the system ``delay`` seconds late."""
        idx = np.searchsorted(np.asarray(self.step_times) + delay, np.asarray(t, dtype=float), side="right")
        return np.asarray(self.levels)[idx]


@dataclass
class LoopCapture:
    """
    A recording of one PI loop responding to a train of steps.

    ``kind`` selects which channels these are:

    ========  ===========================  ==============================  ==========================
    kind      ``y`` (controller input)     ``u`` (controller output)       ``train`` (commanded)
    ========  ===========================  ==============================  ==========================
    ``pll``   ``Phase`` [deg]              ``df`` [Hz]                     ``use`` offset from f_res [Hz]
    ``afl``   amplitude (``QPlusAmpl``)    ``Drive``                       amplitude ``Ref``
    ========  ===========================  ==============================  ==========================

    ``kp_raw`` / ``ki_raw`` are the SXM values that were set while recording.
    """

    kind: str
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    train: StepTrain
    kp_raw: float
    ki_raw: float
    meta: dict = field(default_factory=dict)
