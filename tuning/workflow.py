"""
One step test of one of the SXM's two PI loops (PLL and amplitude feedback): what is stepped, what is recorded,
and what the recording measures.

    define the test (StepTestPlan)  ->  detect what the channels are  ->  analyse the train (analyze_test)

Judging the result and choosing the next gains is tuning/explore.py; the instrument I/O is gui/tuning_runner.py.
Everything here is pure logic on numpy arrays (no Qt, no hardware).

The two tests are the manual's:

* **PLL** - toggle ``DNC use`` by +-1 Hz around f_res and watch ``df`` and ``Phase``;
  ``df`` should be a rectangular, non-overshooting wave.
* **Amplitude loop** - toggle the amplitude ``Ref`` and watch ``QPlusAmpl`` and ``Drive``;
  ``QPlusAmpl`` should be rectangular, ``Drive`` may overshoot but must not saturate. The
  manual steps by +-10 %; the default here is +-5 %, because Drive cannot go below zero: at
  small amplitudes (steady Drive of a few tens of uV) a 10 % down-step with real gains pins
  Drive at zero and the amplitude can only fall at the sensor's ring-down rate.

Up-steps and down-steps are measured separately as well as together (``primary_rising`` /
``primary_falling``): they are not equivalent, above all for the amplitude loop.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from . import metrics as M
from .loopid import LoopModel, identify_loop
from .trial import LoopCapture, StepTrain


# ---------------------------------------------------------------------------
# the two loops
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoopDef:
    key: str
    name: str
    y_channel: str          # controller input as recorded (SXM channel name)
    u_channel: str          # controller output
    primary_channel: str    # the channel whose shape must be rectangular
    kp_param: Tuple[str, object]     # (ptype, pcode) as in gui/common.PARAMS_BASE
    ki_param: Tuple[str, object]
    step_param: Tuple[str, object]   # the parameter the test steps
    gain_sign: int          # raw gains are negative for the PLL, positive for the amplitude loop
    relative_step: bool     # step is a fraction of the base value (amplitude Ref) or absolute (Hz)
    default_step: float
    align_lead_s: float     # how far before its 10 % crossing a step is taken to start
    typical_ratio: float    # manual's starting Ki/Kp (PLL: 100; amplitude: 1e-4)


PLL = LoopDef("pll", "PLL (df / Phase)", y_channel="Phase", u_channel="df", primary_channel="df",
              kp_param=("EDIT", "Edit27"), ki_param=("EDIT", "Edit22"), step_param=("DNC", 3),
              gain_sign=-1, relative_step=False, default_step=1.0, align_lead_s=0.006, typical_ratio=100.0)
AFL = LoopDef("afl", "Amplitude feedback (QPlusAmpl / Drive)", y_channel="QPlusAmpl", u_channel="Drive",
              primary_channel="QPlusAmpl", kp_param=("EDIT", "Edit32"), ki_param=("EDIT", "Edit24"),
              step_param=("EDIT", "Edit23"), gain_sign=+1, relative_step=True, default_step=0.05,
              align_lead_s=0.03, typical_ratio=1e-4)
LOOPS = {"pll": PLL, "afl": AFL}


# ---------------------------------------------------------------------------
# amplitude-loop starting values (manual, pp. 6-7)
# ---------------------------------------------------------------------------
AFL_OUTPUT_GAINS = (0.1, 1.0, 10.0)     # DNC 'Output Gain' ranges, +-V peak
INPUT_GAINS = (1.0, 10.0)               # DNC 'Input Gain InA': 1 = +-7 V, 10 = +-0.7 V (manual)

# Drive floor: Drive is an amplitude and cannot go below zero, so a loop that wants to pull the amplitude down
# faster than the sensor rings down just switches Drive off. A sample counts as "at the floor" when Drive is below
# DRIVE_FLOOR_LEVEL x its settled value before the first step; more than DRIVE_FLOOR_TIME_MAX of the test spent
# there means the loop was on/off, not proportional (seen at 500-550 uV amplitude, steady Drive 10-20 uV).
DRIVE_FLOOR_LEVEL = 0.1
DRIVE_FLOOR_TIME_MAX = 0.02


@dataclass(frozen=True)
class AflStart:
    kp: float
    ki: float
    tau_s: float            # AFL 'Tau' low-pass, Q / (100 f0)
    ring_down_s: float      # amplitude ring-down time of the resonator, Q / (pi f0)
    hold_s: float           # suggested step hold
    settle_s: float         # suggested settle time before recording


def afl_start_values(q: float, f0: float, output_gain_v: float = 1.0) -> AflStart:
    """
    The manual's starting point for the amplitude loop: Ki ~ 5e8/Q and Kp ~ 1e4*Ki at +-1 V output gain,
    both x10 for each range *lower* (+-1 V -> +-0.1 V, stated in the manual). +-10 V is the same rule
    extrapolated (/10); the manual does not state it. Tau = Q / (100 f0).

    ``hold_s`` / ``settle_s`` are this app's heuristic, not the manual's: 3x / 5x the ring-down time
    Q/(pi f0), rounded to 0.5 s, so that a slow (high-Q) sensor can settle between steps.
    """
    if q <= 0 or f0 <= 0 or output_gain_v <= 0:
        raise ValueError("Q, f0 and the output gain must be positive")
    ki = 5e8 / q / output_gain_v
    ring = q / (math.pi * f0)
    return AflStart(kp=1e4 * ki, ki=ki, tau_s=q / (100.0 * f0), ring_down_s=ring,
                    hold_s=min(10.0, max(1.0, round(3.0 * ring * 2) / 2)),
                    settle_s=min(30.0, max(2.0, round(5.0 * ring * 2) / 2)))

_ALIASES = {"df": "df", "phase": "Phase", "qplusampl": "QPlusAmpl", "qplusamplitude": "QPlusAmpl", "drive": "Drive"}


@dataclass(frozen=True)
class Detection:
    """What the recorded scope channels represent."""

    loop: Optional[LoopDef]
    complete: bool                  # both controller channels present (full analysis possible)
    names: Dict[str, str]           # canonical channel name -> the name used in the capture
    note: str


def detect_loop(channel_names: Sequence[str]) -> Detection:
    """
    Decide which loop the channels belong to: ``df`` + ``Phase`` is the PLL, ``QPlusAmpl`` +
    ``Drive`` the amplitude loop. A single channel of a pair still allows a step-shape analysis.
    """
    canon: Dict[str, str] = {}
    for n in channel_names:
        c = _ALIASES.get(str(n).strip().lower())
        if c:
            canon[c] = n
    if {"df", "Phase"} <= canon.keys():
        return Detection(PLL, True, canon, "PLL test: df (output) and Phase (error) recorded")
    if {"QPlusAmpl", "Drive"} <= canon.keys():
        return Detection(AFL, True, canon, "Amplitude-loop test: QPlusAmpl and Drive recorded")
    if "df" in canon:
        return Detection(PLL, False, canon, "PLL test, but Phase is missing: only the df step shape can be analysed")
    if "QPlusAmpl" in canon:
        return Detection(AFL, False, canon, "Amplitude-loop test, but Drive is missing: only the amplitude step can be analysed")
    return Detection(None, False, canon, "Unrecognised channels: record df+Phase (PLL) or QPlusAmpl+Drive (amplitude loop)")


# ---------------------------------------------------------------------------
# the experimental test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepTestPlan:
    """
    One step-train test: ``n_events`` toggles of the stepped parameter between two levels.

    Order of a run: apply (Kp, Ki) and the first level, wait ``settle_s`` so the loop is
    locked and its integral has converged, record ``lead_s`` of the settled level, toggle
    every ``hold_s``, record ``tail_s`` after the last event, then put the parameter back
    to ``base``. The default is the typical train: 7 events (3 square pulses plus one) of
    0.5 s.
    """

    loop: str = "pll"
    base: float = 0.0             # PLL: current `use` frequency [Hz]; amplitude loop: current Ref
    step: float = 1.0             # PLL: +-Hz about base; amplitude loop: +- fraction of base (0.10 = +-10 %)
    hold_s: float = 0.5
    n_events: int = 7
    lead_s: float = 1.0
    tail_s: float = 0.5
    settle_s: float = 2.0
    start_high: bool = False      # the first (settled) level is the high one; the first event goes low

    def __post_init__(self):
        if self.loop not in LOOPS:
            raise ValueError("loop must be 'pll' or 'afl'")
        if self.n_events < 3:
            raise ValueError("need at least 3 events")
        if not (0.15 <= self.hold_s <= 10.0):
            raise ValueError("hold_s must be between 0.15 s and 10 s")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.loop == "afl" and (self.base <= 0 or self.step >= 1.0):
            raise ValueError("amplitude test needs Ref > 0 and a fractional step below 1")

    @property
    def loop_def(self) -> LoopDef:
        return LOOPS[self.loop]

    @property
    def low(self) -> float:
        return self.base * (1.0 - self.step) if self.loop_def.relative_step else self.base - self.step

    @property
    def high(self) -> float:
        return self.base * (1.0 + self.step) if self.loop_def.relative_step else self.base + self.step

    @property
    def levels(self) -> List[float]:
        first, second = (self.high, self.low) if self.start_high else (self.low, self.high)
        return [first if k % 2 == 0 else second for k in range(self.n_events + 1)]

    @property
    def event_times(self) -> List[float]:
        return [self.lead_s + k * self.hold_s for k in range(self.n_events)]

    @property
    def duration(self) -> float:
        return self.lead_s + self.n_events * self.hold_s + self.tail_s

    @property
    def expected_step(self) -> float:
        """Size of one toggle in the units of the stepped parameter."""
        return self.high - self.low

    def train(self) -> StepTrain:
        return StepTrain(tuple(self.event_times), tuple(self.levels))

    def commands(self) -> List[Tuple[float, float]]:
        """(time since the recording started, value to write) for every event."""
        return list(zip(self.event_times, self.levels[1:]))


@dataclass
class CapturedTest:
    """A recorded test: channels by SXM name, the (nominal) event times, and the gains used."""

    plan: StepTestPlan
    t: np.ndarray
    channels: Dict[str, np.ndarray]
    event_times: List[float]        # when each toggle was sent, same time base as ``t``
    kp: float
    ki: float


# ---------------------------------------------------------------------------
# analysing a test
# ---------------------------------------------------------------------------
@dataclass
class StepTestResult:
    kp: float
    ki: float
    loop: str
    failure: Optional[str] = None
    n_steps: int = 0
    window_s: float = 0.0                 # how long after each step the response was followed
    latency_s: float = math.nan           # median command -> 10 % crossing time
    primary: Optional[M.StepMetrics] = None
    error: Optional[M.ErrorMetrics] = None       # PLL Phase transient
    secondary: Optional[M.StepMetrics] = None    # amplitude loop Drive
    noise_rms: float = math.nan           # step-to-step scatter of the primary channel
    # direction-split and tighter-band views, for the amplitude-loop scale-sweep protocol; None where
    # there were fewer than 2 steps of a direction, or (2pct) always alongside `primary`.
    primary_rising: Optional[M.StepMetrics] = None
    primary_falling: Optional[M.StepMetrics] = None
    primary_2pct: Optional[M.StepMetrics] = None      # same as `primary`, with a 2 % settling band
    secondary_rising: Optional[M.StepMetrics] = None
    secondary_falling: Optional[M.StepMetrics] = None
    drive_peak_abs: float = math.nan          # peak |Drive| during the transient, physical units
    drive_floor_frac: float = math.nan        # fraction of the test (from the first event) with Drive at its zero floor
    drive_settled: float = math.nan           # Drive before the first event (the level the floor is judged against)
    drive_rms_excursion: float = math.nan     # RMS(Drive - final) over the whole transient, not just the tail
    grid: Optional[np.ndarray] = None     # averaged responses (relative to the step), for plotting
    mean_primary: Optional[np.ndarray] = None
    std_primary: Optional[np.ndarray] = None
    mean_secondary: Optional[np.ndarray] = None   # Phase (PLL) or Drive (amplitude loop), sign-folded
    # the same direction split as primary_rising/falling etc., but the curves themselves (not just their
    # metrics) - what a plot needs to actually show the rising and falling shapes as two traces, since
    # mean_primary/mean_secondary above are folded over *both* directions and can hide a real asymmetry
    # between them (e.g. the amplitude loop kicking Drive hard to raise QPlusAmpl but just cutting it
    # near zero to let the resonator's own damping bring it back down).
    mean_primary_rising: Optional[np.ndarray] = None
    mean_primary_falling: Optional[np.ndarray] = None
    mean_secondary_rising: Optional[np.ndarray] = None
    mean_secondary_falling: Optional[np.ndarray] = None
    model: Optional[LoopModel] = None
    warnings: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # instrument state when the test ran (SXM read-back), for export


def drive_floor_fraction(t, drive, first_event: float, lead_s: float, level: float = DRIVE_FLOOR_LEVEL):
    """
    ``(fraction, settled)``: the share of samples from ``first_event`` on with Drive below ``level`` x the
    settled Drive (median over the lead before the first event), and that settled value. ``(nan, nan)``
    when there is no settled lead or the settled Drive is not positive.
    """
    t = np.asarray(t, float)
    d = np.asarray(drive, float)
    lead = (t > first_event - 0.6 * lead_s) & (t < first_event - 0.05)
    after = t >= first_event
    if lead.sum() < 10 or after.sum() < 10:
        return math.nan, math.nan
    settled = float(np.median(d[lead]))
    if not settled > 0:
        return math.nan, settled
    return float(np.mean(d[after] <= level * settled)), settled


def align_events(t, y, nominal, signs, expected_step, lead_s, search_pre=0.04, search_post=0.12):
    """
    Locate where each step really starts in the data: the first time the (sign-folded, lightly
    smoothed) signal has covered 10 % of the expected step, minus ``lead_s``. Steps whose crossing
    is not found keep their nominal time. Returns (aligned times, latencies or nan).
    """
    t = np.asarray(t, float)
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(0.002 / dt)))
    ys = ndimage.uniform_filter1d(np.asarray(y, float), size=w, mode="nearest") if w > 1 else np.asarray(y, float)
    aligned, lat = [], []
    for tn, s in zip(nominal, signs):
        pre = ys[(t >= tn - 0.12) & (t < tn - 0.01)]
        win = (t >= tn - search_pre) & (t <= tn + search_post)
        if len(pre) < 3 or win.sum() < 5:
            aligned.append(tn); lat.append(math.nan); continue
        y0 = float(np.median(pre))
        d = s * (ys[win] - y0)
        idx = np.flatnonzero(d >= 0.1 * abs(expected_step))
        if len(idx) == 0:
            aligned.append(tn); lat.append(math.nan); continue
        i = idx[0]
        tw = t[win]
        if i > 0 and d[i] != d[i - 1]:
            t10 = tw[i - 1] + (0.1 * abs(expected_step) - d[i - 1]) / (d[i] - d[i - 1]) * (tw[i] - tw[i - 1])
        else:
            t10 = tw[i]
        aligned.append(float(t10 - lead_s))
        lat.append(float(t10 - tn))
    return aligned, lat


def analyze_test(ct: CapturedTest, *, li_tau: Optional[float] = None, li_stages: int = 2,
                 ctrl_tau: Optional[float] = None, f0: Optional[float] = None, q: Optional[float] = None) -> StepTestResult:
    """
    Measure one recorded test (model-free), and - if the lock-in ``li_tau`` (and, for the amplitude
    loop, its ``ctrl_tau`` = Tau) is given - also identify the physical loop from the same recording.

    The response is folded over all events of either direction, each aligned on its own detected
    onset, so DDE/timer jitter does not smear the edges and holds of 0.2 s work. Slow tails that do
    not fit in the hold are reported as such rather than guessed.
    """
    plan = ct.plan
    ld = plan.loop_def
    res = StepTestResult(kp=ct.kp, ki=ct.ki, loop=plan.loop)
    det = detect_loop(list(ct.channels))
    if det.loop is None or det.loop.key != plan.loop:
        res.failure = f"channels {list(ct.channels)} do not match a {ld.name} test"
        return res
    if not det.complete:
        res.warnings.append(det.note)
    prim_name = det.names.get(ld.primary_channel)
    if prim_name is None:
        res.failure = f"{ld.primary_channel} was not recorded"
        return res

    t = np.asarray(ct.t, float)
    y = np.asarray(ct.channels[prim_name], float)
    dt = float(np.median(np.diff(t)))
    nominal = list(ct.event_times)
    signs = np.sign(plan.train().step_sizes)
    if len(nominal) != len(signs):
        res.failure = "event list does not match the plan"
        return res

    # amplitude loop: amplitude per Ref unit (kappa) from the settled first level
    if plan.loop == "afl":
        first = (t > nominal[0] - 0.6 * plan.lead_s) & (t < nominal[0] - 0.05)
        if first.sum() < 10:
            res.failure = "no settled data before the first event"
            return res
        kappa = float(np.median(y[first]) / plan.levels[0])
        expected = abs(kappa) * plan.expected_step
    else:
        kappa = 1.0
        expected = plan.expected_step

    aligned, lat = align_events(t, y, nominal, signs, expected, ld.align_lead_s)
    finite = [x for x in lat if not math.isnan(x)]
    res.latency_s = float(np.median(finite)) if finite else math.nan
    if len(finite) < len(nominal):
        res.warnings.append(f"the step onset was not found in {len(nominal) - len(finite)} of {len(nominal)} events")

    holds = np.diff(nominal + [nominal[-1] + plan.hold_s])
    pre_s = float(min(0.2, 0.5 * min(plan.lead_s, plan.hold_s)))
    post_s = float(plan.hold_s - 0.05)
    res.window_s = post_s
    try:
        grid, mean, std, n = M.average_steps(t, y, aligned, pre_s, post_s, dt, signs=signs)
    except M.StepNotDetectable:
        res.failure = "unmeasurable: no clean step in the primary channel (loop lost, ringing or not driven)"
        return res
    res.n_steps, res.grid, res.mean_primary, res.std_primary = n, grid, mean, std
    try:
        res.primary = M.step_response_metrics(grid, mean, 0.0, step_override=expected, target_step=expected)
    except M.StepNotDetectable:
        # two different reasons look the same here: a loop too slow to get there (smooth, small change) and a
        # loop oscillating so much that the step drowns (large spread): tell them apart by the spread
        spread = M.detrended_std(y[t >= nominal[0]])
        rest = float(np.median(y[t < nominal[0]])) if (t < nominal[0]).any() else float(np.median(y))
        excursion = float(np.max(np.abs(y - rest)))
        if expected and excursion > 5.0 * abs(expected):
            res.failure = f"unmeasurable: the loop ran away ({excursion / abs(expected):.0f}x the step)"
        elif expected and spread > 0.3 * abs(expected):
            res.failure = (f"unmeasurable: the loop oscillates (signal spread {spread / abs(expected) * 100:.0f} % "
                           "of the step)")
        else:
            res.failure = "unmeasurable: the primary channel did not follow the step"
        return res
    try:
        res.primary_2pct = M.step_response_metrics(grid, mean, 0.0, step_override=expected, target_step=expected, band=0.02)
    except M.StepNotDetectable:
        res.primary_2pct = None
    measured = abs(res.primary.y_final - res.primary.y_initial)
    if measured < 0.15 * expected:
        res.failure = "unmeasurable: the primary channel moved far less than the commanded step"
        return res
    if measured > 0 and not (0.5 * expected <= measured <= 1.6 * expected) and post_s >= 0.6:
        res.warnings.append(f"the {prim_name} step measured {measured:.3g} vs {expected:.3g} expected: "
                            "check channel units or that the loop followed the step")

    if n >= 3:
        late = grid > 0.5 * post_s
        res.noise_rms = float(math.sqrt(np.mean(std[late] ** 2)))

    # direction split (rising vs falling), for protocols that ask whether the two differ - the combined
    # fold above answers "what does a step look like", this answers "does the direction matter".
    signs_arr = np.asarray(signs, dtype=float)
    rising_idx = [i for i, s in enumerate(signs_arr) if s > 0]
    falling_idx = [i for i, s in enumerate(signs_arr) if s < 0]
    have_both_directions = len(rising_idx) >= 2 and len(falling_idx) >= 2
    if have_both_directions:
        try:
            _, mean_up, _, _ = M.average_steps(t, y, [aligned[i] for i in rising_idx], pre_s, post_s, dt,
                                               signs=[signs_arr[i] for i in rising_idx])
            res.mean_primary_rising = mean_up
            res.primary_rising = M.step_response_metrics(grid, mean_up, 0.0, step_override=expected, target_step=expected)
        except M.StepNotDetectable:
            pass
        try:
            _, mean_dn, _, _ = M.average_steps(t, y, [aligned[i] for i in falling_idx], pre_s, post_s, dt,
                                               signs=[signs_arr[i] for i in falling_idx])
            res.mean_primary_falling = mean_dn
            res.primary_falling = M.step_response_metrics(grid, mean_dn, 0.0, step_override=expected, target_step=expected)
        except M.StepNotDetectable:
            pass

    # secondary channel
    if plan.loop == "pll" and "Phase" in det.names:
        _, ph, _, _ = M.average_steps(t, np.asarray(ct.channels[det.names["Phase"]], float), aligned, pre_s, post_s, dt, signs=signs)
        pre = ph[grid < 0]
        ph = ph - float(np.median(pre[-max(5, int(0.3 * len(pre))):]))      # the loop returns to its pre-step level
        res.mean_secondary = ph
        try:
            res.error = M.error_transient_metrics(grid, ph, 0.0, rest_value=0.0)
        except M.StepNotDetectable:
            res.error = None
    elif plan.loop == "afl" and "Drive" in det.names:
        drive_y = np.asarray(ct.channels[det.names["Drive"]], float)
        res.drive_floor_frac, res.drive_settled = drive_floor_fraction(t, drive_y, nominal[0], plan.lead_s)
        if not math.isnan(res.drive_floor_frac) and res.drive_floor_frac > DRIVE_FLOOR_TIME_MAX:
            res.warnings.append(
                f"Drive sat at its zero floor for {res.drive_floor_frac * 100:.0f} % of the test (settled Drive "
                f"{res.drive_settled:.3g}): the loop switched Drive off instead of regulating, and the amplitude fell at "
                "the ring-down rate. Lower both gains, or use a smaller step / larger amplitude.")
        _, dr, _, _ = M.average_steps(t, drive_y, aligned, pre_s, post_s, dt, signs=signs)
        res.mean_secondary = dr
        try:
            res.secondary = M.step_response_metrics(grid, dr, 0.0)
        except M.StepNotDetectable:
            res.secondary = None
        if res.secondary is not None:
            res.drive_peak_abs = float(np.max(np.abs(dr)))
            _, res.drive_rms_excursion = M.transient_excursion(dr, res.secondary.y_final)
        if have_both_directions:
            try:
                _, dr_up, _, _ = M.average_steps(t, drive_y, [aligned[i] for i in rising_idx], pre_s, post_s, dt,
                                                 signs=[signs_arr[i] for i in rising_idx])
                res.mean_secondary_rising = dr_up
                res.secondary_rising = M.step_response_metrics(grid, dr_up, 0.0)
            except M.StepNotDetectable:
                pass
            try:
                _, dr_dn, _, _ = M.average_steps(t, drive_y, [aligned[i] for i in falling_idx], pre_s, post_s, dt,
                                                 signs=[signs_arr[i] for i in falling_idx])
                res.mean_secondary_falling = dr_dn
                res.secondary_falling = M.step_response_metrics(grid, dr_dn, 0.0)
            except M.StepNotDetectable:
                pass

    # physical loop identification from the same recording
    if li_tau and det.complete:
        try:
            y_in = np.asarray(ct.channels[det.names[ld.y_channel]], float)
            u_out = np.asarray(ct.channels[det.names[ld.u_channel]], float)
            # host-clock event times jitter by several ms (timers, DDE); the onsets detected in the data do not
            times = aligned if len(finite) == len(nominal) and all(b > a for a, b in zip(aligned, aligned[1:])) else nominal
            train = StepTrain(tuple(times), tuple(plan.levels))
            cap = LoopCapture(kind=plan.loop, t=t, y=y_in, u=u_out, train=train, kp_raw=ct.kp, ki_raw=ct.ki,
                              meta={"block_s": dt})
            res.model = identify_loop(cap, li_tau=li_tau, li_stages=li_stages, ctrl_tau=ctrl_tau, f0=f0, q=q)
            res.warnings.extend(res.model.warnings)
        except (ValueError, np.linalg.LinAlgError) as e:
            res.warnings.append(f"physical identification skipped: {e}")
    return res


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyLimits:
    """Bounds on what a screening run may write to the instrument."""

    max_gain_factor: float = 16.0       # |gain| may not exceed baseline x this ...
    min_gain_factor: float = 1.0 / 16   # ... nor fall below baseline x this
    phase_limit_deg: float = 75.0       # PLL: |Phase| beyond this means the lock is gone
    amplitude_floor: float = 0.3        # amplitude loop: amplitude below this fraction of Ref is a collapse
    amplitude_ceiling: float = 3.0


def runaway_reason(plan: StepTestPlan, limits: SafetyLimits, channels: Dict[str, np.ndarray],
                   kappa: Optional[float] = None) -> Optional[str]:
    """
    Check the most recent samples of a running test. Returns a reason to abort, or None.

    PLL: ``Phase`` beyond the limit, or ``df`` far outside the stepped range. Amplitude loop: amplitude
    collapsing below / running above a fraction of the expected level.
    """
    if plan.loop == "pll":
        ph = channels.get("Phase")
        if ph is not None and len(ph) and float(np.max(np.abs(ph[-50:]))) > limits.phase_limit_deg:
            return f"|Phase| exceeded {limits.phase_limit_deg:.0f} deg: the PLL lost the resonance"
        df = channels.get("df")
        if df is not None and len(df):
            span = 6.0 * plan.expected_step + 3.0
            if float(np.max(np.abs(df[-50:] - np.median(df[: max(len(df) // 4, 1)])))) > span:
                return "df ran away far beyond the stepped range"
    else:
        amp = channels.get("QPlusAmpl")
        if amp is not None and len(amp) and kappa:
            lvl = kappa * plan.base
            last = float(np.median(amp[-20:]))
            if last < limits.amplitude_floor * lvl:
                return "amplitude collapsed below the floor: the sensor stopped oscillating"
            if last > limits.amplitude_ceiling * lvl:
                return "amplitude ran far above the setpoint"
    return None
