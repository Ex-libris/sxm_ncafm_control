"""
The tuning workflow for the SXM's two PI loops (PLL and amplitude feedback).

    define a test  ->  detect what the scope channels are  ->  analyse the train
        ->  classify the response  ->  advise (higher / lower / different pairing)
        ->  screen a (Kp, Ki) grid into a 2D map with islands of good response

Everything here is pure logic on numpy arrays (no Qt, no hardware); the GUI tab
and the instrument I/O live in ``gui/tuning_tab.py``.

The two tests are the manual's:

* **PLL** - toggle ``DNC use`` by +-1 Hz around f_res and watch ``df`` and ``Phase``;
  ``df`` should be a rectangular, non-overshooting wave.
* **Amplitude loop** - toggle the amplitude ``Ref`` by +-10 % and watch ``QPlusAmpl``
  and ``Drive``; ``QPlusAmpl`` should be rectangular, ``Drive`` may overshoot but must
  not saturate.

The manual's tuning rule is built in: *find a good Ki:Kp ratio, then raise or lower
both together to make the loop faster or slower*. In the map that is a move along a
diagonal (constant ratio); moving across diagonals changes the shape.
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
              step_param=("EDIT", "Edit23"), gain_sign=+1, relative_step=True, default_step=0.10,
              align_lead_s=0.03, typical_ratio=1e-4)
LOOPS = {"pll": PLL, "afl": AFL}


# ---------------------------------------------------------------------------
# amplitude-loop starting values (manual, pp. 6-7)
# ---------------------------------------------------------------------------
AFL_OUTPUT_GAINS = (0.1, 1.0, 10.0)     # DNC 'Output Gain' ranges, +-V peak


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
# what the loop has to achieve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    """Requirements that turn a measured response into a verdict."""

    rise_max: float                      # 10-90 % rise of the primary channel [s]
    overshoot_max: float = 0.10
    max_extrema: int = 3                 # ringing extrema tolerated after the step
    decay_max: Optional[float] = None    # PLL: the Phase transient must have decayed by then [s]
    settle_max: Optional[float] = None   # 5 % settling limit for the primary channel [s]
    name: str = ""

    @classmethod
    def from_scan(cls, t_line: float, n_px: int, rise_fraction: float = 0.5, decay_fraction: float = 0.10):
        """Imaging target: rise within ``rise_fraction`` of a pixel dwell; Phase tail gone within 10 % of a line."""
        t_px = t_line / n_px
        return cls(rise_max=rise_fraction * t_px, settle_max=t_px, decay_max=decay_fraction * t_line,
                   name=f"scan {t_line:g} s/line x {n_px} px (pixel {t_px * 1e3:.1f} ms)")

    @classmethod
    def manual(cls, rise_max: float, overshoot_max: float = 0.10, decay_max: Optional[float] = None):
        """A response-time target given directly (e.g. for a passive background loop)."""
        return cls(rise_max=rise_max, overshoot_max=overshoot_max, decay_max=decay_max,
                   name=f"rise <= {rise_max * 1e3:.0f} ms")


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
    verdict: Optional["Verdict"] = None


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
# verdict and advice
# ---------------------------------------------------------------------------
CATEGORIES = ("lost", "ringing", "overshoot", "slow_tail", "too_slow", "good")
CATEGORY_LABEL = {
    "lost": "lost / unstable", "ringing": "ringing", "overshoot": "overshoot",
    "slow_tail": "slow tail (Ki too low)", "too_slow": "too slow", "good": "good (rectangular)",
}


@dataclass
class Verdict:
    category: str
    reasons: List[str]
    fast_margin: float = math.nan      # rise_max / rise: >= 2 means faster than needed (room to lower the gains)
    score: float = 0.0                 # 0..1 quality of the shape (1 = clean and inside every limit)


def classify(res: StepTestResult, target: Target) -> Verdict:
    """Turn a measured response into one of :data:`CATEGORIES`, with the numbers that decided it."""
    reasons: List[str] = []
    if res.failure or res.primary is None:
        return Verdict("lost", [res.failure or "no measurable step"], score=0.0)
    p = res.primary
    ringing = p.n_extrema > target.max_extrema
    if not math.isnan(res.noise_rms) and abs(p.step_size) > 0 and res.noise_rms > 0.2 * abs(p.step_size):
        return Verdict("ringing", [f"step-to-step scatter {res.noise_rms:.3g} is {res.noise_rms / abs(p.step_size) * 100:.0f} % "
                                   "of the step: sustained oscillation"], score=0.1)
    if not math.isnan(p.damping_ratio) and p.damping_ratio < 0.25 and p.n_extrema >= 2:
        ringing = True
    if res.error is not None and res.error.sign_changes >= 2:
        ringing = True
    if ringing:
        reasons.append(f"{p.n_extrema} ringing extrema" +
                       (f", damping ratio {p.damping_ratio:.2f}" if not math.isnan(p.damping_ratio) else ""))
        return Verdict("ringing", reasons, score=0.15)
    if p.overshoot > target.overshoot_max:
        return Verdict("overshoot", [f"overshoot {p.overshoot * 100:.0f} % (limit {target.overshoot_max * 100:.0f} %)"],
                       score=max(0.2, 1.0 - p.overshoot / max(target.overshoot_max, 1e-9) * 0.3))
    if target.decay_max is not None and res.error is not None:
        d = res.error.decay_time
        if math.isinf(d):
            if res.window_s >= target.decay_max:
                return Verdict("slow_tail", [f"the Phase transient has not decayed within {res.window_s * 1e3:.0f} ms "
                                             f"(limit {target.decay_max * 1e3:.0f} ms)"], score=0.5)
            res.warnings.append(f"the {res.window_s * 1e3:.0f} ms hold is shorter than the {target.decay_max * 1e3:.0f} ms "
                                "Phase-decay limit: lengthen the hold to judge the tail")
        elif d > target.decay_max:
            return Verdict("slow_tail", [f"Phase decays in {d * 1e3:.0f} ms (limit {target.decay_max * 1e3:.0f} ms)"], score=0.55)
    if target.settle_max is not None and not p.settled and res.window_s >= target.settle_max:
        return Verdict("slow_tail", [f"not settled within {res.window_s * 1e3:.0f} ms"], score=0.5)
    if math.isnan(p.rise_time) or p.rise_time > target.rise_max:
        rt = "not reached" if math.isnan(p.rise_time) else f"{p.rise_time * 1e3:.1f} ms"
        return Verdict("too_slow", [f"rise {rt} (needs <= {target.rise_max * 1e3:.1f} ms)"], score=0.6)
    margin = target.rise_max / p.rise_time if p.rise_time > 0 else math.inf
    ok = [f"rise {p.rise_time * 1e3:.1f} ms", f"overshoot {p.overshoot * 100:.0f} %"]
    if not math.isnan(res.noise_rms):
        ok.append(f"noise {res.noise_rms:.3g}")
    return Verdict("good", ok, fast_margin=margin, score=1.0 - 0.5 * p.overshoot / max(target.overshoot_max, 1e-9))


def classify_prediction(pred, target: Target) -> Verdict:
    """Classify a :class:`~tuning.loopid.Prediction` with exactly the rules used for measurements."""
    if not pred.stable or pred.primary is None:
        return Verdict("lost", ["the identified model predicts an unstable loop"], score=0.0)
    r = StepTestResult(kp=pred.kp_raw, ki=pred.ki_raw, loop="", n_steps=1, window_s=math.inf)
    r.primary, r.error, r.secondary = pred.primary, pred.error, pred.secondary
    return classify(r, target)


@dataclass(frozen=True)
class Suggestion:
    kind: str            # 'scale_both' | 'change_ki' | 'change_kp' | 'accept' | 'back_off' | 'new_pairing'
    kp: Optional[float]
    ki: Optional[float]
    why: str


def advise(res: StepTestResult, verdict: Verdict, target: Target) -> List[Suggestion]:
    """
    What to try next, from the manual's rules: shape is set by the Ki:Kp ratio; speed is set by raising
    or lowering *both* while keeping the ratio.
    """
    kp, ki, c = res.kp, res.ki, verdict.category
    out: List[Suggestion] = []
    if c == "lost":
        out.append(Suggestion("back_off", kp * 0.5, ki * 0.5, "Loop lost or unmeasurable: go back to the last good pair, or halve both gains."))
        out.append(Suggestion("new_pairing", kp * 0.5, ki * 0.25, "Or lower Ki more than Kp: a smaller integral term is the safer direction."))
    elif c == "ringing":
        out.append(Suggestion("change_ki", kp, ki * 0.5, "Ringing: lower Ki at the same Kp (the manual's fix for overshoot at high Ki)."))
        out.append(Suggestion("new_pairing", kp * 0.7, ki * 0.5, "If it persists, lower Kp as well - the loop bandwidth is too high for the lock-in TimeConstant."))
    elif c == "overshoot":
        out.append(Suggestion("change_ki", kp, ki * 0.6, "Overshoot: lower Ki at the same Kp."))
        out.append(Suggestion("new_pairing", kp * 0.8, ki * 0.5, "Or lower both, weighting Ki, if you also want less noise."))
    elif c == "slow_tail":
        out.append(Suggestion("change_ki", kp, ki * 1.7, "Slow tail: raise Ki at the same Kp so the residual is removed faster."))
        out.append(Suggestion("scale_both", kp * 1.4, ki * 1.4, "Or raise both together, keeping the Ki:Kp ratio."))
    elif c == "too_slow":
        f, need, hi = 1.25, math.nan, (10.0 if res.loop == "afl" else 2.5)     # the amplitude loop is often decades off
        if res.primary is not None and not math.isnan(res.primary.rise_time) and target.rise_max > 0:
            need = res.primary.rise_time / target.rise_max
            f = float(np.clip(need, 1.25, hi))
        more = f" About x{need:.0f} is needed: a Scale scan gets there faster." if need > hi else ""
        out.append(Suggestion("scale_both", kp * f, ki * f,
                              f"Clean but too slow: raise both by x{f:.2f}, keeping the ratio (the manual's way to a faster loop)." + more))
    else:  # good
        if verdict.fast_margin >= 2.0:
            out.append(Suggestion("scale_both", kp * 0.7, ki * 0.7,
                                  f"Faster than needed by x{verdict.fast_margin:.1f}: lower both by x0.7 to cut noise, then re-test."))
        else:
            out.append(Suggestion("accept", kp, ki, "Meets the target with little margin: a good candidate for this scan."))
    return out


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


def gain_within_limits(kp: float, ki: float, reference: Tuple[float, float], limits: SafetyLimits) -> bool:
    """Both gains within ``[min, max]_gain_factor`` times the known-good reference pair."""
    lo, hi = limits.min_gain_factor, limits.max_gain_factor
    fk = abs(kp / reference[0])
    fi = abs(ki / reference[1])
    return lo * (1 - 1e-9) <= fk <= hi * (1 + 1e-9) and lo * (1 - 1e-9) <= fi <= hi * (1 + 1e-9)


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


# ---------------------------------------------------------------------------
# the (Kp, Ki) map
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """
    Log-spaced grid of raw (Kp, Ki) pairs around a baseline, indexed by two physically meaningful axes
    instead of raw Kp/Ki exponents:

    - ``speed``: Kp and Ki raised or lowered *together*, same ratio (the manual's "faster / slower").
    - ``shape``: the Ki:Kp ratio shifted *relative to the baseline's ratio* (more integral- or
      proportional-heavy), independent of speed.

    ``kp`` depends only on ``speed``; ``ki`` depends on both, since a shape shift is applied on top of
    whatever speed you're at. (The old diagonal-in-exponent-space "same ratio" line is now literally a
    column: fixed shape index, varying speed index.)
    """

    kp0: float
    ki0: float
    factor: float = 2.0
    speed_exps: Tuple[int, ...] = (-2, -1, 0, 1, 2)
    shape_exps: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2)

    def kp(self, i: int) -> float:
        return self.kp0 * self.factor ** self.speed_exps[i]

    def ki(self, i: int, j: int) -> float:
        return self.ki0 * self.factor ** (self.speed_exps[i] + self.shape_exps[j])

    @property
    def dims(self) -> Tuple[int, int]:
        return len(self.speed_exps), len(self.shape_exps)

    @classmethod
    def scan(cls, kp0: float, ki0: float, factor: float, n_down: int, n_up: int) -> "GridSpec":
        """Pure speed scan: Ki:Kp stays at the baseline's ratio (shape fixed at 0), only speed varies."""
        exps = tuple(range(-int(n_down), int(n_up) + 1))
        return cls(kp0=kp0, ki0=ki0, factor=factor, speed_exps=exps, shape_exps=(0,))

    def refine(self, i: int, j: int, factor: Optional[float] = None) -> "GridSpec":
        """A finer 3x3 grid (step factor**0.5 by default) centred on cell (i, j)."""
        return GridSpec(kp0=self.kp(i), ki0=self.ki(i, j), factor=factor or math.sqrt(self.factor),
                        speed_exps=(-1, 0, 1), shape_exps=(-1, 0, 1))


@dataclass
class Island:
    cells: List[Tuple[int, int]]
    best: Tuple[int, int]

    @property
    def size(self) -> int:
        return len(self.cells)


class ScreeningMap:
    """
    Results of screening a :class:`GridSpec`: one measured test per cell, classified against a target.

    Cells are visited from the baseline outward (nearest first). A cell that lost the loop makes every
    more aggressive cell (larger |Kp| *and* |Ki|) skipped, so screening never walks deeper into an unstable region.
    """

    def __init__(self, grid: GridSpec, target: Target, limits: SafetyLimits = SafetyLimits(),
                 reference: Optional[Tuple[float, float]] = None):
        """``reference`` = the known-good baseline the safety limits are measured against (default: the grid centre)."""
        self.grid, self.target, self.limits = grid, target, limits
        self.reference = reference or (grid.kp0, grid.ki0)
        self.results: Dict[Tuple[int, int], StepTestResult] = {}
        self.skipped: Dict[Tuple[int, int], str] = {}
        self.prior: Dict[Tuple[int, int], str] = {}          # optional model-predicted category per cell
        self.baseline = (grid.speed_exps.index(0) if 0 in grid.speed_exps else None,
                         grid.shape_exps.index(0) if 0 in grid.shape_exps else None)

    # -- ordering and safety --------------------------------------------------------------
    def _in_limits(self, i, j) -> bool:
        return gain_within_limits(self.grid.kp(i), self.grid.ki(i, j), self.reference, self.limits)

    def cells(self) -> List[Tuple[int, int]]:
        nk, ni = self.grid.dims
        return [(i, j) for i in range(nk) for j in range(ni)]

    def order(self) -> List[Tuple[int, int]]:
        """Cells nearest the baseline first (in log-gain distance); ties: lower gains first (safer)."""
        def key(c):
            i, j = c
            e_k = self.grid.speed_exps[i]
            e_i = e_k + self.grid.shape_exps[j]        # effective Ki exponent at this (speed, shape)
            return (math.hypot(e_k, e_i), e_k + e_i, e_k)
        return sorted(self.cells(), key=key)

    def _dominating_failure(self, i, j) -> Optional[Tuple[int, int]]:
        kp_e = self.grid.speed_exps[i]
        ki_e = kp_e + self.grid.shape_exps[j]
        for (a, b), r in self.results.items():
            v = r.verdict
            if v is not None and v.category == "lost":
                a_kp = self.grid.speed_exps[a]
                a_ki = a_kp + self.grid.shape_exps[b]
                if kp_e >= a_kp and ki_e >= a_ki:
                    return (a, b)
        return None

    def next_cell(self) -> Optional[Tuple[int, int]]:
        """The next cell to test, or None when the map is complete."""
        for c in self.order():
            if c in self.results or c in self.skipped:
                continue
            i, j = c
            if not self._in_limits(i, j):
                self.skipped[c] = "outside the safety limits"
                continue
            f = self._dominating_failure(i, j)
            if f is not None:
                self.skipped[c] = f"more aggressive than cell {f}, which lost the loop"
                continue
            if self.prior.get(c) == "lost":
                self.skipped[c] = "the identified model predicts this pair is unstable"
                continue
            return c
        return None

    def record(self, cell: Tuple[int, int], res: StepTestResult) -> Verdict:
        res.verdict = classify(res, self.target)
        self.results[cell] = res
        return res.verdict

    # -- views for plotting / decisions --------------------------------------------------------
    def category_grid(self) -> np.ndarray:
        nk, ni = self.grid.dims
        out = np.full((nk, ni), "untested", dtype=object)
        for c, why in self.skipped.items():
            out[c] = "skipped"
        for c, r in self.results.items():
            out[c] = r.verdict.category if r.verdict else "untested"
        return out

    def value_grid(self, what: str) -> np.ndarray:
        """Array (speed index, shape index) of 'rise' [s], 'overshoot', 'noise' or 'score'; nan where untested."""
        nk, ni = self.grid.dims
        out = np.full((nk, ni), np.nan)
        for c, r in self.results.items():
            if r.primary is None:
                continue
            out[c] = {"rise": r.primary.rise_time, "overshoot": r.primary.overshoot, "noise": r.noise_rms,
                      "score": r.verdict.score if r.verdict else np.nan}[what]
        return out

    def islands(self) -> List[Island]:
        """Connected regions (4-neighbour) of 'good' cells: the islands of stable, rectangular response."""
        cat = self.category_grid()
        mask = (cat == "good")
        labels, n = ndimage.label(mask)
        noise = self.value_grid("noise")
        out = []
        for k in range(1, n + 1):
            cells = [tuple(int(v) for v in c) for c in np.argwhere(labels == k)]
            cells.sort()
            def key(c):
                nz = noise[c]
                return (math.inf if math.isnan(nz) else nz, abs(self.grid.kp(c[0])))
            out.append(Island(cells=cells, best=min(cells, key=key)))
        return sorted(out, key=lambda isl: -isl.size)

    def best(self) -> Optional[Tuple[int, int]]:
        """Best 'good' cell overall: lowest noise (ties: lowest gains)."""
        isl = self.islands()
        if not isl:
            return None
        noise = self.value_grid("noise")
        cands = [c for i in isl for c in i.cells]
        return min(cands, key=lambda c: (math.inf if math.isnan(noise[c]) else noise[c], abs(self.grid.kp(c[0]))))

    def speed_line(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Cells with the same shape (Ki:Kp ratio) as ``cell``, slowest (lowest gains) first: the bandwidth line."""
        j = cell[1]
        cs = [c for c in self.cells() if c[1] == j]
        return sorted(cs, key=lambda c: self.grid.speed_exps[c[0]])

    def pair(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return self.grid.kp(cell[0]), self.grid.ki(cell[0], cell[1])

    def refined(self, cell: Tuple[int, int]) -> "ScreeningMap":
        """A finer 3x3 map centred on ``cell``, with the same target and the same safety reference."""
        return ScreeningMap(self.grid.refine(*cell), self.target, self.limits, reference=self.reference)

    def suggest_refinement(self) -> Optional[Tuple[Tuple[int, int], str]]:
        """
        Where to zoom in next: the best good cell if there is an island, otherwise the near miss
        (the measured cell closest to meeting its limit). Returns (cell, reason) or None.
        """
        best = self.best()
        if best is not None:
            kp, ki = self.pair(best)
            return best, f"lowest-noise good cell (Kp={kp:.4g}, Ki={ki:.4g}): map its neighbourhood in finer steps"
        t = self.target
        miss: List[Tuple[float, Tuple[int, int], str]] = []
        for c, r in self.results.items():
            if r.primary is None or r.verdict is None or r.verdict.category in ("lost", "ringing"):
                continue
            cat, p = r.verdict.category, r.primary
            if cat == "too_slow" and not math.isnan(p.rise_time):
                miss.append((p.rise_time / t.rise_max - 1.0, c,
                             f"only {p.rise_time * 1e3:.1f} ms vs {t.rise_max * 1e3:.1f} ms allowed"))
            elif cat == "overshoot":
                miss.append((p.overshoot / t.overshoot_max - 1.0, c,
                             f"overshoot {p.overshoot * 100:.0f} % vs {t.overshoot_max * 100:.0f} % allowed"))
            elif cat == "slow_tail" and r.error is not None and t.decay_max and not math.isinf(r.error.decay_time):
                miss.append((r.error.decay_time / t.decay_max - 1.0, c,
                             f"Phase tail {r.error.decay_time * 1e3:.0f} ms vs {t.decay_max * 1e3:.0f} ms allowed"))
        if not miss:
            return None
        gap, c, why = min(miss)
        return c, f"no island yet; nearest miss ({why}): zoom in around it"

    def set_prior_from_model(self, model) -> None:
        """
        Fill ``prior`` from an identified :class:`~tuning.loopid.LoopModel`. Only pairs the model
        predicts to be unstable are acted upon (they are skipped); the rest is informational.
        """
        self.prior = {}
        for c in self.cells():
            kp, ki = self.pair(c)
            self.prior[c] = classify_prediction(model.predict(kp, ki), self.target).category

    def summary(self) -> str:
        cat = self.category_grid()
        cells = self.cells()
        counts = {k: sum(1 for c in cells if cat[c] == k) for k in list(CATEGORIES) + ["skipped", "untested"]}
        lines = ["Map: " + ", ".join(f"{counts[k]} {CATEGORY_LABEL.get(k, k)}" for k in counts if counts[k])]
        isl = self.islands()
        if not isl:
            lines.append("No island of good response found yet.")
        for n, i in enumerate(isl, 1):
            kps = sorted({self.grid.kp(c[0]) for c in i.cells})
            kis = sorted({self.grid.ki(c[0], c[1]) for c in i.cells})
            kp, ki = self.pair(i.best)
            lines.append(f"Island {n}: {i.size} cells, Kp {kps[0]:.4g}..{kps[-1]:.4g}, Ki {kis[0]:.4g}..{kis[-1]:.4g}; "
                         f"best (lowest noise) Kp={kp:.4g}, Ki={ki:.4g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# amplitude loop: guided scale sweep (hold Kp:Ki, sweep one common scale g)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepLimits:
    """Thresholds for tiering a scale sweep and deciding where to stop it. Proposed defaults - meant
    to be adjusted once there is real hardware data to check them against, not taken as exact."""

    noise_jump_frac: float = 0.5          # Drive post-settling noise rise vs. the previous point: "substantial"
    settle_improve_min_frac: float = 0.10  # below this relative gain in 2 % settling, more scale stops helping
    saturation_frac: float = 0.9          # peak |Drive| at/above this fraction of the output-gain range: saturated


SCALE_TIER_LABEL = {"too_slow": "too slow", "acceptable": "acceptable",
                    "near_optimum": "near optimum", "too_aggressive": "too aggressive"}


@dataclass
class ScalePoint:
    """One tested point of a scale sweep: the gains used, its full measurement, and its tier."""

    g: float
    kp: float
    ki: float
    result: StepTestResult
    tier: str
    reasons: List[str] = field(default_factory=list)


def _p_vs_i_note(points: List[ScalePoint]) -> str:
    """
    A short, data-driven note on whether the sweep favours relatively more P or more I, from the
    measured trend - not an assumption that the nominal ~1e4 ratio is optimal.
    """
    aggressive = next((p for p in points if p.tier == "too_aggressive"), None)
    if aggressive is not None and any("overshoot" in r or "ringing" in r for r in aggressive.reasons):
        return ("Overshoot/ringing appeared before settling was exceeded: the data favours relatively less "
               "proportional weight (lower Kp:Ki) rather than pushing the common scale further.")
    feasible = [p for p in points if p.tier in ("acceptable", "near_optimum")]
    if feasible:
        r = feasible[-1].result
        if r.primary_2pct is not None and not math.isinf(r.primary_2pct.settling_time) and r.primary_2pct.settling_time > 0 \
           and not math.isnan(r.primary.rise_time):
            tail = r.primary_2pct.settling_time - r.primary.rise_time
            if tail > 0.5 * r.primary_2pct.settling_time:
                return ("The rise is fast relative to the 2 % settling time (a lingering tail after the initial "
                       "move): the data favours relatively more integral action to clear that residual faster.")
    return "No clear P-vs-I bias in the tested range; the nominal ~1e4 ratio looks reasonable here."


def assess_scale_sweep(gs: Sequence[float], results: Sequence[StepTestResult], target: Target,
                       output_gain_v: Optional[float] = None,
                       limits: SweepLimits = SweepLimits()) -> Tuple[List[ScalePoint], Optional[float], str]:
    """
    Tier each point of a Kp:Ki-held-fixed scale sweep as too_slow / acceptable / near_optimum /
    too_aggressive and recommend a final scale, per the manual's rule: raise the common gain until
    Drive saturates, overshoot/ringing appears, or Drive noise rises substantially, then back off.

    Stops assessing once a point is ``too_aggressive`` - later (more aggressive) points in ``gs`` are
    not evaluated, matching that same stopping rule. ``output_gain_v`` is the selected DNC Output Gain
    range; saturation is only checked when it is given.

    Returns ``(points, recommended_g, explanation)``. ``recommended_g`` is ``None`` if nothing tested
    both met the target and stayed stable.
    """
    points: List[ScalePoint] = []
    prev_feasible: Optional[ScalePoint] = None
    best_noise, best_idx = math.inf, None
    for g, res in zip(gs, results):
        if res.failure or res.primary is None:
            points.append(ScalePoint(g, res.kp, res.ki, res, "too_aggressive", [res.failure or "unmeasurable"]))
            break
        reasons: List[str] = []
        saturated = (output_gain_v is not None and not math.isnan(res.drive_peak_abs)
                    and res.drive_peak_abs >= limits.saturation_frac * output_gain_v)
        if saturated:
            reasons.append(f"Drive reached {res.drive_peak_abs:.3g} V, >= {limits.saturation_frac * 100:.0f} % of "
                           f"the +-{output_gain_v:g} V output gain range")
        overshoot_bad = res.primary.overshoot > target.overshoot_max or res.primary.n_extrema > target.max_extrema
        if overshoot_bad:
            reasons.append(f"overshoot {res.primary.overshoot * 100:.0f} % / {res.primary.n_extrema} ringing "
                           f"extrema beyond the target")
        noise_jump = False
        if (prev_feasible is not None and res.secondary is not None and prev_feasible.result.secondary is not None
                and not math.isnan(res.secondary.noise_rms) and prev_feasible.result.secondary.noise_rms > 0):
            jump = (res.secondary.noise_rms - prev_feasible.result.secondary.noise_rms) / prev_feasible.result.secondary.noise_rms
            if jump > limits.noise_jump_frac:
                noise_jump = True
                reasons.append(f"Drive noise rose {jump * 100:.0f} % vs. the previous point "
                               f"(limit {limits.noise_jump_frac * 100:.0f} %)")
        if saturated or overshoot_bad or noise_jump:
            points.append(ScalePoint(g, res.kp, res.ki, res, "too_aggressive", reasons))
            break
        settle = res.primary_2pct.settling_time if res.primary_2pct is not None else res.primary.settling_time
        meets_target = (not math.isnan(res.primary.rise_time) and res.primary.rise_time <= target.rise_max
                        and (target.settle_max is None or (not math.isinf(settle) and settle <= target.settle_max)))
        if not meets_target:
            rt = "not reached" if math.isnan(res.primary.rise_time) else f"{res.primary.rise_time * 1e3:.1f} ms"
            points.append(ScalePoint(g, res.kp, res.ki, res, "too_slow",
                                     [f"rise {rt} (target <= {target.rise_max * 1e3:.1f} ms)"]))
            continue
        point = ScalePoint(g, res.kp, res.ki, res, "acceptable", ["meets the target"])
        points.append(point)
        noise = res.primary.noise_rms if not math.isnan(res.primary.noise_rms) else math.inf
        if noise < best_noise:
            best_noise, best_idx = noise, len(points) - 1
        prev_feasible = point

    if best_idx is not None:
        points[best_idx].tier = "near_optimum"
        points[best_idx].reasons = ["lowest primary-channel noise among the feasible points"]

    aggressive = next((p for p in points if p.tier == "too_aggressive"), None)
    if aggressive is not None:
        recommended_g = 0.6 * aggressive.g            # the midpoint of the manual's 50-70 %
        basis = f"60 % of the first too-aggressive scale (g={aggressive.g:g})"
    elif best_idx is not None:
        recommended_g = points[best_idx].g
        basis = "the lowest-noise point that met the target among those tested"
    else:
        recommended_g = None
        basis = "no tested point both met the target and stayed stable - widen the scale list"

    lead = f"Recommend g ≈ {recommended_g:.3g} ({basis})." if recommended_g is not None else basis + "."
    return points, recommended_g, f"{lead} {_p_vs_i_note(points)}"
