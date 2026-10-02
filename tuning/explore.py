"""
Exploring the (Kp, Ki) plane of one loop, judged per step direction, every condition from the same baseline.

Three ideas, each a section below:

* **Assessment per direction.** A step test has up-steps and down-steps, and they are not equivalent: the
  amplitude loop can push Drive up but cannot push it below zero, so a down-step may only fall at the
  sensor's ring-down rate. Each direction gets its own status (clean / overshoot / floor / ringing / lost)
  and rise time; a condition is as good as its *worse* direction.
* **Baseline recovery.** Before every condition the baseline gains are written back and the loop must
  *verifiably* return to the reference state measured at the start of the run (amplitude and Drive, or
  Phase and df), so every condition starts from the same place - not from the previous condition.
* **Exploration in stages** (:class:`Explorer`): the baseline; a coarse grid in decades of Kp and Ki
  (Kp = 0, integral-only, as its own column); bisection along the common scale g at fixed Ki:Kp ratio
  (behaviour there is roughly monotonic: clean and slow at low g, overshoot / Drive at zero / lost at high
  g), which locates each ratio's edge in a few tests; a pattern search on the ratio at the best point
  (not monotonic: overshoot vs. a slow tail); repeats of the best candidates. Results are a trade-off
  (speed vs. noise), not a pass/fail against a target: :func:`trade_off` returns the clean front, the
  fastest clean point and the quietest point that is nearly as fast.

Pure numpy; no Qt, no hardware. Gains are SXM's raw values; nothing assumes what they mean physically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as M
from .workflow import DRIVE_FLOOR_TIME_MAX, StepTestResult

# ---------------------------------------------------------------------------
# assessment, per direction
# ---------------------------------------------------------------------------
STATUSES = ("clean", "slow", "overshoot", "floor", "ringing", "lost")
SEVERITY = {s: i for i, s in enumerate(STATUSES)}
STATUS_LABEL = {"clean": "clean", "slow": "too slow for the hold", "overshoot": "overshoot", "floor": "Drive at zero",
                "ringing": "ringing / oscillating", "lost": "lost / not measurable"}
# analyze_test failures that mean "did not get there within the hold", not "the loop broke": such a point is
# not clean, but it must not make the exploration skip the faster (more aggressive) points beyond it
_SLOW_FAILURES = ("did not follow the step", "moved far less than the commanded step")


@dataclass(frozen=True)
class ExploreLimits:
    overshoot_max: float = 0.10           # fraction of the step
    max_extrema: int = 3                  # ringing extrema tolerated after a step
    floor_max: float = DRIVE_FLOOR_TIME_MAX   # amplitude loop: share of the test with Drive at its zero floor
    scatter_max: float = 0.2              # step-to-step scatter beyond this fraction of the step: oscillating


@dataclass
class DirectionAssessment:
    status: str
    rise_s: float                         # 10-90 % [s]; nan when not reached
    overshoot: float                      # fraction of the step
    extrema: int
    reasons: List[str] = field(default_factory=list)


@dataclass
class Assessment:
    kp: float
    ki: float
    loop: str
    up: DirectionAssessment
    down: DirectionAssessment
    noise: float                          # Drive noise (amplitude loop) / df noise (PLL), loop's own units
    floor_frac: float = math.nan
    ringdown_limited: bool = False        # amplitude loop: the down-step only fell at the ring-down rate
    reasons: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return max((self.up.status, self.down.status), key=SEVERITY.__getitem__)

    @property
    def clean(self) -> bool:
        return self.status == "clean"

    @property
    def speed_s(self) -> float:
        """The slower direction's rise time [s]; inf when either was not measured."""
        r = [d.rise_s for d in (self.up, self.down)]
        return math.inf if any(math.isnan(x) for x in r) else max(r)


def _direction(m: Optional[M.StepMetrics], limits: ExploreLimits) -> DirectionAssessment:
    """Rise and overshoot of one direction. Ringing is judged on all steps together (see :func:`assess`):
    a direction alone averages only half the steps, too few to tell ringing from noise."""
    if m is None:
        return DirectionAssessment("slow", math.nan, math.nan, 0,
                                   ["this direction did not show a measurable step within the hold"])
    reasons = []
    status = "clean"
    if math.isnan(m.rise_time):
        status = "slow"
        reasons.append("did not reach 90 % of the step within the hold")
    elif m.overshoot > limits.overshoot_max:
        status = "overshoot"
        reasons.append(f"overshoot {m.overshoot * 100:.0f} %")
    return DirectionAssessment(status, m.rise_time, m.overshoot, m.n_extrema, reasons)


def _worse(d: DirectionAssessment, status: str, reason: str) -> None:
    if SEVERITY[status] > SEVERITY[d.status]:
        d.status = status
    d.reasons.append(reason)


def assess(res: StepTestResult, limits: ExploreLimits = ExploreLimits(),
           ring_down_s: Optional[float] = None) -> Assessment:
    """Judge one measured condition, each step direction on its own."""
    if res.failure or res.primary is None:
        why = res.failure or "no measurable step"
        st = ("ringing" if "oscillates" in why else
              "slow" if any(k in why for k in _SLOW_FAILURES) else "lost")
        return Assessment(res.kp, res.ki, res.loop, DirectionAssessment(st, math.nan, math.nan, 0, [why]),
                          DirectionAssessment(st, math.nan, math.nan, 0, [why]),
                          math.nan, res.drive_floor_frac, reasons=[why])
    # with fewer than 2 steps of each direction there is no split: both directions are the folded response
    split = res.primary_rising is not None or res.primary_falling is not None or res.n_steps >= 4
    up = _direction(res.primary_rising if split else res.primary, limits)
    down = _direction(res.primary_falling if split else res.primary, limits)
    reasons = []
    step = abs(res.primary.step_size)
    if res.primary.n_extrema > limits.max_extrema:
        why = f"{res.primary.n_extrema} ringing extrema (all steps averaged)"
        _worse(up, "ringing", why)
        _worse(down, "ringing", why)
    # Sustained oscillation: repeated steps differ by more than the noise explains. The averaged trace's noise
    # times sqrt(n) is the single-step noise; plain noise gives a scatter of about that, an oscillation that is
    # not locked to the steps averages out of the mean but not out of the scatter.
    noise_single = res.primary.noise_rms * math.sqrt(max(res.n_steps, 1))
    if (not math.isnan(res.noise_rms) and step > 0 and res.noise_rms > limits.scatter_max * step
            and res.noise_rms > 2.0 * noise_single):
        why = f"step-to-step scatter {res.noise_rms / step * 100:.0f} % of the step, beyond the noise"
        _worse(up, "ringing", why)
        _worse(down, "ringing", why)
    if res.error is not None and res.error.sign_changes >= 2:
        _worse(up, "ringing", "Phase error changes sign repeatedly")
        _worse(down, "ringing", "Phase error changes sign repeatedly")
    floor = res.drive_floor_frac
    if not math.isnan(floor) and floor > limits.floor_max:
        # Drive can only be cut to zero when the loop wants the amplitude *down*: a down-step property
        _worse(down, "floor", f"Drive at zero {floor * 100:.0f} % of the test")
    if res.loop == "afl":
        noise = res.secondary.noise_rms if res.secondary is not None else res.noise_rms
    else:
        noise = res.primary.noise_rms
    limited = False
    if res.loop == "afl" and ring_down_s and not math.isnan(down.rise_s):
        # an amplitude left to itself decays with the ring-down time: 10-90 % of that takes ln(9) * tau
        limited = down.rise_s >= 0.7 * math.log(9.0) * ring_down_s
        if limited:
            reasons.append(f"down-steps fall at the sensor's ring-down rate ({down.rise_s * 1e3:.0f} ms 10-90 %): "
                           "a limit of the sensor, not of the gains")
    return Assessment(res.kp, res.ki, res.loop, up, down, noise, floor, limited, reasons)


# ---------------------------------------------------------------------------
# baseline reference and recovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecoveryCriteria:
    window_s: float = 0.5           # judged on the last window of data
    amp_tol: float = 0.02           # amplitude loop: amplitude within 2 % of the reference
    drive_tol: float = 0.15         # ... and Drive within 15 % (or 4 sigma) of the reference
    drift_tol: float = 0.01         # steady: the two halves of the window agree within 1 % of the level
    phase_tol_deg: float = 0.5      # PLL: Phase within +-0.5 deg (manual's Auto-phase criterion)
    df_tol_hz: float = 0.05         # ... and df within 50 mHz (or 4 sigma) of the reference


@dataclass(frozen=True)
class BaselineReference:
    y: float                        # controller input: amplitude (afl) / Phase (pll)
    u: float                        # controller output: Drive / df
    y_sigma: float
    u_sigma: float


def _halves(x):
    x = np.asarray(x, float)
    h = len(x) // 2
    return float(np.median(x[:h])), float(np.median(x[h:]))


def steady(loop: str, y, u, crit: RecoveryCriteria = RecoveryCriteria()) -> Tuple[bool, str]:
    """No drift across the window (both halves agree); PLL also needs Phase near zero."""
    y, u = np.asarray(y, float), np.asarray(u, float)
    if len(y) < 20:
        return False, "not enough data yet"
    y1, y2 = _halves(y)
    if loop == "afl":
        level = abs(float(np.median(y)))
        if level <= 0:
            return False, "no oscillation amplitude"
        # drift must stand out from the noise (detrended: the noise without the drift itself), not just 1 %
        if abs(y2 - y1) > max(crit.drift_tol * level, M.detrended_std(y)):
            return False, f"amplitude still moving ({(y2 - y1) / level * 100:+.1f} % across {crit.window_s:g} s)"
        return True, ""
    ph = float(np.median(y))
    if abs(ph) > 2 * crit.phase_tol_deg:
        return False, f"Phase {ph:+.2f} deg"
    u1, u2 = _halves(u)
    if abs(u2 - u1) > max(crit.df_tol_hz, 4 * M.detrended_std(u)):
        return False, f"df still moving ({u2 - u1:+.3f} Hz across {crit.window_s:g} s)"
    return True, ""


def reference_from(y, u) -> BaselineReference:
    return BaselineReference(float(np.median(y)), float(np.median(u)), M.detrended_std(y), M.detrended_std(u))


def recovered(loop: str, y, u, ref: BaselineReference, crit: RecoveryCriteria = RecoveryCriteria()) -> Tuple[bool, str]:
    """Back at the reference state (and steady)."""
    ok, why = steady(loop, y, u, crit)
    if not ok:
        return False, why
    my, mu = float(np.median(y)), float(np.median(u))
    if loop == "afl":
        if abs(my - ref.y) > crit.amp_tol * abs(ref.y):
            return False, f"amplitude {(my - ref.y) / ref.y * 100:+.1f} % from the baseline"
        if abs(mu - ref.u) > max(crit.drive_tol * abs(ref.u), 4 * ref.u_sigma):
            return False, f"Drive {mu:.4g} vs {ref.u:.4g} at the baseline"
        return True, ""
    if abs(my) > crit.phase_tol_deg:
        return False, f"Phase {my:+.2f} deg"
    if abs(mu - ref.u) > max(crit.df_tol_hz, 4 * ref.u_sigma):
        return False, f"df {mu:+.3f} Hz vs {ref.u:+.3f} Hz at the baseline"
    return True, ""


def settled_at(loop: str, y, u, expected_y: Optional[float], crit: RecoveryCriteria = RecoveryCriteria()) -> Tuple[bool, str]:
    """Steady under the candidate gains, and (amplitude loop) at the amplitude the first level should give."""
    ok, why = steady(loop, y, u, crit)
    if not ok or loop != "afl" or not expected_y:
        return ok, why
    my = float(np.median(y))
    if abs(my - expected_y) > 2.5 * crit.amp_tol * abs(expected_y):
        return False, f"amplitude {(my - expected_y) / expected_y * 100:+.1f} % from where this level should settle"
    return True, ""


# ---------------------------------------------------------------------------
# the search region
# ---------------------------------------------------------------------------
def decade_values(center: float, span_decades: int, per_decade: int = 1) -> List[float]:
    n = int(span_decades) * int(per_decade)
    return [center * 10.0 ** (k / per_decade) for k in range(-n, n + 1)]


@dataclass(frozen=True)
class SearchRegion:
    """Kp x Ki grid in decades around a centre (by default: the gains set in SXM)."""

    kp_center: float
    ki_center: float
    span_decades: int = 2
    per_decade: int = 1
    include_kp0: bool = True              # integral-only column

    def kps(self) -> List[float]:
        return ([0.0] if self.include_kp0 else []) + decade_values(self.kp_center, self.span_decades, self.per_decade)

    def kis(self) -> List[float]:
        return decade_values(self.ki_center, self.span_decades, self.per_decade)

    def points(self) -> List[Tuple[float, float]]:
        return [(kp, ki) for kp in self.kps() for ki in self.kis()]

    def log_distance(self, kp: float, ki: float) -> float:
        """Distance from the centre in decades (Kp = 0 counts as one decade below the lowest Kp)."""
        lk = (math.log10(abs(kp / self.kp_center)) if kp != 0
              else -(self.span_decades + 1.0))
        return math.hypot(lk, math.log10(abs(ki / self.ki_center)))


def _key(kp: float, ki: float) -> Tuple[float, float]:
    return (float(f"{kp:.6g}"), float(f"{ki:.6g}"))


def _more_aggressive(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """``a`` has at least the |Kp| and |Ki| of ``b`` (Kp = 0 is the least aggressive)."""
    return abs(a[0]) >= abs(b[0]) * (1 - 1e-9) and abs(a[1]) >= abs(b[1]) * (1 - 1e-9)


# ---------------------------------------------------------------------------
# results: aggregation and trade-off
# ---------------------------------------------------------------------------
@dataclass
class PointSummary:
    kp: float
    ki: float
    assessments: List[Assessment]

    @property
    def n(self) -> int:
        return len(self.assessments)

    @property
    def status(self) -> str:
        """Worst status seen: a point that failed once is not trusted."""
        return max((a.status for a in self.assessments), key=SEVERITY.__getitem__)

    @property
    def clean(self) -> bool:
        return self.status == "clean"

    @property
    def speed_s(self) -> float:
        return float(np.median([a.speed_s for a in self.assessments]))

    @property
    def noise(self) -> float:
        vals = [a.noise for a in self.assessments if not math.isnan(a.noise)]
        return float(np.median(vals)) if vals else math.nan

    @property
    def ratio(self) -> float:
        return math.inf if self.kp == 0 else self.ki / self.kp


def summarize(assessments: Sequence[Assessment]) -> Dict[Tuple[float, float], PointSummary]:
    out: Dict[Tuple[float, float], PointSummary] = {}
    for a in assessments:
        k = _key(a.kp, a.ki)
        out.setdefault(k, PointSummary(a.kp, a.ki, [])).assessments.append(a)
    return out


@dataclass
class TradeOff:
    front: List[PointSummary]             # clean points no other clean point beats in both speed and noise
    fastest: Optional[PointSummary]       # fastest clean point with a safety margin to the edge (see trade_off)
    quietest: Optional[PointSummary]      # lowest noise among those within ``speed_slack`` of the fastest
    at_edge: List[PointSummary] = field(default_factory=list)   # clean, but a failure lies within the margin above


def _scale(p: PointSummary) -> float:
    return abs(p.ki) if p.kp == 0 else abs(p.kp)


def near_edge(p: PointSummary, points: Dict[Tuple[float, float], PointSummary], margin: float) -> bool:
    """A not-clean point lies on the same Ki:Kp ratio less than ``margin`` x above ``p`` (the manual: back off from it)."""
    s = _scale(p)
    return any(not q.clean and _same_ratio(q.ratio, p.ratio) and s < _scale(q) < margin * s * (1 - 1e-6)
               for q in points.values())


def trade_off(points: Dict[Tuple[float, float], PointSummary], speed_slack: float = 1.5,
              margin: float = 1.5) -> TradeOff:
    """
    Rank the clean points by speed (slower direction) and noise. The recommendations ``fastest`` and ``quietest``
    exclude points within ``margin`` of a failure along their ratio: a point right at the edge of stability works
    in the test and fails on the first disturbance.
    """
    all_clean = [p for p in points.values() if p.clean and math.isfinite(p.speed_s)]
    at_edge = [p for p in all_clean if near_edge(p, points, margin)]
    clean = [p for p in all_clean if p not in at_edge]
    if not clean:
        return TradeOff([], None, None, at_edge)

    def noise(p):
        return math.inf if math.isnan(p.noise) else p.noise
    front = [p for p in clean
             if not any(q is not p and q.speed_s <= p.speed_s and noise(q) <= noise(p)
                        and (q.speed_s < p.speed_s or noise(q) < noise(p)) for q in clean)]
    front.sort(key=lambda p: p.speed_s)
    fastest = min(clean, key=lambda p: (p.speed_s, noise(p)))
    near = [p for p in clean if p.speed_s <= speed_slack * fastest.speed_s]
    quietest = min(near, key=lambda p: (noise(p), p.speed_s))
    return TradeOff(front, fastest, quietest, at_edge)


# ---------------------------------------------------------------------------
# the exploration plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Proposal:
    kp: float
    ki: float
    stage: str            # 'baseline' | 'grid' | 'edge' | 'ratio' | 'repeat' | 'manual'
    why: str


STAGE_LABEL = {"baseline": "baseline", "grid": "decade grid", "edge": "edge along g", "ratio": "ratio search",
               "repeat": "repeat", "manual": "manual"}


@dataclass(frozen=True)
class ExplorePlan:
    """How far each automatic stage goes."""

    edge_ratios: int = 2          # how many of the best ratios get their g-edge located
    edge_tol: float = 1.25        # bisect until the clean / not-clean bracket is within this factor
    edge_extend_max: float = 1e3  # if a ratio is clean to the end of the grid, look up to this factor beyond it
    margin: float = 1.5           # back off from an edge by this factor (manual: back off to 50-70 %)
    ratio_rounds: int = 2         # pattern search on log(Ki/Kp): +-0.5 decade, then +-0.25
    repeats: int = 2              # extra measurements of the final candidates

    def estimate(self, region: SearchRegion) -> int:
        """Rough number of conditions a full exploration takes (the grid usually skips some)."""
        edge = self.edge_ratios * (2 + math.ceil(math.log(10.0 / region.per_decade, self.edge_tol) / 2))
        return 1 + len(region.points()) + edge + 2 * self.ratio_rounds + 2 * self.repeats


class Explorer:
    """
    The automatic exploration as a sequence of proposals: call :meth:`next` for the next condition to measure,
    measure it, :meth:`record` its assessment, repeat until :meth:`next` returns None. :meth:`add_manual` records
    a point measured by hand (it then counts like any other).
    """

    def __init__(self, region: SearchRegion, plan: ExplorePlan = ExplorePlan()):
        self.region = region
        self.plan = plan
        self.assessments: List[Assessment] = []
        self.skipped: Dict[Tuple[float, float], str] = {}
        self.stage = "baseline"
        self._pending: Optional[Proposal] = None
        self._last: Optional[Assessment] = None
        self._gen = self._run()
        self.done = False

    # -- driving ----------------------------------------------------------------------------------
    def next(self) -> Optional[Proposal]:
        if self.done:
            return None
        if self._pending is not None:
            raise RuntimeError("record() the pending proposal first")
        try:
            p = next(self._gen)
        except StopIteration:
            self.done = True
            self.stage = "done"
            return None
        self._pending = p
        self.stage = p.stage
        return p

    def record(self, a: Assessment) -> None:
        if self._pending is None:
            raise RuntimeError("nothing pending")
        self._pending = None
        self._last = a
        self.assessments.append(a)

    def skip(self, reason: str) -> None:
        """Decline the pending proposal (e.g. beyond the safety limit). The plan treats it as not clean."""
        if self._pending is None:
            raise RuntimeError("nothing pending")
        p, self._pending = self._pending, None
        self.skipped[_key(p.kp, p.ki)] = reason
        lost = DirectionAssessment("lost", math.nan, math.nan, 0, [reason])
        self._last = Assessment(p.kp, p.ki, "", lost, lost, math.nan, reasons=[reason])

    def add_manual(self, a: Assessment) -> None:
        self.assessments.append(a)

    def points(self) -> Dict[Tuple[float, float], PointSummary]:
        return summarize(self.assessments)

    # -- the plan ---------------------------------------------------------------------------------
    def _tested(self, kp, ki) -> Optional[PointSummary]:
        return self.points().get(_key(kp, ki))

    def _test(self, kp, ki, stage, why) -> Iterator[Proposal]:
        """Yield a proposal unless the point was already measured; afterwards ``self._last`` holds its assessment."""
        known = self._tested(kp, ki)
        if known is not None:
            self._last = known.assessments[-1]
            return
        yield Proposal(kp, ki, stage, why)

    def _lost_points(self):
        return [(p.kp, p.ki) for p in self.points().values() if p.status == "lost"]

    def _run(self) -> Iterator[Proposal]:
        r, plan = self.region, self.plan
        yield from self._test(r.kp_center, r.ki_center, "baseline", "the centre of the search (the gains in SXM)")

        # 1. coarse grid, nearest first; never go deeper into a region where the loop was lost
        for kp, ki in sorted(r.points(), key=lambda p: (r.log_distance(*p), abs(p[0]) + abs(p[1]))):
            k = _key(kp, ki)
            lost = next((l for l in self._lost_points() if _more_aggressive((kp, ki), l)), None)
            if lost is not None:
                self.skipped[k] = f"more aggressive than Kp={lost[0]:.3g}, Ki={lost[1]:.3g}, where the loop was lost"
                continue
            yield from self._test(kp, ki, "grid", "coarse grid in decades")

        # 2. edge along g for the best ratios: bisect between the last clean and the first not-clean scale
        for ratio in self._best_ratios(plan.edge_ratios):
            yield from self._edge(ratio)

        # 3. ratio at the best point: pattern search on log10(Ki/Kp) at fixed Kp (or Ki for Kp = 0)
        yield from self._ratio_search()

        # 4. repeat the candidates
        to = trade_off(self.points(), margin=plan.margin)
        for cand in {id(c): c for c in (to.fastest, to.quietest) if c is not None}.values():
            for _ in range(plan.repeats):
                yield Proposal(cand.kp, cand.ki, "repeat", "re-measure a candidate: one test is noisy")

    def _best_ratios(self, n: int) -> List[float]:
        """Distinct Ki/Kp ratios of the fastest clean points (inf = integral-only), best first."""
        clean = sorted((p for p in self.points().values() if p.clean and math.isfinite(p.speed_s)),
                       key=lambda p: p.speed_s)
        out: List[float] = []
        for p in clean:
            if not any(_same_ratio(p.ratio, q) for q in out):
                out.append(p.ratio)
            if len(out) == n:
                break
        return out

    def _ray(self, ratio: float, s: float) -> Tuple[float, float]:
        """The point at scale ``s`` along a ratio: s is Kp (finite ratio) or Ki (integral-only)."""
        return (0.0, s) if math.isinf(ratio) else (s, s * ratio)

    def _edge(self, ratio: float) -> Iterator[Proposal]:
        plan = self.plan
        on_ray = [p for p in self.points().values() if _same_ratio(p.ratio, ratio)]
        scale = (lambda p: abs(p.ki)) if math.isinf(ratio) else (lambda p: abs(p.kp))
        sign = math.copysign(1.0, (on_ray[0].ki if math.isinf(ratio) else on_ray[0].kp))
        good = [scale(p) for p in on_ray if p.clean]
        if not good:
            return
        lo = max(good)
        bad = [scale(p) for p in on_ray if not p.clean and scale(p) > lo]
        hi = min(bad) if bad else None
        label = "integral-only" if math.isinf(ratio) else f"Ki/Kp = {ratio:.3g}"
        while hi is None and lo * 10 <= plan.edge_extend_max * max(good):
            s = lo * 10
            yield from self._test(*self._ray(ratio, sign * s), "edge", f"{label}: clean so far, look a decade higher")
            if self._last.clean:
                lo = s
            else:
                hi = s
        if hi is None:
            return
        while hi / lo > plan.edge_tol:
            s = math.sqrt(lo * hi)
            yield from self._test(*self._ray(ratio, sign * s), "edge",
                                  f"{label}: bisect g between {lo:.3g} (clean) and {hi:.3g}")
            if self._last.clean:
                lo = s
            else:
                hi = s
        # the edge itself works in a test and fails on the first disturbance: measure the backed-off point
        safe = hi / plan.margin
        if safe < lo * (1 - 1e-9):
            yield from self._test(*self._ray(ratio, sign * safe), "edge",
                                  f"{label}: back off from the edge ({hi:.3g}) by x{plan.margin:g}")

    def _ratio_search(self) -> Iterator[Proposal]:
        to = trade_off(self.points(), margin=self.plan.margin)
        if to.fastest is None:
            return
        best = to.fastest
        step = 0.5
        for _ in range(self.plan.ratio_rounds):
            improved = None
            for d in (-step, step):
                if best.kp == 0:
                    kp, ki = 0.0, best.ki * 10 ** d
                else:
                    kp, ki = best.kp, best.ki * 10 ** d
                yield from self._test(kp, ki, "ratio", f"Ki x10^{d:+g} at Kp = {kp:.3g}: shape at the best speed")
                cand = self._tested(kp, ki)
                if (cand is not None and cand.clean and cand.speed_s < (improved or best).speed_s
                        and not near_edge(cand, self.points(), self.plan.margin)):
                    improved = cand
            if improved is not None:
                best = improved
            step /= 2


def _same_ratio(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return math.isinf(a) and math.isinf(b)
    return math.isclose(a, b, rel_tol=1e-3)


def along_ratio(points: Dict[Tuple[float, float], PointSummary], ratio: float) -> List[PointSummary]:
    """Points with this Ki/Kp, lowest scale first: the 'role of g' view."""
    on = [p for p in points.values() if _same_ratio(p.ratio, ratio)]
    return sorted(on, key=lambda p: abs(p.ki) if math.isinf(ratio) else abs(p.kp))


def along_kp(points: Dict[Tuple[float, float], PointSummary], kp: float) -> List[PointSummary]:
    """Points with this Kp, lowest Ki first: the 'role of the ratio' view."""
    on = [p for p in points.values() if math.isclose(p.kp, kp, rel_tol=1e-6, abs_tol=0.0) or (kp == 0 and p.kp == 0)]
    return sorted(on, key=lambda p: abs(p.ki))
