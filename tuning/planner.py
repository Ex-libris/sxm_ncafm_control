"""
Scan-speed requirements, trial scoring and the guided gain search.

The search never assumes what a raw SXM gain means: it starts from the user's
known-good baseline, tries multiples of it, and judges every candidate only on
what the loop actually did (see :mod:`tuning.metrics`).

Workflow (mirrors the manual: "find a good Ki:Kp ratio, then scale both")
--------------------------------------------------------------------------
1. ``baseline``  measure the current gains.
2. ``ratio``     at fixed Kp, try other Ki values; keep the best-shaped response.
3. ``scale``     keep that ratio and scale both gains: down while the scan is
                 still resolved (less noise), up if it is not.
4. ``refine``    bisect (geometrically) to the lowest-noise gains that still
                 meet every constraint.
5. ``verify``    repeat the winner once with fresh noise.

Everything is exposed as a generator (:meth:`GuidedTuner.proposals`): the caller
runs each proposed trial however it likes - after the user approved it - and
hands the result back with :meth:`GuidedTuner.record`. That keeps the search
usable from a GUI-thread state machine as well as from a plain loop.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import metrics as M
from .trial import TrialResult


# ---------------------------------------------------------------------------
# what the scan needs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanSpec:
    """
    What the loop has to deliver for the intended imaging speed.

    The defaults encode simple, adjustable rules of thumb:

    * a step must rise (10-90 %) within ``rise_fraction`` of one pixel dwell,
      so neighbouring pixels are not smeared into each other;
    * it must settle to within 5 % inside ``settle_fraction`` dwells;
    * df overshoot stays below ``overshoot_max`` and ringing is limited;
    * the phase error (whose tail is the integral term's residual) must have
      decayed within ``decay_fraction`` of a line time.
    """

    t_line: float = 6.0             # seconds per scan line
    n_px: int = 256                 # pixels per line
    rise_fraction: float = 0.5
    settle_fraction: float = 1.0
    overshoot_max: float = 0.10
    max_extrema: int = 3            # overshoot + undershoot + one more; beyond = ringing
    decay_fraction: float = 0.10
    noise_metric: str = "full"      # "full" = df RMS, "pixel" = RMS of pixel-averaged df

    @property
    def t_px(self) -> float:
        return self.t_line / self.n_px

    @property
    def rise_max(self) -> float:
        return self.rise_fraction * self.t_px

    @property
    def settle_max(self) -> float:
        return self.settle_fraction * self.t_px

    @property
    def decay_max(self) -> float:
        return self.decay_fraction * self.t_line

    @property
    def required_bandwidth_hz(self) -> float:
        """-3 dB bandwidth of a first-order response whose 10-90 % rise is ``rise_max``."""
        return 0.35 / self.rise_max


# ---------------------------------------------------------------------------
# scoring one trial
# ---------------------------------------------------------------------------
@dataclass
class TrialAnalysis:
    kp: float
    ki: float
    failure: Optional[str] = None            # "lost lock" / "unmeasurable" / "skipped" ...
    df: Optional[M.StepMetrics] = None
    phase: Optional[M.ErrorMetrics] = None
    noise_rms: float = math.nan              # df RMS on the settled tails
    pixel_noise: float = math.nan            # RMS of df averaged over one pixel dwell
    violations: Dict[str, float] = field(default_factory=dict)   # > 0 means the constraint is broken

    @property
    def measurable(self) -> bool:
        return self.failure is None

    @property
    def feasible(self) -> bool:
        return self.measurable and all(v <= 0.0 for v in self.violations.values())

    @property
    def penalty(self) -> float:
        """Worst constraint violation (inf for failed trials); <= 0 when feasible."""
        if not self.measurable:
            return math.inf
        return max(self.violations.values()) if self.violations else 0.0

    def shape_penalty(self) -> float:
        """Violation of the *shape* constraints only (overshoot, ringing, phase decay), used to pick Ki:Kp."""
        if not self.measurable:
            return math.inf
        keys = ("overshoot", "ringing", "phase_decay")
        return max(self.violations.get(k, 0.0) for k in keys)

    def speed_penalty(self) -> float:
        """Violation of the speed constraints only (rise, settling)."""
        if not self.measurable:
            return math.inf
        return max(self.violations.get(k, 0.0) for k in ("rise", "settle"))

    def noise(self, spec: ScanSpec) -> float:
        return self.pixel_noise if spec.noise_metric == "pixel" else self.noise_rms

    def describe_violations(self, spec: ScanSpec) -> List[str]:
        """Human-readable list of broken constraints with the measured value and the limit."""
        out = []
        if not self.measurable:
            return [self.failure or "failed"]
        d, ph = self.df, self.phase
        facts = {
            "rise": (f"rise {d.rise_time * 1e3:.1f} ms (needs <= {spec.rise_max * 1e3:.1f} ms)"),
            "settle": (f"5 % settling {'never' if math.isinf(d.settling_time) else '%.0f ms' % (d.settling_time * 1e3)} "
                       f"(needs <= {spec.settle_max * 1e3:.1f} ms)"),
            "overshoot": f"overshoot {d.overshoot * 100:.0f} % (needs <= {spec.overshoot_max * 100:.0f} %)",
            "ringing": f"{d.n_extrema} ringing extrema (needs <= {spec.max_extrema})",
        }
        if ph is not None:
            facts["phase_decay"] = (f"phase decays in {'never' if math.isinf(ph.decay_time) else '%.0f ms' % (ph.decay_time * 1e3)} "
                                    f"(needs <= {spec.decay_max * 1e3:.0f} ms)")
            facts["phase_ringing"] = f"phase undershoots {ph.sign_changes}x"
        for k, v in self.violations.items():
            if v > 0 and k in facts:
                out.append(facts[k])
        return out


def _rel(value, limit):
    """Normalised violation: value/limit - 1 (inf-safe)."""
    if math.isnan(value):
        return math.inf
    if math.isinf(value):
        return math.inf
    return value / limit - 1.0


def _settled_tails(result: TrialResult, tail_frac=0.4):
    """Residuals of the settled last part of every hold, one array per hold."""
    p = result.protocol
    out = []
    for k in range(p.n_holds):
        t0 = p.hold_s * (k + 1 - tail_frac)
        t1 = p.hold_s * (k + 1)
        m = (result.t >= t0) & (result.t < t1)
        if m.sum() >= 20:
            out.append((result.t[m], result.df[m] - np.median(result.df[m])))
    return out


def analyze_trial(result: TrialResult, spec: ScanSpec) -> TrialAnalysis:
    """Measure one trial and check it against the scan's constraints."""
    a = TrialAnalysis(kp=result.kp, ki=result.ki)
    if not result.locked:
        a.failure = "lost lock"
        return a

    p = result.protocol
    dt = float(np.median(np.diff(result.t)))
    pre_s, post_s = min(0.5 * p.hold_s, 1.0), p.hold_s - 0.05
    signs = np.sign(p.step_sizes)
    try:
        grid, df_mean, _, _ = M.average_steps(result.t, result.df, p.step_times, pre_s, post_s, dt, signs=signs)
        a.df = M.step_response_metrics(grid, df_mean, 0.0, target_step=2.0 * p.step_hz)
    except M.StepNotDetectable:
        a.failure = "unmeasurable"        # ringing / runaway: no clean step to measure
        return a

    try:
        grid, ph_mean, _, _ = M.average_steps(result.t, result.phase, p.step_times, pre_s, post_s, dt, signs=signs)
        a.phase = M.error_transient_metrics(grid, ph_mean, 0.0)
    except M.StepNotDetectable:
        a.phase = None                    # no visible phase transient: nothing to decay

    tails = _settled_tails(result)
    if tails:
        a.noise_rms = float(math.sqrt(np.mean(np.concatenate([r ** 2 for _, r in tails]))))
        blocks = []
        for tt, r in tails:
            if tt[-1] - tt[0] >= 8 * spec.t_px:
                _, means = M.block_mean(tt, r, spec.t_px)
                blocks.append(means)
        if blocks:
            a.pixel_noise = float(np.std(np.concatenate(blocks), ddof=1))

    d = a.df
    a.violations = {
        "overshoot": d.overshoot / spec.overshoot_max - 1.0 if d.overshoot > 0 else -1.0,
        "ringing": d.n_extrema / spec.max_extrema - 1.0 if d.n_extrema > 0 else -1.0,
        "rise": _rel(d.rise_time, spec.rise_max),
        "settle": _rel(d.settling_time, spec.settle_max),
    }
    if a.phase is not None:
        a.violations["phase_decay"] = _rel(a.phase.decay_time, spec.decay_max)
        a.violations["phase_ringing"] = a.phase.sign_changes / 2.0 - 1.0 if a.phase.sign_changes > 0 else -1.0
    return a


# ---------------------------------------------------------------------------
# the guided search
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Limits:
    """Hard bounds on what may be proposed, as multiples of the baseline magnitudes."""

    min_factor: float = 0.05
    max_factor: float = 8.0
    max_trials: int = 16


@dataclass(frozen=True)
class Proposal:
    kp: float
    ki: float
    stage: str
    reason: str


@dataclass
class Recommendation:
    kp: float
    ki: float
    analysis: TrialAnalysis
    meets_all_constraints: bool
    verified: bool
    rationale: str


class GuidedTuner:
    """
    Guided search for one loop's (Kp, Ki).

    Parameters
    ----------
    baseline_kp, baseline_ki : float
        The current, known-good raw gains (both negative for the PLL). They are
        the only thing the search assumes to be safe.
    spec : ScanSpec
    limits : Limits
    ratio_factors : Ki multipliers tried at fixed Kp in the ``ratio`` stage.
    """

    RATIO_FACTORS = (0.3, 3.0, 0.1, 5.0)
    DOWN_STEPS = (0.7, 0.5, 0.35, 0.25)
    UP_STEPS = (1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
    N_BISECT = 3

    def __init__(self, baseline_kp: float, baseline_ki: float, spec: ScanSpec = ScanSpec(),
                 limits: Limits = Limits()):
        if baseline_kp == 0 or baseline_ki == 0:
            raise ValueError("baseline gains must be non-zero")
        self.kp0 = float(baseline_kp)
        self.ki0 = float(baseline_ki)
        self.spec = spec
        self.limits = limits
        self.history: List[TrialAnalysis] = []
        self.proposals_made: List[Proposal] = []
        self._pending: Optional[Proposal] = None
        self.ratio: Optional[float] = None       # chosen Ki/Kp
        self.verified: bool = False

    # ------------------------------------------------------------ bookkeeping
    def record(self, analysis: TrialAnalysis) -> None:
        """Hand back the analysed result of the last proposal."""
        if self._pending is None:
            raise RuntimeError("no proposal is waiting for a result")
        self.history.append(analysis)
        self._pending = None

    def skip(self, reason: str = "skipped by user") -> None:
        """The user declined the last proposal: treat it as an unusable point and carry on."""
        if self._pending is None:
            raise RuntimeError("no proposal is waiting for a result")
        p = self._pending
        self.history.append(TrialAnalysis(kp=p.kp, ki=p.ki, failure=reason))
        self._pending = None

    def _within_limits(self, kp, ki) -> bool:
        lo, hi = self.limits.min_factor, self.limits.max_factor
        fk, fi = abs(kp / self.kp0), abs(ki / self.ki0)
        return lo <= fk <= hi and lo <= fi <= hi

    def _seen(self, kp, ki) -> bool:
        return any(math.isclose(h.kp, kp, rel_tol=1e-3) and math.isclose(h.ki, ki, rel_tol=1e-3)
                   for h in self.history)

    def _trial(self, kp, ki, stage, reason):
        """Generator helper: yield a proposal (if allowed and new), return its analysis or None."""
        if not self._within_limits(kp, ki) or len(self.history) >= self.limits.max_trials:
            return None
        for h in self.history:                  # an identical point was already measured
            if math.isclose(h.kp, kp, rel_tol=1e-3) and math.isclose(h.ki, ki, rel_tol=1e-3):
                return h
        prop = Proposal(kp=kp, ki=ki, stage=stage, reason=reason)
        self._pending = prop
        self.proposals_made.append(prop)
        yield prop
        if self._pending is not None:
            raise RuntimeError("record() or skip() must be called before asking for the next proposal")
        return self.history[-1]

    # ------------------------------------------------------------ the search
    def proposals(self):
        """Generator of :class:`Proposal`; call ``record``/``skip`` after each one."""
        spec = self.spec
        # 1. baseline -------------------------------------------------------------
        base = yield from self._trial(self.kp0, self.ki0, "baseline",
                                      "Measure your current gains as the reference.")
        # 2. ratio ------------------------------------------------------------------
        ratio_pool = [base] if base is not None else []
        for f in self.RATIO_FACTORS:
            a = yield from self._trial(self.kp0, self.ki0 * f, "ratio",
                                       f"Same Kp, Ki x{f:g}: how does the integral term shape the response?")
            if a is not None:
                ratio_pool.append(a)
        measurable = [a for a in ratio_pool if a.measurable]
        if measurable:
            # best-shaped response; ties -> smaller |Ki| (calmer, quieter)
            best_shape = min(measurable, key=lambda a: (round(a.shape_penalty(), 3), abs(a.ki)))
            self.ratio = best_shape.ki / best_shape.kp
        else:
            self.ratio = self.ki0 / self.kp0

        def at_scale(m):
            kp = self.kp0 * m
            return kp, self.ratio * kp

        # 3. scale ------------------------------------------------------------------
        known: Dict[float, TrialAnalysis] = {}
        a1 = yield from self._trial(*at_scale(1.0), "scale", "Best Ki:Kp ratio at the baseline Kp.")
        if a1 is not None:
            known[1.0] = a1
        base_ok = a1 is not None and a1.feasible
        steps = self.DOWN_STEPS if base_ok else self.UP_STEPS
        for m in steps:
            why = ("Lower gains: less noise, if the scan is still resolved." if base_ok
                   else "Higher gains: the response is not fast/clean enough for this scan yet.")
            a = yield from self._trial(*at_scale(m), "scale", f"Both gains x{m:g} along the chosen ratio. {why}")
            if a is None:
                break
            known[m] = a
            if base_ok and not a.feasible:
                break                           # found the lower edge
            if not base_ok and a.feasible:
                break                           # found something that works
            if not a.measurable:
                break

        # 4. refine (geometric bisection between infeasible and feasible scales) ---------
        for _ in range(self.N_BISECT):
            feas = [m for m, a in known.items() if a.feasible]
            infeas = [m for m, a in known.items() if not a.feasible]
            if not feas or not infeas:
                break
            m_hi = min(feas)                    # cheapest known feasible scale
            lows = [m for m in infeas if m < m_hi]
            if not lows:
                break
            m_lo = max(lows)
            if m_hi / m_lo < 1.08:
                break
            m_mid = math.sqrt(m_lo * m_hi)
            a = yield from self._trial(*at_scale(m_mid), "refine",
                                       f"Bisect between x{m_lo:.3g} (fails) and x{m_hi:.3g} (works) for the lowest-noise gains.")
            if a is None:
                break
            known[m_mid] = a

        # 5. verify -----------------------------------------------------------------
        best = self.best()
        if best is not None and best.feasible:
            # measure the winner again with fresh noise; a new object even if the point is identical
            prop = Proposal(kp=best.kp, ki=best.ki, stage="verify",
                            reason="Repeat the recommended gains once more to confirm.")
            if len(self.history) < self.limits.max_trials + 1:
                self._pending = prop
                self.proposals_made.append(prop)
                yield prop
                if self._pending is not None:
                    raise RuntimeError("record() or skip() must be called before asking for the next proposal")
                again = self.history[-1]
                self.verified = again.feasible

    # ------------------------------------------------------------ results
    def best(self) -> Optional[TrialAnalysis]:
        """Lowest-noise feasible trial; else the one closest to feasible; None if nothing was measurable."""
        ok = [a for a in self.history if a.feasible and not math.isnan(a.noise(self.spec))]
        if ok:
            return min(ok, key=lambda a: a.noise(self.spec))
        meas = [a for a in self.history if a.measurable]
        if not meas:
            return None
        # Nothing meets every constraint: prefer a clean (no overshoot/ringing/slow tail) response
        # that is as fast as possible; if none is clean, the least badly shaped one.
        clean = [a for a in meas if a.shape_penalty() <= 0.0]
        if clean:
            return min(clean, key=lambda a: (a.speed_penalty(), abs(a.kp)))
        return min(meas, key=lambda a: (a.shape_penalty(), a.speed_penalty()))

    def recommendation(self) -> Optional[Recommendation]:
        b = self.best()
        if b is None:
            return None
        if b.feasible:
            why = (f"Lowest-noise gains that resolve a {self.spec.t_line:g} s / {self.spec.n_px} px scan "
                   f"(pixel {self.spec.t_px * 1e3:.1f} ms): rise {b.df.rise_time * 1e3:.1f} ms, "
                   f"overshoot {b.df.overshoot * 100:.0f} %, df noise {b.noise_rms * 1e3:.1f} mHz.")
        else:
            broken = b.describe_violations(self.spec)
            keys = {k for k, v in b.violations.items() if v > 0}
            advice = []
            if keys & {"rise", "settle"}:
                advice.append("the scan is faster than this loop can follow cleanly: scan slower or shorten the "
                              "lock-in TimeConstant (which costs noise)")
            if keys & {"overshoot", "ringing", "phase_ringing"}:
                advice.append("the response rings: lower the gains or the Ki:Kp ratio")
            if "phase_decay" in keys:
                advice.append("the phase tail is slow: a larger Ki:Kp ratio would help")
            why = ("No tested gains meet every constraint. Best clean response: " + "; ".join(broken) + ". "
                   + (("; ".join(advice))[0].upper() + ("; ".join(advice))[1:] + "." if advice else ""))
        return Recommendation(kp=b.kp, ki=b.ki, analysis=b, meets_all_constraints=b.feasible,
                              verified=self.verified, rationale=why)

    def pareto(self) -> List[TrialAnalysis]:
        """Feasible-or-not measurable trials not beaten in *both* rise time and noise."""
        pts = [a for a in self.history if a.measurable and not math.isnan(a.noise(self.spec))
               and not math.isnan(a.df.rise_time)]
        front = []
        for a in pts:
            if not any((b.df.rise_time <= a.df.rise_time and b.noise(self.spec) <= a.noise(self.spec)) and
                       (b.df.rise_time < a.df.rise_time or b.noise(self.spec) < a.noise(self.spec))
                       for b in pts):
                front.append(a)
        return sorted(front, key=lambda a: a.df.rise_time)

    def report(self) -> str:
        """Plain-text table of every trial and the recommendation."""
        s = self.spec
        lines = [
            f"Scan: {s.t_line:g} s/line x {s.n_px} px -> pixel dwell {s.t_px * 1e3:.1f} ms; "
            f"need rise <= {s.rise_max * 1e3:.1f} ms (bandwidth ~{s.required_bandwidth_hz:.0f} Hz), "
            f"settle <= {s.settle_max * 1e3:.1f} ms, overshoot <= {s.overshoot_max * 100:.0f} %, "
            f"phase decay <= {s.decay_max * 1e3:.0f} ms.",
            "",
            f"{'#':>2} {'stage':<8} {'Kp':>8} {'Ki':>9} {'rise ms':>8} {'os %':>5} {'ph.decay ms':>11} "
            f"{'noise mHz':>9}  result",
        ]
        for i, a in enumerate(self.history):
            stage = self.proposals_made[i].stage if i < len(self.proposals_made) else "?"
            if not a.measurable:
                lines.append(f"{i + 1:>2} {stage:<8} {a.kp:>8.4g} {a.ki:>9.4g} {'-':>8} {'-':>5} {'-':>11} {'-':>9}  {a.failure}")
                continue
            dec = "-" if a.phase is None else ("inf" if math.isinf(a.phase.decay_time) else f"{a.phase.decay_time * 1e3:.0f}")
            bad = [k for k, v in a.violations.items() if v > 0]
            lines.append(f"{i + 1:>2} {stage:<8} {a.kp:>8.4g} {a.ki:>9.4g} {a.df.rise_time * 1e3:>8.1f} "
                         f"{a.df.overshoot * 100:>5.1f} {dec:>11} {a.noise_rms * 1e3:>9.2f}  "
                         f"{'OK' if a.feasible else 'fails: ' + ', '.join(bad)}")
        rec = self.recommendation()
        lines.append("")
        if rec is None:
            lines.append("No measurable trial.")
        else:
            tag = "verified" if rec.verified else "not re-verified"
            lines.append(f"Recommended: Kp = {rec.kp:.4g}, Ki = {rec.ki:.4g}  ({tag})")
            lines.append(rec.rationale)
        return "\n".join(lines)


def run_guided(tuner: GuidedTuner, backend, approve=None) -> GuidedTuner:
    """
    Drive a tuner with a backend in a plain loop (tests, offline demos).

    ``approve(proposal) -> bool`` plays the guided user; declined proposals are skipped.
    """
    for prop in tuner.proposals():
        if approve is not None and not approve(prop):
            tuner.skip()
            continue
        result = backend.run_trial(prop.kp, prop.ki)
        tuner.record(analyze_trial(result, tuner.spec))
    return tuner
