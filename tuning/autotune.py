"""
Automatic search for the amplitude-loop (AFL) gains, meant to run unattended from the Tuning tab.

The search is a chain of run-sheet stages; each stage is an ordinary list of conditions (tuning/runsheet.py) run by
gui/sheet_runner.py, and the next stage is decided from the results of the one before:

1. **coarse**   G ramp at rho = 1 in 1-3-10 steps, from the gentlest point up to ``top`` x the manual rule.
   If even its first point is not clean (too aggressive), once: **lower**, the same ramp at 0.01 ... 0.3 of it.
   If Drive never moved, the amplitude loop is off: the search stops and says so.
2. **shape**    rho ramp (Ki:Kp) at the fastest clean G of stage 1.
3. **edge**     G ramp at the best rho in x``edge_factor`` steps, from that clean G to just past stage 1's edge.
4. **tau**      (optional) AFL Tau ramp at the chosen point; needs the settings writer to run unattended.
5. **validate** the chosen point ``repeats`` times (+ Ref x0.5 / x2, reported only). If a repeat is not clean, one
   **fallback** validation at G / ``edge_factor``.

The chosen G is ``min(fastest clean G, edge / margin)``: half the speed at which the loop first stops being clean.

Safety: the anchor (G = 1) is the manual rule x ``start_factor`` (default 1/100), i.e. the *gentlest* point of the
search, not a "known good" fast one. G only ever grows from there, the run-sheet stop rule ends a ramp at its first
ringing / Drive-at-zero / lost condition, and a lost condition recovers at this slow anchor.

Judged from QPlusAmpl only: the runner gets :data:`EXPLORE_LIMITS` (Drive at zero does not fail a condition), and
overshoot counts only beyond the gentlest point's (per direction, ``overshoot_excess``), because with the AFL on SXM's
Drive is not the whole excitation and the up-steps overshot by the same amount at every gain (9 Nov 2026).

"Clean", "slow", "overshoot", ... are tuning/explore.assess statuses. Gains are SXM's arbitrary units: nothing here
assumes what a raw gain means; the manual rule only sets where the search starts.

Pure Python; no Qt, no hardware.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import explore as X
from . import runsheet as R
from .workflow import afl_start_values

# statuses that mean "past the edge": the loop is no longer clean because it is too aggressive ('slow' is the
# opposite problem and never counts as an edge)
EDGE_STATUSES = ("overshoot", "floor", "ringing", "lost")
TAU_OPTIONS_MS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
# below a start that was already not clean: these fractions of it (gentlest first), tried once
LOWER_FACTORS = (0.01, 0.03, 0.1, 0.3)
# Drive rms below this in every quiet window = Drive never moved: the amplitude loop is not regulating
DRIVE_STILL_V = 1e-10
# The search judges the amplitude loop from QPlusAmpl only. With the AFL on, SXM's Drive is not the whole excitation
# (9 Nov 2026: it stayed flat while a Ref step moved the amplitude 10 %, and read ~8x below what the open-loop gain
# needs), so "Drive at zero" must not make a condition fail or stop a ramp: it stays in the notes, reported.
EXPLORE_LIMITS = X.ExploreLimits(floor_max=math.inf)
KIND = "ncafm_afl_autotune"


def round_sig(x: float, n: int = 3) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---------------------------------------------------------------------------
# configuration and timing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AutoTuneConfig:
    q: float
    f0: float
    output_gain_v: float = 0.1                # AFL output gain range the search runs at [+-V]
    input_gain: float = 1.0                   # InA gain the search runs at
    base_ref: float = 1.0                     # amplitude Ref the search runs at (SXM units)
    step_frac: float = 0.05                   # Ref step of the train, fraction of Ref (+-5 %, the tab's default)
    start_factor: float = 0.01                # anchor (G = 1) = manual rule x this
    top: float = 3.0                          # highest coarse G = top x the manual rule
    coarse_values: Optional[Tuple[float, ...]] = None   # explicit coarse G ladder instead of 1-3-10 (tests)
    rho_values: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    edge_factor: float = 1.5                  # G step of the edge ramp, and of the fallback
    edge_max_steps: int = 6                   # at most this many points in the edge ramp
    margin: float = 2.0                       # chosen G <= edge / margin
    include_tau: bool = False
    tau_values_ms: Optional[Tuple[float, ...]] = None   # explicit Tau ramp instead of the derived one
    repeats: int = 3
    ref_factors: Tuple[float, ...] = (0.5, 2.0)        # extra validation at Ref x these (reported, not judged)
    quiet_search_s: float = 10.0
    quiet_validate_s: float = 30.0
    tie: float = 0.10                         # within 10 % on the main criterion counts as a tie -> noise decides
    # Overshoot is judged against the gentlest measured point's, per direction: on 9 Nov 2026 the up-steps
    # overshot ~35-65 % at every gain over a 100x range, i.e. not because of Kp / Ki. Only an excess beyond that
    # baseline (fraction of the step) counts as overshoot.
    overshoot_excess: float = 0.20

    def __post_init__(self):
        for name in ("q", "f0", "output_gain_v", "input_gain", "base_ref", "top"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.step_frac < 0.5:
            raise ValueError("step_frac must be between 0 and 0.5")
        if not 0 < self.start_factor <= 1:
            raise ValueError("start_factor must be in (0, 1]: the search starts at or below the manual rule")
        if self.edge_factor <= 1 or self.margin <= 1:
            raise ValueError("edge_factor and margin must be above 1")
        if self.repeats < 1 or self.edge_max_steps < 1:
            raise ValueError("repeats and edge_max_steps must be at least 1")
        if not self.rho_values or any(v <= 0 for v in self.rho_values):
            raise ValueError("rho values must be positive")
        if self.coarse_values is not None and (not self.coarse_values or any(v <= 0 for v in self.coarse_values)):
            raise ValueError("coarse G values must be positive")


@dataclass(frozen=True)
class Timing:
    """Step-train timing from the sensor: the amplitude settles with the ring-down time Q / (pi f0)."""

    ring_down_s: float
    hold_s: float
    settle_s: float
    settle_timeout_s: float


def timing(cfg: AutoTuneConfig) -> Timing:
    s = afl_start_values(cfg.q, cfg.f0, cfg.output_gain_v)
    return Timing(ring_down_s=s.ring_down_s, hold_s=s.hold_s, settle_s=s.settle_s,
                  settle_timeout_s=min(120.0, max(15.0, 20.0 * s.ring_down_s)))


# ---------------------------------------------------------------------------
# one measured condition, as the planner sees it
# ---------------------------------------------------------------------------
@dataclass
class Measured:
    g: float
    rho: float
    settings: Dict[str, Any]
    role: str
    status: str                       # explore status, or 'skipped' / 'stopped' / 'pending'
    up_rise_s: float = math.nan
    down_rise_s: float = math.nan
    settle_s: float = math.nan        # step -> last exit from the settling band (all steps averaged)
    overshoot: float = math.nan       # worse direction, fraction of the step
    noise: float = math.nan           # Drive noise during the steps [V] (reported only)
    drive_rms: float = math.nan       # Drive rms in the quiet window [V] (AFL-off check only)
    note: str = ""
    up_status: str = ""               # explore status of each direction ('' = not known)
    down_status: str = ""
    up_overshoot: float = math.nan
    down_overshoot: float = math.nan
    amp_noise: float = math.nan       # QPlusAmpl rms / mean in the quiet window: the noise the search ranks by

    @classmethod
    def from_outcome(cls, cond: R.Condition, outcome: Any) -> "Measured":
        """From a gui.sheet_runner.Outcome (duck-typed, so this module needs no Qt); None = not run."""
        m = cls(cond.g, cond.rho, dict(cond.settings), cond.role, "pending")
        if outcome is None:
            return m
        m.status = outcome.status
        m.note = outcome.note or ""
        a, res, q = outcome.assessment, outcome.result, outcome.quiet
        if a is not None:
            m.up_rise_s, m.down_rise_s = a.up.rise_s, a.down.rise_s
            m.up_status, m.down_status = a.up.status, a.down.status
            m.up_overshoot, m.down_overshoot = a.up.overshoot, a.down.overshoot
            ov = [v for v in (a.up.overshoot, a.down.overshoot) if _finite(v)]
            m.overshoot = max(ov) if ov else math.nan
            m.noise = a.noise
        if res is not None and getattr(res, "primary", None) is not None:
            m.settle_s = res.primary.settling_time
        if q is not None:
            m.drive_rms = q.get("Drive", "rms")
            m.amp_noise = q.get("QPlusAmpl", "rel_rms")
        return m

    @property
    def clean(self) -> bool:
        return self.status == "clean"

    @property
    def edge(self) -> bool:
        return self.status in EDGE_STATUSES

    @property
    def speed_s(self) -> float:
        r = [self.up_rise_s, self.down_rise_s]
        return max(r) if all(_finite(x) for x in r) else math.inf

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# stages and the result
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    key: str                          # coarse | shape | edge | tau | validate | fallback
    title: str
    why: str
    conditions: List[R.Condition]
    quiet_s: float


@dataclass
class Recommendation:
    g: float
    rho: float
    tau_ms: Optional[float]
    kp: float
    ki: float
    confirmed: bool                   # every validation repeat was clean
    edge_g: Optional[float]           # first G at this rho that was no longer clean (None: not reached)
    note: str = ""

    @property
    def margin(self) -> Optional[float]:
        return self.edge_g / self.g if self.edge_g else None


class AutoTuner:
    """Decides the stages of one automatic search. Feed each stage's results to :meth:`next_stage`."""

    def __init__(self, cfg: AutoTuneConfig):
        self.cfg = cfg
        self.manual = afl_start_values(cfg.q, cfg.f0, cfg.output_gain_v)
        self.anchor = R.Anchor("afl", self.manual.kp * cfg.start_factor, self.manual.ki * cfg.start_factor,
                               cfg.output_gain_v, cfg.input_gain)
        self.timing = timing(cfg)
        self.stages: List[Stage] = []
        self.results: List[List[Measured]] = []
        self.decisions: List[str] = []
        self.finished = False
        self.failure: Optional[str] = None
        self.stopped: Optional[str] = None
        self.recommendation: Optional[Recommendation] = None
        self._g_good: Optional[float] = None
        self._edge1: Optional[float] = None
        self._shape_clean_at_g_good = False
        self._rho = 1.0
        self._g_final: Optional[float] = None
        self._edge2: Optional[float] = None
        self._tau: Optional[float] = None
        self.overshoot_base: Optional[Dict[str, float]] = None    # {'up': .., 'down': ..} of the gentlest point
        self._base_g = math.inf

    # -- helpers -------------------------------------------------------------------------------------
    @property
    def manual_g(self) -> float:
        """The G at which the gains equal the manual rule."""
        return 1.0 / self.cfg.start_factor

    def _note(self, text: str):
        self.decisions.append(text)

    def _group(self, title: str) -> str:
        return f"Auto {len(self.stages) + 1}: {title}"

    def _add(self, stage: Stage) -> Stage:
        self.stages.append(stage)
        return stage

    def _fail(self, why: str) -> None:
        self.failure = why
        self._note(f"No recommendation: {why}")
        return None

    def gains(self, g: float, rho: float) -> Tuple[float, float]:
        return R.gains(self.anchor, R.Condition(g=g, rho=rho), compensate=False)

    def coarse_values(self) -> List[float]:
        if self.cfg.coarse_values is not None:
            return sorted(float(v) for v in self.cfg.coarse_values)
        hi = self.cfg.top * self.manual_g
        vals = []
        for e in range(-1, int(math.ceil(math.log10(hi))) + 2):
            for m in (1.0, 3.0):
                v = m * 10.0 ** e
                if 1.0 - 1e-9 <= v <= hi * (1 + 1e-9):
                    vals.append(v)
        return vals

    def tau_candidates_ms(self) -> List[float]:
        if self.cfg.tau_values_ms is not None:
            return [float(v) for v in self.cfg.tau_values_ms]
        t = self.manual.tau_s * 1e3                      # manual: Tau = Q / (100 f0), the slow end
        vals = [v for v in TAU_OPTIONS_MS if t / 10.0 <= v <= 2.0 * t]
        return vals or [min(TAU_OPTIONS_MS, key=lambda v: abs(math.log(v / t)))]

    def _best(self, items: Sequence[Measured], main: Callable[[Measured], float]) -> Measured:
        """Lowest ``main`` (ties within ``tie`` broken by lower amplitude noise); by speed if ``main`` is never
        finite."""
        scored = [(main(m), m) for m in items if _finite(main(m))]
        if not scored:
            scored = [(m.speed_s, m) for m in items if _finite(m.speed_s)]
        if not scored:
            return items[0]
        lo = min(s for s, _ in scored)
        near = [m for s, m in scored if s <= lo * (1 + self.cfg.tie)]

        return min(near, key=self._amp_noise)

    @staticmethod
    def _amp_noise(m: Measured) -> float:
        """The amplitude noise (QPlusAmpl), the only noise the search ranks by."""
        return m.amp_noise if _finite(m.amp_noise) else math.inf

    # -- overshoot against the gentlest point ------------------------------------------------------------
    def _update_base(self, ms: Sequence[Measured]):
        """The gentlest measured point's overshoot, per direction, is the baseline (a lower G replaces it)."""
        run = [m for m in ms if m.role == "test" and (m.up_status or m.down_status)
               and m.status not in ("skipped", "pending", "stopped", "lost")
               and (_finite(m.up_overshoot) or _finite(m.down_overshoot))]
        if not run:
            return
        g0 = min(m.g for m in run)
        if g0 >= self._base_g:
            return
        m0 = next(m for m in run if m.g == g0)
        self._base_g = g0
        self.overshoot_base = {"up": m0.up_overshoot if _finite(m0.up_overshoot) else 0.0,
                               "down": m0.down_overshoot if _finite(m0.down_overshoot) else 0.0}
        self._note(f"Overshoot baseline (the gentlest point, G {g0:g}): up {self.overshoot_base['up'] * 100:.0f} %, "
                   f"down {self.overshoot_base['down'] * 100:.0f} %. Only overshoot more than "
                   f"{self.cfg.overshoot_excess * 100:.0f} % of the step above it counts.")

    def _rejudge(self, ms: Sequence[Measured]):
        """A direction marked 'overshoot' only because of the gain-independent baseline is clean."""
        base = self.overshoot_base
        if base is None:
            return
        for m in ms:
            if m.status in ("skipped", "pending", "stopped") or not (m.up_status and m.down_status):
                continue
            changed = False
            for d in ("up", "down"):
                ov = getattr(m, f"{d}_overshoot")
                if (getattr(m, f"{d}_status") == "overshoot" and _finite(ov)
                        and ov - base[d] <= self.cfg.overshoot_excess):
                    setattr(m, f"{d}_status", "clean")
                    changed = True
            if changed:
                new = max((m.up_status, m.down_status), key=X.SEVERITY.__getitem__)
                if X.SEVERITY[new] < X.SEVERITY.get(m.status, 0):
                    m.status = new
                    m.note = (m.note + " | " if m.note else "") + "overshoot within the baseline of the gentlest point"

    # -- the stages --------------------------------------------------------------------------------------
    def first_stage(self) -> Stage:
        if self.stages:
            raise RuntimeError("the search has already started")
        vals = self.coarse_values()
        self._note(f"Start: manual rule Kp {self.manual.kp:.3g}, Ki {self.manual.ki:.3g} (Q {self.cfg.q:.4g}, "
                   f"f0 {self.cfg.f0:.6g} Hz, ±{self.cfg.output_gain_v:g} V). Anchor G = 1 is that x "
                   f"{self.cfg.start_factor:g}: Kp {self.anchor.kp:.3g}, Ki {self.anchor.ki:.3g}. "
                   f"The manual rule is G {self.manual_g:g}.")
        title = "speed (G ramp at ρ 1)"
        return self._add(Stage(
            "coarse", title,
            f"From the gentlest point up to {self.cfg.top:g} x the manual rule (G {vals[0]:g} … {vals[-1]:g}); "
            "ends at the first ringing / Drive at zero / lost condition.",
            R.ramp("g", vals, rho=1.0, group=self._group(title)), self.cfg.quiet_search_s))

    def next_stage(self, measured: Sequence[Measured]) -> Optional[Stage]:
        """The results of the stage just run (its conditions, in order). Returns the next stage, or None at the end."""
        if self.finished:
            return None
        if not self.stages:
            raise RuntimeError("call first_stage() first")
        stage = self.stages[-1]
        measured = list(measured)
        if stage.key in ("coarse", "lower"):
            self._update_base(measured)
        self._rejudge(measured)
        self.results.append(measured)
        after = {"coarse": self._after_coarse, "lower": self._after_lower, "shape": self._after_shape,
                 "edge": self._after_edge, "tau": self._after_tau, "validate": self._after_validate,
                 "fallback": self._after_validate}
        nxt = after[stage.key](list(measured))
        if nxt is None:
            self.finished = True
            return None
        return self._add(nxt)

    def stop(self, reason: str, measured: Optional[Sequence[Measured]] = None):
        """The run ended before the search did (Stop, an abort). ``measured``: what the current stage got done."""
        if self.finished:
            return
        if measured is not None and len(self.results) < len(self.stages):
            self.results.append(list(measured))
        self.finished = True
        self.stopped = reason
        self._note(f"Search ended early ({reason}).")

    @staticmethod
    def _afl_off(run: Sequence[Measured]) -> bool:
        """Drive did not move at all in any quiet window: the loop is not regulating (AFL switched off)."""
        q = [m.drive_rms for m in run if _finite(m.drive_rms)]
        return bool(q) and max(q) < DRIVE_STILL_V

    def _after_coarse(self, ms: List[Measured]) -> Optional[Stage]:
        return self._speed(ms, edge_below=None)

    def _after_lower(self, ms: List[Measured]) -> Optional[Stage]:
        return self._speed(ms, edge_below=self._edge1)

    def _speed(self, ms: List[Measured], edge_below: Optional[float]) -> Optional[Stage]:
        """Results of a G ramp at rho 1 (the coarse one, or the lower one after it); ``edge_below``: an edge
        already found above this ramp."""
        tests = [m for m in ms if m.role == "test"]
        run = [m for m in tests if m.status not in ("skipped", "pending", "stopped")]
        if self._afl_off(run):
            return self._fail("Drive did not move at all: the amplitude feedback looks switched OFF (QPlusAmpl "
                              "is then not held at Ref). Switch it on in SXM and run again.")
        found = [m.g for m in tests if m.edge]
        edge = min(found + ([edge_below] if edge_below is not None else []), default=None)
        clean = [m.g for m in tests if m.clean and (edge is None or m.g < edge)]
        if not clean:
            if edge is not None and run and edge <= min(m.g for m in run):
                status = next((m.status for m in tests if m.g == edge), "not clean")
                if self.stages[-1].key == "coarse":
                    return self._lower_stage(edge, status)
                return self._fail(f"even the gentlest point (G {min(m.g for m in run):g} = manual rule x "
                                  f"{min(m.g for m in run) * self.cfg.start_factor:g}) was not clean ({status}).")
            if run and all(m.status == "slow" for m in run):
                return self._fail(f"every point up to G {max(m.g for m in run):g} was too slow for the hold: check "
                                  "Q and f0, or let the search go higher.")
            return self._fail("no clean condition in the speed ramp.")
        self._g_good, self._edge1 = max(clean), edge
        self._note(f"Speed: clean up to G {self._g_good:g}" +
                   (f"; first not clean at G {edge:g}." if edge is not None else "; no edge reached."))
        title = f"shape (ρ ramp at G {self._g_good:g})"
        return Stage("shape", title,
                     "Changes Ki against Kp at the fastest clean speed: low ρ = slow tail, high ρ = overshoot.",
                     R.ramp("rho", self.cfg.rho_values, g=self._g_good, group=self._group(title)),
                     self.cfg.quiet_search_s)

    def _lower_stage(self, edge: float, status: str) -> Stage:
        """The gentlest point was already too aggressive: once, a G ramp below it (same anchor, so a lost
        condition still recovers at G 1, which was measured stable if not clean)."""
        self._edge1 = edge
        vals = [round_sig(edge * f) for f in LOWER_FACTORS]
        self._note(f"Speed: even G {edge:g} was not clean ({status}); trying G {vals[0]:g} … {vals[-1]:g} "
                   f"(down to manual rule x {vals[0] * self.cfg.start_factor:g}).")
        title = "speed below the start (G ramp at ρ 1)"
        return Stage("lower", title,
                     "The start was already too aggressive: the same ramp, below it; ends at the first ringing / "
                     "Drive at zero / lost condition.",
                     R.ramp("g", vals, rho=1.0, group=self._group(title)), self.cfg.quiet_search_s)

    def _after_shape(self, ms: List[Measured]) -> Optional[Stage]:
        clean = [m for m in ms if m.role == "test" and m.clean]
        if not clean:
            self._rho = 1.0
            self._shape_clean_at_g_good = False
            self._note(f"Shape: no ρ gave clean steps at G {self._g_good:g}; keeping ρ 1.")
        else:
            best = self._best(clean, lambda m: m.settle_s)
            self._rho = best.rho
            self._shape_clean_at_g_good = True
            st = f"{best.settle_s * 1e3:.0f} ms" if _finite(best.settle_s) else "n/a"
            self._note(f"Shape: ρ {best.rho:g} settles fastest among the clean ones ({st}).")
        g0, f = self._g_good, self.cfg.edge_factor
        hi = (self._edge1 * f) if self._edge1 is not None else g0 * 10.0
        vals, g = [], g0 * f
        while g <= hi * (1 + 1e-9) and len(vals) < self.cfg.edge_max_steps:
            vals.append(round_sig(g))
            g *= f
        vals = vals or [round_sig(g0 * f)]
        title = f"edge (G ramp at ρ {self._rho:g})"
        return Stage("edge", title,
                     f"Finds where ρ {self._rho:g} stops being clean (G {vals[0]:g} … {vals[-1]:g}, x{f:g} steps).",
                     R.ramp("g", vals, rho=self._rho, group=self._group(title)), self.cfg.quiet_search_s)

    def _after_edge(self, ms: List[Measured]) -> Optional[Stage]:
        tests = [m for m in ms if m.role == "test"]
        edge = min((m.g for m in tests if m.edge), default=None)
        clean = [m.g for m in tests if m.clean and (edge is None or m.g < edge)]
        if self._shape_clean_at_g_good:
            clean.append(self._g_good)
        g_fast = max(clean) if clean else self._g_good
        if edge is not None:
            g_final = min(g_fast, edge / self.cfg.margin)
            self._note(f"Edge: ρ {self._rho:g} first not clean at G {edge:g}; fastest clean G {g_fast:g}. "
                       f"Chosen G {round_sig(g_final):g} = min(fastest clean, edge / {self.cfg.margin:g}).")
        else:
            g_final = g_fast
            top = max((m.g for m in tests), default=g_fast)
            self._note(f"Edge: still clean at G {top:g}; chosen G {round_sig(g_final):g} "
                       "(the safety margin is at least the ramp's range).")
        if not clean:
            self._note("  (no clean point at this ρ: the chosen G is not yet measured clean; validation decides)")
        self._g_final, self._edge2 = round_sig(g_final), edge
        if self.cfg.include_tau:
            vals = self.tau_candidates_ms()
            title = f"AFL Tau at G {self._g_final:g}, ρ {self._rho:g}"
            return Stage("tau", title, f"AFL input filter: {', '.join(f'{v:g}' for v in vals)} ms.",
                         R.ramp("amp_tau_ms", vals, g=self._g_final, rho=self._rho, group=self._group(title)),
                         self.cfg.quiet_search_s)
        return self._validate_stage("validate", self._g_final)

    def _after_tau(self, ms: List[Measured]) -> Optional[Stage]:
        clean = [m for m in ms if m.role == "test" and m.clean]
        if not clean:
            self._note("Tau: no clean value; Tau left as it was.")
        else:
            fastest = min((m.settle_s for m in clean if _finite(m.settle_s)), default=math.nan)
            pool = [m for m in clean if not _finite(fastest) or (_finite(m.settle_s) and m.settle_s <= 1.2 * fastest)]
            best = self._best(pool or clean, self._amp_noise)
            self._tau = float(best.settings["amp_tau_ms"])
            self._note(f"Tau: {self._tau:g} ms, the quietest amplitude among the clean values that settle within "
                       "20 % of the fastest.")
        return self._validate_stage("validate", self._g_final)

    def _validate_stage(self, key: str, g: float) -> Stage:
        base = {"amp_tau_ms": self._tau} if self._tau is not None else {}
        title = f"{'validation' if key == 'validate' else 'fallback validation'} at G {g:g}, ρ {self._rho:g}"
        group = self._group(title)
        conds = [R.Condition(g=g, rho=self._rho, settings=dict(base), group=group, ramp="repeat")
                 for _ in range(self.cfg.repeats)]
        for f in self.cfg.ref_factors:
            conds.append(R.Condition(g=g, rho=self._rho, settings={**base, "amp_ref": round_sig(self.cfg.base_ref * f, 4)},
                                     group=group, ramp="amp_ref"))
        refs = ", ".join(f"x{f:g}" for f in self.cfg.ref_factors)
        return Stage(key, title,
                     f"{self.cfg.repeats} repeats must all be clean" + (f"; Ref {refs} reported only." if refs else "."),
                     conds, self.cfg.quiet_validate_s)

    def _after_validate(self, ms: List[Measured]) -> Optional[Stage]:
        stage = self.stages[-1]
        reps = [m for m in ms if "amp_ref" not in m.settings]
        others = [m for m in ms if "amp_ref" in m.settings]
        ok = bool(reps) and all(m.clean for m in reps)
        self._note(f"{stage.title}: repeats " + ", ".join(m.status for m in reps) +
                   ("; Ref " + ", ".join(f"{m.settings['amp_ref']:g}: {m.status}" for m in others) if others else ""))
        if not ok and stage.key == "validate":
            g2 = round_sig(self._g_final / self.cfg.edge_factor)
            self._note(f"Not every repeat was clean: one more validation at G {g2:g}.")
            self._g_final = g2
            return self._validate_stage("fallback", g2)
        kp, ki = self.gains(self._g_final, self._rho)
        note = "" if ok else "not every repeat was clean, even after the fallback: treat as a starting point only"
        self.recommendation = Recommendation(self._g_final, self._rho, self._tau, kp, ki, ok, self._edge2, note)
        tau = f", Tau {self._tau:g} ms" if self._tau is not None else ""
        self._note(f"Recommended: Kp {kp:.4g}, Ki {ki:.4g}{tau} (G {self._g_final:g}, ρ {self._rho:g}); "
                   + ("confirmed." if ok else note + "."))
        return None

    # -- report ----------------------------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        rec = None
        if self.recommendation is not None:
            rec = {**asdict(self.recommendation), "margin": self.recommendation.margin}
        return {
            "kind": KIND, "version": 1,
            "config": asdict(self.cfg),
            "manual_rule": {"kp": self.manual.kp, "ki": self.manual.ki, "tau_ms": self.manual.tau_s * 1e3,
                            "ring_down_s": self.manual.ring_down_s, "g": self.manual_g},
            "anchor": asdict(self.anchor),
            "timing": asdict(self.timing),
            "stages": [{"key": s.key, "title": s.title, "why": s.why, "quiet_s": s.quiet_s,
                        "conditions": [m.to_dict() for m in (self.results[i] if i < len(self.results) else [])]}
                       for i, s in enumerate(self.stages)],
            "decisions": list(self.decisions),
            "judged_from": "QPlusAmpl only (Drive reported, not judged)",
            "overshoot_base": self.overshoot_base,
            "failure": self.failure, "stopped": self.stopped,
            "recommendation": rec,
        }

    def report_markdown(self) -> str:
        c, t = self.cfg, self.timing
        out = ["# Amplitude loop: automatic search", "",
               f"Sensor: f0 {c.f0:.6g} Hz, Q {c.q:.4g}, ring-down {t.ring_down_s:.3g} s. "
               f"Output gain ±{c.output_gain_v:g} V, InA x{c.input_gain:g}, Ref {c.base_ref:g}.", "",
               f"Step train: Ref ±{c.step_frac * 100:g} %, hold {t.hold_s:g} s per level, settle {t.settle_s:g} s "
               f"(timeout {t.settle_timeout_s:g} s); quiet window {c.quiet_search_s:g} s while searching, "
               f"{c.quiet_validate_s:g} s when validating.", "",
               f"Start: manual rule Kp {self.manual.kp:.3g}, Ki {self.manual.ki:.3g}; anchor G = 1 = manual x "
               f"{c.start_factor:g} (Kp {self.anchor.kp:.3g}, Ki {self.anchor.ki:.3g}).", ""]
        rec = self.recommendation
        out += ["## Result", ""]
        if rec is not None:
            tau = f", Tau {rec.tau_ms:g} ms" if rec.tau_ms is not None else ""
            out.append(f"**Kp {rec.kp:.4g}, Ki {rec.ki:.4g}{tau}** (G {rec.g:g}, ρ {rec.rho:g}): "
                       + ("confirmed by every repeat." if rec.confirmed else rec.note + "."))
            if rec.margin is not None:
                out.append(f"Safety margin: the loop first stopped being clean at {rec.margin:.2g} x this G.")
        elif self.failure:
            out.append(f"No recommendation: {self.failure}")
        else:
            out.append(f"No recommendation: the search ended early ({self.stopped or 'not finished'}).")
        out += ["", "## Decisions", ""] + [f"- {d}" for d in self.decisions]

        def ms(v):
            return "" if not _finite(v) else f"{v * 1e3:.0f}"

        def pct(v):
            return "" if not _finite(v) else f"{v * 100:.0f}"

        def uv(v):
            return "" if not _finite(v) else f"{v * 1e6:.3g}"

        def pct2(v):
            return "" if not _finite(v) else f"{v * 100:.2f}"
        out.insert(5, "Judged from QPlusAmpl only; Drive is reported but decides nothing (with the AFL on, SXM's "
                      "Drive is not the whole excitation).")
        out.insert(6, "")
        for i, s in enumerate(self.stages):
            out += ["", f"## {i + 1}. {s.title}", "", s.why, "",
                    "| G | ρ | Settings | Result | Up 10-90 (ms) | Down 10-90 (ms) | Settle (ms) | Overshoot up / down (%) "
                    "| Amp noise (%) | Drive noise (µV, reported) |",
                    "|---|---|---|---|---|---|---|---|---|---|"]
            for m in (self.results[i] if i < len(self.results) else []):
                sets = ", ".join(f"{R.SETTINGS[k].label} {R.format_setting(k, v)}" for k, v in m.settings.items())
                out.append(f"| {m.g:g} | {m.rho:g} | {sets} | {m.status} | {ms(m.up_rise_s)} | {ms(m.down_rise_s)} | "
                           f"{ms(m.settle_s)} | {pct(m.up_overshoot)} / {pct(m.down_overshoot)} | "
                           f"{pct2(m.amp_noise)} | {uv(m.noise)} |")
        return "\n".join(out) + "\n"
