"""
Runs a run sheet (tuning/runsheet.py) on the instrument, one condition after another, the way a loop is tuned by
hand:

1. **Settings** - the condition's GUI-only settings (AFL Tau, DNC TimeConstant / RollOff, output gain, input
   gain) are set, either by the operator (the run pauses, says what to set, and continues once SXM's read-back
   shows it) or, when enabled, through AnfatecSXMWriter (experimental: not yet verified on the instrument).
2. **Apply** - the gains (G and rho relative to the anchor, compensated for output / input gain) and the base
   setpoint are written over DDE.
3. **Settle** - wait at least ``settle_s``, then until the loop is steady at the first level.
4. **Step train** - the stepped parameter toggles; the response is analysed with tuning/workflow.py.
5. **Quiet window** - back at the base value, wait ``quiet_settle_s``, then record ``quiet_s`` with nothing
   stepped; analysed with tuning/noise.py.

All channels of both loops are recorded together (one capture, round robin), so a step of one loop also shows
what it does to the other. A condition that loses the loop releases it, writes the anchor back and waits for the
loop to recover before the next one; no recovery stops the run. A G or rho ramp stops at its first condition that
rings, loses the loop or keeps Drive at zero (``runsheet.stop_rule``).

DDE writes block on the GUI thread, so this is a QTimer state machine; only the capture runs in a worker thread.
At the end, on Stop and on abort the anchor gains and the start setpoints are written back.
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from sxm_ncafm_control import metadata as MD
from sxm_anfatec.driver import CHANNELS
from sxm_ncafm_control.tuning import explore as X
from sxm_ncafm_control.tuning import noise as N
from sxm_ncafm_control.tuning import runsheet as R
from sxm_ncafm_control.tuning import workflow as W

LOOP_CHANNELS = {"afl": ("QPlusAmpl", "Drive"), "pll": ("Phase", "df")}
ALL_CHANNELS = ("QPlusAmpl", "Drive", "df", "Phase")
DISPLAY_BIN_S = 0.005          # what is kept in memory per condition for the plots


class TrainCaptureThread(QtCore.QThread):
    """
    Reads channels back-to-back and stores their means in ``bin_s`` slices with real time stamps
    (``time.perf_counter``, the same clock the runner uses for the events).

    The GUI thread reads ``t``/``data``/``count`` without a lock: a slice is complete before ``count``
    is incremented, and this is display/safety monitoring only.
    """

    error = QtCore.pyqtSignal(str)

    def __init__(self, driver, channels: Dict[str, Tuple[int, float]], bin_s: float = 0.0005, max_s: float = 30.0):
        super().__init__()
        self.driver = driver
        self.channels = channels                # name -> (driver index, scale to physical units)
        self.bin_s = bin_s
        n = int(max_s / bin_s) + 4
        self.t = np.zeros(n)
        self.data = {name: np.zeros(n) for name in channels}
        self.count = 0
        self.t0 = 0.0
        self._stop = False

    def stop(self):
        self._stop = True

    def now(self) -> float:
        """Seconds since this capture started, on the capture's clock."""
        return time.perf_counter() - self.t0 if self.t0 else 0.0

    def run(self):
        clock = time.perf_counter
        names = list(self.channels)
        idxs = [self.channels[n][0] for n in names]
        scales = [self.channels[n][1] for n in names]
        read = self.driver.read_raw
        sums = [0.0] * len(names)
        cnt = 0
        n_max = len(self.t) - 1
        self.t0 = t0 = clock()
        start = t0
        edge = t0 + self.bin_s
        while not self._stop and self.count < n_max:
            try:
                for q, idx in enumerate(idxs):
                    sums[q] += read(idx)
            except Exception as e:
                self.error.emit(f"Driver read error: {e}")
                return
            cnt += 1
            now = clock()
            if now >= edge:
                i = self.count
                self.t[i] = 0.5 * (start + now) - t0
                for q, name in enumerate(names):
                    self.data[name][i] = sums[q] / cnt * scales[q]
                    sums[q] = 0.0
                cnt = 0
                start = now
                edge = now + self.bin_s
                self.count = i + 1
                time.sleep(0)                   # hand the GIL to the GUI thread once per slice: its timers send the steps


@dataclass
class RunConfig:
    anchor: R.Anchor
    sheet: List[R.Condition]
    plan: W.StepTestPlan                      # the step train; its base is replaced per condition
    start_settings: Dict[str, Any]            # runsheet setting keys -> values in effect when the run starts
    base_use: float                           # DNC use [Hz] at the start (PLL base; restored at the end)
    quiet_s: float = 30.0
    quiet_settle_s: float = 3.0
    settle_timeout_s: float = 60.0
    recover_timeout_s: float = 60.0
    compensate: bool = True
    record_both: bool = True
    other_loop_on: bool = True                # watch the other loop for a lost lock / collapse
    criteria: X.RecoveryCriteria = field(default_factory=X.RecoveryCriteria)
    limits: W.SafetyLimits = field(default_factory=W.SafetyLimits)
    explore_limits: X.ExploreLimits = field(default_factory=X.ExploreLimits)
    ring_down_s: Optional[float] = None
    analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    """One condition, measured (or skipped)."""

    index: int
    condition: R.Condition
    kp: float = math.nan
    ki: float = math.nan
    settings: Dict[str, Any] = field(default_factory=dict)       # all settings in effect
    result: Optional[W.StepTestResult] = None
    assessment: Optional[X.Assessment] = None
    quiet: Optional[N.QuietMetrics] = None
    status: str = "pending"                  # explore.STATUSES, or 'skipped' / 'stopped'
    note: str = ""
    time: str = ""
    meta: Optional[MD.Metadata] = None       # SXM read when the condition ended (see _end_meta)
    plan: Optional[W.StepTestPlan] = None    # the step train as run (its base, levels, hold)
    # display copy of the recording (DISPLAY_BIN_S bins) and its marks, seconds from the capture start
    t: Optional[np.ndarray] = None
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    events: List[float] = field(default_factory=list)
    marks: Dict[str, float] = field(default_factory=dict)      # settle / test / quiet / end
    raw: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None   # full recording, until saved


def _decimate(t, data: Dict[str, np.ndarray], bin_s: float):
    if len(t) < 2:
        return t, data
    idx = np.floor((t - t[0]) / bin_s).astype(int)
    cnt = np.bincount(idx)
    keep = cnt > 0
    td = (np.bincount(idx, weights=t)[keep] / cnt[keep])
    return td, {k: np.bincount(idx, weights=v)[keep] / cnt[keep] for k, v in data.items()}


class SheetRunner(QtCore.QObject):
    condition_started = QtCore.pyqtSignal(int)              # sheet index
    condition_finished = QtCore.pyqtSignal(object)          # Outcome
    skipped = QtCore.pyqtSignal(int, str)                   # sheet index, why
    operator_needed = QtCore.pyqtSignal(str)                # what to set in SXM; the run waits for continue_()
    phase = QtCore.pyqtSignal(str)
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)                       # 'done' | 'stopped' | 'aborted: ...'

    POLL_MS = 100
    PAUSE_MS = 100

    def __init__(self, dde, driver, cfg: RunConfig, *, reader=None, writer=None,
                 meta_fn: Optional[Callable[[], MD.Metadata]] = None, parent=None):
        super().__init__(parent)
        self.dde, self.driver, self.cfg = dde, driver, cfg
        self.reader, self.writer = reader, writer
        self.meta_fn = meta_fn
        self.loop = cfg.plan.loop_def
        self.current: Dict[str, Any] = dict(cfg.start_settings)
        self.running = False
        self.waiting_operator = False
        self.skip: Dict[int, str] = {}
        self._i = -1
        self._token = 0
        self._writing = 0
        self._cap: Optional[TrainCaptureThread] = None
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(self.POLL_MS)
        self._poll.timeout.connect(self._on_poll)
        self._state = "idle"
        self._t_state = 0.0
        self._marks: Dict[str, float] = {}
        self._events: List[Tuple[int, float]] = []
        self._plan: Optional[W.StepTestPlan] = None
        self._out: Optional[Outcome] = None
        self._kappa: Optional[float] = None
        self._amp_start: Optional[float] = None
        self._lost = False
        self._last_why = ""
        self._pending_settings: Dict[str, Any] = {}

    # -- instrument writes ----------------------------------------------------------------------------
    def _write(self, param, value):
        # SendWait pumps the thread's message queue until SXM answers, so Qt timers fire *inside* a write;
        # _on_poll and _later hold off while _writing is set
        ptype, code = param
        self._writing += 1
        try:
            if ptype == "DNC":
                self.dde.send_dncpara(int(code), float(value))
            else:
                self.dde.send_scanpara(str(code), float(value))
        finally:
            self._writing -= 1

    def _gains_for(self, cond: R.Condition) -> Tuple[float, float]:
        return R.gains(self.cfg.anchor, cond, self.current, self.cfg.compensate)

    def _apply_gains(self, kp, ki):
        self._write(self.loop.kp_param, kp)
        self._write(self.loop.ki_param, ki)

    def _restore_anchor(self, release_first: bool):
        if release_first and self.loop.key == "pll":
            self._apply_gains(0.0, 0.0)
        self._apply_gains(*self._gains_for(R.Condition()))

    def _base_of(self, cond: R.Condition) -> float:
        if self.loop.key == "afl":
            return float(cond.settings.get("amp_ref", self.current.get("amp_ref") or self.cfg.plan.base))
        return self.cfg.base_use

    # -- control --------------------------------------------------------------------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        self._i = -1
        self._next()

    def stop(self):
        if self.running:
            self._finish("stopped")

    def continue_(self):
        """The operator has set what operator_needed asked for."""
        if self.running and self.waiting_operator:
            self.waiting_operator = False
            self._check_settings(after_operator=True)

    def _finish(self, reason: str):
        self._token += 1
        self._poll.stop()
        self._stop_capture()
        was = self.running
        self.running = False
        self.waiting_operator = False
        self._state = "idle"
        if not was:
            return
        if self._out is not None and self._out.status == "pending":
            self._out.status, self._out.note = "stopped", reason
            self.condition_finished.emit(self._out)
        self._out = None
        try:
            self._restore_anchor(release_first=self._lost)
            if self.loop.key == "pll":
                self._write(self.loop.step_param, self.cfg.base_use)
            ref0 = self.cfg.start_settings.get("amp_ref") or (self.cfg.plan.base if self.loop.key == "afl" else None)
            if ref0:
                self._write(("EDIT", "Edit23"), ref0)
                self.current["amp_ref"] = ref0
        except Exception as e:  # pragma: no cover - only when the instrument itself is failing
            self.message.emit(f"WARNING: could not restore the anchor: {e}")
        kp, ki = self._gains_for(R.Condition())
        self.message.emit(f"Anchor restored: Kp={kp:.4g}, Ki={ki:.4g}.")
        changed = [k for k, v in self.cfg.start_settings.items()
                   if k != "amp_ref" and v is not None and not R.same_setting(k, self.current.get(k), v)]
        if changed:
            if self.writer is not None and self._write_settings({k: self.cfg.start_settings[k] for k in changed}):
                self.message.emit("Start settings restored: " + ", ".join(R.SETTINGS[k].label for k in changed) + ".")
            else:
                self.message.emit("Set back by hand if wanted: " + "; ".join(
                    f"{R.SETTINGS[k].label} {R.format_setting(k, self.cfg.start_settings[k])}" for k in changed))
        self.finished.emit(reason)

    def _fail_run(self, reason: str):
        self.message.emit(f"STOPPED: {reason}")
        self._finish(f"aborted: {reason}")

    def _later(self, ms, fn, *args):
        tok = self._token

        def call():
            if not (self.running and tok == self._token):
                return
            if self._writing:
                QtCore.QTimer.singleShot(2, QtCore.Qt.PreciseTimer, call)
                return
            fn(*args)
        QtCore.QTimer.singleShot(max(0, int(ms)), QtCore.Qt.PreciseTimer, call)

    def _stop_capture(self):
        if self._cap is not None:
            self._cap.stop()
            self._cap.wait(1000)
            self._cap = None

    # -- sequencing -------------------------------------------------------------------------------------
    def _next(self):
        if not self.running:
            return
        self._i += 1
        while self._i < len(self.cfg.sheet) and self._i in self.skip:
            self._i += 1
        if self._i >= len(self.cfg.sheet):
            self._finish("done")
            return
        self._token += 1
        cond = self.cfg.sheet[self._i]
        self._out = Outcome(self._i, cond, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.condition_started.emit(self._i)
        self._check_settings(after_operator=False)

    # -- 1. settings ----------------------------------------------------------------------------------
    def _read_settings(self) -> Optional[Dict[str, Any]]:
        if self.reader is None:
            return None
        try:
            ro = self.reader.read()
        except Exception as e:
            self.message.emit(f"Reading SXM failed: {e}")
            return None
        if not getattr(ro, "ok", False):
            return None
        return {k: R.sxm_value(k, ro.values.get(s.sxm_key)) for k, s in R.SETTINGS.items()}

    def _needed(self, cond: R.Condition) -> Dict[str, Any]:
        return {k: v for k, v in cond.settings.items()
                if R.SETTINGS[k].writer_name and not R.same_setting(k, self.current.get(k), v)}

    def _write_settings(self, wanted: Dict[str, Any]) -> bool:
        if self.writer is None:
            return False
        ok = True
        for k, v in wanted.items():
            s = R.SETTINGS[k]
            try:
                opt = R.option_for(k, self.writer.options(s.writer_name), v)
                if opt is None:
                    raise ValueError(f"no option for {R.format_setting(k, v)}")
                self._writing += 1
                try:
                    self.writer.set(s.writer_name, opt)
                finally:
                    self._writing -= 1
                self.current[k] = v
            except Exception as e:
                ok = False
                self.message.emit(f"  could not set {s.label} to {R.format_setting(k, v)} automatically: {e}")
        return ok

    def _check_settings(self, after_operator: bool):
        cond = self.cfg.sheet[self._i]
        if after_operator:
            read = self._read_settings()
            if read is None:
                self.message.emit("  SXM cannot be read: taking the operator's word for the settings.")
                self.current.update({k: v for k, v in cond.settings.items() if R.SETTINGS[k].writer_name})
            else:
                for k in cond.settings:
                    if R.SETTINGS[k].writer_name and read.get(k) is not None:
                        self.current[k] = read[k]
        need = self._needed(cond)
        if need and not after_operator and self.writer is not None:
            self._write_settings(need)
            need = self._needed(cond)
        if need:
            self.waiting_operator = True
            text = "; ".join(f"{R.SETTINGS[k].label} = {R.format_setting(k, v)}" for k, v in need.items())
            if after_operator:
                text = "SXM still shows otherwise. " + text
            self.phase.emit(f"#{self._i + 1}: waiting for the operator")
            self.operator_needed.emit(text)
            return
        self._apply()

    # -- 2. apply, 3. settle ------------------------------------------------------------------------------
    def _apply(self):
        cond = self.cfg.sheet[self._i]
        out = self._out
        kp, ki = self._gains_for(cond)
        out.kp, out.ki = kp, ki
        base = self._base_of(cond)
        try:
            self._plan = dataclasses.replace(self.cfg.plan, base=base)
        except ValueError as e:
            self._end_condition(failure=f"step train not possible here: {e}")
            return
        out.plan = self._plan
        out.settings = {**self.current, **cond.settings}
        if self.meta_fn is not None:
            try:
                out.meta = self.meta_fn()
            except Exception:
                out.meta = None
        names = list(ALL_CHANNELS if self.cfg.record_both else LOOP_CHANNELS[self.loop.key])
        specs = {n: (CHANNELS[n][0], CHANNELS[n][3]) for n in names}
        max_s = (self.cfg.settle_timeout_s + self._plan.duration + self.cfg.quiet_settle_s + self.cfg.quiet_s + 10.0)
        self._cap = TrainCaptureThread(self.driver, specs, max_s=max_s)
        self._cap.error.connect(self._fail_run)
        self._cap.start()
        self._events, self._marks = [], {}
        self._kappa = None
        self._amp_start = None
        try:
            if "amp_ref" in cond.settings and self.loop.key == "pll":
                self._write(("EDIT", "Edit23"), cond.settings["amp_ref"])
            if self.loop.key == "afl":
                self.current["amp_ref"] = base
            elif "amp_ref" in cond.settings:
                self.current["amp_ref"] = cond.settings["amp_ref"]
            self._apply_gains(kp, ki)
            self._write(self.loop.step_param, self._plan.levels[0])
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self._lost = False
        self.message.emit(f"#{self._i + 1} {cond.label()}: Kp={kp:.4g}, Ki={ki:.4g}")
        self._enter("settle")
        self._marks["settle"] = self._t_state
        self._poll.start()

    def _enter(self, state: str):
        self._state = state
        self._t_state = self._cap.now() if self._cap is not None else 0.0
        text = {"settle": "settling", "test": "stepping", "quiet_settle": "back at base, settling",
                "quiet": "quiet window", "recover": "recovering at the anchor"}[state]
        self.phase.emit(f"#{self._i + 1}: {text}")

    def _window(self, names: Tuple[str, str]):
        cap = self._cap
        n = cap.count
        if n < 5:
            return None
        t = cap.t[:n]
        t_end = t[-1]
        lo = int(np.searchsorted(t, t_end - self.cfg.criteria.window_s))
        if t_end - self._t_state < self.cfg.criteria.window_s or n - lo < 20:
            return None
        return cap.data[names[0]][lo:n], cap.data[names[1]][lo:n]

    def _runaway(self) -> Optional[str]:
        cap = self._cap
        n = cap.count
        if n < 40:
            return None
        lo = max(0, n - 100)
        tail = {name: arr[lo:n] for name, arr in cap.data.items()}
        own = {k: v for k, v in tail.items() if k in LOOP_CHANNELS[self.loop.key]}
        why = W.runaway_reason(self._plan, self.cfg.limits, own, self._kappa)
        if why:
            return why
        if not self.cfg.other_loop_on:
            return None
        if self.loop.key == "afl" and "Phase" in tail:
            if float(np.max(np.abs(tail["Phase"][-50:]))) > self.cfg.limits.phase_limit_deg:
                return f"the PLL lost the resonance (|Phase| > {self.cfg.limits.phase_limit_deg:.0f} deg)"
        if self.loop.key == "pll" and "QPlusAmpl" in tail:
            if self._amp_start is None and n > 200:
                self._amp_start = float(np.median(cap.data["QPlusAmpl"][:200]))
            if self._amp_start and float(np.median(tail["QPlusAmpl"][-20:])) < self.cfg.limits.amplitude_floor * self._amp_start:
                return "the oscillation amplitude collapsed"
        return None

    def _on_poll(self):
        if not self.running or self._writing or self._cap is None or self._cap.t0 == 0.0:
            return
        elapsed = self._cap.now() - self._t_state
        if self._state in ("settle", "test", "quiet_settle", "quiet"):
            why = self._runaway()
            if why:
                self.message.emit(f"  lost: {why}")
                self._end_condition(failure=f"lost: {why}")
                return
        if self._state == "settle":
            self._poll_settle(elapsed)
        elif self._state == "quiet_settle" and elapsed >= self.cfg.quiet_settle_s:
            self._enter("quiet")
            self._marks["quiet"] = self._t_state
        elif self._state == "quiet" and elapsed >= self.cfg.quiet_s:
            self._marks["end"] = self._cap.now()
            self._end_condition()
        elif self._state == "recover":
            self._poll_recover(elapsed)

    def _poll_settle(self, elapsed):
        if elapsed < self._plan.settle_s:
            return
        y_name, u_name = LOOP_CHANNELS[self.loop.key]
        w = self._window((y_name, u_name))
        if w is not None:
            y, u = w
            ok, why = X.steady(self.loop.key, y, u, self.cfg.criteria)
            if ok:
                if self.loop.key == "afl":
                    self._kappa = float(np.median(y)) / self._plan.levels[0]
                self._start_test()
                return
            self._last_why = why
        if elapsed > self.cfg.settle_timeout_s:
            self._end_condition(failure=f"did not settle within {self.cfg.settle_timeout_s:g} s ({self._last_why})")

    # -- 4. step train -----------------------------------------------------------------------------------
    def _start_test(self):
        self._enter("test")
        self._marks["test"] = self._t_state
        cap = self._cap
        now = time.perf_counter()
        t0 = self._t_state
        for k, (t_rel, value) in enumerate(self._plan.commands()):
            self._later((cap.t0 + t0 + t_rel - now) * 1000, self._send_event, k, value)
        self._later((cap.t0 + t0 + self._plan.duration - now) * 1000, self._end_test)

    def _send_event(self, k, value):
        t_send = self._cap.now()
        try:
            self._write(self.loop.step_param, value)
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self._events.append((k, t_send))

    def _end_test(self):
        try:
            self._write(self.loop.step_param, self._plan.base)
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self._marks["test_end"] = self._cap.now()
        self._enter("quiet_settle")

    # -- 5. end of a condition ----------------------------------------------------------------------------
    def _end_condition(self, failure: Optional[str] = None):
        self._poll.stop()
        self._token += 1
        cap = self._cap
        out = self._out
        n = cap.count if cap is not None else 0
        self._stop_capture()
        self._state = "idle"
        t = cap.t[:n].copy() if n else np.zeros(0)
        data = {k: v[:n].copy() for k, v in cap.data.items()} if n else {}
        events = [tt for _, tt in sorted(self._events)]
        out.events, out.marks = events, dict(self._marks)
        out.raw = (t, data)
        out.t, out.data = _decimate(t, data, DISPLAY_BIN_S) if n else (t, data)
        cond = self.cfg.sheet[self._i]
        plan = self._plan
        y_name, u_name = LOOP_CHANNELS[self.loop.key]
        if failure is None and plan is not None:
            t_a, t_b = self._marks.get("settle", 0.0), self._marks.get("test_end", t[-1] if n else 0.0)
            keep = (t >= t_a) & (t <= t_b)
            if keep.sum() < 100 or len(events) != plan.n_events:
                failure = f"incomplete recording ({int(keep.sum())} slices, {len(events)}/{plan.n_events} events)"
            else:
                ct = W.CapturedTest(plan=plan, t=t[keep], channels={k: data[k][keep] for k in (y_name, u_name)},
                                    event_times=events, kp=out.kp, ki=out.ki)
                out.result = W.analyze_test(ct, **self.cfg.analysis)
            q0, q1 = self._marks.get("quiet"), self._marks.get("end")
            if q0 is not None and q1 is not None:
                m = (t >= q0) & (t <= q1)
                out.quiet = N.quiet_metrics(t[m], {k: v[m] for k, v in data.items()})
        if failure is not None:
            out.result = W.StepTestResult(kp=out.kp, ki=out.ki, loop=self.loop.key, failure=failure)
        out.assessment = X.assess(out.result, self.cfg.explore_limits, self.cfg.ring_down_s)
        out.status = out.assessment.status
        out.note = " | ".join(([out.result.failure] if out.result.failure else []) + out.assessment.reasons
                              + list(out.result.warnings))
        lost = out.status == "lost" or (out.result.failure or "").startswith("lost")
        out.meta = self._end_meta(out, lost)
        for j in R.stop_rule(self.cfg.sheet, self._i, out.status):
            if j not in self.skip:
                self.skip[j] = f"stop rule: #{self._i + 1} was {X.STATUS_LABEL.get(out.status, out.status)}"
                self.skipped.emit(j, self.skip[j])
        self._out = None
        self.condition_finished.emit(out)
        if lost:
            self._recover()
            return
        self._later(self.PAUSE_MS, self._next)

    def _end_meta(self, out: Outcome, lost: bool) -> Optional[MD.Metadata]:
        """
        The settings the condition ran at. The read at its start comes before its gains are written, so SXM is
        read again now, with the gains still in place (as a Step Test export reads SXM when the capture ends).
        A lost condition is not re-read (recovery comes first): its start read gets the gains as written.
        """
        if not lost and self.meta_fn is not None:
            try:
                meta = self.meta_fn()
                if meta is not None and meta.has_sxm:
                    return meta
            except Exception:
                pass
        start = out.meta
        if start is None or not start.has_sxm:
            return start
        meta = MD.Metadata(start.values, start.errors, start.timestamp,
                           f"{start.source}, before the condition; {self.loop.key.upper()} gains as written")
        kp_key, ki_key = ("amp_kp", "amp_ki") if self.loop.key == "afl" else ("pll_kp", "pll_ki")
        meta.values[kp_key], meta.values[ki_key] = out.kp, out.ki
        return meta

    # -- after a lost condition ------------------------------------------------------------------------------
    def _recover(self):
        self._lost = True
        try:
            self._restore_anchor(release_first=True)
            base = self._base_of(R.Condition())
            self._write(self.loop.step_param, base)
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        names = list(LOOP_CHANNELS[self.loop.key])
        self._cap = TrainCaptureThread(self.driver, {n: (CHANNELS[n][0], CHANNELS[n][3]) for n in names},
                                       max_s=self.cfg.recover_timeout_s + 5.0)
        self._cap.error.connect(self._fail_run)
        self._cap.start()
        self._token += 1
        self._enter("recover")
        self._poll.start()

    def _poll_recover(self, elapsed):
        w = self._window(LOOP_CHANNELS[self.loop.key])
        if w is not None:
            ok, why = X.steady(self.loop.key, w[0], w[1], self.cfg.criteria)
            if ok:
                self._poll.stop()
                self._stop_capture()
                self._state = "idle"
                self.message.emit(f"  recovered at the anchor in {elapsed:.1f} s")
                self._lost = False
                self._later(self.PAUSE_MS, self._next)
                return
            self._last_why = why
        if elapsed > self.cfg.recover_timeout_s:
            self._fail_run(f"the loop did not recover at the anchor within {self.cfg.recover_timeout_s:g} s "
                           f"({self._last_why or 'no data'})")
