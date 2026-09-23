"""
Tuning tab: guided screening of the PLL / amplitude-loop gains on the instrument.

Workflow (the manual's, made systematic)
----------------------------------------
1. Define the test: which loop, the step (PLL: +-1 Hz on ``DNC use``; amplitude: +-10 % on ``Ref``),
   hold time, number of events. The channels recorded are implied by the loop
   (PLL: ``df`` + ``Phase``; amplitude loop: ``QPlusAmpl`` + ``Drive``).
2. Enter the known-good baseline gains (SXM cannot be read back, so the tab needs them).
3. Set the target: an imaging scan (line time, pixels) or a response time.
4. Run one test, or screen a log-spaced grid around the baseline indexed by **Speed** (Kp and Ki raised or
   lowered together, same ratio) and **Shape** (the Ki:Kp ratio, shifted relative to the baseline's): every
   cell is a step train, measured and classified. The 2D map shows *islands* of rectangular, well-controlled
   response; moving up (faster speed, same shape) makes the loop faster, down slower; moving sideways changes
   the shape. Click a cell for its averaged response and advice (higher / lower / different pairing), then
   refine around it (by hand, or with "Narrow search": one confirmed round of hardware writes at a time).
5. "Analyze Scope capture" applies the same analysis to a step train already recorded in the Scope tab.

Safety: the loop is only driven inside limits around the baseline; a running test is aborted if the
loop runs away (Phase beyond the limit, amplitude collapse); the baseline gains and the stepped
parameter are always written back at the end, on Stop and on abort. Cells more aggressive than one
that lost the loop are skipped. Run it with the tip retracted.

DDE calls block on the GUI thread, so the runner is a QTimer-driven state machine; only the channel
capture runs in a worker thread.
"""

import dataclasses
import functools
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from sxm_ncafm_control.device_driver import CHANNELS
from sxm_ncafm_control.tuning import metrics as M
from sxm_ncafm_control.tuning import workflow as W

from ..common import VOLTAGE_LIMIT_ABS, append_log_line, format_number
from .sci_spinbox import SciDoubleSpinBox

CATEGORY_COLOR = {
    "good": (70, 175, 95), "overshoot": (240, 200, 60), "ringing": (240, 140, 50), "slow_tail": (120, 170, 230),
    "too_slow": (175, 175, 175), "lost": (205, 65, 65), "skipped": (95, 95, 95), "untested": (238, 238, 238),
}

# the amplitude-loop scale sweep's own 4-tier scheme (tuning/workflow.py's assess_scale_sweep) - a
# different vocabulary from CATEGORY_COLOR above, so its own palette rather than overloading that one.
SCALE_TIER_COLOR = {
    "too_slow": (175, 175, 175), "acceptable": (150, 195, 220), "near_optimum": (70, 175, 95), "too_aggressive": (205, 65, 65),
}


# ---------------------------------------------------------------------------
# persisted UI state: left panel width, which sections are expanded (same JSON-in-home-dir pattern
# as gui_accessibility_manager.py, scoped to this tab instead of the whole app)
# ---------------------------------------------------------------------------
_LAYOUT_FILE = os.path.join(os.path.expanduser("~"), ".scientific_gui_tuning_layout.json")


def _load_tuning_layout() -> dict:
    try:
        with open(_LAYOUT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_tuning_layout(state: dict) -> None:
    try:
        with open(_LAYOUT_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


class CollapsibleSection(QtWidgets.QWidget):
    """
    A QGroupBox-like section that can fold away to a one-line header. Collapsing never hides
    information: the header keeps showing a live summary of the section's current values (set with
    ``set_summary()``), so you never have to expand a section just to read what it is set to - only
    to change it.
    """

    toggled = QtCore.pyqtSignal(bool)          # emits the new expanded state

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title
        self._summary = ""
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.header = QtWidgets.QToolButton()
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setStyleSheet("QToolButton { border: none; font-weight: bold; text-align: left; }")
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        v.addWidget(self.header)
        self.content = QtWidgets.QWidget()
        self.content.setVisible(expanded)
        v.addWidget(self.content)
        self.header.toggled.connect(self._on_toggled)
        self._on_toggled(expanded)

    def set_content_layout(self, layout: QtWidgets.QLayout) -> None:
        self.content.setLayout(layout)

    def set_summary(self, text: str) -> None:
        self._summary = text
        self._refresh_header()

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self.header.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content.setVisible(checked)
        self._refresh_header()
        self.toggled.emit(checked)

    def _refresh_header(self) -> None:
        if self.header.isChecked() or not self._summary:
            self.header.setText(self._title)
        else:
            self.header.setText(f"{self._title}  —  {self._summary}")


# ---------------------------------------------------------------------------
# capture of a step train (worker thread)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# the runner (GUI-thread state machine)
# ---------------------------------------------------------------------------
class TuningRunner(QtCore.QObject):
    """
    Runs tests one after another: apply gains -> settle -> record while toggling the stepped parameter ->
    restore the parameter -> analyse. ``next_item()`` supplies ``(cell, kp, ki)`` or None when done.
    """

    test_started = QtCore.pyqtSignal(object, float, float)
    test_finished = QtCore.pyqtSignal(object, object)        # cell, StepTestResult
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)                         # 'done' | 'stopped' | 'aborted: ...'

    GUARD_MS = 100
    PAUSE_MS = 200

    def __init__(self, dde, driver, plan: W.StepTestPlan, baseline: Tuple[float, float], analysis: dict,
                 limits: W.SafetyLimits = W.SafetyLimits(), parent=None):
        super().__init__(parent)
        self.dde, self.driver, self.plan = dde, driver, plan
        self.baseline = baseline
        self.analysis = analysis                 # kwargs for analyze_test (li_tau, li_stages, ctrl_tau, f0, q)
        self.limits = limits
        self.loop = plan.loop_def
        self.running = False
        self._token = 0
        self._cap: Optional[TrainCaptureThread] = None
        self._events: List[Tuple[int, float]] = []
        self._kappa: Optional[float] = None
        self._cell = None
        self._kp = self._ki = 0.0
        self._guard = QtCore.QTimer(self)
        self._guard.setInterval(self.GUARD_MS)
        self._guard.timeout.connect(self._check_guard)
        self._next_item = None

    # -- instrument writes -------------------------------------------------------------------
    def _write(self, param, value):
        ptype, code = param
        if ptype == "DNC":
            self.dde.send_dncpara(int(code), float(value))
        else:
            self.dde.send_scanpara(str(code), float(value))

    def _apply_gains(self, kp, ki):
        self._write(self.loop.kp_param, kp)
        self._write(self.loop.ki_param, ki)

    def _restore(self):
        """Put the baseline gains and the stepped parameter back. Never raises."""
        for what, fn in (("baseline gains", lambda: self._apply_gains(*self.baseline)),
                         ("stepped parameter", lambda: self._write(self.loop.step_param, self.plan.base))):
            try:
                fn()
            except Exception as e:  # pragma: no cover - only when the instrument itself is failing
                self.message.emit(f"WARNING: could not restore the {what}: {e}")

    # -- control ----------------------------------------------------------------------------------
    def start(self, next_item):
        if self.running:
            return
        self._next_item = next_item
        self.running = True
        self._advance()

    def stop(self):
        if self.running:
            self._finish("stopped")

    def _finish(self, reason: str):
        self._token += 1                        # invalidates every pending timer callback
        self._guard.stop()
        if self._cap is not None:
            self._cap.stop()
            self._cap.wait(1000)
            self._cap = None
        was_running = self.running
        self.running = False
        if was_running:
            self._restore()
            kp, ki = self.baseline
            step_label = "DNC use" if self.loop.key == "pll" else "Ref"
            self.message.emit(f"Baseline restored: Kp={kp:.4g}, Ki={ki:.4g}, {step_label}={self.plan.base:.4g}.")
            self.finished.emit(reason)

    def _later(self, ms, fn, *args):
        """Single-shot precise timer that is ignored if the run was stopped meanwhile."""
        tok = self._token

        def call():
            if self.running and tok == self._token:
                fn(*args)
        QtCore.QTimer.singleShot(max(0, int(ms)), QtCore.Qt.PreciseTimer, call)

    # -- one test -------------------------------------------------------------------------------------
    def _advance(self):
        if not self.running:
            return
        item = self._next_item()
        if item is None:
            self._finish("done")
            return
        self._token += 1
        self._cell, self._kp, self._ki = item
        self._events, self._kappa = [], None
        self.test_started.emit(self._cell, self._kp, self._ki)
        try:
            self._apply_gains(self._kp, self._ki)
            self._write(self.loop.step_param, self.plan.levels[0])
        except Exception as e:
            self._fail(f"writing to SXM failed: {e}")
            return
        self._later(self.plan.settle_s * 1000, self._begin_capture)

    def _begin_capture(self):
        names = [self.loop.y_channel, self.loop.u_channel]
        specs = {n: (CHANNELS[n][0], CHANNELS[n][3]) for n in names}
        self._cap = TrainCaptureThread(self.driver, specs, max_s=self.plan.duration + 2.0)
        self._cap.error.connect(lambda msg: self._fail(msg))
        self._cap.start()
        self._later(5, self._schedule)

    def _schedule(self):
        cap = self._cap
        if cap is None:
            return
        if cap.t0 == 0.0:                       # the thread has not started its clock yet
            self._later(5, self._schedule)
            return
        now = time.perf_counter()
        for k, (t_rel, value) in enumerate(self.plan.commands()):
            self._later((cap.t0 + t_rel - now) * 1000, self._send_event, k, value)
        self._later((cap.t0 + self.plan.duration - now) * 1000, self._end_test)
        self._guard.start()

    def _send_event(self, k, value):
        t_send = time.perf_counter() - self._cap.t0
        try:
            self._write(self.loop.step_param, value)
        except Exception as e:
            self._fail(f"writing to SXM failed: {e}")
            return
        self._events.append((k, t_send))

    def _tail(self):
        cap = self._cap
        n = cap.count
        lo = max(0, n - 100)
        return n, {name: arr[lo:n] for name, arr in cap.data.items()}

    def _check_guard(self):
        if not self.running or self._cap is None:
            return
        n, tail = self._tail()
        if n < 40:
            return
        if self.plan.loop == "afl" and self._kappa is None and self._cap.t[n - 1] > 0.9 * self.plan.lead_s:
            first = self._cap.t[:n] > 0.4 * self.plan.lead_s
            first &= self._cap.t[:n] < 0.9 * self.plan.lead_s
            if first.sum() > 10:
                self._kappa = float(np.median(self._cap.data[self.loop.y_channel][:n][first]) / self.plan.levels[0])
        reason = W.runaway_reason(self.plan, self.limits, tail, self._kappa)
        if reason:
            self._abort_test(reason)

    def _end_test(self):
        self._guard.stop()
        cap = self._cap
        cap.stop()
        cap.wait(1000)
        self._cap = None
        try:
            self._write(self.loop.step_param, self.plan.base)
        except Exception as e:
            self._fail(f"writing to SXM failed: {e}")
            return
        n = cap.count
        events = [t for _, t in sorted(self._events)]
        if n < 100 or len(events) != self.plan.n_events:
            res = W.StepTestResult(kp=self._kp, ki=self._ki, loop=self.plan.loop,
                                   failure=f"incomplete recording ({n} slices, {len(events)}/{self.plan.n_events} events)")
        else:
            ct = W.CapturedTest(plan=self.plan, t=cap.t[:n].copy(), channels={k: v[:n].copy() for k, v in cap.data.items()},
                                event_times=events, kp=self._kp, ki=self._ki)
            res = W.analyze_test(ct, **self.analysis)
        self.test_finished.emit(self._cell, res)
        self._later(self.PAUSE_MS, self._advance)

    def _abort_test(self, reason: str):
        """The loop ran away: stop everything, record the cell as lost, restore the baseline."""
        self.message.emit(f"ABORT: {reason}")
        cell, kp, ki = self._cell, self._kp, self._ki
        res = W.StepTestResult(kp=kp, ki=ki, loop=self.plan.loop, failure=f"aborted: {reason}")
        self._finish(f"aborted: {reason}")
        self.test_finished.emit(cell, res)

    def _fail(self, reason: str):
        self.message.emit(f"ERROR: {reason}")
        self._finish(f"aborted: {reason}")


# ---------------------------------------------------------------------------
# helpers shared with the tests
# ---------------------------------------------------------------------------
_VALUE_RE = None


def _parse_event_value(label: str) -> Optional[float]:
    """The number after the last '=' of a Step Test marker label like 'Used Frequency (f0)=25000.5'."""
    global _VALUE_RE
    import re
    if _VALUE_RE is None:
        _VALUE_RE = re.compile(r"=\s*([-+0-9.eE]+)")
    m = _VALUE_RE.findall(label or "")
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None


def capture_from_scope(scope, kp: float, ki: float, bin_s: float = 0.0005) -> Tuple[Optional[W.CapturedTest], str]:
    """
    Turn the Scope tab's last capture (channels + Step Test event markers) into a :class:`CapturedTest`.

    Returns ``(captured, message)``; ``captured`` is None if it cannot be analysed (the message says why).
    The plan is reconstructed from the events: their values give the two levels, their spacing the hold.
    """
    if scope.last_data1 is None or scope.last_data2 is None or not scope.last_rate:
        return None, "The Scope tab has no capture yet."
    names = [scope.last_chan1, scope.last_chan2]
    det = W.detect_loop(names)
    if det.loop is None:
        return None, det.note
    markers = list(getattr(scope, "_event_markers", []) or [])
    if len(markers) < 3 or scope.capture_start_dt is None:
        return None, "No Step Test events are attached to this capture: run the Step Test with 'Trigger scope capture' on."
    times, values = [], []
    for dt, label in markers:
        times.append(scope.capture_start_dt.msecsTo(dt) / 1000.0)
        values.append(_parse_event_value(label))
    if any(v is None for v in values):
        return None, "Could not read the values from the event labels."
    order = np.argsort(times)
    times = [times[i] for i in order]
    values = [values[i] for i in order]
    lv = sorted(set(round(v, 9) for v in values))
    if len(lv) < 2:
        return None, "The events do not alternate between two levels."
    low, high = lv[0], lv[-1]
    n = len(times)
    hold = float(np.median(np.diff(times)))
    t_end = len(scope.last_data1) / scope.last_rate
    starts_high = abs(values[0] - low) < abs(values[0] - high)      # the first event goes to the low level
    try:
        if det.loop.relative_step:
            base, step = 0.5 * (low + high), (high - low) / (low + high)
        else:
            base, step = 0.5 * (low + high), 0.5 * (high - low)
        plan = W.StepTestPlan(loop=det.loop.key, base=base, step=step, hold_s=hold, n_events=n, lead_s=times[0],
                              tail_s=max(0.0, t_end - times[-1] - hold), start_high=starts_high)
    except ValueError as e:
        return None, f"The events do not form a valid test: {e}"
    t_full = np.arange(len(scope.last_data1)) / scope.last_rate
    ch: Dict[str, np.ndarray] = {}
    t_bins = None
    for name, arr in zip(names, (scope.last_data1, scope.last_data2)):
        t_bins, ch[name] = M.block_mean(t_full, np.asarray(arr, float), bin_s)
    return W.CapturedTest(plan=plan, t=t_bins, channels=ch, event_times=times, kp=kp, ki=ki), det.note


def format_result_html(res: W.StepTestResult, verdict: W.Verdict, suggestions: List[W.Suggestion], loop: W.LoopDef) -> str:
    """The text shown for a selected test."""
    r, c = res, verdict
    col = "#%02x%02x%02x" % CATEGORY_COLOR.get(c.category, (0, 0, 0))
    rows = [f"<b>Kp = {r.kp:.5g}, Ki = {r.ki:.5g}</b> &nbsp; "
            f"<span style='background:{col};padding:2px 6px'><b>{W.CATEGORY_LABEL.get(c.category, c.category)}</b></span>"]
    rows.append("<br>".join(c.reasons))
    p = r.primary
    if p is not None:
        rise = "n/a" if math.isnan(p.rise_time) else f"{p.rise_time * 1e3:.1f} ms"
        settle = "not settled in the hold" if math.isinf(p.settling_time) else f"{p.settling_time * 1e3:.0f} ms"
        rows.append(f"{loop.primary_channel}: rise {rise}, 5 % settling {settle}, overshoot {p.overshoot * 100:.1f} %, "
                    f"{p.n_extrema} ringing extrema; scatter {r.noise_rms:.3g}")
    if r.error is not None:
        d = "not decayed within the hold" if math.isinf(r.error.decay_time) else f"{r.error.decay_time * 1e3:.0f} ms"
        rows.append(f"Phase: peak {r.error.peak:.2f} deg, decays in {d}")
    if r.secondary is not None:
        rows.append(f"Drive: overshoot {r.secondary.overshoot * 100:.0f} % of its final change; "
                    f"post-settling noise {r.secondary.noise_rms:.3g}")
    if r.n_steps:
        rows.append(f"{r.n_steps} events averaged, window {r.window_s * 1e3:.0f} ms, response onset lag {r.latency_s * 1e3:.1f} ms")
    if r.model is not None:
        m = r.model
        pr = m.predict(r.kp, r.ki)
        rows.append(f"<i>Identified loop</i>: Kp {m.scale_p * r.kp:.4g}, Ki {m.scale_i * r.ki:.4g} (physical units), "
                    f"sensor pole {m.gamma:.3g} 1/s, latency {m.delay_s * 1e3:.1f} ms, crossover {pr.crossover_rad_s / (2 * math.pi):.1f} Hz, "
                    f"phase margin {pr.phase_margin_deg:.0f} deg")
    for w in r.warnings:
        rows.append(f"<span style='color:#a06000'>Note: {w}</span>")
    if suggestions:
        rows.append("<b>Suggestions</b>")
        for s in suggestions:
            pair = "" if s.kp is None else f" &rarr; Kp {s.kp:.4g}, Ki {s.ki:.4g}"
            rows.append(f"&bull; {s.why}{pair}")
    return "<br>".join(rows)


# (category, what the response looks like, what to change) - kept in line with W.advise()
CHECKLISTS = {
    "pll": ("Tip retracted / far from the surface", "The loop under test is running and locked",
            "PLL: DNC Lockin Options > Acquire > Auto 0 deg is DISABLED", "Baseline Kp/Ki above match what is set in SXM"),
    "afl": ("Tip retracted / far from the surface", "PLL off (Kp = Ki = 0) and DNC 'use' = the free resonance f_res",
            "Amplitude feedback is ON: QPlusAmpl sits at Ref and Drive has settled",
            "Baseline Kp/Ki, Ref and the output gain above match what is set in SXM"),
}

GUIDE_VERDICTS = (
    ("good", "Fast enough, overshoot within the limit, no ringing.",
     "Keep it. If it is much faster than needed, lower both gains a little: less noise, same shape."),
    ("too_slow", "Clean response, but the rise takes longer than the target.",
     "Raise Kp and Ki together by the same factor. The Ki:Kp ratio stays, so the shape stays."),
    ("overshoot", "Goes past the new value by more than the limit, then comes back.",
     "Lower Ki, keep Kp."),
    ("ringing", "Oscillates around the new value, or never settles (large scatter between repeated steps).",
     "Lower Ki first. If it persists, lower Kp too."),
    ("slow_tail", "Gets there quickly, but the Phase error (PLL) or the last few percent linger.",
     "Raise Ki, keep Kp: the integral term removes what Kp leaves behind."),
    ("lost", "The loop lost lock, ran away, or the step could not be measured.",
     "The run stops and the baseline is restored. Go back to the last good pair, or halve both gains."),
)


def guide_html() -> str:
    """Static help for the tab: what it does, how to use it, what the numbers and verdicts mean."""
    def rgb(cat):
        return "#%02x%02x%02x" % CATEGORY_COLOR[cat]

    rows = "".join(
        f"<tr><td bgcolor='{rgb(cat)}'><b>{W.CATEGORY_LABEL[cat]}</b></td><td>{seen}</td><td>{todo}</td></tr>"
        for cat, seen, todo in GUIDE_VERDICTS)
    return f"""
<h3>What this tab does</h3>
<p>It nudges a setpoint back and forth (PLL: <i>DNC use</i> by &plusmn;step Hz; amplitude loop: <i>Ref</i> by &plusmn;step %),
records how the loop follows it through the driver, measures the response, gives it a verdict and suggests new Kp / Ki.
Only Kp, Ki and the stepped parameter are ever written. They are put back to the baseline when a run ends, is stopped
or is aborted.</p>

<h3>How to use it</h3>
<ol>
<li><b>Test</b> (left, 1): pick the loop. The defaults follow the manual. Every number field accepts scientific notation
(type <code>2.5e8</code>) and shows large or small values that way.</li>
<li><b>Baseline</b> (left, 2): type the Kp / Ki that are set in SXM right now. SXM cannot be read back, so this is
what every change is measured against and what is restored afterwards.</li>
<li><b>Target</b> (left, 3): your scan (line time, pixels) or a response time. It sets the limits used for the verdict.</li>
<li><b>Before running</b> (left, 5): tick every item. Only then are the run buttons enabled.</li>
<li><b>Run single test (baseline)</b>: measures the loop as it is now. The verdict, the measured times and the suggested
new gains appear in the results panel and the plots.</li>
<li><b>Test suggested pair</b>: re-measures with the suggested gains. Repeat until the verdict is <i>good</i>.</li>
<li><b>Narrow search</b> (on the Speed &times; Shape map): one button, one confirmed round - fills the map if it
isn't already, zooms a finer grid onto the best cell, fills that too, then reports what changed and stops.
Click again for another round, until it reports the pair has converged.</li>
<li><b>Stage in Params tab</b>: hands the chosen pair to the Parameters tab. Apply it there.</li>
</ol>

<h3>What is measured</h3>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><td><b>Rise time (10-90 %)</b></td><td>How fast the main channel (df for the PLL, QPlusAmpl for the amplitude loop) follows the step.
Imaging target: at most half a pixel dwell (line time / pixels).</td></tr>
<tr><td><b>5 % settling</b></td><td>Time until it stays within 5 % of the new value.</td></tr>
<tr><td><b>Overshoot</b></td><td>How far it goes past the new value, in % of the step.</td></tr>
<tr><td><b>Ringing extrema</b></td><td>Number of wiggles after the step.</td></tr>
<tr><td><b>Phase tail</b> (PLL)</td><td>The Phase error is what the loop is still correcting. It should have decayed within 10 % of a line time.</td></tr>
<tr><td><b>Scatter</b></td><td>How much repeated steps differ from each other: noise, or a sustained oscillation.</td></tr>
</table>

<h3>Verdicts and what to change</h3>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Verdict</th><th>What you see</th><th>What to change</th></tr>
{rows}
</table>

<h3>Rules of thumb</h3>
<ul>
<li>Kp does the fast tracking; Ki removes what is left (the Phase tail).</li>
<li>The map has two independent axes: <b>speed</b> (Kp and Ki raised or lowered <b>together</b>, same ratio - faster
or slower) and <b>shape</b> (the Ki:Kp <b>ratio</b>, shifted from the baseline's - how it gets there).</li>
<li>A faster loop is a noisier loop: stop at the slowest gains that still meet the target.</li>
<li>Kp and Ki are SXM's raw units. Nothing here assumes what a value means: gains are judged only from measured responses.</li>
</ul>

<h3>Amplitude loop: gains that span decades</h3>
<p>The manual's start is Ki &asymp; 5&middot;10<sup>8</sup>/Q and Kp &asymp; 10<sup>4</sup>&middot;Ki at DNC <i>Output Gain</i> &plusmn;1 V, and both
are &times;10 for each range lower (&plusmn;1 V &rarr; &plusmn;0.1 V). So the same sensor needs 10&times; larger values at &plusmn;0.1 V, and
different sensors differ by decades through Q. Steps of &times;2 cannot cover that.</p>
<ol>
<li>Choose the <b>output gain you really use</b> (left, 2) and press <i>Use the manual's start values</i>. It fills Kp, Ki and Tau from
Q, f0 and the gain, and the hold / settle times from the ring-down time Q/(&pi;f0) (this app's rule, not the manual's). Set the same values in SXM.</li>
<li><b>Run scale scan</b>: tests the baseline, then both gains together in steps of the step factor (default &times;10, three down and three up).
Ki:Kp stays; only the speed changes. It stops going further once a test loses the loop.</li>
<li>Read the scan on the map: grey = still too slow, green = the right decade, orange / red = too high.</li>
<li>Click the best cell and <i>Suggest / refine zoom</i> (or press <i>Narrow search</i> to fill-and-zoom in one click): a finer
3&times;3 map in steps of &radic;(factor), which now also varies the shape. Repeat once more, then use single tests.</li>
<li>Check the <b>Drive</b> line of each result: the manual asks you to avoid a strong (saturating) overshoot there, and Kp amplifies its noise.</li>
</ol>
<p><i>Max gain change vs baseline</i> (left, 4) limits how far from your baseline any test may go (default &times;1000 for this loop, &times;16 for the PLL).
If the baseline warning above the results appears, check first that the output gain here matches the DNC window.</p>

<h3>The Speed &times; Shape map (optional)</h3>
<p>Runs one test per cell of a log-spaced grid around the baseline. The <b>vertical</b> axis is speed: Kp and Ki raised or
lowered together (same ratio) - up is faster, down slower. The <b>horizontal</b> axis is shape: the Ki:Kp ratio shifted
from the baseline's - right is more integral-heavy (higher Ki:Kp), left more proportional-heavy. The crosshair marks the
baseline row and column. Cell colours are the verdict colours above; grey = skipped (more aggressive than a cell that
lost the loop), pale = not tested yet. Click a cell to see its response and advice; <i>Suggest / refine zoom</i> maps the
neighbourhood of a good cell in finer steps by hand, and <i>Narrow search</i> does the fill-and-zoom automatically, one
confirmed round of hardware writes at a time, until it reports the pair has converged.</p>
"""


# ---------------------------------------------------------------------------
# the tab
# ---------------------------------------------------------------------------
class TuningTab(QtWidgets.QWidget):
    """Guided PLL / amplitude-loop tuning: define, run, analyse, screen a Speed x Shape map, refine."""

    LEAD_S = 1.0            # settled recording before the first event
    TAIL_S = 0.5            # recording after the last event's hold

    def __init__(self, dde, driver, scope_tab=None, params_tab=None, parent=None):
        super().__init__(parent)
        self.dde, self.driver = dde, driver
        self.scope_tab, self.params_tab = scope_tab, params_tab
        self.runner: Optional[TuningRunner] = None
        self.maps: List[W.ScreeningMap] = []
        self.singles: List[W.StepTestResult] = []
        self.selected: Optional[Tuple[int, int]] = None
        self._single_result: Optional[W.StepTestResult] = None
        self._map_texts: list = []
        self._confirmed_baseline: Dict[str, Tuple[float, float]] = {}     # per loop, this session only
        self._sweep_gs: List[float] = []
        self._sweep_results: List[Optional[W.StepTestResult]] = []
        self._sweep_active = False
        self._build()
        self._on_loop_changed()
        self._paint_map()          # the clean empty state, not pyqtgraph's raw default axis
        self._update_enabled()

    # ------------------------------------------------------------------ construction
    def _spin(self, lo, hi, val, dec=3, step=None, suffix="", sci_above=1e6, plain=False):
        """
        Number field: accepts scientific notation ('2.5e8') and shows large / small values that way.
        ``plain=True`` keeps fixed decimals and never switches (frequencies: f0 = 25132.457 Hz).
        """
        s = SciDoubleSpinBox(sci_above=sci_above, plain=plain)
        s.setDecimals(dec)
        s.setRange(lo, hi)
        s.setValue(val)
        if step:
            s.setSingleStep(step)
        if suffix:
            s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        return s

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # pinned bar: live "what am I about to do" context, and the Stop control, never scrolled away.
        pinned = QtWidgets.QFrame()
        pinned.setFrameShape(QtWidgets.QFrame.StyledPanel)
        pb = QtWidgets.QHBoxLayout(pinned)
        pb.setContentsMargins(8, 4, 8, 4)
        self.context_label = QtWidgets.QLabel()
        self.context_label.setWordWrap(True)
        self.context_label.setTextFormat(QtCore.Qt.RichText)
        pb.addWidget(self.context_label, 1)
        self.btn_stop = QtWidgets.QPushButton("Stop && restore baseline")
        self.btn_stop.setToolTip("Aborts the running test and immediately writes the baseline Kp/Ki back. "
                                 "Shortcuts: Esc, Ctrl+.")
        self.btn_stop.clicked.connect(self.stop)
        pb.addWidget(self.btn_stop)
        outer.addWidget(pinned)
        self._stop_shortcuts = []
        for seq in ("Esc", "Ctrl+."):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(self.stop)
            self._stop_shortcuts.append(sc)

        root = QtWidgets.QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(root, 1)
        self.main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_split.setChildrenCollapsible(False)     # the left panel can be resized, never dragged to hidden
        root.addWidget(self.main_split)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(260)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        left_scroll.setWidget(left)
        self.main_split.addWidget(left_scroll)

        # 1. test
        g = QtWidgets.QGroupBox("1. Test")
        f = QtWidgets.QFormLayout(g)
        self.loop_combo = QtWidgets.QComboBox()
        self.loop_combo.addItem(W.PLL.name, "pll")
        self.loop_combo.addItem(W.AFL.name, "afl")
        self.loop_combo.currentIndexChanged.connect(self._on_loop_changed)
        f.addRow("Loop:", self.loop_combo)
        self.channels_label = QtWidgets.QLabel()
        self.channels_label.setWordWrap(True)
        f.addRow("Records:", self.channels_label)
        self.base_spin = self._spin(-1e12, 1e12, 25000.0, 4, plain=True)     # `use` frequency in Hz (or Ref): fixed decimals
        self.base_spin.valueChanged.connect(lambda *_: self._update_hint())  # the AFL voltage warning follows it
        self.base_label = QtWidgets.QLabel()
        f.addRow(self.base_label, self.base_spin)
        self.step_spin = self._spin(0.001, 1e6, 1.0, 3)
        self.step_label = QtWidgets.QLabel()
        f.addRow(self.step_label, self.step_spin)
        self.hold_spin = self._spin(0.15, 10.0, 0.5, 2, 0.05, " s")
        f.addRow("Hold per level:", self.hold_spin)
        self.events_spin = QtWidgets.QSpinBox()
        self.events_spin.setRange(3, 15)
        self.events_spin.setValue(7)
        f.addRow("Events:", self.events_spin)
        self.settle_spin = self._spin(0.2, 30.0, 2.0, 1, 0.5, " s")
        f.addRow("Settle before recording:", self.settle_spin)
        self.li_spin = self._spin(0.05, 200.0, 2.0, 2, 0.5, " ms")
        self.li_spin.setToolTip("DNC TimeConstant (12 dB/oct). Used to identify the physical loop; set 0 to skip.")
        f.addRow("Lock-in TimeConstant:", self.li_spin)
        self.tau_spin = self._spin(1.0, 2000.0, 10.0, 1, 1.0, " ms")
        self.tau_label = QtWidgets.QLabel("Amplitude Tau:")
        f.addRow(self.tau_label, self.tau_spin)
        self.f0_spin = self._spin(1.0, 1e7, 25000.0, 3, 100.0, " Hz", plain=True)     # Hz with mHz resolution, e.g. 25132.457
        self.q_spin = self._spin(1.0, 1e7, 25000.0, 0, 1000.0)
        f.addRow("f0 (from the sweep):", self.f0_spin)
        f.addRow("Q (from the sweep):", self.q_spin)
        lv.addWidget(g)

        # 2. baseline
        g = QtWidgets.QGroupBox("2. Baseline (currently set in SXM - it cannot be read back)")
        f = QtWidgets.QFormLayout(g)
        self.kp_spin = self._spin(-1e12, 1e12, -100.0, 4, sci_above=1e4)      # raw gains span decades: 1e4 and up in e-notation
        self.ki_spin = self._spin(-1e12, 1e12, -1e4, 4, sci_above=1e4)
        f.addRow("Kp:", self.kp_spin)
        f.addRow("Ki:", self.ki_spin)
        # amplitude loop only: the gains scale with the DNC Output Gain, and span decades with Q
        self.gain_label = QtWidgets.QLabel("AFL output gain (DNC):")
        self.gain_combo = QtWidgets.QComboBox()
        for v in W.AFL_OUTPUT_GAINS:
            self.gain_combo.addItem(f"+-{v:g} V", v)
        self.gain_combo.setCurrentIndex(W.AFL_OUTPUT_GAINS.index(1.0))
        self.gain_combo.setToolTip("DNC window > Output Gain, as set in SXM. Kp and Ki scale with it: x10 for each range lower "
                                   "(manual: +-1 V -> +-0.1 V). +-10 V (/10) is the same rule extrapolated.")
        f.addRow(self.gain_label, self.gain_combo)
        self.start_label = QtWidgets.QLabel()
        self.start_label.setWordWrap(True)
        f.addRow(self.start_label)
        self.btn_fill = QtWidgets.QPushButton("Use the manual's start values")
        self.btn_fill.setToolTip("Fills Kp, Ki and Tau from Q, f0 and the output gain (manual), and the hold / settle times "
                                 "from the ring-down time (this app's rule of thumb).")
        f.addRow(self.btn_fill)
        self._afl_rows = (self.gain_label, self.gain_combo, self.start_label, self.btn_fill)
        for w in (self.q_spin, self.f0_spin, self.gain_combo):
            (w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox) else w.valueChanged).connect(self._update_start_label)
        self.btn_fill.clicked.connect(lambda: self._fill_afl_start())
        for w in (self.kp_spin, self.ki_spin):
            w.valueChanged.connect(lambda *_: self._update_hint())          # the baseline warning follows what is typed
            w.valueChanged.connect(self._reset_checklist)                   # a different baseline needs re-verifying
        lv.addWidget(g)

        # 3. target
        self.target_section = CollapsibleSection("3. Target", expanded=False)
        f = QtWidgets.QFormLayout()
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItems(["Imaging scan", "Response time"])
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        f.addRow("Based on:", self.target_combo)
        self.line_spin = self._spin(0.1, 600.0, 6.0, 1, 1.0, " s")
        self.px_spin = QtWidgets.QSpinBox()
        self.px_spin.setRange(16, 4096)
        self.px_spin.setValue(256)
        self.rise_spin = self._spin(0.5, 5000.0, 50.0, 1, 5.0, " ms")
        self.os_spin = self._spin(1.0, 50.0, 10.0, 0, 1.0, " %")
        self.line_label, self.px_label, self.rise_label = QtWidgets.QLabel("Line time:"), QtWidgets.QLabel("Pixels:"), QtWidgets.QLabel("Rise time (10-90 %) <=")
        f.addRow(self.line_label, self.line_spin)
        f.addRow(self.px_label, self.px_spin)
        f.addRow(self.rise_label, self.rise_spin)
        f.addRow("Max overshoot:", self.os_spin)
        self.target_note = QtWidgets.QLabel()
        self.target_note.setWordWrap(True)
        f.addRow(self.target_note)
        self.target_section.set_content_layout(f)
        lv.addWidget(self.target_section)
        for w in (self.line_spin, self.px_spin, self.rise_spin, self.os_spin):
            w.valueChanged.connect(self._on_target_changed)

        # 4. grid
        self.search_section = CollapsibleSection("4. Search span", expanded=False)
        f = QtWidgets.QFormLayout()
        self.factor_spin = self._spin(1.2, 10.0, 2.0, 2, 0.1, " x")
        self.factor_spin.setToolTip("How far apart each step is. x2 for a fine PLL search; x10 (a decade) "
                                    "for the amplitude loop, whose gains span many orders of magnitude.")
        self.scan_lo = self._spin(0, 6, 2, 0)
        self.scan_hi = self._spin(0, 6, 2, 0)
        self.speed_lo = self._spin(0, 4, 2, 0)
        self.speed_hi = self._spin(0, 4, 2, 0)
        self.shape_lo = self._spin(0, 5, 3, 0)
        self.shape_hi = self._spin(0, 5, 2, 0)
        self.range_spin = self._spin(2.0, 1e6, 16.0, 0, 10.0, " x")
        self.range_spin.setToolTip("No test is written with a Kp or Ki more than this factor above or below the baseline "
                                   "(map, scale scan and suggested pairs).")
        f.addRow("Step factor:", self.factor_spin)
        f.addRow("Scale scan steps down / up (speed):", self._pair(self.scan_lo, self.scan_hi))
        f.addRow("Map: Speed steps down / up (slower / faster):", self._pair(self.speed_lo, self.speed_hi))
        f.addRow("Map: Shape steps down / up (more P / more I):", self._pair(self.shape_lo, self.shape_hi))
        f.addRow("Max gain change vs baseline:", self.range_spin)
        self.est_label = QtWidgets.QLabel()
        self.est_label.setWordWrap(True)
        f.addRow(self.est_label)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #444;")
        self.preview_label.setToolTip("The actual Kp / Ki values a map with these settings would write to SXM.")
        f.addRow(self.preview_label)
        # amplitude loop only: the guided scale-sweep protocol (non-uniform, hand-picked multipliers -
        # the uniform step-factor scan above cannot express this, so it gets its own list).
        self.scale_list_edit = QtWidgets.QLineEdit("0.1, 0.3, 1, 2, 3")
        self.scale_list_edit.setToolTip("Comma-separated multipliers of the baseline Kp/Ki (Ki:Kp stays fixed), "
                                        "tested in order; the sweep stops at the first one that saturates Drive, "
                                        "overshoots/rings beyond the target, or raises Drive noise substantially.")
        self.scale_list_label = QtWidgets.QLabel("Scale sweep list (g):")
        f.addRow(self.scale_list_label, self.scale_list_edit)
        self.search_section.set_content_layout(f)
        lv.addWidget(self.search_section)
        for w in (self.factor_spin, self.scan_lo, self.scan_hi, self.speed_lo, self.speed_hi, self.shape_lo, self.shape_hi,
                  self.range_spin, self.hold_spin, self.settle_spin, self.events_spin, self.step_spin):
            w.valueChanged.connect(self._update_estimate)

        # 5. checklist + buttons
        g = QtWidgets.QGroupBox("5. Before running")
        v = QtWidgets.QVBoxLayout(g)
        self.checks = []
        for text in CHECKLISTS["pll"]:                      # the texts follow the loop, see _on_loop_changed
            c = QtWidgets.QCheckBox(text)
            c.toggled.connect(self._update_enabled)
            v.addWidget(c)
            self.checks.append(c)
        self.btn_single = QtWidgets.QPushButton("Run single test (baseline)")
        self.btn_scan = QtWidgets.QPushButton("Run scale scan (keep Ki:Kp)")
        self.btn_scan.setToolTip("Tests the baseline, then both gains scaled together in steps of the step factor: Ki:Kp is "
                                 "kept and only the speed changes. Use it first when the right decade of gain is unknown.")
        self.btn_map = QtWidgets.QPushButton("Run / continue map")
        self.btn_sweep = QtWidgets.QPushButton("Run scale sweep (protocol)")
        self.btn_sweep.setToolTip("Steps through the scale list above (Ki:Kp fixed), measuring the full battery "
                                  "for each: rise/fall, 2 % settling, over/undershoot, steady-state error, Drive "
                                  "peak/RMS excursion and noise. Stops at the first sign of trouble and "
                                  "recommends a final scale. Results appear in the 'Scale sweep' tab.")
        self.btn_clear = QtWidgets.QPushButton("Clear map")
        self.btn_scope = QtWidgets.QPushButton("Analyze Scope capture")
        self.btn_scope.setToolTip("Analyze the step train recorded in the Scope tab (needs the Step Test events on it).")
        for b in (self.btn_single, self.btn_scan, self.btn_map, self.btn_sweep, self.btn_clear, self.btn_scope):
            v.addWidget(b)
        self.btn_single.clicked.connect(self._run_single)
        self.btn_scan.clicked.connect(self._run_scan)
        self.btn_map.clicked.connect(self._run_map)
        self.btn_sweep.clicked.connect(self._run_scale_sweep)
        self.btn_clear.clicked.connect(self._clear_map)
        self.btn_scope.clicked.connect(self._analyze_scope)
        lv.addWidget(g)
        lv.addStretch(1)

        # right side: [map | advice] over [selected-test plots | all tests | log]
        right_box = QtWidgets.QWidget()
        rb = QtWidgets.QVBoxLayout(right_box)
        rb.setContentsMargins(0, 0, 0, 0)
        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setTextFormat(QtCore.Qt.RichText)
        self.hint_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.hint_label.setMargin(6)
        rb.addWidget(self.hint_label)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setChildrenCollapsible(False)
        rb.addWidget(right, 1)
        self.main_split.addWidget(right_box)
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.setChildrenCollapsible(False)
        right.addWidget(top)

        self.map_plot = pg.PlotWidget(title="Speed x Shape map: click a cell")
        self.map_plot.setBackground("w")
        self.map_plot.setMouseEnabled(False, False)
        self.map_plot.setLabel("bottom", "Shape  (Ki:Kp ratio vs baseline)")
        self.map_plot.setLabel("left", "Speed  (Kp vs baseline)")
        self.map_plot.setMinimumSize(260, 200)
        self.map_img = pg.ImageItem()
        self.map_plot.addItem(self.map_img)
        self.map_marks = pg.PlotDataItem()
        self.map_plot.addItem(self.map_marks)
        self.map_plot.scene().sigMouseClicked.connect(self._on_map_click)
        top.addWidget(self.map_plot)

        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        self.detail_text = QtWidgets.QTextBrowser()
        self.detail_text.setOpenLinks(False)
        sv.addWidget(self.detail_text, 1)
        grid = QtWidgets.QGridLayout()
        self.btn_test_sug = QtWidgets.QPushButton("Test suggested pair")
        self.btn_apply = QtWidgets.QPushButton("Stage in Params tab")
        self.btn_zoom = QtWidgets.QPushButton("Suggest / refine zoom")
        self.btn_back = QtWidgets.QPushButton("Back to coarse map")
        self.btn_prior = QtWidgets.QPushButton("Predict map (model)")
        self.btn_narrow = QtWidgets.QPushButton("Narrow search")
        self.btn_narrow.setToolTip("Fills the current map (if needed), zooms a finer grid onto its best cell and fills "
                                   "that too - one button, one confirmed round of hardware writes. Click again to "
                                   "narrow further.")
        for n, b in enumerate((self.btn_test_sug, self.btn_apply, self.btn_zoom, self.btn_back, self.btn_prior, self.btn_narrow)):
            b.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # never force the panel wider
            grid.addWidget(b, n // 2, n % 2)
        sv.addLayout(grid)
        side.setMinimumWidth(220)
        top.addWidget(side)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)

        self.detail_tabs = QtWidgets.QTabWidget()
        right.addWidget(self.detail_tabs)
        plots = QtWidgets.QWidget()
        ph = QtWidgets.QHBoxLayout(plots)
        ph.setContentsMargins(0, 0, 0, 0)
        self.plot1 = pg.PlotWidget()
        self.plot2 = pg.PlotWidget()
        for p in (self.plot1, self.plot2):
            p.setBackground("w")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMinimumWidth(200)
            ph.addWidget(p, 1)
        self.plot2.setXLink(self.plot1)
        self.detail_tabs.addTab(plots, "Selected test")
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Kp", "Ki", "Verdict", "Rise ms", "Overshoot %", "Phase decay ms", "Scatter"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.detail_tabs.addTab(self.table, "All tests")

        self.sweep_tab = QtWidgets.QWidget()
        sv2 = QtWidgets.QVBoxLayout(self.sweep_tab)
        sv2.setContentsMargins(0, 0, 0, 0)
        self.sweep_table = QtWidgets.QTableWidget(0, 12)
        self.sweep_table.setHorizontalHeaderLabels(
            ["g", "Kp", "Ki", "Tier", "Rise up ms", "Rise down ms", "2% settle ms", "Overshoot %",
             "SS error %", "Drive peak", "Drive RMS exc.", "Drive noise"])
        self.sweep_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        sv2.addWidget(self.sweep_table, 1)
        self.sweep_summary = QtWidgets.QLabel(
            "Run a scale sweep (amplitude loop, left section 4/5) to see results here.")
        self.sweep_summary.setWordWrap(True)
        self.sweep_summary.setTextFormat(QtCore.Qt.RichText)
        self.sweep_summary.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.sweep_summary.setMargin(6)
        sv2.addWidget(self.sweep_summary)
        self.detail_tabs.addTab(self.sweep_tab, "Scale sweep")

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.detail_tabs.addTab(self.log, "Log")
        self.guide = QtWidgets.QTextBrowser()
        self.guide.setHtml(guide_html())
        self.detail_tabs.addTab(self.guide, "Guide")
        self.detail_tabs.setCurrentWidget(self.guide)         # what a first-time user needs; the first result switches to the plots
        self.detail_text.setHtml("<b>No test yet.</b><br>The verdict, the measured times and the suggested new "
                                 "Kp / Ki appear here after a test. The <b>Guide</b> tab below explains the steps "
                                 "and what each verdict means.")
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        self.btn_apply.clicked.connect(self._stage_selected)
        self.btn_zoom.clicked.connect(self._zoom)
        self.btn_back.clicked.connect(self._back)
        self.btn_prior.clicked.connect(self._predict_prior)
        self.btn_test_sug.clicked.connect(self._test_suggestion)
        self.btn_narrow.clicked.connect(self._narrow_search)
        self._suggestions: List[W.Suggestion] = []
        self._narrow_active = False
        self._narrow_stage: Optional[str] = None
        self._narrow_round = 0
        self._narrow_converged = False

        # left/right split: sane default, then the user's remembered width and section states (if any)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setSizes([360, 900])
        layout_state = _load_tuning_layout()
        self.target_section.set_expanded(bool(layout_state.get("target_expanded", False)))
        self.search_section.set_expanded(bool(layout_state.get("search_expanded", False)))
        if "left_width" in layout_state:
            w = int(layout_state["left_width"])
            QtCore.QTimer.singleShot(0, lambda: self.main_split.setSizes([w, max(500, self.width() - w)]))
        self.main_split.splitterMoved.connect(self._save_layout_state)
        self.target_section.toggled.connect(self._save_layout_state)
        self.search_section.toggled.connect(self._save_layout_state)

    def _save_layout_state(self, *_):
        sizes = self.main_split.sizes()
        _write_tuning_layout({
            "left_width": sizes[0] if sizes else 360,
            "target_expanded": self.target_section.is_expanded(),
            "search_expanded": self.search_section.is_expanded(),
        })

    @staticmethod
    def _pair(a, b):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    # ------------------------------------------------------------------ settings -> objects
    @property
    def loop_def(self) -> W.LoopDef:
        return W.LOOPS[self.loop_combo.currentData()]

    def _on_loop_changed(self):
        ld = self.loop_def
        pll = ld.key == "pll"
        self.channels_label.setText(f"{ld.u_channel} + {ld.y_channel}  (the test steps {'DNC use' if pll else 'Amplitude Ref'})")
        self.base_label.setText("Current `use` frequency:" if pll else "Current amplitude Ref:")
        self.step_label.setText("Step +- (Hz):" if pll else "Step +- (% of Ref):")
        self.base_spin.setValue(25000.0 if pll else 6.0)
        self.step_spin.setValue(1.0 if pll else 10.0)
        self.hold_spin.setValue(0.5 if pll else 1.0)
        if pll:
            self.kp_spin.setValue(-100.0)
            self.ki_spin.setValue(-1e4)
        else:
            self._fill_afl_start(log=False)        # the manual's values for the Q, f0 and output gain on screen
        # the amplitude loop is often decades off the first guess: coarse (x10) steps, wide allowed range
        self.factor_spin.setValue(2.0 if pll else 10.0)
        self.scan_lo.setValue(2 if pll else 3)
        self.scan_hi.setValue(2 if pll else 3)
        self.range_spin.setValue(16.0 if pll else 1000.0)
        for w in (self.tau_spin, self.tau_label, self.scale_list_label, self.scale_list_edit,
                 self.btn_sweep) + self._afl_rows:
            w.setVisible(not pll)
        for c, text in zip(self.checks, CHECKLISTS[ld.key]):
            c.setText(text)
            c.setChecked(False)                      # a different loop needs a different set of checks
        self._update_start_label()
        self.target_combo.setCurrentIndex(0 if pll else 1)
        self.rise_spin.setValue(50.0 if pll else 500.0)
        self._on_target_changed()
        self._update_estimate()
        self._update_enabled()

    def _afl_start(self) -> W.AflStart:
        return W.afl_start_values(self.q_spin.value(), self.f0_spin.value(), self.gain_combo.currentData())

    def _update_start_label(self, *_):
        try:
            s = self._afl_start()
        except ValueError:
            self.start_label.setText("Set Q and f0 (from the sweep) to get the manual's start values.")
            return
        self.start_label.setText(f"Manual start for Q={self.q_spin.value():.0f}, f0={self.f0_spin.value():.10g} Hz, "
                                 f"+-{self.gain_combo.currentData():g} V:<br>Kp = {format_number(s.kp, 4, 1e4)}, "
                                 f"Ki = {format_number(s.ki, 4, 1e4)}, "
                                 f"Tau = {s.tau_s * 1e3:.3g} ms<br>Ring-down Q/(pi f0) = {s.ring_down_s:.2f} s "
                                 f"-> hold {s.hold_s:g} s, settle {s.settle_s:g} s")
        self._update_hint()

    def _fill_afl_start(self, log: bool = True):
        """Baseline gains, Tau and test timing from the manual's rules for the Q, f0 and output gain on screen."""
        try:
            s = self._afl_start()
        except ValueError:
            return
        self.kp_spin.setValue(s.kp)
        self.ki_spin.setValue(s.ki)
        self.tau_spin.setValue(s.tau_s * 1e3)
        self.hold_spin.setValue(s.hold_s)
        self.settle_spin.setValue(s.settle_s)
        if log:
            self._log(f"Baseline set to the manual's start for +-{self.gain_combo.currentData():g} V: Kp={s.kp:.4g}, Ki={s.ki:.4g}, "
                      f"Tau={s.tau_s * 1e3:.3g} ms, hold {s.hold_s:g} s, settle {s.settle_s:g} s. Set the same values in SXM.")

    def _baseline_warning(self) -> str:
        """Amplitude loop: flag a baseline far from the manual's start (typically a wrong output-gain decade)."""
        if self.loop_def.key != "afl":
            return ""
        try:
            s = self._afl_start()
            ratio = self.ki_spin.value() / s.ki
        except (ValueError, ZeroDivisionError):
            return ""
        if 1 / 30 <= ratio <= 30:
            return ""
        return (f"<br><span style='color:#a06000'><b>Check:</b> the baseline Ki is x{ratio:.3g} of the manual's start for this "
                f"Q, f0 and output gain. Is the output gain selected here the one set in the DNC window? "
                f"(It may be right for your sensor: a Scale scan will tell.)</span>")

    def _confirmed_baseline_warning(self) -> str:
        """Either loop: flag that the typed Kp/Ki no longer matches what was last confirmed (this or a
        previous session) - a reminder to re-check SXM before the next run re-confirms it."""
        last = self._load_confirmed_baseline(self.loop_def.key)
        if last is None:
            return ""
        kp, ki = self.kp_spin.value(), self.ki_spin.value()
        if math.isclose(last[0], kp, rel_tol=1e-9) and math.isclose(last[1], ki, rel_tol=1e-9):
            return ""
        return (f"<br><span style='color:#a06000'><b>Note:</b> this differs from the baseline last confirmed for this "
               f"loop (Kp={last[0]:.4g}, Ki={last[1]:.4g}). The next run will ask you to confirm the new one.</span>")

    def _voltage_warning(self) -> str:
        """
        Amplitude loop only: its stepped parameter is Ref, the same voltage-guarded value as
        common.PARAMS_BASE's 'amp_ref' elsewhere in the app. Rather than a permanent app-wide banner,
        warn here, inline, only when the levels this test would actually write exceed the limit.
        """
        if self.loop_def.key != "afl":
            return ""
        base, step_frac = self.base_spin.value(), self.step_spin.value() / 100.0
        high, low = base * (1 + step_frac), base * (1 - step_frac)
        worst = max(abs(high), abs(low))
        if worst <= VOLTAGE_LIMIT_ABS:
            return ""
        return (f"<br><span style='color:#a06000'><b>Check:</b> stepping Ref to {high:.4g} / {low:.4g} exceeds "
               f"&plusmn;{VOLTAGE_LIMIT_ABS:g} V. Do not proceed without a divider/attenuator.</span>")

    def _on_target_changed(self, *_):
        scan = self.target_combo.currentIndex() == 0
        for w in (self.line_spin, self.px_spin, self.line_label, self.px_label):
            w.setVisible(scan)
        for w in (self.rise_spin, self.rise_label):
            w.setVisible(not scan)
        self._target_note()
        self._reset_checklist()

    def _update_context_label(self):
        """The pinned bar's live 'what am I about to do' line: never scrolled out of view."""
        ld = self.loop_def
        conn = "OFFLINE" if not self._online() else "ONLINE"
        kp, ki = format_number(self.kp_spin.value(), 4, 1e4), format_number(self.ki_spin.value(), 4, 1e4)
        running = self.runner_active()
        state = " &middot; <b>RUNNING</b>" if running else ""
        self.context_label.setText(
            f"Tuning: {ld.key.upper()} ({ld.primary_channel}) &middot; Baseline Kp {kp} / Ki {ki} &middot; {conn}{state}")

    def _reset_checklist(self, *_):
        """
        Untick every checklist item: the loop, baseline or target changed (or a run just finished), so a
        previous tick no longer vouches for the current state and the user should re-verify each item.
        None of these are currently driver-verifiable (no lock-state channel exists to tick one automatically).
        """
        for c in self.checks:
            c.setChecked(False)

    def _target_note(self):
        t = self.target()
        txt = f"Rise <= {t.rise_max * 1e3:.1f} ms"
        if t.decay_max:
            txt += f", Phase tail gone within {t.decay_max * 1e3:.0f} ms"
        self.target_note.setText(txt)
        self.target_section.set_summary(txt)          # same real numbers, shown on the collapsed header too

    def target(self) -> W.Target:
        os_max = self.os_spin.value() / 100.0
        if self.target_combo.currentIndex() == 0:
            t = W.Target.from_scan(self.line_spin.value(), self.px_spin.value())
        else:
            t = W.Target.manual(self.rise_spin.value() / 1000.0)
        return dataclasses.replace(t, overshoot_max=os_max)

    def plan(self) -> W.StepTestPlan:
        ld = self.loop_def
        step = self.step_spin.value() / 100.0 if ld.relative_step else self.step_spin.value()
        return W.StepTestPlan(loop=ld.key, base=self.base_spin.value(), step=step, hold_s=self.hold_spin.value(),
                              n_events=self.events_spin.value(), settle_s=self.settle_spin.value(),
                              lead_s=self.LEAD_S, tail_s=self.TAIL_S)

    def grid(self) -> W.GridSpec:
        return W.GridSpec(self.kp_spin.value(), self.ki_spin.value(), factor=self.factor_spin.value(),
                          speed_exps=tuple(range(-int(self.speed_lo.value()), int(self.speed_hi.value()) + 1)),
                          shape_exps=tuple(range(-int(self.shape_lo.value()), int(self.shape_hi.value()) + 1)))

    def scan_grid(self) -> W.GridSpec:
        return W.GridSpec.scan(self.kp_spin.value(), self.ki_spin.value(), self.factor_spin.value(),
                               int(self.scan_lo.value()), int(self.scan_hi.value()))

    def safety_limits(self) -> W.SafetyLimits:
        n = max(self.range_spin.value(), 1.0)
        return W.SafetyLimits(max_gain_factor=n, min_gain_factor=1.0 / n)

    def analysis_kwargs(self) -> dict:
        kw = dict(li_stages=2, f0=self.f0_spin.value(), q=self.q_spin.value())
        if self.li_spin.value() > 0:
            kw["li_tau"] = self.li_spin.value() / 1000.0
        if self.loop_def.key == "afl":
            kw["ctrl_tau"] = self.tau_spin.value() / 1000.0
        return kw

    def _update_estimate(self):
        try:
            g, p = self.grid(), self.plan()
        except ValueError:
            return
        nk, ni = g.dims
        n = nk * ni
        n_scan = int(self.scan_lo.value()) + int(self.scan_hi.value()) + 1
        each = p.settle_s + p.duration + 0.3
        minutes = n * each / 60
        self.est_label.setText(f"Each test takes about {each:.0f} s. Scale scan: {n_scan} tests, about {n_scan * each / 60:.1f} min. "
                               f"Map: {n} tests, about {minutes:.1f} min (worst case; unstable regions are skipped).")
        self.preview_label.setText(self._preview_text(g))
        self.search_section.set_summary(self._search_summary_text(g, n, minutes))
        self._target_note()
        self._update_hint()

    def _preview_text(self, g: W.GridSpec) -> str:
        """Plain-number preview of what a map with this grid would actually write to SXM (no factor^n mental math)."""
        i0 = g.speed_exps.index(0) if 0 in g.speed_exps else 0
        kps = [(g.kp(i), g.speed_exps[i] == 0) for i in range(len(g.speed_exps))]
        kis = [(g.ki(i0, j), g.shape_exps[j] == 0) for j in range(len(g.shape_exps))]

        def fmt(pairs):
            return ", ".join(format_number(v, 4, 1e4) + (" (baseline)" if base else "") for v, base in pairs)
        lo = self.range_spin.value()
        return (f"Kp tested (speed axis): {fmt(kps)}\n"
               f"Ki tested at baseline speed (shape axis): {fmt(kis)}\n"
               f"Max gain change vs baseline keeps both within x{1 / lo:.3g} .. x{lo:.3g} of the baseline.")

    def _search_summary_text(self, g: W.GridSpec, n: int, minutes: float) -> str:
        """
        One-line, real-number summary for the section's collapsed header. Deliberately does NOT report
        step counts or the step factor (e.g. 'speed +-2, shape +-3, step x2') - that notation was exactly
        the log-scale mental math this tab is trying to get rid of. Report the actual Kp/Ki span instead.
        """
        i0 = g.speed_exps.index(0) if 0 in g.speed_exps else 0
        kp_a, kp_b = g.kp(0), g.kp(len(g.speed_exps) - 1)
        ki_a, ki_b = g.ki(i0, 0), g.ki(i0, len(g.shape_exps) - 1)
        return (f"Kp {format_number(kp_a, 4, 1e4)} to {format_number(kp_b, 4, 1e4)}, "
               f"Ki {format_number(ki_a, 4, 1e4)} to {format_number(ki_b, 4, 1e4)} "
               f"— {n} pairs, ~{minutes:.1f} min")

    def _gains_valid(self) -> Optional[str]:
        ld = self.loop_def
        kp, ki = self.kp_spin.value(), self.ki_spin.value()
        if kp == 0 or ki == 0:
            return "Kp and Ki must be non-zero."
        if math.copysign(1, kp) != ld.gain_sign or math.copysign(1, ki) != ld.gain_sign:
            return f"For this loop the raw gains must be {'negative' if ld.gain_sign < 0 else 'positive'} (manual)."
        return None

    def _online(self) -> bool:
        return self.driver is not None and not type(self.dde).__name__.startswith("Mock")

    @staticmethod
    def _accessibility_manager():
        """The app-wide AccessibilityManager, or None (e.g. a bare TuningTab built without one, in tests)."""
        app = QtWidgets.QApplication.instance()
        return getattr(app, "accessibility_manager", None)

    def _update_enabled(self, *_):
        running = self.runner is not None and self.runner.running
        ready = self._online() and all(c.isChecked() for c in self.checks) and not running
        for b in (self.btn_single, self.btn_scan, self.btn_map, self.btn_test_sug):
            b.setEnabled(ready)
        self.btn_narrow.setEnabled(ready and not self._narrow_converged)
        self.btn_sweep.setEnabled(ready)
        self.btn_stop.setEnabled(running)
        mgr = self._accessibility_manager()
        self.btn_stop.setStyleSheet(mgr.stop_button_style(running) if mgr else "")
        self.btn_stop.setToolTip("Aborts the running test and immediately writes the baseline Kp/Ki back. "
                                 "Shortcuts: Esc, Ctrl+." if running else "Nothing is running.")
        self.btn_clear.setEnabled(not running)
        self.btn_clear.setToolTip("Disabled while a test is running." if running else "")
        self.btn_scope.setEnabled(self.scope_tab is not None and not running)
        self.btn_scope.setToolTip(
            "Disabled while a test is running." if running else
            "No Scope tab to read from." if self.scope_tab is None else
            "Analyze the step train recorded in the Scope tab (needs the Step Test events on it).")
        if not self._online():
            tip = "Offline: running tests needs the real SXM and driver."
        elif not all(c.isChecked() for c in self.checks):
            tip = "Tick every item in '5. Before running' first."
        else:
            tip = ""
        for b in (self.btn_single, self.btn_map, self.btn_test_sug):
            b.setToolTip(tip)
        if tip:
            self.btn_scan.setToolTip(tip)
        else:
            self.btn_scan.setToolTip("Tests the baseline, then both gains scaled together in steps of the step factor: Ki:Kp is "
                                     "kept and only the speed changes. Use it first when the right decade of gain is unknown.")
        if tip:
            self.btn_sweep.setToolTip(tip)
        else:
            self.btn_sweep.setToolTip("Steps through the scale list above (Ki:Kp fixed), measuring the full battery "
                                      "for each: rise/fall, 2 % settling, over/undershoot, steady-state error, Drive "
                                      "peak/RMS excursion and noise. Stops at the first sign of trouble and "
                                      "recommends a final scale. Results appear in the 'Scale sweep' tab.")
        if tip:
            self.btn_narrow.setToolTip(tip)
        elif self._narrow_converged:
            self.btn_narrow.setToolTip("Converged: another round would not move this pair. Change the baseline or "
                                       "search span to narrow further.")
        else:
            self.btn_narrow.setToolTip("Fills the current map (if needed), zooms a finer grid onto its best cell and fills "
                                       "that too - one button, one confirmed round of hardware writes. Click again to "
                                       "narrow further.")
        self._update_context_label()
        self._update_hint()

    def _update_hint(self):
        """The one-line 'what now?' above the results: why the buttons are disabled and what to press next."""
        if self.runner_active():
            text = ("<b>Running.</b> The setpoint is being stepped and the response recorded; the verdict appears when the "
                    "test ends. <i>Stop &amp; restore baseline</i> aborts and puts the baseline gains back.")
        elif not self._online():
            text = ("<b>Offline.</b> Running tests needs the real SXM and driver. You can still analyse a step train "
                    "recorded in the Scope tab (<i>Analyze Scope capture</i>).")
        elif not all(c.isChecked() for c in self.checks):
            text = ("<b>First:</b> enter the Kp / Ki that are set in SXM now (left, 2), then tick every item in "
                    "<i>5. Before running</i>. The run buttons stay disabled until you do.")
        elif not self.singles and not (self.current_map() and self.current_map().results):
            what = "DNC use" if self.loop_def.key == "pll" else "Amplitude Ref"
            try:
                p = self.plan()
                secs = f" (about {p.settle_s + p.duration + 0.3:.0f} s)"
            except ValueError:
                secs = ""
            text = (f"<b>Ready.</b> Press <i>Run single test (baseline)</i>: it steps {what} {self.step_spin.value():g} "
                    f"{'Hz' if self.loop_def.key == 'pll' else '%'} up and down, measures the loop at the current gains "
                    f"and suggests what to change{secs}.")
            if self.loop_def.key == "afl":
                text += (" If the right decade of gain is unknown, press <i>Run scale scan</i> first: it tries the baseline and "
                         "both gains scaled together by decades.")
        else:
            text = ("<b>Next:</b> read the verdict and the suggestion below. <i>Test suggested pair</i> re-measures with the "
                    "suggested gains; <i>Stage in Params tab</i> hands a pair to the Parameters tab. Repeat until the "
                    "verdict is good.")
        if not self.runner_active():
            text += self._baseline_warning()
            text += self._confirmed_baseline_warning()
            text += self._voltage_warning()
        self.hint_label.setText(text)

    # ------------------------------------------------------------------ logging
    def _log(self, text):
        append_log_line(self.log, f"[{time.strftime('%H:%M:%S')}] {text}")

    # ------------------------------------------------------------------ running
    def current_map(self) -> Optional[W.ScreeningMap]:
        return self.maps[-1] if self.maps else None

    def _ensure_map(self) -> W.ScreeningMap:
        if not self.maps:
            m = W.ScreeningMap(self.grid(), self.target(), limits=self.safety_limits(),
                               reference=(self.kp_spin.value(), self.ki_spin.value()))
            self.maps.append(m)
            self._narrow_converged = False
            self._paint_map()
        return self.maps[-1]

    def _clear_map(self):
        self.maps.clear()
        self.selected = None
        self._narrow_converged = False
        self._narrow_active = False
        self._narrow_stage = None
        self._narrow_round = 0
        self._paint_map()
        self._refresh_table()
        self._update_enabled()

    # -- baseline confirmation (SXM cannot be read back, so this is what gets restored) --------------
    @staticmethod
    def _baseline_settings() -> QtCore.QSettings:
        return QtCore.QSettings("SXM-NCAFM", "TuningTab")

    def _load_confirmed_baseline(self, loop_key: str) -> Optional[Tuple[float, float]]:
        s = self._baseline_settings()
        kp_key, ki_key = f"confirmed_baseline/{loop_key}/kp", f"confirmed_baseline/{loop_key}/ki"
        # QSettings.value(key, None, type=float) silently returns 0.0 (not None) for a missing key -
        # check existence explicitly, or a never-confirmed baseline reads back as a spurious (0, 0).
        if not (s.contains(kp_key) and s.contains(ki_key)):
            return None
        return (s.value(kp_key, type=float), s.value(ki_key, type=float))

    def _save_confirmed_baseline(self, loop_key: str, kp: float, ki: float) -> None:
        s = self._baseline_settings()
        s.setValue(f"confirmed_baseline/{loop_key}/kp", kp)
        s.setValue(f"confirmed_baseline/{loop_key}/ki", ki)

    def _confirm_baseline(self, kp: float, ki: float) -> bool:
        """
        Gate for every run. If this exact (kp, ki) was already confirmed once this session for the
        current loop, proceed silently - this is what keeps 'Narrow search' (two internal runs per
        round) and repeated manual runs at an unchanged baseline from re-prompting every time. A
        genuinely new or edited baseline always asks, since this is what gets written back to SXM
        when the run stops or aborts, and SXM cannot be read back to check it independently.
        """
        loop_key = self.loop_def.key
        confirmed = self._confirmed_baseline.get(loop_key)
        if confirmed is not None and math.isclose(confirmed[0], kp, rel_tol=1e-9) and math.isclose(confirmed[1], ki, rel_tol=1e-9):
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle("Confirm baseline")
        box.setText(f"Will restore Kp = {kp:.4g}, Ki = {ki:.4g} when this run ends or is stopped.\n\n"
                   "This must match what is actually set in SXM right now - it cannot be read back.")
        box.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        box.setDefaultButton(QtWidgets.QMessageBox.Ok)
        mgr = self._accessibility_manager()
        if mgr:
            mgr.apply_to_widget(box)
        if box.exec_() != QtWidgets.QMessageBox.Ok:
            return False
        self._confirmed_baseline[loop_key] = (kp, ki)
        self._save_confirmed_baseline(loop_key, kp, ki)
        self._update_hint()          # the "differs from last confirmed" warning, if shown, clears now
        return True

    def _start_runner(self, next_item):
        err = self._gains_valid()
        if err:
            QtWidgets.QMessageBox.warning(self, "Baseline gains", err)
            return
        try:
            plan = self.plan()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Test", str(e))
            return
        if not self._confirm_baseline(self.kp_spin.value(), self.ki_spin.value()):
            self._log("Run not started: baseline not confirmed.")
            return
        self.runner = TuningRunner(self.dde, self.driver, plan, (self.kp_spin.value(), self.ki_spin.value()),
                                   self.analysis_kwargs(), parent=self)
        self.runner.message.connect(self._log)
        self.runner.test_started.connect(lambda c, kp, ki: self._log(f"testing {c}: Kp={kp:.4g}, Ki={ki:.4g}"))
        self.runner.test_finished.connect(self._on_test_finished)
        self.runner.finished.connect(self._on_run_finished)
        self._update_enabled()
        self.runner.start(next_item)
        self._update_enabled()

    def _run_single(self):
        self._start_runner(self._one_shot(self.kp_spin.value(), self.ki_spin.value()))

    def _one_shot(self, kp, ki):
        """A ``next_item`` callable that hands out one test at (kp, ki) and then None."""
        given = []

        def nxt():
            if given:
                return None
            given.append(1)
            return ("single", len(self.singles)), kp, ki
        return nxt

    def _run_map(self):
        m = self._ensure_map()

        def nxt():
            c = m.next_cell()
            if c is None:
                return None
            kp, ki = m.pair(c)
            return c, kp, ki
        self._start_runner(nxt)

    def _run_scan(self):
        """Scale scan: the baseline, then Kp and Ki scaled together (Ki:Kp kept) up and down by the step factor."""
        m = W.ScreeningMap(self.scan_grid(), self.target(), limits=self.safety_limits(),
                           reference=(self.kp_spin.value(), self.ki_spin.value()))
        self.maps.append(m)                          # on top of the stack: 'Back to coarse map' returns to the previous one
        self.selected = None
        self._narrow_converged = False
        self._paint_map()
        self._refresh_table()
        self._run_map()

    def _parse_scale_list(self) -> List[float]:
        parts = [x.strip() for x in self.scale_list_edit.text().split(",") if x.strip()]
        if not parts:
            raise ValueError("Scale list is empty. Example: 0.1, 0.3, 1, 2, 3")
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            raise ValueError("Scale list must be comma-separated numbers, e.g. 0.1, 0.3, 1, 2, 3")
        if any(v <= 0 for v in vals):
            raise ValueError("Scale values must be positive: they multiply the baseline Kp/Ki.")
        return vals

    def _run_scale_sweep(self):
        """
        The amplitude-loop protocol sweep: step through the scale list (Ki:Kp fixed), stopping at the
        first point the sweep's own assessment calls too_aggressive (saturation, overshoot/ringing
        beyond the target, or a substantial Drive noise rise) rather than blindly working through the
        whole list - the manual's stopping rule applies to what gets *written*, not just reported.
        """
        try:
            gs = self._parse_scale_list()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Scale list", str(e))
            return
        kp0, ki0 = self.kp_spin.value(), self.ki_spin.value()
        self._sweep_gs = gs
        self._sweep_results = [None] * len(gs)
        self._sweep_active = True
        self._refresh_sweep_view()
        self.detail_tabs.setCurrentWidget(self.sweep_tab)
        state = {"i": 0}

        def nxt():
            i = state["i"]
            if i > 0:
                tested = [(g, r) for g, r in zip(self._sweep_gs[:i], self._sweep_results[:i]) if r is not None]
                if tested:
                    gs_done, res_done = zip(*tested)
                    points, _, _ = W.assess_scale_sweep(list(gs_done), list(res_done), self.target(),
                                                         output_gain_v=self.gain_combo.currentData())
                    if points and points[-1].tier == "too_aggressive":
                        return None             # the sweep's own stopping rule: go no further
            if i >= len(gs):
                return None
            state["i"] = i + 1
            return ("sweep", i), kp0 * gs[i], ki0 * gs[i]
        self._start_runner(nxt)

    def _test_suggestion(self):
        row = self._suggestion_row()
        if row is None:
            return
        s = self._suggestions[row]
        if s.kp is None:
            return
        ref, lim = (self.kp_spin.value(), self.ki_spin.value()), self.safety_limits()
        if not W.gain_within_limits(s.kp, s.ki, ref, lim):
            self._log(f"Not run: Kp={s.kp:.4g}, Ki={s.ki:.4g} is more than x{lim.max_gain_factor:g} away from the baseline. "
                      "Raise 'Max gain change vs baseline', or move the baseline to a pair you have set in SXM.")
            return
        self._start_runner(self._one_shot(s.kp, s.ki))

    def stop(self):
        if self.runner is not None:
            self.runner.stop()

    def _on_run_finished(self, reason):
        self._log(f"run {reason}")
        self._update_enabled()
        self._paint_map()
        if self._narrow_active:
            if reason == "done":
                self._continue_narrow()
            else:
                self._narrow_active = False
                self._narrow_stage = None
        if not self._narrow_active:                  # not mid-sequence: 'Narrow search' chains two runs
            self._reset_checklist()                   # internally and should only reset once, at the end
        if self._sweep_active:
            self._sweep_active = False
            self._log_sweep_recommendation()

    def _log_sweep_recommendation(self):
        tested = [(g, r) for g, r in zip(self._sweep_gs, self._sweep_results) if r is not None]
        if not tested:
            return
        gs_done, res_done = zip(*tested)
        _, rec_g, explanation = W.assess_scale_sweep(list(gs_done), list(res_done), self.target(),
                                                      output_gain_v=self.gain_combo.currentData())
        self._log(f"Scale sweep finished: {explanation}")

    def _refresh_sweep_view(self):
        """Repaint the 'Scale sweep' tab from whatever's in self._sweep_results so far (partial or full)."""
        tested = [(g, r) for g, r in zip(self._sweep_gs, self._sweep_results) if r is not None]
        if not tested:
            self.sweep_table.setRowCount(0)
            self.sweep_summary.setText("Run a scale sweep (amplitude loop, left section 4/5) to see results here.")
            return
        gs_done, res_done = zip(*tested)
        points, rec_g, explanation = W.assess_scale_sweep(list(gs_done), list(res_done), self.target(),
                                                           output_gain_v=self.gain_combo.currentData())
        self.sweep_table.setRowCount(len(points))
        for row, p in enumerate(points):
            r = p.result

            def ms(sm, field="rise_time"):
                if sm is None:
                    return ""
                v = getattr(sm, field)
                return "inf" if math.isinf(v) else ("" if math.isnan(v) else f"{v * 1e3:.1f}")
            vals = [
                f"{p.g:g}", format_number(p.kp, 4, 1e4), format_number(p.ki, 4, 1e4),
                W.SCALE_TIER_LABEL.get(p.tier, p.tier),
                ms(r.primary_rising), ms(r.primary_falling), ms(r.primary_2pct, "settling_time"),
                "" if r.primary is None else f"{r.primary.overshoot * 100:.1f}",
                "" if r.primary is None or math.isnan(r.primary.steady_state_error) else f"{r.primary.steady_state_error * 100:.1f}",
                "" if math.isnan(r.drive_peak_abs) else f"{r.drive_peak_abs:.3g}",
                "" if math.isnan(r.drive_rms_excursion) else f"{r.drive_rms_excursion:.3g}",
                "" if r.secondary is None else f"{r.secondary.noise_rms:.3g}",
            ]
            col = QtGui.QColor(*SCALE_TIER_COLOR.get(p.tier, (255, 255, 255)))
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 3:
                    item.setBackground(col)
                self.sweep_table.setItem(row, c, item)
        if rec_g is not None:
            kp0, ki0 = self.kp_spin.value(), self.ki_spin.value()
            self.sweep_summary.setText(f"<b>Recommended g &asymp; {rec_g:.3g}</b> "
                                       f"(Kp &asymp; {format_number(kp0 * rec_g, 4, 1e4)}, "
                                       f"Ki &asymp; {format_number(ki0 * rec_g, 4, 1e4)})<br>{explanation}")
        else:
            self.sweep_summary.setText(explanation)

    def _on_test_finished(self, cell, res: W.StepTestResult):
        m = self.current_map()
        if isinstance(cell, tuple) and len(cell) == 2 and all(isinstance(c, (int, np.integer)) for c in cell) and m is not None:
            v = m.record(cell, res)
            self.selected = cell
        elif isinstance(cell, tuple) and len(cell) == 2 and cell[0] == "sweep":
            v = W.classify(res, self.target())
            res.verdict = v
            idx = cell[1]
            if 0 <= idx < len(self._sweep_results):
                self._sweep_results[idx] = res
            self._log(f"  g={self._sweep_gs[idx]:g}: {W.CATEGORY_LABEL.get(v.category, v.category)}: " + "; ".join(v.reasons))
            self._refresh_sweep_view()         # stays on the Scale sweep tab - no per-point plot jump
            return
        else:
            v = W.classify(res, self.target())
            res.verdict = v
            self.singles.append(res)
            self._single_result = res
            self.selected = None
            m2 = self._match_grid_cell(res)
            if m2 is not None:
                m2[0].results[m2[1]] = res
                self.selected = m2[1]
        self._log(f"  {W.CATEGORY_LABEL.get(v.category, v.category)}: " + "; ".join(v.reasons))
        self._paint_map()
        self._refresh_table()
        self._show_result(res)

    def _match_grid_cell(self, res):
        m = self.current_map()
        if m is None:
            return None
        for c in m.cells():
            kp, ki = m.pair(c)
            if math.isclose(kp, res.kp, rel_tol=1e-6) and math.isclose(ki, res.ki, rel_tol=1e-6):
                return m, c
        return None

    # ------------------------------------------------------------------ scope analysis
    def _analyze_scope(self):
        if self.scope_tab is None:
            return
        ct, msg = capture_from_scope(self.scope_tab, self.kp_spin.value(), self.ki_spin.value())
        if ct is None:
            self._log(f"Scope analysis: {msg}")
            QtWidgets.QMessageBox.information(self, "Analyze Scope capture", msg)
            return
        idx = self.loop_combo.findData(ct.plan.loop)
        if idx >= 0 and idx != self.loop_combo.currentIndex():
            self.loop_combo.setCurrentIndex(idx)
        res = W.analyze_test(ct, **self.analysis_kwargs())
        res.verdict = W.classify(res, self.target())
        res.warnings.insert(0, f"gains assumed for this capture: Kp={ct.kp:.4g}, Ki={ct.ki:.4g} (from the Baseline fields)")
        self.singles.append(res)
        self._single_result = res
        self.selected = None
        self._log(f"Scope analysis: {msg}; {W.CATEGORY_LABEL.get(res.verdict.category)}")
        self._refresh_table()
        self._show_result(res)
        self.detail_tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ map view
    def _paint_map(self):
        for t in self._map_texts:
            self.map_plot.removeItem(t)
        self._map_texts = []
        m = self.current_map()
        if m is None:
            self.map_img.clear()
            self.map_marks.setData([], [])
            # a clean, honest empty state: no map yet, so no axis numbers that could be mistaken for data
            # (pyqtgraph's own auto-range default is an arbitrary, misleadingly round-looking grid).
            self.map_plot.setXRange(0, 1)
            self.map_plot.setYRange(0, 1)
            self.map_plot.getAxis("left").setTicks([[]])
            self.map_plot.getAxis("bottom").setTicks([[]])
            placeholder = pg.TextItem("Run a test to see the map here", color=(140, 150, 160), anchor=(0.5, 0.5))
            placeholder.setPos(0.5, 0.5)
            self.map_plot.addItem(placeholder)
            self._map_texts.append(placeholder)
            return
        nk, ni = m.grid.dims
        cat = m.category_grid()
        img = np.zeros((ni, nk, 4), dtype=np.uint8)
        for i in range(nk):
            for j in range(ni):
                c = cat[i, j]
                col = CATEGORY_COLOR[c]
                alpha = 255
                if c == "untested" and m.prior.get((i, j)) in CATEGORY_COLOR:      # faint model prediction
                    col, alpha = CATEGORY_COLOR[m.prior[(i, j)]], 70
                img[j, i] = (*col, alpha)
        self.map_img.setImage(img, autoLevels=False)
        self.map_img.setRect(QtCore.QRectF(0, 0, ni, nk))
        self.map_plot.setXRange(-0.2, ni + 0.2)
        self.map_plot.setYRange(-0.2, nk + 0.2)
        # left axis (speed) is a real Kp number - Kp depends only on speed; bottom axis (shape) is a ratio
        # multiple of the baseline's Ki:Kp - real Ki depends on speed *and* shape, so a per-column number
        # would be misleading (see tuning/workflow.py GridSpec docstring).
        self.map_plot.getAxis("left").setTicks([[(i + 0.5, f"{m.grid.kp(i):.3g}") for i in range(nk)]])
        self.map_plot.getAxis("bottom").setTicks(
            [[(j + 0.5, f"x{m.grid.factor ** m.grid.shape_exps[j]:.3g}") for j in range(ni)]])
        rise = m.value_grid("rise")
        for i in range(nk):
            for j in range(ni):
                txt = ""
                if not np.isnan(rise[i, j]):
                    txt = f"{rise[i, j] * 1e3:.0f} ms"
                elif cat[i, j] in ("lost", "skipped"):
                    txt = {"lost": "lost", "skipped": "skip"}[cat[i, j]]
                if txt:
                    t = pg.TextItem(txt, color=(20, 20, 20), anchor=(0.5, 0.5))
                    t.setPos(j + 0.5, i + 0.5)
                    self.map_plot.addItem(t)
                    self._map_texts.append(t)
        # crosshair through the baseline's speed row and shape column, and an outline on the selected cell
        xs, ys = [], []
        b = m.baseline
        if b[0] is not None and b[1] is not None:
            xs += [b[1] + 0.5, b[1] + 0.5, float("nan"), 0.0, ni]
            ys += [0.0, nk, float("nan"), b[0] + 0.5, b[0] + 0.5]
        if self.selected is not None:
            i, j = self.selected
            xs += [float("nan"), j, j + 1, j + 1, j, j]
            ys += [float("nan"), i, i, i + 1, i + 1, i]
        self.map_marks.setData(xs, ys, pen=pg.mkPen((30, 30, 30), width=2), connect="finite")
        self.map_plot.setTitle("Speed x Shape map: up = faster, right = more integral (Ki-heavy)")

    def _on_map_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        m = self.current_map()
        if m is None:
            return
        vb = self.map_plot.getPlotItem().vb
        pos = vb.mapSceneToView(ev.scenePos())
        nk, ni = m.grid.dims
        i, j = int(math.floor(pos.y())), int(math.floor(pos.x()))
        if 0 <= i < nk and 0 <= j < ni and (i, j) in m.cells():
            self.select_cell((i, j))

    def select_cell(self, cell):
        self.selected = cell
        m = self.current_map()
        res = m.results.get(cell) if m else None
        self._paint_map()
        if res is not None:
            self._show_result(res)
        elif m is not None:
            kp, ki = m.pair(cell)
            why = m.skipped.get(cell) or (f"the identified model expects: {W.CATEGORY_LABEL.get(m.prior.get(cell), '?')}" if cell in m.prior else "not tested yet")
            self.detail_text.setHtml(f"<b>Kp = {kp:.5g}, Ki = {ki:.5g}</b><br>{why}")
            self.plot1.clear()
            self.plot2.clear()
            self._suggestions = []

    # ------------------------------------------------------------------ detail view
    def _show_result(self, res: W.StepTestResult):
        v = res.verdict or W.classify(res, self.target())
        sugg = W.advise(res, v, self.target())
        self._suggestions = sugg
        ld = W.LOOPS.get(res.loop, self.loop_def)
        self.detail_text.setHtml(format_result_html(res, v, sugg, ld))
        self.detail_tabs.setCurrentIndex(0)                   # the plots of the selected test
        self.btn_prior.setEnabled(res.model is not None)
        for p in (self.plot1, self.plot2):
            p.clear()
        if res.grid is None or res.mean_primary is None:
            return
        g = res.grid * 1e3
        pen = pg.mkPen((40, 90, 200), width=2)
        self.plot1.plot(g, res.mean_primary, pen=pen)
        if res.std_primary is not None and res.n_steps > 1:
            for sgn in (1, -1):
                self.plot1.plot(g, res.mean_primary + sgn * res.std_primary, pen=pg.mkPen((120, 150, 230), style=QtCore.Qt.DashLine))
        p = res.primary
        if p is not None:
            for level in (p.y_initial, p.y_final):
                self.plot1.addLine(y=level, pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.DotLine))
        self.plot1.addLine(x=0, pen=pg.mkPen((200, 60, 60), style=QtCore.Qt.DashLine))
        self.plot1.setLabel("left", f"{ld.primary_channel} (averaged, sign-folded)")
        if res.mean_secondary is not None:
            self.plot2.plot(g, res.mean_secondary, pen=pg.mkPen((200, 60, 60), width=2))
            self.plot2.addLine(x=0, pen=pg.mkPen((200, 60, 60), style=QtCore.Qt.DashLine))
        self.plot2.setLabel("left", "Phase" if ld.key == "pll" else "Drive")
        self.plot2.setLabel("bottom", "time after the step (ms)")

    def _suggestion_row(self) -> Optional[int]:
        return 0 if self._suggestions else None

    def _refresh_table(self):
        rows = []
        m = self.current_map()
        if m is not None:
            for c in m.order():
                r = m.results.get(c)
                if r is not None:
                    rows.append(r)
        rows += [r for r in self.singles if r not in rows]
        self.table.setRowCount(len(rows))
        for n, r in enumerate(rows):
            v = r.verdict.category if r.verdict else "?"
            p = r.primary
            vals = [f"{r.kp:.4g}", f"{r.ki:.4g}", W.CATEGORY_LABEL.get(v, v),
                    "" if p is None or math.isnan(p.rise_time) else f"{p.rise_time * 1e3:.1f}",
                    "" if p is None else f"{p.overshoot * 100:.1f}",
                    "" if r.error is None else ("inf" if math.isinf(r.error.decay_time) else f"{r.error.decay_time * 1e3:.0f}"),
                    "" if math.isnan(r.noise_rms) else f"{r.noise_rms:.3g}"]
            for k, txt in enumerate(vals):
                self.table.setItem(n, k, QtWidgets.QTableWidgetItem(txt))

    # ------------------------------------------------------------------ actions on the selection
    def _selected_pair(self) -> Optional[Tuple[float, float]]:
        m = self.current_map()
        if self.selected is not None and m is not None:
            return m.pair(self.selected)
        if self._single_result is not None:
            return self._single_result.kp, self._single_result.ki
        return None

    def _stage_selected(self):
        pair = self._selected_pair()
        if pair is None or self.params_tab is None:
            return
        ld = self.loop_def
        ok1 = self.params_tab.stage_value(ld.kp_param[0], ld.kp_param[1], pair[0])
        ok2 = self.params_tab.stage_value(ld.ki_param[0], ld.ki_param[1], pair[1])
        self._log(f"staged Kp={pair[0]:.4g}, Ki={pair[1]:.4g} in the Parameters tab ({'ok' if ok1 and ok2 else 'not all rows found'}); apply it there.")

    def _zoom(self):
        m = self.current_map()
        if m is None:
            return
        cell = self.selected
        if cell is None or cell not in m.results:
            sug = m.suggest_refinement()
            if sug is None:
                self._log("Nothing to zoom into yet: run the map first.")
                return
            cell, why = sug
            self._log(f"Suggested zoom: {why}")
            self.select_cell(cell)
            return
        sub = m.refined(cell)
        center = (sub.grid.speed_exps.index(0), sub.grid.shape_exps.index(0))
        sub.results[center] = m.results[cell]                # already measured: do not repeat it
        self.maps.append(sub)
        self.selected = center
        self._narrow_converged = False
        self._log(f"Refined map around Kp={sub.grid.kp0:.4g}, Ki={sub.grid.ki0:.4g} (step x{sub.grid.factor:.3g}). Run / continue map to fill it.")
        self._paint_map()
        self._refresh_table()
        self._update_enabled()

    def _back(self):
        if len(self.maps) > 1:
            self.maps.pop()
            self.selected = None
            self._paint_map()
            self._refresh_table()

    # ------------------------------------------------------------------ auto-zoom ("Narrow search")
    def _narrow_search(self):
        """
        One click, one confirmed round: fill the current map if it isn't already, zoom a finer grid onto
        its best (or nearest-miss) cell, fill that too, then report what changed and stop. Click again for
        another round. Reuses the same building blocks as 'Run / continue map' and 'Suggest / refine zoom'.
        """
        if self.runner_active():
            return
        m = self.current_map()
        if m is None:
            self._log("Nothing to narrow yet: run the map or scale scan first.")
            return
        self._narrow_active = True
        self._narrow_stage = None
        if m.next_cell() is not None:
            self._run_map()                # fills the base map; _on_run_finished() calls back into _continue_narrow()
        else:
            self._continue_narrow()        # already full: go straight to the zoom step

    def _continue_narrow(self):
        m = self.current_map()
        if self._narrow_stage != "zoomed":
            sug = m.suggest_refinement()
            if sug is None:
                self._narrow_active = False
                self._log("Narrow search: no island or near-miss to zoom onto yet - try a wider search span first.")
                return
            cell, why = sug
            self._narrow_prev_pair = m.pair(cell)
            sub = m.refined(cell)
            center = (sub.grid.speed_exps.index(0), sub.grid.shape_exps.index(0))
            sub.results[center] = m.results[cell]              # already measured: do not repeat it
            self.maps.append(sub)
            self.selected = center
            self._narrow_stage = "zoomed"
            self._narrow_round += 1
            self._log(f"Narrow search, round {self._narrow_round}: zooming onto {why}.")
            self._paint_map()
            self._refresh_table()
            self._run_map()                # fills the finer map; _on_run_finished() calls back in here again
            return
        self._narrow_active = False
        self._narrow_stage = None
        best = m.best()
        pkp, pki = self._narrow_prev_pair
        center = (m.grid.speed_exps.index(0), m.grid.shape_exps.index(0))
        if best is None:
            self._log(f"Narrow search, round {self._narrow_round}: no good cell in the finer grid around "
                      f"Kp={pkp:.4g}, Ki={pki:.4g}. 'Max gain change vs baseline' may be too tight, or the loop is "
                      "genuinely marginal here.")
            self._update_enabled()
            return
        kp, ki = m.pair(best)
        converged = best == center
        self._narrow_converged = converged
        msg = (f"Narrow search, round {self._narrow_round}: centred on Kp={kp:.4g}, Ki={ki:.4g} "
              f"(speed x{kp / pkp:.3g}, shape x{(ki / kp) / (pki / pkp):.3g} of the previous round), "
              f"step now x{m.grid.factor:.3g}.")
        if converged:
            msg += " Converged: another round would not move this pair."
        self._log(msg)
        self.select_cell(best)
        self._update_enabled()

    def _predict_prior(self):
        m = self.current_map()
        res = None
        if m is not None and self.selected is not None:
            res = m.results.get(self.selected)
        res = res or self._single_result
        if m is None or res is None or res.model is None:
            self._log("Predicting a map needs a test analysed with the lock-in TimeConstant set (identification).")
            return
        m.set_prior_from_model(res.model)
        self._log("Faint colours show what the identified model predicts; pairs it predicts unstable will be skipped.")
        self._paint_map()

    # ------------------------------------------------------------------ lifecycle
    def runner_active(self) -> bool:
        return self.runner is not None and self.runner.running

    def closeEvent(self, event):
        self.stop()
        event.accept()
