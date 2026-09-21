"""
Tuning tab: guided screening of the PLL / amplitude-loop gains on the instrument.

Workflow (the manual's, made systematic)
----------------------------------------
1. Define the test: which loop, the step (PLL: +-1 Hz on ``DNC use``; amplitude: +-10 % on ``Ref``),
   hold time, number of events. The channels recorded are implied by the loop
   (PLL: ``df`` + ``Phase``; amplitude loop: ``QPlusAmpl`` + ``Drive``).
2. Enter the known-good baseline gains (SXM cannot be read back, so the tab needs them).
3. Set the target: an imaging scan (line time, pixels) or a response time.
4. Run one test, or screen a log-spaced (Kp, Ki) grid around the baseline: every cell is a
   step train, measured and classified. The 2D map shows *islands* of rectangular, well-controlled
   response; moving up-right along a diagonal (raise both, same ratio) makes the loop faster, down-left
   slower; across diagonals the shape changes. Click a cell for its averaged response and advice
   (higher / lower / different pairing), then refine around it or around the suggested cell.
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
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from sxm_ncafm_control.device_driver import CHANNELS
from sxm_ncafm_control.tuning import metrics as M
from sxm_ncafm_control.tuning import workflow as W

from ..common import append_log_line

CATEGORY_COLOR = {
    "good": (70, 175, 95), "overshoot": (240, 200, 60), "ringing": (240, 140, 50), "slow_tail": (120, 170, 230),
    "too_slow": (175, 175, 175), "lost": (205, 65, 65), "skipped": (95, 95, 95), "untested": (238, 238, 238),
}


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
            self.message.emit("Baseline gains and the stepped parameter were restored.")
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
        rows.append(f"Drive: overshoot {r.secondary.overshoot * 100:.0f} % of its final change")
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


# ---------------------------------------------------------------------------
# the tab
# ---------------------------------------------------------------------------
class TuningTab(QtWidgets.QWidget):
    """Guided PLL / amplitude-loop tuning: define, run, analyse, screen a (Kp, Ki) map, refine."""

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
        self._build()
        self._on_loop_changed()
        self._update_enabled()

    # ------------------------------------------------------------------ construction
    def _spin(self, lo, hi, val, dec=3, step=None, suffix=""):
        s = QtWidgets.QDoubleSpinBox()
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
        root = QtWidgets.QHBoxLayout(self)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(330)
        left_scroll.setMaximumWidth(420)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        left_scroll.setWidget(left)
        root.addWidget(left_scroll)

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
        self.base_spin = self._spin(-1e12, 1e12, 25000.0, 4)
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
        self.f0_spin = self._spin(1.0, 1e7, 25000.0, 1, 100.0, " Hz")
        self.q_spin = self._spin(1.0, 1e7, 25000.0, 0, 1000.0)
        f.addRow("f0 (from the sweep):", self.f0_spin)
        f.addRow("Q (from the sweep):", self.q_spin)
        lv.addWidget(g)

        # 2. baseline
        g = QtWidgets.QGroupBox("2. Baseline (currently set in SXM - it cannot be read back)")
        f = QtWidgets.QFormLayout(g)
        self.kp_spin = self._spin(-1e12, 1e12, -100.0, 4)
        self.ki_spin = self._spin(-1e12, 1e12, -1e4, 4)
        f.addRow("Kp:", self.kp_spin)
        f.addRow("Ki:", self.ki_spin)
        lv.addWidget(g)

        # 3. target
        g = QtWidgets.QGroupBox("3. Target")
        f = QtWidgets.QFormLayout(g)
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
        lv.addWidget(g)

        # 4. grid
        g = QtWidgets.QGroupBox("4. Map grid (log-spaced around the baseline)")
        f = QtWidgets.QFormLayout(g)
        self.factor_spin = self._spin(1.2, 4.0, 2.0, 2, 0.1, " x")
        self.kp_lo = self._spin(0, 4, 2, 0)
        self.kp_hi = self._spin(0, 4, 2, 0)
        self.ki_lo = self._spin(0, 5, 3, 0)
        self.ki_hi = self._spin(0, 5, 2, 0)
        f.addRow("Step factor:", self.factor_spin)
        f.addRow("Kp steps down / up:", self._pair(self.kp_lo, self.kp_hi))
        f.addRow("Ki steps down / up:", self._pair(self.ki_lo, self.ki_hi))
        self.est_label = QtWidgets.QLabel()
        f.addRow(self.est_label)
        lv.addWidget(g)
        for w in (self.factor_spin, self.kp_lo, self.kp_hi, self.ki_lo, self.ki_hi, self.hold_spin, self.settle_spin, self.events_spin):
            w.valueChanged.connect(self._update_estimate)

        # 5. checklist + buttons
        g = QtWidgets.QGroupBox("5. Before running")
        v = QtWidgets.QVBoxLayout(g)
        self.checks = []
        for text in ("Tip retracted / far from the surface", "The loop under test is running and locked",
                     "PLL: DNC Lockin Options > Acquire > Auto 0 deg is DISABLED", "Baseline Kp/Ki above match what is set in SXM"):
            c = QtWidgets.QCheckBox(text)
            c.toggled.connect(self._update_enabled)
            v.addWidget(c)
            self.checks.append(c)
        self.btn_single = QtWidgets.QPushButton("Run single test (baseline)")
        self.btn_map = QtWidgets.QPushButton("Run / continue map")
        self.btn_stop = QtWidgets.QPushButton("Stop && restore baseline")
        self.btn_clear = QtWidgets.QPushButton("Clear map")
        self.btn_scope = QtWidgets.QPushButton("Analyze Scope capture")
        self.btn_scope.setToolTip("Analyze the step train recorded in the Scope tab (needs the Step Test events on it).")
        for b in (self.btn_single, self.btn_map, self.btn_stop, self.btn_clear, self.btn_scope):
            v.addWidget(b)
        self.btn_single.clicked.connect(self._run_single)
        self.btn_map.clicked.connect(self._run_map)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_clear.clicked.connect(self._clear_map)
        self.btn_scope.clicked.connect(self._analyze_scope)
        lv.addWidget(g)
        lv.addStretch(1)

        # right side: [map | advice] over [selected-test plots | all tests | log]
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(right, 1)
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        right.addWidget(top)

        self.map_plot = pg.PlotWidget(title="(Kp, Ki) map: click a cell")
        self.map_plot.setBackground("w")
        self.map_plot.setMouseEnabled(False, False)
        self.map_plot.setLabel("bottom", "Ki  (raw, log-spaced)")
        self.map_plot.setLabel("left", "Kp  (raw, log-spaced)")
        self.map_plot.setMinimumSize(360, 260)
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
        for n, b in enumerate((self.btn_test_sug, self.btn_apply, self.btn_zoom, self.btn_back, self.btn_prior)):
            b.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # never force the panel wider
            grid.addWidget(b, n // 2, n % 2)
        sv.addLayout(grid)
        side.setMinimumWidth(280)
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
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.detail_tabs.addTab(self.log, "Log")
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        self.btn_apply.clicked.connect(self._stage_selected)
        self.btn_zoom.clicked.connect(self._zoom)
        self.btn_back.clicked.connect(self._back)
        self.btn_prior.clicked.connect(self._predict_prior)
        self.btn_test_sug.clicked.connect(self._test_suggestion)
        self._suggestions: List[W.Suggestion] = []

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
        self.kp_spin.setValue(-100.0 if pll else 8.9e7)
        self.ki_spin.setValue(-1e4 if pll else 8900.0)
        for w in (self.tau_spin, self.tau_label):
            w.setVisible(not pll)
        self.target_combo.setCurrentIndex(0 if pll else 1)
        self.rise_spin.setValue(50.0 if pll else 500.0)
        self._on_target_changed()
        self._update_estimate()

    def _on_target_changed(self):
        scan = self.target_combo.currentIndex() == 0
        for w in (self.line_spin, self.px_spin, self.line_label, self.px_label):
            w.setVisible(scan)
        for w in (self.rise_spin, self.rise_label):
            w.setVisible(not scan)
        self._target_note()

    def _target_note(self):
        t = self.target()
        txt = f"Rise <= {t.rise_max * 1e3:.1f} ms"
        if t.decay_max:
            txt += f", Phase tail gone within {t.decay_max * 1e3:.0f} ms"
        self.target_note.setText(txt)

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
                          kp_exps=tuple(range(-int(self.kp_lo.value()), int(self.kp_hi.value()) + 1)),
                          ki_exps=tuple(range(-int(self.ki_lo.value()), int(self.ki_hi.value()) + 1)))

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
        n = g.shape[0] * g.shape[1]
        each = p.settle_s + p.duration + 0.3
        self.est_label.setText(f"{n} tests, about {n * each / 60:.1f} min (worst case; unstable regions are skipped)")
        self._target_note()

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

    def _update_enabled(self, *_):
        running = self.runner is not None and self.runner.running
        ready = self._online() and all(c.isChecked() for c in self.checks) and not running
        for b in (self.btn_single, self.btn_map, self.btn_test_sug):
            b.setEnabled(ready)
        self.btn_stop.setEnabled(running)
        self.btn_clear.setEnabled(not running)
        self.btn_scope.setEnabled(self.scope_tab is not None and not running)
        tip = "" if self._online() else "Offline: running tests needs the real SXM and driver."
        for b in (self.btn_single, self.btn_map):
            b.setToolTip(tip)

    # ------------------------------------------------------------------ logging
    def _log(self, text):
        append_log_line(self.log, f"[{time.strftime('%H:%M:%S')}] {text}")

    # ------------------------------------------------------------------ running
    def current_map(self) -> Optional[W.ScreeningMap]:
        return self.maps[-1] if self.maps else None

    def _ensure_map(self) -> W.ScreeningMap:
        if not self.maps:
            m = W.ScreeningMap(self.grid(), self.target(), reference=(self.kp_spin.value(), self.ki_spin.value()))
            self.maps.append(m)
            self._paint_map()
        return self.maps[-1]

    def _clear_map(self):
        self.maps.clear()
        self.selected = None
        self._paint_map()
        self._refresh_table()

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

    def _test_suggestion(self):
        row = self._suggestion_row()
        if row is None:
            return
        s = self._suggestions[row]
        if s.kp is None:
            return
        self._start_runner(self._one_shot(s.kp, s.ki))

    def stop(self):
        if self.runner is not None:
            self.runner.stop()

    def _on_run_finished(self, reason):
        self._log(f"run {reason}")
        self._update_enabled()
        self._paint_map()

    def _on_test_finished(self, cell, res: W.StepTestResult):
        m = self.current_map()
        if isinstance(cell, tuple) and len(cell) == 2 and all(isinstance(c, (int, np.integer)) for c in cell) and m is not None:
            v = m.record(cell, res)
            self.selected = cell
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
            return
        nk, ni = m.grid.shape
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
        self.map_plot.getAxis("bottom").setTicks([[(j + 0.5, f"{m.grid.ki(j):.3g}") for j in range(ni)]])
        self.map_plot.getAxis("left").setTicks([[(i + 0.5, f"{m.grid.kp(i):.3g}") for i in range(nk)]])
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
        # baseline marker, constant-ratio diagonal (raise both = faster) and the selected cell
        xs, ys = [], []
        b = m.baseline
        if b[0] is not None and b[1] is not None:
            d = b[1] - b[0]
            xs += [max(0, d) + 0.0, min(ni, nk + d)]
            ys += [max(0, -d) + 0.0, min(nk, ni - d)]
        if self.selected is not None:
            i, j = self.selected
            xs += [float("nan"), j, j + 1, j + 1, j, j]
            ys += [float("nan"), i, i, i + 1, i + 1, i]
        self.map_marks.setData(xs, ys, pen=pg.mkPen((30, 30, 30), width=2), connect="finite")
        title = "(Kp, Ki) map: up-right = raise both (faster), down-left = lower both (slower)"
        self.map_plot.setTitle(title)

    def _on_map_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        m = self.current_map()
        if m is None:
            return
        vb = self.map_plot.getPlotItem().vb
        pos = vb.mapSceneToView(ev.scenePos())
        nk, ni = m.grid.shape
        i, j = int(math.floor(pos.y())), int(math.floor(pos.x()))
        if 0 <= i < nk and 0 <= j < ni:
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
        center = (sub.grid.kp_exps.index(0), sub.grid.ki_exps.index(0))
        sub.results[center] = m.results[cell]                # already measured: do not repeat it
        self.maps.append(sub)
        self.selected = center
        self._log(f"Refined map around Kp={sub.grid.kp0:.4g}, Ki={sub.grid.ki0:.4g} (step x{sub.grid.factor:.3g}). Run / continue map to fill it.")
        self._paint_map()
        self._refresh_table()

    def _back(self):
        if len(self.maps) > 1:
            self.maps.pop()
            self.selected = None
            self._paint_map()
            self._refresh_table()

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
