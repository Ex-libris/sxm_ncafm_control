"""
Runs tuning conditions on the instrument, every one from the same verified baseline.

For each condition (a Kp/Ki pair proposed by the explorer, or picked by hand):

1. **Recover** - write the baseline gains and the base setpoint, then wait until the loop is verifiably back
   at the reference state measured at the start of the run (tuning/explore.py: ``recovered``). The first
   condition instead waits until the loop is steady and takes that as the reference. No recovery within
   the timeout stops the run: later conditions would not start from the same place.
2. **Settle** - write the candidate gains and the first level, wait at least ``settle_s``, then until the
   loop is steady at that level. Not steady within the timeout: the condition is recorded as failed and the
   next one starts (with its own recovery).
3. **Test** - the step train (alternating up / down), recorded through the driver; a runaway (amplitude
   collapse, lost PLL lock) ends the condition as lost.

The baseline gains and setpoint are always written back at the end, on Stop and on abort. DDE writes block
on the GUI thread, so this is a QTimer state machine; only the channel capture runs in a worker thread.
"""
import math
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from sxm_ncafm_control.device_driver import CHANNELS
from sxm_ncafm_control.tuning import explore as X
from sxm_ncafm_control.tuning import workflow as W


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


class ConditionRunner(QtCore.QObject):
    """Measures conditions one after another; ``next_item()`` hands out :class:`explore.Proposal` or None."""

    condition_started = QtCore.pyqtSignal(object)              # Proposal
    condition_finished = QtCore.pyqtSignal(object, object)     # Proposal, StepTestResult
    phase = QtCore.pyqtSignal(str)                             # what is happening now, for the status line
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)                          # 'done' | 'stopped' | 'aborted: ...'

    POLL_MS = 100
    PAUSE_MS = 100

    def __init__(self, dde, driver, plan: W.StepTestPlan, baseline: Tuple[float, float], *,
                 criteria: X.RecoveryCriteria = X.RecoveryCriteria(), limits: W.SafetyLimits = W.SafetyLimits(),
                 recover_timeout_s: float = 20.0, settle_timeout_s: Optional[float] = None,
                 reference: Optional[X.BaselineReference] = None, analysis: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.dde, self.driver, self.plan = dde, driver, plan
        self.loop = plan.loop_def
        self.baseline = baseline
        self.criteria, self.limits = criteria, limits
        self.recover_timeout_s = recover_timeout_s
        self.settle_timeout_s = settle_timeout_s or max(3.0 * plan.settle_s, plan.settle_s + 5.0)
        self.reference = reference           # kept across conditions (and runs, if the tab passes it back)
        self.analysis = analysis or {}
        self.running = False
        self._token = 0
        self._cap: Optional[TrainCaptureThread] = None
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(self.POLL_MS)
        self._poll.timeout.connect(self._on_poll)
        self._next_item: Optional[Callable] = None
        self._prop: Optional[X.Proposal] = None
        self._state = "idle"
        self._t_state = 0.0
        self._t_candidate = 0.0
        self._t_test = 0.0
        self._events = []
        self._check: dict = {}
        self._prev_lost = False
        self._last_why = ""

    # -- instrument writes ----------------------------------------------------------------------------
    def _write(self, param, value):
        ptype, code = param
        if ptype == "DNC":
            self.dde.send_dncpara(int(code), float(value))
        else:
            self.dde.send_scanpara(str(code), float(value))

    def _apply_gains(self, kp, ki):
        self._write(self.loop.kp_param, kp)
        self._write(self.loop.ki_param, ki)

    def _write_baseline(self):
        """Baseline gains and base setpoint; a PLL that lost lock is released first (manual: Ki/Kp to 0, then back)."""
        if self._prev_lost and self.loop.key == "pll":
            self._apply_gains(0.0, 0.0)
        self._apply_gains(*self.baseline)
        self._write(self.loop.step_param, self.plan.base)

    # -- control --------------------------------------------------------------------------------------
    def start(self, next_item):
        if self.running:
            return
        self._next_item = next_item
        self.running = True
        self._next()

    def stop(self):
        if self.running:
            self._finish("stopped")

    def _finish(self, reason: str):
        self._token += 1
        self._poll.stop()
        self._stop_capture()
        was = self.running
        self.running = False
        self._state = "idle"
        if was:
            try:
                self._write_baseline()
            except Exception as e:  # pragma: no cover - only when the instrument itself is failing
                self.message.emit(f"WARNING: could not restore the baseline: {e}")
            kp, ki = self.baseline
            step = "DNC use" if self.loop.key == "pll" else "Ref"
            self.message.emit(f"Baseline restored: Kp={kp:.4g}, Ki={ki:.4g}, {step}={self.plan.base:.6g}.")
            self.finished.emit(reason)

    def _fail_run(self, reason: str):
        self.message.emit(f"STOPPED: {reason}")
        self._finish(f"aborted: {reason}")

    def _later(self, ms, fn, *args):
        tok = self._token

        def call():
            if self.running and tok == self._token:
                fn(*args)
        QtCore.QTimer.singleShot(max(0, int(ms)), QtCore.Qt.PreciseTimer, call)

    def _stop_capture(self):
        if self._cap is not None:
            self._cap.stop()
            self._cap.wait(1000)
            self._cap = None

    # -- one condition ------------------------------------------------------------------------------------
    def _next(self):
        if not self.running:
            return
        p = self._next_item()
        if p is None:
            self._finish("done")
            return
        self._token += 1
        self._prop = p
        self._events, self._check = [], {}
        self.condition_started.emit(p)
        names = [self.loop.y_channel, self.loop.u_channel]
        specs = {n: (CHANNELS[n][0], CHANNELS[n][3]) for n in names}
        max_s = self.recover_timeout_s + self.settle_timeout_s + self.plan.duration + 5.0
        self._cap = TrainCaptureThread(self.driver, specs, max_s=max_s)
        self._cap.error.connect(self._fail_run)
        self._cap.start()
        try:
            self._write_baseline()
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self._enter("recover")
        self._poll.start()

    def _enter(self, state: str):
        self._state = state
        self._t_state = self._cap.now() if self._cap is not None else 0.0
        text = {"recover": "recovering to the baseline" if self.reference else "measuring the baseline reference",
                "settle": "settling at the candidate gains", "test": "stepping"}[state]
        self.phase.emit(f"Kp={self._prop.kp:.4g}, Ki={self._prop.ki:.4g}: {text}")

    def _window(self):
        cap = self._cap
        n = cap.count
        if n < 5:
            return None
        t = cap.t[:n]
        t_end = t[-1]
        lo = int(np.searchsorted(t, t_end - self.criteria.window_s))
        if t_end - self._t_state < self.criteria.window_s or n - lo < 20:
            return None
        return cap.data[self.loop.y_channel][lo:n], cap.data[self.loop.u_channel][lo:n], t_end

    def _kappa(self) -> Optional[float]:
        if self.loop.key != "afl" or self.reference is None or not self.plan.base:
            return None
        return self.reference.y / self.plan.base

    def _runaway(self) -> Optional[str]:
        cap = self._cap
        n = cap.count
        if n < 40:
            return None
        lo = max(0, n - 100)
        tail = {name: arr[lo:n] for name, arr in cap.data.items()}
        return W.runaway_reason(self.plan, self.limits, tail, self._kappa())

    def _on_poll(self):
        if not self.running or self._cap is None or self._cap.t0 == 0.0:
            return
        elapsed = self._cap.now() - self._t_state
        if self._state in ("settle", "test"):
            why = self._runaway()
            if why:
                self.message.emit(f"  lost at Kp={self._prop.kp:.4g}, Ki={self._prop.ki:.4g}: {why}")
                self._end_condition(W.StepTestResult(kp=self._prop.kp, ki=self._prop.ki, loop=self.plan.loop,
                                                     failure=f"lost: {why}"))
                return
        if self._state == "recover":
            self._poll_recover(elapsed)
        elif self._state == "settle":
            self._poll_settle(elapsed)

    def _poll_recover(self, elapsed):
        w = self._window()
        if w is not None:
            y, u, _ = w
            if self.reference is None:
                ok, why = X.steady(self.loop.key, y, u, self.criteria)
                if ok:
                    self.reference = X.reference_from(y, u)
                    r = self.reference
                    self.message.emit(f"Baseline reference: {self.loop.y_channel} {r.y:.5g} (+-{r.y_sigma:.2g}), "
                                      f"{self.loop.u_channel} {r.u:.5g} (+-{r.u_sigma:.2g})")
            else:
                ok, why = X.recovered(self.loop.key, y, u, self.reference, self.criteria)
            if ok:
                self._check = {"recover_s": round(elapsed, 3), "y": float(np.median(y)), "u": float(np.median(u))}
                self._apply_candidate()
                return
            self._last_why = why
        if elapsed > self.recover_timeout_s:
            self._fail_run(f"the loop did not return to the baseline within {self.recover_timeout_s:g} s "
                           f"({self._last_why or 'no data'}): later conditions would not start from the same state")

    def _apply_candidate(self):
        try:
            self._apply_gains(self._prop.kp, self._prop.ki)
            self._write(self.loop.step_param, self.plan.levels[0])
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self._prev_lost = False
        self._enter("settle")
        self._t_candidate = self._t_state

    def _poll_settle(self, elapsed):
        if elapsed < self.plan.settle_s:
            return
        w = self._window()
        if w is not None:
            y, u, _ = w
            expected = None
            k = self._kappa()
            if k is not None:
                expected = k * self.plan.levels[0]
            ok, why = X.settled_at(self.loop.key, y, u, expected, self.criteria)
            if ok:
                self._check["settle_s"] = round(elapsed, 3)
                self._start_test()
                return
            self._last_why = why
        if elapsed > self.settle_timeout_s:
            self._end_condition(W.StepTestResult(
                kp=self._prop.kp, ki=self._prop.ki, loop=self.plan.loop,
                failure=f"did not settle at these gains within {self.settle_timeout_s:g} s ({self._last_why})"))

    def _start_test(self):
        self._enter("test")
        self._t_test = self._t_state
        cap = self._cap
        now = time.perf_counter()
        for k, (t_rel, value) in enumerate(self.plan.commands()):
            self._later((cap.t0 + self._t_test + t_rel - now) * 1000, self._send_event, k, value)
        self._later((cap.t0 + self._t_test + self.plan.duration - now) * 1000, self._end_test)

    def _send_event(self, k, value):
        t_send = self._cap.now()
        try:
            self._write(self.loop.step_param, value)
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self._events.append((k, t_send))

    def _end_test(self):
        cap = self._cap
        n = cap.count
        events = [t for _, t in sorted(self._events)]
        try:
            self._write(self.loop.step_param, self.plan.base)
        except Exception as e:
            self._fail_run(f"writing to SXM failed: {e}")
            return
        keep = cap.t[:n] >= self._t_candidate
        if keep.sum() < 100 or len(events) != self.plan.n_events:
            res = W.StepTestResult(kp=self._prop.kp, ki=self._prop.ki, loop=self.plan.loop,
                                   failure=f"incomplete recording ({int(keep.sum())} slices, "
                                           f"{len(events)}/{self.plan.n_events} events)")
        else:
            ct = W.CapturedTest(plan=self.plan, t=cap.t[:n][keep].copy(),
                                channels={k: v[:n][keep].copy() for k, v in cap.data.items()},
                                event_times=events, kp=self._prop.kp, ki=self._prop.ki)
            res = W.analyze_test(ct, **self.analysis)
        self._end_condition(res)

    def _end_condition(self, res: W.StepTestResult):
        self._poll.stop()
        self._token += 1                           # drop pending event / end timers of this condition
        self._stop_capture()
        self._state = "idle"
        res.meta["baseline_check"] = dict(self._check)
        res.meta["stage"] = self._prop.stage
        self._prev_lost = bool(res.failure)
        try:
            self._write_baseline()                 # never leave candidate gains in place between conditions
        except Exception as e:
            self.condition_finished.emit(self._prop, res)
            self._fail_run(f"writing to SXM failed: {e}")
            return
        self.condition_finished.emit(self._prop, res)
        self._later(self.PAUSE_MS, self._next)
