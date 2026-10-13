"""
Single Step tab: one change of one parameter, X0 -> X1, recorded for as long as needed on either side.

The Step Test + Scope pair stores every raw sample, which limits a capture to a few tens of seconds. A single
step that is watched for minutes does not need that everywhere, so this tab has its own recorder
(:class:`SingleStepRecorder`):

* the whole record is kept as time slices (mean, min and max per slice, 1 ms by default) with real time stamps;
* a window around the step is also kept at the full driver rate.

The time at X0 and the time at X1 are set separately. X0 is either written at the start or taken as the value
the parameter already has ("already at X0": nothing is written before the step). Time zero is the step.

Export: ``<stem>.csv`` (the slices), ``<stem>_edge.csv`` (the full-rate window), ``<stem>.json`` and
``<stem>.png``, with the SXM settings read when the recording ends.
"""
import datetime
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .. import metadata as MD
from ..common import PARAMS_BASE, confirm_high_voltage, append_log_line
from sxm_anfatec.driver import CHANNELS
from .export_image import save_scene_png
from .sci_spinbox import SciDoubleSpinBox
from .step_test_tab import StepTestTab, _NAME_TAGS

MAX_SLICES = 5_000_000          # per run: longer records need a longer slice
MAX_EDGE_WINDOW_S = 30.0        # full-rate window, before + after the step
RING_RATE_HZ = 300_000          # samples per second the raw ring is sized for (measured: about 85 kHz, 2 channels)
LATE_STEP_S = 10.0              # room in the record for a step sent later than planned
MAX_COLUMNS = 3000              # plotted columns per trace; the visible part is reduced again on every zoom
PENS = ((50, 100, 200), (200, 50, 50))                  # as the Scope tab


class SingleStepRecorder(QtCore.QThread):
    """
    Reads the channels back-to-back until ``after_s`` past the step (or until stopped).

    Kept: per ``bin_s`` slice the mean, min and max of every channel (``t``, ``mean``, ``lo``, ``hi``, valid up
    to ``count``), and the raw samples from ``edge_pre_s`` before to ``edge_post_s`` after the step (``raw_t``,
    ``raw``, set when the thread ends). Times are seconds since the recording started (``time.perf_counter``).

    The step is sent by the GUI thread, which then calls :meth:`mark_edge`. Raw samples go to a ring, so the
    window before the step is there whenever the step comes.

    The GUI thread reads the slices without a lock: a slice is complete before ``count`` is incremented, and
    that is display only.
    """

    error = QtCore.pyqtSignal(str)

    def __init__(self, driver, channels: List[Tuple[str, int, float]], bin_s: float, before_s: float,
                 after_s: float, edge_pre_s: float, edge_post_s: float):
        super().__init__()
        self.driver = driver
        self.channels = channels                # (name, driver index, scale to physical units)
        self.bin_s = bin_s
        self.after_s = after_s
        self.edge_pre_s = edge_pre_s
        self.edge_post_s = edge_post_s
        self.max_s = before_s + after_s + LATE_STEP_S
        n = int(self.max_s / bin_s) + 4
        self.t = np.zeros(n)
        self.mean = [np.zeros(n) for _ in channels]
        self.lo = [np.zeros(n) for _ in channels]
        self.hi = [np.zeros(n) for _ in channels]
        self.count = 0
        self.t0 = 0.0
        self.edge_t: Optional[float] = None     # the step, seconds since t0 (None: not sent)
        self.raw_t = np.zeros(0)
        self.raw = [np.zeros(0) for _ in channels]
        self.rate = 0.0                         # measured: samples (one read per channel each) per second
        self._stop = False

    def stop(self):
        self._stop = True

    def now(self) -> float:
        """Seconds since this recording started, on the recording's clock."""
        return time.perf_counter() - self.t0 if self.t0 else 0.0

    def mark_edge(self, t: float):
        """The step was sent at ``t`` (from :meth:`now`)."""
        self.edge_t = t

    def run(self):
        try:
            self._record()
        except Exception as e:
            self.error.emit(f"Driver read error: {e}")

    def _record(self):
        clock = time.perf_counter
        read = self.driver.read_raw
        idxs = [c[1] for c in self.channels]
        scales = [c[2] for c in self.channels]
        nch = len(idxs)
        window = self.edge_pre_s + self.edge_post_s

        n_ring = int(window * RING_RATE_HZ) + 16 if window > 0 else 0
        ring_t = np.zeros(n_ring)
        ring = [np.zeros(n_ring, dtype=np.int32) for _ in idxs]
        n_post = int(n_ring * self.edge_post_s / window) if window > 0 else 0

        inf = float("inf")
        sums = [0.0] * nch
        los = [inf] * nch
        his = [-inf] * nch
        cnt = 0
        total = 0
        j = 0                                   # raw samples written to the ring
        j_stop = None                           # ring count at which the window after the step is full
        raw_until = inf
        raw_on = n_ring > 0
        edge_seen = False
        n_max = len(self.t) - 1
        self.t0 = t0 = clock()
        start = t0
        edge = t0 + self.bin_s
        t_end = t0 + self.max_s
        yield_at = t0 + 0.001
        now = t0
        while not self._stop:
            now = clock()
            for q in range(nch):
                v = read(idxs[q])
                sums[q] += v
                if v < los[q]:
                    los[q] = v
                if v > his[q]:
                    his[q] = v
                if raw_on:
                    ring[q][j % n_ring] = v
            cnt += 1
            total += 1
            if raw_on:
                ring_t[j % n_ring] = now
                j += 1
                if j_stop is not None and (j >= j_stop or now >= raw_until):
                    raw_on = False
            if not edge_seen and self.edge_t is not None:
                edge_seen = True
                t_end = t0 + self.edge_t + self.after_s
                raw_until = t0 + self.edge_t + self.edge_post_s
                j_stop = j + n_post
            if now >= edge:
                i = self.count
                self.t[i] = 0.5 * (start + now) - t0
                for q in range(nch):
                    a, b = los[q] * scales[q], his[q] * scales[q]
                    self.mean[q][i] = sums[q] / cnt * scales[q]
                    self.lo[q][i] = min(a, b)   # a negative scale (Topo) swaps them
                    self.hi[q][i] = max(a, b)
                    sums[q] = 0.0
                    los[q] = inf
                    his[q] = -inf
                cnt = 0
                start = now
                edge = now + self.bin_s
                self.count = i + 1
                if now >= t_end or self.count >= n_max:
                    break
            if now >= yield_at:                 # hand the GIL to the GUI thread every ms: its timer sends the step
                yield_at = now + 0.001
                time.sleep(0)

        self.rate = total / (now - t0) if now > t0 else 0.0
        if n_ring and self.edge_t is not None:
            n = min(j, n_ring)
            order = np.arange(j - n, j) % n_ring
            rt = ring_t[order] - t0
            keep = rt >= self.edge_t - self.edge_pre_s
            self.raw_t = rt[keep]
            self.raw = [ring[q][order][keep] * scales[q] for q in range(nch)]


class _OfflineSignal:
    """Stands in for the driver when there is none: every channel relaxes towards the last value sent."""

    def __init__(self, level: float):
        self.target = self.level = level
        self._t = time.perf_counter()
        self._scale = {c[0]: c[3] for c in CHANNELS.values()}

    def read_raw(self, idx):
        time.sleep(0)                           # as the driver call does: let go of the GIL on every read
        now = time.perf_counter()
        self.level += (self.target - self.level) * min(1.0, (now - self._t) / 0.2)
        self._t = now
        noisy = self.level + 0.01 * max(abs(self.level), 1e-3) * random.gauss(0.0, 1.0)
        return int(noisy / self._scale.get(idx, 1.0))


def _columns(t, mean, lo, hi, columns: int = MAX_COLUMNS):
    """Slices merged into at most ``columns`` columns: mean of the means, min of the mins, max of the maxes."""
    n = len(t)
    if n <= columns:
        return t, mean, lo, hi
    per = n // columns
    m = per * columns
    shape = (columns, per)
    return (t[:m].reshape(shape).mean(axis=1), mean[:m].reshape(shape).mean(axis=1),
            lo[:m].reshape(shape).min(axis=1), hi[:m].reshape(shape).max(axis=1))


def _unit(name: str) -> str:
    unit = CHANNELS[name][2] if name in CHANNELS else ""
    return {"°": "deg", "*": "deg"}.get(unit, unit)


class SingleStepTab(QtWidgets.QWidget):
    """One step X0 -> X1 of one parameter, with its own recorder, plot and export."""

    def __init__(self, dde_client, driver=None, reader=None):
        """
        Args:
            dde_client (object): DDE client to communicate with SXM software.
            driver (SXMIOCTL or None): the shared IOCTL driver handle. None: offline, synthetic signals.
            reader (sxm_state.SXMReader, optional): SXM GUI read-back, for X0 and the export settings.
        """
        super().__init__()
        self.dde = dde_client
        self.driver = driver
        self.reader = reader
        self._customs: List[Tuple[str, object, str]] = []
        self._rec: Optional[SingleStepRecorder] = None
        self._offline: Optional[_OfflineSignal] = None
        self._run: Optional[dict] = None        # the run as started, plus what happened
        self.result: Optional[dict] = None      # the last finished recording
        self.last_meta: Optional[MD.Metadata] = None
        self.last_export_path = None
        self._view = None                       # (t, mean[], lo[], hi[]) being plotted
        self._fit = True                        # show the whole record
        self._drawing = False

        v = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)

        # --- the step ---
        self.param = QtWidgets.QComboBox()
        self._populate_params()
        self.x0 = SciDoubleSpinBox()
        self.x0.setDecimals(6); self.x0.setRange(-1e12, 1e12)
        self.x1 = SciDoubleSpinBox()
        self.x1.setDecimals(6); self.x1.setRange(-1e12, 1e12)
        self.btn_x0_sxm = QtWidgets.QPushButton("= SXM value")
        self.btn_x0_sxm.setToolTip("Read the selected parameter's current value from the SXM window into X0.")
        self.btn_x0_sxm.clicked.connect(self.x0_from_sxm)
        self.before = QtWidgets.QDoubleSpinBox()
        self.before.setDecimals(1); self.before.setRange(0.2, 3600); self.before.setValue(5.0)
        self.before.setToolTip("How long X0 is recorded before the step.")
        self.after = QtWidgets.QDoubleSpinBox()
        self.after.setDecimals(1); self.after.setRange(0.2, 7200); self.after.setValue(60.0)
        self.after.setToolTip("How long X1 is recorded after the step.")

        grid.addWidget(self.param, 0, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("X0 (before):"), 0, 2); grid.addWidget(self.x0, 0, 3)
        grid.addWidget(self.btn_x0_sxm, 0, 4)
        grid.addWidget(QtWidgets.QLabel("X1 (after):"), 0, 5); grid.addWidget(self.x1, 0, 6)
        grid.addWidget(QtWidgets.QLabel("Time at X0 (s):"), 1, 2); grid.addWidget(self.before, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Time at X1 (s):"), 1, 5); grid.addWidget(self.after, 1, 6)

        self.chk_write_x0 = QtWidgets.QCheckBox("Write X0 at start")
        self.chk_write_x0.setToolTip(
            "Off: the parameter is taken to be at X0 and settled already. Nothing is written before the\n"
            "step, and X0 is read from SXM at Start when it can be.\n"
            "On: X0 is written when the recording starts, so 'Time at X0' is also the settling time."
        )
        self.chk_return = QtWidgets.QCheckBox("Return to X0 at the end / on Stop")
        self.chk_return.setToolTip("Write X0 back once the recording has ended (not recorded).")
        grid.addWidget(self.chk_write_x0, 2, 0, 1, 3)
        grid.addWidget(self.chk_return, 2, 3, 1, 4)

        # --- the recording ---
        names = list(CHANNELS.keys())
        self.chan1_combo = QtWidgets.QComboBox(); self.chan1_combo.addItems(names)
        self.chan2_combo = QtWidgets.QComboBox(); self.chan2_combo.addItems(names)
        if "QPlusAmpl" in CHANNELS:
            self.chan1_combo.setCurrentIndex(names.index("QPlusAmpl"))
        if "Drive" in CHANNELS:
            self.chan2_combo.setCurrentIndex(names.index("Drive"))
        self.bin_ms = QtWidgets.QDoubleSpinBox()
        self.bin_ms.setDecimals(1); self.bin_ms.setRange(0.5, 1000); self.bin_ms.setValue(1.0)
        self.bin_ms.setToolTip(
            "The whole record is stored as slices of this length: mean, min and max of each channel.\n"
            "Noise read from the means depends on it."
        )
        self.edge_pre = QtWidgets.QDoubleSpinBox()
        self.edge_pre.setDecimals(1); self.edge_pre.setRange(0, MAX_EDGE_WINDOW_S); self.edge_pre.setValue(1.0)
        self.edge_post = QtWidgets.QDoubleSpinBox()
        self.edge_post.setDecimals(1); self.edge_post.setRange(0, MAX_EDGE_WINDOW_S); self.edge_post.setValue(5.0)
        tip = (f"Every sample is also kept from this long before to this long after the step\n"
               f"(at most {MAX_EDGE_WINDOW_S:.0f} s together; 0 and 0: slices only).")
        self.edge_pre.setToolTip(tip); self.edge_post.setToolTip(tip)

        rec = QtWidgets.QHBoxLayout()
        rec.addWidget(QtWidgets.QLabel("Channel 1:")); rec.addWidget(self.chan1_combo)
        rec.addWidget(QtWidgets.QLabel("Channel 2:")); rec.addWidget(self.chan2_combo)
        rec.addWidget(QtWidgets.QLabel("Slice (ms):")); rec.addWidget(self.bin_ms)
        rec.addWidget(QtWidgets.QLabel("Full rate around the step (s): before")); rec.addWidget(self.edge_pre)
        rec.addWidget(QtWidgets.QLabel("after")); rec.addWidget(self.edge_post)
        rec.addStretch(1)
        v.addLayout(rec)

        # --- run ---
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_export.setEnabled(False)
        self.btn_view_all = QtWidgets.QPushButton("Whole record")
        self.btn_view_edge = QtWidgets.QPushButton("Zoom to step")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_view_all.clicked.connect(self.view_all)
        self.btn_view_edge.clicked.connect(self.view_edge)
        self.status_label = QtWidgets.QLabel("No recording yet.")
        self.status_label.setStyleSheet("QLabel { color: #555; }")
        run = QtWidgets.QHBoxLayout()
        for b in (self.btn_start, self.btn_stop, self.btn_export, self.btn_view_all, self.btn_view_edge):
            run.addWidget(b)
        run.addWidget(self.status_label, 1)
        v.addLayout(run)

        # --- plots: slices as a line (mean) in a band (min..max), the full-rate window drawn over them ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        v.addWidget(self.plot_widget, 1)
        self.plots, self._items = [], []
        for row, color in enumerate(PENS):
            p = self.plot_widget.addPlot(row=row, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.enableAutoRange(x=False, y=True)
            lo = pg.PlotCurveItem(pen=pg.mkPen(None))     # curve items: a PlotDataItem without a pen has no path to fill
            hi = pg.PlotCurveItem(pen=pg.mkPen(None))
            band = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(*color, 50))
            raw = pg.PlotDataItem(pen=pg.mkPen(color=(120, 120, 120), width=1))
            raw.setDownsampling(auto=True, method="peak")
            raw.setClipToView(True)
            mean = pg.PlotDataItem(pen=pg.mkPen(color=color, width=2))
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=1, style=QtCore.Qt.DashLine))
            line.hide()
            for item in (lo, hi, raw, mean):
                p.addItem(item)
            for item in (band, line):           # their y bounds include 0: they would drag the y range there
                p.addItem(item, ignoreBounds=True)
            p.getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
            self.plots.append(p)
            self._items.append({"lo": lo, "hi": hi, "raw": raw, "mean": mean, "line": line})
        self.plots[1].setXLink(self.plots[0])
        self.plots[1].setLabel("bottom", "Time since the step (s)")
        self.plots[0].sigXRangeChanged.connect(lambda *_a: self._redraw())

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        v.addWidget(self.log)

        self._step_timer = QtCore.QTimer(self)
        self._step_timer.setSingleShot(True)
        self._step_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._step_timer.timeout.connect(self._do_step)
        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(500)
        self._ui_timer.timeout.connect(self._refresh_live)

    # ---------- parameters ----------
    def set_custom_params(self, customs: List[Tuple[str, object, str]]):
        """Sets custom parameters received from ParamsTab: list of (ptype, pcode, label)."""
        self._customs = customs[:]
        self._populate_params()

    def _populate_params(self):
        self.param.blockSignals(True)
        current = self.param.currentText() if self.param.count() else None
        self.param.clear()
        for _k, ptype, pcode, label, _v in PARAMS_BASE:
            self.param.addItem(label, (ptype, pcode, label))
        for (ptype, pcode, label) in self._customs:
            self.param.addItem(label, (ptype, pcode, label))
        if current:
            idx = self.param.findText(current, QtCore.Qt.MatchExactly)
            if idx >= 0:
                self.param.setCurrentIndex(idx)
        self.param.blockSignals(False)

    def _log_line(self, text: str) -> None:
        append_log_line(self.log, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {text}")

    def _sxm_value(self):
        """The selected parameter as SXM shows it now, or None (reason logged)."""
        ptype, pcode, label = self.param.currentData()
        if self.reader is None:
            self._log_line("SXM read-back is not available.")
            return None
        try:
            value = self.reader.read().by_code(ptype, pcode)
        except Exception as e:
            self._log_line(f"Read from SXM failed: {e}")
            return None
        if value is None:
            self._log_line(f"{label}: no SXM value (not running, or not a readable parameter).")
        return value

    def x0_from_sxm(self) -> bool:
        """Set X0 to the value SXM shows for the selected parameter. Returns True on success."""
        value = self._sxm_value()
        if value is None:
            return False
        self.x0.setValue(value)
        self._log_line(f"X0 <- SXM {self.param.currentData()[2]} = {value}")
        return True

    def _send(self, value: float, what: str) -> bool:
        """Send one value and log it. The high-voltage guard runs once, in start(): no dialog during a run."""
        ptype, pcode, label = self._run["param"]
        try:
            if ptype == "EDIT":
                self.dde.send_scanpara(str(pcode), value)
            else:
                self.dde.send_dncpara(int(pcode), value)
        except Exception as e:
            self._log_line(f"SEND ERROR: {e}")
            return False
        if self._offline is not None:
            self._offline.target = value
        code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
        self._log_line(f"{what}: set {label} ({code_text}) to {value}")
        return True

    # ---------- run ----------
    def is_running(self) -> bool:
        return self._rec is not None

    def start(self):
        """Start the recording; the step follows after 'Time at X0'."""
        if self.is_running():
            return
        ptype, pcode, label = self.param.currentData()
        write_x0 = self.chk_write_x0.isChecked()
        if not write_x0 and self.reader is not None:
            found = self._sxm_value()               # 'already at X0': X0 is what SXM shows, not what was typed
            if found is not None:
                self.x0.setValue(found)
        x0, x1 = self.x0.value(), self.x1.value()
        if x1 == x0:
            self._log_line("Not started: X1 equals X0.")
            return
        bin_s = self.bin_ms.value() / 1000.0
        before, after = self.before.value(), self.after.value()
        pre, post = min(self.edge_pre.value(), before), min(self.edge_post.value(), after)
        if (before + after + LATE_STEP_S) / bin_s > MAX_SLICES:
            self._log_line(f"Not started: more than {MAX_SLICES:,} slices. Use a longer slice.")
            return
        if pre + post > MAX_EDGE_WINDOW_S:
            self._log_line(f"Not started: the full-rate window is limited to {MAX_EDGE_WINDOW_S:.0f} s in total.")
            return
        do_return = self.chk_return.isChecked()
        if StepTestTab._is_voltage_like(ptype, pcode):
            for value in ([x0] if (write_x0 or do_return) else []) + [x1]:
                if not confirm_high_voltage(self, label, value):
                    return

        names = [self.chan1_combo.currentText(), self.chan2_combo.currentText()]
        key = next((k for k, pt, pc, _l, _v in PARAMS_BASE if (pt, pc) == (ptype, pcode)), None)
        self._run = {
            "param": (ptype, pcode, label),
            "parameter": label,
            "code": pcode if ptype == "EDIT" else f"DNC {pcode}",
            "name_tag": _NAME_TAGS.get(key, str(pcode) if ptype == "EDIT" else f"DNC{pcode}"),
            "x0": x0, "x1": x1, "x0_written": write_x0, "return_to_x0": do_return,
            "before_s": before, "after_s": after, "bin_s": bin_s, "edge_pre_s": pre, "edge_post_s": post,
            "channels": names, "started": datetime.datetime.now(), "offline": self.driver is None,
            "step_sent": False, "send_s": None, "step_time": None,
        }
        self._offline = _OfflineSignal(x0) if self.driver is None else None
        channels = [(n, CHANNELS[n][0], CHANNELS[n][3]) for n in names]
        self._rec = SingleStepRecorder(self._offline or self.driver, channels, bin_s, before, after, pre, post)
        self._rec.error.connect(self._log_line)
        self._rec.finished.connect(self._finish)

        self.result = None
        self.last_meta = None
        self._fit = True
        for p, name in zip(self.plots, names):
            p.setLabel("left", f"{name} ({_unit(name)})" if _unit(name) else name)
        self.plots[1].setLabel("bottom", "Time since the start (s)")
        for items in self._items:
            items["raw"].setData([], [])
            items["line"].hide()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)

        self._rec.start()
        if write_x0 and not self._send(x0, "X0"):
            self.stop()
            return
        if not write_x0:
            self._log_line(f"Recording at X0 = {x0} (as found, nothing written).")
        self._step_timer.start(int(before * 1000))
        self._ui_timer.start()

    def _do_step(self):
        rec = self._rec
        if rec is None:
            return
        t_a = rec.now()
        if not self._send(self._run["x1"], "Step"):
            self.stop()
            return
        rec.mark_edge(t_a)
        self._run.update(step_sent=True, send_s=rec.now() - t_a, step_time=datetime.datetime.now())

    def stop(self):
        """Stop now. What was recorded is kept; X0 is written back if asked for and the step was sent."""
        rec = self._rec
        if rec is None:
            return
        rec.stop()
        rec.wait(2000)
        self._finish()

    def _finish(self):
        rec = self._rec
        if rec is None:                             # already finished (Stop, then the thread's finished signal)
            return
        self._rec = None
        self._step_timer.stop()
        self._ui_timer.stop()
        rec.wait(2000)
        run = self._run

        n = rec.count
        edge = rec.edge_t if run["step_sent"] else None
        ref = edge if edge is not None else 0.0
        self.result = {
            "t": rec.t[:n] - ref, "mean": [a[:n] for a in rec.mean], "lo": [a[:n] for a in rec.lo],
            "hi": [a[:n] for a in rec.hi], "raw_t": rec.raw_t - ref, "raw": rec.raw, "rate": rec.rate,
            "edge_s": edge, "names": list(run["channels"]),
        }
        # the SXM settings of this recording, read before X0 is written back so that they show the X1 state
        self.last_meta = self._metadata()
        if run["return_to_x0"] and run["step_sent"]:
            self._send(run["x0"], "Back to X0")
        rec.deleteLater()

        self._view = (self.result["t"], self.result["mean"], self.result["lo"], self.result["hi"])
        self.plots[1].setLabel("bottom", "Time since the step (s)" if edge is not None else "Time since the start (s)")
        for q, items in enumerate(self._items):
            items["raw"].setData(self.result["raw_t"], self.result["raw"][q])
            items["line"].setValue(0.0)
            items["line"].setVisible(edge is not None)
        self._fit = True
        self._redraw()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(n > 0)
        t = self.result["t"]
        if edge is None:
            self.status_label.setText(f"Stopped before the step: {n:,} slices recorded.")
        else:
            self.status_label.setText(
                f"{-t[0]:.1f} s at X0, {t[-1]:.1f} s at X1: {n:,} slices of {run['bin_s'] * 1e3:g} ms, "
                f"{len(self.result['raw_t']):,} full-rate samples around the step "
                f"(measured rate: {rec.rate:,.0f} Hz)")
        self._log_line("Recording finished." if edge is not None else "Recording stopped before the step.")
        rt = self.result["raw_t"]
        if len(rt) and (rt[0] > -run["edge_pre_s"] + 0.05 or rt[-1] < min(run["edge_post_s"], t[-1]) - 0.05):
            self._log_line(f"The full-rate window is shorter than asked for: {rt[0]:.2f} to {rt[-1]:.2f} s "
                           f"(reads came faster than the {RING_RATE_HZ:,} Hz it is sized for, or the run was stopped).")

    def _refresh_live(self):
        rec = self._rec
        if rec is None:
            return
        run = self._run
        now = rec.now()
        edge = rec.edge_t
        # no redraw while the full-rate window is being recorded: the plot code holds the GIL and the
        # recorder would lose samples there
        quiet_from = run["before_s"] - run["edge_pre_s"] - 1.0
        if now < quiet_from or (edge is not None and now > edge + run["edge_post_s"] + 0.2):
            n = rec.count
            self._view = (rec.t[:n], [a[:n] for a in rec.mean], [a[:n] for a in rec.lo], [a[:n] for a in rec.hi])
            for items in self._items:
                if edge is not None:
                    items["line"].setValue(edge)
                items["line"].setVisible(edge is not None)
            self._redraw()
        if edge is None:
            self.status_label.setText(f"At X0: {now:.1f} of {self._run['before_s']:.1f} s")
        else:
            self.status_label.setText(f"At X1: {now - edge:.1f} of {self._run['after_s']:.1f} s")

    # ---------- plot ----------
    def _on_manual_range(self, *_args):
        self._fit = False

    def view_all(self):
        self._fit = True
        self._redraw()

    def view_edge(self):
        """Show the full-rate window (or, without one, a second either side of the step)."""
        if self.result is None or self.result["edge_s"] is None:
            return
        rt = self.result["raw_t"]
        lo, hi = (rt[0], rt[-1]) if len(rt) > 1 else (-1.0, 1.0)
        self._fit = False
        self.plots[0].setXRange(lo, hi, padding=0.02)

    def _redraw(self):
        """Draw the slices in view, merged to at most MAX_COLUMNS columns (so zooming in shows them all)."""
        if self._drawing or self._view is None:
            return
        t, mean, lo, hi = self._view
        if len(t) == 0:
            return
        self._drawing = True
        try:
            if self._fit:
                self.plots[0].setXRange(t[0], t[-1], padding=0.02)
            x0, x1 = self.plots[0].viewRange()[0]
            i0 = max(int(np.searchsorted(t, x0)) - 1, 0)
            i1 = min(int(np.searchsorted(t, x1)) + 1, len(t))
            for q, items in enumerate(self._items):
                tt, m, l, h = _columns(t[i0:i1], mean[q][i0:i1], lo[q][i0:i1], hi[q][i0:i1])
                items["lo"].setData(tt, l)
                items["hi"].setData(tt, h)
                items["mean"].setData(tt, m)
        finally:
            self._drawing = False

    # ---------- export ----------
    def _metadata(self) -> MD.Metadata:
        """SXM read-back now, plus what this recording was."""
        run, res = self._run, self.result
        t, rt = res["t"], res["raw_t"]
        edge = res["edge_s"]
        names = res["names"]
        capture = [
            ("Channels", " / ".join(f"{c} [{_unit(c)}]" for c in names), ""),
            ("Slice", run["bin_s"] * 1e3, "ms"),
            ("Slices", len(t), ""),
            ("Rate (measured)", res["rate"], "Hz"),
            ("Full-rate samples", len(rt), ""),
            ("Full-rate window", f"{rt[0]:.3f} to {rt[-1]:.3f} s" if len(rt) else "none", ""),
            ("Started", run["started"].strftime("%Y-%m-%d %H:%M:%S"), ""),
        ]
        if run["offline"]:
            capture.append(("Data", "OFFLINE MOCK SIGNALS (no driver)", ""))
        step = [
            ("Parameter", f"{run['parameter']} ({run['code']})", ""),
            ("X0", run["x0"], ""),
            ("X0 written at start", "yes" if run["x0_written"] else "no (as found)", ""),
            ("X1", run["x1"], ""),
        ]
        if edge is None:
            step.append(("Step", "not sent (stopped before)", ""))
        else:
            step += [
                ("Time at X0", float(-t[0]) if len(t) else None, "s"),
                ("Time at X1", float(t[-1]) if len(t) else None, "s"),
                ("Step sent", run["step_time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], ""),
                ("Sending took", run["send_s"] * 1e3, "ms"),
            ]
        step.append(("Return to X0", "yes" if run["return_to_x0"] else "no", ""))
        return MD.collect(self.reader, [("Capture", capture), ("Single Step", step)])

    def default_export_name(self) -> str:
        """``<date-time>_singlestep_<param><X0>-<X1>_<chan1>-<chan2>_<key settings>`` (no extension)."""
        meta = self.last_meta or MD.Metadata.unavailable("none")
        run = self._run
        c1, c2 = run["channels"]
        loops = ("pll", "afl") if {c1, c2} & {"df", "Phase"} else ("afl", "pll")
        detail = f"{run['name_tag']}{MD.short_value(run['x0'])}-{MD.short_value(run['x1'])}_{c1}-{c2}"
        return meta.filename("singlestep", detail, loops=loops)

    def export_data(self):
        if self.result is None:
            return
        default_name = self.default_export_name() + ".csv"
        if self.last_export_path:
            default_name = os.path.join(os.path.dirname(self.last_export_path), default_name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Data", default_name, "CSV Files (*.csv)")
        if not path:
            return
        self.last_export_path = path
        try:
            written = self.export_to(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))
            return
        QtWidgets.QMessageBox.information(self, "Export", "Saved:\n" + "\n".join(written))

    def export_to(self, path: str) -> List[str]:
        """
        Write the last recording: ``path`` (the slices; time, then mean / min / max per channel),
        ``<stem>_edge.csv`` (the full-rate window, if any), ``<stem>.json`` and ``<stem>.png``.
        Returns the paths written.
        """
        res, run, meta = self.result, self._run, self.last_meta
        stem = os.path.splitext(path)[0]
        names = res["names"]
        tagged = [f"{c}_{_unit(c)}" if _unit(c) else c for c in names]
        columns = ["time_s"] + [f"{c}_{kind}" for c in tagged for kind in ("mean", "min", "max")]
        written = [path]
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(MD.csv_preamble(meta.header_lines("SXM nc-AFM single step, time slices")))
            f.write(",".join(columns) + "\n")
            cols = [res["t"]]
            for q in range(len(names)):
                cols += [res["mean"][q], res["lo"][q], res["hi"][q]]
            np.savetxt(f, np.column_stack(cols), delimiter=",")
        edge_columns = ["time_s"] + tagged
        edge_path = None
        if len(res["raw_t"]):
            edge_path = stem + "_edge.csv"
            with open(edge_path, "w", encoding="utf-8", newline="") as f:
                f.write(MD.csv_preamble(meta.header_lines("SXM nc-AFM single step, full rate around the step")))
                f.write(",".join(edge_columns) + "\n")
                np.savetxt(f, np.column_stack([res["raw_t"]] + list(res["raw"])), delimiter=",")
            written.append(edge_path)
        written.append(MD.write_sidecar(stem + ".json", meta, {
            "kind": "ncafm_single_step", "data_file": os.path.basename(path), "columns": columns,
            "edge_file": os.path.basename(edge_path) if edge_path else None, "edge_columns": edge_columns,
            "slices": int(len(res["t"])), "slice_s": run["bin_s"], "rate_hz": res["rate"],
            "edge_samples": int(len(res["raw_t"])),
            "time_zero": "the step" if res["edge_s"] is not None else "the start (no step was sent)",
            "single_step": {
                "parameter": run["parameter"], "code": run["code"], "x0": run["x0"], "x1": run["x1"],
                "x0_written": run["x0_written"], "return_to_x0": run["return_to_x0"],
                "step_sent": run["step_sent"], "step_s_after_start": res["edge_s"],
                "send_s": run["send_s"]},
        }))
        try:
            save_scene_png(self.plot_widget.scene(), stem + ".png", legend=meta.legend())
            written.append(stem + ".png")
        except Exception as img_e:
            written.append(f"(plot image not saved: {img_e})")
        return written
