"""
Step Test Tab for NC-AFM Control Suite.

Provides square-wave style parameter stepping with optional integration
to the ScopeTab for signal capture and event annotation.

Features:
    - Select a parameter (including custom EditXX)
    - Define low/high values, period, and step count
    - Visual preview of waveform
    - Triggers Scope capture if enabled
    - Optionally stops Scope capture with test
    - Annotates Scope plots with timing events
"""

import datetime
from typing import List, Tuple
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from ..common import PARAMS_BASE, confirm_high_voltage, append_log_line
from .sci_spinbox import SciDoubleSpinBox


class StepTestTab(QtWidgets.QWidget):
    """Square wave parameter stepping with optional scope trigger."""

    def __init__(self, dde_client):
        """
        Args:
            dde_client (object): DDE client to communicate with SXM software.
        """
        super().__init__()
        self.dde = dde_client
        self.step_index = 0
        self._customs: List[Tuple[str, object, str]] = []
        self.scope_tab = None  # linked externally

        v = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)

        # --- Controls ---
        self.param = QtWidgets.QComboBox()
        self._populate_params()

        self.low = SciDoubleSpinBox()          # Low / High / Base take and show scientific notation (Kp, Ki: 2.5e8)
        self.low.setDecimals(6); self.low.setRange(-1e12, 1e12); self.low.setValue(10.0)
        self.high = SciDoubleSpinBox()
        self.high.setDecimals(6); self.high.setRange(-1e12, 1e12); self.high.setValue(101.0)
        self.period = QtWidgets.QDoubleSpinBox()
        self.period.setDecimals(3); self.period.setRange(0, 3600); self.period.setValue(1.0)
        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(1, 1_000_000); self.steps.setValue(20)

        # Base value the parameter returns to when the test ends or is stopped.
        # SXM parameters are write-only from here, so the original value cannot be
        # read back; the base is explicit. It follows the midpoint of low/high (the
        # centre of a symmetric +-delta test, e.g. f0 +- 1 Hz) until edited by hand.
        self.chk_restore = QtWidgets.QCheckBox("Return to base value at end / on Stop")
        self.chk_restore.setToolTip(
            "After the last step (which is held for one full period), or as soon as you press\n"
            "Stop, write the Base value back to the parameter so the test does not leave it\n"
            "at Low or High."
        )
        self.base = SciDoubleSpinBox()
        self.base.setDecimals(6); self.base.setRange(-1e12, 1e12)
        self.base.setEnabled(False)
        self.base.setToolTip(
            "Value written back after the test. Defaults to the midpoint of Low and High.\n"
            "SXM parameters cannot be read back from here, so enter the true original value\n"
            "if it is not the midpoint."
        )
        self.btn_base_mid = QtWidgets.QPushButton("= midpoint")
        self.btn_base_mid.setEnabled(False)
        self._base_manual = False  # True once the user typed a base value

        c = 0
        grid.addWidget(self.param, 0, c, 1, 2); c += 2
        grid.addWidget(QtWidgets.QLabel("Low:"), 0, c); c += 1; grid.addWidget(self.low, 0, c); c += 1
        grid.addWidget(QtWidgets.QLabel("High:"), 0, c); c += 1; grid.addWidget(self.high, 0, c); c += 1
        grid.addWidget(QtWidgets.QLabel("Period (s):"), 0, c); c += 1; grid.addWidget(self.period, 0, c); c += 1
        grid.addWidget(QtWidgets.QLabel("Steps:"), 0, c); c += 1; grid.addWidget(self.steps, 0, c); c += 1

        self.btn_preview = QtWidgets.QPushButton("Preview")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_preview.clicked.connect(self.preview)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        grid.addWidget(self.btn_preview, 0, c); c += 1
        grid.addWidget(self.btn_start, 0, c); c += 1
        grid.addWidget(self.btn_stop, 0, c)

        # --- Scope integration ---
        self.chk_trigger_scope = QtWidgets.QCheckBox("Trigger scope capture at start")
        self.chk_stop_scope = QtWidgets.QCheckBox("Stop scope with Step Test")
        grid.addWidget(self.chk_trigger_scope, 1, 0, 1, 4)
        grid.addWidget(self.chk_stop_scope, 2, 0, 1, 4)

        # --- Base value / restore ---
        grid.addWidget(self.chk_restore, 3, 0, 1, 4)
        grid.addWidget(QtWidgets.QLabel("Base:"), 3, 4)
        grid.addWidget(self.base, 3, 5)
        grid.addWidget(self.btn_base_mid, 3, 6, 1, 2)
        self.chk_restore.toggled.connect(self.base.setEnabled)
        self.chk_restore.toggled.connect(self.btn_base_mid.setEnabled)
        self.low.valueChanged.connect(self._sync_base_default)
        self.high.valueChanged.connect(self._sync_base_default)
        self.base.valueChanged.connect(self._on_base_edited)
        self.btn_base_mid.clicked.connect(self._reset_base_to_midpoint)
        self._sync_base_default()

        self.tabs_widget = None
        self.scope_tab_index = None

        # --- Plot ---
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        axis_pen = pg.mkPen(color="k", width=1)
        for ax in ["bottom", "left"]:
            self.plot.getAxis(ax).setPen(axis_pen)
            self.plot.getAxis(ax).setTextPen("k")
            self.plot.getAxis(ax).setStyle(tickTextOffset=5, **{"tickFont": QtGui.QFont("", 10)})
        self.plot.getPlotItem().layout.setContentsMargins(50, 10, 10, 40)
        self.plot.setLabel("left", "Value")
        self.plot.setLabel("bottom", "Time", units="s")
        v.addWidget(self.plot)

        # --- Log ---
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        v.addWidget(self.log)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._events = []  # list of (QtCore.QDateTime, str) for scope overlay

    def set_custom_params(self, customs: List[Tuple[str, object, str]]):
        """Sets custom parameters received from ParamsTab.

        Args:
            customs (list): List of (ptype, pcode, label)
        """
        self._customs = customs[:]
        self._populate_params()

    def _populate_params(self):
        """Populates dropdown with base and custom parameters."""
        self.param.blockSignals(True)
        current = self.param.currentText() if self.param.count() else None
        self.param.clear()
        for _k, ptype, pcode, label, _v in PARAMS_BASE:
            self.param.addItem(label, (ptype, pcode, label))
        for (ptype, pcode, label) in getattr(self, "_customs", []):
            self.param.addItem(label, (ptype, pcode, label))
        if current:
            idx = self.param.findText(current, QtCore.Qt.MatchExactly)
            if idx >= 0:
                self.param.setCurrentIndex(idx)
        self.param.blockSignals(False)

    # ---------- base value ----------
    def _sync_base_default(self):
        """Keep Base at the Low/High midpoint until the user sets it by hand."""
        if self._base_manual:
            return
        self.base.blockSignals(True)
        self.base.setValue(0.5 * (self.low.value() + self.high.value()))
        self.base.blockSignals(False)

    def _on_base_edited(self, _value):
        self._base_manual = True

    def _reset_base_to_midpoint(self):
        self._base_manual = False
        self._sync_base_default()

    @staticmethod
    def _is_voltage_like(ptype, pcode) -> bool:
        return (ptype == "EDIT" and str(pcode).lower() == "edit23") or (
            ptype == "DNC" and int(pcode) == 4
        )

    def _send_value(self, ptype, pcode, label, value, base=False) -> bool:
        """Send one value (with the high-voltage guard), log it and record a scope event.

        Returns True if the value was sent, False if it was declined or failed.
        """
        if self._is_voltage_like(ptype, pcode) and not confirm_high_voltage(self, label, value):
            return False

        try:
            if ptype == "EDIT":
                self.dde.send_scanpara(str(pcode), value)
            else:
                self.dde.send_dncpara(int(pcode), value)
        except Exception as e:
            append_log_line(self.log, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SEND ERROR: {e}")
            return False

        self._events.append((QtCore.QDateTime.currentDateTime(),
                             f"{label}={value:.10g}" + (" (base)" if base else "")))

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
        verb = "Restored base:" if base else "Set"
        append_log_line(self.log, f"[{ts}] {verb} {label} ({code_text}) to {value}")
        return True

    def preview(self):
        """Generates a step waveform preview in the plot area."""
        low, high, T, n = self.low.value(), self.high.value(), self.period.value(), self.steps.value()
        x = [0.0]; y = []
        for i in range(n):
            x.append((i + 1) * T)
            y.append(low if (i % 2 == 0) else high)
        restore = self.chk_restore.isChecked()
        if restore:  # one extra period at the end where the base value is written back
            x.append((n + 1) * T)
            y.append(self.base.value())
            n += 1
        self.plot.clear()
        self.plot.plot(x, y, stepMode=True, pen=pg.mkPen("b", width=2))
        self.plot.setXRange(0, n * T, padding=0.02)
        ys = [low, high, self.base.value()] if restore else [low, high]
        ymin, ymax = min(ys), max(ys)
        m = 0.05 * max(1.0, abs(ymax - ymin))
        self.plot.setYRange(ymin - m, ymax + m, padding=0.02)

    def start(self):
        """Begins step test and triggers scope if selected."""
        self.preview()
        self.step_index = 0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._events = []  # clear event list for overlay
        self._timer.start(int(self.period.value() * 1000))

        if self.chk_trigger_scope.isChecked() and self.scope_tab:
            # Size the capture from the test's own duration, not whatever
            # happens to be in the Scope tab's "Samples to acquire" field -
            # otherwise a capture that finishes early leaves the back half
            # of the test's events with nowhere valid to be drawn.
            n_periods = self.steps.value() + (1 if self.chk_restore.isChecked() else 0)
            test_duration_s = n_periods * self.period.value()
            npts = self.scope_tab.estimate_capture_npoints(test_duration_s)
            if npts >= self.scope_tab.npoints_spin.maximum():
                append_log_line(
                    self.log,
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Warning: this test "
                    f"(~{test_duration_s:.0f} s) may be too long for Scope to fully capture "
                    f"in one shot even at its maximum sample count - consider the Live Scope "
                    f"tab for tests this long."
                )
            self.scope_tab.start_capture(npoints_override=npts)
        if self.tabs_widget and self.scope_tab_index is not None:
            self.tabs_widget.setCurrentIndex(self.scope_tab_index)

    def stop(self, restore=True):
        """Stops the step test and optionally stops the scope.

        If "Return to base value" is checked and at least one step was sent, the base
        value is written back first (unless restore=False, used when the base was
        already written by the final tick).
        """
        if self._timer.isActive():
            self._timer.stop()
            append_log_line(self.log, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Test stopped.")
            if restore and self.chk_restore.isChecked() and self.step_index > 0:
                ptype, pcode, label = self.param.currentData()
                self._send_value(ptype, pcode, label, self.base.value(), base=True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if self.chk_trigger_scope.isChecked() and self.chk_stop_scope.isChecked() and self.scope_tab:
            self.scope_tab.stop_capture()

        if self.scope_tab:
            try:
                self.scope_tab.set_event_markers(self._events)
            except Exception:
                pass

    def _tick(self):
        """Called on each timer tick: sends the next step (or, on the final tick, the base value)."""
        ptype, pcode, label = self.param.currentData()

        if self.step_index >= self.steps.value():
            # Extra tick after the last step (only scheduled when "Return to base" is on):
            # the last step has now been held for a full period, so write the base back.
            if self.chk_restore.isChecked():
                self._send_value(ptype, pcode, label, self.base.value(), base=True)
            self.stop(restore=False)
            return

        value = self.low.value() if (self.step_index % 2 == 0) else self.high.value()
        if not self._send_value(ptype, pcode, label, value):
            self.stop()  # declined / failed: abort (restores the base if a step was already sent)
            return

        self.step_index += 1
        if self.step_index >= self.steps.value() and not self.chk_restore.isChecked():
            self.stop()
