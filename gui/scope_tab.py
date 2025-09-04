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
    - Sweep mode: repeat step test for a series of base values
"""

import datetime
import numpy as np
from typing import List, Tuple
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from ..common import PARAMS_BASE, confirm_high_voltage


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

        self.low = QtWidgets.QDoubleSpinBox()
        self.low.setDecimals(6); self.low.setRange(-1e12, 1e12); self.low.setValue(10.0)
        self.high = QtWidgets.QDoubleSpinBox()
        self.high.setDecimals(6); self.high.setRange(-1e12, 1e12); self.high.setValue(101.0)
        self.period = QtWidgets.QDoubleSpinBox()
        self.period.setDecimals(3); self.period.setRange(0, 3600); self.period.setValue(1.0)
        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(1, 1_000_000); self.steps.setValue(20)

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

        self.tabs_widget = None
        self.scope_tab_index = None

        # --- Sweep mode ---
        sweep_group = QtWidgets.QGroupBox("Sweep mode (optional)")
        sweep_layout = QtWidgets.QFormLayout(sweep_group)

        self.chk_sweep = QtWidgets.QCheckBox("Enable sweep mode")
        sweep_layout.addRow(self.chk_sweep)

        self.sweep_start = QtWidgets.QDoubleSpinBox()
        self.sweep_start.setRange(-1e6, 1e6)
        self.sweep_start.setDecimals(3)
        sweep_layout.addRow("Start value", self.sweep_start)

        self.sweep_end = QtWidgets.QDoubleSpinBox()
        self.sweep_end.setRange(-1e6, 1e6)
        self.sweep_end.setDecimals(3)
        sweep_layout.addRow("End value", self.sweep_end)

        self.sweep_step = QtWidgets.QDoubleSpinBox()
        self.sweep_step.setRange(1e-6, 1e6)
        self.sweep_step.setDecimals(3)
        self.sweep_step.setValue(1.0)
        sweep_layout.addRow("Increment", self.sweep_step)

        self.sweep_wait = QtWidgets.QDoubleSpinBox()
        self.sweep_wait.setRange(0.0, 1e4)
        self.sweep_wait.setDecimals(1)
        self.sweep_wait.setValue(1.0)
        sweep_layout.addRow("Wait between values (s)", self.sweep_wait)

        v.addWidget(sweep_group)

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

        # sweep state
        self.sweep_values = []
        self.current_sweep_index = 0
        self.sweep_wait_time = 0.0
        self.current_base_value = None

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

    def preview(self):
        """Generates a step waveform preview in the plot area."""
        low, high, T, n = self.low.value(), self.high.value(), self.period.value(), self.steps.value()
        x = [0.0]; y = []
        for i in range(n):
            x.append((i + 1) * T)
            y.append(low if (i % 2 == 0) else high)
        self.plot.clear()
        self.plot.plot(x, y, stepMode=True, pen=pg.mkPen("b", width=2))
        self.plot.setXRange(0, n * T, padding=0.02)
        ymin, ymax = sorted([low, high])
        m = 0.05 * max(1.0, abs(ymax - ymin))
        self.plot.setYRange(ymin - m, ymax + m, padding=0.02)

    def start(self):
        """Begins step test (single or sweep)."""
        if self.chk_sweep.isChecked():
            # build sweep list
            start = self.sweep_start.value()
            end   = self.sweep_end.value()
            step  = self.sweep_step.value()

            if step == 0:
                QtWidgets.QMessageBox.warning(self, "Sweep error", "Increment cannot be zero")
                return

            if start <= end:
                self.sweep_values = list(np.arange(start, end + 0.5*step, step))
            else:
                self.sweep_values = list(np.arange(start, end - 0.5*step, -abs(step)))

            self.sweep_wait_time = self.sweep_wait.value()
            self.current_sweep_index = 0
            self.run_next_sweep()
        else:
            # normal single step test
            self._run_single_test(base_value=None)

    def run_next_sweep(self):
        """Run the next sweep value if any remain."""
        if self.current_sweep_index >= len(self.sweep_values):
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        base_val = self.sweep_values[self.current_sweep_index]
        self._run_single_test(base_value=base_val)

    def _run_single_test(self, base_value=None):
        """Actually runs one step test, optionally at a sweep base value."""
        self.preview()
        self.step_index = 0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._events = []

        # build metadata for scope
        if self.scope_tab:
            ptype, pcode, label = self.param.currentData()
            meta = {
                "param_label": label,
                "ptype": ptype,
                "pcode": pcode,
                "low": self.low.value(),
                "high": self.high.value(),
                "period_s": self.period.value(),
                "steps": self.steps.value(),
            }
            if base_value is not None:
                meta["sweep_base"] = base_value

            if hasattr(self.scope_tab, "set_test_metadata"):
                try:
                    self.scope_tab.set_test_metadata(meta)
                except Exception:
                    pass

        if self.chk_trigger_scope.isChecked() and self.scope_tab:
            self.scope_tab.start_capture()
        if self.tabs_widget and self.scope_tab_index is not None:
            self.tabs_widget.setCurrentIndex(self.scope_tab_index)

        self._timer.start(int(self.period.value() * 1000))
        self.current_base_value = base_value

    def stop(self):
        """Stops the step test and optionally stops the scope."""
        if self._timer.isActive():
            self._timer.stop()
            self.log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Test stopped.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if self.chk_trigger_scope.isChecked() and self.chk_stop_scope.isChecked() and self.scope_tab:
            self.scope_tab.stop_capture()

        if self.scope_tab:
            try:
                self.scope_tab.set_event_markers(self._events)
            except Exception:
                pass

        # if sweep mode, queue next sweep run
        if self.chk_sweep.isChecked() and hasattr(self, "sweep_values"):
            self.current_sweep_index += 1
            if self.current_sweep_index < len(self.sweep_values):
                QtCore.QTimer.singleShot(
                    int(self.sweep_wait_time * 1000), self.run_next_sweep
                )

    def _tick(self):
        """Called on each timer tick: sends value and logs event."""
        ptype, pcode, label = self.param.currentData()
        value = self.low.value() if (self.step_index % 2 == 0) else self.high.value()

        is_voltage_like = (ptype == "EDIT" and str(pcode).lower() == "edit23") or (
            ptype == "DNC" and int(pcode) == 4
        )
        if is_voltage_like and not confirm_high_voltage(self, label, value):
            self.stop()
            return

        try:
            if ptype == "EDIT":
                self.dde.send_scanpara(str(pcode), value)
            else:
                self.dde.send_dncpara(int(pcode), value)
        except Exception as e:
            self.log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SEND ERROR: {e}")
            self.stop()
            return

        ts_dt = QtCore.QDateTime.currentDateTime()
        self._events.append((ts_dt, f"{label}={value:g}"))

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
        self.log.append(f"[{ts}] Set {label} ({code_text}) to {value}")

        self.step_index += 1
        if self.step_index >= self.steps.value():
            self.stop()
