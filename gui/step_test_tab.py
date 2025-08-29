import datetime
from typing import List, Tuple
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from .common import PARAMS_BASE, confirm_high_voltage


class StepTestTab(QtWidgets.QWidget):
    """Square wave parameter stepping with optional scope trigger."""

    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self.step_index = 0
        self._customs: List[Tuple[str, object, str]] = []
        self.scope_tab = None  # linked from MainWindow

        v = QtWidgets.QVBoxLayout(self)

        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)

        # --- Controls ---
        self.param = QtWidgets.QComboBox()
        self._populate_params()

        self.low = QtWidgets.QDoubleSpinBox();  self.low.setDecimals(6)
        self.low.setMinimum(-1e12); self.low.setMaximum(1e12); self.low.setValue(10.0)
        self.high = QtWidgets.QDoubleSpinBox(); self.high.setDecimals(6)
        self.high.setMinimum(-1e12); self.high.setMaximum(1e12); self.high.setValue(101.0)
        self.period = QtWidgets.QDoubleSpinBox(); self.period.setDecimals(3)
        self.period.setMaximum(3600); self.period.setValue(1.000)
        self.steps = QtWidgets.QSpinBox(); self.steps.setMaximum(1_000_000); self.steps.setValue(20)

        c = 0
        grid.addWidget(self.param, 0, c, 1, 2); c += 2
        grid.addWidget(QtWidgets.QLabel("Low:"), 0, c); c += 1; grid.addWidget(self.low, 0, c); c += 1
        grid.addWidget(QtWidgets.QLabel("High:"),0, c); c += 1; grid.addWidget(self.high,0, c); c += 1
        grid.addWidget(QtWidgets.QLabel("Period (s):"),0, c); c += 1; grid.addWidget(self.period,0, c); c += 1
        grid.addWidget(QtWidgets.QLabel("Steps:"),0, c); c += 1; grid.addWidget(self.steps,0, c); c += 1

        self.btn_preview = QtWidgets.QPushButton("Preview")
        self.btn_start   = QtWidgets.QPushButton("Start")
        self.btn_stop    = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_preview.clicked.connect(self.preview)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        grid.addWidget(self.btn_preview, 0, c); c += 1
        grid.addWidget(self.btn_start,   0, c); c += 1
        grid.addWidget(self.btn_stop,    0, c)

        # --- New checkboxes ---
        self.chk_trigger_scope = QtWidgets.QCheckBox("Trigger scope capture at start")
        self.chk_stop_scope = QtWidgets.QCheckBox("Stop scope with Step Test")
        grid.addWidget(self.chk_trigger_scope, 1, 0, 1, 4)
        grid.addWidget(self.chk_stop_scope, 2, 0, 1, 4)
        self.tabs_widget = None
        self.scope_tab_index = None
        # --- Plot + log ---
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

        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True); v.addWidget(self.log)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        # --- NEW: (timestamp, label) for scope overlay ---
        self._events = []  # list of (QtCore.QDateTime, str)

    # ------------------------------------------------------------------
    def set_custom_params(self, customs: List[Tuple[str, object, str]]):
        self._customs = customs[:]
        self._populate_params()

    def _populate_params(self):
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

    # ------------------------------------------------------------------
    def preview(self):
        low, high, T, n = self.low.value(), self.high.value(), self.period.value(), self.steps.value()
        x = [0.0]; y = []
        for i in range(n):
            x.append((i + 1) * T)
            y.append(low if (i % 2 == 0) else high)
        self.plot.clear()
        self.plot.plot(x, y, stepMode=True, pen=pg.mkPen("b", width=2))
        self.plot.setXRange(0, n * T, padding=0.02)
        ymin, ymax = sorted([low, high]); m = 0.05 * max(1.0, abs(ymax - ymin))
        self.plot.setYRange(ymin - m, ymax + m, padding=0.02)

    def start(self):
        self.preview()
        self.step_index = 0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # --- NEW: fresh event list for this run ---
        self._events = []
        self._timer.start(int(self.period.value() * 1000))
        # If you already have these options, leave them as-is
        if getattr(self, "chk_trigger_scope", None) is not None and self.chk_trigger_scope.isChecked():
            if getattr(self, "scope_tab", None) is not None:
                self.scope_tab.start_capture()
        if getattr(self, "tabs_widget", None) is not None and getattr(self, "scope_tab_index", None) is not None:
            self.tabs_widget.setCurrentIndex(self.scope_tab_index)



    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
            self.log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Test stopped.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if self.chk_trigger_scope.isChecked() and self.chk_stop_scope.isChecked() and self.scope_tab is not None:
            self.scope_tab.stop_capture()
                # --- NEW: hand timings to the scope for overlay ---
        if getattr(self, "scope_tab", None) is not None:
            try:
                self.scope_tab.set_event_markers(self._events)
            except Exception:
                pass


    def _tick(self):
        ptype, pcode, label = self.param.currentData()
        value = self.low.value() if (self.step_index % 2 == 0) else self.high.value()

        is_voltage_like = (ptype == "EDIT" and str(pcode).lower() == "edit23") or (
            ptype == "DNC" and int(pcode) == 4
        )
        if is_voltage_like and not confirm_high_voltage(self, label, value):
            self.stop(); return

        try:
            if ptype == "EDIT":
                self.dde.send_scanpara(str(pcode), value)
            else:
                self.dde.send_dncpara(int(pcode), value)
        except Exception as e:
            self.log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SEND ERROR: {e}")
            self.stop(); return
                # --- NEW: log event for overlay (wall-clock) ---
        ts_dt = QtCore.QDateTime.currentDateTime()
        # Keep labels short; include the UI label and value
        self._events.append((ts_dt, f"{label}={value:g}"))

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
        self.log.append(f"[{ts}] Set {label} ({code_text}) to {value}")

        self.step_index += 1
        if self.step_index >= self.steps.value():
            self.stop()
