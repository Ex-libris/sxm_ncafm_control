# z_const_acquisition.py
# Full script with a second plot + IOCTL channel dropdown (uses device_driver.CHANNELS)

import sys
import datetime
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Uses your mapping. CHANNELS[name] -> (idx, short, unit, scale) or similar.
from ..device_driver import CHANNELS


class FlexibleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)  # Only emit valueChanged when editing is finished
        self.custom_step_enabled = True
        
    def stepBy(self, steps):
        if not self.custom_step_enabled:
            super().stepBy(steps)
            return
            
        # Get current cursor position in the line edit
        line_edit = self.lineEdit()
        cursor_pos = line_edit.cursorPosition()
        text = line_edit.text()
        
        # Remove suffix for position calculation
        suffix = self.suffix()
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
        
        # Find decimal point position
        decimal_pos = text.find('.')
        
        # Determine step size based on cursor position
        step_size = self.determine_step_size(cursor_pos, text, decimal_pos)
        
        # Apply the step with strict boundary checking
        current_value = self.value()
        new_value = current_value + (steps * step_size)
        
        # CRITICAL: Prevent wraparound by strictly enforcing boundaries
        if new_value > self.maximum():
            new_value = self.maximum()
            print(f"Warning: Clamped to maximum value {self.maximum()}")
        elif new_value < self.minimum():
            new_value = self.minimum()
            print(f"Warning: Clamped to minimum value {self.minimum()}")
        
        # Only set value if it actually changed to prevent unnecessary signals
        if abs(new_value - current_value) > 1e-10:  # Account for floating point precision
            self.setValue(new_value)
        
        # Restore cursor position (approximately)
        QtCore.QTimer.singleShot(0, lambda: line_edit.setCursorPosition(cursor_pos))
    
    def determine_step_size(self, cursor_pos, text, decimal_pos):
        """Determine step size based on cursor position"""
        if decimal_pos == -1:  # No decimal point
            # Count digits from right to cursor position
            digits_from_right = len(text) - cursor_pos
            if digits_from_right <= 0:
                return 1.0
            return 10 ** (digits_from_right - 1)
        else:
            if cursor_pos <= decimal_pos:
                # Before decimal point
                digits_from_right = decimal_pos - cursor_pos
                return 10 ** digits_from_right if digits_from_right > 0 else 1.0
            else:
                # After decimal point
                decimal_places = cursor_pos - decimal_pos - 1
                return 10 ** (-decimal_places - 1)
    
    def wheelEvent(self, event):
        """Enhanced wheel event with position-aware stepping"""
        if not self.custom_step_enabled:
            super().wheelEvent(event)
            return
            
        # Get the number of steps (usually +1 or -1)
        steps = event.angleDelta().y() // 120
        
        # Check for modifier keys for different behaviors
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        
        if modifiers == QtCore.Qt.ControlModifier:
            # Ctrl+wheel: use the default single step
            self.custom_step_enabled = False
            super().wheelEvent(event)
            self.custom_step_enabled = True
        elif modifiers == QtCore.Qt.ShiftModifier:
            # Shift+wheel: larger steps (10x current position step)
            self.stepBy(steps * 10)
        else:
            # Normal wheel: position-aware stepping
            self.stepBy(steps)
            
        event.accept()
    
    def keyPressEvent(self, event):
        """Enhanced key press event for better navigation"""
        if event.key() == QtCore.Qt.Key_Up:
            self.stepBy(1)
            event.accept()
        elif event.key() == QtCore.Qt.Key_Down:
            self.stepBy(-1)
            event.accept()
        else:
            super().keyPressEvent(event)


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde=None, driver=None):
        super().__init__()
        self.setWindowTitle("Z-Const Acquisition")
        self.resize(1000, 700)

        self.dde = dde
        self.driver = driver

        # state
        self.live_mode = True
        self.feedback_enabled = True
        self.last_z = 0.0
        self.window_seconds = 10

        # topography trace
        self.t_stamps = []
        self.z_hist = []

        # aux trace
        default_aux = "Frequency" if "Frequency" in CHANNELS else list(CHANNELS.keys())[0]
        self.aux_channel = default_aux
        self.aux_unit = CHANNELS[self.aux_channel][2]
        self.aux_hist = []

        # ui
        main = QtWidgets.QVBoxLayout(self)

        # --- top controls ---
        ctrl = QtWidgets.QHBoxLayout()

        self.btn_toggle = QtWidgets.QPushButton("Disable Feedback")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self.toggle_feedback)
        ctrl.addWidget(self.btn_toggle)

        ctrl.addWidget(QtWidgets.QLabel("Z Position:"))
        self.z_spin = FlexibleDoubleSpinBox()
        self.z_spin.setRange(-1000.0, 1000.0)
        self.z_spin.setSuffix(" nm")
        self.z_spin.setEnabled(False)  # enabled only if manual
        self.z_spin.valueChanged.connect(self.manual_update)
        ctrl.addWidget(self.z_spin)

        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Window (s):"))
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(["2", "5", "10", "30", "60", "120"])
        self.combo_window.setCurrentText("10")
        self.combo_window.currentTextChanged.connect(
            lambda v: setattr(self, "window_seconds", int(v))
        )
        ctrl.addWidget(self.combo_window)

        self.btn_clear = QtWidgets.QPushButton("Clear Trace")
        self.btn_clear.clicked.connect(self.clear_trace)
        ctrl.addWidget(self.btn_clear)

        ctrl.addStretch(1)
        main.addLayout(ctrl)

        # help row
        help_row = QtWidgets.QHBoxLayout()
        help_lbl = QtWidgets.QLabel(
            "Wheel = pos-aware step | Shift+wheel = 10× | Ctrl+wheel = fixed step | ↑↓ = pos-aware step"
        )
        help_lbl.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        help_lbl.setWordWrap(True)
        help_row.addWidget(help_lbl)
        main.addLayout(help_row)

        # status row
        status = QtWidgets.QHBoxLayout()
        self.status_lbl = QtWidgets.QLabel("Status: Live mode, Feedback ON")
        self.status_lbl.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status.addWidget(self.status_lbl)
        status.addStretch(1)
        main.addLayout(status)

        # --- topography plot ---
        self.plot_topo = pg.PlotWidget()
        self.plot_topo.setBackground("w")
        self.curve_topo = self.plot_topo.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot_topo.setLabel("bottom", "Time", units="s")
        self.plot_topo.setLabel("left", "Z Position", units="nm")
        self.plot_topo.showGrid(x=True, y=True, alpha=0.3)
        main.addWidget(self.plot_topo)

        # --- aux controls + plot ---
        aux_ctrl = QtWidgets.QHBoxLayout()
        aux_ctrl.addWidget(QtWidgets.QLabel("Aux channel:"))

        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.addItems(sorted(CHANNELS.keys()))
        i_def = self.combo_channel.findText(self.aux_channel)
        if i_def >= 0:
            self.combo_channel.setCurrentIndex(i_def)
        self.combo_channel.currentTextChanged.connect(self.on_channel_changed)
        aux_ctrl.addWidget(self.combo_channel)

        self.lbl_aux_unit = QtWidgets.QLabel(f"[{self.aux_unit}]")
        self.lbl_aux_unit.setStyleSheet("QLabel { color: #444; }")
        aux_ctrl.addWidget(self.lbl_aux_unit)
        aux_ctrl.addStretch(1)
        main.addLayout(aux_ctrl)

        self.plot_aux = pg.PlotWidget()
        self.plot_aux.setBackground("w")
        self.curve_aux = self.plot_aux.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot_aux.setLabel("bottom", "Time", units="s")
        self.plot_aux.setLabel("left", self.aux_channel, units=self.aux_unit)
        self.plot_aux.showGrid(x=True, y=True, alpha=0.3)
        main.addWidget(self.plot_aux)

        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

        # initial sync
        self.initialize_z_position()

    # ---------- actions ----------
    def initialize_z_position(self):
        try:
            if self.driver:
                z = self.driver.read_scaled("Topo")
                self.last_z = float(z)
                self.z_spin.setValue(self.last_z)
        except Exception as e:
            print(f"Init error: {e}")

    def toggle_feedback(self, checked: bool):
        # checked means button is pressed -> we *disable* feedback
        self.feedback_enabled = not checked
        if self.feedback_enabled:
            self.btn_toggle.setText("Disable Feedback")
            self.status_lbl.setText("Status: Live mode, Feedback ON")
            self.status_lbl.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.z_spin.setEnabled(False)
        else:
            self.btn_toggle.setText("Enable Feedback")
            self.status_lbl.setText("Status: Manual mode, Feedback OFF")
            self.status_lbl.setStyleSheet("QLabel { color: #c77; font-weight: bold; }")
            self.z_spin.setEnabled(True)

    def manual_update(self, val: float):
        if self.feedback_enabled:
            return
        # Here you would send setpoint to hardware if needed
        self.last_z = float(val)

    def clear_trace(self):
        self.t_stamps.clear()
        self.z_hist.clear()
        self.curve_topo.setData([], [])
        self.aux_hist.clear()
        self.curve_aux.setData([], [])
        print("Trace cleared")

    def on_channel_changed(self, name: str):
        try:
            unit = CHANNELS[name][2]
        except Exception:
            # keep previous if missing
            name = self.aux_channel
            unit = self.aux_unit
        self.aux_channel = name
        self.aux_unit = unit
        self.lbl_aux_unit.setText(f"[{unit}]")
        self.plot_aux.setLabel("left", name, units=unit)
        self.aux_hist.clear()
        self.curve_aux.setData([], [])

    # ---------- polling ----------
    def poll(self):
        now = datetime.datetime.now()
        # time in seconds since midnight; stable increasing
        t_now = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        # read topo
        z = self.last_z
        try:
            if self.live_mode and self.driver:
                z = float(self.driver.read_scaled("Topo"))
                self.last_z = z
                if self.feedback_enabled:
                    self.z_spin.blockSignals(True)
                    self.z_spin.setValue(z)
                    self.z_spin.blockSignals(False)
        except Exception as e:
            print(f"Polling Topo error: {e}")

        # read aux
        aux_val = None
        try:
            if self.driver:
                aux_val = float(self.driver.read_scaled(self.aux_channel))
        except Exception as e:
            print(f"Polling Aux '{self.aux_channel}' error: {e}")

        # push
        self.t_stamps.append(t_now)
        self.z_hist.append(z)
        if aux_val is not None:
            self.aux_hist.append(aux_val)

        # trim to window
        win = self.window_seconds
        while (
            self.t_stamps
            and len(self.t_stamps) > 1
            and self.t_stamps[-1] - self.t_stamps[0] > win
        ):
            self.t_stamps.pop(0)
            self.z_hist.pop(0)
            if self.aux_hist:
                self.aux_hist.pop(0)

        # redraw topo
        if self.t_stamps:
            self.curve_topo.setData(self.t_stamps, self.z_hist)
            x_min = max(0.0, t_now - win)
            x_max = t_now
            self.plot_topo.setXRange(x_min, x_max, padding=0.02)
            y_min = min(self.z_hist)
            y_max = max(self.z_hist)
            if y_max > y_min:
                pad = 0.1 * (y_max - y_min)
                self.plot_topo.setYRange(y_min - pad, y_max + pad)

        # redraw aux
        if self.t_stamps and self.aux_hist:
            # align lengths if aux started later
            n_aux = len(self.aux_hist)
            self.curve_aux.setData(self.t_stamps[-n_aux:], self.aux_hist)
            x_min = max(0.0, t_now - win)
            x_max = t_now
            self.plot_aux.setXRange(x_min, x_max, padding=0.02)
            y_min = min(self.aux_hist)
            y_max = max(self.aux_hist)
            if y_max > y_min:
                pad = 0.1 * (y_max - y_min)
                self.plot_aux.setYRange(y_min - pad, y_max + pad)

    # ---------- Qt ----------
    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        event.accept()


def run(dde=None, driver=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ZConstAcquisition(dde=dde, driver=driver)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    # Expect caller to pass a real driver with read_scaled(name).
    # If you want to test without hardware, define a stub here.
    class _StubDriver:
        def __init__(self):
            self._t0 = datetime.datetime.now()

        def read_scaled(self, name):
            dt = (datetime.datetime.now() - self._t0).total_seconds()
            if name == "Topo":
                return 100.0 + 2.0 * pg.np.sin(0.8 * dt)
            # generic aux
            return 1.0 * pg.np.sin(2.0 * dt) + 5.0

    # Comment out the stub when wiring to your stack.
    sys.exit(run(driver=_StubDriver()))
