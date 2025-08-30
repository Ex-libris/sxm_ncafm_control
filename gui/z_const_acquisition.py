import datetime
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde, driver):
        super().__init__()
        self.dde = dde  # For DDE read/write
        self.driver = driver  # For IOCTL read
        self.live_mode = True
        self.last_z = 0.0
        self.base_z = 0.0
        self.z_history = []
        self.timestamps = []
        self.window_seconds = 10
        self.feedback_enabled = True

        layout = QtWidgets.QVBoxLayout(self)

        # ---- Top Controls ----
        ctrl = QtWidgets.QHBoxLayout()

        # Feedback toggle
        self.btn_toggle = QtWidgets.QPushButton("Disable Feedback")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self.toggle_feedback)
        ctrl.addWidget(self.btn_toggle)

        # Z position spinbox
        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setDecimals(6)
        self.z_spin.setSingleStep(0.001)
        self.z_spin.setRange(-1000.0, 1000.0)
        self.z_spin.setSuffix(" nm")
        self.z_spin.setEnabled(False)
        self.z_spin.valueChanged.connect(self.manual_update)
        ctrl.addWidget(QtWidgets.QLabel("Z Position:"))
        ctrl.addWidget(self.z_spin)

        # Time window selection
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(["2", "5", "10", "30", "60", "120"])
        self.combo_window.setCurrentText("10")
        self.combo_window.currentTextChanged.connect(
            lambda val: setattr(self, "window_seconds", int(val))
        )
        ctrl.addWidget(QtWidgets.QLabel("Window (s):"))
        ctrl.addWidget(self.combo_window)

        # Clear trace button
        self.btn_clear = QtWidgets.QPushButton("Clear Trace")
        self.btn_clear.clicked.connect(self.clear_trace)
        ctrl.addWidget(self.btn_clear)

        layout.addLayout(ctrl)

        # ---- Status Display ----
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Live mode, Feedback ON")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # ---- Plot ----
        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.curve = self.plot.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Z Position", units="nm")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        # ---- Timer for polling ----
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

        self.initialize_z_position()

    def initialize_z_position(self):
        try:
            if self.driver:
                current_z = self.driver.read_scaled("Topo")
                self.last_z = current_z
                self.z_spin.setValue(current_z)
                print(f"Initialized Z position: {current_z:.6f} nm")
            else:
                print("No driver available - using mock initialization")
                self.last_z = 0.0
        except Exception as e:
            print(f"Error initializing Z position: {e}")
            self.last_z = 0.0

    def toggle_feedback(self, checked: bool):
        if checked:
            # Disabling feedback
            self.btn_toggle.setText("Enable Feedback")
            try:
                if self.driver:
                    current_z = self.driver.read_scaled("Topo")
                    self.base_z = current_z
                    self.last_z = current_z
                    self.z_spin.setValue(current_z)
                    print(f"Feedback disabled. Base Z: {current_z:.6f} nm")

                # Disable feedback via DDE
                self.dde.feed_para("enable", 1)
                self.feedback_enabled = False

            except Exception as e:
                print(f"Error reading Z before disabling feedback: {e}")
                self.z_spin.setValue(self.last_z)

            self.live_mode = False
            self.z_spin.setEnabled(True)
            self.status_label.setText("Status: Manual mode, Feedback OFF")
            self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

        else:
            # Enabling feedback
            self.btn_toggle.setText("Disable Feedback")
            self.dde.feed_para("enable", 0)
            self.feedback_enabled = True
            self.live_mode = True
            self.z_spin.setEnabled(False)
            self.status_label.setText("Status: Live mode, Feedback ON")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            print("Feedback enabled - returning to live mode")

    def manual_update(self, value: float):
        if self.live_mode:
            return
        try:
            delta = self.base_z -value 
            self.dde.set_channel(0, delta)
            self.last_z = value
            print(f"Manual Z update: target {value:.6f} nm (Δ {delta:+.6f} nm from base)")
        except Exception as e:
            print(f"Manual Z write error: {e}")
            self.z_spin.setValue(self.last_z)

    def poll(self):
        now = datetime.datetime.now()
        elapsed = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        try:
            if self.live_mode and self.driver:
                z = self.driver.read_scaled("Topo")
                self.z_spin.setValue(z)
                self.last_z = z
            else:
                z = self.last_z
        except Exception as e:
            print(f"Polling error: {e}")
            z = self.last_z if hasattr(self, 'last_z') else 0.0

        self.timestamps.append(elapsed)
        self.z_history.append(z)

        while (self.timestamps and 
               len(self.timestamps) > 1 and 
               self.timestamps[-1] - self.timestamps[0] > self.window_seconds):
            self.timestamps.pop(0)
            self.z_history.pop(0)

        if len(self.timestamps) > 0:
            self.curve.setData(self.timestamps, self.z_history)
            x_min = max(0, elapsed - self.window_seconds)
            x_max = elapsed
            self.plot.setXRange(x_min, x_max, padding=0.02)
            if len(self.z_history) > 0:
                y_min = min(self.z_history)
                y_max = max(self.z_history)
                y_range = y_max - y_min
                if y_range > 0:
                    padding = y_range * 0.1
                    self.plot.setYRange(y_min - padding, y_max + padding)

    def clear_trace(self):
        self.z_history.clear()
        self.timestamps.clear()
        self.curve.setData([], [])
        print("Trace cleared")

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
        event.accept()
