# z_const_acquisition.py
"""
Z-Constant acquisition tab with ring buffer for stable memory usage.
"""

import datetime
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from sxm_ncafm_control.device_driver import CHANNELS


class RingBuffer:
    """Fixed-size circular buffer for numeric data."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=float)
        self.index = 0
        self.full = False

    def append(self, value):
        self.data[self.index] = value
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def get_all(self):
        if not self.full:
            return self.data[:self.index]
        return np.concatenate((self.data[self.index:], self.data[:self.index]))


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde, driver, parent=None):
        super().__init__(parent)
        self.dde = dde
        self.driver = driver

        self.timer_interval_ms = 100      # poll interval
        self.window_seconds = 60          # visible window

        # Ring buffers sized for maximum samples in window
        max_samples = int(self.window_seconds * 1000 / self.timer_interval_ms)
        self.timestamps = RingBuffer(max_samples)
        self.z_history = RingBuffer(max_samples)

        self.last_z = 0.0

        # ---- UI ----
        vbox = QtWidgets.QVBoxLayout(self)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot = self.plot_widget.plot(pen=pg.mkPen("b", width=1))
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Topo (nm)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_widget)

        # Status
        self.status_label = QtWidgets.QLabel("Ready")
        vbox.addWidget(self.status_label)

        # Timer
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.poll)

    def start(self):
        self._timer.start(self.timer_interval_ms)
        self.status_label.setText("Running...")

    def stop(self):
        self._timer.stop()
        self.status_label.setText("Stopped")

    def poll(self):
        """Poll driver for Z value and update plot."""
        now = datetime.datetime.now()
        try:
            z = self.driver.read_scaled("Topo")
            self.last_z = z
        except Exception as e:
            print(f"Driver read error: {e}")
            z = self.last_z

        # Append to ring buffers
        self.timestamps.append(now.timestamp())
        self.z_history.append(z)

        # Retrieve all valid samples
        t_all = self.timestamps.get_all()
        z_vals = self.z_history.get_all()

        if len(t_all) == 0:
            return

        # Relative time axis
        t_rel = t_all - t_all[0]

        # Update plot
        self.plot.setData(t_rel, z_vals)

        self.status_label.setText(
            f"Latest z = {z:.3f} nm at {now.strftime('%H:%M:%S')}"
        )
