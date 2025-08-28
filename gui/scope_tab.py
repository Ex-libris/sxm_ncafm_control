import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from sxm_ncafm_control.device_driver import SXMIOCTL, CHANNELS


class CaptureThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(np.ndarray, float)  # data, rate Hz

    def __init__(self, driver, chan_idx, npoints=50000):
        super().__init__()
        self.driver = driver
        self.chan_idx = chan_idx
        self.npoints = npoints
        self._stop = False

    def run(self):
        vals = np.zeros(self.npoints, dtype=np.float64)
        t0 = QtCore.QTime.currentTime()
        for i in range(self.npoints):
            if self._stop:
                vals = vals[:i]
                break
            try:
                raw = self.driver.read_raw(self.chan_idx)
            except Exception:
                raw = np.random.randn() * 0.01  # offline fallback
            # Lookup scale
            _, _, _, scale = [c for c in CHANNELS.values() if c[0] == self.chan_idx][0]
            vals[i] = raw * scale
        elapsed_ms = t0.msecsTo(QtCore.QTime.currentTime())
        rate = len(vals) / max(elapsed_ms / 1000.0, 1e-9)
        self.finished.emit(vals, rate)

    def stop(self):
        self._stop = True


class ScopeTab(QtWidgets.QWidget):
    """Single-shot scope for SXM channels."""

    def __init__(self, driver=None):
        super().__init__()
        try:
            self.driver = driver or SXMIOCTL()
        except Exception as e:
            print("⚠ Could not open SXM driver, using mock:", e)
            self.driver = None

        self.capture_thread = None
        self.last_data = None
        self.last_rate = None
        self.last_chan = None
        # --- NEW: time + markers state ---
        self.capture_start_dt = None          # QDateTime when capture starts
        self._event_markers = []              # list[(QtCore.QDateTime, str)]
        self._marker_items = []               # pg items drawn on the plot

        vbox = QtWidgets.QVBoxLayout(self)

        # Controls
        hbox = QtWidgets.QHBoxLayout()
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItems(list(CHANNELS.keys()))
        hbox.addWidget(self.chan_combo)

        self.npoints_spin = QtWidgets.QSpinBox()
        self.npoints_spin.setRange(1000, 2_000_000)
        self.npoints_spin.setValue(50000)
        hbox.addWidget(QtWidgets.QLabel("Samples:"))
        hbox.addWidget(self.npoints_spin)

        self.start_btn = QtWidgets.QPushButton("Start Capture")
        self.start_btn.clicked.connect(self.start_capture)
        hbox.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_capture)
        hbox.addWidget(self.stop_btn)

        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_data)
        hbox.addWidget(self.export_btn)

        vbox.addLayout(hbox)

        # Plot

        self.plot = pg.PlotWidget()
        self.plot.setBackground("white")
        self.plot.setLabel("bottom", "Sample index")
        self.plot.setLabel("left", "Value")

        # Professional oscilloscope-style grid
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Clean and simple - just enable the grid with custom styling
        plot_item = self.plot.getPlotItem()
        # Make x-axis show time
        self.plot.setLabel('bottom', 'Time (s)')

        # Optional: customize grid pen color
        grid_pen = pg.mkPen(color=(70, 70, 70), width=0.5)
        plot_item.getAxis('bottom').setPen(grid_pen)
        plot_item.getAxis('left').setPen(grid_pen)
        vbox.addWidget(self.plot)

    # ------------------------------------------------------------------
    def start_capture(self):
        chan_name = self.chan_combo.currentText()
        idx, _, unit, scale = CHANNELS[chan_name]
        npts = self.npoints_spin.value()
        self.plot.clear()
        # --- NEW: start-of-capture time + wipe old overlays ---
        self._clear_markers()
        self.capture_start_dt = QtCore.QDateTime.currentDateTime()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

        self.last_data = None
        self.last_rate = None
        self.last_chan = chan_name

        if self.driver is not None:
            # Launch capture thread
            self.capture_thread = CaptureThread(self.driver, idx, npoints=npts)
            self.capture_thread.finished.connect(self.show_data)
            self.capture_thread.start()
        else:
            # Offline fallback: generate mock signal
            t = np.linspace(0, 1, npts)
            arr = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(npts)
            self.show_data(arr, rate=npts)

    def stop_capture(self):
        if self.capture_thread is not None:
            self.capture_thread.stop()

    def show_data(self, arr, rate):
        try:
            self.plot.clear()

            self.last_data = np.asarray(arr)
            self.last_rate = float(rate) if rate else 0.0

            # Build time axis in seconds (0 ... (N-1)/rate). Guard rate=0.
            if self.last_rate > 0:
                t = np.arange(len(self.last_data), dtype=float) / self.last_rate
            else:
                # Fallback: index as seconds with 1 Hz if rate missing
                t = np.arange(len(self.last_data), dtype=float)

            # Draw the trace with time on X
            self.plot.plot(t, self.last_data, pen=pg.mkPen('b', width=1))

            # Buttons and state
            self.export_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # Add markers (if any)
            self._update_markers()

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Plot error", str(e))

    def export_data(self):
        if self.last_data is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", f"{self.last_chan}_capture.csv", "CSV Files (*.csv);;NumPy Files (*.npy)"
        )
        if not path:
            return
        try:
            if path.endswith(".npy"):
                np.save(path, self.last_data)
            else:
                t = np.arange(len(self.last_data)) / max(self.last_rate, 1)
                header = f"channel={self.last_chan}, rate={self.last_rate:.2f} Hz"
                np.savetxt(
                    path,
                    np.column_stack((t, self.last_data)),
                    delimiter=",",
                    header="time,value\n" + header,
                    comments="",
                )
            QtWidgets.QMessageBox.information(self, "Export", f"Data saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))

    def set_event_markers(self, events):
        """
        Accept a list of (QtCore.QDateTime, str_label) to overlay as vertical
        lines with small text. Units on X are seconds from capture start.
        """
        self._event_markers = list(events or [])
        # If data is already shown, render now; else show_data() will call this.
        self._update_markers()

    def _clear_markers(self):
        for it in self._marker_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._marker_items = []

    def _update_markers(self):
        # Need data, a valid rate (or at least X axis) and a start time
        if (
            self.last_data is None
            or self.capture_start_dt is None
        ):
            return

        # Compute plot X range in seconds
        if self.last_rate and self.last_rate > 0:
            tmax = (len(self.last_data) - 1) / self.last_rate if len(self.last_data) else 0.0
        else:
            tmax = float(len(self.last_data) - 1) if len(self.last_data) else 0.0

        # Y position for labels: top of current data
        try:
            ymax = float(np.nanmax(self.last_data)) if len(self.last_data) else 0.0
        except Exception:
            ymax = 0.0

        self._clear_markers()

        for dt, label in self._event_markers:
            try:
                # seconds since capture_start_dt
                secs = max(0.0, self.capture_start_dt.msecsTo(dt) / 1000.0)
            except Exception:
                secs = 0.0

            # Clamp into plotted window
            x = min(secs, tmax)

            # Vertical line
            line = pg.InfiniteLine(pos=x, angle=90,
                                   pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot.addItem(line)

            # Small text tag
            txt = pg.TextItem(label, anchor=(0, 1), color='r')
            txt.setPos(x, ymax)
            self.plot.addItem(txt)

            self._marker_items.extend([line, txt])
