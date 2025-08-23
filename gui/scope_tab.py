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
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "Sample index")
        self.plot.setLabel("left", "Value")
        vbox.addWidget(self.plot)

    # ------------------------------------------------------------------
    def start_capture(self):
        chan_name = self.chan_combo.currentText()
        idx, _, unit, scale = CHANNELS[chan_name]
        npts = self.npoints_spin.value()
        self.plot.clear()

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
        self.plot.plot(arr, pen="y")
        self.last_data = arr
        self.last_rate = rate
        self.export_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

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
