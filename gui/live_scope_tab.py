# live_scope_tab.py
"""
Live Scope tab: a continuously-updating rolling oscilloscope view.

This is deliberately a separate tab from ScopeTab, not a mode within it.
ScopeTab's job is discrete "trigger and capture" snapshots (and stays that
way - Step Test overlays its event markers onto exactly those captures).
This tab instead mirrors the native SXM software's live oscilloscope: once
started it never stops acquiring, and the plot always shows a rolling
window over the last `window` seconds rather than a fixed sample count.

Measured on real hardware, the IOCTL read path sustains roughly
150,000-175,000 combined samples/s with ~6 microsecond per-call latency
(see measure_driver_throughput.py) - fast enough that raw throughput is not
the bottleneck. What a one-shot "capture N points" tab cannot give you is a
continuously live, always-rolling trace, which is what this tab is for.
"""

import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from sxm_ncafm_control.device_driver import CHANNELS


class LiveCaptureThread(QtCore.QThread):
    """
    Continuously reads two channels into fixed-size ring buffers until
    stopped - unlike ScopeTab's CaptureThread, there is no fixed sample
    count; it runs indefinitely.

    Thread-safety note: the GUI thread reads write_index/buf1/buf2/t0
    directly, without a lock. Single-writer/single-reader access to plain
    numpy float64 element writes is safe enough for a live monitoring
    display (the GIL prevents a torn individual write), and a redraw that
    lands exactly on the wrap boundary can show at most one visually
    negligible inconsistent frame, which self-heals on the next redraw.
    This is not appropriate for anything safety-critical.
    """

    error = QtCore.pyqtSignal(str)

    def __init__(self, driver, chan_idx1, chan_idx2, scale1, scale2, capacity):
        super().__init__()
        self.driver = driver
        self.chan_idx1 = chan_idx1
        self.chan_idx2 = chan_idx2
        self.scale1 = scale1
        self.scale2 = scale2
        self.capacity = capacity
        self.buf1 = np.zeros(capacity, dtype=np.float64)
        self.buf2 = np.zeros(capacity, dtype=np.float64)
        self.write_index = 0
        self.t0 = 0.0
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        self.t0 = time.perf_counter()
        idx = 0
        cap = self.capacity
        buf1 = self.buf1
        buf2 = self.buf2
        driver = self.driver
        chan1 = self.chan_idx1
        chan2 = self.chan_idx2
        scale1 = self.scale1
        scale2 = self.scale2

        while not self._stop:
            try:
                raw1 = driver.read_raw(chan1)
                raw2 = driver.read_raw(chan2)
            except Exception as e:
                self.error.emit(f"Driver read error: {e}")
                return

            pos = idx % cap
            buf1[pos] = raw1 * scale1
            buf2[pos] = raw2 * scale2
            idx += 1
            self.write_index = idx


class LiveScopeTab(QtWidgets.QWidget):
    """Continuously rolling dual-channel oscilloscope view."""

    # Generous margin above the ~150-175 kHz combined rate measured on real
    # hardware - used only to size the ring buffer up front. If the real
    # rate differs, the buffer just ends up covering a different span than
    # the nominal window request; the status line reports the actual span.
    ASSUMED_MAX_RATE = 200_000  # samples/s per channel
    MAX_PLOT_POINTS = 5000      # live view - small and fast, not for export
    REDRAW_INTERVAL_MS = 40     # ~25 fps

    def __init__(self, driver=None):
        """
        Parameters
        ----------
        driver : SXMIOCTL or None
            The shared IOCTL driver handle, provided by SXMConnection.
            There is no offline/mock path for this tab - Start Live is
            disabled without a real driver.
        """
        super().__init__()
        self.driver = driver
        self.capture_thread = None

        vbox = QtWidgets.QVBoxLayout(self)

        hbox = QtWidgets.QHBoxLayout()

        hbox.addWidget(QtWidgets.QLabel("Channel 1:"))
        self.chan1_combo = QtWidgets.QComboBox()
        self.chan1_combo.addItems(list(CHANNELS.keys()))
        if "QPlusAmpl" in CHANNELS:
            self.chan1_combo.setCurrentIndex(list(CHANNELS.keys()).index("QPlusAmpl"))
        hbox.addWidget(self.chan1_combo)

        hbox.addWidget(QtWidgets.QLabel("Channel 2:"))
        self.chan2_combo = QtWidgets.QComboBox()
        self.chan2_combo.addItems(list(CHANNELS.keys()))
        if "Drive" in CHANNELS:
            self.chan2_combo.setCurrentIndex(list(CHANNELS.keys()).index("Drive"))
        elif len(CHANNELS) > 1:
            self.chan2_combo.setCurrentIndex(1)
        hbox.addWidget(self.chan2_combo)

        hbox.addWidget(QtWidgets.QLabel("Window (s):"))
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(["1", "2", "5", "10", "30"])
        self.window_combo.setCurrentText("5")
        hbox.addWidget(self.window_combo)

        self.start_btn = QtWidgets.QPushButton("Start Live")
        self.start_btn.clicked.connect(self.start_live)
        hbox.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_live)
        hbox.addWidget(self.stop_btn)

        vbox.addLayout(hbox)

        self.status_label = QtWidgets.QLabel("Not running.")
        self.status_label.setStyleSheet("QLabel { color: #555; }")
        vbox.addWidget(self.status_label)

        self.plot_widget = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.plot_widget)

        self.plot1 = self.plot_widget.addPlot(row=0, col=0)
        self.plot1.setLabel("left", "Channel 1")
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        self.curve1 = self.plot1.plot([], [], pen=pg.mkPen(color=(50, 100, 200), width=2))

        self.plot2 = self.plot_widget.addPlot(row=1, col=0)
        self.plot2.setLabel("bottom", "Time (s, 0 = now)")
        self.plot2.setLabel("left", "Channel 2")
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.curve2 = self.plot2.plot([], [], pen=pg.mkPen(color=(200, 50, 50), width=2))
        self.plot2.setXLink(self.plot1)

        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.timeout.connect(self._redraw)

    def start_live(self):
        if self.driver is None:
            QtWidgets.QMessageBox.warning(
                self, "No driver",
                "No IOCTL driver available (offline mode). Live Scope needs "
                "the real hardware driver - there is no mock data source for it."
            )
            return

        chan1_name = self.chan1_combo.currentText()
        chan2_name = self.chan2_combo.currentText()
        idx1, _, unit1, scale1 = CHANNELS[chan1_name]
        idx2, _, unit2, scale2 = CHANNELS[chan2_name]
        window_s = float(self.window_combo.currentText())
        capacity = max(1000, int(window_s * self.ASSUMED_MAX_RATE))

        self.plot1.setLabel("left", f"{chan1_name} ({unit1})")
        self.plot2.setLabel("left", f"{chan2_name} ({unit2})")

        self.capture_thread = LiveCaptureThread(self.driver, idx1, idx2, scale1, scale2, capacity)
        self.capture_thread.error.connect(self._on_error)
        self.capture_thread.start()

        self.redraw_timer.start(self.REDRAW_INTERVAL_MS)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.chan1_combo.setEnabled(False)
        self.chan2_combo.setEnabled(False)
        self.window_combo.setEnabled(False)
        self.status_label.setText("Starting...")

    def stop_live(self):
        self.redraw_timer.stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.wait(1000)
            self.capture_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.chan1_combo.setEnabled(True)
        self.chan2_combo.setEnabled(True)
        self.window_combo.setEnabled(True)
        self.status_label.setText("Stopped.")

    def _on_error(self, message):
        self.stop_live()
        QtWidgets.QMessageBox.warning(self, "Live capture error", message)

    def _redraw(self):
        thread = self.capture_thread
        if thread is None:
            return

        idx = thread.write_index
        cap = thread.capacity
        n_valid = min(idx, cap)
        if n_valid == 0:
            return

        elapsed = time.perf_counter() - thread.t0
        rate = idx / elapsed if elapsed > 0 else 0.0

        step = max(1, n_valid // self.MAX_PLOT_POINTS)
        if idx <= cap:
            data1 = thread.buf1[:n_valid:step]
            data2 = thread.buf2[:n_valid:step]
        else:
            # Buffer has wrapped: the position about to be overwritten next
            # holds the oldest still-valid sample. Chronological order is
            # [oldest-chunk][newest-chunk].
            start = idx % cap
            data1 = np.concatenate((thread.buf1[start::step], thread.buf1[:start:step]))
            data2 = np.concatenate((thread.buf2[start::step], thread.buf2[:start:step]))

        n_shown = len(data1)
        if rate > 0 and n_shown > 0:
            t = (np.arange(n_shown, dtype=np.float64) - (n_shown - 1)) * step / rate
        else:
            t = np.arange(n_shown, dtype=np.float64)

        self.curve1.setData(t, data1)
        self.curve2.setData(t, data2)

        window_covered = cap / rate if rate > 0 else 0.0
        state = "filling" if idx < cap else "full"
        self.status_label.setText(
            f"Live - measured rate: {rate:,.0f} Hz, buffer covers ~{window_covered:.1f} s ({state})"
        )

    def closeEvent(self, event):
        self.stop_live()
        event.accept()
