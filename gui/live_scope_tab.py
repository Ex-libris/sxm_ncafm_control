# live_scope_tab.py
"""
Live Scope tab: a continuously-updating rolling oscilloscope view.

This is deliberately a separate tab from ScopeTab, not a mode within it.
ScopeTab's job is discrete "trigger and capture" snapshots (and stays that
way - Step Test overlays its event markers onto exactly those captures).
This tab instead mirrors the native SXM software's live oscilloscope: once
started it never stops acquiring, and the plot always shows a rolling
window over the last few seconds or minutes.

How the data is kept
--------------------
Measured on real hardware, the IOCTL read path sustains roughly
150,000-175,000 combined samples/s (see measure_driver_throughput.py), far too
many points to store or draw for a long window. So the acquisition thread
does not keep raw samples: it reduces the stream, in ~1 ms time slices, to one
record per slice - (timestamp, min, max) for each channel - in a fixed-size
ring buffer. Consequences:

* Memory is constant (MAX_WINDOW_S of history, ~24 MB) whatever the window
  is, so windows of up to 10 minutes are possible.
* The plot draws min/max envelopes ("peak detect"), so a narrow spike is
  never lost to decimation, at any window length.
* Every record carries its own timestamp, so the time axis is exact instead
  of derived from an average sample rate, and stalls show up as gaps.
* The window is only a *view* on the ring, so it can be changed while
  running without restarting the acquisition.
"""

import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from sxm_ncafm_control.device_driver import CHANNELS


class LiveCaptureThread(QtCore.QThread):
    """
    Reads two channels back-to-back forever and stores one (t, min, max)
    record per channel per CHUNK_S of wall time in a ring buffer.

    Thread-safety note: the GUI thread reads the rec_* arrays and rec_count
    directly, without a lock. The writer fills a slot completely before
    incrementing rec_count, so every record below rec_count is complete; a
    redraw can at worst see a slot being overwritten at the very oldest edge
    of a completely full ring, which self-heals on the next redraw. This is
    a display-only monitor, not a data-acquisition path.
    """

    error = QtCore.pyqtSignal(str)

    CHUNK_S = 0.001            # time covered by one record
    MAX_WINDOW_S = 600.0       # history kept (ring capacity)
    CAPACITY = int(MAX_WINDOW_S / CHUNK_S)
    MAX_CHUNK_SAMPLES = 8192   # safety cap on samples per record

    def __init__(self, driver, chan_idx1, chan_idx2, scale1, scale2):
        super().__init__()
        self.driver = driver
        self.chan_idx1 = chan_idx1
        self.chan_idx2 = chan_idx2
        self.scale1 = scale1
        self.scale2 = scale2

        n = self.CAPACITY
        self.rec_t = np.zeros(n, dtype=np.float64)    # seconds since t0 (end of the slice)
        self.rec_lo1 = np.zeros(n, dtype=np.float64)
        self.rec_hi1 = np.zeros(n, dtype=np.float64)
        self.rec_lo2 = np.zeros(n, dtype=np.float64)
        self.rec_hi2 = np.zeros(n, dtype=np.float64)
        self.rec_count = 0        # total records ever written (ring index = count % CAPACITY)
        self.n_samples = 0        # total sample pairs read
        self.latest1 = float("nan")
        self.latest2 = float("nan")
        self.t0 = 0.0
        self._stop = False

    def stop(self):
        self._stop = True

    def _commit(self, a, b, n, t):
        """Reduce the n samples in a[:n], b[:n] to one record ending at time t."""
        s1, s2 = self.scale1, self.scale2
        lo1, hi1 = a[:n].min() * s1, a[:n].max() * s1
        lo2, hi2 = b[:n].min() * s2, b[:n].max() * s2
        if lo1 > hi1:            # negative scale factors flip min/max
            lo1, hi1 = hi1, lo1
        if lo2 > hi2:
            lo2, hi2 = hi2, lo2

        i = self.rec_count % self.CAPACITY
        self.rec_t[i] = t
        self.rec_lo1[i], self.rec_hi1[i] = lo1, hi1
        self.rec_lo2[i], self.rec_hi2[i] = lo2, hi2
        self.latest1 = float(a[n - 1]) * s1
        self.latest2 = float(b[n - 1]) * s2
        self.n_samples += n
        self.rec_count += 1      # publish last: the slot above is complete before this

    def run(self):
        clock = time.perf_counter
        read_raw = self.driver.read_raw
        chan1, chan2 = self.chan_idx1, self.chan_idx2
        cap = self.MAX_CHUNK_SAMPLES
        a = np.empty(cap, dtype=np.int64)
        b = np.empty(cap, dtype=np.int64)

        n = 0
        self.t0 = t0 = clock()
        chunk_end = t0 + self.CHUNK_S
        while not self._stop:
            try:
                a[n] = read_raw(chan1)
                b[n] = read_raw(chan2)
            except Exception as e:
                self.error.emit(f"Driver read error: {e}")
                return
            n += 1

            now = clock()
            if now >= chunk_end or n == cap:
                self._commit(a, b, n, now - t0)
                n = 0
                chunk_end = now + self.CHUNK_S   # never try to "catch up" after a stall


class LiveScopeTab(QtWidgets.QWidget):
    """Continuously rolling dual-channel oscilloscope view (peak-detect display)."""

    WINDOWS_S = ["1", "2", "5", "10", "30", "60", "120", "300", "600"]
    DEFAULT_WINDOW_S = "5"
    N_COLUMNS = 1500           # envelope columns drawn per channel
    FAST_REDRAW_MS = 40        # ~25 fps
    SLOW_REDRAW_MS = 250       # long windows move lots of data per frame
    SLOW_ABOVE_S = 30

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
        self._units = ("", "")

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
        self.window_combo.addItems(self.WINDOWS_S)
        self.window_combo.setCurrentText(self.DEFAULT_WINDOW_S)
        self.window_combo.setToolTip(
            "How much history the plots show. This is only a view: it can be changed "
            "while running.\nUp to 10 minutes are kept in memory."
        )
        hbox.addWidget(self.window_combo)

        self.start_btn = QtWidgets.QPushButton("Start Live")
        self.start_btn.clicked.connect(self.start_live)
        hbox.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_live)
        hbox.addWidget(self.stop_btn)

        self.pause_chk = QtWidgets.QCheckBox("Pause display")
        self.pause_chk.setToolTip(
            "Freeze the plots so you can zoom and inspect them. Acquisition keeps running "
            "in the background."
        )
        hbox.addWidget(self.pause_chk)

        vbox.addLayout(hbox)

        self.status_label = QtWidgets.QLabel("Not running.")
        self.status_label.setStyleSheet("QLabel { color: #555; }")
        vbox.addWidget(self.status_label)

        self.plot_widget = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.plot_widget)

        self.plot1 = self.plot_widget.addPlot(row=0, col=0)
        self.plot1.setLabel("left", "Channel 1")
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        self.curve1 = self.plot1.plot([], [], pen=pg.mkPen(color=(50, 100, 200), width=1))

        self.plot2 = self.plot_widget.addPlot(row=1, col=0)
        self.plot2.setLabel("bottom", "Time (s, 0 = now)")
        self.plot2.setLabel("left", "Channel 2")
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.curve2 = self.plot2.plot([], [], pen=pg.mkPen(color=(200, 50, 50), width=1))
        self.plot2.setXLink(self.plot1)

        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.timeout.connect(self._redraw)

        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        self.pause_chk.toggled.connect(self._on_pause_toggled)
        self.chan1_combo.currentIndexChanged.connect(self._on_channels_changed)
        self.chan2_combo.currentIndexChanged.connect(self._on_channels_changed)
        self._apply_window()

    # ------------------------------------------------------------------ control
    def _window_s(self) -> float:
        return float(self.window_combo.currentText())

    def _apply_window(self):
        """Fix the visible time range to the chosen window and pick a redraw rate."""
        w = self._window_s()
        self.plot1.setXRange(-w, 0, padding=0)
        self.redraw_timer.setInterval(
            self.SLOW_REDRAW_MS if w > self.SLOW_ABOVE_S else self.FAST_REDRAW_MS
        )

    def _on_window_changed(self, _text):
        self._apply_window()
        if self.capture_thread is not None:
            self._redraw(force=True)

    def _on_pause_toggled(self, paused):
        if not paused and self.capture_thread is not None:
            self._redraw(force=True)

    def _on_channels_changed(self, _index):
        """Channels are fixed per acquisition thread: restart it (history is cleared)."""
        if self.capture_thread is not None:
            self.stop_live()
            self.start_live()

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
        self._units = (unit1, unit2)

        self.plot1.setLabel("left", f"{chan1_name} ({unit1})")
        self.plot2.setLabel("left", f"{chan2_name} ({unit2})")
        self.curve1.setData([], [])
        self.curve2.setData([], [])
        self._apply_window()

        self.capture_thread = LiveCaptureThread(self.driver, idx1, idx2, scale1, scale2)
        self.capture_thread.error.connect(self._on_error)
        self.capture_thread.start()

        self.redraw_timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Starting...")

    def stop_live(self):
        self.redraw_timer.stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.wait(1000)
            self.capture_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped.")

    def _on_error(self, message):
        self.stop_live()
        QtWidgets.QMessageBox.warning(self, "Live capture error", message)

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def _window_records(thread, window_s):
        """
        Copy the records inside the last window_s seconds out of the ring.

        Returns (t, lo1, hi1, lo2, hi2) in chronological order, or None if
        nothing has been recorded yet.
        """
        count = thread.rec_count
        if count == 0:
            return None
        cap = thread.CAPACITY
        # Records are never shorter than CHUNK_S, so this many always covers the window.
        m = min(count, cap, int(window_s / thread.CHUNK_S) + 2)
        idx = np.arange(count - m, count) % cap
        t = thread.rec_t[idx]
        first = int(np.argmax(t >= t[-1] - window_s))   # t ascends; last element always qualifies
        idx = idx[first:]
        return (t[first:], thread.rec_lo1[idx], thread.rec_hi1[idx],
                thread.rec_lo2[idx], thread.rec_hi2[idx])

    @classmethod
    def _envelope(cls, t, lo, hi, columns=None):
        """
        Reduce records to at most `columns` min/max columns, then interleave
        (min, max) so a plain line plot draws the envelope. Keeps narrow spikes.
        """
        columns = columns or cls.N_COLUMNS
        m = len(t)
        if m > columns:
            edges = (np.arange(columns + 1) * m) // columns   # strictly increasing since m > columns
            starts = edges[:-1]
            lo = np.minimum.reduceat(lo, starts)
            hi = np.maximum.reduceat(hi, starts)
            t = t[edges[1:] - 1]
        x = np.repeat(t, 2)
        y = np.empty(2 * len(t), dtype=np.float64)
        y[0::2] = lo
        y[1::2] = hi
        return x, y

    def _redraw(self, force=False):
        thread = self.capture_thread
        if thread is None or (self.pause_chk.isChecked() and not force):
            return

        window_s = self._window_s()
        rec = self._window_records(thread, window_s)
        if rec is None:
            return
        t, lo1, hi1, lo2, hi2 = rec
        t_rel = t - t[-1]           # 0 = newest record

        x, y = self._envelope(t_rel, lo1, hi1)
        self.curve1.setData(x, y)
        x, y = self._envelope(t_rel, lo2, hi2)
        self.curve2.setData(x, y)

        elapsed = time.perf_counter() - thread.t0
        rate = thread.n_samples / elapsed if elapsed > 0 else 0.0
        covered = t[-1] - t[0]
        state = "" if covered >= 0.98 * window_s else f" - filling, {covered:.1f} s of {window_s:g} s"
        u1, u2 = self._units
        self.status_label.setText(
            f"Live - {rate:,.0f} samples/s per channel | now: {thread.latest1:.6g} {u1}, "
            f"{thread.latest2:.6g} {u2}{state}"
        )

    def closeEvent(self, event):
        self.stop_live()
        event.accept()
