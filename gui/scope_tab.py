import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from sxm_ncafm_control.device_driver import SXMIOCTL, CHANNELS


class CaptureThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(np.ndarray, np.ndarray, float)  # data1, data2, rate Hz

    def __init__(self, driver, chan_idx1, chan_idx2, npoints=50000):
        super().__init__()
        self.driver = driver
        self.chan_idx1 = chan_idx1
        self.chan_idx2 = chan_idx2
        self.npoints = npoints
        self._stop = False

    def run(self):
        vals1 = np.zeros(self.npoints, dtype=np.float64)
        vals2 = np.zeros(self.npoints, dtype=np.float64)
        t0 = QtCore.QTime.currentTime()
        
        # Get scaling factors for both channels
        _, _, _, scale1 = [c for c in CHANNELS.values() if c[0] == self.chan_idx1][0]
        _, _, _, scale2 = [c for c in CHANNELS.values() if c[0] == self.chan_idx2][0]
        
        for i in range(self.npoints):
            if self._stop:
                vals1 = vals1[:i]
                vals2 = vals2[:i]
                break
            try:
                raw1 = self.driver.read_raw(self.chan_idx1)
                raw2 = self.driver.read_raw(self.chan_idx2)
            except Exception:
                raw1 = np.random.randn() * 0.01  # offline fallback
                raw2 = np.random.randn() * 0.01 + 0.5  # different signal
                
            vals1[i] = raw1 * scale1
            vals2[i] = raw2 * scale2
            
        elapsed_ms = t0.msecsTo(QtCore.QTime.currentTime())
        rate = len(vals1) / max(elapsed_ms / 1000.0, 1e-9)
        self.finished.emit(vals1, vals2, rate)

    def stop(self):
        self._stop = True


class ScopeTab(QtWidgets.QWidget):
    """Dual-channel scope for SXM channels with shared time axis."""

    def __init__(self, driver=None):
        super().__init__()
        try:
            self.driver = driver or SXMIOCTL()
        except Exception as e:
            print("⚠ Could not open SXM driver, using mock:", e)
            self.driver = None

        self.capture_thread = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_rate = None
        self.last_chan1 = None
        self.last_chan2 = None
        
        # Time + markers state
        self.capture_start_dt = None
        self._event_markers = []
        self._marker_items1 = []  # markers for plot1
        self._marker_items2 = []  # markers for plot2
        
        # Reference to test tab for repeat functionality
        self.test_tab = None

        vbox = QtWidgets.QVBoxLayout(self)

        # Controls
        hbox = QtWidgets.QHBoxLayout()
        
        # Channel 1 selection
        hbox.addWidget(QtWidgets.QLabel("Channel 1:"))
        self.chan1_combo = QtWidgets.QComboBox()
        self.chan1_combo.addItems(list(CHANNELS.keys()))
        hbox.addWidget(self.chan1_combo)
        
        # Channel 2 selection
        hbox.addWidget(QtWidgets.QLabel("Channel 2:"))
        self.chan2_combo = QtWidgets.QComboBox()
        self.chan2_combo.addItems(list(CHANNELS.keys()))
        # Set different default for second channel if available
        if len(CHANNELS) > 1:
            self.chan2_combo.setCurrentIndex(1)
        hbox.addWidget(self.chan2_combo)

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

        self.repeat_test_btn = QtWidgets.QPushButton("Repeat Test")
        self.repeat_test_btn.setEnabled(False)
        self.repeat_test_btn.clicked.connect(self.repeat_test)
        hbox.addWidget(self.repeat_test_btn)

        vbox.addLayout(hbox)

        # Create dual plots with shared X-axis
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("white")  # Set background on the widget
        vbox.addWidget(self.plot_widget)
        
        # First plot
        self.plot1 = self.plot_widget.addPlot(row=0, col=0)
        self.plot1.setLabel("left", "Channel 1")
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        
        # Second plot (shares X-axis with first)
        self.plot2 = self.plot_widget.addPlot(row=1, col=0)
        self.plot2.setLabel("bottom", "Time (s)")
        self.plot2.setLabel("left", "Channel 2")
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        
        # Link X-axes so they zoom/pan together
        self.plot2.setXLink(self.plot1)

        # Style the grids
        for plot in [self.plot1, self.plot2]:
            grid_pen = pg.mkPen(color=(70, 70, 70), width=0.5)
            plot.getAxis('bottom').setPen(grid_pen)
            plot.getAxis('left').setPen(grid_pen)

    def start_capture(self):
        chan1_name = self.chan1_combo.currentText()
        chan2_name = self.chan2_combo.currentText()
        idx1, _, unit1, scale1 = CHANNELS[chan1_name]
        idx2, _, unit2, scale2 = CHANNELS[chan2_name]
        npts = self.npoints_spin.value()
        
        # Clear both plots
        self.plot1.clear()
        self.plot2.clear()
        
        # Update plot labels with units
        self.plot1.setLabel("left", f"{chan1_name} ({unit1})")
        self.plot2.setLabel("left", f"{chan2_name} ({unit2})")
        
        # Clear markers and set capture start time
        self._clear_markers()
        self.capture_start_dt = QtCore.QDateTime.currentDateTime()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

        self.last_data1 = None
        self.last_data2 = None
        self.last_rate = None
        self.last_chan1 = chan1_name
        self.last_chan2 = chan2_name

        if self.driver is not None:
            # Launch capture thread for both channels
            self.capture_thread = CaptureThread(self.driver, idx1, idx2, npoints=npts)
            self.capture_thread.finished.connect(self.show_data)
            self.capture_thread.start()
        else:
            # Offline fallback: generate mock signals
            t = np.linspace(0, 1, npts)
            arr1 = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(npts)
            arr2 = np.cos(2 * np.pi * 3 * t) * 2 + 0.2 * np.random.randn(npts)
            self.show_data(arr1, arr2, rate=npts)

    def stop_capture(self):
        if self.capture_thread is not None:
            self.capture_thread.stop()

    def show_data(self, arr1, arr2, rate):
        try:
            self.plot1.clear()
            self.plot2.clear()

            self.last_data1 = np.asarray(arr1)
            self.last_data2 = np.asarray(arr2)
            self.last_rate = float(rate) if rate else 0.0

            # Build time axis in seconds
            if self.last_rate > 0:
                t = np.arange(len(self.last_data1), dtype=float) / self.last_rate
            else:
                t = np.arange(len(self.last_data1), dtype=float)

            # Draw both traces
            self.plot1.plot(t, self.last_data1, pen=pg.mkPen('b', width=1))
            self.plot2.plot(t, self.last_data2, pen=pg.mkPen('r', width=1))

            # Update button states
            self.export_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # Add markers to both plots
            self._update_markers()

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Plot error", str(e))

    def export_data(self):
        if self.last_data1 is None or self.last_data2 is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", f"{self.last_chan1}_{self.last_chan2}_capture.csv", 
            "CSV Files (*.csv);;NumPy Files (*.npy)"
        )
        if not path:
            return
        try:
            if path.endswith(".npy"):
                # Save as structured array with both channels
                data = np.column_stack((self.last_data1, self.last_data2))
                np.save(path, data)
            else:
                t = np.arange(len(self.last_data1)) / max(self.last_rate, 1)
                header = f"chan1={self.last_chan1}, chan2={self.last_chan2}, rate={self.last_rate:.2f} Hz"
                np.savetxt(
                    path,
                    np.column_stack((t, self.last_data1, self.last_data2)),
                    delimiter=",",
                    header="time,channel1,channel2\n" + header,
                    comments="",
                )
            QtWidgets.QMessageBox.information(self, "Export", f"Data saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))

    def set_event_markers(self, events):
        """
        Accept a list of (QtCore.QDateTime, str_label) to overlay as vertical
        lines with small text on both plots. Units on X are seconds from capture start.
        """
        self._event_markers = list(events or [])
        self._update_markers()

    def _clear_markers(self):
        for it in self._marker_items1 + self._marker_items2:
            try:
                if it in self._marker_items1:
                    self.plot1.removeItem(it)
                else:
                    self.plot2.removeItem(it)
            except Exception:
                pass
        self._marker_items1 = []
        self._marker_items2 = []

    def _update_markers(self):
        # Need data, a valid rate and a start time
        if (
            self.last_data1 is None
            or self.last_data2 is None
            or self.capture_start_dt is None
        ):
            return

        # Compute plot X range in seconds
        if self.last_rate and self.last_rate > 0:
            tmax = (len(self.last_data1) - 1) / self.last_rate if len(self.last_data1) else 0.0
        else:
            tmax = float(len(self.last_data1) - 1) if len(self.last_data1) else 0.0

        # Y positions for labels on each plot
        try:
            ymax1 = float(np.nanmax(self.last_data1)) if len(self.last_data1) else 0.0
            ymax2 = float(np.nanmax(self.last_data2)) if len(self.last_data2) else 0.0
        except Exception:
            ymax1 = ymax2 = 0.0

        self._clear_markers()

        for dt, label in self._event_markers:
            try:
                # seconds since capture_start_dt
                secs = max(0.0, self.capture_start_dt.msecsTo(dt) / 1000.0)
            except Exception:
                secs = 0.0

            # Clamp into plotted window
            x = min(secs, tmax)

            # Add markers to both plots
            # Plot 1
            line1 = pg.InfiniteLine(pos=x, angle=90,
                                   pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot1.addItem(line1)
            txt1 = pg.TextItem(label, anchor=(0, 1), color='r')
            txt1.setPos(x, ymax1)
            self.plot1.addItem(txt1)
            self._marker_items1.extend([line1, txt1])

            # Plot 2
            line2 = pg.InfiniteLine(pos=x, angle=90,
                                   pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot2.addItem(line2)
            txt2 = pg.TextItem(label, anchor=(0, 1), color='r')
            txt2.setPos(x, ymax2)
            self.plot2.addItem(txt2)
            self._marker_items2.extend([line2, txt2])

    def set_test_tab_reference(self, test_tab):
        """Set reference to test tab for repeat functionality."""
        self.test_tab = test_tab
        self.repeat_test_btn.setEnabled(test_tab is not None)

    def repeat_test(self):
        """Trigger the test tab to repeat the last test configuration."""
        if self.test_tab is None:
            QtWidgets.QMessageBox.warning(self, "No Test Tab", 
                "No test tab reference available. Please run a test first.")
            return
            
        # Check if test is already running
        if hasattr(self.test_tab, '_timer') and self.test_tab._timer.isActive():
            QtWidgets.QMessageBox.information(self, "Test Running", 
                "A test is already in progress. Please wait for it to complete.")
            return
            
        try:
            # Enable scope triggering for this repeat
            if hasattr(self.test_tab, 'chk_trigger_scope'):
                self.test_tab.chk_trigger_scope.setChecked(True)
            if hasattr(self.test_tab, 'chk_stop_scope'):
                self.test_tab.chk_stop_scope.setChecked(True)
                
            # Start the test
            self.test_tab.start()
            
            # Show a brief message
            QtWidgets.QMessageBox.information(self, "Test Started", 
                "Repeating last test configuration. The scope will be triggered automatically.")
                
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Test Error", 
                f"Failed to start test: {str(e)}")