# scope_tab.py
"""
Scope tab (dual-channel oscilloscope view).

Provides live capture of two SXM channels, plotting them against a shared
time axis. Supports export of data to CSV/NumPy, and overlay of event
markers from external test tabs.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
from sxm_ncafm_control.device_driver import CHANNELS


class CaptureThread(QtCore.QThread):
    """
    Background capture thread for two SXM channels.

    Reads raw values from the IOCTL driver as fast as the driver allows,
    until the requested number of points is acquired or the thread is
    stopped. There is no fixed sample rate: the effective rate is measured
    from how long the loop actually took, and reported after the fact.

    Emits
    -----
    finished : np.ndarray, np.ndarray, float
        Arrays of channel 1 and channel 2 values (scaled to physical units),
        and the effective sampling rate in Hz.
    error : str
        A read failure message. Emitted at most once, before `finished`,
        with whatever data was captured before the failure.
    """
    finished = QtCore.pyqtSignal(np.ndarray, np.ndarray, float)  # data1, data2, rate Hz
    error = QtCore.pyqtSignal(str)

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

        flush_interval = 10000  # yield to other threads every N samples

        n_captured = self.npoints
        for i in range(self.npoints):
            if self._stop:
                n_captured = i
                break

            try:
                raw1 = self.driver.read_raw(self.chan_idx1)
                raw2 = self.driver.read_raw(self.chan_idx2)
            except Exception as e:
                self.error.emit(f"Driver read error at sample {i}: {e}")
                n_captured = i
                break

            if abs(raw1) > 1e10 or abs(raw2) > 1e10:
                print(f"Warning: Extreme values detected at sample {i}: {raw1}, {raw2}")
                continue

            vals1[i] = raw1 * scale1
            vals2[i] = raw2 * scale2

            if i > 0 and i % flush_interval == 0:
                self.msleep(1)

        vals1 = vals1[:n_captured]
        vals2 = vals2[:n_captured]
        elapsed_ms = t0.msecsTo(QtCore.QTime.currentTime())
        rate = len(vals1) / max(elapsed_ms / 1000.0, 1e-9)
        self.finished.emit(vals1, vals2, rate)

    def stop(self):
        self._stop = True


class ScopeTab(QtWidgets.QWidget):
    """Dual-channel scope for SXM channels with shared time axis."""

    def __init__(self, driver=None):
        """
        Parameters
        ----------
        driver : SXMIOCTL or None
            The shared IOCTL driver handle, provided by SXMConnection.
            Pass None to run in offline mode (mock data).
        """
        super().__init__()
        self.driver = driver

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
        
        # Remember last export path
        self.last_export_path = None

        vbox = QtWidgets.QVBoxLayout(self)

        # Controls
        hbox = QtWidgets.QHBoxLayout()
        
        # Channel 1 selection - default to QPlusAmpl
        hbox.addWidget(QtWidgets.QLabel("Channel 1:"))
        self.chan1_combo = QtWidgets.QComboBox()
        self.chan1_combo.addItems(list(CHANNELS.keys()))
        # Set default to QPlusAmpl if available
        if "QPlusAmpl" in CHANNELS:
            idx = list(CHANNELS.keys()).index("QPlusAmpl")
            self.chan1_combo.setCurrentIndex(idx)
        hbox.addWidget(self.chan1_combo)
        
        # Channel 2 selection - default to Drive
        hbox.addWidget(QtWidgets.QLabel("Channel 2:"))
        self.chan2_combo = QtWidgets.QComboBox()
        self.chan2_combo.addItems(list(CHANNELS.keys()))
        # Set default to Drive if available, otherwise second channel
        if "Drive" in CHANNELS:
            idx = list(CHANNELS.keys()).index("Drive")
            self.chan2_combo.setCurrentIndex(idx)
        elif len(CHANNELS) > 1:
            self.chan2_combo.setCurrentIndex(1)
        hbox.addWidget(self.chan2_combo)

        self.npoints_spin = QtWidgets.QSpinBox()
        self.npoints_spin.setRange(1000, 2_000_000)
        self.npoints_spin.setValue(500_000)
        self.npoints_spin.setToolTip(
            "Number of raw reads to perform, back-to-back, as fast as the driver allows.\n"
            "This is NOT a sample rate or a duration - both are measured after capture\n"
            "and shown below the plots once the capture finishes."
        )
        samples_label = QtWidgets.QLabel("Samples to acquire:")
        samples_label.setToolTip(self.npoints_spin.toolTip())
        hbox.addWidget(samples_label)
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

        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_plots)
        hbox.addWidget(self.clear_btn)

        vbox.addLayout(hbox)

        self.status_label = QtWidgets.QLabel("No capture yet.")
        self.status_label.setStyleSheet("QLabel { color: #555; }")
        vbox.addWidget(self.status_label)

        # Create dual plots with shared X-axis
        self.plot_widget = pg.GraphicsLayoutWidget()
        # self.plot_widget.setBackground("white")  # Set background on the widget
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

    def __del__(self):
        """Ensure proper cleanup when widget is destroyed."""
        try:
            if hasattr(self, 'capture_thread') and self.capture_thread is not None:
                self.capture_thread.stop()
                self.capture_thread.wait(1000)
                if self.capture_thread.isRunning():
                    self.capture_thread.terminate()
                    self.capture_thread.wait(1000)
        except Exception as e:
            print(f"Warning during ScopeTab cleanup: {e}")

    def _cleanup_data(self):
        """Drop references to the last capture's data arrays."""
        self.last_data1 = None
        self.last_data2 = None
        self.last_rate = None
        self.last_chan1 = None
        self.last_chan2 = None

    def _force_clear_plots(self):
        """Clear all traces/markers from both plots.

        Only clear the plots themselves - do NOT touch plot.scene() or
        plot.getViewBox() here. Both plots share one GraphicsLayoutWidget
        scene, so calling scene().clear() on either plot also destroys the
        other plot's axes/viewbox, which is what previously required a
        plot-recreation workaround on every capture.
        """
        try:
            self.plot1.clear()
            self.plot2.clear()
        except Exception as e:
            print(f"Warning during plot clearing: {e}")

    def estimate_capture_npoints(self, duration_s):
        """
        Estimate how many samples (i.e. loop iterations - one read per
        channel each) are needed for a capture to last at least duration_s.

        Uses the most recently measured rate if one is available (adapts to
        whatever this specific PC/hardware actually achieves), otherwise
        falls back to a conservative default that's deliberately on the
        generous side - a rate assumed too LOW would under-size the capture
        and cause it to finish before duration_s elapses, reproducing the
        exact "capture ends before the test does" problem this exists to
        avoid. A 20% margin is added on top for run-to-run rate variance.
        """
        assumed_rate = self.last_rate if self.last_rate and self.last_rate > 0 else 100_000
        npoints = int(duration_s * assumed_rate * 1.2)
        lo, hi = self.npoints_spin.minimum(), self.npoints_spin.maximum()
        return max(lo, min(hi, npoints))

    def start_capture(self, npoints_override=None):
        # 1. PROPERLY CLEANUP PREVIOUS THREAD
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.wait(1000)  # Wait up to 1 second
            if self.capture_thread.isRunning():
                print("Warning: Force terminating capture thread")
                self.capture_thread.terminate()
                self.capture_thread.wait(1000)
            try:
                self.capture_thread.deleteLater()
            except Exception:
                pass
            self.capture_thread = None

        chan1_name = self.chan1_combo.currentText()
        chan2_name = self.chan2_combo.currentText()
        idx1, _, unit1, scale1 = CHANNELS[chan1_name]
        idx2, _, unit2, scale2 = CHANNELS[chan2_name]
        npts = npoints_override if npoints_override is not None else self.npoints_spin.value()
        self.last_chan1 = chan1_name
        self.last_chan2 = chan2_name

        self._force_clear_plots()

        # Update plot labels with units
        self.plot1.setLabel("left", f"{chan1_name} ({unit1})")
        self.plot2.setLabel("left", f"{chan2_name} ({unit2})")

        # Clear markers and set capture start time
        self._clear_markers()
        self.capture_start_dt = QtCore.QDateTime.currentDateTime()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.status_label.setText("Capturing...")

        self._cleanup_data()

        if self.driver is not None:
            # Launch capture thread for both channels
            self.capture_thread = CaptureThread(self.driver, idx1, idx2, npoints=npts)
            self.capture_thread.error.connect(self._on_capture_error)
            self.capture_thread.finished.connect(self.show_data)
            self.capture_thread.start()
        else:
            # Offline fallback: generate mock signals (no real hardware to poll)
            t = np.linspace(0, 1, npts)
            arr1 = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(npts)
            arr2 = np.cos(2 * np.pi * 3 * t) * 2 + 0.2 * np.random.randn(npts)
            self.show_data(arr1, arr2, rate=npts)

    def stop_capture(self):
        if self.capture_thread is not None:
            self.capture_thread.stop()

    def _on_capture_error(self, message):
        QtWidgets.QMessageBox.warning(self, "Capture error", message)

    def show_data(self, arr1, arr2, rate):
        try:
            # Force cleanup first
            self._force_clear_plots()

            # Convert to numpy arrays explicitly with proper dtype
            self.last_data1 = np.asarray(arr1, dtype=np.float64)
            self.last_data2 = np.asarray(arr2, dtype=np.float64)
            self.last_rate = float(rate) if rate else 0.0

            # Build time axis in seconds
            if self.last_rate > 0:
                t = np.arange(len(self.last_data1), dtype=np.float64) / self.last_rate
            else:
                t = np.arange(len(self.last_data1), dtype=np.float64)

            # Downsample for display only above this many points (keep full data for export)
            max_plot_points = 100000
            downsampled = len(self.last_data1) > max_plot_points
            if downsampled:
                step = len(self.last_data1) // max_plot_points
                t_plot = t[::step]
                data1_plot = self.last_data1[::step]
                data2_plot = self.last_data2[::step]
            else:
                t_plot = t
                data1_plot = self.last_data1
                data2_plot = self.last_data2

            # Draw both traces with explicit pen creation and antialiasing disabled for performance
            pen1 = pg.mkPen(color=(50,100,200), width=2)
            pen2 = pg.mkPen(color=(200,50,50), width=2)
            
            self.plot1.plot(t_plot, data1_plot, pen=pen1, antialias=False)
            self.plot2.plot(t_plot, data2_plot, pen=pen2, antialias=False)

            # Update button states
            self.export_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # Add markers to both plots
            self._update_markers()

            duration_s = len(self.last_data1) / self.last_rate if self.last_rate > 0 else 0.0
            status = (
                f"Captured {len(self.last_data1):,} samples in {duration_s:.3f} s "
                f"(measured rate: {self.last_rate:,.0f} Hz)"
            )
            if downsampled:
                status += f" - plot downsampled to {len(data1_plot):,} points, full data kept for export"
            self.status_label.setText(status)

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Plot error", str(e))
            print(f"Detailed plot error: {e}")
            import traceback
            traceback.print_exc()

    def export_data(self):
        if self.last_data1 is None or self.last_data2 is None:
            return
        
        # Use last path or default filename
        default_name = f"{self.last_chan1}_{self.last_chan2}_capture.csv"
        if self.last_export_path:
            import os
            default_name = os.path.join(os.path.dirname(self.last_export_path), default_name)
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", default_name, 
            "CSV Files (*.csv);;NumPy Files (*.npy)"
        )
        if not path:
            return
            
        # Remember this path for next time
        self.last_export_path = path
        
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
                
            # Also save PNG screenshot of the plots
            png_path = path.rsplit('.', 1)[0] + '.png'
            try:
                exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
                exporter.parameters()['width'] = 1200  # High resolution
                exporter.export(png_path)
                QtWidgets.QMessageBox.information(self, "Export", 
                    f"Data saved to:\n{path}\n\nPlot image saved to:\n{png_path}")
            except Exception as img_e:
                QtWidgets.QMessageBox.information(self, "Export", 
                    f"Data saved to:\n{path}\n\nNote: Could not save plot image: {img_e}")
                    
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
        """Improved marker cleanup with explicit error handling."""
        # Clear from plot1
        for item in self._marker_items1:
            try:
                self.plot1.removeItem(item)
            except RuntimeError as e:
                # Item may already be removed - this is OK
                print(f"Warning: Could not remove marker from plot1: {e}")
            except Exception as e:
                print(f"Error removing marker from plot1: {e}")
        
        # Clear from plot2  
        for item in self._marker_items2:
            try:
                self.plot2.removeItem(item)
            except RuntimeError as e:
                # Item may already be removed - this is OK
                print(f"Warning: Could not remove marker from plot2: {e}")
            except Exception as e:
                print(f"Error removing marker from plot2: {e}")
        
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
            if len(self.last_data1) > 0:
                ymax1 = float(np.nanmax(self.last_data1))
            else:
                ymax1 = 0.0
            if len(self.last_data2) > 0:
                ymax2 = float(np.nanmax(self.last_data2))
            else:
                ymax2 = 0.0
        except Exception:
            ymax1 = ymax2 = 0.0

        # Clear existing markers first
        self._clear_markers()

        skipped = 0
        for dt, label in self._event_markers:
            try:
                # seconds since capture_start_dt
                secs = max(0.0, self.capture_start_dt.msecsTo(dt) / 1000.0)
            except Exception:
                secs = 0.0

            # An event that happened after the capture ended has no honest
            # position in this plot - skip it rather than stacking it at the
            # edge, which just looks like a pile of overlapping markers.
            if secs > tmax:
                skipped += 1
                continue
            x = secs

            # Add markers to both plots
            try:
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
            except Exception as e:
                print(f"Error adding marker '{label}' at {x}s: {e}")

        if skipped > 0:
            current = self.status_label.text()
            suffix = f" - {skipped} event(s) occurred after the capture ended and are not shown"
            if suffix not in current:
                self.status_label.setText(current + suffix)

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

    def clear_plots(self):
        """Clear plots, markers, and the last captured data."""
        self._force_clear_plots()
        self._clear_markers()
        self._cleanup_data()
        self.export_btn.setEnabled(False)
        self.status_label.setText("No capture yet.")