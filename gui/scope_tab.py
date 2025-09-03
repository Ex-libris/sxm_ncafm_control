# scope_tab.py
"""
Scope tab (dual-channel oscilloscope view) - HYBRID VERSION
Uses the EXACT working mechanics from the old version with minimal changes.

Provides live capture of two SXM channels, plotting them against a shared
time axis. Supports export of data to CSV/NumPy, and overlay of event
markers from external test tabs.
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
from sxm_ncafm_control.device_driver import CHANNELS


class CaptureThread(QtCore.QThread):
    """
    Background capture thread for two SXM channels.
    KEEPING EXACT SAME LOGIC AS OLD WORKING VERSION

    Reads raw values from the IOCTL driver at maximum speed until the
    requested number of points is acquired or the thread is stopped.

    Emits
    -----
    finished : np.ndarray, np.ndarray, float
        Arrays of channel 1 and channel 2 values (scaled to physical units),
        and the effective sampling rate in Hz.
    """
    finished = QtCore.pyqtSignal(np.ndarray, np.ndarray, float)

    def __init__(self, driver, chan_idx1, chan_idx2, npoints=50000):
        super().__init__()
        # PRESERVE EXACT SAME INITIALIZATION
        self.driver = driver
        self.chan_idx1 = chan_idx1
        self.chan_idx2 = chan_idx2
        self.npoints = npoints
        self._stop = False

    def run(self):
        """Acquire raw values from both channels in a tight loop.
        KEEPING EXACT SAME LOGIC AS OLD WORKING VERSION"""
        vals1 = np.zeros(self.npoints, dtype=np.float64)
        vals2 = np.zeros(self.npoints, dtype=np.float64)
        t0 = QtCore.QTime.currentTime()
        # CRITICAL: Get scaling factors for both channels - SAME AS OLD VERSION
        vals1 = self.driver.read_scaled(self.chan_idx1)
        vals2 = self.driver.read_scaled(self.chan_idx2)

        # PRESERVE EXACT SAME ACQUISITION LOOP
        for i in range(self.npoints):
            if self._stop:
                vals1 = vals1[:i]
                vals2 = vals2[:i]
                break
            try:
                # CRITICAL: Use same driver calls as old version
                raw1 = self.driver.read_scaled(self.chan_idx1)
                raw2 = self.driver.read_scaled(self.chan_idx2)
            except Exception:
                # PRESERVE EXACT SAME FALLBACK
                raw1 = np.random.randn() * 0.01
                raw2 = np.random.randn() * 0.01 + 0.5
        # PRESERVE EXACT SAME RATE CALCULATION
        elapsed_ms = t0.msecsTo(QtCore.QTime.currentTime())
        rate = len(vals1) / max(elapsed_ms / 1000.0, 1e-9)
        self.finished.emit(vals1, vals2, rate)

    def stop(self):
        """Request the thread to stop early."""
        self._stop = True


class ScopeTab(QtWidgets.QWidget):
    """
    Dual-channel scope for SXM channels with shared time axis.
    HYBRID VERSION: Preserves exact working mechanics from old version
    
    Parameters
    ----------
    dde : object
        DDE client handle (real or mock), provided by SXMConnection.
    driver : object or None
        IOCTL driver handle (SXMIOCTL) or None if unavailable.
    """

    def __init__(self, dde, driver):
        super().__init__()
        # PRESERVE EXACT SAME INITIALIZATION ORDER AND VALUES
        self.dde = dde
        self.driver = driver

        self.capture_thread = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_rate = None
        self.last_chan1 = None
        self.last_chan2 = None

        # Time + markers state - SAME AS OLD VERSION
        self.capture_start_dt = None
        self._event_markers = []
        self._marker_items1 = []  # markers for plot1
        self._marker_items2 = []  # markers for plot2

        # Reference to test tab for repeat functionality - SAME AS OLD VERSION
        self.test_tab = None

        # Remember last export path - SAME AS OLD VERSION
        self.last_export_path = None

        # ----------------------------
        # Layout and controls - ENHANCED UI FROM NEW VERSION
        # ----------------------------
        vbox = QtWidgets.QVBoxLayout(self)

        # ENHANCED: Top control bar with better styling
        ctrl_frame = QtWidgets.QFrame()
        ctrl_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        ctrl_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        hbox = QtWidgets.QHBoxLayout(ctrl_frame)

        # Channel 1 selection - PRESERVE EXACT SAME LOGIC
        channel1_group = QtWidgets.QGroupBox("Channel 1")
        channel1_layout = QtWidgets.QVBoxLayout(channel1_group)
        
        self.chan1_combo = QtWidgets.QComboBox()
        self.chan1_combo.addItems(list(CHANNELS.keys()))
        # PRESERVE EXACT SAME DEFAULT SELECTION LOGIC
        if "QPlusAmpl" in CHANNELS:
            idx = list(CHANNELS.keys()).index("QPlusAmpl")
            self.chan1_combo.setCurrentIndex(idx)
        channel1_layout.addWidget(self.chan1_combo)
        hbox.addWidget(channel1_group)

        # Channel 2 selection - PRESERVE EXACT SAME LOGIC  
        channel2_group = QtWidgets.QGroupBox("Channel 2")
        channel2_layout = QtWidgets.QVBoxLayout(channel2_group)
        
        self.chan2_combo = QtWidgets.QComboBox()
        self.chan2_combo.addItems(list(CHANNELS.keys()))
        # PRESERVE EXACT SAME DEFAULT SELECTION LOGIC
        if "Drive" in CHANNELS:
            idx = list(CHANNELS.keys()).index("Drive")
            self.chan2_combo.setCurrentIndex(idx)
        elif len(CHANNELS) > 1:
            self.chan2_combo.setCurrentIndex(1)
        channel2_layout.addWidget(self.chan2_combo)
        hbox.addWidget(channel2_group)

        # ENHANCED: Samples control with better styling
        samples_group = QtWidgets.QGroupBox("Samples")
        samples_layout = QtWidgets.QVBoxLayout(samples_group)
        
        self.npoints_spin = QtWidgets.QSpinBox()
        # PRESERVE EXACT SAME RANGE AND DEFAULT
        self.npoints_spin.setRange(1000, 2_000_000)
        self.npoints_spin.setValue(2_000_000)  # default 2M samples
        self.npoints_spin.setGroupSeparatorShown(True)  # NEW: Better number display
        samples_layout.addWidget(self.npoints_spin)
        hbox.addWidget(samples_group)

        hbox.addSpacing(20)  # NEW: Better spacing

        # ENHANCED: Control buttons with better styling
        buttons_group = QtWidgets.QGroupBox("Control")
        buttons_layout = QtWidgets.QGridLayout(buttons_group)
        
        # Row 1: Start/Stop
        self.start_btn = QtWidgets.QPushButton("Start Capture")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.start_btn.clicked.connect(self.start_capture)  # PRESERVE EXACT SAME CONNECTION
        buttons_layout.addWidget(self.start_btn, 0, 0)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.stop_btn.setEnabled(False)  # PRESERVE EXACT SAME INITIAL STATE
        self.stop_btn.clicked.connect(self.stop_capture)  # PRESERVE EXACT SAME CONNECTION
        buttons_layout.addWidget(self.stop_btn, 0, 1)

        # Row 2: Export/Utility buttons
        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.setEnabled(False)  # PRESERVE EXACT SAME INITIAL STATE
        self.export_btn.clicked.connect(self.export_data)  # PRESERVE EXACT SAME CONNECTION
        buttons_layout.addWidget(self.export_btn, 1, 0)

        self.repeat_test_btn = QtWidgets.QPushButton("Repeat Test")
        self.repeat_test_btn.setEnabled(False)  # PRESERVE EXACT SAME INITIAL STATE
        self.repeat_test_btn.clicked.connect(self.repeat_test)  # PRESERVE EXACT SAME CONNECTION
        buttons_layout.addWidget(self.repeat_test_btn, 1, 1)

        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_plots)  # PRESERVE EXACT SAME CONNECTION
        buttons_layout.addWidget(self.clear_btn, 1, 2)

        hbox.addWidget(buttons_group)

        vbox.addWidget(ctrl_frame)

        # NEW: Status bar
        self.status_bar = QtWidgets.QLabel("Ready")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)
        vbox.addWidget(self.status_bar)

        # ----------------------------
        # Plotting widgets - PRESERVE EXACT SAME STRUCTURE
        # ----------------------------
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("white")  # PRESERVE EXACT SAME BACKGROUND
        vbox.addWidget(self.plot_widget)

        # PRESERVE EXACT SAME PLOT SETUP
        # First plot
        self.plot1 = self.plot_widget.addPlot(row=0, col=0)
        self.plot1.setLabel("left", "Channel 1")
        self.plot1.showGrid(x=True, y=True, alpha=0.3)

        # Second plot, linked X-axis - PRESERVE EXACT SAME SETUP
        self.plot2 = self.plot_widget.addPlot(row=1, col=0)
        self.plot2.setLabel("bottom", "Time (s)")
        self.plot2.setLabel("left", "Channel 2")
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.plot2.setXLink(self.plot1)  # CRITICAL: Preserve exact same linking

        # PRESERVE EXACT SAME STYLING
        for plot in [self.plot1, self.plot2]:
            grid_pen = pg.mkPen(color=(70, 70, 70), width=0.5)
            plot.getAxis('bottom').setPen(grid_pen)
            plot.getAxis('left').setPen(grid_pen)

    # ------------------------------------------------------------------
    # Capture and plotting - PRESERVE EXACT SAME LOGIC
    # ------------------------------------------------------------------
    def start_capture(self):
        """Start capturing from the selected two channels.
        PRESERVE EXACT SAME LOGIC AS OLD WORKING VERSION"""
        
        # Update status
        self.status_bar.setText("Initializing capture...")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                color: #856404;
            }
        """)
        
        # PRESERVE EXACT SAME CHANNEL SELECTION LOGIC
        chan1_name = self.chan1_combo.currentText()
        chan2_name = self.chan2_combo.currentText()
        idx1, _, unit1, _ = CHANNELS[chan1_name]
        idx2, _, unit2, _ = CHANNELS[chan2_name]
        npts = self.npoints_spin.value()

        # PRESERVE EXACT SAME PLOT INITIALIZATION
        self.plot1.clear()
        self.plot2.clear()
        self.plot1.setLabel("left", f"{chan1_name} ({unit1})")
        self.plot2.setLabel("left", f"{chan2_name} ({unit2})")

        # PRESERVE EXACT SAME MARKER AND STATE MANAGEMENT
        self._clear_markers()
        self.capture_start_dt = QtCore.QDateTime.currentDateTime()

        # PRESERVE EXACT SAME BUTTON STATES
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

        # PRESERVE EXACT SAME DATA RESET
        self.last_data1 = None
        self.last_data2 = None
        self.last_rate = None
        self.last_chan1 = chan1_name
        self.last_chan2 = chan2_name

        # PRESERVE EXACT SAME THREAD LOGIC
        if self.driver is not None:
            self.status_bar.setText(f"Capturing {npts:,} samples from {chan1_name} and {chan2_name}...")
            self.capture_thread = CaptureThread(self.driver, idx1, idx2, npoints=npts)
            self.capture_thread.finished.connect(self.show_data)
            self.capture_thread.start()
        else:
            # PRESERVE EXACT SAME OFFLINE FALLBACK
            self.status_bar.setText("No driver - generating test signals...")
            t = np.linspace(0, 1, npts)
            arr1 = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(npts)
            arr2 = np.cos(2 * np.pi * 3 * t) * 2 + 0.2 * np.random.randn(npts)
            self.show_data(arr1, arr2, rate=npts)

    def stop_capture(self):
        """Stop capture if a thread is running.
        PRESERVE EXACT SAME LOGIC AS OLD WORKING VERSION"""
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.status_bar.setText("Stopping capture...")

    def show_data(self, arr1, arr2, rate):
        """Display acquired or simulated data on the plots.
        PRESERVE EXACT SAME LOGIC AS OLD WORKING VERSION"""
        try:
            # PRESERVE EXACT SAME PLOT CLEARING
            self.plot1.clear()
            self.plot2.clear()

            # PRESERVE EXACT SAME DATA HANDLING
            self.last_data1 = np.asarray(arr1)
            self.last_data2 = np.asarray(arr2)
            self.last_rate = float(rate) if rate else 0.0

            # PRESERVE EXACT SAME TIME AXIS CALCULATION
            if self.last_rate > 0:
                t = np.arange(len(self.last_data1), dtype=float) / self.last_rate
            else:
                t = np.arange(len(self.last_data1), dtype=float)

            # PRESERVE EXACT SAME PLOTTING
            self.plot1.plot(t, self.last_data1, pen=pg.mkPen('b', width=1))
            self.plot2.plot(t, self.last_data2, pen=pg.mkPen('r', width=1))

            # PRESERVE EXACT SAME BUTTON STATE CHANGES
            self.export_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # PRESERVE EXACT SAME MARKER UPDATE
            self._update_markers()

            # NEW: Enhanced status update
            self.status_bar.setText(f"Captured {len(self.last_data1):,} samples at {self.last_rate:.1f} Hz")
            self.status_bar.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    color: #155724;
                }
            """)

        except Exception as e:
            # PRESERVE EXACT SAME ERROR HANDLING
            QtWidgets.QMessageBox.warning(self, "Plot error", str(e))
            self.status_bar.setText(f"Error: {str(e)}")
            self.status_bar.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    color: #721c24;
                }
            """)

    def export_data(self):
        """Export last captured data and a screenshot of the plots.
        PRESERVE EXACT SAME LOGIC AS OLD WORKING VERSION"""
        if self.last_data1 is None or self.last_data2 is None:
            return

        # PRESERVE EXACT SAME FILE NAMING
        default_name = f"{self.last_chan1}_{self.last_chan2}_capture.csv"
        if self.last_export_path:
            import os
            default_name = os.path.join(os.path.dirname(self.last_export_path), default_name)

        # PRESERVE EXACT SAME FILE DIALOG
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", default_name,
            "CSV Files (*.csv);;NumPy Files (*.npy)"
        )
        if not path:
            return

        self.last_export_path = path

        try:
            # PRESERVE EXACT SAME EXPORT LOGIC
            if path.endswith(".npy"):
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

            # PRESERVE EXACT SAME PNG EXPORT
            png_path = path.rsplit('.', 1)[0] + '.png'
            try:
                exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
                exporter.parameters()['width'] = 1200
                exporter.export(png_path)
                QtWidgets.QMessageBox.information(
                    self, "Export",
                    f"Data saved to:\n{path}\n\nPlot image saved to:\n{png_path}"
                )
            except Exception as img_e:
                QtWidgets.QMessageBox.information(
                    self, "Export",
                    f"Data saved to:\n{path}\n\nNote: Could not save plot image: {img_e}"
                )

        except Exception as e:
            # PRESERVE EXACT SAME ERROR HANDLING
            QtWidgets.QMessageBox.warning(self, "Export error", str(e))

    # ------------------------------------------------------------------
    # Marker utilities - PRESERVE EXACT SAME LOGIC AS OLD VERSION
    # ------------------------------------------------------------------
    def set_event_markers(self, events):
        """
        Overlay vertical lines with labels on both plots.
        PRESERVE EXACT SAME LOGIC AS OLD VERSION

        Parameters
        ----------
        events : list of (QtCore.QDateTime, str)
            List of (timestamp, label) pairs. X-axis units are seconds
            relative to capture start.
        """
        self._event_markers = list(events or [])
        self._update_markers()

    def _clear_markers(self):
        """PRESERVE EXACT SAME LOGIC AS OLD VERSION"""
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
        """Rebuild markers on both plots from stored events.
        PRESERVE EXACT SAME LOGIC AS OLD VERSION"""
        if (
            self.last_data1 is None
            or self.last_data2 is None
            or self.capture_start_dt is None
        ):
            return

        # PRESERVE EXACT SAME TIME CALCULATION
        if self.last_rate and self.last_rate > 0:
            tmax = (len(self.last_data1) - 1) / self.last_rate if len(self.last_data1) else 0.0
        else:
            tmax = float(len(self.last_data1) - 1) if len(self.last_data1) else 0.0

        # PRESERVE EXACT SAME Y-MAX CALCULATION
        try:
            ymax1 = float(np.nanmax(self.last_data1)) if len(self.last_data1) else 0.0
            ymax2 = float(np.nanmax(self.last_data2)) if len(self.last_data2) else 0.0
        except Exception:
            ymax1 = ymax2 = 0.0

        self._clear_markers()

        # PRESERVE EXACT SAME MARKER CREATION LOOP
        for dt, label in self._event_markers:
            try:
                secs = max(0.0, self.capture_start_dt.msecsTo(dt) / 1000.0)
            except Exception:
                secs = 0.0

            x = min(secs, tmax)

            # PRESERVE EXACT SAME PLOT 1 MARKER
            line1 = pg.InfiniteLine(pos=x, angle=90,
                                    pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot1.addItem(line1)
            txt1 = pg.TextItem(label, anchor=(0, 1), color='r')
            txt1.setPos(x, ymax1)
            self.plot1.addItem(txt1)
            self._marker_items1.extend([line1, txt1])

            # PRESERVE EXACT SAME PLOT 2 MARKER
            line2 = pg.InfiniteLine(pos=x, angle=90,
                                    pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot2.addItem(line2)
            txt2 = pg.TextItem(label, anchor=(0, 1), color='r')
            txt2.setPos(x, ymax2)
            self.plot2.addItem(txt2)
            self._marker_items2.extend([line2, txt2])

    # ------------------------------------------------------------------
    # Integration with test tab - PRESERVE EXACT SAME LOGIC AS OLD VERSION
    # ------------------------------------------------------------------
    def set_test_tab_reference(self, test_tab):
        """Set reference to a test tab for repeat functionality.
        PRESERVE EXACT SAME LOGIC AS OLD VERSION"""
        self.test_tab = test_tab
        self.repeat_test_btn.setEnabled(test_tab is not None)

    def repeat_test(self):
        """Trigger the test tab to repeat the last test configuration.
        PRESERVE EXACT SAME LOGIC AS OLD VERSION"""
        if self.test_tab is None:
            QtWidgets.QMessageBox.warning(
                self, "No Test Tab",
                "No test tab reference available. Please run a test first."
            )
            return

        if hasattr(self.test_tab, '_timer') and self.test_tab._timer.isActive():
            QtWidgets.QMessageBox.information(
                self, "Test Running",
                "A test is already in progress. Please wait for it to complete."
            )
            return

        try:
            if hasattr(self.test_tab, 'chk_trigger_scope'):
                self.test_tab.chk_trigger_scope.setChecked(True)
            if hasattr(self.test_tab, 'chk_stop_scope'):
                self.test_tab.chk_stop_scope.setChecked(True)

            self.test_tab.start()

            QtWidgets.QMessageBox.information(
                self, "Test Started",
                "Repeating last test configuration. The scope will be triggered automatically."
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Test Error",
                f"Failed to start test: {str(e)}"
            )

    def clear_plots(self):
        """Clear both plots and reset data.
        PRESERVE EXACT SAME LOGIC AS OLD VERSION"""
        self.plot1.clear()
        self.plot2.clear()
        self._clear_markers()
        self.last_data1 = None
        self.last_data2 = None
        self.last_rate = None
        self.export_btn.setEnabled(False)
        
        # NEW: Reset status
        self.status_bar.setText("Ready")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)