# gui/qplus_calibration_tab.py
import datetime
import numpy as np
from typing import List, Tuple, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy import stats

from .common import confirm_high_voltage


class QplusCalibrationTab(QtWidgets.QWidget):
    """Q+ amplitude calibration through topography measurement."""

    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self.measurement_data: List[Tuple[float, float]] = []  # (amplitude_mv, topo_change_pm)
        self.calibration_factor = None
        self.is_measuring = False
        self.baseline_topo = None
        
        vbox = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "<b>Q+ Amplitude Calibration Procedure:</b><br>"
            "1. Engage tip in topography constant current mode<br>"
            "2. Set amplitude range and number of points<br>"
            "3. Click 'Start Calibration' - system will automatically:<br>"
            "&nbsp;&nbsp;• Set each amplitude setpoint<br>"
            "&nbsp;&nbsp;• Wait for stabilization<br>"
            "&nbsp;&nbsp;• Record topography change<br>"
            "4. Review linear fit to get calibration factor"
        )
        instructions.setStyleSheet(
            "background: #f0f8ff; border: 1px solid #87ceeb; "
            "border-radius: 5px; padding: 10px; margin: 5px;"
        )
        vbox.addWidget(instructions)
        
        # Controls
        controls_group = QtWidgets.QGroupBox("Calibration Parameters")
        controls_layout = QtWidgets.QGridLayout(controls_group)
        
        # Amplitude range
        controls_layout.addWidget(QtWidgets.QLabel("Amplitude Start (mV):"), 0, 0)
        self.amp_start = QtWidgets.QDoubleSpinBox()
        self.amp_start.setRange(0.1, 1000.0)
        self.amp_start.setValue(50.0)
        self.amp_start.setDecimals(1)
        controls_layout.addWidget(self.amp_start, 0, 1)
        
        controls_layout.addWidget(QtWidgets.QLabel("Amplitude End (mV):"), 0, 2)
        self.amp_end = QtWidgets.QDoubleSpinBox()
        self.amp_end.setRange(0.1, 1000.0)
        self.amp_end.setValue(200.0)
        self.amp_end.setDecimals(1)
        controls_layout.addWidget(self.amp_end, 0, 3)
        
        controls_layout.addWidget(QtWidgets.QLabel("Number of Points:"), 1, 0)
        self.num_points = QtWidgets.QSpinBox()
        self.num_points.setRange(3, 50)
        self.num_points.setValue(10)
        controls_layout.addWidget(self.num_points, 1, 1)
        
        controls_layout.addWidget(QtWidgets.QLabel("Stabilization Time (s):"), 1, 2)
        self.stab_time = QtWidgets.QDoubleSpinBox()
        self.stab_time.setRange(0.5, 30.0)
        self.stab_time.setValue(2.0)
        self.stab_time.setDecimals(1)
        controls_layout.addWidget(self.stab_time, 1, 3)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Calibration")
        self.btn_start.clicked.connect(self.start_calibration)
        button_layout.addWidget(self.btn_start)
        
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_calibration)
        button_layout.addWidget(self.btn_stop)
        
        self.btn_clear = QtWidgets.QPushButton("Clear Data")
        self.btn_clear.clicked.connect(self.clear_data)
        button_layout.addWidget(self.btn_clear)
        
        self.btn_export = QtWidgets.QPushButton("Export Results")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
        button_layout.addWidget(self.btn_export)
        
        button_layout.addStretch()
        controls_layout.addLayout(button_layout, 2, 0, 1, 4)
        
        vbox.addWidget(controls_group)
        
        # Results display
        results_group = QtWidgets.QGroupBox("Calibration Results")
        results_layout = QtWidgets.QHBoxLayout(results_group)
        
        # Current measurement info
        info_layout = QtWidgets.QVBoxLayout()
        self.lbl_current_amp = QtWidgets.QLabel("Current Amplitude: --")
        self.lbl_progress = QtWidgets.QLabel("Progress: --")
        self.lbl_calibration = QtWidgets.QLabel("Calibration Factor: --")
        self.lbl_r_squared = QtWidgets.QLabel("R²: --")
        
        for lbl in [self.lbl_current_amp, self.lbl_progress, 
                   self.lbl_calibration, self.lbl_r_squared]:
            lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
            info_layout.addWidget(lbl)
        
        info_layout.addStretch()
        results_layout.addLayout(info_layout)
        
        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground("white")
        self.plot.setLabel("left", "Topography Change", units="pm")
        self.plot.setLabel("bottom", "Q+ Amplitude Setpoint", units="mV")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Style the plot
        grid_pen = pg.mkPen(color=(70, 70, 70), width=0.5)
        for axis in ['bottom', 'left']:
            self.plot.getAxis(axis).setPen(grid_pen)
            
        results_layout.addWidget(self.plot, 2)  # Give more space to plot
        vbox.addWidget(results_group)
        
        # Log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        vbox.addWidget(QtWidgets.QLabel("Calibration Log:"))
        vbox.addWidget(self.log)
        
        # Timer for automated measurements
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_measurement)
        
        self.current_point = 0
        self.amplitude_points = []
        
    def _log(self, message: str):
        """Add timestamped message to log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        
    def start_calibration(self):
        """Start the automated calibration sequence."""
        start_amp = self.amp_start.value()
        end_amp = self.amp_end.value()
        num_points = self.num_points.value()
        
        if start_amp >= end_amp:
            QtWidgets.QMessageBox.warning(self, "Invalid Range", 
                "Start amplitude must be less than end amplitude.")
            return
            
        # Generate amplitude points
        self.amplitude_points = np.linspace(start_amp, end_amp, num_points).tolist()
        self.measurement_data = []
        self.current_point = 0
        self.is_measuring = True
        self.baseline_topo = None
        
        # Update UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        
        self._log(f"Starting calibration: {start_amp:.1f} to {end_amp:.1f} mV, {num_points} points")
        
        # Get baseline topography reading
        self._log("Establishing baseline topography...")
        try:
            # Read current topography (assuming DNC channel 1 is topography)
            self.baseline_topo = self.dde.read_dncpara(1)
            self._log(f"Baseline topography: {self.baseline_topo:.2f}")
            
            # Start first measurement
            self.next_measurement()
            
        except Exception as e:
            self._log(f"Error reading baseline: {e}")
            self.stop_calibration()
            
    def next_measurement(self):
        """Perform next measurement in the sequence."""
        if not self.is_measuring or self.current_point >= len(self.amplitude_points):
            self.finish_calibration()
            return
            
        amp_setpoint = self.amplitude_points[self.current_point]
        
        try:
            # Set amplitude reference (Edit23)
            if not confirm_high_voltage(self, "Amplitude Reference", amp_setpoint / 1000.0):
                self.stop_calibration()
                return
                
            self.dde.send_scanpara("Edit23", amp_setpoint / 1000.0)  # Convert mV to V
            
            # Update display
            self.lbl_current_amp.setText(f"Current Amplitude: {amp_setpoint:.1f} mV")
            self.lbl_progress.setText(f"Progress: {self.current_point + 1}/{len(self.amplitude_points)}")
            
            self._log(f"Set amplitude to {amp_setpoint:.1f} mV, waiting for stabilization...")
            
            # Wait for stabilization then read topography
            QtCore.QTimer.singleShot(
                int(self.stab_time.value() * 1000), 
                self.read_topography
            )
            
        except Exception as e:
            self._log(f"Error setting amplitude: {e}")
            self.stop_calibration()
            
    def read_topography(self):
        """Read topography after stabilization."""
        try:
            # Read current topography
            current_topo = self.dde.read_dncpara(1)
            topo_change_pm = (current_topo - self.baseline_topo) * 1000  # Convert to pm
            
            amp_mv = self.amplitude_points[self.current_point]
            self.measurement_data.append((amp_mv, topo_change_pm))
            
            self._log(f"Point {self.current_point + 1}: {amp_mv:.1f} mV → {topo_change_pm:.1f} pm change")
            
            # Update plot
            self.update_plot()
            
            # Move to next point
            self.current_point += 1
            
            # Continue with next measurement after short delay
            QtCore.QTimer.singleShot(200, self.next_measurement)
            
        except Exception as e:
            self._log(f"Error reading topography: {e}")
            self.stop_calibration()
            
    def finish_calibration(self):
        """Complete calibration and calculate results."""
        if len(self.measurement_data) < 3:
            self._log("Insufficient data points for calibration.")
            self.stop_calibration()
            return
            
        # Perform linear regression
        amps = [point[0] for point in self.measurement_data]
        topo_changes = [point[1] for point in self.measurement_data]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(amps, topo_changes)
        
        # Calibration factor: nm per mV
        self.calibration_factor = slope / 1000.0  # Convert pm/mV to nm/mV
        r_squared = r_value ** 2
        
        # Update display
        self.lbl_calibration.setText(f"Calibration Factor: {self.calibration_factor:.6f} nm/mV")
        self.lbl_r_squared.setText(f"R²: {r_squared:.4f}")
        
        # Add fit line to plot
        fit_amps = np.array([min(amps), max(amps)])
        fit_topo = slope * fit_amps + intercept
        self.plot.plot(fit_amps, fit_topo, pen=pg.mkPen('r', width=2), name='Linear Fit')
        
        self._log(f"Calibration complete!")
        self._log(f"Factor: {self.calibration_factor:.6f} nm/mV (R² = {r_squared:.4f})")
        
        if r_squared < 0.95:
            self._log("WARNING: Low R² value suggests poor linear fit.")
        
        self.stop_calibration()
        
    def stop_calibration(self):
        """Stop the calibration process."""
        self.is_measuring = False
        self.timer.stop()
        
        # Update UI
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(len(self.measurement_data) > 0)
        
        self.lbl_current_amp.setText("Current Amplitude: --")
        self.lbl_progress.setText("Progress: --")
        
    def update_plot(self):
        """Update the calibration plot with current data."""
        if not self.measurement_data:
            return
            
        self.plot.clear()
        
        amps = [point[0] for point in self.measurement_data]
        topo_changes = [point[1] for point in self.measurement_data]
        
        # Plot data points
        scatter = pg.ScatterPlotItem(
            x=amps, y=topo_changes, 
            pen=pg.mkPen('b', width=2), 
            brush=pg.mkBrush('lightblue'), 
            size=8, 
            symbol='o'
        )
        self.plot.addItem(scatter)
        
        # If we have enough points, show preliminary fit
        if len(self.measurement_data) >= 3:
            slope, intercept, r_value, _, _ = stats.linregress(amps, topo_changes)
            fit_amps = np.array([min(amps), max(amps)])
            fit_topo = slope * fit_amps + intercept
            self.plot.plot(fit_amps, fit_topo, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            
    def clear_data(self):
        """Clear all measurement data."""
        reply = QtWidgets.QMessageBox.question(
            self, "Clear Data", 
            "Clear all calibration data?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.measurement_data = []
            self.calibration_factor = None
            self.plot.clear()
            self.log.clear()
            
            self.lbl_calibration.setText("Calibration Factor: --")
            self.lbl_r_squared.setText("R²: --")
            self.btn_export.setEnabled(False)
            
    def export_results(self):
        """Export calibration data and results."""
        if not self.measurement_data:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"qplus_calibration_{timestamp}.csv"
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Calibration Results", default_name,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not path:
            return
            
        try:
            with open(path, 'w') as f:
                # Header with calibration info
                f.write(f"# Q+ Amplitude Calibration Results\n")
                f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Amplitude Range: {self.amp_start.value():.1f} to {self.amp_end.value():.1f} mV\n")
                f.write(f"# Stabilization Time: {self.stab_time.value():.1f} s\n")
                
                if self.calibration_factor is not None:
                    # Calculate R² again
                    amps = [point[0] for point in self.measurement_data]
                    topo_changes = [point[1] for point in self.measurement_data]
                    _, _, r_value, _, _ = stats.linregress(amps, topo_changes)
                    r_squared = r_value ** 2
                    
                    f.write(f"# Calibration Factor: {self.calibration_factor:.6f} nm/mV\n")
                    f.write(f"# R²: {r_squared:.4f}\n")
                
                f.write(f"#\n")
                f.write("Amplitude_mV,Topography_Change_pm\n")
                
                # Data
                for amp_mv, topo_pm in self.measurement_data:
                    f.write(f"{amp_mv:.1f},{topo_pm:.2f}\n")
                    
            QtWidgets.QMessageBox.information(self, "Export Complete", 
                f"Calibration results exported to:\n{path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Error", 
                f"Failed to export data:\n{e}")