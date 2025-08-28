# gui/qplus_calibration_tab.py
import datetime
import numpy as np
from typing import List, Tuple, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy import stats
# IOCTL is imported lazily in a helper to avoid import errors on non-Windows

from .common import confirm_high_voltage


class QplusCalibrationTab(QtWidgets.QWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self.setWindowTitle("Q+ Amplitude Calibration")
        self.layout = QtWidgets.QVBoxLayout(self)

        # UI elements
        self.controls_group = QtWidgets.QGroupBox("Setup")
        self.controls_layout = QtWidgets.QGridLayout(self.controls_group)

        self.amp_start = QtWidgets.QDoubleSpinBox()
        self.amp_start.setRange(0.0, 1000.0)
        self.amp_start.setValue(10.0)
        self.amp_start.setSuffix(" mV")

        self.amp_end = QtWidgets.QDoubleSpinBox()
        self.amp_end.setRange(0.0, 1000.0)
        self.amp_end.setValue(100.0)
        self.amp_end.setSuffix(" mV")

        self.num_points = QtWidgets.QSpinBox()
        self.num_points.setRange(2, 200)
        self.num_points.setValue(10)

        self.stab_time = QtWidgets.QDoubleSpinBox()
        self.stab_time.setRange(0.0, 60.0)
        self.stab_time.setValue(2.0)
        self.stab_time.setSuffix(" s")

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        # Labels reflect that we sweep Amplitude setpoint (Edit23)
        self.controls_layout.addWidget(QtWidgets.QLabel("Setpoint start (Edit23):"), 0, 0)
        self.controls_layout.addWidget(self.amp_start, 0, 1)
        self.controls_layout.addWidget(QtWidgets.QLabel("Setpoint end (Edit23):"), 1, 0)
        self.controls_layout.addWidget(self.amp_end, 1, 1)
        self.controls_layout.addWidget(QtWidgets.QLabel("Points:"), 2, 0)
        self.controls_layout.addWidget(self.num_points, 2, 1)
        self.controls_layout.addWidget(QtWidgets.QLabel("Stabilize:"), 3, 0)
        self.controls_layout.addWidget(self.stab_time, 3, 1)
        self.controls_layout.addWidget(self.start_btn, 4, 0)
        self.controls_layout.addWidget(self.stop_btn, 4, 1)

        self.layout.addWidget(self.controls_group)

        # Plot
        self.plot_widget = pg.PlotWidget(title="Topography change vs amplitude setpoint (Edit23)")
        self.plot_widget.setLabel("left", "ΔTopography", units="pm")
        self.plot_widget.setLabel("bottom", "Amplitude setpoint", units="mV")
        self.plot_data = self.plot_widget.plot([], [], pen=None, symbol="o")
        self.layout.addWidget(self.plot_widget)

        # Log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.layout.addWidget(self.log)

        # Data
        self.measurement_data: List[Tuple[float, float]] = []
        self.amplitude_points: List[float] = []
        self.current_point: int = 0
        self.baseline_topo: Optional[float] = None
        self.calibration_factor: Optional[float] = None

        # Signals
        self.start_btn.clicked.connect(self.start_calibration)
        self.stop_btn.clicked.connect(self.stop_calibration)

    def _read_topography_nm(self) -> float:
        """Read one topography sample in nm using DDE GetChannel(0) or IOCTL fallback."""
        # Try DDE first
        try:
            if hasattr(self.dde, "read_topography"):
                return float(self.dde.read_topography())
            if hasattr(self.dde, "read_channel"):
                return float(self.dde.read_channel(0))
        except Exception:
            pass
        # IOCTL fallback
        try:
            try:
                from sxm_ncafm_control.device_driver import SXMIOCTL, CHANNELS
            except Exception:
                from device_driver import SXMIOCTL, CHANNELS
            if not hasattr(self, "_ioctl"):
                self._ioctl = SXMIOCTL()
            idx, _short, _unit, scale = CHANNELS["Topo"]  # unit: nm
            raw = self._ioctl.read_raw(idx)
            return float(raw) * float(scale)
        except Exception as e:
            raise RuntimeError(f"Cannot read topography: {e}")

    def _log(self, text: str) -> None:
        self.log.append(text)
        self.log.moveCursor(QtGui.QTextCursor.End)

    def _build_amplitude_points(self) -> None:
        start_amp = self.amp_start.value()
        end_amp = self.amp_end.value()
        num_points = self.num_points.value()

        if num_points < 2 or end_amp <= start_amp:
            raise ValueError("Invalid amplitude range or num_points.")

        self.amplitude_points = list(np.linspace(start_amp, end_amp, num_points))
        self.current_point = 0
        self.measurement_data.clear()
        self.plot_data.setData([], [])
        self.calibration_factor = None

        self._log(f"Sweep Edit23 from {start_amp:.1f} to {end_amp:.1f} mV with {num_points} points")

        # Baseline
        self._log("Reading baseline topography...")
        try:
            self.baseline_topo = self._read_topography_nm()
            self._log(f"Baseline topography: {self.baseline_topo:.2f} nm")
            self.next_measurement()
        except Exception as e:
            self._log(f"Failed to read baseline topography: {e}")
            self.stop_calibration()

    def start_calibration(self) -> None:
        if not confirm_high_voltage(self, "Start calibration with high voltage enabled?"):
            return
        try:
            self._build_amplitude_points()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Setup error", str(e))

    def stop_calibration(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def next_measurement(self) -> None:
        if self.current_point >= len(self.amplitude_points):
            self.finish_calibration()
            return

        amp_mv = self.amplitude_points[self.current_point]
        self._log(f"Set amplitude setpoint (Edit23) to {amp_mv:.1f} mV and wait {self.stab_time.value():.1f} s")

        try:
            # Sweep the amplitude feedback setpoint (Parameter mode) -> Edit23
            self.dde.send_scanpara("Edit23", amp_mv)

            # Wait, then sample
            QtCore.QTimer.singleShot(
                int(self.stab_time.value() * 1000),
                self.read_topography
            )

        except Exception as e:
            self._log(f"Error setting Edit23: {e}")
            self.stop_calibration()

    def read_topography(self):
        """Read topography after stabilization."""
        try:
            current_topo = self._read_topography_nm()
            topo_change_pm = (current_topo - self.baseline_topo) * 1000.0  # nm -> pm

            amp_mv = self.amplitude_points[self.current_point]
            self.measurement_data.append((amp_mv, topo_change_pm))

            # Update plot
            xs = [p[0] for p in self.measurement_data]
            ys = [p[1] for p in self.measurement_data]
            self.plot_data.setData(xs, ys)
            self._log(f"ΔTopo = {topo_change_pm:.2f} pm at Edit23 = {amp_mv:.1f} mV")

            # Next point
            self.current_point += 1
            if self.current_point < len(self.amplitude_points):
                self.next_measurement()
            else:
                self.finish_calibration()

        except Exception as e:
            self._log(f"Error reading topography: {e}")
            self.stop_calibration()

    def finish_calibration(self) -> None:
        if len(self.measurement_data) < 2:
            self._log("Not enough data to fit a line.")
            self.stop_calibration()
            return

        try:
            amps = [point[0] for point in self.measurement_data]
            topo_changes = [point[1] for point in self.measurement_data]

            slope, intercept, r_value, _, _ = stats.linregress(amps, topo_changes)
            r_squared = r_value ** 2
            self.calibration_factor = slope  # pm/mV

            self._log(f"Fit: ΔTopo(pm) = {slope:.3f} * Edit23(mV) + {intercept:.2f} (R²={r_squared:.4f})")
            QtWidgets.QMessageBox.information(
                self,
                "Calibration complete",
                f"Slope = {slope:.3f} pm/mV\nR² = {r_squared:.4f}"
            )
        finally:
            self.stop_calibration()

    def export_results(self) -> None:
        if not self.measurement_data:
            QtWidgets.QMessageBox.information(self, "No data", "Run a calibration first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save results", "qplus_calibration.csv", "CSV (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, 'w') as f:
                f.write(f"# Q+ Amplitude Calibration Results\n")
                f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Edit23 Range: {self.amp_start.value():.1f} to {self.amp_end.value():.1f} mV\n")
                f.write(f"# Stabilization Time: {self.stab_time.value():.1f} s\n")

                if self.calibration_factor is not None:
                    amps = [point[0] for point in self.measurement_data]
                    topo_changes = [point[1] for point in self.measurement_data]
                    _, _, r_value, _, _ = stats.linregress(amps, topo_changes)
                    r_squared = r_value ** 2

                    f.write(f"# Calibration slope (pm/mV): {self.calibration_factor:.3f}\n")
                    f.write(f"# R²: {r_squared:.4f}\n")

                f.write(f"#\n")
                f.write("Edit23_mV,Topography_Change_pm\n")

                for amp_mv, topo_pm in self.measurement_data:
                    f.write(f"{amp_mv:.1f},{topo_pm:.2f}\n")

            QtWidgets.QMessageBox.information(self, "Export Complete",
                                              f"Calibration results exported to:\n{path}")

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Error",
                                          f"Failed to export data:\n{e}")
