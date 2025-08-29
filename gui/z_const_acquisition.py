# z_const_acquisition.py
"""
Constant-height (Z-constant) acquisition tab.

This tab allows:
  • Real-time plotting of topography (nm)
  • Enabling/disabling the SXM feedback loop (via DDE FeedPara)
  • Capturing the last topography value when feedback is disabled
  • Holding and adjusting Z with a QDoubleSpinBox (pm/nm resolution)
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class ZConstAcquisitionTab(QtWidgets.QWidget):
    """
    Constant-height control tab.

    Provides live topography plotting and manual control of Z when feedback
    is disabled. When feedback is enabled, the spinbox is locked and Z is
    controlled by the SXM feedback loop.

    Parameters
    ----------
    dde : object
        DDE client handle (real or mock), provided by SXMConnection.
    driver : object or None
        IOCTL driver handle (SXMIOCTL) or None if unavailable.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, dde, driver=None, parent=None):
        super().__init__(parent)
        self.dde = dde
        self.driver = driver

        self._holding = False
        self._last_topo_nm = 0.0
        self._history_len = 4000

        # ----------------------------
        # UI layout
        # ----------------------------
        vbox = QtWidgets.QVBoxLayout(self)

        # Feedback toggle button
        toolbar = QtWidgets.QHBoxLayout()
        self.btn_toggle = QtWidgets.QPushButton("ENABLED")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self._update_toggle_style()
        self.btn_toggle.clicked.connect(self.toggle_fb)
        toolbar.addWidget(self.btn_toggle)
        vbox.addLayout(toolbar)

        # Plot widget
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Topography (nm)")
        self.plot.setLabel("bottom", "Time (s)")
        self.curve = self.plot.plot(pen=pg.mkPen('b', width=1))
        vbox.addWidget(self.plot)

        # Spinbox for Z hold
        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("Held Z (nm):"))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(6)         # pm precision
        self.spin.setRange(-1e6, 1e6)    # nm range
        self.spin.setSingleStep(0.001)   # default 1 pm step
        self.spin.setEnabled(False)
        self.spin.valueChanged.connect(self.apply_spin)
        form.addWidget(self.spin, 1)
        vbox.addLayout(form)

        # Timer for live plot updates
        self._t0 = QtCore.QTime.currentTime()
        self._xs = np.empty(self._history_len, dtype=float)
        self._ys = np.empty(self._history_len, dtype=float)
        self._n = 0

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)  # ~50 Hz
        self.timer.timeout.connect(self._sample_topo)
        self.timer.start()

    # ------------------------------------------------------------------
    # Feedback control
    # ------------------------------------------------------------------
    def toggle_fb(self):
        """
        Toggle topography feedback control.

        This sends a DDE command:
            FeedPara('enable', value)

        where `value` is an integer flag:
            1 → enable feedback
            0 → disable feedback
        """
        FB_ENABLE = 1
        FB_DISABLE = 0

        if self.btn_toggle.isChecked():
            # Feedback ON (send integer 1)
            try:
                self.dde.feed_para("enable", FB_ENABLE)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "DDE error", str(e))
                return
            self._holding = False
            self.spin.setEnabled(False)
            self.btn_toggle.setText("ENABLED")
        else:
            # Feedback OFF (send integer 0)
            try:
                self.dde.feed_para("enable", FB_DISABLE)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "DDE error", str(e))
                return
            topo_nm = self._last_topo_nm
            self.spin.setValue(topo_nm)
            self._holding = True
            self.spin.setEnabled(True)
            self.btn_toggle.setText("DISABLED")
            if self.driver:
                try:
                    self.driver.write_unit("Topo", topo_nm)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "IOCTL error", str(e))
        self._update_toggle_style()

    def _update_toggle_style(self):
        """Update the background color of the toggle button."""
        if self.btn_toggle.isChecked():
            self.btn_toggle.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        else:
            self.btn_toggle.setStyleSheet("background-color: lightgrey; font-weight: bold;")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_topo(self):
        """Sample the topography channel and update the plot."""
        try:
            topo_nm = float(self.dde.read_topography())  # channel 0 in nm
        except Exception:
            topo_nm = getattr(self, "_sim", 0.0) + 0.001
            self._sim = topo_nm

        self._last_topo_nm = topo_nm

        # Append to ring buffer
        t = self._t0.msecsTo(QtCore.QTime.currentTime()) / 1000.0
        i = self._n % self._history_len
        self._xs[i] = t
        self._ys[i] = topo_nm
        self._n += 1

        if self._n < self._history_len:
            self.curve.setData(self._xs[:self._n], self._ys[:self._n])
        else:
            idx = (np.arange(self._history_len) + i + 1) % self._history_len
            self.curve.setData(self._xs[idx], self._ys[idx])

    # ------------------------------------------------------------------
    # Spinbox editing
    # ------------------------------------------------------------------
    def apply_spin(self, value: float):
        """
        Apply a new Z value when feedback is disabled.

        Parameters
        ----------
        value : float
            Target Z position in nm.
        """
        if not self._holding or self.driver is None:
            return
        try:
            self.driver.write_unit("Topo", value)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "IOCTL error", str(e))
