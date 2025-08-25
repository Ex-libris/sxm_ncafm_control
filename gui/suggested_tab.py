import datetime
import numpy as np
from scipy.optimize import curve_fit
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from .common import PARAMS_BASE, confirm_high_voltage


def lorentzian_amp(f, A0, f0, Q):
    """
    Calculate amplitude response using a Lorentzian model.

    Parameters
    ----------
    f : array-like
        Frequency values.
    A0 : float
        Peak amplitude.
    f0 : float
        Resonance frequency.
    Q : float
        Quality factor.

    Returns
    -------
    array-like
        Amplitude values for each frequency.
    """
    return A0 / np.sqrt(1.0 + 4 * Q**2 * ((f - f0) / f0) ** 2)


class SuggestedTab(QtWidgets.QWidget):
    """
    Tab for computing recommended NC-AFM parameters and fitting resonance spectra.

    This tab provides input fields for Q factor, resonance frequency, and PLL bandwidth.
    It calculates suggested feedback gains and time constants, displays formulas,
    and allows loading and fitting of resonance spectra using a Lorentzian model.

    Inputs
    ------
    dde_client : object
        DDE client for communication with the SXM system.
    params_tab : ParamsTab
        Reference to the parameters tab for staging values.

    Outputs
    -------
    Updates recommended Ki, Kp, tau, and PLL time constant fields.
    Can stage or send values to the SXM system or parameters tab.
    Displays fit results and plots for loaded spectra.
    """

    def __init__(self, dde_client, params_tab):
        """
        Recalculate recommended parameters based on current input values.

        Inputs
        ------
        Q : float
            Quality factor.
        f0 : float
            Resonance frequency (Hz).
        BW_PLL : float
            PLL bandwidth (Hz).

        Outputs
        -------
        Updates Ki, Kp, amplitude tau, and PLL time constant fields.
        """
        super().__init__()
        self.dde = dde_client
        self.params_tab = params_tab
        self._last_fit = None

        main_layout = QtWidgets.QHBoxLayout(self)

        # ---------------- Left column ----------------
        left = QtWidgets.QVBoxLayout()

        # Input form
        form = QtWidgets.QGridLayout()
        r = 0
        self.q_val = QtWidgets.QDoubleSpinBox()
        self.q_val.setRange(1, 1e9)
        self.q_val.setDecimals(1)
        self.q_val.setValue(30000.0)

        self.f0_val = QtWidgets.QDoubleSpinBox()
        self.f0_val.setRange(1.0, 1e9)
        self.f0_val.setDecimals(1)
        self.f0_val.setValue(300000.0)
        self.f0_val.setSuffix(" Hz")

        self.bw_pll = QtWidgets.QDoubleSpinBox()
        self.bw_pll.setRange(1.0, 5000.0)
        self.bw_pll.setDecimals(1)
        self.bw_pll.setValue(50.0)
        self.bw_pll.setSuffix(" Hz")

        form.addWidget(QtWidgets.QLabel("Q factor:"), r, 0)
        form.addWidget(self.q_val, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel("f₀ (Hz):"), r, 0)
        form.addWidget(self.f0_val, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel("PLL bandwidth (Hz):"), r, 0)
        form.addWidget(self.bw_pll, r, 1)
        r += 1

        # Outputs
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        form.addWidget(sep, r, 0, 1, 2)
        r += 1

        self.ki_out = QtWidgets.QLineEdit(); self.ki_out.setReadOnly(True)
        self.kp_out = QtWidgets.QLineEdit(); self.kp_out.setReadOnly(True)
        self.tau_amp = QtWidgets.QLineEdit(); self.tau_amp.setReadOnly(True)
        self.tpll_out = QtWidgets.QLineEdit(); self.tpll_out.setReadOnly(True)

        form.addWidget(QtWidgets.QLabel("Suggested Amplitude Ki:"), r, 0)
        form.addWidget(self.ki_out, r, 1); r += 1
        form.addWidget(QtWidgets.QLabel("Suggested Amplitude Kp:"), r, 0)
        form.addWidget(self.kp_out, r, 1); r += 1
        form.addWidget(QtWidgets.QLabel("Suggested Amplitude tau (ms):"), r, 0)
        form.addWidget(self.tau_amp, r, 1); r += 1
        form.addWidget(QtWidgets.QLabel("PLL TimeConstant (ms):"), r, 0)
        form.addWidget(self.tpll_out, r, 1); r += 1

        left.addLayout(form)

        # Notes
        note = QtWidgets.QTextEdit()
        note.setReadOnly(True)
        note.setPlainText(
            "Formulas:\n"
            "  • Ki ≈ 5×10⁸/Q (for 1 V output gain), Kp ≈ 10⁴·Ki.\n"
            "  • Amplitude bandwidth ≈ 10·Q / f₀.\n"
            "  • PLL τ ≈ 1 / (10·BW_PLL). Default BW_PLL = 50 Hz → τ = 2 ms."
        )
        note.setMaximumHeight(100)
        left.addWidget(note)

        # Buttons
        h = QtWidgets.QHBoxLayout()
        self.btn_recalc = QtWidgets.QPushButton("Recalculate")
        self.btn_stage = QtWidgets.QPushButton("Load to Parameters Tab")
        self.btn_send = QtWidgets.QPushButton("Send Now")
        h.addWidget(self.btn_recalc); h.addStretch(1)
        h.addWidget(self.btn_stage); h.addWidget(self.btn_send)
        left.addLayout(h)

        # Spectrum buttons
        hh = QtWidgets.QHBoxLayout()
        self.btn_load_spec = QtWidgets.QPushButton("Load Spectrum")
        self.btn_apply_spec = QtWidgets.QPushButton("Apply from Spectrum")
        self.chk_fit = QtWidgets.QCheckBox("Fit Lorentzian")
        self.chk_fit.setChecked(True)
        hh.addWidget(self.btn_load_spec); hh.addWidget(self.btn_apply_spec); hh.addWidget(self.chk_fit)
        left.addLayout(hh)

        # Results box
        self.results_box = QtWidgets.QLabel()
        self.results_box.setStyleSheet("background:#f9f9f9; border:1px solid #ccc; padding:4px;")
        left.addWidget(self.results_box)

        # ---------------- Right column ----------------
        right = QtWidgets.QVBoxLayout()
        self.plot_amp = pg.PlotWidget(title="Amplitude")
        self.plot_phase = pg.PlotWidget(title="Phase")
        for p in (self.plot_amp, self.plot_phase):
            p.setBackground("w")
            for ax in ("left", "bottom"):
                p.getAxis(ax).setPen("k")
                p.getAxis(ax).setTextPen("k")
        right.addWidget(self.plot_amp, stretch=1)
        right.addWidget(self.plot_phase, stretch=1)

        # Combine
        main_layout.addLayout(left, stretch=0)
        main_layout.addLayout(right, stretch=1)

        # Wiring
        self.btn_recalc.clicked.connect(self._recalc)
        for w in (self.q_val, self.f0_val, self.bw_pll):
            w.valueChanged.connect(self._recalc)
        self.btn_stage.clicked.connect(self._stage)
        self.btn_send.clicked.connect(self._send)
        self.btn_load_spec.clicked.connect(self._load_spectrum)
        self.btn_apply_spec.clicked.connect(self._apply_from_spectrum)
        self.chk_fit.stateChanged.connect(self._reload_fit_visibility)

        # Initial compute
        self._recalc()

    # ------------------------------------------------------------------
    def _recalc(self):
        """
        Recalculate recommended parameters based on current input values.

        Inputs
        ------
        Q : float
            Quality factor.
        f0 : float
            Resonance frequency (Hz).
        BW_PLL : float
            PLL bandwidth (Hz).

        Outputs
        -------
        Updates Ki, Kp, amplitude tau, and PLL time constant fields.
        """
        Q = float(self.q_val.value()); f0 = float(self.f0_val.value()); BW_PLL = float(self.bw_pll.value())
        Ki = 5e8 / max(Q, 1e-9); Kp = 1e4 * Ki; BW_amp = 10 * Q / f0
        tau_amp = 1.0 / BW_amp * 1000; tau_pll = 1.0 / (10.0 * max(BW_PLL, 1e-9)) * 1000
        self.ki_out.setText(f"{Ki:.6g}"); self.kp_out.setText(f"{Kp:.6g}")
        self.tau_amp.setText(f"{tau_amp:.6g}"); self.tpll_out.setText(f"{tau_pll:.6g}")

    def _stage(self):
        """
        Stage the calculated Ki and Kp values to the parameters tab.

        Inputs
        ------
        Ki : float
            Calculated amplitude integral gain.
        Kp : float
            Calculated amplitude proportional gain.

        Outputs
        -------
        Stages values to Edit24 and Edit32 in the parameters tab.
        Shows a message indicating which parameters were loaded.
        """
        try: Ki = float(self.ki_out.text()); Kp = float(self.kp_out.text())
        except Exception: 
            QtWidgets.QMessageBox.warning(self, "Numbers", "Invalid Ki/Kp."); return
        ok1 = self.params_tab.stage_value("EDIT", "Edit24", Ki)
        ok2 = self.params_tab.stage_value("EDIT", "Edit32", Kp)
        msg = []; 
        if ok1: msg.append("Edit24 ← Ki")
        if ok2: msg.append("Edit32 ← Kp")
        QtWidgets.QMessageBox.information(self, "Loaded", " / ".join(msg) if msg else "Nothing loaded.")

    def _send(self):
        """
        Send the calculated Ki and Kp values directly to the SXM system.

        Inputs
        ------
        Ki : float
            Calculated amplitude integral gain.
        Kp : float
            Calculated amplitude proportional gain.

        Outputs
        -------
        Sends values to SXM system.
        Shows a message indicating the result of the operation.
        """

        try: 
            Ki = float(self.ki_out.text()); Kp = float(self.kp_out.text())
            self.dde.send_scanpara("Edit24", Ki); self.dde.send_scanpara("Edit32", Kp)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "DDE error", f"Failed to send: {e}"); return
        QtWidgets.QMessageBox.information(self, "Sent", "Amplitude Ki (Edit24) and Kp (Edit32) sent.")

    # ------------------------------------------------------------------
    def _load_spectrum(self):
        """
        Load a resonance spectrum from a file and fit a Lorentzian model if enabled.

        Inputs
        ------
        File with three columns: frequency, phase, amplitude.

        Outputs
        -------
        Displays fit results and plots amplitude and phase spectra.
        Updates internal fit parameters for use in calculations.
        """

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Spectrum", "", "Data Files (*.txt *.csv);;All Files (*)")
        if not path: return

        try:
            data = np.genfromtxt(path, comments="#", skip_header=1, delimiter=None, dtype=float)
            if data.ndim == 1 and data.size >= 3: data = data.reshape(1, -1)
            data = data[~np.isnan(data).any(axis=1)]
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not read file:\n{e}"); return

        if data.shape[1] != 3:
            QtWidgets.QMessageBox.warning(self, "Error", f"Spectrum file must have 3 columns (freq | phase | amplitude). Got {data.shape[1]}."); return

        freq, phase, amp = data[:, 0], data[:, 1], data[:, 2]
        sort_idx = np.argsort(freq); freq, phase, amp = freq[sort_idx], phase[sort_idx], amp[sort_idx]
        
        

        # Estimate from amplitude
        A0_guess, f0_guess = amp.max(), freq[np.argmax(amp)]
        half_max = A0_guess / np.sqrt(2)
        mask = amp > half_max
        if mask.any():
            f_lo, f_hi = freq[mask][0], freq[mask][-1]
            Q_guess = f0_guess / (f_hi - f_lo) if f_hi > f_lo else 1000
        else: Q_guess = 1000

        # Fit?
        A0_fit, f0_fit, Q_fit = A0_guess, f0_guess, Q_guess
        perr = [np.nan, np.nan, np.nan]; fit_ok = False
        if self.chk_fit.isChecked():
            try:
                popt, pcov = curve_fit(lorentzian_amp, freq, amp, p0=[A0_guess, f0_guess, Q_guess], maxfev=10000)
                A0_fit, f0_fit, Q_fit = popt
                perr = np.sqrt(np.diag(pcov))  # 1-sigma errors
                fit_ok = True
            except Exception as e: print("Fit failed:", e)

        phase_at_f0 = phase[np.argmin(np.abs(freq - f0_fit))]
        self._last_fit = (Q_fit, f0_fit, phase_at_f0)

        # ---- Plot ----
        self.plot_amp.clear(); self.plot_amp.plot(freq, amp, pen=pg.mkPen("k", width=3))
        self.plot_amp.addLine(x=f0_fit, pen=pg.mkPen("gray", style=QtCore.Qt.DashLine))
        if fit_ok: 
            f_fit = np.linspace(freq.min(), freq.max(), 1000)
            self.plot_amp.plot(f_fit, lorentzian_amp(f_fit, A0_fit, f0_fit, Q_fit), pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine))
        self.plot_phase.clear(); self.plot_phase.plot(freq, phase, pen=pg.mkPen("b", width=3))
        self.plot_phase.addLine(x=f0_fit, pen=pg.mkPen("gray", style=QtCore.Qt.DashLine))

        # ---- Results ----
        if fit_ok:
            self.results_box.setText(
                f"Results from Lorentzian fit:\n"
                f"  f₀   = {f0_fit:.2f} ± {perr[1]:.2f} Hz\n"
                f"  Q    = {Q_fit:.1f} ± {perr[2]:.1f}\n"
                f"  A₀   = {A0_fit:.3g} ± {perr[0]:.3g}\n"
                f"  Phase(f₀) = {phase_at_f0:.1f}°"
            )
        else:
            self.results_box.setText(
                f"Results from FWHM estimate:\n"
                f"  f₀   = {f0_fit:.2f} Hz\n"
                f"  Q    = {Q_fit:.1f}\n"
                f"  A₀   = {A0_fit:.3g}\n"
                f"  Phase(f₀) = {phase_at_f0:.1f}°"
            )

    def _reload_fit_visibility(self):
        """
        Reload the spectrum plot with or without Lorentzian fitting,
        depending on the checkbox state.

        Inputs
        ------
        Checkbox state for fitting.

        Outputs
        -------
        Updates spectrum plot.
        """
        """Reload spectrum with or without fit depending on checkbox."""
        self._load_spectrum()

    def _apply_from_spectrum(self):
        """
        Apply fitted Q and f₀ values from the loaded spectrum to the input fields.

        Inputs
        ------
        Internal fit results (Q, f₀).

        Outputs
        -------
        Updates Q factor and resonance frequency fields.
        Recalculates recommended parameters.
        """
        if not self._last_fit:
            QtWidgets.QMessageBox.information(self, "No fit", "Please load and fit a spectrum first."); return
        Q_fit, f0_fit, _ = self._last_fit
        self.q_val.setValue(Q_fit); self.f0_val.setValue(f0_fit)
        self._recalc()
