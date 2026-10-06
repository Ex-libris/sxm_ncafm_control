"""
Tuning tab: run sheets of step tests, the way the loops are tuned by hand.

What the user does
------------------
1. Pick the loop and read SXM. The **anchor** is a known-good (Kp0, Ki0), by default the gains SXM has now.
2. Build a run sheet from **ramps**: G (common scale of both gains, sets the speed) at a fixed rho, rho (Ki:Kp
   ratio factor, sets the shape) at a fixed G, or one setting (AFL Tau, DNC TimeConstant / RollOff, output gain,
   input gain, Ref) at fixed G and rho. Kp = G Kp0, Ki = G rho Ki0, so the orders of magnitude between Kp and Ki
   never change by accident.
3. Run. Every condition: settings (by hand when asked, or automatically), gains, settle, step train, quiet
   window. A G or rho ramp stops at its first condition that rings, loses the loop or keeps Drive at zero.
4. Click a row to see its step response, its quiet-window noise and its recording; *Ramp trends* plots speed and
   noise against the ramped value. The best condition is the cleanest one with the lowest df noise per pixel.

The tests themselves: gui/sheet_runner.py; the sheet logic: tuning/runsheet.py; noise: tuning/noise.py.
"""
from __future__ import annotations

import csv
import dataclasses
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from sxm_ncafm_control import metadata as MD
from sxm_ncafm_control.tuning import explore as X
from sxm_ncafm_control.tuning import runsheet as R
from sxm_ncafm_control.tuning import workflow as W

from ..common import append_log_line, format_number
from . import condition_export as CX
from .sci_spinbox import SciDoubleSpinBox
from .sheet_runner import LOOP_CHANNELS, Outcome, RunConfig, SheetRunner

STATUS_COLOR = {
    "clean": (70, 175, 95), "slow": (120, 170, 230), "overshoot": (240, 200, 60), "floor": (240, 140, 50),
    "ringing": (220, 90, 50), "lost": (205, 65, 65), "skipped": (150, 150, 150), "stopped": (150, 150, 150),
    "pending": (230, 230, 230), "running": (180, 210, 255),
}
STATUS_TEXT = {**X.STATUS_LABEL, "skipped": "skipped", "stopped": "stopped", "pending": "", "running": "running…"}
LOOP_NAME = {"afl": "Amplitude loop (QPlusAmpl / Drive)", "pll": "PLL (df / Phase)"}
# per loop: step, hold, events, settle, quiet, quiet settle, settle timeout, steady window
TRAIN_DEFAULTS = {
    "afl": dict(step=5.0, hold=4.0, events=8, settle=8.0, quiet=30.0, quiet_settle=5.0, timeout=60.0, window=1.0),
    "pll": dict(step=1.0, hold=1.5, events=8, settle=2.0, quiet=20.0, quiet_settle=1.5, timeout=20.0, window=0.5),
}
PIXEL_CHOICES = (0.01, 0.03, 0.1, 0.3)
COLUMNS = ("#", "Ramp", "G", "ρ", "Settings", "Kp", "Ki", "Result", "Up 10-90", "Down 10-90", "Overshoot",
           "Drive σ / Phase pk", "Amp noise", "df / pixel", "Notes")
SERIES = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]     # categorical slots 1-4


def _fmt(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    return format_number(float(f"{v:.4g}"))


def _gain(v) -> str:
    """Gains span decades: scientific from 1e4 up (5e5, -1.65e4), plain below (-200)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    return format_number(float(f"{v:.4g}"), sci_above=1e4)


def _copy_meta(meta: Optional[MD.Metadata]) -> MD.Metadata:
    """The SXM values of ``meta`` without its app sections, so adding a section never changes the original."""
    if meta is None:
        return MD.Metadata.unavailable("not read")
    return MD.Metadata(meta.values, meta.errors, meta.timestamp, meta.source)


def _ms(v) -> str:
    return "" if v is None or not math.isfinite(v) else f"{v * 1e3:.0f} ms"


class RunSheetTab(QtWidgets.QWidget):
    def __init__(self, dde, driver, reader=None, params_tab=None, parent=None):
        super().__init__(parent)
        self.dde, self.driver, self.reader, self.params_tab = dde, driver, reader, params_tab
        self.sxm: Dict[str, Any] = {}
        self.sxm_time: Optional[str] = None
        self.sheet: List[R.Condition] = []
        self.outcomes: Dict[int, Outcome] = {}
        self.runner: Optional[SheetRunner] = None
        self._writer = None
        self._last_dir = ""
        self._raw_dir = ""
        self._best: Optional[int] = None
        self._build()
        self._on_loop_changed()
        self.refresh_from_sxm(quiet=True)

    # ================================================================== construction
    def _spin(self, lo, hi, val, dec=2, step=None, suffix="", sci=False):
        s = SciDoubleSpinBox(plain=not sci, sci_above=1e4)
        s.setDecimals(dec if not sci else 6)
        s.setRange(lo, hi)
        s.setValue(val)
        if step:
            s.setSingleStep(step)
        if suffix:
            s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        return s

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)

        # ---------------------------------------------------------------- left: set-up
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(2, 2, 2, 2)

        g = QtWidgets.QGroupBox("1. Loop and instrument")
        f = QtWidgets.QFormLayout(g)
        self.loop_combo = QtWidgets.QComboBox()
        for key in ("afl", "pll"):
            self.loop_combo.addItem(LOOP_NAME[key], key)
        self.loop_combo.currentIndexChanged.connect(self._on_loop_changed)
        f.addRow("Loop:", self.loop_combo)
        self.btn_read = QtWidgets.QPushButton("Read SXM")
        self.btn_read.clicked.connect(lambda: self.refresh_from_sxm())
        f.addRow(self.btn_read)
        self.instrument_label = QtWidgets.QLabel()
        self.instrument_label.setWordWrap(True)
        self.instrument_label.setTextFormat(QtCore.Qt.RichText)
        f.addRow(self.instrument_label)
        lv.addWidget(g)

        g = QtWidgets.QGroupBox("2. Anchor (Kp₀, Ki₀)")
        f = QtWidgets.QFormLayout(g)
        self.kp0 = self._spin(-1e12, 1e12, 1e6, sci=True)
        self.ki0 = self._spin(-1e12, 1e12, 100.0, sci=True)
        for s in (self.kp0, self.ki0):
            s.valueChanged.connect(self._refresh_panels)
        f.addRow("Kp₀:", self.kp0)
        f.addRow("Ki₀:", self.ki0)
        self.btn_anchor_sxm = QtWidgets.QPushButton("Anchor = gains in SXM")
        self.btn_anchor_sxm.clicked.connect(self._anchor_from_sxm)
        f.addRow(self.btn_anchor_sxm)
        self.anchor_og = QtWidgets.QComboBox()
        for v in W.AFL_OUTPUT_GAINS:
            self.anchor_og.addItem(f"±{v:g} V", v)
        self.anchor_ina = QtWidgets.QComboBox()
        for v in W.INPUT_GAINS:
            self.anchor_ina.addItem(f"x{v:g}", v)
        f.addRow("Found at output gain:", self.anchor_og)
        f.addRow("… and input gain:", self.anchor_ina)
        self.compensate = QtWidgets.QCheckBox("Rescale gains when output / input gain changes")
        self.compensate.setChecked(True)
        self.compensate.setToolTip("Amplitude loop: gains x1/output gain and x1/input gain relative to the anchor, "
                                   "so the loop gain stays the same (manual: x10 from ±1 V to ±0.1 V).")
        f.addRow(self.compensate)
        self.anchor_label = QtWidgets.QLabel()
        f.addRow(self.anchor_label)
        lv.addWidget(g)

        g = QtWidgets.QGroupBox("3. Each condition")
        f = QtWidgets.QFormLayout(g)
        self.step_spin = self._spin(0.01, 50.0, 5.0, dec=2)
        self.step_label = QtWidgets.QLabel("Step ±:")
        f.addRow(self.step_label, self.step_spin)
        self.hold_spin = self._spin(0.2, 10.0, 4.0, dec=1, suffix=" s")
        f.addRow("Hold per level:", self.hold_spin)
        self.events_spin = QtWidgets.QSpinBox()
        self.events_spin.setRange(4, 40)
        self.events_spin.setSingleStep(2)
        f.addRow("Steps:", self.events_spin)
        self.settle_spin = self._spin(0.5, 120.0, 8.0, dec=1, suffix=" s")
        f.addRow("Settle before steps:", self.settle_spin)
        self.quiet_spin = self._spin(0.0, 600.0, 30.0, dec=0, suffix=" s")
        self.quiet_spin.setToolTip("Recorded at the base value with nothing stepped: the noise of the condition. "
                                   "0 = no quiet window.")
        f.addRow("Quiet window:", self.quiet_spin)
        self.quiet_settle_spin = self._spin(0.0, 60.0, 5.0, dec=1, suffix=" s")
        f.addRow("… after settling for:", self.quiet_settle_spin)
        self.timeout_spin = self._spin(5.0, 600.0, 60.0, dec=0, suffix=" s")
        f.addRow("Settle timeout:", self.timeout_spin)
        self.both_check = QtWidgets.QCheckBox("Record both loops (QPlusAmpl, Drive, df, Phase)")
        self.both_check.setChecked(True)
        f.addRow(self.both_check)
        self.pixel_combo = QtWidgets.QComboBox()
        for p in PIXEL_CHOICES:
            self.pixel_combo.addItem(f"{p * 1e3:g} ms", p)
        self.pixel_combo.setCurrentIndex(2)
        self.pixel_combo.setToolTip("Pixel dwell time for 'df / pixel': the df noise after averaging over one pixel. "
                                    "The best condition is the clean one with the lowest value.")
        self.pixel_combo.currentIndexChanged.connect(self._refresh_table)
        f.addRow("Rank by df noise per pixel of:", self.pixel_combo)
        for w in (self.step_spin, self.hold_spin, self.events_spin, self.settle_spin, self.quiet_spin,
                  self.quiet_settle_spin):
            w.valueChanged.connect(self._refresh_ramp_preview)
        lv.addWidget(g)

        g = QtWidgets.QGroupBox("4. Add a ramp to the sheet")
        f = QtWidgets.QFormLayout(g)
        self.ramp_combo = QtWidgets.QComboBox()
        for key in R.RAMPABLE:
            self.ramp_combo.addItem(R.RAMP_LABEL[key], key)
        self.ramp_combo.currentIndexChanged.connect(self._on_ramp_param)
        f.addRow("Ramp:", self.ramp_combo)
        self.values_edit = QtWidgets.QLineEdit()
        self.values_edit.textChanged.connect(self._refresh_ramp_preview)
        f.addRow("Values:", self.values_edit)
        self.fixed_g = self._spin(0.001, 1000.0, 1.0, dec=3)
        self.fixed_rho = self._spin(0.001, 1000.0, 1.0, dec=3)
        f.addRow("at G:", self.fixed_g)
        f.addRow("at ρ:", self.fixed_rho)
        self.ref_every = QtWidgets.QSpinBox()
        self.ref_every.setRange(0, 20)
        self.ref_every.setValue(0)
        self.ref_every.setSpecialValueText("never")
        self.ref_every.setToolTip("Insert the anchor (G = ρ = 1) at the start and after every N conditions, "
                                  "to see drift.")
        f.addRow("Anchor repeat every:", self.ref_every)
        self.ramp_preview = QtWidgets.QLabel()
        self.ramp_preview.setWordWrap(True)
        f.addRow(self.ramp_preview)
        self.btn_add = QtWidgets.QPushButton("Add to sheet")
        self.btn_add.clicked.connect(self.add_ramp)
        f.addRow(self.btn_add)
        lv.addWidget(g)

        g = QtWidgets.QGroupBox("5. Run")
        v = QtWidgets.QVBoxLayout(g)
        self.retracted_check = QtWidgets.QCheckBox("Tip retracted / far from the surface")
        self.retracted_check.toggled.connect(self._update_enabled)
        v.addWidget(self.retracted_check)
        self.auto_check = QtWidgets.QCheckBox("Set Tau / TimeConstant / RollOff / gains in SXM automatically "
                                              "(experimental)")
        self.auto_check.setToolTip("Uses AnfatecSXMWriter, not yet verified on the instrument. Off: the run pauses "
                                   "and asks you to set them, then checks SXM's read-back.")
        v.addWidget(self.auto_check)
        row = QtWidgets.QHBoxLayout()
        self.raw_check = QtWidgets.QCheckBox("Save every condition (CSV + JSON + PNG, as a Step Test) to:")
        self.raw_check.setToolTip("Each condition is saved when it ends, named and laid out like a Step Test export "
                                  "from the Scope tab; the results table is saved there too at the end of the run. "
                                  "Run asks for the folder if none is set.")
        self.raw_check.setChecked(True)
        self.raw_check.toggled.connect(self._update_enabled)
        self.btn_raw_dir = QtWidgets.QPushButton("Folder…")
        self.btn_raw_dir.clicked.connect(self._choose_raw_dir)
        row.addWidget(self.raw_check)
        row.addWidget(self.btn_raw_dir)
        v.addLayout(row)
        self.raw_label = QtWidgets.QLabel("<i>no folder</i>")
        self.raw_label.setWordWrap(True)
        v.addWidget(self.raw_label)
        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run sheet")
        self.btn_run.clicked.connect(self.run)
        self.btn_continue = QtWidgets.QPushButton("Continue")
        self.btn_continue.clicked.connect(self._continue)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop)
        for b in (self.btn_run, self.btn_continue, self.btn_stop):
            row.addWidget(b)
        v.addLayout(row)
        lv.addWidget(g)
        lv.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        split.addWidget(scroll)

        # ---------------------------------------------------------------- right: sheet, plots, log
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        self.operator_banner = QtWidgets.QLabel()
        self.operator_banner.setWordWrap(True)
        self.operator_banner.setStyleSheet("QLabel{background:#fff3c4;color:#3d2e00;border:1px solid #d9b84a;"
                                           "padding:6px;font-weight:bold}")
        self.operator_banner.hide()
        tv.addWidget(self.operator_banner)
        bar = QtWidgets.QHBoxLayout()
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QtWidgets.QPushButton("Clear sheet")
        self.btn_clear.clicked.connect(self._clear_sheet)
        self.btn_load = QtWidgets.QPushButton("Load sheet…")
        self.btn_load.clicked.connect(self._load_sheet)
        self.btn_save = QtWidgets.QPushButton("Save sheet…")
        self.btn_save.clicked.connect(self._save_sheet)
        self.btn_export = QtWidgets.QPushButton("Export results…")
        self.btn_export.clicked.connect(self._export_results)
        self.btn_stage = QtWidgets.QPushButton("Stage gains in Parameters")
        self.btn_stage.clicked.connect(self._stage_selected)
        for b in (self.btn_remove, self.btn_clear, self.btn_load, self.btn_save, self.btn_export, self.btn_stage):
            bar.addWidget(b)
        bar.addStretch(1)
        tv.addLayout(bar)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        tv.addWidget(self.table, 1)
        right.addWidget(top)

        self.plot_tabs = QtWidgets.QTabWidget()
        cond = pg.GraphicsLayoutWidget()
        self.p_step = cond.addPlot(row=0, col=0, title="Step response (averaged)")
        self.p_second = cond.addPlot(row=0, col=1, title="Controller output during the steps")
        self.p_asd = cond.addPlot(row=1, col=0, title="Quiet window: noise spectrum")
        self.p_rec = cond.addPlot(row=1, col=1, title="Recording")
        for p in (self.p_step, self.p_second, self.p_rec):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.addLegend(offset=(5, 5))
        self.p_asd.showGrid(x=True, y=True, alpha=0.25)
        self.p_asd.setLogMode(x=True, y=True)
        self.p_asd.addLegend(offset=(5, 5))
        self.plot_tabs.addTab(cond, "Selected condition")
        trend = pg.GraphicsLayoutWidget()
        self.p_speed = trend.addPlot(row=0, col=0, title="Speed: 10-90 % rise")
        self.p_noise = trend.addPlot(row=0, col=1, title="Noise in the quiet window")
        self.p_drive = trend.addPlot(row=1, col=0, title="Controller output noise")
        self.p_amp = trend.addPlot(row=1, col=1, title="Amplitude noise")
        for p in (self.p_speed, self.p_noise, self.p_drive, self.p_amp):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.addLegend(offset=(5, 5))
        self.plot_tabs.addTab(trend, "Ramp trends")
        right.addWidget(self.plot_tabs)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log)
        right.setSizes([330, 420, 110])
        split.addWidget(right)
        split.setSizes([380, 1000])

    # ================================================================== accessors
    @property
    def loop(self) -> str:
        return self.loop_combo.currentData()

    @property
    def loop_def(self) -> W.LoopDef:
        return W.LOOPS[self.loop]

    def anchor(self) -> R.Anchor:
        return R.Anchor(self.loop, self.kp0.value(), self.ki0.value(), self.anchor_og.currentData(),
                        self.anchor_ina.currentData())

    def current_settings(self) -> Dict[str, Any]:
        return {k: R.sxm_value(k, self.sxm.get(s.sxm_key)) for k, s in R.SETTINGS.items()}

    def plan(self) -> W.StepTestPlan:
        ld = self.loop_def
        step = self.step_spin.value() / 100.0 if ld.relative_step else self.step_spin.value()
        base = (self.sxm.get("amp_ref") or 1.0) if ld.key == "afl" else (self.sxm.get("used_freq") or 25000.0)
        lead = 2.0 if ld.key == "afl" else 1.0
        return W.StepTestPlan(loop=ld.key, base=float(base), step=step, hold_s=self.hold_spin.value(),
                              n_events=self.events_spin.value(), settle_s=self.settle_spin.value(),
                              lead_s=lead, tail_s=0.5)

    def ring_down_s(self) -> Optional[float]:
        f0, q = self.sxm.get("f_peak"), self.sxm.get("q")
        return q / (math.pi * f0) if isinstance(f0, float) and isinstance(q, float) and f0 > 0 and q > 0 else None

    def condition_seconds(self) -> float:
        try:
            p = self.plan()
        except ValueError:
            return math.nan
        q = self.quiet_spin.value()
        return p.settle_s + p.duration + (self.quiet_settle_s() + q if q > 0 else 0.0) + 2.0

    def quiet_settle_s(self) -> float:
        return self.quiet_settle_spin.value()

    # ================================================================== SXM
    def refresh_from_sxm(self, quiet: bool = False) -> bool:
        ro = None
        if self.reader is not None:
            try:
                ro = self.reader.read()
            except Exception as e:
                self._log(f"Reading SXM failed: {e}")
        if ro is None or not getattr(ro, "ok", False):
            if not quiet:
                self._log("SXM could not be read (is it running with its windows open?).")
            self._refresh_panels()
            return False
        self.sxm = dict(ro.values)
        self.sxm_time = ro.timestamp.strftime("%H:%M:%S")
        if not self.sheet and not self.outcomes:
            self._anchor_from_sxm(quiet=True)
        self._log("Read SXM at " + self.sxm_time + ".")
        self._refresh_panels()
        return True

    def _anchor_from_sxm(self, quiet: bool = False):
        kp_key, ki_key = ("amp_kp", "amp_ki") if self.loop == "afl" else ("pll_kp", "pll_ki")
        kp, ki = self.sxm.get(kp_key), self.sxm.get(ki_key)
        if isinstance(kp, float) and isinstance(ki, float) and kp != 0 and ki != 0:
            self.kp0.setValue(kp)
            self.ki0.setValue(ki)
        elif not quiet:
            self._log("SXM has no usable gains for this loop (both must be non-zero): anchor unchanged.")
        for combo, key in ((self.anchor_og, "afl_output_gain"), (self.anchor_ina, "input_gain_ina")):
            i = combo.findData(self.sxm.get(key)) if self.sxm.get(key) is not None else -1
            if i >= 0:
                combo.setCurrentIndex(i)
        self._refresh_panels()

    def _writer_or_none(self):
        if not self.auto_check.isChecked():
            return None
        if self._writer is None:
            try:
                from sxm_ncafm_control.AnfatecSXMWriter import AnfatecSXMWriter
                self._writer = AnfatecSXMWriter()
            except Exception as e:
                self._log(f"Automatic settings unavailable ({e}): the run will ask you instead.")
                return None
        return self._writer

    # ================================================================== checks and panels
    def checks(self) -> List[tuple]:
        """``(level, text)``; 'block' disables Run."""
        out = []
        s, ld = self.sxm, self.loop_def
        if not self._online():
            out.append(("block", "Offline: running needs SXM and the driver. Sheets can still be built and saved."))
        out.append(("ok", f"SXM read at {self.sxm_time}.") if s else
                   ("warn", "SXM not read: setpoints and settings are unknown. Press Read SXM."))
        kp, ki = self.kp0.value(), self.ki0.value()
        if kp == 0 or ki == 0:
            out.append(("block", "The anchor needs Kp₀ and Ki₀ both non-zero."))
        elif any(math.copysign(1, v) != ld.gain_sign for v in (kp, ki)):
            out.append(("block", f"Gains of this loop are {'negative' if ld.gain_sign < 0 else 'positive'}."))
        if not self.sheet:
            out.append(("block", "The sheet is empty: add a ramp."))
        if s:
            if ld.key == "afl":
                pk, pi = s.get("pll_kp"), s.get("pll_ki")
                if pk == 0 and pi == 0:
                    out.append(("warn", "PLL is off: the quiet window cannot measure df noise. Tune in the "
                                        "configuration you image with (PLL on)."))
            else:
                ak, ai = s.get("amp_kp"), s.get("amp_ki")
                if ak == 0 and ai == 0:
                    out.append(("warn", "Amplitude loop is off (constant excitation)."))
            ina, og = s.get("input_gain_ina"), s.get("afl_output_gain")
            if ld.key == "afl" and isinstance(og, float) and og != self.anchor_og.currentData() and not self.compensate.isChecked():
                out.append(("warn", f"SXM output gain ±{og:g} V differs from the anchor's, and gains are not rescaled."))
            if ld.key == "afl" and isinstance(ina, float) and ina != self.anchor_ina.currentData() and not self.compensate.isChecked():
                out.append(("warn", f"SXM input gain x{ina:g} differs from the anchor's, and gains are not rescaled."))
        if not self.retracted_check.isChecked():
            out.append(("block", "Confirm that the tip is retracted."))
        if self.raw_check.isChecked() and not self._raw_dir:
            out.append(("warn", "No folder for the recordings yet: Run will ask for one."))
        elif not self.raw_check.isChecked():
            out.append(("warn", "Recordings are not saved (only the results table, on Export)."))
        return out

    def _refresh_panels(self, *_):
        s = self.sxm
        cur = self.current_settings()

        def val(k):
            return R.format_setting(k, cur[k]) if cur.get(k) is not None else "n/a"
        rows = [("AFL Kp / Ki", f"{_gain(s.get('amp_kp'))} / {_gain(s.get('amp_ki'))}"),
                ("PLL Kp / Ki", f"{_gain(s.get('pll_kp'))} / {_gain(s.get('pll_ki'))}"),
                ("Ref / use", f"{_fmt(s.get('amp_ref'))} / {_fmt(s.get('used_freq'))} Hz"),
                ("Tau / TC / RollOff", f"{val('amp_tau_ms')} / {val('dnc_tc_ms')} / {val('dnc_rolloff')}"),
                ("Output / input gain", f"{val('output_gain_v')} / {val('input_gain')}"),
                ("f₀ / Q", f"{_fmt(s.get('f_peak'))} Hz / {_fmt(s.get('q'))}")]
        icon = {"ok": "<span style='color:#2a8a3a'>✔</span>", "warn": "<span style='color:#b07000'>⚠</span>",
                "block": "<span style='color:#c03030'>✖</span>"}
        self.instrument_label.setText(
            "<table cellspacing='0' cellpadding='1'>" +
            "".join(f"<tr><td>{a}</td><td>&nbsp;<b>{b}</b></td></tr>" for a, b in rows) + "</table>" +
            "<br>".join(f"{icon[lvl]} {txt}" for lvl, txt in self.checks()))
        kp, ki = self.kp0.value(), self.ki0.value()
        if kp:
            self.anchor_label.setText(f"Ki₀/Kp₀ = {format_number(float(f'{ki / kp:.3g}'), sci_above=1e4)}  ·  "
                                      "Kp = G·Kp₀, Ki = G·ρ·Ki₀")
        self._refresh_ramp_preview()
        self._update_enabled()

    def _on_loop_changed(self, *_):
        d = TRAIN_DEFAULTS[self.loop]
        pll = self.loop == "pll"
        self.step_label.setText("Step ± (Hz):" if pll else "Step ± (% of Ref):")
        self.step_spin.setValue(d["step"])
        self.hold_spin.setValue(d["hold"])
        self.events_spin.setValue(d["events"])
        self.settle_spin.setValue(d["settle"])
        self.quiet_spin.setValue(d["quiet"])
        self.quiet_settle_spin.setValue(d["quiet_settle"])
        self.timeout_spin.setValue(d["timeout"])
        if pll:
            self.kp0.setValue(-200.0)
            self.ki0.setValue(-1.65e4)
        else:
            self.kp0.setValue(1e6)
            self.ki0.setValue(100.0)
        if self.sxm:
            self._anchor_from_sxm(quiet=True)
        if self.sheet and not self.runner_active():
            self._log("Loop changed: the sheet was cleared (its gains are relative to the other loop's anchor).")
            self._clear_sheet()
        self._on_ramp_param()

    def _on_ramp_param(self, *_):
        key = self.ramp_combo.currentData()
        vals = R.DEFAULT_VALUES[self.loop][key]
        self.values_edit.setText(", ".join(str(v) for v in vals))
        self.fixed_g.setEnabled(key != "g")
        self.fixed_rho.setEnabled(key != "rho")
        self._refresh_ramp_preview()

    def _ramp_conditions(self) -> List[R.Condition]:
        key = self.ramp_combo.currentData()
        vals = R.parse_values(key, self.values_edit.text())
        return R.ramp(key, vals, g=self.fixed_g.value(), rho=self.fixed_rho.value(),
                      reference_every=self.ref_every.value())

    def _refresh_ramp_preview(self, *_):
        try:
            conds = self._ramp_conditions()
        except ValueError as e:
            self.ramp_preview.setText(f"<span style='color:#c03030'>{e}</span>")
            self.btn_add.setEnabled(False)
            return
        each = self.condition_seconds()
        a = None
        try:
            a = self.anchor()
        except ValueError:
            pass
        first, last = conds[0], conds[-1]
        span = ""
        if a is not None:
            cur = self.current_settings()
            k1, i1 = R.gains(a, first, cur, self.compensate.isChecked())
            k2, i2 = R.gains(a, last, cur, self.compensate.isChecked())
            span = f"<br>Kp {_gain(k1)} … {_gain(k2)}, Ki {_gain(i1)} … {_gain(i2)}"
        total = len(self.sheet) + len(conds)
        self.ramp_preview.setText(f"Adds {len(conds)} conditions (~{len(conds) * each / 60:.0f} min).{span}"
                                  f"<br>Sheet would have {total} (~{total * each / 60:.0f} min).")
        self.btn_add.setEnabled(not self.runner_active())

    # ================================================================== enabling, status, log
    def _online(self) -> bool:
        return self.driver is not None and not type(self.dde).__name__.startswith("Mock")

    def runner_active(self) -> bool:
        return self.runner is not None and self.runner.running

    def _blocked(self) -> Optional[str]:
        b = [t for lvl, t in self.checks() if lvl == "block"]
        return b[0] if b else None

    def _update_enabled(self, *_):
        running = self.runner_active()
        why = self._blocked()
        self.btn_run.setEnabled(not running and why is None)
        self.btn_run.setToolTip(why or "")
        self.btn_stop.setEnabled(running)
        self.btn_continue.setEnabled(running and self.runner.waiting_operator)
        app = QtWidgets.QApplication.instance()
        mgr = getattr(app, "accessibility_manager", None)
        self.btn_stop.setStyleSheet(mgr.stop_button_style(running) if mgr else "")
        for b in (self.btn_add, self.btn_remove, self.btn_clear, self.btn_load, self.loop_combo):
            b.setEnabled(not running)
        self.btn_save.setEnabled(bool(self.sheet))
        self.btn_export.setEnabled(bool(self.outcomes) and not running)
        self.btn_stage.setEnabled(self.params_tab is not None and self._selected_outcome() is not None and not running)
        if not running:
            if why:
                self._status(f"<b>Not ready:</b> {why}")
            else:
                n = len(self.sheet)
                self._status(f"<b>Ready:</b> {n} conditions, ~{n * self.condition_seconds() / 60:.0f} min.")

    def _status(self, html: str):
        conn = "ONLINE" if self._online() else "OFFLINE"
        self.status_label.setText(f"{LOOP_NAME[self.loop]} &middot; {conn} &middot; {html}")

    def _log(self, text: str):
        append_log_line(self.log, f"[{time.strftime('%H:%M:%S')}] {text}")

    # ================================================================== sheet editing
    def add_ramp(self):
        try:
            conds = self._ramp_conditions()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Add ramp", str(e))
            return
        self.sheet.extend(conds)
        self._log(f"Added {len(conds)} conditions: {conds[-1].group}.")
        self._refresh_table()
        self._refresh_panels()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            del self.sheet[r]
        self.outcomes = {}
        self._refresh_table()
        self._refresh_panels()

    def _clear_sheet(self):
        self.sheet = []
        self.outcomes = {}
        self._best = None
        self._refresh_table()
        self._refresh_panels()
        self._plot_selected()

    def _save_sheet(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save run sheet", os.path.join(
            self._last_dir, f"{time.strftime('%Y%m%d-%H%M%S')}_runsheet_{self.loop.upper()}.json"), "JSON (*.json)")
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        try:
            R.save_sheet(path, self.anchor(), self.sheet, {"train": self._train_dict()})
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Save sheet", str(e))
            return
        self._log(f"Saved the sheet to {path}.")

    def _load_sheet(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load run sheet", self._last_dir, "JSON (*.json)")
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        try:
            anchor, sheet, extra = R.load_sheet(path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Load sheet", f"Could not load:\n{e}")
            return
        self.loop_combo.setCurrentIndex(self.loop_combo.findData(anchor.loop))
        self.kp0.setValue(anchor.kp)
        self.ki0.setValue(anchor.ki)
        self.anchor_og.setCurrentIndex(max(0, self.anchor_og.findData(anchor.output_gain_v)))
        self.anchor_ina.setCurrentIndex(max(0, self.anchor_ina.findData(anchor.input_gain)))
        self._apply_train_dict(extra.get("train") or {})
        self.sheet, self.outcomes, self._best = sheet, {}, None
        self._log(f"Loaded {len(sheet)} conditions from {path}.")
        self._refresh_table()
        self._refresh_panels()

    def _train_dict(self) -> Dict[str, Any]:
        return {"step": self.step_spin.value(), "hold_s": self.hold_spin.value(), "events": self.events_spin.value(),
                "settle_s": self.settle_spin.value(), "quiet_s": self.quiet_spin.value(),
                "quiet_settle_s": self.quiet_settle_spin.value(), "settle_timeout_s": self.timeout_spin.value(),
                "record_both": self.both_check.isChecked(), "compensate": self.compensate.isChecked()}

    def _apply_train_dict(self, d: Dict[str, Any]):
        for key, w in (("step", self.step_spin), ("hold_s", self.hold_spin), ("events", self.events_spin),
                       ("settle_s", self.settle_spin), ("quiet_s", self.quiet_spin),
                       ("quiet_settle_s", self.quiet_settle_spin), ("settle_timeout_s", self.timeout_spin)):
            if key in d:
                w.setValue(d[key])
        if "record_both" in d:
            self.both_check.setChecked(bool(d["record_both"]))
        if "compensate" in d:
            self.compensate.setChecked(bool(d["compensate"]))

    def _choose_raw_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the recordings", self._raw_dir or self._last_dir)
        if d:
            self._raw_dir = d
            self.raw_label.setText(d)
            self.raw_check.setChecked(True)
        self._refresh_panels()

    # ================================================================== running
    def run(self):
        why = self._blocked()
        if why:
            QtWidgets.QMessageBox.warning(self, "Run sheet", why)
            return
        if self.raw_check.isChecked() and not self._raw_dir:
            self._choose_raw_dir()
            if not self._raw_dir:
                return
        try:
            plan = self.plan()
            anchor = self.anchor()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Run sheet", str(e))
            return
        ld = self.loop_def
        other_on = True
        if self.sxm:
            ok, ik = ("pll_kp", "pll_ki") if ld.key == "afl" else ("amp_kp", "amp_ki")
            other_on = not (self.sxm.get(ok) == 0 and self.sxm.get(ik) == 0)
        crit = X.RecoveryCriteria(window_s=TRAIN_DEFAULTS[ld.key]["window"])
        cfg = RunConfig(anchor=anchor, sheet=list(self.sheet), plan=plan, start_settings=self.current_settings(),
                        base_use=float(self.sxm.get("used_freq") or plan.base),
                        quiet_s=self.quiet_spin.value(), quiet_settle_s=self.quiet_settle_spin.value(),
                        settle_timeout_s=self.timeout_spin.value(),
                        recover_timeout_s=max(30.0, 10 * (self.ring_down_s() or 2.0)),
                        compensate=self.compensate.isChecked(), record_both=self.both_check.isChecked(),
                        other_loop_on=other_on, criteria=crit, ring_down_s=self.ring_down_s())
        self.outcomes = {}
        self._best = None
        self.runner = SheetRunner(self.dde, self.driver, cfg, reader=self.reader, writer=self._writer_or_none(),
                                  meta_fn=(lambda: MD.collect(self.reader)) if self.reader is not None else None,
                                  parent=self)
        self.runner.message.connect(self._log)
        self.runner.phase.connect(lambda t: self._status(f"<b>Running</b> {t}"))
        self.runner.condition_started.connect(self._on_started)
        self.runner.condition_finished.connect(self._on_finished_condition)
        self.runner.skipped.connect(self._on_skipped)
        self.runner.operator_needed.connect(self._on_operator)
        self.runner.finished.connect(self._on_run_finished)
        self._refresh_table()
        self._log(f"Run started: {len(self.sheet)} conditions, anchor Kp₀={_gain(anchor.kp)}, Ki₀={_gain(anchor.ki)}.")
        self.runner.start()
        self._update_enabled()

    def stop(self):
        if self.runner is not None:
            self.runner.stop()

    def _continue(self):
        self.operator_banner.hide()
        if self.runner is not None:
            self.runner.continue_()
        self._update_enabled()

    def _on_operator(self, text: str):
        self.operator_banner.setText(f"Set in SXM, then press Continue: {text}")
        self.operator_banner.show()
        self._log(f"Waiting: set {text}")
        self._update_enabled()

    def _on_started(self, i: int):
        self.operator_banner.hide()
        self._set_row_status(i, "running")
        self.table.selectRow(i)

    def _on_skipped(self, i: int, why: str):
        self.outcomes[i] = Outcome(i, self.sheet[i], status="skipped", note=why)
        self._fill_row(i)

    def _on_finished_condition(self, out: Outcome):
        if out.raw is not None and len(out.raw[0]) and self.raw_check.isChecked() and self._raw_dir:
            try:
                written = self._save_raw(out)
                out.note = (out.note + " | " if out.note else "") + f"saved {os.path.basename(written[0])}"
                for w in written[1:]:
                    if w.startswith("("):
                        self._log(f"  {w}")
            except OSError as e:
                self._log(f"  could not save the recording: {e}")
        out.raw = None                             # the display copy stays; the full one is on disk (or dropped)
        self.outcomes[out.index] = out
        a = out.assessment
        msg = f"  -> {STATUS_TEXT.get(out.status, out.status)}"
        if a is not None and out.status not in ("skipped", "stopped"):
            msg += f" (up {_ms(a.up.rise_s)}, down {_ms(a.down.rise_s)})"
        px = self._df_pixel(out)
        if math.isfinite(px):
            msg += f"; df {px * 1e3:.2f} mHz per {self.pixel_combo.currentData() * 1e3:g} ms pixel"
        self._log(msg)
        self._rank()
        self._fill_row(out.index)
        self._plot_selected()

    def _on_run_finished(self, reason: str):
        self.operator_banner.hide()
        self._log(f"Run {reason}.")
        if self._best is not None:
            o = self.outcomes[self._best]
            self._log(f"Best so far: #{self._best + 1} {o.condition.label()} (Kp {_gain(o.kp)}, Ki {_gain(o.ki)}).")
        if self.raw_check.isChecked() and self._raw_dir and any(o.status != "skipped" for o in self.outcomes.values()):
            try:
                path = os.path.join(self._raw_dir, self._results_name())
                jp = self.export_results_to(path)
                self._log(f"Results saved: {os.path.basename(path)} (+ {os.path.basename(jp)}).")
            except OSError as e:
                self._log(f"Could not save the results: {e}")
        self._update_enabled()
        self._refresh_table()

    def _save_raw(self, out: Outcome) -> List[str]:
        """The condition as a Step Test export (CSV + JSON + PNG) in the recordings folder. Returns the paths."""
        c = out.condition
        rows = [("Index", f"#{out.index + 1} of {len(self.sheet)}", ""), ("Ramp", c.group, ""),
                ("G", c.g, ""), ("rho", c.rho, ""), ("Kp written", out.kp, ""), ("Ki written", out.ki, ""),
                ("Settings", c.label(), ""), ("Result", STATUS_TEXT.get(out.status, out.status), "")]
        if out.note:
            rows.append(("Notes", out.note, ""))
        a = self.runner.cfg.anchor if self.runner is not None else self.anchor()
        rows += [("Anchor Kp0 / Ki0", f"{_gain(a.kp)} / {_gain(a.ki)}", "")]
        title = f"#{out.index + 1}  {c.label()}  (Kp {_gain(out.kp)}, Ki {_gain(out.ki)}): "                 f"{STATUS_TEXT.get(out.status, out.status)}"
        return CX.save_condition(self._raw_dir, out, self.loop, _copy_meta(out.meta), rows,
                                 offline=self.driver is None, title=title)

    # ================================================================== results
    def _df_pixel(self, out: Outcome) -> float:
        if out.quiet is None:
            return math.nan
        return out.quiet.get("df", "pixel", self.pixel_combo.currentData())

    def _rank(self):
        """Best = clean, lowest df noise per pixel (or the loop's own noise when df was not recorded)."""
        best, score = None, math.inf
        for i, o in self.outcomes.items():
            if o.status != "clean":
                continue
            s = self._df_pixel(o)
            if not math.isfinite(s) and o.assessment is not None:
                s = o.assessment.noise
            if math.isfinite(s) and s < score:
                best, score = i, s
        self._best = best

    def _cells(self, i: int) -> List[str]:
        c = self.sheet[i]
        o = self.outcomes.get(i)
        settings = ", ".join(f"{R.SETTINGS[k].label} {R.format_setting(k, v)}" for k, v in c.settings.items())
        if c.role == "reference":
            settings = "anchor (reference)" + (f", {settings}" if settings else "")
        try:
            kp, ki = R.gains(self.anchor(), c, self.current_settings(), self.compensate.isChecked())
        except ValueError:
            kp = ki = math.nan
        if o is not None and math.isfinite(o.kp):
            kp, ki = o.kp, o.ki
        cells = [str(i + 1) + (" ★" if i == self._best else ""), c.group, f"{c.g:g}", f"{c.rho:g}", settings,
                 _gain(kp), _gain(ki)]
        if o is None:
            return cells + [""] * (len(COLUMNS) - len(cells))
        a = o.assessment
        up = down = os_ = sec = amp = px = ""
        if a is not None and o.status not in ("skipped", "stopped"):
            up, down = _ms(a.up.rise_s), _ms(a.down.rise_s)
            ov = max((v for v in (a.up.overshoot, a.down.overshoot) if math.isfinite(v)), default=math.nan)
            os_ = "" if not math.isfinite(ov) else f"{ov * 100:.0f} %"
        q = o.quiet
        if q is not None:
            if self.loop == "afl":
                d = q.channels.get("Drive")
                sec = "" if d is None else f"{d.rms * 1e6:.2f} µV ({d.rel_rms * 100:.0f} %)" if math.isfinite(d.rel_rms) else f"{d.rms * 1e6:.2f} µV"
            amp_c = q.channels.get("QPlusAmpl")
            if amp_c is not None and math.isfinite(amp_c.rel_rms):
                amp = f"{amp_c.rel_rms * 100:.2f} %"
            v = self._df_pixel(o)
            px = "" if not math.isfinite(v) else f"{v * 1e3:.2f} mHz"
        if self.loop == "pll" and o.result is not None and o.result.error is not None:
            sec = f"{o.result.error.peak:.1f}°"
        return cells + [STATUS_TEXT.get(o.status, o.status), up, down, os_, sec, amp, px, o.note]

    def _fill_row(self, i: int):
        if i >= self.table.rowCount():
            return
        cells = self._cells(i)
        o = self.outcomes.get(i)
        status = o.status if o is not None else "pending"
        for col, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            if col == COLUMNS.index("Result"):
                item.setBackground(QtGui.QColor(*STATUS_COLOR.get(status, (255, 255, 255))))
                item.setForeground(QtGui.QColor(20, 20, 20))
            if col == len(COLUMNS) - 1:
                item.setToolTip(text)
            if i == self._best:
                fnt = item.font()
                fnt.setBold(True)
                item.setFont(fnt)
            self.table.setItem(i, col, item)

    def _set_row_status(self, i: int, status: str):
        item = QtWidgets.QTableWidgetItem(STATUS_TEXT.get(status, status))
        item.setBackground(QtGui.QColor(*STATUS_COLOR.get(status, (255, 255, 255))))
        item.setForeground(QtGui.QColor(20, 20, 20))
        self.table.setItem(i, COLUMNS.index("Result"), item)

    def _refresh_table(self, *_):
        sel = self._selected_row()
        self._rank()
        self.table.setRowCount(len(self.sheet))
        for i in range(len(self.sheet)):
            self._fill_row(i)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        if sel is not None and sel < len(self.sheet):
            self.table.selectRow(sel)

    def _selected_row(self) -> Optional[int]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else None

    def _selected_outcome(self) -> Optional[Outcome]:
        i = self._selected_row()
        o = self.outcomes.get(i) if i is not None else None
        return o if o is not None and math.isfinite(o.kp) else None

    def _on_select(self):
        self._plot_selected()
        self._update_enabled()

    def _stage_selected(self):
        o = self._selected_outcome()
        if o is None or self.params_tab is None:
            return
        ld = self.loop_def
        ok = (self.params_tab.stage_value(*ld.kp_param, o.kp) and self.params_tab.stage_value(*ld.ki_param, o.ki))
        self._log(f"Staged Kp={_gain(o.kp)}, Ki={_gain(o.ki)} in Parameters." if ok else
                  "These gains have no row in the Parameters tab.")

    # ================================================================== plots
    @staticmethod
    def _pen(k: int, width: float = 2.0, style=QtCore.Qt.SolidLine):
        return pg.mkPen(SERIES[k % len(SERIES)], width=width, style=style)

    def _plot_selected(self):
        for p in (self.p_step, self.p_second, self.p_asd, self.p_rec):
            p.clear()
        i = self._selected_row()
        o = self.outcomes.get(i) if i is not None else None
        self._plot_trends(i)
        if o is None:
            return
        res = o.result
        ld = self.loop_def
        if res is not None and res.grid is not None:
            g = res.grid
            prim = ld.primary_channel
            self.p_step.setTitle(f"{prim} step (averaged, fraction of the step)")
            curves = [(res.mean_primary_rising, "up-steps", 0), (res.mean_primary_falling, "down-steps", 1)]
            if all(c is None for c, _, _ in curves) and res.mean_primary is not None:
                curves = [(res.mean_primary, "all steps", 0)]
            step = abs(res.primary.step_size) if res.primary is not None and res.primary.step_size else 1.0
            for y, name, k in curves:
                if y is not None:
                    self.p_step.plot(g, (y - y[g < 0].mean() if (g < 0).any() else y) / step, pen=self._pen(k), name=name)
            self.p_step.setLabel("bottom", "time after the step", units="s")
            sec_name = "Drive" if ld.key == "afl" else "Phase"
            self.p_second.setTitle(f"{sec_name} during the steps")
            sec = [(res.mean_secondary_rising, "up-steps", 0), (res.mean_secondary_falling, "down-steps", 1)]
            if all(c is None for c, _, _ in sec) and res.mean_secondary is not None:
                sec = [(res.mean_secondary, "all steps (sign-folded)", 0)]
            for y, name, k in sec:
                if y is not None:
                    self.p_second.plot(g, y * (1e6 if ld.key == "afl" else 1.0), pen=self._pen(k), name=name)
            self.p_second.setLabel("left", "Drive change (µV)" if ld.key == "afl" else "Phase (°)")
            self.p_second.setLabel("bottom", "time after the step", units="s")
        q = o.quiet
        if q is not None:
            self.p_asd.setTitle("Quiet window: noise spectrum (relative / √Hz; df in mHz/√Hz)")
            k = 0
            for name in ("df", "QPlusAmpl", "Drive"):
                if name not in q.asd:
                    continue
                f, a = q.asd[name]
                c = q.channels[name]
                if name == "df":
                    y, label = a * 1e3, "df (mHz/√Hz)"
                elif math.isfinite(c.rel_rms) and abs(c.mean) > 0:
                    y, label = a / abs(c.mean), f"{name} (relative/√Hz)"
                else:
                    continue
                keep = (f > 0) & (y > 0)
                self.p_asd.plot(f[keep], y[keep], pen=self._pen(k, 1.5), name=label)
                k += 1
            self.p_asd.setLabel("bottom", "frequency", units="Hz")
        if o.t is not None and len(o.t):
            prim = ld.primary_channel
            if prim in o.data:
                self.p_rec.setTitle(f"Recording: {prim} (dashed: start of settle, steps, quiet window)")
                self.p_rec.plot(o.t, o.data[prim], pen=self._pen(0, 1.2), name=prim)
                for key in ("settle", "test", "quiet", "end"):
                    if key in o.marks:
                        self.p_rec.addLine(x=o.marks[key], pen=pg.mkPen((140, 140, 140), style=QtCore.Qt.DashLine))
                for e in o.events:
                    self.p_rec.addLine(x=e, pen=pg.mkPen((200, 200, 200), width=1))
                self.p_rec.setLabel("bottom", "time", units="s")

    def _plot_trends(self, i: Optional[int]):
        for p in (self.p_speed, self.p_noise, self.p_drive, self.p_amp):
            p.clear()
        if i is None or i >= len(self.sheet):
            return
        grp = self.sheet[i].group
        ramp = self.sheet[i].ramp
        idx = [j for j, c in enumerate(self.sheet) if c.group == grp and c.role == "test" and j in self.outcomes]
        if not idx:
            return

        def xval(c: R.Condition):
            if ramp == "g":
                return c.g
            if ramp == "rho":
                return c.rho
            v = c.settings.get(ramp)
            try:
                return float(v)
            except (TypeError, ValueError):
                return math.nan
        label = R.RAMP_LABEL.get(ramp, ramp)
        pts = sorted((xval(self.sheet[j]), self.outcomes[j]) for j in idx)
        pts = [(x, o) for x, o in pts if math.isfinite(x)]
        if not pts:
            return
        xs = np.array([x for x, _ in pts])
        logx = bool(ramp in ("g", "rho") and xs.min() > 0)
        for p in (self.p_speed, self.p_noise, self.p_drive, self.p_amp):
            p.setLogMode(x=logx, y=False)
            p.setLabel("bottom", label)

        def series(plot, vals, name, k, scale=1.0):
            v = np.array([np.nan if x is None else x for x in vals], float) * scale
            m = np.isfinite(v)
            if m.any():
                plot.plot(xs[m], v[m], pen=self._pen(k), symbol="o", symbolSize=7,
                          symbolBrush=SERIES[k % len(SERIES)], name=name)

        series(self.p_speed, [o.assessment.up.rise_s if o.assessment else None for _, o in pts], "up-steps", 0, 1e3)
        series(self.p_speed, [o.assessment.down.rise_s if o.assessment else None for _, o in pts], "down-steps", 1, 1e3)
        self.p_speed.setLabel("left", "10-90 % rise (ms)")
        px = self.pixel_combo.currentData()
        series(self.p_noise, [self._df_pixel(o) for _, o in pts], f"df per {px * 1e3:g} ms pixel", 0, 1e3)
        series(self.p_noise, [o.quiet.get("df", "rms") if o.quiet else None for _, o in pts], "df rms", 1, 1e3)
        self.p_noise.setLabel("left", "df noise (mHz)")
        if self.loop == "afl":
            series(self.p_drive, [o.quiet.get("Drive", "rms") if o.quiet else None for _, o in pts], "Drive σ", 0, 1e6)
            self.p_drive.setLabel("left", "Drive σ (µV)")
        else:
            series(self.p_drive, [o.quiet.get("Phase", "rms") if o.quiet else None for _, o in pts], "Phase σ", 0)
            self.p_drive.setLabel("left", "Phase σ (°)")
        series(self.p_amp, [o.quiet.get("QPlusAmpl", "rel_rms") if o.quiet else None for _, o in pts],
               "QPlusAmpl σ / mean", 0, 100)
        self.p_amp.setLabel("left", "amplitude noise (%)")

    # ================================================================== export
    def _export_rows(self) -> List[Dict[str, Any]]:
        rows = []
        px = self.pixel_combo.currentData()
        for i, c in enumerate(self.sheet):
            o = self.outcomes.get(i)
            if o is None:
                continue
            a, q = o.assessment, o.quiet

            def qv(ch, what="rms", key=None, scale=1.0):
                v = q.get(ch, what, key) if q is not None else math.nan
                return "" if not math.isfinite(v) else f"{v * scale:.6g}"

            def num(v, scale=1.0):
                return "" if v is None or not math.isfinite(v) else f"{v * scale:.6g}"
            row = {"n": i + 1, "time": o.time, "loop": self.loop, "ramp": c.group, "role": c.role, "G": c.g,
                   "rho": c.rho, "settings": c.label(), "kp": num(o.kp), "ki": num(o.ki), "result": o.status,
                   "best": "yes" if i == self._best else "",
                   "up_rise_ms": num(a.up.rise_s, 1e3) if a else "", "down_rise_ms": num(a.down.rise_s, 1e3) if a else "",
                   "up_overshoot_pct": num(a.up.overshoot, 100) if a else "",
                   "down_overshoot_pct": num(a.down.overshoot, 100) if a else "",
                   "drive_floor_pct": num(a.floor_frac, 100) if a else "",
                   "df_rms_mHz": qv("df", scale=1e3), f"df_pixel_{px * 1e3:g}ms_mHz": qv("df", "pixel", px, 1e3),
                   "phase_rms_deg": qv("Phase"), "amp_rel_rms_pct": qv("QPlusAmpl", "rel_rms", scale=100),
                   "drive_mean_uV": qv("Drive", "mean", scale=1e6), "drive_rms_uV": qv("Drive", scale=1e6),
                   "drive_floor_quiet_pct": qv("Drive", "floor_frac", scale=100), "notes": o.note}
            for k in R.SETTINGS:
                v = o.settings.get(k) if o.settings else None
                row[k] = "" if v is None else str(v)
            rows.append(row)
        return rows

    def export_results_to(self, path: str) -> str:
        """Write the measured conditions to ``path`` (CSV) and a .json next to it. Returns the JSON path."""
        rows = self._export_rows()
        if not rows:
            raise OSError("nothing measured yet")
        meta = _copy_meta(next((o.meta for o in self.outcomes.values() if o.meta is not None), None)
                          or MD.collect(self.reader))
        a = self.anchor()
        meta.add_section("Run sheet", [("Loop", self.loop, ""), ("Anchor Kp0", a.kp, ""), ("Anchor Ki0", a.ki, ""),
                                       ("Anchor output gain", a.output_gain_v, "V"),
                                       ("Anchor input gain", a.input_gain, "x"),
                                       ("Conditions measured", len(rows), ""),
                                       ("Best", "" if self._best is None else f"#{self._best + 1}", "")])
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(MD.csv_preamble(meta.header_lines("SXM nc-AFM run sheet results"), numeric=False))
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        detail = []
        for i, o in sorted(self.outcomes.items()):
            q = o.quiet
            detail.append({"n": i + 1, "condition": dataclasses.asdict(o.condition),
                           "kp": o.kp, "ki": o.ki, "result": o.status, "note": o.note, "marks_s": o.marks,
                           "events_s": o.events,
                           "quiet": None if q is None else {
                               "duration_s": q.duration_s,
                               "channels": {n: {"mean": c.mean, "rms": c.rms, "rel_rms": c.rel_rms, "bands": c.bands,
                                                "pixel": {str(k): v for k, v in c.pixel.items()},
                                                "floor_frac": c.floor_frac} for n, c in q.channels.items()},
                               "coherence": q.coherence},
                           "sxm": o.meta.to_dict()["groups"] if o.meta is not None else None})
        return MD.write_sidecar(os.path.splitext(path)[0] + ".json", meta, {
            "kind": "ncafm_runsheet_results", "sheet": R.sheet_to_dict(a, self.sheet, {"train": self._train_dict()}),
            "results": detail})

    def _results_name(self) -> str:
        meta = next((o.meta for o in self.outcomes.values() if o.meta is not None), None) or MD.Metadata.unavailable("none")
        return meta.filename("runsheet", f"{self.loop.upper()}-{len(self.outcomes)}conditions",
                             loops=(self.loop, "pll" if self.loop == "afl" else "afl")) + ".csv"

    def _export_results(self):
        if not self.outcomes:
            return
        name = self._results_name()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", os.path.join(self._last_dir, name),
                                                        "CSV (*.csv)")
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        try:
            jp = self.export_results_to(path)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Export results", f"Could not write:\n{e}")
            return
        self._log(f"Exported {len(self.outcomes)} conditions to {path} (+ {os.path.basename(jp)}).")

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
