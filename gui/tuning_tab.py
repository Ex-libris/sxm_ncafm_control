"""
Tuning tab: explore the (Kp, Ki) plane of the PLL or the amplitude loop on the instrument.

What the user does
------------------
1. Pick the loop. Everything about the instrument is read from SXM (gains, setpoint, gains of the DNC, Tau,
   TimeConstant, f0, Q) and shown with automatic checks; the Advanced drawer can override it (offline use).
2. Tick the two things SXM cannot report (tip retracted; amplitude feedback on / Auto 0 deg off).
3. Optionally adjust the search region (centre = the gains in SXM, span in decades, Kp = 0 column).
4. Press Explore, or click any point of the Kp-Ki map and test it.

What the tab does (tuning/explore.py, gui/tuning_runner.py)
-----------------------------------------------------------
Every condition starts from the same verified baseline (baseline gains written back, the loop must return
to the reference state), then the candidate gains settle and a step train runs. Up-steps and down-steps are
judged separately; a condition is as good as its worse direction. The automatic exploration runs a coarse
grid in decades, bisects the common scale g along the best Ki:Kp ratios to find their edge, searches the
ratio at the best point, and repeats the candidates. Results are shown as a trade-off (speed vs. noise), with
the fastest and the quietest clean point that keep a margin to the edge.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from sxm_ncafm_control import metadata as MD
from sxm_ncafm_control.tuning import explore as X
from sxm_ncafm_control.tuning import metrics as M
from sxm_ncafm_control.tuning import workflow as W

from ..common import append_log_line, format_number
from .sci_spinbox import SciDoubleSpinBox
from .tuning_runner import ConditionRunner

STATUS_COLOR = {
    "clean": (70, 175, 95), "slow": (120, 170, 230), "overshoot": (240, 200, 60), "floor": (240, 140, 50), "ringing": (220, 90, 50),
    "lost": (205, 65, 65), "untested": (205, 205, 205), "skipped": (120, 120, 120),
}
CONFIRM = {
    "afl": ("Tip retracted / far from the surface", "Amplitude feedback is ON in SXM"),
    "pll": ("Tip retracted / far from the surface", "DNC Lockin Options > Acquire > Auto 0 deg is OFF"),
}
LOOP_NAME = {"afl": "Amplitude loop (QPlusAmpl / Drive)", "pll": "PLL (df / Phase)"}


def _rgb(status: str) -> str:
    return "#%02x%02x%02x" % STATUS_COLOR.get(status, (0, 0, 0))


def _ms(v: float) -> str:
    return "n/a" if v is None or math.isnan(v) else ("inf" if math.isinf(v) else f"{v * 1e3:.3g} ms")


def _fmt(v: float) -> str:
    return format_number(v, 4, 1e4)


@dataclass
class TestRecord:
    """One measured condition, with everything needed to show and export it."""

    proposal: X.Proposal
    result: W.StepTestResult
    assessment: X.Assessment
    meta: Optional[MD.Metadata]
    plan: W.StepTestPlan
    time: str


class CollapsibleSection(QtWidgets.QWidget):
    """A section that folds to its header; the header keeps a one-line summary of what is inside."""

    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title, self._summary = title, ""
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.header = QtWidgets.QToolButton()
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setStyleSheet("QToolButton { border: none; font-weight: bold; text-align: left; }")
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        v.addWidget(self.header)
        self.content = QtWidgets.QWidget()
        v.addWidget(self.content)
        self.header.toggled.connect(self._on_toggled)
        self._on_toggled(expanded)

    def set_content_layout(self, layout):
        self.content.setLayout(layout)

    def set_summary(self, text: str):
        self._summary = text
        self._refresh()

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool):
        self.header.setChecked(expanded)

    def _on_toggled(self, checked):
        self.header.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content.setVisible(checked)
        self._refresh()
        self.toggled.emit(checked)

    def _refresh(self):
        self.header.setText(self._title if self.header.isChecked() or not self._summary
                            else f"{self._title}  —  {self._summary}")


class DecadeAxis(pg.AxisItem):
    """An axis in log10 units that labels decades with the real value (1e8) and the Kp = 0 column as '0'."""

    def __init__(self, orientation, zero_at: Optional[float] = None, sign: float = 1.0):
        super().__init__(orientation)
        self.zero_at, self.sign = zero_at, sign

    def tickValues(self, minVal, maxVal, size):
        lo, hi = math.floor(minVal), math.ceil(maxVal)
        major = [float(v) for v in range(lo, hi + 1)]
        return [(1.0, major)]

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            if self.zero_at is not None and abs(v - self.zero_at) < 1e-6:
                out.append("0")
            elif self.zero_at is not None and v < self.zero_at + 0.5:
                out.append("")
            else:
                out.append(_fmt(self.sign * 10 ** v))
        return out


# ---------------------------------------------------------------------------
# scope capture -> CapturedTest (the step train recorded in the Scope tab, with Step Test events)
# ---------------------------------------------------------------------------
def _parse_event_value(label: str) -> Optional[float]:
    import re
    m = re.findall(r"=\s*([-+0-9.eE]+)", label or "")
    try:
        return float(m[-1]) if m else None
    except ValueError:
        return None


def capture_from_scope(scope, kp: float, ki: float, bin_s: float = 0.0005):
    """The Scope tab's last capture (channels + Step Test events) as a CapturedTest, or (None, why)."""
    if scope.last_data1 is None or scope.last_data2 is None or not scope.last_rate:
        return None, "The Scope tab has no capture yet."
    names = [scope.last_chan1, scope.last_chan2]
    det = W.detect_loop(names)
    if det.loop is None:
        return None, det.note
    markers = list(getattr(scope, "_event_markers", []) or [])
    if len(markers) < 3 or scope.capture_start_dt is None:
        return None, "No Step Test events are attached to this capture: run the Step Test with 'Trigger scope capture' on."
    pairs = sorted((scope.capture_start_dt.msecsTo(dt) / 1000.0, _parse_event_value(lbl)) for dt, lbl in markers)
    times, values = [p[0] for p in pairs], [p[1] for p in pairs]
    if any(v is None for v in values):
        return None, "Could not read the values from the event labels."
    lv = sorted(set(round(v, 9) for v in values))
    if len(lv) < 2:
        return None, "The events do not alternate between two levels."
    low, high = lv[0], lv[-1]
    hold = float(np.median(np.diff(times)))
    t_end = len(scope.last_data1) / scope.last_rate
    starts_high = abs(values[0] - low) < abs(values[0] - high)
    try:
        base = 0.5 * (low + high)
        step = (high - low) / (low + high) if det.loop.relative_step else 0.5 * (high - low)
        plan = W.StepTestPlan(loop=det.loop.key, base=base, step=step, hold_s=hold, n_events=len(times), lead_s=times[0],
                              tail_s=max(0.0, t_end - times[-1] - hold), start_high=starts_high)
    except ValueError as e:
        return None, f"The events do not form a valid test: {e}"
    t_full = np.arange(len(scope.last_data1)) / scope.last_rate
    ch, t_bins = {}, None
    for name, arr in zip(names, (scope.last_data1, scope.last_data2)):
        t_bins, ch[name] = M.block_mean(t_full, np.asarray(arr, float), bin_s)
    return W.CapturedTest(plan=plan, t=t_bins, channels=ch, event_times=times, kp=kp, ki=ki), det.note


# ---------------------------------------------------------------------------
# the tab
# ---------------------------------------------------------------------------
class TuningTab(QtWidgets.QWidget):
    LEAD_S = 1.0            # settled recording before the first event (seconds)
    TAIL_S = 0.5

    def __init__(self, dde, driver, scope_tab=None, params_tab=None, reader=None, parent=None):
        super().__init__(parent)
        self.dde, self.driver = dde, driver
        self.reader, self.scope_tab, self.params_tab = reader, scope_tab, params_tab
        self.sxm: Dict = {}                              # last SXM read-back values
        self.sxm_time: Optional[str] = None
        self.tests: List[TestRecord] = []
        self.explorer: Optional[X.Explorer] = None
        self.runner: Optional[ConditionRunner] = None
        self.selected: Optional[Tuple[float, float]] = None
        self._run_meta: Optional[MD.Metadata] = None
        self._last_export_dir = ""
        self._build()
        self._on_loop_changed()
        self.refresh_from_sxm(quiet=True)

    # ================================================================== construction
    def _spin(self, lo, hi, val, dec=3, step=None, suffix="", sci_above=1e6, plain=False):
        s = SciDoubleSpinBox(sci_above=sci_above, plain=plain)
        s.setDecimals(dec)
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
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        pinned = QtWidgets.QFrame()
        pinned.setFrameShape(QtWidgets.QFrame.StyledPanel)
        pb = QtWidgets.QHBoxLayout(pinned)
        pb.setContentsMargins(8, 4, 8, 4)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.RichText)
        pb.addWidget(self.status_label, 1)
        self.btn_stop = QtWidgets.QPushButton("Stop && restore baseline")
        self.btn_stop.clicked.connect(self.stop)
        pb.addWidget(self.btn_stop)
        outer.addWidget(pinned)
        for seq in ("Esc", "Ctrl+."):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(self.stop)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        outer.addWidget(split, 1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(300)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        left_scroll.setWidget(left)
        split.addWidget(left_scroll)

        # 1. loop
        g = QtWidgets.QGroupBox("1. Loop")
        f = QtWidgets.QFormLayout(g)
        self.loop_combo = QtWidgets.QComboBox()
        for key in ("afl", "pll"):
            self.loop_combo.addItem(LOOP_NAME[key], key)
        self.loop_combo.currentIndexChanged.connect(self._on_loop_changed)
        f.addRow(self.loop_combo)
        lv.addWidget(g)

        # 2. instrument
        g = QtWidgets.QGroupBox("2. Instrument (read from SXM)")
        v = QtWidgets.QVBoxLayout(g)
        self.btn_refresh = QtWidgets.QPushButton("Read SXM again")
        self.btn_refresh.clicked.connect(lambda: self.refresh_from_sxm())
        v.addWidget(self.btn_refresh)
        self.instrument_label = QtWidgets.QLabel()
        self.instrument_label.setTextFormat(QtCore.Qt.RichText)
        self.instrument_label.setWordWrap(True)
        v.addWidget(self.instrument_label)
        self.checks_label = QtWidgets.QLabel()
        self.checks_label.setTextFormat(QtCore.Qt.RichText)
        self.checks_label.setWordWrap(True)
        v.addWidget(self.checks_label)
        lv.addWidget(g)

        # 3. confirmations
        g = QtWidgets.QGroupBox("3. Confirm (SXM cannot tell)")
        v = QtWidgets.QVBoxLayout(g)
        self.confirms = []
        for _ in range(2):
            c = QtWidgets.QCheckBox()
            c.toggled.connect(self._update_enabled)
            v.addWidget(c)
            self.confirms.append(c)
        lv.addWidget(g)

        # 4. search region
        g = QtWidgets.QGroupBox("4. Search region")
        f = QtWidgets.QFormLayout(g)
        self.kp_center = self._spin(-1e15, 1e15, 2e8, 4, sci_above=1e4)
        self.ki_center = self._spin(-1e15, 1e15, 2e4, 4, sci_above=1e4)
        f.addRow("Centre Kp:", self.kp_center)
        f.addRow("Centre Ki:", self.ki_center)
        row = QtWidgets.QHBoxLayout()
        self.btn_center_sxm = QtWidgets.QPushButton("= gains in SXM")
        self.btn_center_manual = QtWidgets.QPushButton("= manual's start")
        self.btn_center_sxm.clicked.connect(self._center_on_sxm)
        self.btn_center_manual.clicked.connect(self._center_on_manual)
        row.addWidget(self.btn_center_sxm)
        row.addWidget(self.btn_center_manual)
        f.addRow(row)
        self.span_spin = QtWidgets.QSpinBox()
        self.span_spin.setRange(1, 3)
        self.span_spin.setValue(1)
        self.span_spin.setSuffix(" decade(s) each way")
        f.addRow("Span:", self.span_spin)
        self.per_decade = QtWidgets.QComboBox()
        self.per_decade.addItem("1 per decade", 1)
        self.per_decade.addItem("2 per decade", 2)
        f.addRow("Grid points:", self.per_decade)
        self.kp0_check = QtWidgets.QCheckBox("Include Kp = 0 (integral only)")
        self.kp0_check.setChecked(True)
        f.addRow(self.kp0_check)
        self.region_label = QtWidgets.QLabel()
        self.region_label.setWordWrap(True)
        f.addRow(self.region_label)
        for w in (self.kp_center, self.ki_center):
            w.valueChanged.connect(self._on_region_changed)
        self.span_spin.valueChanged.connect(self._on_region_changed)
        self.per_decade.currentIndexChanged.connect(self._on_region_changed)
        self.kp0_check.toggled.connect(self._on_region_changed)
        lv.addWidget(g)

        # 5. actions
        g = QtWidgets.QGroupBox("5. Run")
        v = QtWidgets.QVBoxLayout(g)
        self.btn_explore = QtWidgets.QPushButton("Explore automatically")
        self.btn_explore.setToolTip("Grid in decades -> edge along g (bisection) -> ratio search -> repeats. "
                                    "Every condition starts from the verified baseline.")
        self.btn_test = QtWidgets.QPushButton("Test selected point")
        self.btn_test.setToolTip("Click a point of the map (or anywhere on it) first.")
        self.btn_stage = QtWidgets.QPushButton("Stage selected in Params tab")
        self.btn_export = QtWidgets.QPushButton("Export results...")
        self.btn_clear = QtWidgets.QPushButton("Clear results")
        for b in (self.btn_explore, self.btn_test, self.btn_stage, self.btn_export, self.btn_clear):
            v.addWidget(b)
        self.btn_explore.clicked.connect(self.explore)
        self.btn_test.clicked.connect(self.test_selected)
        self.btn_stage.clicked.connect(self._stage_selected)
        self.btn_export.clicked.connect(self._export_results)
        self.btn_clear.clicked.connect(self._clear)
        lv.addWidget(g)

        # advanced
        self.advanced = CollapsibleSection("Advanced", expanded=False)
        f = QtWidgets.QFormLayout()
        self.override_check = QtWidgets.QCheckBox("Use these values instead of SXM's")
        self.override_check.setToolTip("Normally every value below is read from SXM and not editable. Tick to type "
                                        "them (offline analysis, or SXM not readable).")
        self.override_check.toggled.connect(self._on_override)
        f.addRow(self.override_check)
        self.base_spin = self._spin(-1e12, 1e12, 1.0, 4, plain=True)
        self.base_label = QtWidgets.QLabel()
        f.addRow(self.base_label, self.base_spin)
        self.kp_spin = self._spin(-1e15, 1e15, 2e8, 4, sci_above=1e4)
        self.ki_spin = self._spin(-1e15, 1e15, 2e4, 4, sci_above=1e4)
        f.addRow("Baseline Kp:", self.kp_spin)
        f.addRow("Baseline Ki:", self.ki_spin)
        self.gain_combo = QtWidgets.QComboBox()
        for gv in W.AFL_OUTPUT_GAINS:
            self.gain_combo.addItem(f"+-{gv:g} V", gv)
        self.gain_combo.setCurrentIndex(W.AFL_OUTPUT_GAINS.index(1.0))
        f.addRow("Output gain:", self.gain_combo)
        self.ina_combo = QtWidgets.QComboBox()
        for gv in W.INPUT_GAINS:
            self.ina_combo.addItem(f"x{gv:g}", gv)
        f.addRow("Input gain InA:", self.ina_combo)
        self.f0_spin = self._spin(1.0, 1e7, 25000.0, 3, 100.0, " Hz", plain=True)
        self.q_spin = self._spin(1.0, 1e8, 25000.0, 0, 1000.0)
        self.li_spin = self._spin(0.0, 1000.0, 2.0, 3, 0.5, " ms")
        self.tau_spin = self._spin(0.0, 5000.0, 10.0, 3, 1.0, " ms")
        f.addRow("f0:", self.f0_spin)
        f.addRow("Q:", self.q_spin)
        f.addRow("DNC TimeConstant:", self.li_spin)
        f.addRow("Amplitude Tau:", self.tau_spin)
        self._sxm_fields = (self.base_spin, self.kp_spin, self.ki_spin, self.gain_combo, self.ina_combo,
                            self.f0_spin, self.q_spin, self.li_spin, self.tau_spin)
        f.addRow(QtWidgets.QLabel("<b>Method</b> (derived; change only if needed)"))
        self.step_spin = self._spin(0.001, 1e6, 5.0, 3)
        self.step_label = QtWidgets.QLabel()
        f.addRow(self.step_label, self.step_spin)
        self.hold_spin = self._spin(0.15, 30.0, 1.0, 2, 0.05, " s")
        self.events_spin = QtWidgets.QSpinBox()
        self.events_spin.setRange(5, 15)
        self.events_spin.setValue(7)
        self.settle_spin = self._spin(0.2, 60.0, 2.0, 1, 0.5, " s")
        self.recover_spin = self._spin(2.0, 600.0, 20.0, 0, 5.0, " s")
        self.window_spin = self._spin(0.1, 5.0, 0.5, 2, 0.1, " s")
        f.addRow("Hold per level:", self.hold_spin)
        f.addRow("Steps per test:", self.events_spin)
        f.addRow("Settle at candidate (min):", self.settle_spin)
        f.addRow("Recovery timeout:", self.recover_spin)
        f.addRow("Judged over the last:", self.window_spin)
        self.os_spin = self._spin(1.0, 100.0, 10.0, 0, 1.0, " %")
        self.floor_spin = self._spin(0.0, 100.0, 2.0, 1, 0.5, " %")
        self.margin_spin = self._spin(1.0, 10.0, 1.5, 2, 0.1, " x")
        self.range_spin = self._spin(2.0, 1e9, 1e4, 0, 10.0, " x")
        f.addRow("Overshoot allowed:", self.os_spin)
        f.addRow("Drive at zero allowed:", self.floor_spin)
        f.addRow("Margin to the edge:", self.margin_spin)
        f.addRow("Max gain change vs baseline:", self.range_spin)
        self.btn_scope = QtWidgets.QPushButton("Analyze Scope capture")
        self.btn_scope.setToolTip("Assess a step train recorded in the Scope tab (with Step Test events), at the "
                                  "baseline gains above.")
        self.btn_scope.clicked.connect(self._analyze_scope)
        f.addRow(self.btn_scope)
        self.advanced.set_content_layout(f)
        for w in self._sxm_fields:                 # read from SXM; editable only with the override ticked
            w.setEnabled(False)
        lv.addWidget(self.advanced)
        for w in self._sxm_fields + (self.step_spin, self.hold_spin, self.settle_spin, self.events_spin):
            sig = getattr(w, "valueChanged", None) or w.currentIndexChanged
            sig.connect(lambda *_: self._refresh_panels())
        lv.addStretch(1)

        # right side
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setChildrenCollapsible(False)
        split.addWidget(right)
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        right.addWidget(top)
        self.kp_axis = DecadeAxis("bottom")
        self.ki_axis = DecadeAxis("left")
        self.map_plot = pg.PlotWidget(axisItems={"bottom": self.kp_axis, "left": self.ki_axis})
        self.map_plot.setBackground("w")
        self.map_plot.setMouseEnabled(False, False)
        self.map_plot.showGrid(x=True, y=True, alpha=0.2)
        self.map_plot.setLabel("bottom", "Kp")
        self.map_plot.setLabel("left", "Ki")
        self.map_plot.setTitle("Kp-Ki map: ▲ up-steps, ▼ down-steps; rings: gold = fastest, blue = quietest")
        self.map_plot.setMinimumSize(320, 260)
        self.map_items = []
        self.map_plot.scene().sigMouseClicked.connect(self._on_map_click)
        top.addWidget(self.map_plot)
        self.detail = QtWidgets.QTextBrowser()
        self.detail.setMinimumWidth(260)
        top.addWidget(self.detail)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)

        self.tabs = QtWidgets.QTabWidget()
        right.addWidget(self.tabs)
        resp = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(resp)
        h.setContentsMargins(0, 0, 0, 0)
        self.plot_y, self.plot_u = pg.PlotWidget(), pg.PlotWidget()
        for p in (self.plot_y, self.plot_u):
            p.setBackground("w")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.addLegend(offset=(5, 5))
            h.addWidget(p, 1)
        self.plot_u.setXLink(self.plot_y)
        self.tabs.addTab(resp, "Response")
        self.plot_g = self._metric_plot("Along g (same Ki:Kp)", "scale")
        self.tabs.addTab(self.plot_g, "Along g")
        self.plot_r = self._metric_plot("Along the ratio (same Kp)", "Ki")
        self.tabs.addTab(self.plot_r, "Along ratio")
        self.plot_t = pg.PlotWidget()
        self.plot_t.setBackground("w")
        self.plot_t.setLogMode(True, True)
        self.plot_t.showGrid(x=True, y=True, alpha=0.3)
        self.plot_t.setLabel("bottom", "speed: slower direction's rise (ms)")
        self.plot_t.setLabel("left", "noise")
        for ax in ("bottom", "left"):
            self.plot_t.getAxis(ax).enableAutoSIPrefix(False)
        self.plot_t.addLegend(offset=(-10, 10))
        self.tabs.addTab(self.plot_t, "Trade-off")
        self.table = QtWidgets.QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(["#", "stage", "Kp", "Ki", "result", "up", "up rise", "down", "down rise",
                                              "Drive at 0", "noise", "recovered in"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.cellClicked.connect(self._on_table_click)
        self.tabs.addTab(self.table, "All tests")
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Log")
        self.guide = QtWidgets.QTextBrowser()
        self.guide.setHtml(guide_html())
        self.tabs.addTab(self.guide, "Guide")
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 900])

    def _metric_plot(self, title, xlabel):
        p = pg.PlotWidget(title=title)
        p.setBackground("w")
        p.setLogMode(True, True)
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setLabel("bottom", xlabel)
        p.setLabel("left", "10-90 % rise (ms)")
        for ax in ("bottom", "left"):
            p.getAxis(ax).enableAutoSIPrefix(False)            # real values (2e8), not '0.2 (x1e+09)'
        p.addLegend(offset=(-10, 10))
        return p

    # ================================================================== state -> objects
    @property
    def loop(self) -> str:
        return self.loop_combo.currentData()

    @property
    def loop_def(self) -> W.LoopDef:
        return W.LOOPS[self.loop]

    def baseline(self) -> Tuple[float, float]:
        return self.kp_spin.value(), self.ki_spin.value()

    def ring_down_s(self) -> Optional[float]:
        f0, q = self.f0_spin.value(), self.q_spin.value()
        return q / (math.pi * f0) if f0 > 0 and q > 0 else None

    def plan(self) -> W.StepTestPlan:
        ld = self.loop_def
        step = self.step_spin.value() / 100.0 if ld.relative_step else self.step_spin.value()
        return W.StepTestPlan(loop=ld.key, base=self.base_spin.value(), step=step, hold_s=self.hold_spin.value(),
                              n_events=self.events_spin.value(), settle_s=self.settle_spin.value(),
                              lead_s=self.LEAD_S, tail_s=self.TAIL_S)

    def region(self) -> X.SearchRegion:
        return X.SearchRegion(self.kp_center.value(), self.ki_center.value(), self.span_spin.value(),
                              self.per_decade.currentData(), self.kp0_check.isChecked())

    def limits(self) -> X.ExploreLimits:
        return X.ExploreLimits(overshoot_max=self.os_spin.value() / 100.0, floor_max=self.floor_spin.value() / 100.0)

    def criteria(self) -> X.RecoveryCriteria:
        return X.RecoveryCriteria(window_s=self.window_spin.value())

    def points(self) -> Dict[Tuple[float, float], X.PointSummary]:
        return X.summarize([r.assessment for r in self.tests if r.assessment.loop == self.loop])

    def trade(self) -> X.TradeOff:
        return X.trade_off(self.points(), margin=self.margin_spin.value())

    # ================================================================== SXM read-back
    def refresh_from_sxm(self, quiet: bool = False) -> bool:
        """Read SXM and (unless overridden) fill every instrument value from it. True if SXM was read."""
        readout = None
        if self.reader is not None:
            try:
                readout = self.reader.read()
            except Exception as e:
                self._log(f"Reading SXM failed: {e}")
        if readout is None or not readout.ok:
            self.sxm, self.sxm_time = {}, None
            if not quiet:
                self._log("SXM could not be read (is it running with its windows open?).")
            self._refresh_panels()
            return False
        self.sxm = dict(readout.values)
        self.sxm_time = readout.timestamp.strftime("%H:%M:%S")
        if not self.override_check.isChecked():
            self._fill_from_sxm()
        self._log("Read SXM: " + ", ".join(f"{k}={v:.6g}" for k, v in self.sxm.items()
                                           if isinstance(v, float) and k in ("amp_kp", "amp_ki", "pll_kp", "pll_ki",
                                                                             "amp_ref", "used_freq", "input_gain_ina")))
        self._refresh_panels()
        return True

    def _fill_from_sxm(self):
        s = self.sxm
        kp_key, ki_key = ("amp_kp", "amp_ki") if self.loop == "afl" else ("pll_kp", "pll_ki")
        old_baseline = self.baseline()

        def put(widget, key, factor=1.0):
            v = s.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                widget.setValue(v * factor)

        put(self.kp_spin, kp_key)
        put(self.ki_spin, ki_key)
        put(self.base_spin, "amp_ref" if self.loop == "afl" else "used_freq")
        put(self.f0_spin, "f_peak")
        put(self.q_spin, "q")
        put(self.li_spin, "dnc_time_constant_s", 1e3)
        put(self.tau_spin, "amp_tau_s", 1e3)
        for combo, key in ((self.gain_combo, "afl_output_gain"), (self.ina_combo, "input_gain_ina")):
            i = combo.findData(s.get(key)) if s.get(key) is not None else -1
            if i >= 0:
                combo.setCurrentIndex(i)
        if self.baseline() != old_baseline:
            self._center_on_sxm()
        self._derive_method()

    def _derive_method(self):
        """Hold, settle and recovery timeout from the sensor's ring-down time (the slowest thing in the loop)."""
        if self.override_check.isChecked():
            return
        if self.loop == "afl":
            try:
                s = W.afl_start_values(self.q_spin.value(), self.f0_spin.value(), self.gain_combo.currentData())
            except ValueError:
                return
            self.hold_spin.setValue(s.hold_s)
            self.settle_spin.setValue(s.settle_s)
            self.recover_spin.setValue(min(600.0, max(10.0, 10 * s.ring_down_s)))
        else:
            self.hold_spin.setValue(0.5)
            self.settle_spin.setValue(2.0)
            self.recover_spin.setValue(20.0)

    def _on_override(self, on: bool):
        for w in self._sxm_fields:
            w.setEnabled(on)
        if not on:
            self.refresh_from_sxm(quiet=True)
        self._refresh_panels()

    def checks(self) -> List[Tuple[str, str]]:
        """``(level, text)``: level 'ok', 'warn' or 'block' (a block disables the run buttons)."""
        out = []
        s, ld = self.sxm, self.loop_def
        if not self._online():
            out.append(("block", "Offline: running tests needs SXM and the driver."))
        if s:
            out.append(("ok", f"Values read from SXM at {self.sxm_time}."))
        elif self.override_check.isChecked():
            out.append(("warn", "SXM not readable: using the values typed in Advanced."))
        else:
            out.append(("block", "SXM not readable: press 'Read SXM again', or type the values in Advanced "
                                 "(tick 'Use these values')."))
        kp, ki = self.baseline()
        if kp == 0 and ki == 0:
            out.append(("block", "Baseline Kp and Ki are both 0: the loop is off."))
        elif any(v != 0 and math.copysign(1, v) != ld.gain_sign for v in (kp, ki)):
            out.append(("block", f"The gains of this loop must be {'negative' if ld.gain_sign < 0 else 'positive'} "
                                 "(manual)."))
        if self.loop == "afl":
            pk, pi = s.get("pll_kp"), s.get("pll_ki")
            if pk is not None and pi is not None:
                out.append(("ok", "PLL off.") if pk == 0 and pi == 0 else
                           ("block", f"PLL is on (Kp={_fmt(pk)}, Ki={_fmt(pi)}): set PLL Kp = Ki = 0 for amplitude tuning."))
            ina = self.ina_combo.currentData()
            out.append(("ok", f"Input gain x{ina:g}.") if ina >= 10 else
                       ("warn", f"Input gain x{ina:g}: use the highest gain that does not overload (x10 for small "
                                "amplitudes). Results are only comparable at one input gain."))
            f0, q, tau = self.f0_spin.value(), self.q_spin.value(), self.tau_spin.value() / 1e3
            if f0 > 0 and q > 0 and tau > 0:
                want = q / (100 * f0)
                ratio = tau / want
                out.append(("ok", f"Tau {tau * 1e3:.3g} ms (manual: Q/(100 f0) = {want * 1e3:.3g} ms).")
                           if 1 / 3 <= ratio <= 3 else
                           ("warn", f"Tau {tau * 1e3:.3g} ms vs Q/(100 f0) = {want * 1e3:.3g} ms (manual)."))
        fb = s.get("feedback_off")
        if fb is True:
            out.append(("ok", "z feedback off."))
        elif fb is False:
            out.append(("warn", "z feedback is ON: make sure the tip is far from the surface."))
        return out

    # ================================================================== panels
    def _refresh_panels(self, *_):
        ld = self.loop_def
        kp, ki = self.baseline()
        rd = self.ring_down_s()
        rows = [("Baseline Kp", _fmt(kp)), ("Baseline Ki", _fmt(ki)),
                ("Steps " + ("Ref" if ld.key == "afl" else "DNC use"),
                 f"{self.base_spin.value():.6g}" + ("" if ld.key == "afl" else " Hz")),
                ("Input / output gain", f"x{self.ina_combo.currentData():g} / +-{self.gain_combo.currentData():g} V"),
                ("TimeConstant / Tau", f"{self.li_spin.value():.3g} ms / {self.tau_spin.value():.3g} ms"),
                ("f0 / Q", f"{self.f0_spin.value():.3f} Hz / {self.q_spin.value():.0f}"),
                ("Ring-down Q/(π f0)", _ms(rd) if rd else "n/a")]
        src = "typed (Advanced)" if self.override_check.isChecked() else ("SXM" if self.sxm else "not read")
        self.instrument_label.setText(
            f"<table cellspacing='0' cellpadding='1'>" +
            "".join(f"<tr><td>{a}</td><td>&nbsp;<b>{b}</b></td></tr>" for a, b in rows) +
            f"</table><i>source: {src}</i>")
        icon = {"ok": "<span style='color:#2a8a3a'>✔</span>", "warn": "<span style='color:#b07000'>⚠</span>",
                "block": "<span style='color:#c03030'>✖</span>"}
        self.checks_label.setText("<br>".join(f"{icon[lvl]} {txt}" for lvl, txt in self.checks()))
        self._refresh_region_label()
        self._update_enabled()

    def _refresh_region_label(self):
        r = self.region()
        try:
            p = self.plan()
            each = p.settle_s * 2 + p.duration + 1.0           # recovery ~ one settle, settle, the test
        except ValueError:
            each = math.nan
        n = X.ExplorePlan(margin=self.margin_spin.value()).estimate(r)
        kps = ", ".join(_fmt(v) for v in r.kps())
        kis = ", ".join(_fmt(v) for v in r.kis())
        self.region_label.setText(f"Kp: {kps}\nKi: {kis}\nAbout {n} tests, ~{n * each / 60:.0f} min "
                                  f"(~{each:.0f} s each, from the ring-down time)")

    def _on_region_changed(self, *_):
        self._refresh_region_label()
        self.explorer = None                    # a new region: the next Explore plans again (tests are kept)
        self._paint()

    def _center_on_sxm(self):
        kp, ki = self.baseline()
        if kp != 0:
            self.kp_center.setValue(kp)
        if ki != 0:
            self.ki_center.setValue(ki)

    def _center_on_manual(self):
        if self.loop == "afl":
            try:
                s = W.afl_start_values(self.q_spin.value(), self.f0_spin.value(), self.gain_combo.currentData())
            except ValueError:
                return
            self.kp_center.setValue(s.kp)
            self.ki_center.setValue(s.ki)
        else:
            self.kp_center.setValue(-100.0)
            self.ki_center.setValue(-1e4)

    def _on_loop_changed(self, *_):
        ld = self.loop_def
        pll = ld.key == "pll"
        for c, text in zip(self.confirms, CONFIRM[ld.key]):
            c.setText(text)
            c.setChecked(False)
        self.base_label.setText("DNC use (Hz):" if pll else "Amplitude Ref:")
        self.step_label.setText("Step +- (Hz):" if pll else "Step +- (% of Ref):")
        self.step_spin.setValue(ld.default_step if pll else 100.0 * ld.default_step)
        if pll:
            self.kp_spin.setValue(-100.0)
            self.ki_spin.setValue(-1e4)
            self.base_spin.setValue(25000.0)
        else:
            self.base_spin.setValue(1.0)
            try:
                s = W.afl_start_values(self.q_spin.value(), self.f0_spin.value(), self.gain_combo.currentData())
                self.kp_spin.setValue(s.kp)
                self.ki_spin.setValue(s.ki)
            except ValueError:
                pass
        self.explorer = None
        self.selected = None
        if self.sxm and not self.override_check.isChecked():
            self._fill_from_sxm()
        else:
            self._derive_method()
        self._center_on_sxm()
        self._refresh_panels()
        self._paint()

    # ================================================================== enabling, status
    def _online(self) -> bool:
        return self.driver is not None and not type(self.dde).__name__.startswith("Mock")

    def runner_active(self) -> bool:
        return self.runner is not None and self.runner.running

    def _blocked(self) -> Optional[str]:
        blocks = [t for lvl, t in self.checks() if lvl == "block"]
        if blocks:
            return blocks[0]
        if not all(c.isChecked() for c in self.confirms):
            return "Tick both items in '3. Confirm'."
        return None

    def _update_enabled(self, *_):
        running = self.runner_active()
        why = self._blocked()
        for b in (self.btn_explore, self.btn_test):
            b.setEnabled(not running and why is None)
            b.setToolTip(why or "")
        self.btn_test.setEnabled(self.btn_test.isEnabled() and self.selected is not None)
        self.btn_stop.setEnabled(running)
        app = QtWidgets.QApplication.instance()
        mgr = getattr(app, "accessibility_manager", None)
        self.btn_stop.setStyleSheet(mgr.stop_button_style(running) if mgr else "")
        has = bool(self.tests)
        self.btn_export.setEnabled(has and not running)
        self.btn_clear.setEnabled(has and not running)
        self.btn_stage.setEnabled(self.selected is not None and self.params_tab is not None and not running)
        self.btn_scope.setEnabled(self.scope_tab is not None and not running)
        if not running:
            if why:
                self._status(f"<b>Not ready:</b> {why}")
            elif not self.tests:
                self._status("<b>Ready.</b> Press <i>Explore automatically</i>, or click a point on the map and "
                             "<i>Test selected point</i>.")
            else:
                self._status("<b>Ready.</b> Click points on the map to inspect them; the recommendation is on the right.")

    def _status(self, html: str):
        conn = "ONLINE" if self._online() else "OFFLINE"
        self.status_label.setText(f"{LOOP_NAME[self.loop]} &middot; {conn} &middot; {html}")

    def _log(self, text: str):
        append_log_line(self.log, f"[{time.strftime('%H:%M:%S')}] {text}")

    # ================================================================== running
    def _gain_ok(self, kp: float, ki: float) -> Optional[str]:
        """None when (kp, ki) is within the allowed change from the baseline (Kp = 0 is always allowed)."""
        n = self.range_spin.value()
        bkp, bki = self.baseline()
        for v, b, name in ((kp, bkp, "Kp"), (ki, bki, "Ki")):
            if v == 0 or b == 0:
                continue
            if not (1 / n <= abs(v / b) <= n):
                return f"{name}={_fmt(v)} is more than x{n:g} from the baseline {_fmt(b)} (Advanced: max gain change)"
            if math.copysign(1, v) != self.loop_def.gain_sign:
                return f"{name}={_fmt(v)} has the wrong sign for this loop"
        return None

    def _start(self, next_item, what: str) -> bool:
        why = self._blocked()
        if why:
            QtWidgets.QMessageBox.warning(self, "Tuning", why)
            return False
        try:
            plan = self.plan()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Tuning", str(e))
            return False
        self._run_meta = MD.collect(self.reader) if self.reader is not None else MD.Metadata.unavailable("no read-back")
        self.runner = ConditionRunner(self.dde, self.driver, plan, self.baseline(), criteria=self.criteria(),
                                      recover_timeout_s=self.recover_spin.value(), parent=self)
        self.runner.message.connect(self._log)
        self.runner.phase.connect(lambda t: self._status(f"<b>Running:</b> {t}"))
        self.runner.condition_started.connect(
            lambda p: self._log(f"[{X.STAGE_LABEL.get(p.stage, p.stage)}] Kp={_fmt(p.kp)}, Ki={_fmt(p.ki)}: {p.why}"))
        self.runner.condition_finished.connect(self._on_condition)
        self.runner.finished.connect(self._on_finished)
        self._log(f"{what} started. Baseline Kp={_fmt(self.baseline()[0])}, Ki={_fmt(self.baseline()[1])}.")
        self.runner.start(next_item)
        self._update_enabled()
        return True

    def explore(self):
        """Run (or continue) the automatic exploration of the search region."""
        if self.explorer is None or self.explorer.done:
            self.explorer = X.Explorer(self.region(), X.ExplorePlan(margin=self.margin_spin.value()))
            for r in self.tests:                          # what was already measured counts: no re-testing
                if r.assessment.loop == self.loop:
                    self.explorer.add_manual(r.assessment)
        ex = self.explorer
        reported = set(ex.skipped)

        def report_skips():
            for k, why in ex.skipped.items():
                if k not in reported:
                    reported.add(k)
                    if not why.startswith(("Kp=", "Ki=")):            # _gain_ok skips are logged below
                        self._log(f"skipped Kp={_fmt(k[0])}, Ki={_fmt(k[1])}: {why}")

        def nxt():
            while True:
                p = ex.next()
                report_skips()
                if p is None:
                    self._log(self._why_exploration_ended(ex))
                    return None
                why = self._gain_ok(p.kp, p.ki)
                if why is None:
                    return p
                self._log(f"skipped Kp={_fmt(p.kp)}, Ki={_fmt(p.ki)}: {why}")
                ex.skip(why)
        self._start(nxt, "Exploration")

    def _why_exploration_ended(self, ex: X.Explorer) -> str:
        pts = ex.points()
        n_clean = sum(p.clean for p in pts.values())
        text = (f"Exploration complete: {len(pts)} point(s) measured, {n_clean} clean, "
                f"{len(ex.skipped)} skipped.")
        if not n_clean:
            text += (" No clean point, so there was nothing to refine (edge, ratio and repeat stages need one). "
                     "Look at the statuses on the map: relax the overshoot / Drive-floor limits, lengthen the hold, "
                     "or move the search centre.")
        return text

    def test_selected(self):
        if self.selected is None:
            return
        kp, ki = self.selected
        why = self._gain_ok(kp, ki)
        if why:
            QtWidgets.QMessageBox.warning(self, "Test selected point", why)
            return
        given = []

        def nxt():
            if given:
                return None
            given.append(1)
            return X.Proposal(kp, ki, "manual", "selected on the map")
        self._start(nxt, "Manual test")

    def stop(self):
        if self.runner is not None:
            self.runner.stop()

    def _on_condition(self, prop: X.Proposal, res: W.StepTestResult):
        a = X.assess(res, self.limits(), self.ring_down_s())
        if self.explorer is not None and prop.stage != "manual" and self.explorer._pending is not None:
            self.explorer.record(a)
        elif self.explorer is not None:
            self.explorer.add_manual(a)
        self.tests.append(TestRecord(prop, res, a, self._run_meta, self.runner.plan, time.strftime("%Y-%m-%d %H:%M:%S")))
        rc = res.meta.get("baseline_check", {})
        self._log(f"  -> {X.STATUS_LABEL[a.status]} (up {a.up.status} {_ms(a.up.rise_s)}, down {a.down.status} "
                  f"{_ms(a.down.rise_s)})" + (f"; recovered in {rc['recover_s']:.1f} s" if "recover_s" in rc else ""))
        self.selected = (prop.kp, prop.ki)
        self._refresh_all()

    def _on_finished(self, reason: str):
        if self.explorer is not None and self.explorer._pending is not None:
            self.explorer = None                   # stopped mid-proposal: plan again next time (tests are kept)
        self._log(f"Run {reason}.")
        to = self.trade()
        if reason == "done" and to.fastest is not None:
            self._log(f"Recommended: fastest Kp={_fmt(to.fastest.kp)}, Ki={_fmt(to.fastest.ki)} "
                      f"({_ms(to.fastest.speed_s)}); quietest Kp={_fmt(to.quietest.kp)}, Ki={_fmt(to.quietest.ki)}.")
        self._update_enabled()
        self._refresh_all()

    def _clear(self):
        self.tests.clear()
        self.explorer = None
        self.selected = None
        self._refresh_all()

    # ================================================================== map
    def _map_coords(self, kp: float, ki: float, x0: float) -> Tuple[float, float]:
        return (x0 if kp == 0 else math.log10(abs(kp))), math.log10(abs(ki))

    def _x0(self) -> float:
        kps = [abs(v) for v in self.region().kps() if v != 0]
        kps += [abs(r.assessment.kp) for r in self.tests if r.assessment.kp != 0]
        return math.floor(math.log10(min(kps))) - 1.0 if kps else 0.0

    def _paint(self):
        p = self.map_plot
        for it in self.map_items:
            p.removeItem(it)
        self.map_items = []
        sign_kp = 1.0 if self.loop_def.gain_sign > 0 else -1.0
        x0 = self._x0()
        self.kp_axis.zero_at, self.kp_axis.sign = (x0 if self.kp0_check.isChecked() or any(
            r.assessment.kp == 0 for r in self.tests) else None), sign_kp
        self.ki_axis.sign = sign_kp

        def add(item):
            p.addItem(item)
            self.map_items.append(item)

        pts = self.points()
        tested = set(pts)
        grid = [g for g in self.region().points() if X._key(*g) not in tested]
        if grid:
            xy = [self._map_coords(kp, ki, x0) for kp, ki in grid]
            skipped = self.explorer.skipped if self.explorer is not None else {}
            brushes = [pg.mkBrush(*STATUS_COLOR["skipped" if X._key(*g) in skipped else "untested"]) for g in grid]
            add(pg.ScatterPlotItem([a for a, _ in xy], [b for _, b in xy], symbol="o", size=7, brush=brushes,
                                   pen=pg.mkPen(150, 150, 150)))
        if pts:
            xs, ys, bu, bd = [], [], [], []
            for s in pts.values():
                x, y = self._map_coords(s.kp, s.ki, x0)
                xs.append(x)
                ys.append(y)
                worst_up = max((a.up.status for a in s.assessments), key=X.SEVERITY.__getitem__)
                worst_dn = max((a.down.status for a in s.assessments), key=X.SEVERITY.__getitem__)
                bu.append(pg.mkBrush(*STATUS_COLOR[worst_up]))
                bd.append(pg.mkBrush(*STATUS_COLOR[worst_dn]))
            add(pg.ScatterPlotItem(xs, [y + 0.07 for y in ys], symbol="t1", size=14, brush=bu, pen=pg.mkPen(60, 60, 60)))
            add(pg.ScatterPlotItem(xs, [y - 0.07 for y in ys], symbol="t", size=14, brush=bd, pen=pg.mkPen(60, 60, 60)))
            to = self.trade()
            for cand, color, size in ((to.fastest, (230, 170, 0), 38), (to.quietest, (40, 140, 240), 46)):
                if cand is not None:                      # rings around the recommended points (gold / blue)
                    x, y = self._map_coords(cand.kp, cand.ki, x0)
                    add(pg.ScatterPlotItem([x], [y], symbol="o", size=size, brush=pg.mkBrush(0, 0, 0, 0),
                                           pen=pg.mkPen(*color, width=3)))
        if self.selected is not None:
            x, y = self._map_coords(*self.selected, x0)
            add(pg.ScatterPlotItem([x], [y], symbol="o", size=30, brush=pg.mkBrush(0, 0, 0, 0),
                                   pen=pg.mkPen(20, 20, 20, width=2)))
        p.enableAutoRange()

    def _on_map_click(self, ev):
        if self.runner_active():
            return
        vb = self.map_plot.getViewBox()
        if not self.map_plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = vb.mapSceneToView(ev.scenePos())
        self.select_at(pos.x(), pos.y())

    def select_at(self, x: float, y: float):
        """Select the nearest tested or grid point within 0.35 decades, otherwise the clicked spot (0.25-decade grid)."""
        x0 = self._x0()
        cands = list({X._key(s.kp, s.ki): (s.kp, s.ki) for s in self.points().values()}.values())
        cands += self.region().points()
        best, dist = None, 0.35
        for kp, ki in cands:
            cx, cy = self._map_coords(kp, ki, x0)
            d = math.hypot(cx - x, cy - y)
            if d < dist:
                best, dist = (kp, ki), d
        if best is None:
            sign = 1.0 if self.loop_def.gain_sign > 0 else -1.0
            kp = 0.0 if x < x0 + 0.5 else sign * 10 ** (round(x * 4) / 4)
            best = (kp, sign * 10 ** (round(y * 4) / 4))
        self.select(best)

    def select(self, point: Tuple[float, float]):
        self.selected = point
        self._refresh_all()

    def _on_table_click(self, row, _col):
        if 0 <= row < len(self.tests):
            a = self.tests[row].assessment
            self.select((a.kp, a.ki))

    # ================================================================== views
    def _refresh_all(self):
        self._paint()
        self._refresh_detail()
        self._refresh_table()
        self._plot_response()
        self._plot_along()
        self._plot_tradeoff()
        self._update_enabled()

    def _selected_summary(self) -> Optional[X.PointSummary]:
        if self.selected is None:
            return None
        return self.points().get(X._key(*self.selected))

    def _refresh_detail(self):
        to = self.trade()
        parts = []
        if to.fastest is not None:
            parts.append("<h3>Recommendation</h3>")
            parts.append(f"<b>Fastest</b> (with x{self.margin_spin.value():g} margin to the edge): "
                         f"Kp {_fmt(to.fastest.kp)}, Ki {_fmt(to.fastest.ki)}: {_ms(to.fastest.speed_s)}, "
                         f"noise {to.fastest.noise:.3g}<br>")
            parts.append(f"<b>Quietest</b> within 1.5x of that speed: Kp {_fmt(to.quietest.kp)}, "
                         f"Ki {_fmt(to.quietest.ki)}: {_ms(to.quietest.speed_s)}, noise {to.quietest.noise:.3g}")
            if any(a.ringdown_limited for p in (to.fastest, to.quietest) for a in p.assessments):
                parts.append("<br><i>Down-steps are limited by the sensor's ring-down time, not by the gains.</i>")
        elif self.tests:
            parts.append("<h3>Recommendation</h3>No clean point with a margin to the edge yet.")
        s = self._selected_summary()
        if self.selected is not None:
            kp, ki = self.selected
            ratio = "∞ (integral only)" if kp == 0 else f"{ki / kp:.3g}"
            parts.append(f"<h3>Kp {_fmt(kp)}, Ki {_fmt(ki)}</h3>Ki:Kp = {ratio}")
            if s is None:
                why = (self.explorer.skipped.get(X._key(kp, ki)) if self.explorer else None)
                parts.append("<br>Not tested yet." + (f" Skipped: {why}" if why else
                                                      " Press <i>Test selected point</i>."))
            else:
                parts.append(f" &middot; tested {s.n}x &middot; <span style='background:{_rgb(s.status)};"
                             f"padding:1px 5px'>{X.STATUS_LABEL[s.status]}</span>")
                a = s.assessments[-1]
                parts.append("<table border='1' cellspacing='0' cellpadding='3'><tr><th></th><th>result</th>"
                             "<th nowrap>rise</th><th nowrap>overshoot</th><th nowrap>wiggles</th></tr>")
                for name, d in (("up", a.up), ("down", a.down)):
                    parts.append(f"<tr><td>{name}</td><td bgcolor='{_rgb(d.status)}'>{X.STATUS_LABEL[d.status]}</td>"
                                 f"<td>{_ms(d.rise_s)}</td><td>{'' if math.isnan(d.overshoot) else f'{d.overshoot * 100:.0f} %'}"
                                 f"</td><td>{d.extrema}</td></tr>")
                parts.append("</table>")
                extra = []
                if not math.isnan(a.floor_frac):
                    extra.append(f"Drive at zero {a.floor_frac * 100:.0f} % of the test")
                if not math.isnan(a.noise):
                    extra.append(f"noise {a.noise:.3g}")
                rec = [r for r in self.tests if X._key(r.assessment.kp, r.assessment.ki) == X._key(kp, ki)][-1]
                rc = rec.result.meta.get("baseline_check", {})
                if "recover_s" in rc:
                    extra.append(f"recovered to the baseline in {rc['recover_s']:.1f} s, settled in "
                                 f"{rc.get('settle_s', math.nan):.1f} s")
                if X.near_edge(s, self.points(), self.margin_spin.value()) and s.clean:
                    extra.append("<b>close to the edge</b>: a failure lies within the margin above it along g")
                parts.append("<br>".join(extra))
                for why in a.up.reasons + a.down.reasons + a.reasons + rec.result.warnings:
                    parts.append(f"<br><span style='color:#805000'>{why}</span>")
        elif not self.tests:
            parts.append("<b>No tests yet.</b><br>Grey dots are the grid the exploration will measure. "
                         "▲ / ▼ show the up- and down-step result of each tested point.")
        self.detail.setHtml("".join(parts))

    def _refresh_table(self):
        self.table.setRowCount(len(self.tests))
        for i, r in enumerate(self.tests):
            a = r.assessment
            rc = r.result.meta.get("baseline_check", {})
            vals = [str(i + 1), X.STAGE_LABEL.get(r.proposal.stage, r.proposal.stage), _fmt(a.kp), _fmt(a.ki),
                    X.STATUS_LABEL[a.status], a.up.status, _ms(a.up.rise_s), a.down.status, _ms(a.down.rise_s),
                    "" if math.isnan(a.floor_frac) else f"{a.floor_frac * 100:.0f} %",
                    "" if math.isnan(a.noise) else f"{a.noise:.3g}",
                    f"{rc['recover_s']:.1f} s" if "recover_s" in rc else ""]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                if c in (4, 5, 7):
                    st = a.status if c == 4 else (a.up.status if c == 5 else a.down.status)
                    item.setBackground(QtGui.QColor(*STATUS_COLOR[st]))
                self.table.setItem(i, c, item)

    def _plot_response(self):
        for p in (self.plot_y, self.plot_u):
            p.clear()
        if self.selected is None:
            return
        recs = [r for r in self.tests if X._key(r.assessment.kp, r.assessment.ki) == X._key(*self.selected)]
        if not recs or recs[-1].result.grid is None:
            return
        res, ld = recs[-1].result, self.loop_def
        t = res.grid * 1e3
        self.plot_y.setTitle(f"{ld.primary_channel} (sign-folded, from each step)")
        self.plot_u.setTitle("Drive" if ld.key == "afl" else "Phase")
        self.plot_y.setLabel("bottom", "ms after the step")
        self.plot_u.setLabel("bottom", "ms after the step")
        for arr, pen, name in ((res.mean_primary_rising, (40, 110, 220), "up-steps"),
                               (res.mean_primary_falling, (220, 80, 40), "down-steps")):
            if arr is not None:
                self.plot_y.plot(t, arr, pen=pg.mkPen(*pen, width=2), name=name)
        sec = [(res.mean_secondary_rising, (40, 110, 220), "up-steps"), (res.mean_secondary_falling, (220, 80, 40), "down-steps")]
        if all(a is None for a, _, _ in sec) and res.mean_secondary is not None:
            sec = [(res.mean_secondary, (90, 90, 90), "both (folded)")]
        for arr, pen, name in sec:
            if arr is not None:
                self.plot_u.plot(t, arr, pen=pg.mkPen(*pen, width=2), name=name)

    def _plot_series(self, plot, series, xfun):
        plot.clear()
        if not series:
            return
        xs = [xfun(s) for s in series]
        for attr, pen, name in (("up", (40, 110, 220), "up-steps"), ("down", (220, 80, 40), "down-steps")):
            ys = [getattr(s.assessments[-1], attr).rise_s * 1e3 for s in series]
            ok = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0 and math.isfinite(y)]
            if ok:
                plot.plot([x for x, _ in ok], [y for _, y in ok], pen=pg.mkPen(*pen, width=2), name=name)
            pts = [(x, y, getattr(s.assessments[-1], attr).status) for x, y, s in zip(xs, ys, series)
                   if x > 0 and y > 0 and math.isfinite(y)]
            if pts:
                plot.addItem(pg.ScatterPlotItem([math.log10(x) for x, _, _ in pts], [math.log10(y) for _, y, _ in pts],
                                                symbol="t1" if attr == "up" else "t", size=12,
                                                brush=[pg.mkBrush(*STATUS_COLOR[st]) for _, _, st in pts]))

    def _plot_along(self):
        if self.selected is None:
            self.plot_g.clear()
            self.plot_r.clear()
            return
        kp, ki = self.selected
        pts = self.points()
        ratio = math.inf if kp == 0 else ki / kp
        along_g = X.along_ratio(pts, ratio)
        self.plot_g.setTitle("Integral only: along Ki" if kp == 0 else f"Along g at Ki:Kp = {ratio:.3g}  (role of g)")
        self.plot_g.setLabel("bottom", "|Ki|" if kp == 0 else "|Kp|  (Ki follows the ratio)")
        self._plot_series(self.plot_g, along_g, lambda s: abs(s.ki) if kp == 0 else abs(s.kp))
        along_r = X.along_kp(pts, kp)
        self.plot_r.setTitle(f"Along the ratio at Kp = {_fmt(kp)}  (role of Ki:Kp)")
        self.plot_r.setLabel("bottom", "|Ki|" if kp == 0 else "Ki:Kp")
        self._plot_series(self.plot_r, along_r, lambda s: abs(s.ki) if kp == 0 else abs(s.ki / s.kp))

    def _plot_tradeoff(self):
        p = self.plot_t
        p.clear()
        pts = self.points()
        to = self.trade()
        clean = [s for s in pts.values() if s.clean and math.isfinite(s.speed_s) and s.noise > 0]
        if not clean:
            return

        def xy(ss):
            return [math.log10(s.speed_s * 1e3) for s in ss], [math.log10(s.noise) for s in ss]
        edge = [s for s in clean if s in to.at_edge]
        safe = [s for s in clean if s not in to.at_edge]
        for ss, color, name in ((safe, STATUS_COLOR["clean"], "clean"), (edge, (180, 180, 60), "clean, near the edge")):
            if ss:
                x, y = xy(ss)
                p.addItem(pg.ScatterPlotItem(x, y, symbol="o", size=10, brush=pg.mkBrush(*color), name=name))
        if to.front:
            p.plot([s.speed_s * 1e3 for s in to.front], [s.noise for s in to.front], pen=pg.mkPen(60, 60, 60, width=1),
                   name="front")
        for cand, color in ((to.fastest, (230, 170, 0)), (to.quietest, (40, 140, 240))):
            if cand is not None and cand.noise > 0:
                x, y = xy([cand])
                p.addItem(pg.ScatterPlotItem(x, y, symbol="star", size=18, brush=pg.mkBrush(*color)))

    # ================================================================== hand-off, analysis
    def _stage_selected(self):
        if self.selected is None or self.params_tab is None:
            return
        kp, ki = self.selected
        ld = self.loop_def
        ok = self.params_tab.stage_value(ld.kp_param[0], str(ld.kp_param[1]), kp)
        ok &= self.params_tab.stage_value(ld.ki_param[0], str(ld.ki_param[1]), ki)
        self._log(f"Staged Kp={_fmt(kp)}, Ki={_fmt(ki)} in the Parameters tab." if ok else "Could not stage the pair.")

    def _analyze_scope(self):
        if self.scope_tab is None:
            return
        kp, ki = self.baseline()
        ct, msg = capture_from_scope(self.scope_tab, kp, ki)
        if ct is None:
            self._log(f"Scope analysis: {msg}")
            QtWidgets.QMessageBox.information(self, "Analyze Scope capture", msg)
            return
        if ct.plan.loop != self.loop:
            self.loop_combo.setCurrentIndex(self.loop_combo.findData(ct.plan.loop))
        res = W.analyze_test(ct)
        res.warnings.insert(0, f"gains assumed for this capture: Kp={_fmt(kp)}, Ki={_fmt(ki)} (the baseline)")
        a = X.assess(res, self.limits(), self.ring_down_s())
        self.tests.append(TestRecord(X.Proposal(kp, ki, "manual", "Scope capture"), res, a,
                                     getattr(self.scope_tab, "last_meta", None), ct.plan,
                                     time.strftime("%Y-%m-%d %H:%M:%S")))
        self.selected = (kp, ki)
        self._log(f"Scope analysis: {msg}; {X.STATUS_LABEL[a.status]}")
        self._refresh_all()

    # ================================================================== export
    EXPORT_COLUMNS = (
        ("n", "n"), ("time", "time"), ("stage", "stage"), ("loop", "loop"), ("kp", "Kp"), ("ki", "Ki"),
        ("ratio", "Ki:Kp"), ("result", "result"), ("up", "up-steps"), ("up_rise_ms", "up rise 10-90 % [ms]"),
        ("up_overshoot_pct", "up overshoot [%]"), ("down", "down-steps"), ("down_rise_ms", "down rise 10-90 % [ms]"),
        ("down_overshoot_pct", "down overshoot [%]"), ("ringdown_limited", "down limited by ring-down"),
        ("drive_floor_pct", "Drive at zero [%]"), ("noise", "noise"), ("recover_s", "recovered to baseline [s]"),
        ("settle_s", "settled at candidate [s]"), ("base", "base"), ("low", "low"), ("high", "high"),
        ("hold_s", "hold [s]"), ("events", "steps"), ("input_gain", "input gain InA (SXM)"),
        ("output_gain_v", "output gain [+-V] (SXM)"), ("amp_tau_ms", "Amplitude Tau [ms] (SXM)"),
        ("dnc_tc_ms", "DNC TimeConstant [ms] (SXM)"), ("q", "Q (SXM)"), ("notes", "notes"),
    )

    @staticmethod
    def _export_row(i: int, r: TestRecord) -> dict:
        def num(v, f=1.0):
            return "" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{v * f:.6g}"
        a, p = r.assessment, r.plan
        sx = r.meta.values if r.meta is not None else {}
        rc = r.result.meta.get("baseline_check", {})
        return {
            "n": i + 1, "time": r.time, "stage": r.proposal.stage, "loop": a.loop, "kp": f"{a.kp:.6g}", "ki": f"{a.ki:.6g}",
            "ratio": "inf" if a.kp == 0 else f"{a.ki / a.kp:.6g}", "result": a.status,
            "up": a.up.status, "up_rise_ms": num(a.up.rise_s, 1e3), "up_overshoot_pct": num(a.up.overshoot, 100),
            "down": a.down.status, "down_rise_ms": num(a.down.rise_s, 1e3),
            "down_overshoot_pct": num(a.down.overshoot, 100), "ringdown_limited": "yes" if a.ringdown_limited else "",
            "drive_floor_pct": num(a.floor_frac, 100), "noise": num(a.noise), "recover_s": num(rc.get("recover_s")),
            "settle_s": num(rc.get("settle_s")), "base": num(p.base), "low": num(p.low), "high": num(p.high),
            "hold_s": num(p.hold_s), "events": p.n_events, "input_gain": num(sx.get("input_gain_ina")),
            "output_gain_v": num(sx.get("afl_output_gain")), "amp_tau_ms": num(sx.get("amp_tau_s"), 1e3),
            "dnc_tc_ms": num(sx.get("dnc_time_constant_s"), 1e3), "q": num(sx.get("q")),
            "notes": " | ".join(([r.result.failure] if r.result.failure else []) + a.reasons + list(r.result.warnings)),
        }

    def default_export_name(self) -> str:
        meta = self.tests[-1].meta or MD.Metadata.unavailable("none")
        loops = sorted({r.assessment.loop for r in self.tests})
        order = tuple(loops) + tuple(x for x in ("afl", "pll") if x not in loops)
        return meta.filename("tuning", "-".join(lp.upper() for lp in loops) + f"-{len(self.tests)}tests", loops=order) + ".csv"

    def export_results_to(self, path: str) -> str:
        """Write every test to ``path`` (CSV) and the matching .json. Returns the JSON path. Raises OSError."""
        import csv
        meta = self.tests[-1].meta or MD.collect(self.reader)
        header = MD.Metadata(meta.values, meta.errors, meta.timestamp, meta.source)
        to = self.trade()
        rows = [("Tests", len(self.tests), ""), ("Settings above", "as read when the last run started", "")]
        if to.fastest is not None:
            rows += [("Fastest (with margin)", f"Kp {_fmt(to.fastest.kp)}, Ki {_fmt(to.fastest.ki)}, "
                                               f"{_ms(to.fastest.speed_s)}", ""),
                     ("Quietest within 1.5x", f"Kp {_fmt(to.quietest.kp)}, Ki {_fmt(to.quietest.ki)}, "
                                              f"{_ms(to.quietest.speed_s)}", "")]
        header.add_section("Tuning session", rows)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(MD.csv_preamble(header.header_lines("SXM nc-AFM tuning results"), numeric=False))
            w = csv.writer(f)
            w.writerow([t for _k, t in self.EXPORT_COLUMNS])
            for i, r in enumerate(self.tests):
                row = self._export_row(i, r)
                w.writerow([row[k] for k, _t in self.EXPORT_COLUMNS])
        tests = []
        for i, r in enumerate(self.tests):
            tests.append({**self._export_row(i, r), "why": r.proposal.why,
                          "baseline_check": r.result.meta.get("baseline_check"),
                          "sxm": r.meta.to_dict()["groups"] if r.meta is not None else None})
        return MD.write_sidecar(os.path.splitext(path)[0] + ".json", header, {"kind": "ncafm_tuning_results", "tests": tests})

    def _export_results(self):
        if not self.tests:
            return
        name = self.default_export_name()
        start = os.path.join(self._last_export_dir, name) if self._last_export_dir else name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export tuning results", start, "CSV (*.csv)")
        if not path:
            return
        self._last_export_dir = os.path.dirname(path)
        try:
            json_path = self.export_results_to(path)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Export results", f"Could not write:\n{e}")
            return
        self._log(f"Exported {len(self.tests)} tests to {path} (+ {os.path.basename(json_path)})")

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


def guide_html() -> str:
    rows = "".join(f"<tr><td bgcolor='{_rgb(s)}'><b>{X.STATUS_LABEL[s]}</b></td><td>{t}</td></tr>" for s, t in (
        ("clean", "Follows the step within the overshoot limit, no ringing."),
        ("slow", "Did not get to the new value within the hold: too little gain (or lengthen the hold in Advanced)."),
        ("overshoot", "Goes past the new value by more than the limit (Advanced). Usually too much Ki for the Kp."),
        ("floor", "Amplitude loop, down-steps: Drive sat at zero. The loop switched Drive off instead of regulating; "
                  "the amplitude then falls only at the ring-down rate. Usually too much gain."),
        ("ringing", "Oscillates, or repeated steps differ a lot."),
        ("lost", "The loop lost the amplitude / the lock, did not settle, or the step could not be measured.")))
    return f"""
<h3>What this tab does</h3>
<p>It measures the loop at many (Kp, Ki) pairs and shows how it behaves: up-steps and down-steps separately, as
speed against noise. Only Kp, Ki and the stepped setpoint (amplitude Ref or DNC use) are written; the baseline (the
gains in SXM) is written back between conditions and at the end.</p>
<h3>Using it</h3>
<ol>
<li><b>Loop</b>: amplitude loop or PLL. The instrument values are read from SXM; fix anything marked ✖ in SXM and
press <i>Read SXM again</i>.</li>
<li><b>Confirm</b> the two items SXM cannot report.</li>
<li><b>Explore automatically</b>, or click any point of the map and <b>Test selected point</b>.</li>
<li>Read the <b>recommendation</b>, inspect points by clicking them, then <b>Stage selected in Params tab</b>.</li>
</ol>
<h3>Every condition starts from the same baseline</h3>
<p>Before each condition the baseline gains are written back and the tab waits until the loop is <i>verifiably</i>
back at the reference state measured at the start of the run (amplitude within 2 % and Drive back; or Phase within
0.5° and df back; Drive and df are compared with the previous recovery, so slow drift does not stop the run). Only then are the candidate gains applied; they must settle before the steps start. A loop that does not come
back stops the run: later results would not be comparable.</p>
<h3>The automatic exploration</h3>
<ol>
<li><b>Grid in decades</b> of Kp and Ki around the centre, including <b>Kp = 0</b> (integral only). Nothing is assumed
about the right Ki:Kp ratio. Points more aggressive than one where the loop was lost are skipped.</li>
<li><b>Edge along g</b>: for the best Ki:Kp ratios, both gains are scaled together (g) and the edge between clean and
not clean is found by <b>bisection</b> (a few tests to within x1.25).</li>
<li><b>Ratio search</b> at the best point: Ki x10<sup>±0.5</sup>, then x10<sup>±0.25</sup>, at fixed Kp.</li>
<li><b>Repeats</b> of the candidates: one test is noisy.</li>
</ol>
<h3>Results</h3>
<table border='1' cellspacing='0' cellpadding='4'>{rows}</table>
<p>On the map each point is <b>▲</b> (up-steps) over <b>▼</b> (down-steps); stars mark the recommendations. A point is
as good as its worse direction. The <b>fastest</b> recommendation keeps a margin to the edge (Advanced, default x1.5):
a point right at the edge works in a test and fails on the first disturbance. <b>Quietest</b> is the lowest-noise clean
point within 1.5x of that speed. <i>Along g</i> and <i>Along ratio</i> show rise times of the selected point's
neighbours; <i>Trade-off</i> shows all clean points.</p>
<p>Amplitude loop: a down-step cannot be faster than the sensor's ring-down (10-90 % in ln 9 &times; Q/(π f0)). When
it falls at that rate the result says so: it is the sensor, not the gains.</p>
"""
