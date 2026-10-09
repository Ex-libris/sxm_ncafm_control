"""
One run-sheet condition saved the way a Step Test capture is exported from the Scope tab:

* ``<stem>.csv``  - ``#`` settings block, then ``time_s`` and every recorded channel with its unit
  (``QPlusAmpl_V,Drive_V,...``; the loop's two channels first);
* ``<stem>.json`` - the same settings, the columns, the step events and the phase marks;
* ``<stem>.png``  - the loop's two channels against time with the step events (dashed) and the phases
  (dotted), and the settings legend underneath.

``<stem>`` follows the Step Test name: ``<date-time>_steptest_<param><low>-<high>_<chan1>-<chan2>_cond<NN>_<key
settings>``. Needs Qt (the PNG is drawn with pyqtgraph, offscreen); no hardware.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from .. import metadata as MD
from sxm_anfatec.driver import CHANNELS
from .export_image import save_scene_png

# the stepped parameter per loop: label, code and file-name tag as the Step Test tab writes them
STEPPED = {"afl": ("Amplitude Ref", "Edit23", "AmpRef"), "pll": ("Used Frequency (f0)", "DNC 3", "f0")}
# plotted channels, in the order the Step Test examples use (amplitude: QPlusAmpl / Drive; PLL: df / Phase)
PLOTTED = {"afl": ("QPlusAmpl", "Drive"), "pll": ("df", "Phase")}
PENS = ((50, 100, 200), (200, 50, 50))                  # as the Scope tab
PHASE_LABEL = {"settle": "settle", "test": "steps", "test_end": "back at base", "quiet": "quiet", "end": "end"}
PNG_SIZE = (1200, 700)                                  # plot area, px (legend added below)
MAX_PLOT_COLUMNS = 4000                                 # min/max envelope above this many samples


def channel_unit(name: str) -> str:
    unit = CHANNELS[name][2] if name in CHANNELS else ""
    return {"°": "deg", "*": "deg"}.get(unit, unit)


def _envelope(t: np.ndarray, y: np.ndarray, columns: int = MAX_PLOT_COLUMNS) -> Tuple[np.ndarray, np.ndarray]:
    """``(t, y)`` reduced to a min/max pair per column, so spikes survive in the image."""
    n = len(t)
    if n <= 2 * columns:
        return t, y
    per = n // columns
    m = per * columns
    tb = t[:m].reshape(columns, per)
    yb = y[:m].reshape(columns, per)
    tt = np.repeat(tb.mean(axis=1), 2)
    yy = np.column_stack((yb.min(axis=1), yb.max(axis=1))).ravel()
    return tt, yy


def channel_order(loop: str, names: Sequence[str]) -> List[str]:
    first = [n for n in PLOTTED[loop] if n in names]
    return first + [n for n in names if n not in first]


def step_rows(loop: str, plan, n_sent: int) -> List[Tuple[str, object, str]]:
    label, code, _tag = STEPPED[loop]
    unit = "Hz" if loop == "pll" else ""
    return [("Parameter", f"{label} ({code})", ""), ("Low", plan.low, unit), ("High", plan.high, unit),
            ("Hold per level", plan.hold_s, "s"),
            ("Steps", f"{n_sent} of {plan.n_events}" + ("" if n_sent >= plan.n_events else " (stopped early)"), ""),
            ("Return to base", "yes", ""), ("Base", plan.base, unit)]


def event_labels(loop: str, plan, n: int) -> List[str]:
    label = STEPPED[loop][0]
    return [f"{label}={MD.short_value(v) if loop == 'afl' else f'{v:.2f}'}" for _t, v in plan.commands()[:n]]


def export_name(meta: MD.Metadata, loop: str, plan, index: int) -> str:
    """``<date-time>_steptest_<param><low>-<high>_<chan1>-<chan2>_cond<NN>_<key settings>`` (no extension)."""
    tag = STEPPED[loop][2]
    a, b = PLOTTED[loop]
    if plan is None:
        detail = f"{tag}_{a}-{b}_cond{index + 1:02d}"
    else:
        detail = f"{tag}{MD.short_value(plan.low)}-{MD.short_value(plan.high)}_{a}-{b}_cond{index + 1:02d}"
    other = "pll" if loop == "afl" else "afl"
    return meta.filename("steptest", detail, loops=(loop, other))


def save_condition(folder: str, out, loop: str, meta: MD.Metadata, condition_rows: Sequence[Tuple[str, object, str]],
                   offline: bool = False, title: str = "") -> List[str]:
    """
    Write one measured condition (``sheet_runner.Outcome`` with its full recording in ``out.raw``) to ``folder``.
    ``meta`` is a fresh copy (sections are added to it). Returns the paths written; a PNG that cannot be drawn is
    reported in the list instead of raising.
    """
    t, data = out.raw
    plan = out.plan
    names = channel_order(loop, list(data))
    n = len(t)
    rate = (n - 1) / (t[-1] - t[0]) if n > 1 and t[-1] > t[0] else 0.0
    stem = os.path.join(folder, export_name(meta, loop, plan, out.index))

    capture = [("Channels", " / ".join(f"{c} [{channel_unit(c)}]" for c in names), ""),
               ("Samples", n, ""), ("Rate (averaged slices)", rate, "Hz"),
               ("Duration", float(t[-1] - t[0]) if n > 1 else 0.0, "s")]
    if out.time:
        capture.append(("Started", out.time, ""))
    if offline:
        capture.append(("Data", "OFFLINE (no driver)", ""))
    meta.add_section("Capture", capture)
    labels = event_labels(loop, plan, len(out.events)) if plan is not None else []
    if out.events:
        meta.add_section("Step Test events", [(f"t = {e:.3f} s", lbl, "") for e, lbl in zip(out.events, labels)])
    if plan is not None:
        meta.add_section("Step Test", step_rows(loop, plan, len(out.events)))
    meta.add_section("Run sheet condition", list(condition_rows))

    columns = ["time_s"] + [f"{c}_{channel_unit(c)}" if channel_unit(c) else c for c in names]
    written = [stem + ".csv"]
    with open(stem + ".csv", "w", encoding="utf-8", newline="") as f:
        f.write(MD.csv_preamble(meta.header_lines("SXM nc-AFM run sheet step test")))
        f.write(",".join(columns) + "\n")
        if n:
            np.savetxt(f, np.column_stack([t] + [data[c] for c in names]), delimiter=",")
    written.append(MD.write_sidecar(stem + ".json", meta, {
        "kind": "ncafm_runsheet_steptest", "data_file": os.path.basename(stem + ".csv"), "columns": columns,
        "samples": int(n), "rate_hz": rate, "loop": loop,
        "step_test": None if plan is None else {
            "parameter": STEPPED[loop][0], "code": STEPPED[loop][1], "low": plan.low, "high": plan.high,
            "base": plan.base, "hold_s": plan.hold_s, "steps": plan.n_events, "steps_sent": len(out.events)},
        "events_s": list(out.events), "event_labels": labels, "marks_s": dict(out.marks)}))
    try:
        save_plot_png(stem + ".png", t, data, loop, out.events, labels, out.marks, meta, title)
        written.append(stem + ".png")
    except Exception as e:                  # the data is on disk; a missing image must not stop the run
        written.append(f"(plot image not saved: {e})")
    return written


def save_plot_png(path: str, t: np.ndarray, data: Dict[str, np.ndarray], loop: str, events: Sequence[float],
                  labels: Sequence[str], marks: Dict[str, float], meta: MD.Metadata, title: str = "") -> None:
    """The loop's two channels as the Scope tab plots a Step Test, drawn offscreen, with ``meta``'s legend."""
    win = pg.GraphicsLayoutWidget()
    try:
        win.setBackground("w")
        win.resize(*PNG_SIZE)
        plots = []
        shown = [c for c in PLOTTED[loop] if c in data]
        for row, (name, color) in enumerate(zip(shown, PENS)):
            p = win.addPlot(row=row, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            unit = channel_unit(name)
            p.setLabel("left", f"{name} ({unit})" if unit else name)
            for ax in ("left", "bottom"):
                p.getAxis(ax).setPen(pg.mkPen(60, 60, 60))
                p.getAxis(ax).setTextPen(pg.mkPen(30, 30, 30))
            if len(t):
                p.plot(*_envelope(t, data[name]), pen=pg.mkPen(color=color, width=2))
            for x, lbl in zip(events, labels):
                # the value only ('1.05'): full labels overlap at the hold times used; the CSV has them in full
                line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen("r", width=1, style=QtCore.Qt.DashLine),
                                       label=lbl.split("=")[-1] if row == 0 else None,
                                       labelOpts={"position": 0.97, "color": (220, 60, 60),
                                                  "anchors": [(0, 0), (0, 0)]})
                p.addItem(line)
            for key, x in marks.items():
                line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen((120, 120, 120), width=1,
                                                                      style=QtCore.Qt.DotLine),
                                       label=PHASE_LABEL.get(key, key) if row == 0 else None,
                                       labelOpts={"position": 0.05, "color": (100, 100, 100),
                                                  "anchors": [(0, 1), (0, 1)]})
                p.addItem(line)
            if plots:
                p.setXLink(plots[0])
            plots.append(p)
        if plots:
            plots[-1].setLabel("bottom", "Time (s)")
            if title:
                plots[0].setTitle(title, color=(30, 30, 30))
        # never shown, so the view has no size: lay the plots out on a fixed canvas and export that item
        win.ci.setGeometry(QtCore.QRectF(0, 0, PNG_SIZE[0], PNG_SIZE[1]))
        save_scene_png(win.ci, path, width=PNG_SIZE[0], legend=meta.legend())
    finally:
        win.deleteLater()
