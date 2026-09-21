"""
High-DPI awareness (gui/hidpi.py). Offscreen Qt; a scaled display is simulated with QT_SCALE_FACTOR.
"""
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg                                                   # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets                                # noqa: E402

from sxm_ncafm_control.gui import hidpi                                  # noqa: E402
from sxm_ncafm_control.gui.gui_accessibility_manager import AccessibilityManager   # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

RENDER = r'''
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = sys.argv[1]
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg
from sxm_ncafm_control.gui import hidpi
hidpi.enable_high_dpi()                       # before the QApplication, as app.main() does
app = QtWidgets.QApplication([])

def thickness(width):
    w = pg.PlotWidget(); w.setBackground("w"); w.resize(400, 200)
    w.hideAxis("bottom"); w.hideAxis("left")
    w.setYRange(0, 1); w.setXRange(0, 1, padding=0)
    w.plot([0.0, 1.0], [0.5, 0.5], pen=pg.mkPen((0, 0, 0), width=width))
    w.show(); app.processEvents()
    img = w.grab().toImage(); col = img.width() // 2
    dark = sum(1 for y in range(img.height()) if QtGui.QColor(img.pixel(col, y)).red() < 128)
    return dark, img.width() / w.width()

before, dpr = thickness(2)
hidpi.install_pen_scaling()
after, _ = thickness(2)
print(json.dumps({"dpr": dpr, "before": before, "after": after}))
'''


def render(scale: str) -> dict:
    out = subprocess.run([sys.executable, "-c", RENDER, scale], capture_output=True, text=True, timeout=90,
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


class RenderedLineThickness(unittest.TestCase):
    """The measured symptom: pyqtgraph pens stay N *device* pixels, so plots get thin on a scaled display."""

    def test_without_the_fix_a_width_2_line_is_2_device_pixels_at_any_scale(self):
        for scale in ("1", "2"):
            self.assertEqual(render(scale)["before"], 2, scale)

    def test_with_the_fix_the_line_scales_with_the_display(self):
        # The offscreen platform reports 100 logical DPI, so the pass-through rounding policy yields x1.04 here
        # (a real 150 % Windows display gives exactly 1.5): compare with the measured ratio, not the nominal one.
        for scale in ("1", "1.5", "2"):
            r = render(scale)
            self.assertAlmostEqual(r["dpr"], float(scale), delta=0.1 * float(scale))
            self.assertLessEqual(abs(r["after"] - 2 * r["dpr"]), 1, (scale, r))       # width 2 -> 2 x the ratio
            self.assertGreater(r["after"], r["before"] if scale != "1" else 1)


class PenScaling(unittest.TestCase):
    def setUp(self):
        hidpi.uninstall_pen_scaling()
        patcher = mock.patch.dict(os.environ, {hidpi.ENV_LINE_SCALE: "2"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(hidpi.uninstall_pen_scaling)
        hidpi.install_pen_scaling()

    def test_widths_are_multiplied_and_the_rest_of_the_pen_is_kept(self):
        self.assertEqual(pg.mkPen((0, 0, 0), width=3).widthF(), 6.0)
        self.assertEqual(pg.mkPen("r").widthF(), 2.0)                        # the default width 1 is scaled too
        pen = pg.mkPen((10, 20, 30), width=2, style=QtCore.Qt.DashLine)
        self.assertEqual(pen.style(), QtCore.Qt.DashLine)
        self.assertEqual(pen.color().getRgb()[:3], (10, 20, 30))

    def test_finished_pens_none_and_non_cosmetic_pens_are_left_alone(self):
        self.assertEqual(pg.mkPen(QtGui.QPen(QtGui.QColor("r"), 5)).widthF(), 5.0)
        self.assertEqual(pg.mkPen(None).style(), QtCore.Qt.NoPen)
        self.assertEqual(pg.mkPen("r", width=3, cosmetic=False).widthF(), 3.0)

    def test_installing_twice_does_not_compound(self):
        hidpi.install_pen_scaling()
        hidpi.install_pen_scaling()
        self.assertEqual(pg.mkPen("r", width=3).widthF(), 6.0)
        self.assertIs(pg.mkPen, pg.functions.mkPen)

    def test_pyqtgraphs_own_pens_scale_too(self):
        axis = pg.PlotWidget().getAxis("bottom")
        self.assertEqual(axis.pen().widthF(), 2.0)


class ScaleDetection(unittest.TestCase):
    def test_environment_override(self):
        for raw, expected in (("3", 3.0), ("1.5", 1.5)):
            with mock.patch.dict(os.environ, {hidpi.ENV_LINE_SCALE: raw}):
                self.assertEqual(hidpi.pen_scale(), expected)
        for raw in ("", "abc", "-2", "0"):
            with mock.patch.dict(os.environ, {hidpi.ENV_LINE_SCALE: raw}):
                self.assertEqual(hidpi.pen_scale(), hidpi.device_pixel_ratio(), raw)

    def test_the_ratio_is_never_below_one(self):
        self.assertGreaterEqual(hidpi.device_pixel_ratio(), 1.0)

    def test_the_display_line_is_ascii_and_says_what_was_detected(self):
        line = hidpi.describe_display()
        self.assertTrue(line.isascii())                                      # a cp1252 console must not choke on it
        self.assertIn("scale x", line)
        self.assertIn("plot line widths", line)

    def test_the_window_fits_the_screen(self):
        size = hidpi.initial_window_size()
        avail = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.assertGreater(size.width(), 0)
        self.assertLessEqual(size.width(), avail.width())
        self.assertLessEqual(size.height(), avail.height())
        self.assertEqual(hidpi.initial_window_size(wanted=(10, 10)), QtCore.QSize(10, 10))


class FontSize(unittest.TestCase):
    def setUp(self):
        self.old = app.font()
        self.addCleanup(app.setFont, self.old)
        self.mgr = AccessibilityManager()
        self.mgr.settings["font_scale"] = 1.0

    def test_a_pixel_sized_application_font_no_longer_gives_a_6_pt_interface(self):
        f = QtGui.QFont()
        f.setPixelSize(16)
        self.assertEqual(f.pointSize(), -1)                                  # the case that used to give 6 pt
        app.setFont(f)
        pt = self.mgr.get_scaled_font().pointSizeF()
        self.assertGreater(pt, 8.0)
        self.assertLess(pt, 20.0)

    def test_point_sized_fonts_scale_with_the_accessibility_factor(self):
        f = QtGui.QFont()
        f.setPointSize(10)
        app.setFont(f)
        self.mgr.settings["font_scale"] = 1.5
        self.assertAlmostEqual(self.mgr.get_scaled_font().pointSizeF(), 15.0)
        self.assertAlmostEqual(self.mgr.get_scaled_font(base_size=8).pointSizeF(), 12.0)


if __name__ == "__main__":
    unittest.main()
