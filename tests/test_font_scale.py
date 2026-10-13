"""
The font the interface starts with: a scale picked from the screen's pixel density until the user chooses one
(gui/hidpi.suggested_font_scale, AccessibilityManager), a 9 pt floor, and plot axis labels that follow it.
Offscreen Qt; the settings file is a temporary one.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtGui, QtWidgets                                      # noqa: E402
import pyqtgraph as pg                                                  # noqa: E402  (after PyQt5: it binds to the Qt already imported)

from sxm_ncafm_control.gui import hidpi                                 # noqa: E402
from sxm_ncafm_control.gui.gui_accessibility_manager import AccessibilityManager   # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
KEEP = []        # plot widgets stay alive: a half-deleted pyqtgraph scene crashes later tests' event loops


class Screen:
    def __init__(self, px_per_inch):
        self.px_per_inch = px_per_inch

    def physicalDotsPerInch(self):
        return self.px_per_inch


class SuggestedScale(unittest.TestCase):
    def test_follows_the_pixel_density(self):
        for density, scale in ((87, 1.0), (96, 1.0), (109, 1.25), (137, 1.5), (163, 1.8), (220, 2.2)):
            self.assertEqual(hidpi.suggested_font_scale(screen=Screen(density)), scale, density)

    def test_an_unusable_panel_size_gives_one(self):
        for density in (0, 20, 900):
            self.assertEqual(hidpi.suggested_font_scale(screen=Screen(density)), 1.0)


class Manager(unittest.TestCase):
    def manager(self, saved=None, density=163):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        if saved is not None:
            with open(os.path.join(d.name, ".scientific_gui_accessibility.json"), "w") as f:
                json.dump(saved, f)
        with mock.patch("os.path.expanduser", return_value=d.name), \
                mock.patch.object(hidpi.QtGui.QGuiApplication, "primaryScreen", return_value=Screen(density)):
            return AccessibilityManager()

    def test_a_dense_screen_starts_larger(self):
        self.assertEqual(self.manager().settings["font_scale"], 1.8)
        self.assertEqual(self.manager(density=96).settings["font_scale"], 1.0)

    def test_a_settings_file_from_before_with_the_default_follows_the_screen(self):
        self.assertEqual(self.manager({"font_scale": 1.0, "dark_mode": False}).settings["font_scale"], 1.8)

    def test_a_chosen_scale_is_kept(self):
        self.assertEqual(self.manager({"font_scale": 1.25}).settings["font_scale"], 1.25)          # chosen before
        mgr = self.manager()
        mgr.set_font_scale("Normal")                                                              # chosen now
        self.assertFalse(mgr.settings["font_scale_auto"])
        with open(mgr.settings_file) as f:
            saved = json.load(f)
        self.assertEqual(self.manager(saved).settings["font_scale"], 1.0)

    def test_the_base_font_is_at_least_the_windows_9_pt(self):
        mgr = self.manager(density=96)
        f = QtGui.QFont()
        # app.font is patched, not set: setFont re-lays-out every widget earlier tests left behind
        with mock.patch.object(app, "font", return_value=f):
            f.setPointSizeF(7.8)                                    # what Qt reports on a 125 % display
            self.assertEqual(mgr.get_scaled_font().pointSizeF(), 9.0)
            f.setPointSizeF(11.0)
            self.assertEqual(mgr.get_scaled_font().pointSizeF(), 11.0)

    def test_plot_axis_labels_follow_the_font(self):
        mgr = self.manager({"font_scale": 1.5})
        glw = pg.GraphicsLayoutWidget()
        KEEP.append(glw)
        plot = glw.addPlot()
        plot.setLabel("left", "Drive (V)")
        mgr.apply_to_plotitem(plot)
        size = f"{mgr.get_scaled_font().pointSizeF():.1f}pt"
        axis = plot.getAxis("left")
        self.assertEqual(axis.labelStyle["font-size"], size)
        self.assertIn(f"font-size: {size}", axis.labelString())
        self.assertEqual(axis.style["tickFont"].pointSizeF(), mgr.get_scaled_font().pointSizeF())
        plot.setLabel("left", "QPlusAmpl (V)")                      # a tab relabelling the axis keeps the size
        self.assertIn(f"font-size: {size}", axis.labelString())
        self.assertIn("QPlusAmpl", axis.labelString())


if __name__ == "__main__":
    unittest.main()
