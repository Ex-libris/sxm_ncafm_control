"""
metadata.py (grouped SXM settings for exports) and the exports that use it: the Scope tab (CSV / NPY + JSON +
captioned PNG), the Parameters tab's tune JSON, the calibration CSV. Offscreen Qt; no hardware.
"""
import datetime
import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                     # noqa: E402
from PyQt5 import QtGui, QtWidgets                     # noqa: E402

from sxm_ncafm_control import metadata as MD           # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

VALUES = {"amp_ref": 0.55, "amp_kp": 1e7, "amp_ki": 1200.0, "amp_tau_s": 0.05, "amp_pull_back": 30.0,
          "pll_kp": -100.0, "pll_ki": -1e4, "input_gain_ina": 10.0, "afl_output_gain": 0.1,
          "dnc_time_constant_s": 0.002, "dnc_rolloff": "12 db/oct", "used_freq": 25562.49, "drive": 0.02,
          "q": 148699.0, "f_peak": 25562.0, "ring_down_s": 1.851628, "feedback_off": False}
STAMP = datetime.datetime(2026, 10, 2, 14, 30, 12)


class Readout:
    def __init__(self, values, ok=True):
        self.values, self.errors, self.timestamp, self.ok = dict(values), {"topo_ref": "not readable"}, STAMP, ok


class Reader:
    def __init__(self, values=VALUES, fail=None):
        self.values, self.fail = values, fail

    def read(self):
        if self.fail:
            raise RuntimeError(self.fail)
        return Readout(self.values)


class Rendering(unittest.TestCase):
    def meta(self):
        return MD.collect(Reader(), [("Capture", [("Samples", 1000, ""), ("Rate (measured)", 150000.0, "Hz")])])

    def test_filename_tag_packs_the_key_settings(self):
        m = self.meta()
        self.assertEqual(m.filename_tag(), "AFL-Ref0.55-Kp1e7-Ki1200-Tau50ms_PLL-Kp-100-Ki-1e4_InA10_OG0.1V_TC2ms")
        self.assertTrue(m.filename_tag(("pll",)).startswith("PLL-Kp-100"))
        self.assertEqual(m.filename("scope", "QPlusAmpl-Drive").split("_")[:3], ["20261002-143012", "scope", "QPlusAmpl-Drive"])

    def test_missing_values_are_left_out_of_the_name_and_names_are_windows_safe(self):
        m = MD.Metadata({"amp_kp": 2e8, "amp_ki": 2e4})
        self.assertEqual(m.filename_tag(), "AFL-Kp2e8-Ki2e4")
        self.assertEqual(MD.safe_filename('a:b/c d*?"e'), "a_b_c_d_e")
        self.assertEqual(MD.Metadata.unavailable("x").filename_tag(), "")

    def test_header_groups_values_with_units_aligned(self):
        lines = self.meta().header_lines("Title")
        text = "\n".join(lines)
        self.assertEqual(lines[:2], ["Title", "====="])
        for group in ("[Amplitude feedback (AFL)]", "[PLL]", "[DNC (lock-in and excitation)]",
                      "[Resonance (last DNC sweep)]", "[Capture]"):
            self.assertIn(group, text)
        self.assertIn("Tau (AFL input filter)", text)
        self.assertRegex(text, r"Tau \(AFL input filter\) +50 ms")              # seconds shown as ms
        self.assertRegex(text, r"TimeConstant t +2 ms")
        self.assertRegex(text, r"Output gain +0\.1 \+-V")
        self.assertRegex(text, r"Ref \(amplitude setpoint\) +0\.55 SXM units")   # unit not reported: said so
        self.assertRegex(text, r"Q +148699\n")
        self.assertRegex(text, r"use \(PLL centre / excitation\) +25562\.49 Hz")   # mHz kept
        self.assertRegex(text, r"\n  Ref +n/a\n")                                # topography Ref: not read, said so
        rows = [ln for ln in lines if ln.startswith("  ")]
        width = max(len(ln) - len(ln.lstrip()) for ln in rows)                     # labels padded to one column
        self.assertTrue(all(ln[2:].find("  ") >= 0 for ln in rows))
        self.assertGreater(width, 0)

    def test_the_preamble_says_how_to_read_the_file_and_both_ways_work(self):
        import re
        m = self.meta()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write(MD.csv_preamble(m.header_lines("T")))
                f.write("time_s,a_V\n0,1\n1,2\n")
            with open(p, encoding="utf-8") as f:
                first = f.readline()
            skip = int(re.search(r"skiprows=(\d+)", first).group(1))
            arr = np.loadtxt(p, delimiter=",", skiprows=skip)
            self.assertEqual(arr.tolist(), [[0.0, 1.0], [1.0, 2.0]])
            import pandas as pd
            df = pd.read_csv(p, comment="#")
            self.assertEqual(list(df.columns), ["time_s", "a_V"])
            self.assertEqual(df["a_V"].tolist(), [1, 2])
        self.assertNotIn("numpy", MD.csv_preamble(["x"], numeric=False))

    def test_json_has_groups_raw_values_and_what_was_not_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = MD.write_sidecar(os.path.join(d, "x.json"), self.meta(), {"kind": "test"})
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
        self.assertEqual(payload["kind"], "test")
        self.assertEqual(payload["groups"]["Amplitude feedback (AFL)"]["Tau (AFL input filter)"], {"value": 50.0, "unit": "ms"})
        self.assertEqual(payload["sxm_values"]["amp_tau_s"], 0.05)
        self.assertEqual(payload["sxm_not_read"], {"topo_ref": "not readable"})
        self.assertEqual(payload["sxm_read_at"], "2026-10-02T14:30:12")

    def test_without_sxm_the_export_says_why(self):
        for reader, why in ((None, "no SXM read-back"), (Reader(fail="window gone"), "window gone")):
            m = MD.collect(reader)
            self.assertFalse(m.has_sxm)
            text = "\n".join(m.header_lines())
            self.assertIn(why, text)
            self.assertNotIn("[PLL]", text)

    def test_caption_is_a_few_dense_lines(self):
        cap = self.meta().caption_lines()
        self.assertLessEqual(len(cap), 6)
        self.assertTrue(cap[0].startswith("AFL: Ref 0.55"))
        self.assertIn("InA x10", cap[2])
        self.assertIn("Capture:", " ".join(cap))
        self.assertIn("2026-10-02 14:30:12", cap[-1])


class ScopeExport(unittest.TestCase):
    def test_csv_json_and_png_carry_the_settings(self):
        from sxm_ncafm_control.gui.scope_tab import ScopeTab
        tab = ScopeTab(None, reader=Reader())
        tab.chan1_combo.setCurrentText("QPlusAmpl")
        tab.chan2_combo.setCurrentText("Drive")
        tab.npoints_spin.setValue(tab.npoints_spin.minimum())
        tab.start_capture()                                                      # offline: mock signals
        self.assertIsNotNone(tab.last_meta)
        self.assertEqual(tab.last_meta.get("input_gain_ina"), 10.0)
        name = tab.default_export_name()
        self.assertIn("_scope_QPlusAmpl-Drive_AFL-Ref0.55", name)
        with tempfile.TemporaryDirectory() as d:
            written = tab.export_to(os.path.join(d, name + ".csv"))
            self.assertEqual([os.path.splitext(w)[1] for w in written], [".csv", ".json", ".png"])
            text = open(written[0], encoding="utf-8").read()
            self.assertIn("# SXM nc-AFM scope capture", text)
            self.assertIn("OFFLINE MOCK SIGNALS", text)
            import pandas as pd
            df = pd.read_csv(written[0], comment="#")
            self.assertEqual(list(df.columns), ["time_s", "QPlusAmpl_V", "Drive_V"])
            self.assertEqual(len(df), len(tab.last_data1))
            payload = json.load(open(written[1], encoding="utf-8"))
            self.assertEqual(payload["kind"], "ncafm_scope_capture")
            self.assertEqual(payload["columns"], ["time_s", "QPlusAmpl_V", "Drive_V"])
            img = QtGui.QImage(written[2])
            self.assertEqual(img.width(), 1200)
            npy = tab.export_to(os.path.join(d, "raw.npy"))
            self.assertEqual(np.load(npy[0]).shape, (len(tab.last_data1), 2))       # .npy layout unchanged
            self.assertEqual(json.load(open(npy[1], encoding="utf-8"))["columns"], ["QPlusAmpl_V", "Drive_V"])

    def test_step_test_events_are_listed(self):
        from sxm_ncafm_control.gui.scope_tab import ScopeTab
        tab = ScopeTab(None, reader=None)
        tab.npoints_spin.setValue(tab.npoints_spin.minimum())
        tab.start_capture()
        tab.set_event_markers([(tab.capture_start_dt.addMSecs(500), "Amplitude Ref=0.5775"),
                               (tab.capture_start_dt.addMSecs(1500), "Amplitude Ref=0.5225")])
        text = "\n".join(tab._export_metadata().header_lines())
        self.assertIn("[Step Test events]", text)
        self.assertRegex(text, r"t = 0\.500 s +Amplitude Ref=0\.5775")
        self.assertIn("no SXM read-back", text)

    def test_step_test_settings_and_name(self):
        from sxm_ncafm_control.gui.scope_tab import ScopeTab
        from sxm_ncafm_control.gui.step_test_tab import StepTestTab
        from sxm_anfatec.dde import MockDDEClient
        step = StepTestTab(MockDDEClient())
        step.param.setCurrentIndex(step.param.findText("Amplitude Ref"))
        step.low.setValue(0.5225); step.high.setValue(0.5775); step.period.setValue(0.5); step.steps.setValue(4)
        step.chk_restore.setChecked(True)
        tab = ScopeTab(None, reader=Reader())
        step.scope_tab = tab
        tab.npoints_spin.setValue(tab.npoints_spin.minimum())
        tab.start_capture()
        step._run_settings = step.settings()
        step.step_index = 2                                                      # stopped half-way
        tab.set_event_markers([(tab.capture_start_dt.addMSecs(500), "Amplitude Ref=0.5225")],
                              step_test=step._scope_settings())
        text = "\n".join(tab._export_metadata().header_lines())
        self.assertIn("[Step Test]", text)
        self.assertIn("Amplitude Ref (Edit23)", text)
        self.assertIn("2 of 4 (stopped early)", text)
        self.assertRegex(text, r"Base +0\.55")
        name = tab.default_export_name()
        self.assertIn("_steptest_AmpRef0.5225-0.5775_", name)
        with tempfile.TemporaryDirectory() as d:
            written = tab.export_to(os.path.join(d, name + ".csv"))
            payload = json.load(open(written[1], encoding="utf-8"))
            self.assertEqual(payload["step_test"]["steps_sent"], 2)
            self.assertFalse(payload["step_test"]["completed"])
            img = QtGui.QImage(written[2])
            self.assertGreater(img.height(), 0)
        tab.start_capture()                                                      # a plain capture afterwards
        self.assertIn("_scope_", tab.default_export_name())
        self.assertNotIn("Step Test", "\n".join(tab._export_metadata().header_lines()))

    def test_legend_shows_every_loop_and_marks_unread_values(self):
        values = dict(VALUES)
        del values["pll_ki"]
        cols, footer = MD.Metadata(values, timestamp=STAMP).add_section("Capture", [("Samples", 10, "")]).legend()
        titles = [t for t, _r in cols]
        self.assertEqual(titles, ["Amplitude (AFL)", "PLL", "DNC", "Resonance", "Capture"])
        d = {t: dict(r) for t, r in cols}
        self.assertEqual(d["PLL"]["Ki"], "n/a")
        self.assertEqual(d["DNC"]["use"], "25562.49 Hz")
        self.assertEqual(d["DNC"]["InA"], "x10")
        self.assertEqual(d["Amplitude (AFL)"]["Out gain"], "+-0.1 V")
        self.assertEqual(d["Amplitude (AFL)"]["Tau"], "50 ms")
        self.assertIn("2026-10-02 14:30:12", footer)


if __name__ == "__main__":
    unittest.main()
