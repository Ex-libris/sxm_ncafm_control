"""
gui/runsheet_tab.py, offscreen: building a sheet from ramps, the checks that gate Run, a whole run on the fake
amplitude loop (table, plots, saved recordings, export). No hardware.
"""
import csv
import json
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets                                             # noqa: E402

from sxm_anfatec.dde import MockDDEClient                  # noqa: E402
from sxm_ncafm_control.gui import runsheet_tab as T                     # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeAFLInstrument   # noqa: E402
from sxm_ncafm_control.tuning import autotune as A                      # noqa: E402

SXM_AFL = {"f_peak": 25564.0, "q": 40000.0, "amp_ref": 1.0, "afl_output_gain": 0.1, "input_gain_ina": 1.0}

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def quick(tab):
    """Short trains, so a run takes seconds."""
    tab.hold_spin.setValue(0.6)
    tab.events_spin.setValue(4)
    tab.settle_spin.setValue(0.5)
    tab.quiet_spin.setValue(1.5)
    tab.quiet_settle_spin.setValue(0.3)
    tab.timeout_spin.setValue(10)


class Building(unittest.TestCase):
    def test_ramps_fill_the_sheet_and_the_table(self):
        tab = T.RunSheetTab(MockDDEClient(), None)
        self.assertEqual(tab.ramp_combo.currentData(), "g")
        self.assertEqual(tab.values_edit.text(), "0.1, 0.2, 0.3, 0.5, 1, 2, 3")
        self.assertIn("Adds 7 conditions", tab.ramp_preview.text())
        self.assertIn("Kp 1e5 … 3e6, Ki 10 … 300", tab.ramp_preview.text())     # G 0.1 ... 3 x (1e6, 100)
        tab.add_ramp()
        tab.ramp_combo.setCurrentIndex(tab.ramp_combo.findData("amp_tau_ms"))
        self.assertEqual(tab.values_edit.text(), "5, 10, 20, 50, 100")
        tab.fixed_g.setValue(0.5)
        tab.add_ramp()
        self.assertEqual(len(tab.sheet), 12)
        self.assertEqual(tab.table.rowCount(), 12)
        self.assertEqual(tab.table.item(7, T.COLUMNS.index("Settings")).text(), "AFL Tau 5 ms")
        self.assertEqual(tab.table.item(7, T.COLUMNS.index("Kp")).text(), "5e5")
        tab.values_edit.setText("fast")
        self.assertFalse(tab.btn_add.isEnabled())

    def test_run_is_gated(self):
        tab = T.RunSheetTab(MockDDEClient(), None)
        why = tab._blocked()
        self.assertIn("Offline", why)
        tab2 = T.RunSheetTab(FakeAFLInstrument(), FakeAFLInstrument())
        self.assertIn("empty", tab2._blocked())
        tab2.add_ramp()
        self.assertIn("retracted", tab2._blocked())
        tab2.retracted_check.setChecked(True)
        self.assertIsNone(tab2._blocked())
        self.assertTrue(tab2.btn_run.isEnabled())
        tab2.raw_check.setChecked(True)            # saving is on by default; no folder yet: Run asks, not blocks
        self.assertIsNone(tab2._blocked())
        self.assertIn("ask", " ".join(t for _k, t in tab2.checks()))

    def test_switching_loop_resets_the_anchor_and_clears_the_sheet(self):
        tab = T.RunSheetTab(MockDDEClient(), None)
        tab.add_ramp()
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("pll"))
        self.assertEqual(tab.sheet, [])
        self.assertEqual((tab.kp0.value(), tab.ki0.value()), (-200.0, -1.65e4))
        self.assertEqual(tab.values_edit.text(), "0.25, 0.4, 0.6, 1, 1.5, 2")


class Running(unittest.TestCase):
    def test_a_run_fills_the_table_plots_saves_and_exports(self):
        inst = FakeAFLInstrument()
        tab = T.RunSheetTab(inst, inst)
        tab.kp0.setValue(2e8)
        tab.ki0.setValue(2e4)
        quick(tab)
        tab.values_edit.setText("0.5")
        tab.add_ramp()
        tab.values_edit.setText("1")
        tab.add_ramp()
        tab.retracted_check.setChecked(True)
        with tempfile.TemporaryDirectory() as d:
            tab._raw_dir = d
            tab.raw_check.setChecked(True)
            tab.run()
            self.assertTrue(tab.runner_active())
            self.assertFalse(tab.btn_run.isEnabled())
            end = time.perf_counter() + 120
            while tab.runner_active() and time.perf_counter() < end:
                app.processEvents()
                time.sleep(0.005)
            self.assertFalse(tab.runner_active(), "run did not finish")
            self.assertEqual(sorted(tab.outcomes), [0, 1])
            for i in (0, 1):
                self.assertNotEqual(tab.table.item(i, T.COLUMNS.index("Result")).text(), "")
                self.assertTrue(tab.table.item(i, T.COLUMNS.index("Amp noise")).text().endswith("%"))
            files = os.listdir(d)
            recordings = sorted(f for f in files if "_steptest_AmpRef" in f and f.endswith(".csv"))
            self.assertEqual(len(recordings), 2, files)
            self.assertIn("_QPlusAmpl-Drive_cond01", recordings[0])
            for r in recordings:
                stem = r[:-4]
                self.assertIn(stem + ".json", files)
                self.assertIn(stem + ".png", files)
            with open(os.path.join(d, recordings[0]), encoding="utf-8") as f:
                lines = f.readlines()
            head = [ln for ln in lines if not ln.startswith("#")][0]
            self.assertTrue(head.startswith("time_s,QPlusAmpl_V,Drive_V,df_"), head)
            text = "".join(lines)
            self.assertIn("[Step Test]", text)
            self.assertIn("[Run sheet condition]", text)
            self.assertEqual(len([f for f in files if "_runsheet_AFL-2conditions" in f and f.endswith(".csv")]), 1)
            tab.table.selectRow(1)
            self.assertGreater(len(tab.p_rec.listDataItems()), 0)
            self.assertGreater(len(tab.p_asd.listDataItems()), 0)
            self.assertGreater(len(tab.p_speed.listDataItems()), 0)
            path = os.path.join(d, "results.csv")
            jp = tab.export_results_to(path)
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
            self.assertEqual([r["G"] for r in rows], ["0.5", "1.0"])
            self.assertEqual(rows[0]["kp"], "1e+08")
            with open(jp, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["kind"], "ncafm_runsheet_results")
            self.assertEqual(len(payload["sheet"]["conditions"]), 2)
            self.assertIn("QPlusAmpl", payload["results"][0]["quiet"]["channels"])
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])


class AutoSearch(unittest.TestCase):
    def test_it_is_gated_until_sxm_gave_the_sensor_and_the_tip_is_retracted(self):
        inst = FakeAFLInstrument()
        tab = T.RunSheetTab(inst, inst)
        self.assertIn("Read SXM", tab.auto_blocked())
        self.assertFalse(tab.btn_auto.isEnabled())
        self.assertIn("Not ready", tab.auto_why.text())               # the reason is shown, not only a tooltip
        # SXM read, but its DNC status bar was empty: f0 / Q can be typed
        tab.sxm = {k: v for k, v in SXM_AFL.items() if k not in ("f_peak", "q")}
        tab._auto_from_sxm()
        self.assertIn("f₀ and Q are unknown", tab.auto_blocked())
        tab.auto_f0.setValue(25564.0)
        tab.auto_q.setValue(40000.0)
        self.assertIn("retracted", tab.auto_blocked())
        tab.sxm = {**SXM_AFL, "amp_ref": 0.0}                       # no amplitude setpoint in SXM
        self.assertIn("set an amplitude setpoint", tab.auto_blocked())
        tab.sxm = dict(SXM_AFL)
        tab._auto_from_sxm()
        tab.retracted_check.setChecked(True)
        self.assertIsNone(tab.auto_blocked())
        tab._refresh_panels()
        self.assertTrue(tab.btn_auto.isEnabled())
        self.assertEqual(tab.auto_why.text(), "")
        self.assertIn("the manual rule is G 100", tab.auto_preview.text())
        cfg = tab.auto_config()
        self.assertEqual((cfg.start_factor, cfg.q, cfg.step_frac, cfg.include_tau), (0.01, 40000.0, 0.05, False))
        tab.auto_tau.setChecked(True)
        self.assertFalse(tab.auto_config().include_tau)           # Tau only with the automatic settings writer
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("pll"))
        self.assertIn("amplitude loop", tab.auto_blocked())

    def test_a_short_search_runs_its_stages_and_writes_the_summary(self):
        inst = FakeAFLInstrument()
        tab = T.RunSheetTab(inst, inst)
        quick(tab)
        tab.retracted_check.setChecked(True)
        # Q and output gain chosen so that the start (manual rule / 10) is the fake's comfortable 2e8 / 2e4
        cfg = A.AutoTuneConfig(q=25000.0, f0=25564.0, output_gain_v=0.1, base_ref=1.0, start_factor=0.1,
                               coarse_values=(1.0,), rho_values=(1.0,), edge_max_steps=1, repeats=1, ref_factors=(),
                               quiet_search_s=1.0, quiet_validate_s=1.0)
        with tempfile.TemporaryDirectory() as d:
            tab._raw_dir = d
            tab.start_auto(cfg, use_timing=False)
            self.assertTrue(tab.auto_active())
            self.assertFalse(tab.btn_auto.isEnabled())
            self.assertTrue(tab.btn_stop.isEnabled())
            self.assertEqual((tab.kp0.value(), tab.ki0.value()), (2e8, 2e4))
            self.assertIs(tab.runner.cfg.explore_limits, A.EXPLORE_LIMITS)      # judged from QPlusAmpl only
            end = time.perf_counter() + 300
            while tab.auto_active() and time.perf_counter() < end:
                app.processEvents()
                time.sleep(0.005)
            self.assertFalse(tab.auto_active(), "the search did not finish")
            files = os.listdir(d)
            summary = [f for f in files if f.endswith("_autotune_AFL_summary.json")]
            self.assertEqual(len(summary), 1, files)
            self.assertTrue(any(f.endswith("_autotune_AFL_summary.md") for f in files))
            with open(os.path.join(d, summary[0]), encoding="utf-8") as f:
                s = json.load(f)
            self.assertEqual(s["kind"], A.KIND)
            self.assertEqual(s["stages"][0]["key"], "coarse")
            self.assertEqual(tab.table.rowCount(), sum(len(st["conditions"]) for st in s["stages"]))
            self.assertTrue(s["recommendation"] is not None or s["failure"] or s["stopped"])
            if s["results_csv"]:
                self.assertIn(s["results_csv"], files)
            self.assertTrue(tab.btn_auto.isEnabled() or tab.auto_blocked() is not None)
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])   # back at the start

    def test_stop_ends_the_search_and_still_writes_the_summary(self):
        inst = FakeAFLInstrument()
        tab = T.RunSheetTab(inst, inst)
        quick(tab)
        cfg = A.AutoTuneConfig(q=25000.0, f0=25564.0, output_gain_v=0.1, start_factor=0.1, coarse_values=(1.0, 2.0),
                               quiet_search_s=1.0)
        with tempfile.TemporaryDirectory() as d:
            tab._raw_dir = d
            tab.start_auto(cfg, use_timing=False)
            end = time.perf_counter() + 30
            while not tab.outcomes and time.perf_counter() < end:
                app.processEvents()
                time.sleep(0.005)
            tab.stop()
            end = time.perf_counter() + 30
            while tab.auto_active() and time.perf_counter() < end:
                app.processEvents()
                time.sleep(0.005)
            self.assertFalse(tab.auto_active())
            with open(os.path.join(d, [f for f in os.listdir(d) if f.endswith("_summary.json")][0]),
                      encoding="utf-8") as f:
                s = json.load(f)
            self.assertEqual(s["stopped"], "stopped")
            self.assertIsNone(s["recommendation"])


if __name__ == "__main__":
    unittest.main()
