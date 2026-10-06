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

from sxm_ncafm_control.dde_client import MockDDEClient                  # noqa: E402
from sxm_ncafm_control.gui import runsheet_tab as T                     # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeAFLInstrument   # noqa: E402

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
        tab2.raw_check.setChecked(True)
        self.assertIn("folder", tab2._blocked())

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
            recordings = [f for f in os.listdir(d) if f.endswith(".csv")]
            self.assertEqual(len(recordings), 2)
            with open(os.path.join(d, recordings[0]), encoding="utf-8") as f:
                head = [ln for ln in f if not ln.startswith("#")][0]
            self.assertTrue(head.startswith("time_s,QPlusAmpl,Drive,df,Phase"))
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


if __name__ == "__main__":
    unittest.main()
