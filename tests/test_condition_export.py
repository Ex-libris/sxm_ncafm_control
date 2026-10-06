"""gui/condition_export.py, offscreen: one run-sheet condition written as a Step Test export (CSV + JSON + PNG)."""
import datetime
import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                                      # noqa: E402
from PyQt5 import QtGui, QtWidgets                                      # noqa: E402

from sxm_ncafm_control import metadata as MD                            # noqa: E402
from sxm_ncafm_control.gui import condition_export as CX               # noqa: E402
from sxm_ncafm_control.gui.sheet_runner import Outcome                  # noqa: E402
from sxm_ncafm_control.tuning import runsheet as R                      # noqa: E402
from sxm_ncafm_control.tuning import workflow as W                      # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def outcome(loop):
    plan = W.StepTestPlan(loop=loop, base=1.0 if loop == "afl" else 25564.0, step=0.05 if loop == "afl" else 1.0,
                          hold_s=0.5, n_events=4, settle_s=0.5)
    t = np.arange(0, 6, 0.0005)
    data = {n: np.random.randn(len(t)) for n in ("Phase", "df", "QPlusAmpl", "Drive")}
    out = Outcome(4, R.Condition(g=2.0, rho=1.0, group="G ramp"), kp=2e6, ki=200, status="clean",
                  time="2026-10-06 10:00:00")
    out.raw, out.plan = (t, data), plan
    out.events = [1.0 + k * plan.hold_s for k in range(plan.n_events)]
    out.marks = {"settle": 0.0, "test": 1.0, "test_end": 3.5, "quiet": 4.0, "end": 6.0}
    return out


class SaveCondition(unittest.TestCase):
    def test_afl_condition_in_the_step_test_layout(self):
        meta = MD.Metadata({"amp_ref": 1.0, "amp_kp": 2e6, "amp_ki": 200.0, "pll_kp": -200.0, "pll_ki": -1.65e4},
                           timestamp=datetime.datetime(2026, 10, 6, 10, 0, 30))
        with tempfile.TemporaryDirectory() as d:
            written = CX.save_condition(d, outcome("afl"), "afl", meta, [("G", 2.0, "")], title="#5")
            self.assertEqual([os.path.splitext(p)[1] for p in written], [".csv", ".json", ".png"])
            name = os.path.basename(written[0])
            self.assertTrue(name.startswith("20261006-100030_steptest_AmpRef0.95-1.05_QPlusAmpl-Drive_cond05_AFL-Ref1"),
                            name)
            with open(written[0], encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual([ln for ln in lines if not ln.startswith("#")][0].strip(),
                             "time_s,QPlusAmpl_V,Drive_V,Phase_deg,df_Hz")
            text = "".join(lines)
            for section in ("[Capture]", "[Step Test events]", "[Step Test]", "[Run sheet condition]"):
                self.assertIn(section, text)
            self.assertIn("Amplitude Ref=1.05", text)
            with open(written[1], encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["kind"], "ncafm_runsheet_steptest")
            self.assertEqual(payload["step_test"]["steps_sent"], 4)
            img = QtGui.QImage(written[2])
            self.assertEqual(img.width(), CX.PNG_SIZE[0])
            self.assertGreater(img.height(), CX.PNG_SIZE[1])           # legend below the plots

    def test_pll_condition_plots_df_and_phase(self):
        with tempfile.TemporaryDirectory() as d:
            written = CX.save_condition(d, outcome("pll"), "pll", MD.Metadata.unavailable("test"), [])
            self.assertIn("_steptest_f0", os.path.basename(written[0]))
            self.assertIn("_df-Phase_cond05", os.path.basename(written[0]))
            with open(written[0], encoding="utf-8") as f:
                head = [ln for ln in f if not ln.startswith("#")][0]
            self.assertTrue(head.startswith("time_s,df_Hz,Phase_deg,"), head)

    def test_envelope_keeps_the_extremes(self):
        t = np.arange(100000.0)
        y = np.zeros_like(t)
        y[12345] = 5.0
        tt, yy = CX._envelope(t, y, columns=100)
        self.assertEqual(len(tt), 200)
        self.assertEqual(yy.max(), 5.0)


if __name__ == "__main__":
    unittest.main()
