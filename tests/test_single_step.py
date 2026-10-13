"""
The Single Step tab: its recorder (time slices for the whole record, full rate around the step) and the tab's
sequence and export. Offscreen Qt, real time (a few seconds); no hardware.
"""
import json
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                     # noqa: E402
from PyQt5 import QtWidgets                            # noqa: E402

from sxm_anfatec.driver import CHANNELS                # noqa: E402
from sxm_ncafm_control.gui.single_step_tab import SingleStepRecorder, SingleStepTab, _columns   # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class StepDriver:
    """read_raw gives ``level`` (raw counts) on every channel, and takes about as long as the driver (5 us)."""

    def __init__(self, level=1000):
        self.level = level

    def read_raw(self, idx):
        t_end = time.perf_counter() + 5e-6
        while time.perf_counter() < t_end:
            pass
        return self.level


class FakeDDE:
    def __init__(self):
        self.writes = []

    def send_scanpara(self, code, value):
        self.writes.append((code, value))

    def send_dncpara(self, index, value):
        self.writes.append((f"DNC{index}", value))


def wait_until(done, timeout_s=15.0):
    t_end = time.perf_counter() + timeout_s
    while not done() and time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)
    return done()


class Recorder(unittest.TestCase):
    def record(self, driver, edge_at=0.3, **kw):
        args = dict(bin_s=0.005, before_s=0.3, after_s=0.6, edge_pre_s=0.1, edge_post_s=0.2)
        args.update(kw)
        rec = SingleStepRecorder(driver, [("a", 1, 2.0), ("b", 2, -1.0)], **args)
        rec.start()
        if edge_at is not None:
            while rec.now() < edge_at:
                time.sleep(0.001)
            t = rec.now()
            driver.level = 3000
            rec.mark_edge(t)
        return rec

    def test_slices_cover_the_record_and_raw_covers_the_window(self):
        rec = self.record(StepDriver())
        self.assertTrue(rec.wait(5000))                       # ends by itself, after_s past the step
        n, edge = rec.count, rec.edge_t
        t = rec.t[:n]
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertLess(t[0], 0.05)
        self.assertAlmostEqual(t[-1] - edge, 0.6, delta=0.05)
        before, after = t < edge - 0.01, t > edge + 0.01
        np.testing.assert_allclose(rec.mean[0][:n][before], 2000.0)
        np.testing.assert_allclose(rec.mean[0][:n][after], 6000.0)
        # a negative scale: min and max stay in order
        np.testing.assert_allclose(rec.mean[1][:n][after], -3000.0)
        self.assertTrue(np.all(rec.lo[1][:n] <= rec.hi[1][:n]))
        # the full-rate window
        self.assertGreater(len(rec.raw_t), 1000)
        self.assertTrue(np.all(np.diff(rec.raw_t) > 0))
        self.assertAlmostEqual(rec.raw_t[0], edge - 0.1, delta=0.02)
        self.assertAlmostEqual(rec.raw_t[-1], edge + 0.2, delta=0.03)
        self.assertEqual(set(np.unique(rec.raw[0])), {2000.0, 6000.0})
        step = rec.raw_t[np.argmax(rec.raw[0] > 4000)]
        self.assertAlmostEqual(step, edge, delta=0.005)
        self.assertGreater(rec.rate, 1000)

    def test_without_a_step_there_is_no_raw_window(self):
        rec = self.record(StepDriver(), edge_at=None)
        time.sleep(0.2)
        rec.stop()
        self.assertTrue(rec.wait(2000))
        self.assertGreater(rec.count, 10)
        self.assertEqual(len(rec.raw_t), 0)

    def test_no_window_asked_for(self):
        rec = self.record(StepDriver(), edge_pre_s=0.0, edge_post_s=0.0)
        self.assertTrue(rec.wait(5000))
        self.assertEqual(len(rec.raw_t), 0)
        self.assertGreater(rec.count, 100)

    def test_a_read_error_is_reported(self):
        class Broken:
            def read_raw(self, idx):
                raise OSError("handle closed")
        got = []
        rec = SingleStepRecorder(Broken(), [("a", 1, 1.0)], 0.005, 0.3, 0.3, 0.1, 0.1)
        rec.error.connect(got.append)
        rec.start()
        self.assertTrue(rec.wait(2000))
        app.processEvents()
        self.assertIn("handle closed", got[0])


class Columns(unittest.TestCase):
    def test_spikes_survive(self):
        t = np.arange(10000.0)
        mean = np.zeros(10000)
        lo, hi = mean - 1, mean + 1
        hi[4321] = 50.0
        tt, m, l, h = _columns(t, mean, lo, hi, columns=100)
        self.assertEqual(len(tt), 100)
        self.assertEqual(h.max(), 50.0)
        self.assertEqual(l.min(), -1.0)


class Tab(unittest.TestCase):
    def tab(self, x0=0.5, x1=0.55, before=0.3, after=0.8):
        dde = FakeDDE()
        tab = SingleStepTab(dde, None, reader=None)
        tab.param.setCurrentIndex(tab.param.findText("Amplitude Ref"))
        tab.x0.setValue(x0); tab.x1.setValue(x1)
        tab.before.setValue(before); tab.after.setValue(after)
        tab.bin_ms.setValue(5.0)
        tab.edge_pre.setValue(0.1); tab.edge_post.setValue(0.3)
        return tab, dde

    def test_uneven_spans_nothing_written_before_the_step(self):
        tab, dde = self.tab()
        tab.start()
        self.assertTrue(tab.is_running())
        self.assertEqual(dde.writes, [])                       # 'already at X0'
        self.assertTrue(wait_until(lambda: not tab.is_running()))
        self.assertEqual(dde.writes, [("Edit23", 0.55)])
        res = tab.result
        self.assertAlmostEqual(-res["t"][0], 0.3, delta=0.08)
        self.assertAlmostEqual(res["t"][-1], 0.8, delta=0.08)
        self.assertGreater(len(res["raw_t"]), 100)
        self.assertLess(abs(np.mean(res["mean"][0][res["t"] < -0.05]) - 0.5), 0.01)      # mock signal follows
        self.assertLess(abs(np.mean(res["mean"][0][res["t"] > 0.6]) - 0.55), 0.01)
        self.assertTrue(tab.btn_export.isEnabled())

        name = tab.default_export_name()
        self.assertIn("_singlestep_AmpRef0.5-0.55_QPlusAmpl-Drive", name)
        with tempfile.TemporaryDirectory() as d:
            written = tab.export_to(os.path.join(d, name + ".csv"))
            self.assertEqual([w[len(os.path.join(d, name)):] for w in written],
                             [".csv", "_edge.csv", ".json", ".png"])
            with open(written[0], encoding="utf-8") as f:
                text = f.read()
            self.assertIn("[Single Step]", text)
            self.assertIn("no (as found)", text)
            self.assertIn("OFFLINE MOCK SIGNALS", text)
            import pandas as pd
            df = pd.read_csv(written[0], comment="#")
            self.assertEqual(list(df.columns), ["time_s", "QPlusAmpl_V_mean", "QPlusAmpl_V_min", "QPlusAmpl_V_max",
                                                "Drive_V_mean", "Drive_V_min", "Drive_V_max"])
            self.assertEqual(len(df), len(res["t"]))
            edge = pd.read_csv(written[1], comment="#")
            self.assertEqual(list(edge.columns), ["time_s", "QPlusAmpl_V", "Drive_V"])
            self.assertEqual(len(edge), len(res["raw_t"]))
            with open(written[2], encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["kind"], "ncafm_single_step")
            self.assertEqual(payload["single_step"]["x1"], 0.55)
            self.assertTrue(payload["single_step"]["step_sent"])
            self.assertEqual(payload["time_zero"], "the step")

    def test_x0_written_and_restored(self):
        tab, dde = self.tab(before=0.3, after=0.3)
        tab.chk_write_x0.setChecked(True)
        tab.chk_return.setChecked(True)
        tab.start()
        self.assertEqual(dde.writes, [("Edit23", 0.5)])
        self.assertTrue(wait_until(lambda: not tab.is_running()))
        self.assertEqual(dde.writes, [("Edit23", 0.5), ("Edit23", 0.55), ("Edit23", 0.5)])

    def test_stop_before_the_step_writes_nothing(self):
        tab, dde = self.tab(before=5.0)
        tab.chk_return.setChecked(True)
        tab.start()
        wait_until(lambda: tab._rec.count > 10)
        tab.stop()
        self.assertFalse(tab.is_running())
        self.assertEqual(dde.writes, [])
        self.assertIsNone(tab.result["edge_s"])
        self.assertIn("not sent", "\n".join(tab.last_meta.header_lines()))
        self.assertTrue(tab.btn_start.isEnabled())

    def test_stop_after_the_step_returns_to_x0(self):
        tab, dde = self.tab(before=0.3, after=30.0)
        tab.chk_return.setChecked(True)
        tab.start()
        self.assertTrue(wait_until(lambda: tab._run["step_sent"]))
        tab.stop()
        self.assertEqual(dde.writes, [("Edit23", 0.55), ("Edit23", 0.5)])

    def test_refused_when_x1_equals_x0(self):
        tab, dde = self.tab(x0=0.5, x1=0.5)
        tab.start()
        self.assertFalse(tab.is_running())
        self.assertEqual(dde.writes, [])
        self.assertIn("X1 equals X0", tab.log.toPlainText())

    def test_already_at_x0_takes_the_sxm_value(self):
        class Readout:
            ok = False                                         # nothing for the export metadata

            def by_code(self, ptype, pcode):
                return 0.42 if (ptype, pcode) == ("EDIT", "Edit23") else None

        class Reader:
            def read(self):
                return Readout()
        tab, dde = self.tab(x0=0.5, x1=0.55, before=5.0)
        tab.reader = Reader()
        tab.start()
        self.assertEqual(tab._run["x0"], 0.42)
        self.assertEqual(tab.x0.value(), 0.42)
        tab.stop()


if __name__ == "__main__":
    unittest.main()
