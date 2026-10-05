"""
gui/tuning_tab.py, offscreen: values from the SXM read-back, automatic checks, the two confirmations, the search
region, the map (selection), running conditions end to end against the real-time fake amplitude loop, the
recommendation and export.
"""
import json
import math
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets                                     # noqa: E402

from sxm_ncafm_control.gui import tuning_tab as T              # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeAFLInstrument, FakeInstrument   # noqa: E402
from sxm_ncafm_control.tests.test_sxm_state import GUI, reader   # noqa: E402
from sxm_ncafm_control.tuning import explore as X               # noqa: E402
from sxm_ncafm_control.tuning import workflow as W              # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(cond, timeout=60.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


class MockDDEClient:          # the name starts with "Mock": treated as offline
    pass


def typed_afl_tab(inst, **kw):
    """An amplitude-loop tab on the fake instrument, values typed (no SXM), fast method settings, confirmed."""
    tab = T.TuningTab(inst, inst, **kw)
    tab.LEAD_S, tab.TAIL_S = 0.3, 0.1
    tab.override_check.setChecked(True)
    tab.base_spin.setValue(1.0)
    tab.kp_spin.setValue(2e8)
    tab.ki_spin.setValue(2e4)
    tab.ina_combo.setCurrentIndex(tab.ina_combo.findData(10.0))
    tab.f0_spin.setValue(25000.0)
    tab.q_spin.setValue(25000.0)
    tab.hold_spin.setValue(0.6)
    tab.settle_spin.setValue(0.6)
    tab.events_spin.setValue(5)
    tab.window_spin.setValue(0.3)
    tab._center_on_sxm()
    for c in tab.confirms:
        c.setChecked(True)
    return tab


class ReadBack(unittest.TestCase):
    def test_instrument_values_come_from_sxm_and_are_not_editable(self):
        rd, _ = reader()
        tab = T.TuningTab(FakeInstrument(), FakeInstrument(), reader=rd)
        self.assertEqual(tab.loop, "afl")
        self.assertEqual(tab.baseline(), (5e8, 5e4))
        self.assertEqual(tab.base_spin.value(), 0.5)                              # amplitude Ref is stepped
        self.assertEqual(tab.gain_combo.currentData(), 0.1)
        self.assertEqual(tab.ina_combo.currentData(), 1.0)
        self.assertAlmostEqual(tab.tau_spin.value(), 2.0)
        self.assertEqual((tab.q_spin.value(), tab.f0_spin.value()), (148699.0, 25562.0))
        self.assertEqual((tab.kp_center.value(), tab.ki_center.value()), (5e8, 5e4))   # the search centres on SXM
        self.assertFalse(tab.kp_spin.isEnabled())
        self.assertIn("5e8", tab.instrument_label.text())
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("pll"))
        self.assertEqual(tab.baseline(), (-100.0, -1e4))
        self.assertAlmostEqual(tab.base_spin.value(), 25562.49)                  # the PLL steps DNC use

    def test_method_timing_is_derived_from_the_ring_down(self):
        rd, _ = reader()
        tab = T.TuningTab(FakeInstrument(), FakeInstrument(), reader=rd)
        ring = 148699.0 / (math.pi * 25562.0)                                     # 1.85 s
        self.assertAlmostEqual(tab.ring_down_s(), ring, places=3)
        self.assertEqual(tab.hold_spin.value(), W.afl_start_values(148699.0, 25562.0, 0.1).hold_s)
        self.assertGreaterEqual(tab.recover_spin.value(), 10 * ring - 1)

    def test_checks(self):
        rd, fake = reader()
        inst = FakeAFLInstrument()
        tab = T.TuningTab(inst, inst, reader=rd)
        texts = {t: lvl for lvl, t in tab.checks()}
        self.assertTrue(any("PLL is on" in t for t in texts))                     # the fixture has PLL -100 / -1e4
        self.assertTrue(any("Input gain x1" in t and lvl == "warn" for t, lvl in texts.items()))
        self.assertIn("PLL is on", tab._blocked())
        fake.gui["pll"] = {"Kp": 0.0, "Ki": 0.0}
        tab.refresh_from_sxm()
        self.assertTrue(any(t == "PLL off." for _, t in tab.checks()))
        self.assertIn("Confirm", tab._blocked())                                  # now only the two ticks are missing
        for c in tab.confirms:
            c.setChecked(True)
        self.assertIsNone(tab._blocked())
        self.assertTrue(tab.btn_explore.isEnabled())

    def test_without_sxm_the_values_must_be_typed(self):
        inst = FakeAFLInstrument()
        tab = T.TuningTab(inst, inst, reader=reader(broken=set(GUI))[0])
        for c in tab.confirms:
            c.setChecked(True)
        self.assertIn("SXM not readable", tab._blocked())
        tab.override_check.setChecked(True)
        self.assertTrue(tab.kp_spin.isEnabled())
        self.assertIsNone(tab._blocked())

    def test_offline_blocks(self):
        tab = T.TuningTab(MockDDEClient(), None)
        tab.override_check.setChecked(True)
        for c in tab.confirms:
            c.setChecked(True)
        self.assertIn("Offline", tab._blocked())
        self.assertFalse(tab.btn_explore.isEnabled())


class RegionAndMap(unittest.TestCase):
    def test_region_lists_the_real_values_and_a_time_estimate(self):
        tab = typed_afl_tab(FakeAFLInstrument())
        txt = tab.region_label.text()
        self.assertIn("Kp: 0, 2e7, 2e8, 2e9", txt)
        self.assertIn("Ki: 2000, 2e4, 2e5", txt)
        self.assertIn("min", txt)
        tab.span_spin.setValue(2)
        self.assertIn("2e10", tab.region_label.text())
        tab.btn_center_manual.click()
        self.assertEqual((tab.kp_center.value(), tab.ki_center.value()), (2e8, 2e4))   # Q = f0 = 25k at +-1 V

    def test_clicking_selects_the_nearest_point_or_the_spot(self):
        tab = typed_afl_tab(FakeAFLInstrument())
        tab.select_at(math.log10(2e8) + 0.1, math.log10(2e4) - 0.1)
        self.assertEqual(tab.selected, (2e8, 2e4))                                # snapped to the grid point
        tab.select_at(8.6, 3.6)
        self.assertAlmostEqual(tab.selected[0], 10 ** 8.5)                        # a free spot: 0.25-decade grid
        self.assertAlmostEqual(tab.selected[1], 10 ** 3.5)
        tab.select_at(tab._x0(), 4.0)
        self.assertEqual(tab.selected[0], 0.0)                                    # the Kp = 0 column
        self.assertIn("Not tested yet", tab.detail.toPlainText())

    def test_points_outside_the_allowed_change_are_refused(self):
        tab = typed_afl_tab(FakeAFLInstrument())
        tab.range_spin.setValue(10.0)
        self.assertIsNone(tab._gain_ok(2e9, 2e4))
        self.assertIn("more than x10", tab._gain_ok(2e10, 2e4))
        self.assertIsNone(tab._gain_ok(0.0, 2e4))                                 # Kp = 0 is always allowed
        self.assertIn("wrong sign", tab._gain_ok(-2e8, 2e4))


class Running(unittest.TestCase):
    def test_a_selected_point_is_measured_from_the_baseline_and_shown(self):
        inst = FakeAFLInstrument()
        tab = typed_afl_tab(inst)
        tab.select((4e8, 4e4))
        tab.test_selected()
        self.assertTrue(tab.runner_active())
        self.assertTrue(wait_until(lambda: not tab.runner_active(), 90), "test did not finish")
        self.assertEqual(len(tab.tests), 1)
        rec = tab.tests[0]
        self.assertEqual((rec.assessment.kp, rec.assessment.ki), (4e8, 4e4))
        self.assertIn("recover_s", rec.result.meta["baseline_check"])
        self.assertIsNotNone(tab.runner.reference)                                # measured at the start of the run
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])
        text = tab.detail.toPlainText()
        self.assertIn("Kp 4e8, Ki 4e4", text)
        self.assertIn("down", text)
        self.assertEqual(tab.table.rowCount(), 1)
        self.assertTrue(tab.btn_export.isEnabled())

    def test_exploration_stops_cleanly_and_does_not_retest(self):
        inst = FakeAFLInstrument()
        tab = typed_afl_tab(inst)
        tab.explore()
        self.assertTrue(wait_until(lambda: len(tab.tests) >= 2, 120), "exploration did not measure two points")
        tab.stop()
        self.assertFalse(tab.runner_active())
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])
        done = {X._key(r.assessment.kp, r.assessment.ki) for r in tab.tests}
        ex = X.Explorer(tab.region(), X.ExplorePlan())                           # how the next Explore plans
        for r in tab.tests:
            ex.add_manual(r.assessment)
        p = ex.next()
        self.assertNotIn(X._key(p.kp, p.ki), done)

    def test_export_writes_up_and_down_per_test(self):
        tab = typed_afl_tab(FakeAFLInstrument())
        res = W.StepTestResult(kp=2e8, ki=2e4, loop="afl", failure="lost: amplitude collapsed")
        res.meta["baseline_check"] = {"recover_s": 1.2, "settle_s": 0.7}
        tab.tests.append(T.TestRecord(X.Proposal(2e8, 2e4, "grid", "coarse grid"), res, X.assess(res), None,
                                      tab.plan(), "now"))
        tab._refresh_all()
        name = tab.default_export_name()
        self.assertIn("_tuning_AFL-1tests", name)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name)
            js = tab.export_results_to(path)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("# SXM nc-AFM tuning results", text)
            header = [ln for ln in text.splitlines() if not ln.startswith("#")][0]
            for col in ("up-steps", "down-steps", "up rise 10-90 % [ms]", "recovered to baseline [s]", "Drive at zero [%]"):
                self.assertIn(col, header)
            with open(js, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["tests"][0]["result"], "lost")
            self.assertEqual(payload["tests"][0]["baseline_check"]["recover_s"], 1.2)


class Guide(unittest.TestCase):
    def test_guide_explains_the_method(self):
        html = T.guide_html().lower()
        for word in ("baseline", "bisection", "kp = 0", "up-steps", "down-steps", "margin", "ring-down"):
            self.assertIn(word, html)


if __name__ == "__main__":
    unittest.main()
