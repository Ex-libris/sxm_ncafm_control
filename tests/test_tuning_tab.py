"""
Tests for gui/tuning_tab.py, offscreen. The runner is exercised end to end against tests/fake_instrument.py,
a virtual PLL that runs in real time behind the same DDE/driver calls the real instrument gets.
"""
import math
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                     # noqa: E402
from PyQt5 import QtCore, QtWidgets                    # noqa: E402

from sxm_ncafm_control.gui import tuning_tab as T     # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeInstrument                 # noqa: E402
from sxm_ncafm_control.tests.fixtures import pll_plan, record_afl, record_pll, afl_plan   # noqa: E402
from sxm_ncafm_control.tuning import workflow as W     # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
FAST = dict(hold=0.15, events=3, settle=0.2)           # shortest legal test (~1.6 s each)


def wait_until(cond, timeout=25.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


class MockDDEClient:                                    # name starts with "Mock": the tab must treat it as offline
    pass


def make_tab(inst, **kw):
    tab = T.TuningTab(inst, inst, **kw)
    tab.LEAD_S, tab.TAIL_S = 0.4, 0.2
    tab.base_spin.setValue(25000.0)
    tab.step_spin.setValue(1.0)
    tab.hold_spin.setValue(FAST["hold"])
    tab.events_spin.setValue(FAST["events"])
    tab.settle_spin.setValue(FAST["settle"])
    tab.kp_spin.setValue(-100.0)
    tab.ki_spin.setValue(-1e4)
    for c in tab.checks:
        c.setChecked(True)
    # pre-confirm the baseline so a real (blocking) QMessageBox never pops up in these offscreen tests
    tab._confirmed_baseline[tab.loop_def.key] = (tab.kp_spin.value(), tab.ki_spin.value())
    return tab


class FakeScope:
    """The attributes of ScopeTab that the analysis reads."""

    def __init__(self, cap, plan, chan1="df", chan2="Phase", label="Used Frequency (f0)={:.10g}", events=True):
        self.last_data1, self.last_data2 = cap.u, cap.y
        self.last_chan1, self.last_chan2 = chan1, chan2
        self.last_rate = 1.0 / float(np.median(np.diff(cap.t)))
        self.capture_start_dt = QtCore.QDateTime.currentDateTime()
        self._event_markers = []
        if events:
            for t, v in plan.commands():
                self._event_markers.append((self.capture_start_dt.addMSecs(int(round(t * 1000))), label.format(v)))


class Offline(unittest.TestCase):
    def test_builds_and_run_buttons_are_disabled_without_the_instrument(self):
        tab = T.TuningTab(MockDDEClient(), None)
        for c in tab.checks:
            c.setChecked(True)
        self.assertFalse(tab.btn_single.isEnabled())
        self.assertFalse(tab.btn_map.isEnabled())
        self.assertIn("Offline", tab.btn_single.toolTip())

    def test_run_needs_the_checklist(self):
        inst = FakeInstrument()
        tab = T.TuningTab(inst, inst)
        self.assertFalse(tab.btn_single.isEnabled())
        for c in tab.checks:
            c.setChecked(True)
        self.assertTrue(tab.btn_single.isEnabled())
        tab.checks[0].setChecked(False)
        self.assertFalse(tab.btn_single.isEnabled())

    def test_switching_loop_loads_the_manuals_defaults(self):
        tab = T.TuningTab(FakeInstrument(), None)
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        self.assertAlmostEqual(tab.kp_spin.value(), 2e8)      # the manual start for Q = f0 = 25 k at +-1 V
        self.assertAlmostEqual(tab.ki_spin.value(), 2e4)
        self.assertEqual(tab.step_spin.value(), 10.0)
        self.assertIn("QPlusAmpl", tab.channels_label.text())
        self.assertEqual(tab.plan().step, 0.10)
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("pll"))
        self.assertEqual(tab.kp_spin.value(), -100.0)
        self.assertIn("Phase", tab.channels_label.text())

    def test_plan_grid_and_target_come_from_the_widgets(self):
        tab = T.TuningTab(FakeInstrument(), None)
        tab.line_spin.setValue(12.0)
        tab.px_spin.setValue(128)
        self.assertAlmostEqual(tab.target().rise_max, 0.5 * 12.0 / 128)
        g = tab.grid()
        self.assertEqual(g.dims, (5, 6))
        self.assertEqual(g.kp0, -100.0)
        tab.speed_lo.setValue(1)
        tab.shape_hi.setValue(0)
        self.assertEqual(tab.grid().dims, (4, 4))

    def test_gain_signs_are_validated(self):
        tab = T.TuningTab(FakeInstrument(), FakeInstrument())
        self.assertIsNone(tab._gains_valid())
        tab.kp_spin.setValue(100.0)
        self.assertIn("negative", tab._gains_valid())
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        self.assertIsNone(tab._gains_valid())
        tab.kp_spin.setValue(-5.0)
        self.assertIn("positive", tab._gains_valid())


class ScopeAdapter(unittest.TestCase):
    def test_pll_capture_with_step_test_events(self):
        plan = pll_plan(hold_s=0.5)
        ct0, cap = record_pll(plan, -100, -1e4, seed=2)
        ct, msg = T.capture_from_scope(FakeScope(cap, plan), -100, -1e4)
        self.assertIsNotNone(ct, msg)
        self.assertEqual(ct.plan.loop, "pll")
        self.assertAlmostEqual(ct.plan.base, 25000.0, places=3)
        self.assertAlmostEqual(ct.plan.step, 1.0, places=3)
        self.assertAlmostEqual(ct.plan.hold_s, 0.5, places=2)
        self.assertEqual(ct.plan.n_events, 7)
        res = W.analyze_test(ct)
        self.assertIsNone(res.failure)
        self.assertAlmostEqual(res.primary.rise_time, 0.0122, delta=0.004)

    def test_train_that_starts_on_the_high_level(self):
        plan = pll_plan(hold_s=0.5, start_high=True)
        _, cap = record_pll(plan, -100, -1e4, seed=2)
        ct, msg = T.capture_from_scope(FakeScope(cap, plan), -100, -1e4)
        self.assertTrue(ct.plan.start_high)
        self.assertIsNone(W.analyze_test(ct).failure)

    def test_amplitude_capture(self):
        plan = afl_plan()
        _, cap = record_afl(plan, 8.9e7, 8900)
        ct, msg = T.capture_from_scope(FakeScope(cap, plan, "QPlusAmpl", "Drive", label="Amplitude Ref (Edit23)={:.10g}"), 8.9e7, 8900)
        self.assertIsNotNone(ct, msg)
        self.assertEqual(ct.plan.loop, "afl")
        self.assertAlmostEqual(ct.plan.step, 0.10, places=3)                 # +-10 % of Ref
        self.assertIsNone(W.analyze_test(ct).failure)

    def test_refuses_with_a_clear_message(self):
        plan = pll_plan()
        _, cap = record_pll(plan, -100, -1e4)
        self.assertIn("no capture", T.capture_from_scope(type("S", (), dict(last_data1=None, last_data2=None, last_rate=None))(), 1, 1)[1].lower())
        self.assertIn("Step Test events", T.capture_from_scope(FakeScope(cap, plan, events=False), 1, 1)[1])
        ct, msg = T.capture_from_scope(FakeScope(cap, plan, "Topo", "Bias"), 1, 1)
        self.assertIsNone(ct)
        self.assertIn("Unrecognised", msg)

    def test_event_labels_are_parsed(self):
        self.assertEqual(T._parse_event_value("Used Frequency (f₀)=25000.5"), 25000.5)
        self.assertEqual(T._parse_event_value("Amplitude Ref=6.6 (base)"), 6.6)
        self.assertIsNone(T._parse_event_value("no number here"))

    def test_tab_analyzes_the_scope_capture_and_shows_advice(self):
        plan = pll_plan(hold_s=1.5)
        _, cap = record_pll(plan, -100, -1e4, seed=5)
        tab = T.TuningTab(FakeInstrument(), None, scope_tab=FakeScope(cap, plan))
        tab.line_spin.setValue(12.0)
        tab.px_spin.setValue(128)
        tab._analyze_scope()
        self.assertEqual(len(tab.singles), 1)
        html = tab.detail_text.toHtml()
        self.assertIn("good", html.lower())
        self.assertIn("Suggestions", html)
        self.assertIn("gains assumed", html)                                  # tells the user which gains it assumed
        self.assertEqual(tab.table.rowCount(), 1)


class RunnerEndToEnd(unittest.TestCase):
    def test_single_test_writes_the_train_and_restores_everything(self):
        inst = FakeInstrument()
        tab = make_tab(inst)
        reasons = []
        tab._start_runner(tab._one_shot(-100.0, -1e4))
        tab.runner.finished.connect(reasons.append)
        self.assertTrue(tab.runner_active())
        self.assertTrue(wait_until(lambda: reasons), "the test did not finish")
        self.assertEqual(reasons, ["done"])
        self.assertEqual(len(tab.singles), 1)
        res = tab.singles[0]
        self.assertEqual(res.n_steps, 3)
        self.assertIsNone(res.failure, res.failure)
        self.assertIsNotNone(res.primary)
        # the exact sequence of writes: gains, first level, the 3 toggles, then the baseline back
        w = inst.writes
        self.assertEqual(w[:3], [("Edit27", -100.0), ("Edit22", -1e4), ("DNC3", 24999.0)])
        self.assertEqual([v for c, v in w if c == "DNC3"][:5], [24999.0, 25001.0, 24999.0, 25001.0, 25000.0])
        self.assertEqual((inst.kp_raw, inst.ki_raw, inst.use), (-100.0, -1e4, 25000.0))
        self.assertFalse(tab.runner_active())

    def test_events_are_sent_on_schedule(self):
        inst = FakeInstrument()
        tab = make_tab(inst)
        stamps = []
        orig = inst.send_dncpara
        inst.send_dncpara = lambda i, v: (stamps.append((time.perf_counter(), v)), orig(i, v))[1]
        tab._start_runner(tab._one_shot(-100.0, -1e4))
        self.assertTrue(wait_until(lambda: not tab.runner_active()))
        toggles = [t for t, v in stamps if v in (25001.0, 24999.0)][1:]        # skip the pre-recording level write
        gaps = np.diff(toggles)
        self.assertEqual(len(gaps), 2)
        for g in gaps:
            self.assertAlmostEqual(g, FAST["hold"], delta=0.04)                # hold 0.15 s +- 40 ms of timer jitter

    def test_a_small_map_fills_in_paints_and_restores(self):
        inst = FakeInstrument()
        tab = make_tab(inst)
        tab.speed_lo.setValue(0)
        tab.speed_hi.setValue(0)
        tab.shape_lo.setValue(1)
        tab.shape_hi.setValue(0)                                                # 1 x 2 cells: Ki/2 and Ki
        done = []
        tab._run_map()
        tab.runner.finished.connect(done.append)
        self.assertTrue(wait_until(lambda: done, 40), "map did not finish")
        m = tab.current_map()
        self.assertEqual(len(m.results), 2)
        self.assertEqual(set(m.results), {(0, 0), (0, 1)})
        self.assertTrue(all(r.verdict is not None for r in m.results.values()))
        self.assertEqual(tab.table.rowCount(), 2)
        self.assertEqual((inst.kp_raw, inst.ki_raw, inst.use), (-100.0, -1e4, 25000.0))
        self.assertEqual(tab.map_img.image.shape, (2, 1, 4))
        self.assertIn("Kp", tab.detail_text.toPlainText())

    def test_stop_restores_the_baseline_mid_run(self):
        inst = FakeInstrument()
        tab = make_tab(inst)
        tab.speed_lo.setValue(1)
        tab.speed_hi.setValue(1)
        tab._run_map()
        reasons = []
        tab.runner.finished.connect(reasons.append)
        time.sleep(0.05)
        wait_until(lambda: False, 0.5)                                          # let it start a test (gains changed)
        tab.stop()
        self.assertEqual(reasons, ["stopped"])
        self.assertEqual((inst.kp_raw, inst.ki_raw, inst.use), (-100.0, -1e4, 25000.0))
        self.assertFalse(tab.runner_active())
        wait_until(lambda: False, 0.4)                                          # nothing may fire after the stop
        self.assertEqual((inst.kp_raw, inst.ki_raw, inst.use), (-100.0, -1e4, 25000.0))

    def test_runaway_aborts_the_test_and_restores_the_baseline(self):
        inst = FakeInstrument()
        plan = W.StepTestPlan(loop="pll", base=25000.0, step=1.0, hold_s=0.5, n_events=3, lead_s=1.0, tail_s=0.5, settle_s=0.3)
        runner = T.TuningRunner(inst, inst, plan, baseline=(-100.0, -1e4), analysis={})
        got, msgs, reasons = [], [], []
        runner.test_finished.connect(lambda c, r: got.append((c, r)))
        runner.message.connect(msgs.append)
        runner.finished.connect(reasons.append)
        runner.start(self._items([("bad", +100.0, +1e4)]))                       # positive gains = positive feedback
        self.assertTrue(wait_until(lambda: reasons, 20), "no abort")
        self.assertTrue(reasons[0].startswith("aborted"), reasons)
        self.assertEqual(len(got), 1)
        self.assertIn("aborted", got[0][1].failure)
        self.assertTrue(any("ABORT" in m for m in msgs))
        self.assertEqual((inst.kp_raw, inst.ki_raw, inst.use), (-100.0, -1e4, 25000.0))
        # the aborted cell counts as lost and blocks more aggressive cells in a map
        self.assertEqual(W.classify(got[0][1], W.Target(rise_max=0.05)).category, "lost")

    def test_dde_failure_aborts_cleanly(self):
        inst = FakeInstrument()

        def boom(*a, **k):
            raise RuntimeError("DDE down")
        plan = W.StepTestPlan(loop="pll", base=25000.0, hold_s=0.5, settle_s=0.2)
        runner = T.TuningRunner(inst, inst, plan, baseline=(-100.0, -1e4), analysis={})
        reasons = []
        runner.finished.connect(reasons.append)
        inst.send_scanpara = boom
        runner.start(self._items([("c", -100.0, -1e4)]))
        self.assertTrue(wait_until(lambda: reasons, 5))
        self.assertIn("DDE down", reasons[0])
        self.assertFalse(runner.running)

    @staticmethod
    def _items(items):
        q = list(items)
        return lambda: q.pop(0) if q else None


class MapInteraction(unittest.TestCase):
    """The map view, selection, advice and refinement with pre-recorded (fixture) results."""

    @classmethod
    def setUpClass(cls):
        cls.tab = T.TuningTab(FakeInstrument(), None)
        cls.tab.line_spin.setValue(12.0)
        cls.tab.px_spin.setValue(128)
        cls.tab.speed_lo.setValue(1)
        cls.tab.speed_hi.setValue(1)
        cls.tab.shape_lo.setValue(1)
        cls.tab.shape_hi.setValue(1)
        m = cls.tab._ensure_map()
        n = 0
        while (c := m.next_cell()) is not None:
            kp, ki = m.pair(c)
            ct, _ = record_pll(pll_plan(hold_s=1.5), kp, ki, seed=n + 1)
            m.record(c, W.analyze_test(ct, li_tau=0.002, f0=25000.0, q=25000.0))
            n += 1
        cls.tab._paint_map()

    def test_every_cell_is_painted_with_its_category_colour(self):
        m = self.tab.current_map()
        img = self.tab.map_img.image                                            # (ki index, kp index, rgba)
        self.assertEqual(img.shape, (3, 3, 4))
        cat = m.category_grid()
        for i in range(3):
            for j in range(3):
                self.assertEqual(tuple(img[j, i][:3]), T.CATEGORY_COLOR[cat[i, j]])

    def test_clicking_a_cell_shows_its_response_and_advice(self):
        self.tab.select_cell((1, 1))
        html = self.tab.detail_text.toHtml()
        self.assertIn("Kp = -100", html)
        self.assertIn("Suggestions", html)
        self.assertIn("Identified loop", html)                                  # identification ran (lock-in set)
        self.assertGreater(len(self.tab.plot1.listDataItems()), 0)
        self.assertGreater(len(self.tab.plot2.listDataItems()), 0)

    def test_untested_or_skipped_cells_explain_themselves(self):
        tab = T.TuningTab(FakeInstrument(), None)
        m = tab._ensure_map()
        m.skipped[(0, 0)] = "more aggressive than cell (1, 1), which lost the loop"
        tab.select_cell((0, 0))
        self.assertIn("lost the loop", tab.detail_text.toPlainText())
        tab.select_cell((4, 5))
        self.assertIn("not tested yet", tab.detail_text.toPlainText())

    def test_zoom_refines_around_the_selection_and_back_returns(self):
        tab = self.tab
        before = len(tab.maps)
        tab.select_cell((1, 1))
        tab._zoom()
        self.assertEqual(len(tab.maps), before + 1)
        sub = tab.current_map()
        self.assertEqual(sub.grid.dims, (3, 3))
        center = (1, 1)
        self.assertIn(center, sub.results)                                      # already measured: not repeated
        self.assertAlmostEqual(sub.grid.kp0, tab.maps[0].grid.kp(1))
        tab._back()
        self.assertEqual(len(tab.maps), before)

    def test_prior_from_a_test_with_a_model_paints_predictions(self):
        tab = self.tab
        tab.select_cell((1, 1))
        self.assertTrue(tab.btn_prior.isEnabled())
        tab._predict_prior()
        m = tab.current_map()
        self.assertEqual(len(m.prior), 9)

    def test_stage_selected_pair_in_the_parameters_tab(self):
        staged = []

        class Params:
            def stage_value(self, ptype, code, value):
                staged.append((ptype, code, value))
                return True
        tab = T.TuningTab(FakeInstrument(), None, params_tab=Params())
        m = tab._ensure_map()
        tab.select_cell((2, 3))
        tab._stage_selected()
        kp, ki = m.pair((2, 3))
        self.assertEqual(staged, [("EDIT", "Edit27", kp), ("EDIT", "Edit22", ki)])


class Guidance(unittest.TestCase):
    """The on-screen explanations: the Guide tab and the 'what now?' line."""

    def test_guide_covers_every_verdict_with_an_action(self):
        html = T.guide_html()
        for cat in W.CATEGORIES:
            self.assertIn(W.CATEGORY_LABEL[cat], html)
            self.assertIn(cat, {c for c, _, _ in T.GUIDE_VERDICTS})
        for word in ("Baseline", "Kp", "Ki", "Stage in Params tab"):
            self.assertIn(word, html)

    def test_guide_is_the_first_thing_shown_and_the_first_result_switches_to_the_plots(self):
        tab = make_tab(FakeInstrument())
        self.assertIs(tab.detail_tabs.currentWidget(), tab.guide)
        self.assertIn("No test yet", tab.detail_text.toPlainText())
        tab._show_result(W.StepTestResult(kp=-100, ki=-1e4, loop="pll", failure="x"))
        self.assertEqual(tab.detail_tabs.currentIndex(), 0)

    def test_hint_says_why_the_buttons_are_disabled_and_what_to_press(self):
        tab = T.TuningTab(FakeInstrument(), FakeInstrument())
        self.assertIn("tick every item", tab.hint_label.text())
        self.assertIn("Tick every item", tab.btn_single.toolTip())
        self.assertFalse(tab.btn_single.isEnabled())
        for c in tab.checks:
            c.setChecked(True)
        self.assertIn("Run single test", tab.hint_label.text())
        self.assertTrue(tab.btn_single.isEnabled())
        off = T.TuningTab(MockDDEClient(), None)
        self.assertIn("Offline", off.hint_label.text())


class AmplitudeLoopUI(unittest.TestCase):
    def make(self):
        tab = T.TuningTab(FakeInstrument(), FakeInstrument())
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        return tab

    def test_switching_to_the_amplitude_loop_uses_the_manuals_values_at_1v(self):
        tab = self.make()
        self.assertEqual(tab.gain_combo.currentData(), 1.0)
        self.assertAlmostEqual(tab.ki_spin.value(), 5e8 / tab.q_spin.value())
        self.assertAlmostEqual(tab.kp_spin.value() / tab.ki_spin.value(), 1e4)
        self.assertAlmostEqual(tab.tau_spin.value(), 10.0 * tab.q_spin.value() / tab.f0_spin.value())
        self.assertEqual((tab.factor_spin.value(), tab.range_spin.value()), (10.0, 1000.0))
        self.assertFalse(tab.gain_combo.isHidden())
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("pll"))
        self.assertTrue(tab.gain_combo.isHidden())
        self.assertEqual((tab.factor_spin.value(), tab.range_spin.value()), (2.0, 16.0))

    def test_a_lower_output_gain_needs_ten_times_larger_gains(self):
        tab = self.make()
        ki1, kp1 = tab.ki_spin.value(), tab.kp_spin.value()
        tab.gain_combo.setCurrentIndex(tab.gain_combo.findData(0.1))
        self.assertEqual(tab.ki_spin.value(), ki1)                              # the selector alone never touches the baseline
        self.assertIn("+-0.1 V", tab.start_label.text())
        tab.btn_fill.click()
        self.assertAlmostEqual(tab.ki_spin.value() / ki1, 10.0)
        self.assertAlmostEqual(tab.kp_spin.value() / kp1, 10.0)

    def test_a_baseline_far_from_the_manual_start_is_flagged(self):
        tab = self.make()
        for c in tab.checks:
            c.setChecked(True)
        self.assertNotIn("Check:", tab.hint_label.text())
        tab.ki_spin.setValue(tab.ki_spin.value() * 100)
        self.assertIn("Check:", tab.hint_label.text())
        tab.gain_combo.setCurrentIndex(tab.gain_combo.findData(0.1))            # x10 explains part of it: x10 left
        self.assertNotIn("Check:", tab.hint_label.text())

    def test_the_checklist_follows_the_loop_and_starts_unticked(self):
        tab = T.TuningTab(FakeInstrument(), FakeInstrument())
        for c in tab.checks:
            c.setChecked(True)
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        self.assertFalse(any(c.isChecked() for c in tab.checks))
        self.assertIn("PLL off", tab.checks[1].text())
        self.assertNotIn("Auto 0", " ".join(c.text() for c in tab.checks))
        self.assertFalse(tab.btn_scan.isEnabled())
        for c in tab.checks:
            c.setChecked(True)
        self.assertTrue(tab.btn_scan.isEnabled())

    def test_scale_scan_and_limits_come_from_the_widgets(self):
        tab = self.make()
        g = tab.scan_grid()
        self.assertEqual(g.dims, (7, 1))                                        # pure speed: a single shape column
        self.assertEqual(g.factor, 10.0)
        self.assertAlmostEqual(g.ki(6, 0) / g.kp(6), tab.ki_spin.value() / tab.kp_spin.value())
        lim = tab.safety_limits()
        self.assertEqual(lim.max_gain_factor, 1000.0)
        self.assertAlmostEqual(lim.min_gain_factor, 1e-3)
        self.assertIn("Scale scan: 7 tests", tab.est_label.text())

    def test_running_a_scale_scan_pushes_a_speed_only_map_and_stop_restores_the_baseline(self):
        inst = FakeInstrument()
        tab = T.TuningTab(inst, inst)
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        for c in tab.checks:
            c.setChecked(True)
        kp, ki = tab.kp_spin.value(), tab.ki_spin.value()
        tab._confirmed_baseline[tab.loop_def.key] = (kp, ki)
        tab._run_scan()
        self.assertTrue(tab.runner_active())
        m = tab.current_map()
        self.assertEqual(m.grid.dims, (7, 1))
        self.assertEqual(len(m.cells()), 7)
        tab.stop()
        self.assertFalse(tab.runner_active())
        self.assertEqual(inst.writes[-3:], [("Edit32", kp), ("Edit24", ki), ("Edit23", 6.0)])    # baseline + Ref back

    def test_a_suggestion_beyond_the_allowed_range_is_not_run(self):
        tab = self.make()
        for c in tab.checks:
            c.setChecked(True)
        tab.range_spin.setValue(4.0)
        tab._suggestions = [W.Suggestion("scale_both", tab.kp_spin.value() * 10, tab.ki_spin.value() * 10, "x")]
        tab._test_suggestion()
        self.assertFalse(tab.runner_active())
        self.assertIn("Not run", tab.log.toPlainText())

    def test_the_guide_explains_the_amplitude_loop(self):
        html = T.guide_html()
        for word in ("Output Gain", "scale scan", "decades", "Drive"):
            self.assertIn(word.lower(), html.lower())


class BaselineProtection(unittest.TestCase):
    """Priority 1: confirmation gate, checklist hygiene, restored-value logging, voltage warning."""

    def test_declining_the_confirmation_aborts_the_run(self):
        inst = FakeInstrument()
        tab = T.TuningTab(inst, inst)
        for c in tab.checks:
            c.setChecked(True)
        tab._confirm_baseline = lambda kp, ki: False           # simulate Cancel, without a real dialog
        tab._run_single()
        self.assertIsNone(tab.runner)
        self.assertIn("not started", tab.log.toPlainText())

    def test_an_already_confirmed_baseline_does_not_reprompt(self):
        tab = make_tab(FakeInstrument())                        # make_tab pre-confirms the baseline

        def boom(*a, **k):
            raise AssertionError("QMessageBox.exec_ must not be called for an already-confirmed baseline")
        orig = QtWidgets.QMessageBox.exec_
        QtWidgets.QMessageBox.exec_ = boom
        try:
            self.assertTrue(tab._confirm_baseline(tab.kp_spin.value(), tab.ki_spin.value()))
        finally:
            QtWidgets.QMessageBox.exec_ = orig

    def test_an_edited_baseline_is_no_longer_considered_confirmed(self):
        tab = make_tab(FakeInstrument())                        # pre-confirmed at Kp=-100, Ki=-1e4
        tab.kp_spin.setValue(tab.kp_spin.value() * 2)            # edited: no longer matches the confirmed pair
        orig = QtWidgets.QMessageBox.exec_
        QtWidgets.QMessageBox.exec_ = lambda self: QtWidgets.QMessageBox.Cancel   # a real dialog is now shown
        try:
            self.assertFalse(tab._confirm_baseline(tab.kp_spin.value(), tab.ki_spin.value()))
        finally:
            QtWidgets.QMessageBox.exec_ = orig

    def test_checklist_resets_when_the_baseline_or_target_changes(self):
        tab = T.TuningTab(FakeInstrument(), None)
        for c in tab.checks:
            c.setChecked(True)
        tab.kp_spin.setValue(tab.kp_spin.value() * 2)
        self.assertFalse(any(c.isChecked() for c in tab.checks))
        for c in tab.checks:
            c.setChecked(True)
        tab.rise_spin.setValue(tab.rise_spin.value() + 1)
        self.assertFalse(any(c.isChecked() for c in tab.checks))

    def test_checklist_resets_after_a_run_finishes(self):
        tab = make_tab(FakeInstrument())
        tab._start_runner(tab._one_shot(-100.0, -1e4))
        self.assertTrue(wait_until(lambda: not tab.runner_active()))
        self.assertFalse(any(c.isChecked() for c in tab.checks))

    def test_restored_baseline_is_logged_with_real_values(self):
        tab = make_tab(FakeInstrument())
        tab._start_runner(tab._one_shot(-100.0, -1e4))
        self.assertTrue(wait_until(lambda: not tab.runner_active()))
        self.assertIn("Baseline restored: Kp=-100, Ki=-1e+04", tab.log.toPlainText())

    def test_afl_voltage_warning_appears_only_beyond_the_limit(self):
        tab = T.TuningTab(FakeInstrument(), None)
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        tab.base_spin.setValue(6.0)
        tab.step_spin.setValue(10.0)                            # +-10 % of 6 V: well inside +-10 V
        self.assertEqual(tab._voltage_warning(), "")
        tab.base_spin.setValue(12.0)                            # 12 V + 10 % = 13.2 V: over the limit
        self.assertIn("Check:", tab._voltage_warning())
        self.assertIn("Check:", tab.hint_label.text())

    def test_pinned_stop_button_is_enabled_only_while_running(self):
        tab = make_tab(FakeInstrument())
        self.assertFalse(tab.btn_stop.isEnabled())
        tab._start_runner(tab._one_shot(-100.0, -1e4))
        self.assertTrue(tab.btn_stop.isEnabled())
        tab.stop()
        self.assertFalse(tab.btn_stop.isEnabled())

    def test_context_label_reflects_loop_baseline_and_connection(self):
        tab = T.TuningTab(FakeInstrument(), None)
        text = tab.context_label.text()
        self.assertIn("PLL", text)
        self.assertIn("-100", text)
        self.assertIn("OFFLINE", text)
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        self.assertIn("AFL", tab.context_label.text())


class SuggestedSetupGain(unittest.TestCase):
    def test_output_gain_scales_the_amplitude_gains(self):
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        tab = SuggestedTab(FakeInstrument(), None)
        tab.q_val.setValue(25000.0)
        tab.f0_val.setValue(25000.0)
        tab.out_gain.setCurrentIndex(tab.out_gain.findData(1.0))
        tab._recalc()
        ki1, kp1 = float(tab.ki_out.text()), float(tab.kp_out.text())
        self.assertAlmostEqual(ki1, 2e4)
        tab.out_gain.setCurrentIndex(tab.out_gain.findData(0.1))                # recalculates by itself
        self.assertAlmostEqual(float(tab.ki_out.text()) / ki1, 10.0)
        self.assertAlmostEqual(float(tab.kp_out.text()) / kp1, 10.0)


if __name__ == "__main__":
    unittest.main()
