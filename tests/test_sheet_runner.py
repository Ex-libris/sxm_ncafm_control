"""
gui/sheet_runner.py against the real-time fake instruments: conditions run in order with a quiet window, the
gains follow G and rho, the run pauses for settings the operator must set (or writes them through a writer), a
lost condition is recovered at the anchor and stops its ramp, and the anchor is restored at the end. Offscreen Qt.

The fakes simulate the loops in pure Python, which competes with the GUI thread for the GIL, so step timing here
is looser than on the instrument: these tests check what the runner does, not how clean the steps look.
"""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets                                    # noqa: E402

from sxm_anfatec.driver import CHANNELS                   # noqa: E402
from sxm_ncafm_control.gui.sheet_runner import RunConfig, SheetRunner  # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeAFLInstrument, FakeInstrument   # noqa: E402
from sxm_ncafm_control.tuning import explore as X                      # noqa: E402
from sxm_ncafm_control.tuning import runsheet as R                     # noqa: E402
from sxm_ncafm_control.tuning import workflow as W                     # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
FAST = X.RecoveryCriteria(window_s=0.3)
AFL_ANCHOR = R.Anchor("afl", 2e8, 2e4)
PLL_ANCHOR = R.Anchor("pll", -100.0, -1e4)


def afl_cfg(sheet, **kw):
    plan = W.StepTestPlan(loop="afl", base=1.0, step=0.05, hold_s=0.6, n_events=4, lead_s=0.3, tail_s=0.1,
                          settle_s=0.5)
    args = dict(anchor=AFL_ANCHOR, sheet=sheet, plan=plan, start_settings={"amp_ref": 1.0, "amp_tau_ms": 20.0},
                base_use=25000.0, quiet_s=1.5, quiet_settle_s=0.3, settle_timeout_s=10.0, recover_timeout_s=10.0,
                record_both=False, criteria=FAST)
    args.update(kw)
    return RunConfig(**args)


def pll_cfg(sheet, **kw):
    plan = W.StepTestPlan(loop="pll", base=25000.0, step=1.0, hold_s=0.3, n_events=4, lead_s=0.3, tail_s=0.1,
                          settle_s=0.4)
    args = dict(anchor=PLL_ANCHOR, sheet=sheet, plan=plan, start_settings={}, base_use=25000.0, quiet_s=1.0,
                quiet_settle_s=0.2, settle_timeout_s=5.0, recover_timeout_s=10.0, record_both=False, criteria=FAST)
    args.update(kw)
    return RunConfig(**args)


def run(runner, on_operator=None, timeout=120.0):
    outs, done, log, skipped, asked = [], [], [], [], []
    runner.condition_finished.connect(outs.append)
    runner.finished.connect(done.append)
    runner.message.connect(log.append)
    runner.skipped.connect(lambda i, why: skipped.append(i))

    def operator(text):
        asked.append(text)
        if on_operator is not None:
            QtCore.QTimer.singleShot(50, on_operator)
    runner.operator_needed.connect(operator)
    runner.start()
    end = time.perf_counter() + timeout
    while not done and time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)
    assert done, "run did not finish"
    return outs, done[0], log, skipped, asked


class LosingPLL(FakeInstrument):
    """The fake PLL, except that it loses the resonance (|Phase| 80 deg) whenever |Kp| exceeds 500."""

    def read_raw(self, idx):
        if abs(self.kp_raw) > 500 and idx == CHANNELS["Phase"][0]:
            return int(80.0 / CHANNELS["Phase"][3])
        return super().read_raw(idx)


class FakeWriter:
    def __init__(self):
        self.sets = []

    def options(self, name):
        return {"amp.tau": ["5 ms", "20 ms", "50 ms"]}[name]

    def set(self, name, value):
        self.sets.append((name, value))
        return {"value": value}


class AmplitudeLoop(unittest.TestCase):
    def test_a_g_ramp_measures_steps_and_a_quiet_window_at_each_gain(self):
        inst = FakeAFLInstrument()
        # two ramps of one condition each: the fake's loose timing must not trip the stop rule between them
        r = SheetRunner(inst, inst, afl_cfg(R.ramp("g", [0.5]) + R.ramp("g", [1])))
        outs, reason, log, skipped, asked = run(r)
        self.assertEqual(reason, "done")
        self.assertEqual([o.index for o in outs], [0, 1])
        self.assertEqual(asked, [])
        for o, g in zip(outs, (0.5, 1.0)):
            self.assertAlmostEqual(o.kp, 2e8 * g)
            self.assertAlmostEqual(o.ki, 2e4 * g)
            self.assertIsNotNone(o.result)                 # step quality is judged on the instrument, not the fake
            self.assertIn("QPlusAmpl", o.quiet.channels)
            self.assertGreater(o.quiet.duration_s, 1.0)
            self.assertGreater(o.quiet.get("Drive", "mean"), 0)
            self.assertLess(o.marks["test"], o.marks["quiet"])
            self.assertIn("QPlusAmpl", o.data)
        self.assertIn(("Edit32", 1e8), inst.writes)
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])   # anchor restored

    def test_the_run_waits_for_the_operator_to_set_what_dde_cannot(self):
        inst = FakeAFLInstrument()
        r = SheetRunner(inst, inst, afl_cfg([R.Condition(settings={"amp_tau_ms": 50.0})]))
        outs, reason, log, _, asked = run(r, on_operator=r.continue_)
        self.assertEqual(reason, "done")
        self.assertEqual(len(asked), 1)
        self.assertIn("AFL Tau = 50 ms", asked[0])
        self.assertEqual(outs[0].settings["amp_tau_ms"], 50.0)
        self.assertTrue(any("Set back by hand" in m and "AFL Tau 20 ms" in m for m in log))

    def test_a_writer_sets_it_without_asking_and_puts_it_back(self):
        inst = FakeAFLInstrument()
        w = FakeWriter()
        r = SheetRunner(inst, inst, afl_cfg([R.Condition(settings={"amp_tau_ms": 50.0})]), writer=w)
        outs, reason, log, _, asked = run(r)
        self.assertEqual(reason, "done")
        self.assertEqual(asked, [])
        self.assertEqual(w.sets, [("amp.tau", "50 ms"), ("amp.tau", "20 ms")])

    def test_output_gain_change_rescales_the_gains(self):
        inst = FakeAFLInstrument()
        cfg = afl_cfg([R.Condition(settings={"output_gain_v": 1.0})],
                      anchor=R.Anchor("afl", 2e8, 2e4, output_gain_v=0.1),
                      start_settings={"amp_ref": 1.0, "output_gain_v": 0.1})
        r = SheetRunner(inst, inst, cfg)
        r.current["output_gain_v"] = 1.0          # as if already set: no pause, and no hardware range in the fake
        r.cfg.start_settings["output_gain_v"] = 1.0
        outs, reason, *_ = run(r)
        self.assertAlmostEqual(outs[0].kp, 2e7)
        self.assertAlmostEqual(outs[0].ki, 2e3)

    def test_stop_mid_condition_restores_the_anchor(self):
        inst = FakeAFLInstrument()
        r = SheetRunner(inst, inst, afl_cfg(R.ramp("g", [0.5])))
        stopped = []
        r.finished.connect(stopped.append)
        r.condition_finished.connect(lambda o: stopped.append(o.status))
        r.start()
        end = time.perf_counter() + 20
        while r._state != "test" and time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)
        r.stop()
        self.assertFalse(r.running)
        self.assertEqual(stopped, ["stopped", "stopped"])
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])


class PLL(unittest.TestCase):
    def test_a_lost_condition_recovers_at_the_anchor_and_stops_its_ramp(self):
        inst = LosingPLL()
        sheet = R.ramp("g", [1]) + R.ramp("g", [10, 20]) + R.ramp("g", [1], rho=0.5)
        r = SheetRunner(inst, inst, pll_cfg(sheet))
        outs, reason, log, skipped, _ = run(r)
        self.assertEqual(reason, "done", reason)
        self.assertEqual([o.index for o in outs], [0, 1, 3])
        self.assertEqual(skipped, [2])
        self.assertEqual(outs[1].status, "lost")
        self.assertIn("lost", outs[1].result.failure)
        i = inst.writes.index(("Edit27", -1000.0))
        self.assertIn(("Edit27", 0.0), inst.writes[i:])                 # released before the anchor is written
        self.assertTrue(any("recovered at the anchor" in m for m in log))
        self.assertNotEqual(outs[2].status, "lost")                     # the next ramp runs normally
        self.assertEqual(inst.writes[-3:], [("Edit27", -100.0), ("Edit22", -1e4), ("DNC3", 25000.0)])


if __name__ == "__main__":
    unittest.main()
