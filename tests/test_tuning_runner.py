"""
gui/tuning_runner.py against the real-time fake instruments: every condition starts from the same verified
baseline, a bad candidate does not end the run, a baseline that never comes back does. Offscreen Qt.
"""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets                                    # noqa: E402

from sxm_ncafm_control.gui.tuning_runner import ConditionRunner   # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeAFLInstrument, FakeInstrument   # noqa: E402
from sxm_ncafm_control.tuning import explore as X              # noqa: E402
from sxm_ncafm_control.tuning import workflow as W             # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
FAST = X.RecoveryCriteria(window_s=0.3)


def wait_until(cond, timeout=40.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


def afl_plan():
    return W.StepTestPlan(loop="afl", base=1.0, step=0.05, hold_s=0.6, n_events=5, lead_s=0.3, tail_s=0.1, settle_s=0.5)


def run(runner, proposals):
    items = list(proposals)
    got = []
    runner.condition_finished.connect(lambda p, r: got.append((p, r)))
    log = []
    runner.message.connect(log.append)
    done = []
    runner.finished.connect(done.append)
    runner.start(lambda: items.pop(0) if items else None)
    assert wait_until(lambda: bool(done), 90), "run did not finish"
    return got, done[0], log


class AmplitudeLoop(unittest.TestCase):
    def test_every_condition_starts_from_the_verified_baseline(self):
        inst = FakeAFLInstrument()
        r = ConditionRunner(inst, inst, afl_plan(), (2e8, 2e4), criteria=FAST, recover_timeout_s=10)
        props = [X.Proposal(2e8, 2e4, "baseline", ""), X.Proposal(4e8, 4e4, "grid", "")]
        got, reason, log = run(r, props)
        self.assertEqual(reason, "done")
        self.assertEqual(len(got), 2)
        self.assertTrue(any("Baseline reference" in m for m in log))
        for p, res in got:
            self.assertIsNone(res.failure, res.failure)
            self.assertIn("recover_s", res.meta["baseline_check"])
            self.assertIn("settle_s", res.meta["baseline_check"])
            self.assertAlmostEqual(res.meta["baseline_check"]["y"], r.reference.y, delta=0.03 * r.reference.y)
            a = X.assess(res, ring_down_s=0.32)
            self.assertFalse(any(map(lambda d: d != d, (a.up.rise_s, a.down.rise_s))))   # both directions measured
        # between the two conditions the baseline gains were written again, before the second candidate
        i_second = max(i for i, w in enumerate(inst.writes) if w == ("Edit32", 4e8))
        before = inst.writes[:i_second]
        self.assertIn(("Edit32", 2e8), before[-4:])
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])

    def test_a_candidate_that_never_settles_is_recorded_and_the_run_continues(self):
        inst = FakeAFLInstrument()
        plan = W.StepTestPlan(loop="afl", base=1.0, step=0.10, hold_s=0.6, n_events=5, lead_s=0.3, tail_s=0.1,
                              settle_s=0.5)
        r = ConditionRunner(inst, inst, plan, (2e8, 2e4), criteria=FAST, recover_timeout_s=10, settle_timeout_s=1.5)
        # Kp = 0, Ki ~ 0: Drive stays frozen at its baseline value, so the amplitude never moves to the first level
        props = [X.Proposal(2e8, 2e4, "baseline", ""), X.Proposal(0.0, 2e-3, "grid", ""),
                 X.Proposal(2e8, 2e4, "repeat", "")]
        got, reason, _ = run(r, props)
        self.assertEqual(reason, "done")
        self.assertEqual(len(got), 3)
        self.assertIn("did not settle", got[1][1].failure)
        self.assertIsNone(got[2][1].failure)                     # recovered to the baseline and measured normally

    def test_no_recovery_stops_the_run_and_restores_the_baseline(self):
        inst = FakeAFLInstrument()
        ref = X.BaselineReference(y=2.0, u=0.1, y_sigma=0.001, u_sigma=0.001)     # a state the loop cannot reach
        r = ConditionRunner(inst, inst, afl_plan(), (2e8, 2e4), criteria=FAST, recover_timeout_s=1.0, reference=ref)
        got, reason, log = run(r, [X.Proposal(2e8, 2e4, "grid", "")])
        self.assertTrue(reason.startswith("aborted"))
        self.assertIn("did not return to the baseline", reason)
        self.assertEqual(got, [])
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])

    def test_stop_mid_condition_restores_the_baseline(self):
        inst = FakeAFLInstrument()
        r = ConditionRunner(inst, inst, afl_plan(), (2e8, 2e4), criteria=FAST)
        r.start(lambda: X.Proposal(5e8, 5e4, "grid", ""))
        self.assertTrue(wait_until(lambda: r._state == "settle", 15))
        r.stop()
        self.assertFalse(r.running)
        self.assertEqual(inst.writes[-3:], [("Edit32", 2e8), ("Edit24", 2e4), ("Edit23", 1.0)])


class PLL(unittest.TestCase):
    def test_pll_conditions_recover_and_a_lost_lock_is_released_first(self):
        inst = FakeInstrument()
        plan = W.StepTestPlan(loop="pll", base=25000.0, step=1.0, hold_s=0.3, n_events=5, lead_s=0.3, tail_s=0.1,
                              settle_s=0.4)
        r = ConditionRunner(inst, inst, plan, (-100.0, -1e4), criteria=FAST, recover_timeout_s=10)
        props = [X.Proposal(-100.0, -1e4, "baseline", ""), X.Proposal(100.0, 1e4, "grid", ""),   # wrong sign: runs away
                 X.Proposal(-100.0, -1e4, "repeat", "")]
        got, reason, _ = run(r, props)
        self.assertEqual(reason, "done", reason)
        self.assertIn("lost", got[1][1].failure)
        i = inst.writes.index(("Edit27", 100.0))
        self.assertIn(("Edit27", 0.0), inst.writes[i:])                     # released (0, 0) before the baseline
        self.assertIsNone(got[2][1].failure, got[2][1].failure)


if __name__ == "__main__":
    unittest.main()
