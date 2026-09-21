"""Tests for tuning.planner: requirements, trial analysis and the guided search (on the simulator)."""
import math
import unittest

from sxm_ncafm_control.tuning.planner import (GuidedTuner, Limits, ScanSpec, analyze_trial, run_guided)
from sxm_ncafm_control.tuning.simulator import SimulatedPLLBackend, run_step_trial
from sxm_ncafm_control.tuning.trial import StepProtocol

FAST = ScanSpec(t_line=3.0, n_px=256)
SLOW = ScanSpec(t_line=12.0, n_px=128)


class Spec(unittest.TestCase):
    def test_pixel_dwell_and_required_bandwidth(self):
        self.assertAlmostEqual(FAST.t_px, 3.0 / 256)
        self.assertAlmostEqual(FAST.required_bandwidth_hz, 60.0, delta=1.0)      # ~60 Hz
        self.assertAlmostEqual(SLOW.t_px, 12.0 / 128)
        self.assertAlmostEqual(SLOW.required_bandwidth_hz, 7.5, delta=0.2)      # ~7.5 Hz

    def test_faster_scans_need_more_bandwidth(self):
        bw = [ScanSpec(t_line=t, n_px=256).required_bandwidth_hz for t in (12, 6, 3)]
        self.assertLess(bw[0], bw[1])
        self.assertLess(bw[1], bw[2])


class Analysis(unittest.TestCase):
    proto = StepProtocol()

    def test_good_gains_are_feasible_for_a_slow_scan_but_not_a_fast_one(self):
        r = run_step_trial(-100, -1e4, self.proto, seed=1)
        self.assertTrue(analyze_trial(r, SLOW).feasible)
        a = analyze_trial(r, FAST)
        self.assertFalse(a.feasible)
        self.assertGreater(a.violations["rise"], 0)

    def test_lost_lock_is_a_failure(self):
        a = analyze_trial(run_step_trial(+100, +1e4, self.proto, seed=1), SLOW)
        self.assertEqual(a.failure, "lost lock")
        self.assertFalse(a.feasible)
        self.assertTrue(math.isinf(a.penalty))

    def test_ringing_runaway_is_unmeasurable_or_infeasible(self):
        a = analyze_trial(run_step_trial(-100, -2e5, self.proto, seed=1), SLOW)
        self.assertFalse(a.feasible)

    def test_noise_reported(self):
        a = analyze_trial(run_step_trial(-100, -1e4, self.proto, seed=1), SLOW)
        self.assertTrue(0.001 < a.noise_rms < 0.2)                # Hz
        self.assertFalse(math.isnan(a.pixel_noise))
        self.assertLess(a.pixel_noise, a.noise_rms)               # pixel averaging removes noise

    def test_violation_descriptions_carry_numbers(self):
        a = analyze_trial(run_step_trial(-100, -1e4, self.proto, seed=1), FAST)
        text = " ".join(a.describe_violations(FAST))
        self.assertIn("rise", text)
        self.assertIn("ms", text)


class Search(unittest.TestCase):
    def tune(self, spec, seed=10, **kw):
        t = GuidedTuner(-100, -1e4, spec, **kw)
        run_guided(t, SimulatedPLLBackend(seed=seed))
        return t

    def test_slow_scan_finds_quieter_gains_that_still_resolve_the_scan(self):
        t = self.tune(SLOW)
        base, rec = t.history[0], t.recommendation()
        self.assertTrue(rec.meets_all_constraints)
        self.assertTrue(rec.verified)
        self.assertLess(rec.analysis.noise_rms, 0.6 * base.noise_rms)         # substantially quieter
        self.assertLess(abs(rec.kp), abs(base.kp))                             # by lowering the gains
        # the winner really satisfies the spec on its own measured numbers
        self.assertLessEqual(rec.analysis.df.rise_time, SLOW.rise_max)
        self.assertLessEqual(rec.analysis.df.overshoot, SLOW.overshoot_max)

    def test_fast_scan_reports_that_it_cannot_be_met_cleanly(self):
        t = self.tune(FAST)
        rec = t.recommendation()
        self.assertFalse(rec.meets_all_constraints)
        self.assertIn("rise", rec.rationale)
        self.assertLessEqual(rec.analysis.shape_penalty(), 0.0)                # ... but it is a clean response
        self.assertEqual(t.verified, False)

    def test_baseline_first_and_stage_order(self):
        t = self.tune(SLOW)
        stages = [p.stage for p in t.proposals_made]
        self.assertEqual(stages[0], "baseline")
        order = ["baseline", "ratio", "scale", "refine", "verify"]
        self.assertEqual(stages, sorted(stages, key=order.index))
        self.assertEqual(stages[-1], "verify")

    def test_proposals_stay_inside_the_limits(self):
        lim = Limits(min_factor=0.2, max_factor=3.0, max_trials=16)
        t = self.tune(SLOW, limits=lim)
        for p in t.proposals_made:
            for v, v0 in ((p.kp, -100), (p.ki, -1e4)):
                self.assertGreaterEqual(abs(v / v0), lim.min_factor - 1e-9)
                self.assertLessEqual(abs(v / v0), lim.max_factor + 1e-9)
            self.assertLess(p.kp, 0)                                            # signs are never flipped
            self.assertLess(p.ki, 0)

    def test_trial_budget_is_respected(self):
        t = self.tune(SLOW, limits=Limits(max_trials=6))
        self.assertLessEqual(len(t.history), 7)                                 # +1 for the verify repeat

    def test_search_is_reproducible_for_a_seed_and_robust_across_seeds(self):
        a = self.tune(SLOW, seed=10).recommendation()
        b = self.tune(SLOW, seed=10).recommendation()
        self.assertEqual((a.kp, a.ki), (b.kp, b.ki))
        for seed in (21, 33):
            r = self.tune(SLOW, seed=seed).recommendation()
            self.assertTrue(r.meets_all_constraints)
            self.assertLess(abs(math.log(abs(r.kp / a.kp))), math.log(1.6))     # same neighbourhood

    def test_declined_proposals_are_skipped_and_never_recommended(self):
        t = GuidedTuner(-100, -1e4, SLOW)
        seen = []
        run_guided(t, SimulatedPLLBackend(seed=3), approve=lambda p: (seen.append(p.stage) or p.stage != "ratio"))
        self.assertIn("ratio", seen)
        skipped = [a for a in t.history if a.failure == "skipped by user"]
        self.assertGreaterEqual(len(skipped), 1)
        rec = t.recommendation()
        self.assertTrue(rec is None or rec.analysis.measurable)

    def test_protocol_misuse_raises(self):
        t = GuidedTuner(-100, -1e4, SLOW)
        gen = t.proposals()
        next(gen)
        with self.assertRaises(RuntimeError):
            next(gen)                                                           # forgot record()
        t2 = GuidedTuner(-100, -1e4, SLOW)
        with self.assertRaises(RuntimeError):
            t2.record(None)                                                     # nothing pending

    def test_unstable_baseline_does_not_crash(self):
        t = GuidedTuner(-100, -1e6, SLOW)                                       # baseline that loses lock
        run_guided(t, SimulatedPLLBackend(seed=4))
        self.assertFalse(t.history[0].measurable)
        self.assertGreaterEqual(len(t.history), 3)                              # it kept exploring

    def test_zero_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            GuidedTuner(0, -1e4, SLOW)

    def test_report_lists_every_trial(self):
        t = self.tune(SLOW)
        rep = t.report()
        self.assertIn("Recommended", rep)
        self.assertEqual(sum(1 for line in rep.splitlines() if line[:2].strip().isdigit()), len(t.history))

    def test_pareto_front_is_non_dominated(self):
        t = self.tune(FAST)
        front = t.pareto()
        self.assertGreaterEqual(len(front), 2)
        for a in front:
            for b in front:
                if a is not b:
                    self.assertFalse(b.df.rise_time <= a.df.rise_time and b.noise_rms <= a.noise_rms)


if __name__ == "__main__":
    unittest.main()
