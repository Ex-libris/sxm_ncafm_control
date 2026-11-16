"""
tuning/autotune.py: the stages of the automatic amplitude-loop search and the decisions between them, driven by a
synthetic judge instead of the instrument (pure logic, no Qt).
"""
import json
import math
import unittest
from types import SimpleNamespace

from sxm_ncafm_control.tuning import autotune as A

CFG = dict(q=40000.0, f0=25564.0, output_gain_v=0.1, base_ref=1.0)


def status_of(g, rho):
    """A loop whose edge moves with rho: slow at low G, overshoot / ringing as G * sqrt(rho) grows."""
    x = g * math.sqrt(rho)
    if g < 2:
        return "slow"
    if x >= 100:
        return "ringing"
    if x >= 40:
        return "overshoot"
    return "clean"


def judge(stage, c, status=status_of):
    s = status(c.g, c.rho)
    settle = 1.0 / c.g + 0.05 * abs(math.log(c.rho / 0.3))          # rho 0.3 settles fastest
    tau = c.settings.get("amp_tau_ms")
    noise = 1e-6 * (1.0 + (abs(math.log(tau / 10.0)) if tau else 0.0))   # Tau 10 ms is quietest
    ov = 0.5 if s == "overshoot" else 0.0
    return A.Measured(c.g, c.rho, dict(c.settings), c.role, s, up_rise_s=0.05, down_rise_s=0.06, settle_s=settle,
                      noise=noise, amp_noise=noise, up_status=s, down_status=s, up_overshoot=ov, down_overshoot=ov)


def drive(tuner, j=judge, max_stages=10):
    stage, keys = tuner.first_stage(), []
    while stage is not None and len(keys) < max_stages:
        keys.append(stage.key)
        stage = tuner.next_stage([j(stage, c) for c in stage.conditions])
    return keys


class Start(unittest.TestCase):
    def test_the_anchor_is_the_manual_rule_scaled_down_and_the_ladder_reaches_past_it(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertAlmostEqual(t.manual.ki, 1.25e5)                 # 5e8 / Q x10 at +-0.1 V
        self.assertAlmostEqual(t.manual.kp, 1.25e9)
        self.assertAlmostEqual(t.anchor.kp, 1.25e7)                 # x 1/100
        self.assertAlmostEqual(t.anchor.ki, 1.25e3)
        self.assertEqual(t.manual_g, 100)
        self.assertEqual(t.coarse_values(), [1, 3, 10, 30, 100, 300])
        stage = t.first_stage()
        self.assertEqual([c.g for c in stage.conditions], [1, 3, 10, 30, 100, 300])
        self.assertTrue(all(c.rho == 1 and c.ramp == "g" for c in stage.conditions))

    def test_timing_follows_the_ring_down(self):
        tm = A.timing(A.AutoTuneConfig(**CFG))
        self.assertAlmostEqual(tm.ring_down_s, 40000 / (math.pi * 25564), places=6)
        self.assertEqual((tm.hold_s, tm.settle_s, tm.settle_timeout_s), (1.5, 2.5, 15.0))
        slow = A.timing(A.AutoTuneConfig(**{**CFG, "q": 343561.0}))
        self.assertEqual(slow.hold_s, 10.0)                         # capped: StepTestPlan allows 10 s at most
        self.assertGreater(slow.settle_timeout_s, 60.0)

    def test_bad_settings_are_refused(self):
        for bad in ({"start_factor": 2.0}, {"margin": 1.0}, {"q": 0.0}, {"rho_values": ()}, {"repeats": 0}):
            with self.assertRaises(ValueError):
                A.AutoTuneConfig(**{**CFG, **bad})


class Search(unittest.TestCase):
    def test_a_full_search_picks_half_the_edge_at_the_best_ratio(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t), ["coarse", "shape", "edge", "validate"])
        # coarse: clean 3 ... 30, ringing at 100 -> shape at G 30; rho 0.3 settles fastest of the clean ones
        self.assertEqual([c.rho for c in t.stages[1].conditions], [0.1, 0.3, 1, 3, 10])
        self.assertTrue(all(c.g == 30 for c in t.stages[1].conditions))
        # edge at rho 0.3: 45, 67.5, 101 (x1.5 up to 1.5 x the coarse edge); 101 overshoots
        self.assertEqual([c.g for c in t.stages[2].conditions], [45, 67.5, 101])
        self.assertTrue(all(c.rho == 0.3 for c in t.stages[2].conditions))
        rec = t.recommendation
        self.assertTrue(rec.confirmed)
        self.assertEqual((rec.g, rec.rho, rec.edge_g), (50.5, 0.3, 101))     # min(67.5, 101 / 2)
        self.assertAlmostEqual(rec.margin, 2.0)
        self.assertAlmostEqual(rec.kp, 1.25e7 * 50.5)
        self.assertAlmostEqual(rec.ki, 1.25e3 * 50.5 * 0.3)
        val = t.stages[3].conditions
        self.assertEqual(len(val), 5)                                  # 3 repeats + Ref x0.5, x2
        self.assertEqual([c.settings.get("amp_ref") for c in val], [None, None, None, 0.5, 2.0])
        self.assertIsNone(t.failure)
        self.assertTrue(t.finished)
        self.assertIsNone(t.next_stage([]))

    def test_a_start_that_is_already_too_aggressive_continues_below_it(self):
        def strong(g, rho):                     # what the instrument showed: Drive at zero already at G 1
            return "slow" if g < 0.05 else ("floor" if g >= 1 else "clean")

        def j(stage, c):
            m = judge(stage, c, strong)
            m.drive_rms = 5e-6                  # Drive moves: the loop is on
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t, j), ["coarse", "lower", "shape", "edge", "validate"])
        self.assertTrue(all(c.g == 0.3 for c in t.stages[2].conditions))          # fastest clean below the start
        self.assertEqual([c.g for c in t.stages[3].conditions], [0.45, 0.675, 1.01])  # up to 1.5 x the edge (G 1)
        rec = t.recommendation
        self.assertEqual((rec.g, rec.edge_g), (0.505, 1.01))                       # min(0.675, 1.01 / 2)
        self.assertTrue(rec.confirmed)

    def test_overshoot_that_does_not_change_with_gain_is_not_counted(self):
        def ff(stage, c):     # every up-step overshoots 40 % whatever the gain (as on 9 Nov 2026); ringing from 30
            m = judge(stage, c, lambda g, r: "clean" if g * math.sqrt(r) < 30 else "ringing")
            if m.status == "clean":
                m.up_status, m.up_overshoot, m.status = "overshoot", 0.40, "overshoot"
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t, ff), ["coarse", "shape", "edge", "validate"])
        self.assertEqual(t.overshoot_base, {"up": 0.40, "down": 0.0})
        self.assertTrue(t.recommendation.confirmed)
        self.assertTrue(any("Overshoot baseline" in d for d in t.decisions))

    def test_only_overshoot_beyond_the_gentlest_points_counts(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        t.first_stage()

        def mk(g, ov):
            return A.Measured(g, 1.0, {}, "test", "overshoot", up_status="overshoot", down_status="clean",
                              up_overshoot=ov, down_overshoot=0.0)
        ms = [mk(1, 0.40), mk(3, 0.55), mk(10, 0.75)]
        t._update_base(ms)
        t._rejudge(ms)
        self.assertEqual([m.status for m in ms], ["clean", "clean", "overshoot"])    # 0.75 - 0.40 > 0.20
        self.assertIn("baseline", ms[0].note)

    def test_drive_at_zero_does_not_fail_a_condition_in_the_search(self):
        self.assertEqual(A.EXPLORE_LIMITS.floor_max, math.inf)
        self.assertEqual(A.EXPLORE_LIMITS.overshoot_max, A.X.ExploreLimits().overshoot_max)

    def test_a_failed_validation_falls_back_once_to_a_gentler_gain(self):
        def picky(stage, c):
            m = judge(stage, c)
            if stage.key == "validate" and "amp_ref" not in c.settings:
                m.status = "overshoot"
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t, picky), ["coarse", "shape", "edge", "validate", "fallback"])
        self.assertEqual(t.recommendation.g, A.round_sig(50.5 / 1.5))
        self.assertTrue(t.recommendation.confirmed)

    def test_no_confirmation_after_the_fallback_is_reported_not_hidden(self):
        def never(stage, c):
            m = judge(stage, c)
            if stage.key in ("validate", "fallback") and "amp_ref" not in c.settings:
                m.status = "ringing"
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        drive(t, never)
        self.assertFalse(t.recommendation.confirmed)
        self.assertIn("starting point only", t.report_markdown())

    def test_ref_checks_are_reported_but_do_not_decide(self):
        def small_amp_floor(stage, c):
            m = judge(stage, c)
            if c.settings.get("amp_ref") == 0.5:
                m.status = "floor"
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        drive(t, small_amp_floor)
        self.assertTrue(t.recommendation.confirmed)
        self.assertTrue(any("0.5: floor" in d for d in t.decisions))

    def test_conditions_cut_by_the_stop_rule_are_ignored(self):
        def stop_rule(stage, c):
            m = judge(stage, c)
            if stage.key == "coarse" and c.g > 100:
                m.status = "skipped"
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        drive(t, stop_rule)
        self.assertIsNotNone(t.recommendation)

    def test_the_tau_stage_picks_the_quietest_of_the_fast_clean_values(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG, include_tau=True))
        self.assertEqual(drive(t), ["coarse", "shape", "edge", "tau", "validate"])
        self.assertEqual([c.settings["amp_tau_ms"] for c in t.stages[3].conditions], [2, 5, 10, 20])  # 16 ms /10..x2
        self.assertEqual(t.recommendation.tau_ms, 10.0)
        self.assertTrue(all(c.settings.get("amp_tau_ms") == 10.0 for c in t.stages[4].conditions))

    def test_when_no_rho_is_clean_the_ratio_stays_and_the_edge_still_decides(self):
        def shape_bad(stage, c):
            m = judge(stage, c)
            if stage.key == "shape":
                m.status = "overshoot"
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        drive(t, shape_bad)
        self.assertEqual(t.recommendation.rho, 1.0)
        self.assertTrue(any("keeping ρ 1" in d for d in t.decisions))


class Failures(unittest.TestCase):
    def test_everything_too_slow_gives_no_recommendation(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t, lambda s, c: judge(s, c, lambda g, r: "slow")), ["coarse"])
        self.assertIsNone(t.recommendation)
        self.assertIn("too slow", t.failure)

    def test_an_unclean_gentlest_point_tries_once_below_it_then_gives_up(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t, lambda s, c: judge(s, c, lambda g, r: "ringing")), ["coarse", "lower"])
        self.assertEqual([c.g for c in t.stages[1].conditions], [0.01, 0.03, 0.1, 0.3])
        self.assertIn("even the gentlest point (G 0.01 = manual rule x 0.0001)", t.failure)

    def test_a_drive_that_never_moves_means_the_amplitude_loop_is_off(self):
        def afl_off(stage, c):
            m = judge(stage, c, lambda g, r: "slow")
            m.drive_rms = 0.0
            return m
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        self.assertEqual(drive(t, afl_off), ["coarse"])
        self.assertIn("switched OFF", t.failure)
        self.assertIsNone(t.recommendation)

    def test_stopping_keeps_what_was_measured(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        stage = t.first_stage()
        t.stop("stopped", [judge(stage, c) for c in stage.conditions[:2]])
        self.assertTrue(t.finished)
        self.assertIsNone(t.next_stage([]))
        s = t.summary()
        self.assertEqual(s["stopped"], "stopped")
        self.assertEqual(len(s["stages"][0]["conditions"]), 2)
        self.assertIn("ended early", t.report_markdown())


class Reporting(unittest.TestCase):
    def test_summary_is_json_and_the_report_has_every_stage(self):
        t = A.AutoTuner(A.AutoTuneConfig(**CFG))
        drive(t)
        s = json.loads(json.dumps(t.summary()))
        self.assertEqual(s["kind"], A.KIND)
        self.assertEqual([st["key"] for st in s["stages"]], ["coarse", "shape", "edge", "validate"])
        self.assertAlmostEqual(s["recommendation"]["margin"], 2.0)
        md = t.report_markdown()
        for title in (st.title for st in t.stages):
            self.assertIn(title, md)
        self.assertIn("**Kp", md)

    def test_measured_from_an_outcome(self):
        d = SimpleNamespace(rise_s=0.04, overshoot=0.02, status="clean")
        u = SimpleNamespace(rise_s=0.05, overshoot=float("nan"), status="clean")
        out = SimpleNamespace(status="clean", note="n", assessment=SimpleNamespace(up=d, down=u, noise=2e-6),
                              result=SimpleNamespace(primary=SimpleNamespace(settling_time=0.3)),
                              quiet=SimpleNamespace(get=lambda ch, what="rms": 3e-6))
        from sxm_ncafm_control.tuning import runsheet as R
        m = A.Measured.from_outcome(R.Condition(g=3, rho=0.5), out)
        self.assertEqual((m.status, m.overshoot, m.settle_s, m.drive_rms, m.amp_noise), ("clean", 0.02, 0.3, 3e-6, 3e-6))
        self.assertAlmostEqual(m.speed_s, 0.05)
        self.assertEqual(A.Measured.from_outcome(R.Condition(), None).status, "pending")
        self.assertIsNone(A.Measured(1, 1, {}, "test", "slow").to_dict()["settle_s"])


if __name__ == "__main__":
    unittest.main()
