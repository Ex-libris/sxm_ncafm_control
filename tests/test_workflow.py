"""Tests for tuning.workflow: test definition, channel detection, analysis, safety, the manual's start values."""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tests.fixtures import (AFL_SETUP, PLL_SCALE, PLL_SETUP, afl_plan, pll_plan, record_afl,
                                              record_pll)
from sxm_ncafm_control.tuning import explore as X
from sxm_ncafm_control.tuning import workflow as W


class Detection(unittest.TestCase):
    def test_pll_channels(self):
        d = W.detect_loop(["df", "Phase"])
        self.assertIs(d.loop, W.PLL)
        self.assertTrue(d.complete)

    def test_amplitude_channels_and_aliases(self):
        d = W.detect_loop(["Drive", "QplusAmplitude"])
        self.assertIs(d.loop, W.AFL)
        self.assertTrue(d.complete)
        self.assertEqual(d.names["QPlusAmpl"], "QplusAmplitude")

    def test_partial_and_unknown(self):
        d = W.detect_loop(["df", "Topo"])
        self.assertIs(d.loop, W.PLL)
        self.assertFalse(d.complete)
        self.assertIn("Phase", d.note)
        self.assertIsNone(W.detect_loop(["Topo", "Bias"]).loop)

    def test_pll_wins_when_all_four_are_present(self):
        self.assertIs(W.detect_loop(["df", "Phase", "QPlusAmpl", "Drive"]).loop, W.PLL)


class Plan(unittest.TestCase):
    def test_default_is_the_typical_train(self):
        p = W.StepTestPlan(loop="pll", base=25000.0)
        self.assertEqual(p.n_events, 7)
        self.assertEqual(len(p.levels), 8)
        self.assertEqual(p.levels[:4], [24999.0, 25001.0, 24999.0, 25001.0])          # +-1 Hz about the base
        self.assertEqual(p.event_times[0], p.lead_s)
        self.assertAlmostEqual(p.event_times[1] - p.event_times[0], p.hold_s)
        self.assertAlmostEqual(p.duration, p.lead_s + 7 * p.hold_s + p.tail_s)
        self.assertEqual(len(p.commands()), 7)
        self.assertEqual(p.commands()[0], (p.lead_s, 25001.0))
        self.assertEqual(p.train().step_sizes, [2.0, -2.0] * 3 + [2.0])

    def test_a_plan_can_start_on_the_high_level(self):
        p = W.StepTestPlan(loop="pll", base=25000.0, start_high=True)
        self.assertEqual(p.levels[:3], [25001.0, 24999.0, 25001.0])
        self.assertEqual(p.train().step_sizes[:2], [-2.0, 2.0])
        self.assertEqual(p.commands()[0][1], 24999.0)

    def test_amplitude_step_is_relative(self):
        p = W.StepTestPlan(loop="afl", base=6.0, step=0.10)
        self.assertAlmostEqual(p.low, 5.4)
        self.assertAlmostEqual(p.high, 6.6)
        self.assertAlmostEqual(p.expected_step, 1.2)

    def test_parameters_map_to_dde_codes(self):
        self.assertEqual(W.PLL.kp_param, ("EDIT", "Edit27"))
        self.assertEqual(W.PLL.ki_param, ("EDIT", "Edit22"))
        self.assertEqual(W.PLL.step_param, ("DNC", 3))
        self.assertEqual(W.AFL.kp_param, ("EDIT", "Edit32"))
        self.assertEqual(W.AFL.ki_param, ("EDIT", "Edit24"))
        self.assertEqual(W.AFL.step_param, ("EDIT", "Edit23"))

    def test_validation(self):
        for kw in (dict(loop="x"), dict(n_events=2), dict(hold_s=0.05), dict(step=0.0),
                   dict(loop="afl", base=0.0), dict(loop="afl", base=6.0, step=1.5)):
            with self.assertRaises(ValueError, msg=str(kw)):
                W.StepTestPlan(**{**dict(loop="pll", base=1.0), **kw})


class Alignment(unittest.TestCase):
    def test_jittered_event_times_are_corrected_from_the_data(self):
        plan = pll_plan(hold_s=1.0)
        ct, cap = record_pll(plan, -100, -1e4, jitter_s=0.012, latency_s=0.008)
        signs = np.sign(plan.train().step_sizes)
        aligned, lat = W.align_events(ct.t, ct.channels["df"], ct.event_times, signs, plan.expected_step, W.PLL.align_lead_s)
        true_onset = np.array(plan.event_times) + 0.008
        # after alignment the residual jitter is far below the +-12 ms scatter of the raw event times
        self.assertLess(np.std(np.array(aligned) + W.PLL.align_lead_s - true_onset), 0.004)
        self.assertGreater(np.std(np.array(ct.event_times) - np.array(plan.event_times)), 0.004)


class AnalysePLL(unittest.TestCase):
    def analyse(self, kp, ki, **kw):
        plan = pll_plan(**{k: kw.pop(k) for k in list(kw) if k in ("hold_s", "n_events", "lead_s")})
        ct, _ = record_pll(plan, kp, ki, **kw)
        res = W.analyze_test(ct)
        return res, X.assess(res)

    def test_good_gains(self):
        r, a = self.analyse(-100, -1e4, hold_s=2.0)
        self.assertIsNone(r.failure)
        self.assertEqual(r.n_steps, 7)
        self.assertAlmostEqual(r.primary.rise_time, 0.0122, delta=0.003)       # simulator ground truth ~12 ms
        self.assertAlmostEqual(r.primary.overshoot, 0.095, delta=0.04)
        # the manual calls these gains fast with ~10 % overshoot: right at the limit, so either verdict is honest;
        # what matters is that each direction agrees with the folded measurement
        self.assertIn(a.status, ("clean", "overshoot"))
        for d in (a.up, a.down):
            self.assertAlmostEqual(d.overshoot, r.primary.overshoot, delta=0.06)
        self.assertTrue(0.005 < r.noise_rms < 0.08)
        self.assertGreater(r.latency_s, 0.006)                                  # command latency was 8 ms

    def test_high_ki_overshoots_and_very_high_ki_rings_or_is_lost(self):
        self.assertEqual(self.analyse(-100, -5e4, hold_s=2.0)[1].status, "overshoot")
        self.assertIn(self.analyse(-100, -3e5, hold_s=2.0)[1].status, ("ringing", "lost"))

    def test_short_holds_still_give_the_df_shape(self):
        long_ = self.analyse(-100, -1e4, hold_s=2.0)[0]
        short = self.analyse(-100, -1e4, hold_s=0.2, lead_s=0.6)[0]
        self.assertAlmostEqual(short.primary.rise_time, long_.primary.rise_time, delta=0.4 * long_.primary.rise_time)
        self.assertAlmostEqual(short.primary.overshoot, long_.primary.overshoot, delta=0.06)

    def test_timing_jitter_does_not_smear_the_edges(self):
        clean = self.analyse(-100, -1e4, hold_s=1.0)[0]
        jit = self.analyse(-100, -1e4, hold_s=1.0, jitter_s=0.012)[0]
        self.assertAlmostEqual(jit.primary.rise_time, clean.primary.rise_time, delta=0.35 * clean.primary.rise_time)

    def test_physical_identification_rides_along(self):
        plan = pll_plan(hold_s=0.5)
        ct, _ = record_pll(plan, -100, -1e4)
        res = W.analyze_test(ct, li_tau=PLL_SETUP.lockin_tau, li_stages=2, f0=PLL_SETUP.f0, q=PLL_SETUP.q)
        self.assertIsNotNone(res.model)
        self.assertAlmostEqual(abs(res.model.scale_p) / PLL_SCALE.kp_hz_per_deg, 1.0, delta=0.03)
        self.assertAlmostEqual(abs(res.model.scale_i) / PLL_SCALE.ki_hz_per_deg_s, 1.0, delta=0.03)
        # the model's delay is relative to the onsets detected in the data; the command latency is reported separately
        self.assertGreater(res.latency_s, 0.008)
        self.assertLess(abs(res.model.delay_s), 0.006)

    def test_identification_is_robust_to_event_timing_jitter(self):
        plan = pll_plan(hold_s=0.5)
        ct, _ = record_pll(plan, -100, -1e4, jitter_s=0.008, seed=4)            # +-8 ms scatter of the host-clock event times
        res = W.analyze_test(ct, li_tau=PLL_SETUP.lockin_tau, li_stages=2, f0=PLL_SETUP.f0, q=PLL_SETUP.q)
        self.assertIsNotNone(res.model)
        self.assertAlmostEqual(abs(res.model.scale_p) / PLL_SCALE.kp_hz_per_deg, 1.0, delta=0.05)
        self.assertAlmostEqual(abs(res.model.scale_i) / PLL_SCALE.ki_hz_per_deg_s, 1.0, delta=0.05)
        self.assertGreater(res.model.delay_s, -0.004)                            # not pinned at the edge of the search

    def test_missing_phase_still_analyses_df(self):
        plan = pll_plan(hold_s=1.0)
        ct, _ = record_pll(plan, -100, -1e4)
        del ct.channels["Phase"]
        res = W.analyze_test(ct)
        self.assertIsNone(res.failure)
        self.assertIsNone(res.error)
        self.assertTrue(any("Phase is missing" in w for w in res.warnings))

    def test_wrong_channels_fail_cleanly(self):
        plan = pll_plan()
        ct, _ = record_pll(plan, -100, -1e4)
        ct.channels = {"Topo": ct.channels["df"], "Bias": ct.channels["Phase"]}
        self.assertIn("do not match", W.analyze_test(ct).failure)

    def test_flat_channel_is_unmeasurable_not_a_crash(self):
        plan = pll_plan()
        ct, _ = record_pll(plan, -100, -1e4)
        ct.channels["df"] = np.zeros_like(ct.channels["df"])
        res = W.analyze_test(ct)
        self.assertIn("unmeasurable", res.failure)
        self.assertNotEqual(X.assess(res).status, "clean")


class AnalyseAmplitude(unittest.TestCase):
    def test_amplitude_loop_and_kappa(self):
        plan = afl_plan()
        ct, _ = record_afl(plan, 8.9e7, 8900)
        res = W.analyze_test(ct, li_tau=AFL_SETUP.lockin_tau, li_stages=2, ctrl_tau=AFL_SETUP.tau,
                             f0=AFL_SETUP.f0, q=AFL_SETUP.q)
        self.assertIsNone(res.failure)
        self.assertEqual(res.n_steps, 7)
        self.assertAlmostEqual(res.primary.rise_time, 0.17, delta=0.06)
        self.assertGreater(res.secondary.overshoot, 0.5)                       # Drive kick, as the manual warns
        self.assertAlmostEqual(res.model.kappa, 1.0, delta=0.02)
        self.assertEqual(X.assess(res).status, "clean")

    def test_amplitude_assessments(self):
        plan = afl_plan()
        gentle = X.assess(W.analyze_test(record_afl(plan, 3e7, 3000)[0]))
        base = X.assess(W.analyze_test(record_afl(plan, 8.9e7, 8900)[0]))
        self.assertGreater(gentle.speed_s, 1.5 * base.speed_s)                  # gentle gains: slower
        ct, _ = record_afl(plan, 8.9e7, 3e4)                                   # more Ki -> overshoot
        self.assertEqual(X.assess(W.analyze_test(ct)).status, "overshoot")

    def test_steady_state_error_is_populated_for_the_amplitude_loop(self):
        # was always nan before: analyze_test never passed target_step to the primary channel
        plan = afl_plan()
        ct, _ = record_afl(plan, 8.9e7, 8900)
        res = W.analyze_test(ct)
        self.assertFalse(math.isnan(res.primary.steady_state_error))
        self.assertLess(abs(res.primary.steady_state_error), 0.2)              # QPlusAmpl tracks Ref closely here

    def test_2pct_settling_is_at_least_as_slow_as_5pct(self):
        plan = afl_plan()
        ct, _ = record_afl(plan, 8.9e7, 8900)
        res = W.analyze_test(ct)
        self.assertIsNotNone(res.primary_2pct)
        self.assertGreaterEqual(res.primary_2pct.settling_time, res.primary.settling_time)

    def test_rising_and_falling_are_split_when_both_directions_are_present(self):
        plan = afl_plan()                                                      # default 7 events: both directions
        ct, _ = record_afl(plan, 8.9e7, 8900)
        res = W.analyze_test(ct)
        self.assertIsNotNone(res.primary_rising)
        self.assertIsNotNone(res.primary_falling)
        self.assertIsNotNone(res.secondary_rising)
        self.assertIsNotNone(res.secondary_falling)
        # the curves themselves are kept too (not just their summary metrics) - a plot needs them,
        # and they must be genuinely different arrays, not the same folded-together one reused
        for arr in (res.mean_primary_rising, res.mean_primary_falling,
                   res.mean_secondary_rising, res.mean_secondary_falling):
            self.assertIsNotNone(arr)
            self.assertEqual(len(arr), len(res.grid))
        self.assertFalse(np.array_equal(res.mean_primary_rising, res.mean_primary_falling))
        # a symmetric synthetic loop should give closely matched rise and fall times
        self.assertAlmostEqual(res.primary_rising.rise_time, res.primary_falling.rise_time,
                               delta=0.3 * res.primary.rise_time)

    def test_drive_at_its_zero_floor_is_measured_and_warned_about(self):
        plan = afl_plan()
        ct, _ = record_afl(plan, 8.9e7, 8900)
        clean = W.analyze_test(ct)
        self.assertLess(clean.drive_floor_frac, W.DRIVE_FLOOR_TIME_MAX)          # a healthy loop never sits at zero
        self.assertGreater(clean.drive_settled, 0.0)
        self.assertFalse(any("zero floor" in w for w in clean.warnings))
        drive = ct.channels["Drive"].copy()
        after = np.flatnonzero(ct.t >= ct.event_times[0])
        drive[after[: len(after) // 5]] = 0.0                                     # pinned at zero for 20 % of the test
        ct.channels["Drive"] = drive
        res = W.analyze_test(ct)
        self.assertAlmostEqual(res.drive_floor_frac, 0.2, delta=0.02)
        self.assertTrue(any("zero floor" in w for w in res.warnings))
        self.assertEqual(X.assess(res).down.status, "floor")

    def test_drive_floor_fraction_needs_a_settled_positive_drive(self):
        t = np.linspace(0, 3, 3001)
        self.assertTrue(math.isnan(W.drive_floor_fraction(t, np.zeros_like(t), 1.0, 1.0)[0]))
        d = np.where(t < 1.0, 1.0, np.where(t < 1.5, 0.0, 1.0))
        frac, settled = W.drive_floor_fraction(t, d, 1.0, 1.0)
        self.assertAlmostEqual(settled, 1.0)
        self.assertAlmostEqual(frac, 0.25, delta=0.01)

    def test_the_amplitude_step_defaults_to_5_percent(self):
        self.assertEqual(W.AFL.default_step, 0.05)

    def test_drive_excursion_and_peak_are_populated(self):
        plan = afl_plan()
        ct, _ = record_afl(plan, 8.9e7, 8900)
        res = W.analyze_test(ct)
        self.assertFalse(math.isnan(res.drive_peak_abs))
        self.assertFalse(math.isnan(res.drive_rms_excursion))
        self.assertGreaterEqual(res.drive_peak_abs, 0.0)
        self.assertGreaterEqual(res.drive_rms_excursion, 0.0)


class Safety(unittest.TestCase):
    lim = W.SafetyLimits()

    def test_pll_phase_runaway(self):
        plan = pll_plan()
        self.assertIn("Phase", W.runaway_reason(plan, self.lim, {"Phase": np.full(100, 80.0)}))
        self.assertIsNone(W.runaway_reason(plan, self.lim, {"Phase": np.full(100, 5.0), "df": np.zeros(100)}))

    def test_df_runaway(self):
        plan = pll_plan()
        df = np.concatenate([np.zeros(200), np.full(100, 50.0)])
        self.assertIn("df", W.runaway_reason(plan, self.lim, {"df": df}))

    def test_amplitude_collapse(self):
        plan = afl_plan()
        self.assertIn("collapsed", W.runaway_reason(plan, self.lim, {"QPlusAmpl": np.full(50, 0.5)}, kappa=1.0))
        self.assertIsNone(W.runaway_reason(plan, self.lim, {"QPlusAmpl": np.full(50, 6.0)}, kappa=1.0))


class AmplitudeLoopSearch(unittest.TestCase):
    """The amplitude loop spans decades of gain: the manual's start values and the output gain."""

    def test_manual_start_values_at_1v(self):
        s = W.afl_start_values(q=25000.0, f0=25000.0, output_gain_v=1.0)
        self.assertAlmostEqual(s.ki, 2e4)
        self.assertAlmostEqual(s.kp, 2e8)
        self.assertAlmostEqual(s.tau_s, 0.01)                                   # Q / (100 f0) = 10 ms
        self.assertAlmostEqual(s.ring_down_s, 1.0 / math.pi)                    # Q / (pi f0)

    def test_each_lower_output_gain_range_needs_ten_times_larger_gains(self):
        base = W.afl_start_values(25000.0, 25000.0, 1.0)
        low = W.afl_start_values(25000.0, 25000.0, 0.1)
        high = W.afl_start_values(25000.0, 25000.0, 10.0)
        self.assertAlmostEqual(low.ki / base.ki, 10.0)
        self.assertAlmostEqual(low.kp / base.kp, 10.0)
        self.assertAlmostEqual(high.ki / base.ki, 0.1)
        self.assertAlmostEqual(low.kp / low.ki, 1e4)                            # the ratio never changes

    def test_start_values_span_many_decades_with_q_and_gain(self):
        ks = [W.afl_start_values(q, 25000.0, g).kp for q in (2e3, 1e5) for g in (10.0, 1.0, 0.1)]
        self.assertGreater(max(ks) / min(ks), 1e3)

    def test_test_timing_follows_the_ring_down_and_is_bounded(self):
        slow = W.afl_start_values(1e5, 25000.0)                                 # ring-down 1.27 s
        self.assertEqual(slow.hold_s, 4.0)
        self.assertEqual(slow.settle_s, 6.5)
        fast = W.afl_start_values(2000.0, 25000.0)
        self.assertEqual((fast.hold_s, fast.settle_s), (1.0, 2.0))
        huge = W.afl_start_values(1e9, 1000.0)
        self.assertEqual((huge.hold_s, huge.settle_s), (10.0, 30.0))

    def test_invalid_inputs_are_rejected(self):
        for args in ((0, 25000.0, 1.0), (25000.0, 0, 1.0), (25000.0, 25000.0, 0)):
            with self.assertRaises(ValueError):
                W.afl_start_values(*args)


if __name__ == "__main__":
    unittest.main()
