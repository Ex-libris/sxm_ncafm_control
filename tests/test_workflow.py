"""Tests for tuning.workflow: test definition, channel detection, analysis, verdicts, advice, safety, map."""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tests.fixtures import (AFL_SETUP, PLL_SCALE, PLL_SETUP, afl_plan, pll_plan, record_afl,
                                              record_pll)
from sxm_ncafm_control.tuning import metrics as M
from sxm_ncafm_control.tuning import workflow as W

SLOW_SCAN = W.Target.from_scan(t_line=12.0, n_px=128)     # rise <= 47 ms, Phase tail <= 1.2 s
FAST_SCAN = W.Target.from_scan(t_line=3.0, n_px=256)      # rise <= 5.9 ms


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
    def analyse(self, kp, ki, target, **kw):
        plan = pll_plan(**{k: kw.pop(k) for k in list(kw) if k in ("hold_s", "n_events", "lead_s")})
        ct, _ = record_pll(plan, kp, ki, **kw)
        res = W.analyze_test(ct)
        res.verdict = W.classify(res, target)
        return res

    def test_good_gains_for_a_slow_scan(self):
        r = self.analyse(-100, -1e4, SLOW_SCAN, hold_s=2.0)
        self.assertIsNone(r.failure)
        self.assertEqual(r.n_steps, 7)
        self.assertAlmostEqual(r.primary.rise_time, 0.0122, delta=0.003)       # simulator ground truth ~12 ms
        self.assertAlmostEqual(r.primary.overshoot, 0.095, delta=0.04)
        self.assertEqual(r.verdict.category, "good", r.verdict.reasons)
        self.assertTrue(0.005 < r.noise_rms < 0.08)
        self.assertGreater(r.latency_s, 0.006)                                  # command latency was 8 ms

    def test_same_gains_are_too_slow_for_a_fast_scan(self):
        r = self.analyse(-100, -1e4, FAST_SCAN, hold_s=2.0)
        self.assertEqual(r.verdict.category, "too_slow")

    def test_high_ki_overshoots_and_very_high_ki_rings_or_is_lost(self):
        self.assertEqual(self.analyse(-100, -5e4, SLOW_SCAN, hold_s=2.0).verdict.category, "overshoot")
        self.assertIn(self.analyse(-100, -3e5, SLOW_SCAN, hold_s=2.0).verdict.category, ("ringing", "lost"))

    def test_low_ki_is_a_slow_tail(self):
        r = self.analyse(-100, -1e3, SLOW_SCAN, hold_s=2.0)
        self.assertEqual(r.verdict.category, "slow_tail", r.verdict.reasons)

    def test_short_holds_do_not_judge_the_tail(self):
        r = self.analyse(-100, -1e3, SLOW_SCAN, hold_s=0.2, lead_s=0.6)
        self.assertNotEqual(r.verdict.category, "slow_tail")
        self.assertTrue(any("lengthen the hold" in w for w in r.warnings), r.warnings)
        self.assertIsNotNone(r.primary)                                        # the df edge is still measured

    def test_short_holds_still_give_the_df_shape(self):
        long_ = self.analyse(-100, -1e4, SLOW_SCAN, hold_s=2.0)
        short = self.analyse(-100, -1e4, SLOW_SCAN, hold_s=0.2, lead_s=0.6)
        self.assertAlmostEqual(short.primary.rise_time, long_.primary.rise_time, delta=0.4 * long_.primary.rise_time)
        self.assertAlmostEqual(short.primary.overshoot, long_.primary.overshoot, delta=0.06)

    def test_timing_jitter_does_not_smear_the_edges(self):
        clean = self.analyse(-100, -1e4, SLOW_SCAN, hold_s=1.0)
        jit = self.analyse(-100, -1e4, SLOW_SCAN, hold_s=1.0, jitter_s=0.012)
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
        self.assertEqual(W.classify(res, SLOW_SCAN).category, "lost")


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
        self.assertEqual(W.classify(res, W.Target.manual(rise_max=0.5)).category, "good")

    def test_amplitude_verdicts(self):
        plan = afl_plan()
        ct, _ = record_afl(plan, 3e7, 3000)                                    # gentle -> slow
        res = W.analyze_test(ct)
        self.assertEqual(W.classify(res, W.Target.manual(rise_max=0.3)).category, "too_slow")
        ct, _ = record_afl(plan, 8.9e7, 3e4)                                   # more Ki -> overshoot
        res = W.analyze_test(ct)
        self.assertEqual(W.classify(res, W.Target.manual(rise_max=0.5, overshoot_max=0.10)).category, "overshoot")


def _metrics(**kw):
    d = dict(step_size=1.0, y_initial=0.0, y_final=1.0, delay_time=0.0, rise_time=0.01, settling_time=0.05,
             settled=True, overshoot=0.0, n_extrema=0, damping_ratio=math.nan, steady_state_error=math.nan,
             noise_rms=0.01, band=0.05)
    d.update(kw)
    return M.StepMetrics(**d)


def _result(kp=-100.0, ki=-1e4, **kw):
    r = W.StepTestResult(kp=kp, ki=ki, loop="pll", n_steps=7, window_s=1.0)
    r.primary = _metrics(**kw.pop("primary", {}))
    for k, v in kw.items():
        setattr(r, k, v)
    return r


TARGET = W.Target(rise_max=0.03, overshoot_max=0.10, decay_max=0.5)


class Verdicts(unittest.TestCase):
    def cat(self, **kw):
        r = _result(**kw)
        return W.classify(r, TARGET).category

    def test_each_category(self):
        self.assertEqual(self.cat(), "good")
        self.assertEqual(self.cat(primary=dict(overshoot=0.25)), "overshoot")
        self.assertEqual(self.cat(primary=dict(n_extrema=6, damping_ratio=0.1)), "ringing")
        self.assertEqual(self.cat(primary=dict(rise_time=0.1)), "too_slow")
        self.assertEqual(self.cat(primary=dict(rise_time=math.nan)), "too_slow")
        self.assertEqual(self.cat(failure="lost lock"), "lost")

    def test_phase_tail(self):
        slow = M.ErrorMetrics(peak=5.0, time_to_peak=0.0, iae=1.0, decay_time=2.0, sign_changes=0, noise_rms=0.1)
        self.assertEqual(self.cat(error=slow), "slow_tail")
        never = M.ErrorMetrics(peak=5.0, time_to_peak=0.0, iae=1.0, decay_time=math.inf, sign_changes=0, noise_rms=0.1)
        self.assertEqual(self.cat(error=never), "slow_tail")
        under = M.ErrorMetrics(peak=5.0, time_to_peak=0.0, iae=1.0, decay_time=0.1, sign_changes=3, noise_rms=0.1)
        self.assertEqual(self.cat(error=under), "ringing")

    def test_fast_margin(self):
        v = W.classify(_result(primary=dict(rise_time=0.01)), TARGET)
        self.assertAlmostEqual(v.fast_margin, 3.0)


class Advice(unittest.TestCase):
    def advise(self, **kw):
        r = _result(**kw)
        v = W.classify(r, TARGET)
        return r, v, W.advise(r, v, TARGET)

    def test_overshoot_lowers_ki_only(self):
        r, v, s = self.advise(primary=dict(overshoot=0.3))
        self.assertEqual(s[0].kind, "change_ki")
        self.assertEqual(s[0].kp, r.kp)
        self.assertLess(abs(s[0].ki), abs(r.ki))

    def test_slow_tail_raises_ki(self):
        slow = M.ErrorMetrics(peak=5.0, time_to_peak=0.0, iae=1.0, decay_time=2.0, sign_changes=0, noise_rms=0.1)
        r, v, s = self.advise(error=slow)
        self.assertEqual(s[0].kind, "change_ki")
        self.assertGreater(abs(s[0].ki), abs(r.ki))

    def test_too_slow_scales_both_keeping_the_ratio(self):
        r, v, s = self.advise(primary=dict(rise_time=0.09))
        self.assertEqual(s[0].kind, "scale_both")
        self.assertAlmostEqual(s[0].ki / s[0].kp, r.ki / r.kp)
        self.assertGreater(abs(s[0].kp), abs(r.kp))
        self.assertAlmostEqual(s[0].kp / r.kp, 2.5)                              # rise 3x too slow, capped at x2.5

    def test_good_with_margin_suggests_lower_gains_to_cut_noise(self):
        r, v, s = self.advise(primary=dict(rise_time=0.008))
        self.assertEqual(s[0].kind, "scale_both")
        self.assertLess(abs(s[0].kp), abs(r.kp))

    def test_good_without_margin_is_accepted(self):
        r, v, s = self.advise(primary=dict(rise_time=0.025))
        self.assertEqual(s[0].kind, "accept")

    def test_lost_backs_off(self):
        r, v, s = self.advise(failure="lost lock")
        self.assertEqual(s[0].kind, "back_off")
        self.assertLess(abs(s[0].kp), abs(r.kp))

    def test_signs_are_preserved(self):
        for kw in (dict(primary=dict(overshoot=0.3)), dict(primary=dict(rise_time=0.09)), dict(failure="x")):
            r, v, s = self.advise(**kw)
            for sug in s:
                self.assertLess(sug.kp, 0)
                self.assertLess(sug.ki, 0)


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


def _cat_result(cat, kp, ki, noise=0.02, rise=0.02):
    r = _result(kp=kp, ki=ki, primary=dict(rise_time=rise))
    r.noise_rms = noise
    r.verdict = W.Verdict(cat, [], score=1.0 if cat == "good" else 0.3)
    return r


class MapLogic(unittest.TestCase):
    def setUp(self):
        self.grid = W.GridSpec(-100.0, -1e4)
        self.map = W.ScreeningMap(self.grid, TARGET)

    def test_grid_is_log_spaced_and_signed(self):
        self.assertEqual(self.grid.dims, (5, 6))
        self.assertAlmostEqual(self.grid.kp(2), -100.0)
        self.assertAlmostEqual(self.grid.kp(4), -400.0)
        self.assertAlmostEqual(self.grid.ki(2, 0), -1e4 / 8)
        self.assertEqual(self.map.baseline, (2, 3))

    def test_order_starts_at_the_baseline_and_moves_outward(self):
        o = self.map.order()
        self.assertEqual(o[0], (2, 3))
        dist = [math.hypot(self.grid.speed_exps[i], self.grid.speed_exps[i] + self.grid.shape_exps[j]) for i, j in o]
        self.assertEqual(dist, sorted(dist))
        self.assertEqual(len(set(o)), 30)

    def test_a_lost_cell_skips_everything_more_aggressive(self):
        self.map.results[(2, 3)] = _cat_result("good", -100, -1e4)
        self.map.results[(3, 4)] = _cat_result("lost", -200, -2e4)
        visited = []
        while (c := self.map.next_cell()) is not None:
            visited.append(c)
            self.map.results[c] = _cat_result("good", *self.map.pair(c))
        for i in range(3, 5):
            for j in range(4, 6):
                if (i, j) == (3, 4):
                    continue                                                    # the failed cell itself
                self.assertNotIn((i, j), visited)
                self.assertIn((i, j), self.map.skipped)
        self.assertIn((2, 4), visited)                                          # less aggressive in Kp: still tested
        self.assertIn((3, 3), visited)                                          # less aggressive in Ki: still tested
        self.assertIn("lost", self.map.skipped[(4, 5)])

    def test_safety_limits_skip_cells(self):
        m = W.ScreeningMap(self.grid, TARGET, W.SafetyLimits(max_gain_factor=2.0, min_gain_factor=0.5))
        seen = []
        while (c := m.next_cell()) is not None:
            seen.append(c)
            m.results[c] = _cat_result("good", *m.pair(c))
        for i, j in seen:
            self.assertLessEqual(abs(self.grid.factor ** self.grid.speed_exps[i]), 2.0)
            self.assertGreaterEqual(abs(self.grid.factor ** (self.grid.speed_exps[i] + self.grid.shape_exps[j])), 0.5)
        self.assertTrue(any("safety" in why for why in m.skipped.values()))

    def test_model_predicted_instability_is_skipped(self):
        self.map.prior[(4, 5)] = "lost"
        self.map.results[(2, 3)] = _cat_result("good", -100, -1e4)
        visited = []
        while (c := self.map.next_cell()) is not None:
            visited.append(c)
            self.map.results[c] = _cat_result("good", *self.map.pair(c))
        self.assertNotIn((4, 5), visited)
        self.assertIn("model", self.map.skipped[(4, 5)])

    def test_islands_best_and_diagonals(self):
        m = self.map
        good = [(1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]
        for c in good:
            m.results[c] = _cat_result("good", *m.pair(c), noise=0.01 * (c[0] + 1))
        m.results[(0, 0)] = _cat_result("good", *m.pair((0, 0)), noise=0.001)           # a separate island
        m.results[(4, 5)] = _cat_result("lost", *m.pair((4, 5)))
        isl = m.islands()
        self.assertEqual([i.size for i in isl], [5, 1])
        self.assertEqual(isl[0].best, (1, 2))                                           # lowest noise in the big island
        self.assertEqual(m.best(), (0, 0))                                              # lowest noise overall
        line = m.speed_line((2, 3))
        ratios = {round(m.grid.ki(i, j) / m.grid.kp(i), 6) for i, j in line}
        self.assertEqual(len(ratios), 1)
        self.assertEqual([m.grid.speed_exps[i] for i, _ in line], sorted(m.grid.speed_exps[i] for i, _ in line))
        self.assertIn("Island 1: 5 cells", m.summary())

    def test_value_and_category_grids(self):
        self.map.results[(2, 3)] = _cat_result("good", -100, -1e4, noise=0.05, rise=0.012)
        self.map.skipped[(0, 0)] = "why"
        cat = self.map.category_grid()
        self.assertEqual(cat[2, 3], "good")
        self.assertEqual(cat[0, 0], "skipped")
        self.assertEqual(cat[4, 5], "untested")
        self.assertAlmostEqual(self.map.value_grid("rise")[2, 3], 0.012)
        self.assertTrue(np.isnan(self.map.value_grid("noise")[0, 0]))

    def test_refine_centres_a_finer_grid(self):
        g = self.grid.refine(3, 4)
        self.assertEqual(g.dims, (3, 3))
        self.assertAlmostEqual(g.kp(1), self.grid.kp(3))
        self.assertAlmostEqual(g.ki(1, 1), self.grid.ki(3, 4))
        self.assertAlmostEqual(g.kp(2) / g.kp(1), math.sqrt(2))


class Refinement(unittest.TestCase):
    def test_refined_map_keeps_the_original_safety_reference(self):
        m = W.ScreeningMap(W.GridSpec(-100.0, -1e4), TARGET, W.SafetyLimits(max_gain_factor=4.0, min_gain_factor=0.25))
        sub = m.refined((4, 3))                                             # centred on Kp x4: its upper half is out of bounds
        self.assertEqual(sub.reference, m.reference)
        self.assertFalse(sub._in_limits(2, 1))                              # Kp x4*sqrt2 exceeds the original x4 limit
        self.assertTrue(sub._in_limits(0, 1))

    def test_suggests_the_best_good_cell_when_an_island_exists(self):
        m = W.ScreeningMap(W.GridSpec(-100.0, -1e4), TARGET)
        m.results[(2, 3)] = _cat_result("good", -100, -1e4, noise=0.05)
        m.results[(1, 2)] = _cat_result("good", -50, -2500, noise=0.01)
        cell, why = m.suggest_refinement()
        self.assertEqual(cell, (1, 2))
        self.assertIn("lowest-noise", why)

    def test_suggests_the_nearest_miss_when_there_is_no_island(self):
        m = W.ScreeningMap(W.GridSpec(-100.0, -1e4), W.Target(rise_max=0.010, overshoot_max=0.10))
        for cell, rise in (((2, 3), 0.0125), ((1, 3), 0.030), ((3, 3), 0.0105)):
            r = _result(kp=-100, ki=-1e4, primary=dict(rise_time=rise))
            m.record(cell, r)
        cell, why = m.suggest_refinement()
        self.assertEqual(cell, (3, 3))                                       # 10.5 ms vs 10 ms allowed
        self.assertIn("nearest miss", why)
        self.assertIsNone(W.ScreeningMap(W.GridSpec(-100.0, -1e4), TARGET).suggest_refinement())


class ModelPrior(unittest.TestCase):
    def test_one_recording_predicts_the_map_and_flags_unstable_pairs(self):
        target = W.Target.from_scan(12.0, 128)
        ct, _ = record_pll(pll_plan(hold_s=0.5), -100, -1e4, seed=3)
        res = W.analyze_test(ct, li_tau=PLL_SETUP.lockin_tau, f0=PLL_SETUP.f0, q=PLL_SETUP.q)
        m = W.ScreeningMap(W.GridSpec(-100.0, -1e4), target)
        m.set_prior_from_model(res.model)
        self.assertEqual(m.prior[(2, 3)], "good")                                    # the pair that was recorded
        self.assertEqual(m.prior[(2, 5)], "overshoot")                               # Ki x4: measured map agrees
        self.assertIn(m.prior[(4, 5)], ("overshoot", "ringing", "lost"))            # the aggressive corner is never predicted good
        self.assertGreaterEqual(sum(v == "good" for v in m.prior.values()), 3)

    def test_prediction_agrees_with_measurement_on_most_cells(self):
        target = W.Target.from_scan(12.0, 128)
        ct, _ = record_pll(pll_plan(hold_s=0.5), -100, -1e4, seed=3)
        res = W.analyze_test(ct, li_tau=PLL_SETUP.lockin_tau, f0=PLL_SETUP.f0, q=PLL_SETUP.q)
        m = W.ScreeningMap(W.GridSpec(-100.0, -1e4, speed_exps=(-1, 0, 1), shape_exps=(-2, -1, 0, 1)), target)
        m.set_prior_from_model(res.model)
        agree = total = 0
        for c in m.cells():
            kp, ki = m.pair(c)
            meas, _ = record_pll(pll_plan(hold_s=1.5), kp, ki, seed=7)
            r = W.analyze_test(meas)
            cat = W.classify(r, target).category
            total += 1
            agree += (cat == m.prior[c]) or (cat in ("ringing", "lost") and m.prior[c] in ("ringing", "lost"))
        self.assertGreaterEqual(agree / total, 0.7, f"{agree}/{total}")


class EndToEndScreening(unittest.TestCase):
    """Screen a real (fixture-recorded) map: the workflow as the GUI drives it."""

    @classmethod
    def setUpClass(cls):
        cls.target = SLOW_SCAN
        cls.map = W.ScreeningMap(W.GridSpec(-100.0, -1e4, speed_exps=(-2, -1, 0, 1), shape_exps=(-2, -1, 0, 1, 2)), cls.target)
        cls.plan = pll_plan(hold_s=1.5)
        n = 0
        while (c := cls.map.next_cell()) is not None:
            kp, ki = cls.map.pair(c)
            ct, _ = record_pll(cls.plan, kp, ki, seed=n + 1, jitter_s=0.008)
            cls.map.record(c, W.analyze_test(ct))
            n += 1
        cls.n_tested = n

    def cat(self, i, j):
        return self.map.category_grid()[i, j]

    def test_map_has_islands_and_a_bad_corner(self):
        self.assertGreaterEqual(len(self.map.islands()), 1)
        self.assertEqual(self.cat(2, 2), "good")                                # the baseline
        self.assertNotEqual(self.cat(3, 4), "good")                             # most aggressive corner (Kp x2, shape x4 -> Ki x8)

    def test_shape_changes_across_the_ratio_and_speed_along_it(self):
        self.assertEqual(self.cat(2, 4), "overshoot")                           # same speed (Kp fixed), shape x4 (Ki x4)
        base, low = self.map.results[(2, 2)], self.map.results[(2, 0)]                # same speed, shape /4 (Ki /4)
        self.assertGreater(low.error.decay_time, 2.0 * base.error.decay_time)         # a longer Phase tail
        line = self.map.speed_line((2, 2))                                      # constant shape: constant Ki:Kp ratio
        rises = [self.map.results[c].primary.rise_time for c in line if c in self.map.results and self.map.results[c].primary]
        self.assertGreaterEqual(len(rises), 3)
        self.assertTrue(all(a > b for a, b in zip(rises, rises[1:])), rises)    # raising both = faster

    def test_lowering_both_gains_reduces_noise_and_the_best_cell_is_quieter_than_baseline(self):
        noise = self.map.value_grid("noise")
        line = [c for c in self.map.speed_line((2, 2)) if not np.isnan(noise[c])]
        vals = [noise[c] for c in line]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), vals)
        best = self.map.best()
        self.assertIsNotNone(best)
        self.assertLessEqual(noise[best], noise[2, 2])

    def test_advice_from_the_map_points_back_toward_the_island(self):
        r = self.map.results[(2, 4)]                                            # overshoot cell
        s = W.advise(r, r.verdict, self.target)
        self.assertEqual(s[0].kind, "change_ki")
        self.assertLess(abs(s[0].ki), abs(r.ki))


class AmplitudeLoopSearch(unittest.TestCase):
    """The amplitude loop spans decades of gain: manual start values, output gain, scale scan, wide limits."""

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

    LIM = W.SafetyLimits(max_gain_factor=1000.0, min_gain_factor=1e-3)

    def scan_map(self):
        g = W.GridSpec.scan(2e8, 2e4, 10.0, 3, 3)
        return g, W.ScreeningMap(g, TARGET, self.LIM, reference=(2e8, 2e4))

    def test_scale_scan_moves_both_gains_together(self):
        g, m = self.scan_map()
        self.assertEqual(g.speed_exps, tuple(range(-3, 4)))
        self.assertEqual(g.shape_exps, (0,))                                    # pure speed: shape fixed at the baseline's
        self.assertEqual(len(m.cells()), 7)
        for i, j in m.cells():
            self.assertAlmostEqual(g.ki(i, j) / g.kp(i), 1e-4)                  # Ki:Kp is the baseline's
        self.assertEqual(m.order()[0], (3, 0))                                  # the baseline first
        self.assertIn("7 untested", m.summary())

    def test_scale_scan_does_not_go_on_after_a_lost_cell(self):
        g, m = self.scan_map()
        m.results[(3, 0)] = _cat_result("good", g.kp(3), g.ki(3, 0))
        m.results[(4, 0)] = _cat_result("lost", g.kp(4), g.ki(4, 0))            # one decade up loses the loop
        visited = []
        while True:
            c = m.next_cell()
            if c is None:
                break
            visited.append(c)
            m.results[c] = _cat_result("too_slow", g.kp(c[0]), g.ki(c[0], c[1]))
        self.assertEqual(sorted(visited), [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(sorted(m.skipped), [(5, 0), (6, 0)])

    def test_the_scan_is_bounded_by_the_gain_range(self):
        g = W.GridSpec.scan(2e8, 2e4, 10.0, 4, 4)
        m = W.ScreeningMap(g, TARGET, self.LIM, reference=(2e8, 2e4))
        seen = []
        while True:
            c = m.next_cell()
            if c is None:
                break
            seen.append(c)
            m.results[c] = _cat_result("too_slow", g.kp(c[0]), g.ki(c[0], c[1]))
        self.assertEqual(len(seen), 7)                                          # +-4 decades asked, +-3 allowed
        self.assertEqual(sorted(m.skipped), [(0, 0), (8, 0)])

    def test_refining_a_scan_gives_a_full_finer_map_with_the_same_limits(self):
        g, m = self.scan_map()
        sub = m.refined((3, 0))
        self.assertEqual(len(sub.cells()), 9)                                   # no longer locked to a single column
        self.assertAlmostEqual(sub.grid.factor, math.sqrt(10.0))
        self.assertIs(sub.limits, m.limits)
        self.assertEqual(sub.reference, m.reference)

    def test_gain_range_check(self):
        ref = (2e8, 2e4)
        self.assertTrue(W.gain_within_limits(2e11, 2e7, ref, self.LIM))         # exactly x1000 both ways
        self.assertFalse(W.gain_within_limits(2e12, 2e7, ref, self.LIM))
        self.assertFalse(W.gain_within_limits(2e8, 2.0, ref, self.LIM))

    def test_amplitude_loop_may_be_scaled_by_a_decade_and_points_to_the_scan(self):
        t = W.Target(rise_max=0.05, overshoot_max=0.10)
        r = _result(kp=2e8, ki=2e4, primary=dict(rise_time=0.5))                # 10x too slow
        r.loop = "afl"
        s = W.advise(r, W.classify(r, t), t)
        self.assertEqual(s[0].kind, "scale_both")
        self.assertAlmostEqual(s[0].kp / r.kp, 10.0)
        self.assertNotIn("Scale scan", s[0].why)
        r.primary = _metrics(rise_time=2.0)                                     # 40x too slow: more than one step
        s = W.advise(r, W.classify(r, t), t)
        self.assertAlmostEqual(s[0].kp / r.kp, 10.0)
        self.assertIn("Scale scan", s[0].why)


if __name__ == "__main__":
    unittest.main()
