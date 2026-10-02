"""
tuning/explore.py: per-direction assessment, baseline recovery checks, the decade grid, and the staged
exploration (grid -> edge along g -> ratio search -> repeats) against a synthetic Kp-Ki landscape.
"""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tests.fixtures import afl_plan, record_afl
from sxm_ncafm_control.tuning import explore as X
from sxm_ncafm_control.tuning import workflow as W
from sxm_ncafm_control.tuning.metrics import StepMetrics


def _m(rise=0.01, overshoot=0.0, extrema=0, noise=0.01, step=1.0):
    return StepMetrics(step_size=step, y_initial=0.0, y_final=step, delay_time=0.0, rise_time=rise, settling_time=3 * rise,
                       settled=True, overshoot=overshoot, n_extrema=extrema, damping_ratio=math.nan,
                       steady_state_error=math.nan, noise_rms=noise, band=0.05)


def _res(kp=1.0, ki=1.0, loop="afl", up=None, down=None, floor=0.0, drive_noise=0.01, scatter=0.01):
    r = W.StepTestResult(kp=kp, ki=ki, loop=loop)
    r.primary = _m()
    r.primary_rising, r.primary_falling = up or _m(), down or _m()
    r.secondary = _m(noise=drive_noise)
    r.drive_floor_frac = floor
    r.noise_rms = scatter
    return r


class Assessing(unittest.TestCase):
    def test_directions_are_judged_separately_and_the_worse_one_counts(self):
        a = X.assess(_res(up=_m(rise=0.01), down=_m(rise=0.05, overshoot=0.3)))
        self.assertEqual(a.up.status, "clean")
        self.assertEqual(a.down.status, "overshoot")
        self.assertEqual(a.status, "overshoot")
        self.assertFalse(a.clean)
        self.assertAlmostEqual(a.speed_s, 0.05)                                   # the slower direction

    def test_a_direction_that_does_not_get_there_is_slow_not_clean(self):
        r = _res(down=_m(rise=math.nan))
        a = X.assess(r)
        self.assertEqual((a.up.status, a.down.status), ("clean", "slow"))
        self.assertFalse(a.clean)
        r = _res()
        r.primary_falling, r.n_steps = None, 7
        self.assertEqual(X.assess(r).down.status, "slow")

    def test_drive_at_zero_is_a_down_step_failure(self):
        a = X.assess(_res(floor=0.2))
        self.assertEqual((a.up.status, a.down.status), ("clean", "floor"))
        self.assertIn("Drive at zero", a.down.reasons[0])

    def test_ringdown_limited_down_steps_are_flagged_not_failed(self):
        a = X.assess(_res(down=_m(rise=0.8)), ring_down_s=0.32)                 # ln(9)*0.32 = 0.70 s
        self.assertTrue(a.ringdown_limited)
        self.assertTrue(a.clean)
        self.assertIn("ring-down", a.reasons[0])

    def test_scatter_and_failures(self):
        r = _res(scatter=0.5)
        r.n_steps = 7
        self.assertEqual(X.assess(r).status, "ringing")                          # 0.5 >> 0.01 * sqrt(7): oscillating
        r.primary = _m(noise=0.2)
        self.assertEqual(X.assess(r).status, "clean")                            # 0.5 ~ 0.2 * sqrt(7): just noise
        r = _res()
        r.primary = _m(extrema=6)
        self.assertEqual(X.assess(r).status, "ringing")
        slow = X.assess(W.StepTestResult(kp=1, ki=1, loop="afl",
                                         failure="unmeasurable: the primary channel did not follow the step"))
        self.assertEqual(slow.status, "slow")                                    # not lost: no skipping beyond it
        lost = X.assess(W.StepTestResult(kp=1, ki=1, loop="afl", failure="aborted: amplitude collapsed"))
        self.assertEqual(lost.status, "lost")
        self.assertTrue(math.isinf(lost.speed_s))

    def test_a_simulated_amplitude_loop_is_assessed_per_direction(self):
        plan = afl_plan(step=0.05)
        ct, _ = record_afl(plan, 8.9e7, 8900)
        a = X.assess(W.analyze_test(ct), ring_down_s=0.32)
        self.assertNotEqual(a.status, "lost")
        self.assertFalse(math.isnan(a.up.rise_s))
        self.assertFalse(math.isnan(a.down.rise_s))


class Recovery(unittest.TestCase):
    def test_amplitude_loop_recovery(self):
        ref = X.reference_from(np.full(200, 1.0), np.full(200, 0.02))
        rng = np.random.default_rng(0)
        y, u = 1.0 + 0.001 * rng.standard_normal(200), 0.02 + 1e-4 * rng.standard_normal(200)
        self.assertEqual(X.recovered("afl", y, u, ref), (True, ""))
        ok, why = X.recovered("afl", y * 1.05, u, ref)
        self.assertFalse(ok)
        self.assertIn("+5.0 % from the baseline", why)
        ok, why = X.recovered("afl", np.linspace(0.9, 1.0, 200), u, ref)       # still climbing
        self.assertFalse(ok)
        self.assertIn("still moving", why)
        self.assertFalse(X.recovered("afl", y, u * 0.0, ref)[0])                 # Drive not back
        noisy = 1.0 + 0.03 * rng.standard_normal(200)                             # 3 % noise, no drift: steady
        self.assertTrue(X.steady("afl", noisy, u)[0])

    def test_pll_recovery(self):
        ref = X.reference_from(np.zeros(200), np.full(200, 0.3))
        rng = np.random.default_rng(1)
        ph, df = 0.05 * rng.standard_normal(200), 0.3 + 0.005 * rng.standard_normal(200)
        self.assertTrue(X.recovered("pll", ph, df, ref)[0])
        self.assertIn("Phase", X.recovered("pll", ph + 3, df, ref)[1])
        self.assertIn("df", X.recovered("pll", ph, df + 0.5, ref)[1])

    def test_settled_at_the_first_level(self):
        y, u = np.full(100, 0.95), np.full(100, 0.02)
        self.assertTrue(X.settled_at("afl", y, u, 0.95)[0])
        self.assertFalse(X.settled_at("afl", y, u, 1.10)[0])


def landscape(kp, ki):
    """
    Synthetic amplitude loop (raw gains, manual-like magnitudes): speed grows with Kp and Ki; too much Ki for
    the Kp overshoots; too much gain overall pins Drive at zero on down-steps, then loses the loop.
    Integral-only works but is slow and overshoots above Ki = 2e5.
    """
    if kp == 0:
        rise = 2000.0 / ki
        if ki > 2e6:
            return "lost", rise
        return ("overshoot" if ki > 2e5 else "clean"), rise
    g = kp / 2e8
    rise = 0.05 / g / (1 + ki / 2e4)
    if g > 30:
        return "lost", rise
    if g > 7:
        return "floor", rise
    if ki / kp > 2e-3:
        return "overshoot", rise
    return "clean", rise


def synthetic(kp, ki):
    status, rise = landscape(kp, ki)
    up, down = _m(rise=rise), _m(rise=rise * 1.3)
    res = _res(kp=kp, ki=ki, up=up, down=down, drive_noise=1e-3 * (1 + abs(kp) / 1e8))
    if status == "overshoot":
        res.primary_falling = _m(rise=rise, overshoot=0.4)
    elif status == "floor":
        res.drive_floor_frac = 0.3
    elif status == "lost":
        res = W.StepTestResult(kp=kp, ki=ki, loop="afl", failure="aborted: amplitude collapsed")
    return X.assess(res)


class Exploring(unittest.TestCase):
    def run_all(self, region, plan=X.ExplorePlan(), limit=200):
        ex = X.Explorer(region, plan)
        seen = []
        while True:
            p = ex.next()
            if p is None:
                break
            seen.append(p)
            ex.record(synthetic(p.kp, p.ki))
            self.assertLess(len(seen), limit, "exploration does not terminate")
        return ex, seen

    def test_grid_has_decades_and_an_integral_only_column(self):
        r = X.SearchRegion(2e8, 2e4, span_decades=1)
        self.assertEqual(r.kps(), [0.0, 2e7, 2e8, 2e9])
        self.assertEqual([round(v) for v in r.kis()], [2000, 20000, 200000])
        self.assertEqual(len(r.points()), 12)
        self.assertEqual(len(X.SearchRegion(2e8, 2e4, 1, per_decade=2).kis()), 5)

    def test_the_stages_run_in_order_and_every_point_is_measured_once_except_repeats(self):
        ex, seen = self.run_all(X.SearchRegion(2e8, 2e4, span_decades=1))
        stages = [p.stage for p in seen]
        self.assertEqual(stages[0], "baseline")
        order = ["baseline", "grid", "edge", "ratio", "repeat"]
        self.assertEqual(stages, sorted(stages, key=order.index))               # stages never interleave
        keys = [(p.kp, p.ki) for p in seen if p.stage != "repeat"]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(ex.done)

    def test_more_aggressive_cells_than_a_lost_one_are_skipped(self):
        ex, seen = self.run_all(X.SearchRegion(2e8, 2e4, span_decades=2))
        lost = [p for p in ex.points().values() if p.status == "lost"]
        self.assertTrue(lost)
        for k, why in ex.skipped.items():
            self.assertIn("where the loop was lost", why)
        self.assertNotIn(X._key(2e10, 2e6), {X._key(p.kp, p.ki) for p in seen})

    def test_bisection_locates_the_edge_along_g(self):
        ex, seen = self.run_all(X.SearchRegion(2e8, 2e4, span_decades=1))
        edge = [p for p in seen if p.stage == "edge"]
        self.assertTrue(edge)
        ratios = {X._key(1.0, p.ki / p.kp)[1] for p in edge if p.kp != 0}
        self.assertTrue(ratios)
        for ratio in ratios:
            on = X.along_ratio(ex.points(), ratio)
            clean = [p.kp for p in on if p.clean]
            bad = [p.kp for p in on if not p.clean and p.kp > max(clean)]
            self.assertTrue(bad, f"no edge found for Ki/Kp = {ratio}")
            self.assertLessEqual(min(bad) / max(clean), 1.25 + 1e-9)             # bracketed to the tolerance
            self.assertLess(max(clean), 7 * 2e8 * 1.0001)
            self.assertGreater(min(bad), 7 * 2e8)                                # the true edge: g = 7

    def test_after_bisection_the_backed_off_point_is_measured(self):
        ex, seen = self.run_all(X.SearchRegion(2e8, 2e4, span_decades=1))
        backs = [p for p in seen if "back off" in p.why]
        self.assertTrue(backs)
        for p in backs:
            s = ex.points()[X._key(p.kp, p.ki)]
            self.assertFalse(X.near_edge(s, ex.points(), 1.5))                    # it has the margin by construction

    def test_trade_off_and_repeats(self):
        ex, seen = self.run_all(X.SearchRegion(2e8, 2e4, span_decades=1))
        to = X.trade_off(ex.points())
        self.assertIsNotNone(to.fastest)
        self.assertTrue(to.fastest.clean)
        self.assertGreaterEqual(to.fastest.n, 3)                                  # measured, then repeated twice
        self.assertLessEqual(to.quietest.speed_s, 1.5 * to.fastest.speed_s)
        self.assertLessEqual(to.quietest.noise, to.fastest.noise)
        for p in to.front:
            self.assertTrue(p.clean)
        for p in (to.fastest, to.quietest):                                       # never right at the edge
            self.assertFalse(X.near_edge(p, ex.points(), 1.5))
        self.assertTrue(to.at_edge)                                              # the bisection's last clean points

    def test_skipping_a_proposal_counts_as_not_clean(self):
        ex = X.Explorer(X.SearchRegion(2e8, 2e4, span_decades=1))
        p = ex.next()
        ex.skip("beyond the safety limit")
        self.assertIn(X._key(p.kp, p.ki), ex.skipped)
        self.assertEqual(ex.assessments, [])
        self.assertIsNotNone(ex.next())

    def test_along_kp_is_the_ratio_view(self):
        ex, _ = self.run_all(X.SearchRegion(2e8, 2e4, span_decades=1))
        row = X.along_kp(ex.points(), 2e8)
        self.assertGreaterEqual(len(row), 3)
        self.assertEqual([p.ki for p in row], sorted(p.ki for p in row))


if __name__ == "__main__":
    unittest.main()
