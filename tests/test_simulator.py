"""Tests for tuning.simulator: physics sanity and the properties the tuner relies on."""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tuning import metrics as M
from sxm_ncafm_control.tuning.simulator import PLLSetup, SXMScale, run_step_trial, simulate_pll
from sxm_ncafm_control.tuning.trial import StepProtocol

PROTO = StepProtocol(step_hz=1.0, hold_s=3.0, n_holds=7)
QUIET = PLLSetup(phase_noise_deg_rthz=0.0)


def responses(r):
    """Ensemble df step metrics and phase-transient metrics of a trial."""
    dt = float(np.median(np.diff(r.t)))
    signs = np.sign(r.protocol.step_sizes)
    g, d, _, _ = M.average_steps(r.t, r.df, r.protocol.step_times, 1.0, 2.9, dt, signs=signs)
    m = M.step_response_metrics(g, d, 0.0, target_step=2.0)
    g, p, _, _ = M.average_steps(r.t, r.phase, r.protocol.step_times, 1.0, 2.9, dt, signs=signs)
    return m, M.error_transient_metrics(g, p, 0.0)


class Protocol(unittest.TestCase):
    def test_levels_alternate_and_steps_are_twice_the_offset(self):
        p = StepProtocol(step_hz=1.0, hold_s=2.0, n_holds=5)
        self.assertEqual(p.levels, [-1.0, 1.0, -1.0, 1.0, -1.0])
        self.assertEqual(p.step_sizes, [2.0, -2.0, 2.0, -2.0])
        self.assertEqual(p.step_times, [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(p.duration, 10.0)


class Physics(unittest.TestCase):
    def test_holds_lock_at_a_constant_offset(self):
        n = int(2.0 / QUIET.dt)
        df, ph, locked = simulate_pll(-100, -1e4, np.full(n, 0.7), QUIET)
        self.assertTrue(locked)
        self.assertLess(abs(df[-1] - 0.7), 1e-6)         # starts locked: delta = dF
        self.assertLess(abs(ph[-1]), 1e-6)

    def test_df_tracks_the_commanded_step(self):
        r = run_step_trial(-100, -1e4, PROTO, QUIET, seed=0)
        self.assertTrue(r.locked)
        m, _ = responses(r)
        self.assertAlmostEqual(m.step_size, 2.0, delta=0.02)          # rectangular, full size
        self.assertLess(abs(m.steady_state_error), 0.01)

    def test_wrong_sign_gains_run_away(self):
        r = run_step_trial(+100, +1e4, PROTO, QUIET, seed=0)          # the manual: both must be negative
        self.assertFalse(r.locked)

    def test_huge_gains_lose_lock(self):
        self.assertFalse(run_step_trial(-100, -2e6, PROTO, QUIET, seed=0).locked)

    def test_integral_gain_shapes_the_response_like_the_manual_figure(self):
        """Kp=-100 fixed: Ki=-1e3 slow phase tail / -1e4 fast / -5e4 overshooting."""
        out = {}
        for ki in (-1e3, -1e4, -5e4):
            r = run_step_trial(-100, ki, PROTO, QUIET, seed=0)
            self.assertTrue(r.locked, ki)
            out[ki] = responses(r)
        (m1, e1), (m2, e2), (m3, e3) = out[-1e3], out[-1e4], out[-5e4]
        self.assertGreater(e1.decay_time, 4 * e2.decay_time)          # slow tail at low Ki
        self.assertLess(m1.overshoot, 0.02)                            # ... but no overshoot
        self.assertLess(m2.overshoot, 0.15)                            # 'good': mild
        self.assertGreater(m3.overshoot, 0.25)                         # large Ki overshoots
        self.assertLess(m1.overshoot, m2.overshoot)
        self.assertLess(m2.overshoot, m3.overshoot)

    def test_higher_gains_are_faster(self):
        rises = []
        for mult in (0.25, 0.5, 1.0, 2.0):
            r = run_step_trial(-100 * mult, -1e4 * mult, PROTO, QUIET, seed=0)
            self.assertTrue(r.locked)
            rises.append(responses(r)[0].rise_time)
        self.assertTrue(all(a > b for a, b in zip(rises, rises[1:])), rises)


class Noise(unittest.TestCase):
    @staticmethod
    def tail_noise(r):
        p = r.protocol
        vals = []
        for k in range(1, p.n_holds):
            m = (r.t >= p.hold_s * k + 0.6 * p.hold_s) & (r.t < p.hold_s * (k + 1))
            vals.append(r.df[m] - np.median(r.df[m]))
        return float(np.std(np.concatenate(vals)))

    def test_noise_grows_with_gain(self):
        pts = []
        for mult in (0.25, 0.5, 1.0, 2.0):
            r = run_step_trial(-100 * mult, -1e4 * mult, PROTO, seed=7)
            self.assertTrue(r.locked)
            pts.append((mult, self.tail_noise(r)))
        self.assertTrue(all(a[1] < b[1] for a, b in zip(pts, pts[1:])), pts)
        slope = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]
        self.assertGreater(slope, 0.9)                                 # at least linear in the gain
        self.assertLess(slope, 1.6)

    def test_longer_lockin_time_constant_reduces_noise(self):
        n = [self.tail_noise(run_step_trial(-100, -1e4, PROTO, PLLSetup(lockin_tau=tau), seed=8))
             for tau in (0.5e-3, 2e-3, 4e-3)]
        self.assertGreater(n[0], n[1])
        self.assertGreater(n[1], n[2])

    def test_seeds(self):
        a = run_step_trial(-100, -1e4, PROTO, seed=1)
        b = run_step_trial(-100, -1e4, PROTO, seed=1)
        c = run_step_trial(-100, -1e4, PROTO, seed=2)
        self.assertTrue(np.array_equal(a.df, b.df))
        self.assertFalse(np.array_equal(a.df, c.df))


class Output(unittest.TestCase):
    def test_shapes_and_time_axis(self):
        r = run_step_trial(-100, -1e4, PROTO, seed=0, out_fs=1000.0)
        self.assertEqual(len(r.t), len(r.df))
        self.assertEqual(len(r.t), len(r.phase))
        self.assertAlmostEqual(float(np.median(np.diff(r.t))), 1e-3, places=6)
        self.assertAlmostEqual(r.t[-1], PROTO.duration, delta=2e-3)
        self.assertTrue(r.meta["simulated"])


if __name__ == "__main__":
    unittest.main()
