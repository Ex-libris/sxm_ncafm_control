"""Tests for tuning.metrics against analytic responses (run: see CLAUDE.md)."""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tuning import metrics as M


def first_order(t, tau):
    return 1.0 - np.exp(-t / tau)


def second_order(t, zeta, wn):
    wd = wn * math.sqrt(1 - zeta ** 2)
    return 1.0 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / math.sqrt(1 - zeta ** 2) * np.sin(wd * t))


def with_pre(t_post, y_post, pre_s=1.0, dt=None):
    """Prepend a settled 0 level so the response starts at t = 0 (the step time)."""
    dt = dt or (t_post[1] - t_post[0])
    t_pre = np.arange(-pre_s, 0.0, dt)
    return np.concatenate([t_pre, t_post]), np.concatenate([np.zeros_like(t_pre), y_post])


class StepMetricsAnalytic(unittest.TestCase):
    dt = 1e-3

    def test_first_order_noise_free(self):
        tau = 0.1
        tp = np.arange(0, 3.0, self.dt)
        t, y = with_pre(tp, first_order(tp, tau))
        m = M.step_response_metrics(t, y, 0.0)
        self.assertAlmostEqual(m.rise_time, tau * math.log(9), delta=0.003)       # 2.197 tau
        self.assertAlmostEqual(m.settling_time, tau * math.log(20), delta=0.004)  # 5 % band: 2.996 tau
        self.assertEqual(m.overshoot, 0.0)
        self.assertEqual(m.n_extrema, 0)
        self.assertTrue(m.settled)
        self.assertAlmostEqual(m.step_size, 1.0, places=3)

    def test_second_order_overshoot_and_damping(self):
        zeta, wn = 0.3, 20.0
        tp = np.arange(0, 4.0, self.dt)
        t, y = with_pre(tp, second_order(tp, zeta, wn))
        m = M.step_response_metrics(t, y, 0.0)
        expected_os = math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))       # 0.372
        self.assertAlmostEqual(m.overshoot, expected_os, delta=0.01)
        self.assertAlmostEqual(m.damping_ratio, zeta, delta=0.03)
        self.assertGreaterEqual(m.n_extrema, 2)

    def test_well_damped_has_no_overshoot(self):
        tp = np.arange(0, 3.0, self.dt)
        t, y = with_pre(tp, second_order(tp, 0.95, 20.0))
        m = M.step_response_metrics(t, y, 0.0)
        self.assertLess(m.overshoot, 0.01)

    def test_noisy_first_order_within_tolerance(self):
        tau, rng = 0.1, np.random.default_rng(1)
        tp = np.arange(0, 3.0, self.dt)
        t, y = with_pre(tp, first_order(tp, tau))
        y = y + rng.normal(0, 0.03, len(y))                                      # 3 % noise
        m = M.step_response_metrics(t, y, 0.0)
        self.assertAlmostEqual(m.rise_time, tau * math.log(9), delta=0.2 * tau * math.log(9))
        self.assertAlmostEqual(m.settling_time, tau * math.log(20), delta=0.35 * tau * math.log(20))
        self.assertEqual(m.overshoot, 0.0, "noise must not be reported as overshoot")
        self.assertAlmostEqual(m.noise_rms, 0.03, delta=0.006)

    def test_negative_step_and_offset(self):
        tau = 0.2
        tp = np.arange(0, 3.0, self.dt)
        t, y = with_pre(tp, 5.0 - 2.0 * first_order(tp, tau))                    # 5 -> 3
        y[t < 0] = 5.0
        m = M.step_response_metrics(t, y, 0.0)
        self.assertAlmostEqual(m.step_size, -2.0, places=2)
        self.assertAlmostEqual(m.y_initial, 5.0, places=3)
        self.assertAlmostEqual(m.rise_time, tau * math.log(9), delta=0.005)

    def test_too_slow_response_is_flagged_unsettled(self):
        tp = np.arange(0, 2.0, self.dt)
        t, y = with_pre(tp, first_order(tp, 5.0))                                # tau = 5 s, hold = 2 s
        m = M.step_response_metrics(t, y, 0.0)
        self.assertFalse(m.settled)
        self.assertTrue(math.isinf(m.settling_time))

    def test_steady_state_error_uses_target(self):
        tp = np.arange(0, 3.0, self.dt)
        t, y = with_pre(tp, 0.9 * first_order(tp, 0.1))                          # reaches 0.9 of the command
        m = M.step_response_metrics(t, y, 0.0, target_step=1.0)
        self.assertAlmostEqual(m.steady_state_error, -0.1, places=2)

    def test_step_buried_in_noise_is_rejected(self):
        rng = np.random.default_rng(2)
        t = np.arange(-1.0, 3.0, self.dt)
        y = rng.normal(0, 1.0, len(t))
        with self.assertRaises(M.StepNotDetectable):
            M.step_response_metrics(t, y, 0.0)

    def test_too_little_data(self):
        with self.assertRaises(M.StepNotDetectable):
            M.step_response_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], 1.0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            M.step_response_metrics([0.0, 2.0, 1.0], [0.0, 1.0, 1.0], 1.0)       # not increasing


class AverageSteps(unittest.TestCase):
    dt = 1e-3

    def _train(self, tau=0.1, noise=0.2, n=6, hold=1.5, seed=3):
        """A +-1 toggling train (steps of size 2): high SNR only after averaging."""
        rng = np.random.default_rng(seed)
        t = np.arange(0, n * hold, self.dt)
        y = np.empty_like(t)
        level, step_times = -1.0, []
        y[:] = -1.0
        for k in range(1, n):
            ts = k * hold
            step_times.append(ts)
            new = -level
            m = t >= ts
            y[m] = new + (level - new) * np.exp(-(t[m] - ts) / tau)
            level = new
        return t, y + rng.normal(0, noise, len(t)), step_times

    def test_up_and_down_steps_add_coherently(self):
        t, y, st = self._train()
        grid, mean, std, n = M.average_steps(t, y, st, pre_s=0.5, post_s=1.4, dt=self.dt)
        self.assertEqual(n, 5)
        self.assertAlmostEqual(float(np.median(mean[grid > 1.0])), 2.0, delta=0.05)   # all folded to 'up'
        self.assertAlmostEqual(float(np.median(mean[grid < 0])), 0.0, delta=0.05)

    def test_averaging_reduces_noise_by_sqrt_n(self):
        t, y, st = self._train(noise=0.3)
        grid, mean, std, n = M.average_steps(t, y, st, pre_s=0.5, post_s=1.4, dt=self.dt)
        tail_single = M.detrended_std(np.interp(st[0] + grid[grid > 1.0], t, y))
        tail_avg = M.detrended_std(mean[grid > 1.0])
        self.assertLess(tail_avg, tail_single / math.sqrt(n) * 1.5)

    def test_metrics_on_averaged_response(self):
        tau = 0.1
        t, y, st = self._train(tau=tau, noise=0.3)
        grid, mean, _, _ = M.average_steps(t, y, st, pre_s=0.5, post_s=1.4, dt=self.dt)
        m = M.step_response_metrics(grid, mean, 0.0)
        self.assertAlmostEqual(m.rise_time, tau * math.log(9), delta=0.05)
        self.assertAlmostEqual(m.step_size, 2.0, delta=0.05)

    def test_explicit_signs_for_signals_that_return_to_rest(self):
        # a decaying spike whose sign follows the step direction (like a loop error signal)
        t = np.arange(0, 6.0, self.dt)
        e = np.zeros_like(t)
        st, signs = [1.5, 3.0, 4.5], [+1, -1, +1]
        for ts, s in zip(st, signs):
            m = t >= ts
            e[m] += s * np.exp(-(t[m] - ts) / 0.1)
        grid, mean, _, n = M.average_steps(t, e, st, pre_s=0.3, post_s=1.2, dt=self.dt, signs=signs)
        self.assertEqual(n, 3)
        self.assertGreater(float(mean[np.argmin(np.abs(grid - 0.001))]), 0.9)         # all spikes positive

    def test_no_usable_steps(self):
        t = np.arange(0, 1.0, self.dt)
        with self.assertRaises(M.StepNotDetectable):
            M.average_steps(t, np.zeros_like(t), [0.01], pre_s=0.5, post_s=0.5, dt=self.dt)


class ErrorTransient(unittest.TestCase):
    dt = 1e-3

    def test_exponential_decay(self):
        a, tau = 20.0, 0.15
        tp = np.arange(0, 3.0, self.dt)
        t, e = with_pre(tp, a * np.exp(-tp / tau))
        m = M.error_transient_metrics(t, e, 0.0)
        self.assertAlmostEqual(m.peak, a, delta=0.2)
        self.assertAlmostEqual(m.time_to_peak, 0.0, delta=0.005)
        self.assertAlmostEqual(m.iae, a * tau, delta=0.05 * a * tau)                  # integral of a*exp(-t/tau)
        self.assertAlmostEqual(m.decay_time, tau * math.log(20), delta=0.02)           # to 5 % of the peak
        self.assertEqual(m.sign_changes, 0)

    def test_undershoot_is_counted(self):
        tp = np.arange(0, 3.0, self.dt)
        e = np.exp(-tp / 0.1) - 0.4 * np.exp(-tp / 0.4)                               # goes negative after the peak
        t, e = with_pre(tp, e)
        m = M.error_transient_metrics(t, e, 0.0)
        self.assertGreaterEqual(m.sign_changes, 1)

    def test_never_decaying_is_inf(self):
        tp = np.arange(0, 2.0, self.dt)
        t, e = with_pre(tp, np.exp(-tp / 50.0))
        m = M.error_transient_metrics(t, e, 0.0)
        self.assertTrue(math.isinf(m.decay_time))


class NoiseMetrics(unittest.TestCase):
    def test_white_noise_band_rms(self):
        fs, sigma = 2000.0, 0.5
        y = np.random.default_rng(4).normal(0, sigma, 40000)
        self.assertAlmostEqual(M.band_rms(y, fs, 0.0, fs / 2), sigma, delta=0.03)
        self.assertAlmostEqual(M.band_rms(y, fs, 0.0, fs / 4), sigma / math.sqrt(2), delta=0.03)

    def test_noise_rms_ignores_drift(self):
        y = np.random.default_rng(5).normal(0, 0.2, 5000) + np.linspace(0, 50, 5000)
        self.assertAlmostEqual(M.noise_rms(y), 0.2, delta=0.02)

    def test_pixel_noise_averages_down(self):
        fs = 1000.0
        t = np.arange(0, 20.0, 1 / fs)
        y = np.random.default_rng(6).normal(0, 1.0, len(t))
        self.assertAlmostEqual(M.pixel_noise(t, y, 0.1), 0.1, delta=0.02)             # 100 samples per pixel

    def test_pixel_noise_needs_enough_blocks(self):
        t = np.arange(0, 0.5, 1e-3)
        self.assertTrue(math.isnan(M.pixel_noise(t, np.zeros_like(t), 0.1)))

    def test_block_mean_shapes(self):
        t = np.arange(0, 1.0, 0.01)
        tc, ym = M.block_mean(t, np.arange(len(t), dtype=float), 0.1)
        self.assertEqual(len(tc), 10)
        self.assertAlmostEqual(float(ym[0]), 4.5, places=6)


class TransientExcursion(unittest.TestCase):
    def test_peak_and_rms_of_a_known_overshoot(self):
        # already at the final level, a bump to 1.3, then back to a flat noise-free tail at 1.0
        y = np.concatenate([np.full(20, 1.0), np.linspace(1.0, 1.3, 30), np.full(200, 1.0)])
        peak, rms = M.transient_excursion(y, y_final=1.0)
        self.assertAlmostEqual(peak, 0.3, places=6)
        self.assertLess(rms, peak)                        # most samples are at the flat, zero-deviation tail
        self.assertGreater(rms, 0.0)

    def test_zero_when_already_at_the_final_value(self):
        y = np.full(100, 2.5)
        peak, rms = M.transient_excursion(y, y_final=2.5)
        self.assertEqual((peak, rms), (0.0, 0.0))

    def test_empty_is_nan_not_a_crash(self):
        peak, rms = M.transient_excursion(np.array([]), y_final=1.0)
        self.assertTrue(math.isnan(peak) and math.isnan(rms))


if __name__ == "__main__":
    unittest.main()
