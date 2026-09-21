"""Tests for tuning.identify: recover a hidden raw->physical gain mapping from noisy trials."""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tuning import metrics as M
from sxm_ncafm_control.tuning.identify import identify_scale
from sxm_ncafm_control.tuning.simulator import PLLSetup, SXMScale, run_step_trial
from sxm_ncafm_control.tuning.trial import StepProtocol

PROTO = StepProtocol(step_hz=1.0, hold_s=2.0, n_holds=7)
SETUP = PLLSetup()
HIDDEN = SXMScale(kp_hz_per_deg=0.0040, ki_hz_per_deg_s=0.00020)     # 1.4x and 0.57x the default guess
GAINS = [(-100, -1e3), (-100, -1e4), (-100, -3e4), (-60, -6e3)]       # like the manual's Ki sweep + one more


def measured(scale=HIDDEN, seed0=100, gains=GAINS):
    return [run_step_trial(kp, ki, PROTO, SETUP, scale, seed=seed0 + i) for i, (kp, ki) in enumerate(gains)]


def overshoot(scale, kp, ki):
    r = run_step_trial(kp, ki, PROTO, PLLSetup(phase_noise_deg_rthz=0.0), scale, seed=0)
    dt = float(np.median(np.diff(r.t)))
    g, d, _, _ = M.average_steps(r.t, r.df, r.protocol.step_times, 1.0, 1.9, dt, signs=np.sign(r.protocol.step_sizes))
    return M.step_response_metrics(g, d, 0.0).overshoot


class Identify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials = measured()
        cls.fit = identify_scale(cls.trials, SETUP, initial=SXMScale())      # default guess is well off

    def test_recovers_both_scales(self):
        self.assertAlmostEqual(self.fit.scale.kp_hz_per_deg / HIDDEN.kp_hz_per_deg, 1.0, delta=0.15)
        self.assertAlmostEqual(self.fit.scale.ki_hz_per_deg_s / HIDDEN.ki_hz_per_deg_s, 1.0, delta=0.15)

    def test_fit_is_good(self):
        self.assertLess(self.fit.rms_residual, 0.08)
        self.assertEqual(len(self.fit.per_trial_rms), len(GAINS))

    def test_identified_model_predicts_an_untested_gain(self):
        """The point of identifying: predict a gain that was never run on the 'instrument'."""
        truth = overshoot(HIDDEN, -100, -2e4)
        pred = overshoot(self.fit.scale, -100, -2e4)
        naive = overshoot(SXMScale(), -100, -2e4)                              # the un-identified default
        self.assertLess(abs(pred - truth), 0.08)
        self.assertLess(abs(pred - truth), abs(naive - truth))

    def test_invariant_to_the_instruments_sign_convention(self):
        flipped = []
        for r in self.trials:
            flipped.append(type(r)(kp=r.kp, ki=r.ki, t=r.t, df=-r.df, phase=-r.phase, protocol=r.protocol,
                                   locked=r.locked, meta=r.meta))
        fit = identify_scale(flipped, SETUP, initial=SXMScale())
        self.assertAlmostEqual(fit.scale.kp_hz_per_deg, self.fit.scale.kp_hz_per_deg, delta=0.05 * self.fit.scale.kp_hz_per_deg)

    def test_needs_two_trials(self):
        with self.assertRaises(ValueError):
            identify_scale(self.trials[:1], SETUP)


if __name__ == "__main__":
    unittest.main()
