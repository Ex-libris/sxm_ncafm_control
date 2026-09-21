"""
Identify the unknown raw-SXM-gain -> physical-gain mapping from measured trials.

SXM's Kp/Ki are in arbitrary internal units. Given a few real step-response
trials at *known raw gains*, this fits the two scale factors of the loop model
in :mod:`tuning.simulator` so that its noise-free response reproduces what was
measured. Once identified, the model can predict the response (and stability)
of gains that were never tried on the instrument - the basis for a
model-assisted search and for expressing targets as physical performance
(bandwidth, damping) rather than raw numbers.

The sensor (f0, Q) and the lock-in time constant are inputs: Q and f0 come from
the resonance sweep, the TimeConstant is a setting.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize

from . import metrics as M
from .simulator import PLLSetup, SXMScale, run_step_trial
from .trial import StepProtocol, TrialResult


@dataclass(frozen=True)
class Identified:
    scale: SXMScale
    rms_residual: float          # of the fit, in units of the (normalised) response
    n_evaluations: int
    per_trial_rms: tuple         # one entry per input trial


def _peak_sign(x) -> float:
    """+1 or -1: the sign of the largest-magnitude sample."""
    return 1.0 if x[int(np.argmax(np.abs(x)))] >= 0 else -1.0


def _grid(fit_window_s: float, n: int = 80) -> np.ndarray:
    """Log-spaced times after the step: resolves the ms-scale rise and the seconds-scale tail."""
    return np.geomspace(2e-3, fit_window_s, n)


def measured_response(result: TrialResult, fit_window_s: float):
    """
    Ensemble-averaged, sign-folded step responses of a measured trial, on the fit grid.

    Returns (df_normalised, phase_normalised, phase_peak): the df response divided by the commanded
    step, and the phase transient divided by its own peak. Both are sign-normalised (df to a positive
    final value, phase to a positive peak), so the fit does not depend on the instrument's sign
    conventions.
    """
    p = result.protocol
    dt = float(np.median(np.diff(result.t)))
    signs = np.sign(p.step_sizes)
    post = min(fit_window_s, p.hold_s - 0.05)
    g, df_mean, _, _ = M.average_steps(result.t, result.df, p.step_times, 1.0, post, dt, signs=signs)
    _, ph_mean, _, _ = M.average_steps(result.t, result.phase, p.step_times, 1.0, post, dt, signs=signs)
    tt = _grid(post)
    z_df = np.interp(tt, g, df_mean) / (2.0 * p.step_hz)
    if np.median(z_df[-10:]) < 0:
        z_df = -z_df
    ph = np.interp(tt, g, ph_mean)
    ph = ph * _peak_sign(ph)
    peak = float(np.max(np.abs(ph))) or 1.0
    return z_df, ph / peak, peak


def _model_response(kp, ki, protocol, setup, scale, fit_window_s):
    """Noise-free single-step response of the model on the same grid."""
    post = min(fit_window_s, protocol.hold_s - 0.05)
    proto = StepProtocol(step_hz=protocol.step_hz, hold_s=protocol.hold_s, n_holds=2)
    quiet = PLLSetup(f0=setup.f0, q=setup.q, lockin_tau=setup.lockin_tau, lockin_stages=setup.lockin_stages,
                     phase_noise_deg_rthz=0.0, dt=setup.dt)
    r = run_step_trial(kp, ki, proto, quiet, scale, seed=0)
    if not r.locked:
        return None
    dt = float(np.median(np.diff(r.t)))
    g, df_mean, _, _ = M.average_steps(r.t, r.df, proto.step_times, 1.0, post, dt, signs=[1.0])
    _, ph_mean, _, _ = M.average_steps(r.t, r.phase, proto.step_times, 1.0, post, dt, signs=[1.0])
    tt = _grid(post)
    ph = np.interp(tt, g, ph_mean)
    return np.interp(tt, g, df_mean) / (2.0 * protocol.step_hz), ph * _peak_sign(ph)


def identify_scale(trials: Sequence[TrialResult], setup: PLLSetup = PLLSetup(),
                   initial: SXMScale = SXMScale(), fit_window_s: float = 1.5) -> Identified:
    """
    Fit ``SXMScale`` to measured trials.

    Parameters
    ----------
    trials : measured :class:`TrialResult` objects at known raw gains; use several
        different gains (e.g. the manual's Ki sweep at fixed Kp) so both scales are constrained.
    setup : sensor and lock-in settings the trials were taken with.
    initial : starting guess; a coarse grid around it is searched first, so it may be well off.
    """
    if len(trials) < 2:
        raise ValueError("need at least two trials at different gains")
    meas = []
    for tr in trials:
        z, ph_n, peak = measured_response(tr, fit_window_s)
        meas.append((tr, z, ph_n, peak))

    n_eval = [0]
    bad = 5.0        # residual assigned to a model that loses lock

    def residuals(theta):
        n_eval[0] += 1
        scale = SXMScale(math.exp(theta[0]), math.exp(theta[1]))
        out = []
        for tr, z, ph_n, peak in meas:
            m = _model_response(tr.kp, tr.ki, tr.protocol, setup, scale, fit_window_s)
            if m is None:
                out.append(np.full(2 * len(z), bad))
                continue
            z_m, ph_m = m
            out.append(np.concatenate([z - z_m, ph_n - ph_m / peak]))
        return np.concatenate(out)

    # coarse grid, then Levenberg-Marquardt-style refinement from the best cell
    x0 = np.log([initial.kp_hz_per_deg, initial.ki_hz_per_deg_s])
    best, best_cost = x0, math.inf
    for a in np.log([0.25, 0.5, 1.0, 2.0, 4.0]):
        for b in np.log([0.25, 0.5, 1.0, 2.0, 4.0]):
            th = x0 + np.array([a, b])
            cost = float(np.sum(residuals(th) ** 2))
            if cost < best_cost:
                best, best_cost = th, cost
    fit = optimize.least_squares(residuals, best, method="trf", diff_step=0.02, x_scale=1.0, max_nfev=60)

    final = residuals(fit.x)
    per = []
    k = 0
    for _, z, _, _ in meas:
        n = 2 * len(z)
        per.append(float(np.sqrt(np.mean(final[k:k + n] ** 2))))
        k += n
    scale = SXMScale(kp_hz_per_deg=math.exp(fit.x[0]), ki_hz_per_deg_s=math.exp(fit.x[1]))
    return Identified(scale=scale, rms_residual=float(np.sqrt(np.mean(final ** 2))),
                      n_evaluations=n_eval[0], per_trial_rms=tuple(per))
