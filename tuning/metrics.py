"""
Step-response and noise metrics for loop tuning.

Everything works on plain ``(t, y)`` arrays (seconds, any unit), so the same
code analyses a live capture, an exported CSV, or simulator output.

Conventions
-----------
* Step metrics follow the usual definitions: rise time = 10 % -> 90 %,
  settling time = last exit from a +-band around the final value (default
  5 %), overshoot = peak beyond the final value as a fraction of the step.
* Measurements are noise aware: the effective settling band is never
  narrower than a few noise sigmas, and overshoot/ringing below the noise
  floor is not reported.
* Repeated steps of either sign are folded together with
  :func:`average_steps` before measuring, which cuts the noise by sqrt(n).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal


class StepNotDetectable(ValueError):
    """The data around the step is too short, or the step is buried in noise."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _prep(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError("t and y must be 1-D arrays of the same length")
    if len(t) >= 2 and not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    return t, y


def detrended_std(y) -> float:
    """Standard deviation of y after removing a straight line (drift-tolerant noise estimate)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    return float(np.std(y - np.polyval(coef, x), ddof=1))


def block_mean(t, y, block_s):
    """Average consecutive blocks of ``block_s`` seconds. Returns (t_centres, y_means)."""
    t, y = _prep(t, y)
    idx = np.floor((t - t[0]) / block_s).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    sums_y = np.bincount(idx, weights=y)
    sums_t = np.bincount(idx, weights=t)
    return sums_t[keep] / counts[keep], sums_y[keep] / counts[keep]


def _first_crossing(t, z, level):
    """Time at which z first reaches ``level`` (linear interpolation), or nan."""
    above = np.flatnonzero(z >= level)
    if len(above) == 0:
        return math.nan
    i = above[0]
    if i == 0:
        return float(t[0])
    z0, z1 = z[i - 1], z[i]
    frac = 0.0 if z1 == z0 else (level - z0) / (z1 - z0)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


# ---------------------------------------------------------------------------
# step response
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepMetrics:
    """Metrics of one (usually ensemble-averaged) step response."""

    step_size: float          # y_final - y_initial (signed, in y units)
    y_initial: float
    y_final: float
    delay_time: float         # step -> 10 %  [s]
    rise_time: float          # 10 % -> 90 %  [s]
    settling_time: float      # step -> last exit from +-band  [s]; inf if it never settles
    settled: bool
    overshoot: float          # fraction of |step| beyond the final value (0 if none above noise)
    n_extrema: int            # number of ringing extrema above the noise floor after the step
    damping_ratio: float      # from the first two extrema; nan if fewer than two
    steady_state_error: float  # (step - target)/target; nan if no target given
    noise_rms: float          # noise of the settled tail, in y units
    band: float               # effective settling band actually used (fraction of the step)


def step_response_metrics(t, y, t_step, *, hold_end=None, pre_tail_frac=0.3,
                          final_frac=0.25, band=0.05, target_step=None, step_override=None) -> StepMetrics:
    """
    Measure a single step response.

    Parameters
    ----------
    t, y : arrays
        The response, including some settled data *before* the step.
    t_step : float
        Time of the step (start of the response).
    hold_end : float, optional
        End of the hold after the step (default: end of data). Everything in
        ``[t_step, hold_end]`` is the response; its last ``final_frac`` defines
        the final value.
    pre_tail_frac : float
        Fraction of the pre-step data (its latest part) used as the initial level.
    band : float
        Nominal settling band as a fraction of the step (0.05 = 5 %).
    target_step : float, optional
        Commanded step size, for the steady-state error.
    step_override : float, optional
        Size of the step to normalise by (only its magnitude is used; the direction comes from the
        data). Use it when the hold is too short for the response to settle, so the tail of the hold
        is not the final value: e.g. the expected amplitude change of a Ref step.

    Raises
    ------
    StepNotDetectable
        Too little data, or the step is smaller than ~4 noise sigmas.
    """
    t, y = _prep(t, y)
    if hold_end is None:
        hold_end = float(t[-1])
    pre = y[t < t_step]
    post_mask = (t >= t_step) & (t <= hold_end)
    if len(pre) < 5 or post_mask.sum() < 20:
        raise StepNotDetectable("not enough data around the step")

    tt = t[post_mask] - t_step
    yy = y[post_mask]
    pre_tail = pre[-max(5, int(pre_tail_frac * len(pre))):]
    n_final = max(5, int(final_frac * len(yy)))
    tail = yy[-n_final:]

    y0 = float(np.median(pre_tail))
    y1 = float(np.median(tail))
    step = y1 - y0
    sigma = max(detrended_std(tail), detrended_std(pre_tail))
    if abs(step) < 4.0 * sigma or step == 0.0:
        raise StepNotDetectable("step is not distinguishable from the noise")
    if step_override is not None:      # normalise by the expected size (the direction still comes from the data)
        step = abs(float(step_override)) * (1.0 if step >= 0 else -1.0)

    z = (yy - y0) / step
    sigma_z = sigma / abs(step)

    # Smooth just enough that the noise on z is ~1 % of the step.
    w = int(min(max(1, math.ceil((sigma_z / 0.01) ** 2)), max(1, len(z) // 20)))
    zs = ndimage.uniform_filter1d(z, size=w, mode="nearest") if w > 1 else z
    sigma_zs = sigma_z / math.sqrt(w)

    t10 = _first_crossing(tt, zs, 0.1)
    t90 = _first_crossing(tt, zs, 0.9)
    rise = t90 - t10 if not (math.isnan(t10) or math.isnan(t90)) else math.nan

    overshoot = float(zs.max() - 1.0)
    # The max of ~hundreds of independent smoothed noise samples reaches ~3-4.7 sigma on its own,
    # so only call it overshoot above 5 sigma.
    if overshoot < max(5.0 * sigma_zs, 0.005):
        overshoot = 0.0

    band_eff = max(band, 4.0 * sigma_zs)
    outside = np.flatnonzero(np.abs(zs - 1.0) > band_eff)
    guard = max(3, int(0.05 * len(zs)))
    if len(outside) == 0:
        settling, settled = 0.0, True
    elif outside[-1] >= len(zs) - 1 - guard:
        settling, settled = math.inf, False
    else:
        settling, settled = float(tt[outside[-1] + 1]), True

    # Ringing: alternating extrema of the deviation from the final value.
    e = zs - 1.0
    h = max(4.0 * sigma_zs, 0.05)      # ringing must be material: >= 5 % of the step
    pk_pos, _ = signal.find_peaks(e, height=h, prominence=h)
    pk_neg, _ = signal.find_peaks(-e, height=h, prominence=h)
    extrema = sorted([(int(i), float(e[i])) for i in pk_pos] + [(int(i), float(e[i])) for i in pk_neg])
    damping = math.nan
    if len(extrema) >= 2:
        a = math.log(abs(extrema[0][1]) / abs(extrema[1][1]))
        if a > 0:
            damping = a / math.sqrt(math.pi ** 2 + a ** 2)   # log decrement over a half cycle
        else:
            damping = 0.0                                     # not decaying

    sse = math.nan
    if target_step:
        sse = (step - target_step) / target_step

    return StepMetrics(step_size=step, y_initial=y0, y_final=y1, delay_time=t10, rise_time=rise,
                       settling_time=settling, settled=settled, overshoot=overshoot,
                       n_extrema=len(extrema), damping_ratio=damping, steady_state_error=sse,
                       noise_rms=sigma, band=band_eff)


def average_steps(t, y, step_times, pre_s, post_s, dt, signs=None):
    """
    Fold several steps (of either direction) into one averaged 'up' response.

    Each window ``[t_step - pre_s, t_step + post_s]`` is interpolated onto a
    common grid, its initial level removed, and multiplied by the step's sign
    so down-steps and up-steps add up coherently.

    Parameters
    ----------
    signs : sequence of +-1, optional
        Direction of each step. If omitted it is inferred from the data
        (final level minus initial level), which fails for signals that
        return to their initial level (e.g. a loop's error signal): pass the
        commanded directions for those.

    Returns
    -------
    grid, mean, std, n : arrays / int
        ``grid`` is relative to the step (negative = before it); ``std`` is the
        spread across repeats (zeros if n == 1).
    """
    t, y = _prep(t, y)
    grid = np.arange(-pre_s, post_s, dt)
    segs = []
    for k, ts in enumerate(step_times):
        if ts - pre_s < t[0] or ts + post_s > t[-1]:
            continue
        seg = np.interp(ts + grid, t, y)
        y0 = float(np.median(seg[grid < 0][-max(5, int(0.3 * np.sum(grid < 0))):]))
        if signs is not None:
            s = float(np.sign(signs[k]))
        else:
            y1 = float(np.median(seg[-max(5, int(0.25 * len(seg))):]))
            s = float(np.sign(y1 - y0))
        if s == 0.0:
            continue
        segs.append(s * (seg - y0))
    if not segs:
        raise StepNotDetectable("no usable steps")
    stack = np.vstack(segs)
    std = stack.std(axis=0, ddof=1) if len(segs) > 1 else np.zeros_like(grid)
    return grid, stack.mean(axis=0), std, len(segs)


# ---------------------------------------------------------------------------
# error-signal transient (e.g. the PLL phase after a frequency step)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ErrorMetrics:
    """Transient of a loop's error signal, which should return to its rest value."""

    peak: float          # largest |deviation| after the step, in signal units
    time_to_peak: float  # [s]
    iae: float           # integral of |deviation| dt  [unit * s]
    decay_time: float    # last time |deviation| exceeded band*peak [s]; inf if never decays
    sign_changes: int    # undershoots of the error (a sign of ringing)
    noise_rms: float


def error_transient_metrics(t, e, t_step, *, hold_end=None, final_frac=0.25, band=0.05, rest_value=None) -> ErrorMetrics:
    """
    Measure how an error signal (already sign-folded, see :func:`average_steps`) decays after a step.

    ``rest_value`` is the level the signal returns to. By default it is the median of the last part of
    the window, which is wrong when the hold is too short for the transient to finish: pass the
    known level (e.g. the pre-step baseline) then.
    """
    t, e = _prep(t, e)
    if hold_end is None:
        hold_end = float(t[-1])
    m = (t >= t_step) & (t <= hold_end)
    if m.sum() < 20:
        raise StepNotDetectable("not enough data after the step")
    tt = t[m] - t_step
    ee = e[m]
    n_final = max(5, int(final_frac * len(ee)))
    rest = float(np.median(ee[-n_final:])) if rest_value is None else float(rest_value)
    sigma = detrended_std(ee[-n_final:])
    d = ee - rest
    ad = np.abs(d)
    i_pk = int(np.argmax(ad))
    peak = float(ad[i_pk])
    if peak < 4.0 * sigma:
        raise StepNotDetectable("error transient is not distinguishable from the noise")

    iae = float(np.trapezoid(ad, tt)) if hasattr(np, "trapezoid") else float(np.trapz(ad, tt))
    thr = max(band * peak, 3.0 * sigma)
    above = np.flatnonzero(ad > thr)
    if len(above) == 0:
        decay = 0.0
    elif above[-1] >= len(ad) - 1 - max(3, int(0.05 * len(ad))):
        decay = math.inf
    else:
        decay = float(tt[above[-1] + 1])

    sgn0 = np.sign(d[i_pk])
    opposite = (np.sign(d) == -sgn0) & (ad > thr)
    sign_changes = int(np.sum(np.diff(opposite.astype(int)) == 1) + (1 if opposite[0] else 0))
    return ErrorMetrics(peak=peak, time_to_peak=float(tt[i_pk]), iae=iae, decay_time=decay,
                        sign_changes=sign_changes, noise_rms=sigma)


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
def noise_rms(y) -> float:
    """RMS of y after removing a straight line (drift-tolerant)."""
    return detrended_std(y)


def band_rms(y, fs, f_lo, f_hi) -> float:
    """RMS of the noise inside [f_lo, f_hi] Hz, from a Welch spectrum of uniformly sampled ``y``."""
    y = np.asarray(y, dtype=float)
    if len(y) < 16:
        return math.nan
    nper = int(min(len(y), max(256, 2 ** int(math.log2(len(y) // 4 or 1)))))
    f, pxx = signal.welch(y - np.mean(y), fs=fs, nperseg=nper, detrend="linear")
    m = (f >= f_lo) & (f <= f_hi)
    if m.sum() < 2:
        return math.nan
    integ = np.trapezoid(pxx[m], f[m]) if hasattr(np, "trapezoid") else np.trapz(pxx[m], f[m])
    return float(math.sqrt(integ))


def pixel_noise(t, y, t_px) -> float:
    """
    Noise as it ends up in an image: the standard deviation of ``y`` averaged
    over blocks of one pixel dwell time. NaN if there are fewer than 8 blocks.
    """
    _, means = block_mean(t, y, t_px)
    if len(means) < 8:
        return math.nan
    return float(np.std(means, ddof=1))
