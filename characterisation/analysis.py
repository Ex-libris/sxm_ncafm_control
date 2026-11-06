"""Analysis of one condition record and of a whole campaign. Pure numpy/scipy; no instrument.

Every number is reported with the window it came from, the baseline method, the fit interval, the model
and its uncertainty (``params`` / ``*_window`` / ``*_model`` fields), so a summary can be traced back to
the raw record.

Conventions
- Times are seconds from the record start. Step times come from the **Drive DAC channel** (where the
  hardware stepped), not from the command time; the command -> DAC delay is reported separately.
- QPlusAmpl is in volts as SXM outputs it. Nothing here converts it to displacement.
- Ring-down model: R(t)^2 = P0 exp(-2 (t - t_s) / tau) + c, weighted least squares, with c free (the noise
  power: E[R^2] = A^2 + 2 sigma^2 for a signal in Gaussian noise, so subtracting a power, not an amplitude,
  is the consistent baseline). t_s = DAC step + the filter's settling to 0.1 % (settle_time(TC, roll-off)),
  so the lock-in transient is outside the window: after it, an exponential leaves an n-stage filter as the
  same exponential (same tau), only scaled and delayed by about n TC. tau should therefore not depend on TC;
  the campaign summary tests that.
- Uncertainties: least-squares standard errors inflated by sqrt(N / N_eff), N_eff = window x 2 ENBW
  (samples closer than the lock-in correlation time are not independent).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import filters as P
from .config import rolloff_db, tc_seconds

MODEL_RINGDOWN = 'R^2 = P0 exp(-2 (t - t_s)/tau) + c; weighted LSQ (sigma_i = 2 R_i s + s^2, s = baseline std)'
MODEL_RINGUP = '|A_ss (1 - exp(-(1/tau + j 2 pi delta)(t - t_0)))|, t_0 = DAC step + n TC; LSQ'


@dataclass
class AnalysisParams:
    discard_s: float = 5.0             # baseline statistics start this long after the step off (or TC change)
    discard_auto: bool = True          # ... or later, until the fitted ring-down tail < residual_max x std
    residual_max: float = 0.1
    min_baseline_s: float = 5.0        # never shorten the baseline window below this when extending
    fit_settle_accuracy: float = 1e-3  # ring-down window starts when the filter transient is below this
    fit_min_start_s: float = 0.005
    fit_floor_factor: float = 5.0      # ring-down window ends where R falls below this x baseline mean ...
    fit_max_tau: float = 6.0           # ... or after this many expected tau
    loglin_floor_factor: float = 10.0  # log-linear cross-check: only where R > this x baseline mean
    steady_margin_s: float = 0.1       # steady window ends this long before the step-off command
    pre_margin_s: float = 0.05
    blocks: int = 10                   # block standard errors
    psd_bands: tuple = ((0.5, 5.0), (5.0, 50.0), (50.0, 500.0))


# ------------------------------------------------------------------------------------------ small helpers

def window(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (t >= t0) & (t < t1)


def stats(y: np.ndarray, blocks: int = 10) -> dict:
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return {'mean': math.nan, 'std': math.nan, 'se': math.nan, 'n': 0}
    mean, std = float(np.mean(y)), float(np.std(y, ddof=1)) if n > 1 else 0.0
    k = min(blocks, n)
    if k >= 2:
        bm = [float(np.mean(b)) for b in np.array_split(y, k)]
        se = float(np.std(bm, ddof=1) / math.sqrt(k))
    else:
        se = math.nan
    return {'mean': mean, 'std': std, 'se': se, 'n': n}


def slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3 or np.ptp(t) == 0:
        return math.nan
    return float(np.polyfit(np.asarray(t, float), np.asarray(y, float), 1)[0])


def circ_stats(deg: np.ndarray) -> tuple[float, float]:
    """Circular mean and circular standard deviation, degrees."""
    if len(deg) == 0:
        return math.nan, math.nan
    z = np.exp(1j * np.radians(np.asarray(deg, float)))
    m = np.mean(z)
    r = min(1.0, abs(m))
    return float(np.degrees(np.angle(m))), float(np.degrees(math.sqrt(-2 * math.log(r)))) if r > 0 else math.inf


def step_time(t: np.ndarray, drive: np.ndarray, t_from: float, lo: float, hi: float) -> float | None:
    """First time after ``t_from`` where the Drive DAC crosses the midpoint from ``lo`` towards ``hi``."""
    if len(t) == 0 or lo == hi:
        return None
    mid = (lo + hi) / 2
    sel = t >= t_from
    idx = np.nonzero(sel & ((drive > mid) if hi > lo else (drive < mid)))[0]
    return float(t[idx[0]]) if len(idx) else None


def uniform(t: np.ndarray, y: np.ndarray, fs: float | None = None) -> tuple[np.ndarray, float]:
    if fs is None:
        fs = (len(t) - 1) / (t[-1] - t[0])
    tu = np.arange(t[0], t[-1], 1 / fs)
    return np.interp(tu, t, y), fs


def asd(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectral density (unit/sqrt(Hz)) of a polled record, resampled to a uniform grid."""
    if len(t) < 64:
        return np.zeros(0), np.zeros(0)
    from scipy.signal import welch
    yu, fs = uniform(t, np.asarray(y, float))
    nper = int(min(len(yu), max(256, fs * 4)))
    f, pxx = welch(yu - np.mean(yu), fs=fs, nperseg=nper)
    return f, np.sqrt(pxx)


def band_density(f: np.ndarray, a: np.ndarray, lo: float, hi: float) -> float:
    sel = (f >= lo) & (f < hi)
    return float(np.sqrt(np.mean(a[sel] ** 2))) if np.any(sel) else math.nan


# ------------------------------------------------------------------------------------- PLL off (physical)

# The Frequency DAC is a DDS: 10 MHz / 2^32 = 0.00232830644 Hz per count. sxm_anfatec's driver table has the
# 6-digit 0.00232831 (+1.5 ppm), which reads +39 mHz at 25.56 kHz. Seen on the instrument 2026-11-06:
# Frequency DAC 25563.7495 Hz (flat) at use = 25563.71 Hz; x (1e7/2^32)/0.00232831 = 25563.7104 Hz.
DDS_SCALE_HZ = 1e7 / 2 ** 32
SCALE_REL_TOL = 3e-6          # what the 6-digit driver scale can be off by (+-2 ppm, with margin)


def pll_off_check(freq, df, f_use: float, tol_hz: float,
                  scale: float | None = None) -> tuple[list[str], list[str], dict]:
    """The excitation frequency during a record. Returns (problems, notes, values).

    Open loop, what matters is that the excitation frequency is CONSTANT (and known), not that it equals
    ``use``: a ring-down decays at f0 whatever the drive frequency, and a constant detuning scales every
    steady amplitude by the same factor. So:

    - problem (refuse / abort): the Frequency DAC moves (std > tol), i.e. a PLL is acting during the record;
    - notes (recorded, not blocking): df held away from 0 (an offset frozen in when the PLL gains were set to
      0) and the level differing from ``use`` beyond the driver scale's rounding. Both are constant offsets.

    ``f_exc_hz`` is the best estimate of the excitation frequency (the exact DDS scale when ``scale`` is given).
    """
    f = np.asarray(freq, float)
    s = stats(f)
    v = {'freq_dac': s['mean'], 'freq_dac_std': s['std'], 'freq_dac_minus_use': s['mean'] - f_use,
         'f_exc_hz': s['mean']}
    if scale:
        v['freq_dac_dds_scale'] = v['f_exc_hz'] = s['mean'] * DDS_SCALE_HZ / scale
        v['freq_dds_minus_use'] = v['freq_dac_dds_scale'] - f_use
    v['f_exc_minus_use'] = v['f_exc_hz'] - f_use
    problems, notes = [], []
    if not s['std'] <= tol_hz:
        problems.append(f'Frequency DAC moves (std {s["std"]:.2e} Hz > {tol_hz:g} Hz): a PLL is acting')
    if df is not None and len(df):
        d = stats(np.asarray(df, float))
        v.update(df_dac=d['mean'], df_dac_std=d['std'])
        if abs(d['mean']) > tol_hz:
            notes.append(f'df = {d["mean"]:+.4f} Hz held (an offset frozen in when the PLL gains were set to 0): '
                         'constant, so harmless open loop; the excitation frequency is recorded')
    allowed = max(tol_hz, SCALE_REL_TOL * abs(f_use))
    if abs(v['freq_dac_minus_use']) > allowed:
        notes.append(f'excitation {v["f_exc_hz"]:.4f} Hz is {v["f_exc_minus_use"]:+.4f} Hz from use {f_use} Hz')
    return problems, notes, v


# ---------------------------------------------------------------------------------------------- fits

def fit_ringdown(t, amp, t_step, tc_s, roll, f0, floor_mean, floor_std, tau_guess, p: AnalysisParams) -> dict:
    """Fit the amplitude ring-down after a drive step off at ``t_step`` (DAC time). See module docstring."""
    from scipy.optimize import curve_fit
    out = {'ringdown_model': MODEL_RINGDOWN, 'ringdown_ok': False}
    t_s = t_step + max(P.settle_time(tc_s, roll, p.fit_settle_accuracy), p.fit_min_start_s)
    t_max = t_s + p.fit_max_tau * tau_guess
    sel = t >= t_s
    below = np.nonzero(sel & (amp < p.fit_floor_factor * floor_mean))[0]
    t_end = min(t_max, float(t[below[0]]) if len(below) else float(t[-1]))
    w = window(t, t_s, t_end)
    out.update(ringdown_window=[t_s - t_step, t_end - t_step], ringdown_n=int(np.sum(w)))
    if np.sum(w) < 20:
        out['ringdown_note'] = 'too few samples in the fit window'
        return out
    tt, rr = t[w] - t_s, np.asarray(amp[w], float)
    s = floor_std if floor_std > 0 else max(1e-12, 1e-3 * float(np.max(rr)))
    c0 = max(floor_mean ** 2, 0.0)
    pos = rr ** 2 - c0 > 0
    if np.sum(pos) < 10:
        out['ringdown_note'] = 'signal not above the noise in the fit window'
        return out
    k, b = np.polyfit(tt[pos], np.log(rr[pos] ** 2 - c0), 1)
    tau0 = -2 / k if k < 0 else tau_guess
    sigma = 2 * rr * s + s ** 2

    def model(x, p0, tau, c):
        return p0 * np.exp(-2 * x / tau) + c

    try:
        popt, pcov = curve_fit(model, tt, rr ** 2, p0=[math.exp(b), tau0, c0], sigma=sigma, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        out['ringdown_note'] = f'fit failed: {exc}'
        return out
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(3, math.nan)
    n_eff = max(2.0, (t_end - t_s) * 2 * P.enbw(tc_s, roll))
    infl = math.sqrt(max(1.0, len(tt) / n_eff))
    p0, tau, c = (float(v) for v in popt)
    resid = (rr ** 2 - model(tt, *popt)) / sigma
    a0 = math.sqrt(max(p0, 0.0))
    out.update(ringdown_ok=bool(tau > 0 and p0 > 0), tau_s=tau, tau_se=float(perr[1] * infl),
               q=math.pi * f0 * tau, q_se=float(math.pi * f0 * perr[1] * infl), a0_v=a0,
               noise_power_c=c, ringdown_chi_red=float(np.sum(resid ** 2) / max(1, len(tt) - 3)),
               ringdown_n_eff=n_eff, ringdown_se_inflation=infl, ringdown_f0_used=f0)
    hi = w & (amp > p.loglin_floor_factor * floor_mean)
    if np.sum(hi) >= 10:
        kk = np.polyfit(t[hi] - t_s, np.log(np.asarray(amp[hi], float)), 1)[0]
        out['tau_loglin_s'] = float(-1 / kk) if kk < 0 else math.nan
        out['loglin_window'] = [float(t[hi][0] - t_step), float(t[hi][-1] - t_step)]
    return out


def fit_ringup(t, amp, t_step, t_end, tc_s, roll, tau_guess) -> dict:
    """Ring-up from rest after a drive step on: gives tau again and |detuning| (beats when f_use != f0)."""
    from scipy.optimize import curve_fit
    out = {'ringup_model': MODEL_RINGUP, 'ringup_ok': False}
    t0 = t_step + P.stages(roll) * tc_s
    w = window(t, t0, t_end)
    if np.sum(w) < 50:
        return out
    tt, rr = t[w] - t0, np.asarray(amp[w], float)
    a_ss0 = float(np.mean(rr[-max(10, len(rr) // 20):]))

    def model(x, a_ss, tau, delta):
        return np.abs(a_ss * (1 - np.exp(-(1 / tau + 2j * np.pi * delta) * x)))

    try:
        popt, pcov = curve_fit(model, tt, rr, p0=[a_ss0, tau_guess, 0.05],
                               bounds=([0, tau_guess / 20, 0], [10 * a_ss0 + 1e-12, tau_guess * 20, 50]),
                               maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(3, math.nan)
    n_eff = max(2.0, (t_end - t0) * 2 * P.enbw(tc_s, roll))
    infl = math.sqrt(max(1.0, len(tt) / n_eff))
    out.update(ringup_ok=True, tau_up_s=float(popt[1]), tau_up_se=float(perr[1] * infl),
               detune_abs_hz=float(popt[2]), detune_se_hz=float(perr[2] * infl),
               ringup_window=[t0 - t_step, t_end - t_step])
    return out


# ---------------------------------------------------------------------------------------- one condition

def analyse_condition(t: np.ndarray, data: dict, meta: dict, p: AnalysisParams | None = None) -> dict:
    """Metrics of one record. ``meta`` is the condition's sidecar (kind, drive, off_drive, tc, rolloff,
    marks, expect, preflight k_cal)."""
    p = p or AnalysisParams()
    amp = np.asarray(data['QPlusAmpl'], float)
    drv = np.asarray(data['Drive'], float)
    tc_s, roll = tc_seconds(meta['tc']), rolloff_db(meta.get('rolloff', '12 db/oct'))
    f_use = meta.get('f_use') or meta['expect']['f0_hz']
    tau_guess = meta['expect']['q'] / (math.pi * meta['expect']['f0_hz'])
    marks = {m['label']: m for m in meta.get('marks', [])}
    k_cal = meta.get('k_cal')
    out = {'idx': meta['idx'], 'kind': meta['kind'], 'block': meta['block'], 'drive_cmd': meta.get('drive'),
           'tc': meta['tc'], 'tc_readback': meta.get('tc_readback'), 'tc_s': tc_s, 'rolloff_db': roll, 'start': meta.get('start'),
           'status': meta.get('status'), 'n_samples': len(t),
           'rate_hz': (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else math.nan,
           'params': asdict(p)}
    if len(t) < 10:
        out['note'] = 'empty record'
        return out
    t_end = float(t[-1])
    off = meta.get('off_drive', 0.0)

    if meta['kind'] == 'baseline':
        m = marks.get('tc')
        t_tr = (m['t_ack'] if m else 0.0)
        t_b = t_tr + p.discard_s
        out.update(_baseline(t, amp, data, t_b, t_end, p, 'baseline'))
        out['baseline_window'] = [t_b, t_end]
        out['baseline_method'] = f'mean/std of QPlusAmpl from {p.discard_s:g} s after the TC write to the end'
        out['drive_dac_baseline'] = stats(drv[window(t, t_b, t_end)])['mean']
        _add_noise_density(out, tc_s, roll)
        return out

    # driven condition: pre | on (ring-up, steady) | off (ring-down, baseline)
    m_on, m_off = marks.get('on'), marks.get('off')
    if not m_on or not m_off:
        out['note'] = 'record has no on/off marks (stopped or aborted early)'
        return out
    pre = window(t, 0.0, m_on['t_cmd'] - p.pre_margin_s)
    d_pre = stats(drv[pre])['mean'] if np.any(pre) else off
    steady = window(t, m_off['t_cmd'] - meta['steady_s'], m_off['t_cmd'] - p.steady_margin_s)
    d_on = stats(drv[steady])['mean']
    t_on = step_time(t, drv, m_on['t_cmd'] - 0.2, d_pre, d_on)
    t_off = step_time(t, drv, m_off['t_cmd'] - 0.2, d_on, d_pre)
    out.update(t_on_dac=t_on, t_off_dac=t_off,
               delay_cmd_dac_on=(t_on - m_on['t_cmd']) if t_on is not None else math.nan,
               delay_cmd_dac_off=(t_off - m_off['t_cmd']) if t_off is not None else math.nan,
               write_ack_s=m_off['t_ack'] - m_off['t_cmd'])

    s_amp = stats(amp[steady], p.blocks)
    s_drv = stats(drv[steady], p.blocks)
    out.update(amp_pre=stats(amp[pre])['mean'], drive_dac_pre=d_pre,
               amp_ss=s_amp['mean'], amp_ss_std=s_amp['std'], amp_ss_se=s_amp['se'],
               amp_ss_drift_rel=(slope(t[steady], amp[steady]) * meta['steady_s'] / s_amp['mean']
                                 if s_amp['mean'] else math.nan),
               drive_dac=s_drv['mean'], drive_dac_std=s_drv['std'],
               drive_dac_flat_rel=s_drv['std'] / abs(s_drv['mean']) if s_drv['mean'] else math.nan,
               steady_window=[float(m_off['t_cmd'] - meta['steady_s']), float(m_off['t_cmd'] - p.steady_margin_s)])
    if out['drive_cmd']:
        out['drive_ratio'] = (s_drv['mean'] - d_pre) / (out['drive_cmd'] - off)
        if k_cal:
            out['drive_echo_rel'] = out['drive_ratio'] / k_cal - 1
        out['amp_per_drive'] = s_amp['mean'] / (s_drv['mean'] - d_pre) if s_drv['mean'] != d_pre else math.nan
    if 'Phase' in data:
        ph = np.asarray(data['Phase'], float)
        out['phase_ss'], out['phase_ss_std'] = circ_stats(ph[steady])
    if 'Frequency' in data:
        df = np.asarray(data['df'], float)[steady] if 'df' in data else None
        problems, notes, vals = pll_off_check(np.asarray(data['Frequency'], float)[steady], df, f_use,
                                              meta.get('freq_tol_hz', 0.005), (meta.get('scales') or {}).get('Frequency'))
        out.update(vals)
        out['pll_off_problems'] = problems
        out['excitation_notes'] = notes
    if 'InA' in data:
        out['ina_peak'] = float(np.max(np.abs(np.asarray(data['InA'], float)))) if len(t) else math.nan
    f, a = asd(t[steady], amp[steady])
    for lo, hi in p.psd_bands:
        out[f'amp_ss_asd_{lo:g}_{hi:g}Hz'] = band_density(f, a, lo, hi)

    if t_off is None:
        out['note'] = 'drive step off not seen on the Drive DAC'
        return out
    t_b = t_off + p.discard_s
    out.update(_baseline(t, amp, data, t_b, t_end, p, 'baseline'))
    rd = fit_ringdown(t, amp, t_off, tc_s, roll, f_use, out['baseline_mean'], out['baseline_std'], tau_guess, p)
    out.update(rd)
    discard = p.discard_s
    if rd.get('ringdown_ok'):
        t_s = t_off + rd['ringdown_window'][0]
        if s_amp['mean'] > 0 and rd['a0_v'] > 0:
            out['ringdown_delay_s'] = (t_s - rd['tau_s'] * math.log(s_amp['mean'] / rd['a0_v'])) - t_off
        out['ringdown_delay_expected_s'] = P.group_delay(tc_s, roll)
        # The baseline must start where the ring-down tail is below a tenth of the noise: extend the discard
        # when the fit says so (the configured value is the minimum), keeping at least min_baseline_s.
        if p.discard_auto and out['baseline_std'] > 0:
            need = (t_s - t_off) + rd['tau_s'] * math.log(max(1.0, rd['a0_v'] / (p.residual_max * out['baseline_std'])))
            if need > discard and t_end - (t_off + need) >= p.min_baseline_s:
                discard = need
                out.update(_baseline(t, amp, data, t_off + discard, t_end, p, 'baseline'))
        residual = rd['a0_v'] * math.exp(-(t_off + discard - t_s) / rd['tau_s'])
        out['residual_at_baseline_v'] = residual
        out['residual_over_baseline_std'] = residual / out['baseline_std'] if out['baseline_std'] else math.nan
    t_b = t_off + discard
    out['baseline_discard_s'] = discard
    out['baseline_window'] = [discard, t_end - t_off]
    out['baseline_method'] = (f'mean/std of QPlusAmpl from {discard:.2f} s after the DAC step off to the end '
                              f'(minimum {p.discard_s:g} s; extended until the fitted ring-down tail is below '
                              f'{p.residual_max:g} x baseline std); the ring-down fit has the noise power as offset c')
    out['drive_dac_off'] = stats(drv[window(t, t_b, t_end)])['mean']
    _add_noise_density(out, tc_s, roll)
    if t_on is not None and out['amp_pre'] < 0.1 * s_amp['mean']:
        out.update(fit_ringup(t, amp, t_on, m_off['t_cmd'] - p.steady_margin_s, tc_s, roll, tau_guess))
    return out


def _baseline(t, amp, data, t0, t1, p: AnalysisParams, key: str) -> dict:
    w = window(t, t0, t1)
    s = stats(amp[w], p.blocks)
    out = {f'{key}_mean': s['mean'], f'{key}_std': s['std'], f'{key}_se': s['se'], f'{key}_n': s['n'],
           f'{key}_median': float(np.median(amp[w])) if s['n'] else math.nan,
           f'{key}_slope_v_per_s': slope(t[w], amp[w]),
           f'{key}_rms': float(np.sqrt(np.mean(np.asarray(amp[w], float) ** 2))) if s['n'] else math.nan}
    f, a = asd(t[w], amp[w])
    for lo, hi in p.psd_bands:
        out[f'{key}_asd_{lo:g}_{hi:g}Hz'] = band_density(f, a, lo, hi)
    if 'Phase' in data and s['n']:
        out[f'{key}_phase_circ_std'] = circ_stats(np.asarray(data['Phase'], float)[w])[1]
    return out


def _add_noise_density(out: dict, tc_s: float, roll: int) -> None:
    """Input noise density implied by the zero-drive amplitude, if it is Rayleigh noise:
    E[R] = sigma sqrt(pi/2), sigma^2 = S ENBW per quadrature -> S^0.5 = E[R] / sqrt(pi/2 ENBW)."""
    enbw = P.enbw(tc_s, roll)
    out['enbw_hz'] = enbw
    if math.isfinite(out.get('baseline_mean', math.nan)):
        out['noise_density_v_rthz'] = out['baseline_mean'] / math.sqrt(math.pi / 2 * enbw)
        out['noise_density_method'] = 'Rayleigh: baseline_mean / sqrt(pi/2 * ENBW(TC, roll-off))'


# ------------------------------------------------------------------------------------------- whole run

def t_crit(n: int, level: float = 0.95) -> float:
    if n < 2:
        return math.nan
    from scipy.stats import t as tdist
    return float(tdist.ppf(0.5 + level / 2, n - 1))


def summarise_cells(rows: list[dict], keys=('amp_ss', 'phase_ss', 'drive_dac', 'tau_s', 'baseline_mean',
                                              'baseline_std', 'noise_density_v_rthz', 'ringdown_delay_s',
                                              'amp_per_drive')) -> list[dict]:
    """Mean, SE and 95 % CI over repeats, per (kind, drive, TC) cell."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get('status') not in ('ok', 'warn'):
            continue
        cells.setdefault((r['kind'], r.get('drive_cmd'), r['tc']), []).append(r)
    out = []
    for (kind, drive, tc), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, tc_seconds(kv[0][2]))):
        row = {'kind': kind, 'drive_cmd': drive, 'tc': tc, 'n': len(rs)}
        for k in keys:
            v = np.array([r.get(k, math.nan) for r in rs], float)
            v = v[np.isfinite(v)]
            n = len(v)
            row[f'{k}_mean'] = float(np.mean(v)) if n else math.nan
            row[f'{k}_sd'] = float(np.std(v, ddof=1)) if n > 1 else math.nan
            row[f'{k}_ci95'] = t_crit(n) * row[f'{k}_sd'] / math.sqrt(n) if n > 1 else math.nan
        out.append(row)
    return out


def linearity(rows: list[dict]) -> dict:
    """Per TC: steady amplitude vs measured drive (Drive DAC minus its zero level), straight line with offset."""
    out = {}
    for tc in sorted({r['tc'] for r in rows if r['kind'] == 'matrix'}, key=tc_seconds):
        rs = [r for r in rows if r['kind'] == 'matrix' and r['tc'] == tc and r.get('status') in ('ok', 'warn')
              and math.isfinite(r.get('amp_ss', math.nan)) and math.isfinite(r.get('drive_dac', math.nan))]
        if len({r['drive_cmd'] for r in rs}) < 2:
            continue
        x = np.array([r['drive_dac'] - r.get('drive_dac_pre', 0.0) for r in rs])
        y = np.array([r['amp_ss'] for r in rs])
        (k, b), cov = np.polyfit(x, y, 1, cov=True) if len(rs) > 3 else (np.polyfit(x, y, 1), np.full((2, 2), math.nan))
        by_d = {}
        for r, xi, yi in zip(rs, x, y):
            by_d.setdefault(r['drive_cmd'], []).append(yi / xi if xi else math.nan)
        ratios = {d: float(np.nanmean(v)) for d, v in sorted(by_d.items())}
        r0 = ratios[min(ratios)]
        out[tc] = {'slope_v_per_v': float(k), 'slope_se': float(math.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else math.nan,
                   'offset_v': float(b), 'offset_se': float(math.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else math.nan,
                   'max_rel_resid': float(np.max(np.abs(y - (k * x + b)) / y)),
                   'amp_per_drive_rel_to_lowest': {f'{d:g}': v / r0 for d, v in ratios.items()},
                   'n': len(rs), 'method': 'least squares A_ss = k (DriveDAC - DriveDAC_zero) + b over all repeats'}
    return out


def tc_dependence(rows: list[dict], key: str) -> dict:
    """Regression of ``key`` on TC over the matrix conditions: slope ~ 0 means TC-independent."""
    rs = [r for r in rows if r['kind'] == 'matrix' and r.get('status') in ('ok', 'warn')
          and math.isfinite(r.get(key, math.nan))]
    if len({r['tc'] for r in rs}) < 2:
        return {}
    x = np.array([r['tc_s'] for r in rs])
    y = np.array([r[key] for r in rs])
    (k, b), cov = np.polyfit(x, y, 1, cov=True) if len(rs) > 3 else (np.polyfit(x, y, 1), np.full((2, 2), math.nan))
    se = float(math.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else math.nan
    per_tc = {}
    for tc in sorted({r['tc'] for r in rs}, key=tc_seconds):
        v = np.array([r[key] for r in rs if r['tc'] == tc])
        per_tc[tc] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else math.nan,
                      'n': len(v)}
    return {'slope_per_s': float(k), 'slope_se': se, 'intercept': float(b),
            'slope_over_se': float(k / se) if se and math.isfinite(se) else math.nan,
            'change_over_range_rel': float(k * (x.max() - x.min()) / np.mean(y)), 'per_tc': per_tc,
            'method': f'least squares {key} = a + slope * TC over all matrix conditions'}


def baseline_vs_tc(rows: list[dict]) -> dict:
    """log-log slope of the zero-drive amplitude vs TC: -0.5 for white noise (R ~ sqrt(ENBW))."""
    rs = [r for r in rows if r.get('status') in ('ok', 'warn') and math.isfinite(r.get('baseline_mean', math.nan))
          and r.get('baseline_mean', 0) > 0]
    if len({r['tc'] for r in rs}) < 2:
        return {}
    x, y = np.log([r['tc_s'] for r in rs]), np.log([r['baseline_mean'] for r in rs])
    k = float(np.polyfit(x, y, 1)[0])
    dens = np.array([r['noise_density_v_rthz'] for r in rs if math.isfinite(r.get('noise_density_v_rthz', math.nan))])
    return {'loglog_slope': k, 'expected_white': -0.5,
            'noise_density_v_rthz_median': float(np.median(dens)) if len(dens) else math.nan,
            'noise_density_spread_rel': float(np.std(dens) / np.mean(dens)) if len(dens) > 1 else math.nan,
            'n': len(rs), 'method': 'least squares ln(baseline_mean) = a + k ln(TC), driven and baseline conditions'}


def drift(rows: list[dict], key: str = 'amp_ss') -> dict:
    """Anchors (the reference condition repeated): linear trend of ``key`` with time."""
    import datetime as dt
    rs = [r for r in rows if r['kind'] == 'anchor' and r.get('status') in ('ok', 'warn')
          and math.isfinite(r.get(key, math.nan)) and r.get('start')]
    if len(rs) < 2:
        return {}
    t0 = dt.datetime.fromisoformat(rs[0]['start'])
    hours = np.array([(dt.datetime.fromisoformat(r['start']) - t0).total_seconds() / 3600 for r in rs])
    y = np.array([r[key] for r in rs])
    if np.ptp(hours) == 0:
        return {'n': len(rs), 'values': y.tolist()}
    k = float(np.polyfit(hours, y, 1)[0])
    return {'n': len(rs), 'hours': hours.tolist(), 'values': y.tolist(), 'slope_per_h': k,
            'slope_rel_per_h': k / float(np.mean(y)), 'spread_rel': float(np.std(y, ddof=1) / np.mean(y))}


# --------------------------------------------------------------------------------------------- I/O

def load_condition(json_path: Path) -> tuple[np.ndarray, dict, dict]:
    meta = json.loads(Path(json_path).read_text(encoding='utf-8'))
    with np.load(Path(json_path).with_suffix('.npz')) as z:
        t = z['t'].astype(np.float64)
        data = {k[4:]: z[k].astype(np.float64) * float(z['scale_' + k[4:]]) for k in z.files if k.startswith('cnt_')}
    return t, data, meta


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys and k != 'params']
    with open(path, 'x', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
