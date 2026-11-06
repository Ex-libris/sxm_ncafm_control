"""Diagnostic PNGs for a campaign (matplotlib, Agg). Optional: without matplotlib the analysis still runs.

Encoding: the lock-in time constant is ordered, so it is an ordinal one-hue (blue) ramp, light = short TC,
dark = long TC, always with a distinct marker and a legend (identity never by colour alone). One y-axis per
panel; drive and amplitude of a trace are stacked panels with a shared time axis, not a twin axis.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import tc_seconds

RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']   # ordinal blue, 250..700
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
INK, INK2, GRID, SURFACE, FIT = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#eb6834'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'lines.linewidth': 2, 'font.size': 9, 'legend.frameon': False})
    return plt


def _style(tcs: list[str]) -> dict:
    ordered = sorted(set(tcs), key=tc_seconds)
    n = len(ordered)
    idx = [round(i * (len(RAMP) - 1) / max(1, n - 1)) for i in range(n)] if n <= len(RAMP) else list(range(n))
    return {tc: (RAMP[min(k, len(RAMP) - 1)], MARKERS[i % len(MARKERS)]) for i, (tc, k) in enumerate(zip(ordered, idx))}


def amp_vs_drive(cells: list[dict], path: Path) -> None:
    plt = _plt()
    rs = [c for c in cells if c['kind'] == 'matrix' and math.isfinite(c.get('amp_ss_mean', math.nan))]
    if not rs:
        return
    st = _style([c['tc'] for c in rs])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for tc, (col, mk) in st.items():
        s = sorted([c for c in rs if c['tc'] == tc], key=lambda c: c['drive_cmd'])
        x = [c['drive_cmd'] for c in s]
        y = [c['amp_ss_mean'] for c in s]
        e = [0 if not math.isfinite(c['amp_ss_ci95']) else c['amp_ss_ci95'] for c in s]
        ax.errorbar(x, y, yerr=e, color=col, marker=mk, ms=7, lw=1.5, capsize=2, label=f'TC {tc}')
    ax.set(xscale='log', yscale='log', xlabel='drive command (SXM units)', ylabel='steady QPlusAmpl (V)',
           title='Steady amplitude vs drive (mean ± 95 % CI over repeats)')
    ax.legend(title='lock-in TC', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tau_vs_tc(rows: list[dict], expected_tau: float, path: Path) -> None:
    plt = _plt()
    rs = [r for r in rows if r['kind'] in ('matrix', 'anchor', 'headroom') and r.get('ringdown_ok')]
    if not rs:
        return
    drives = sorted({r['drive_cmd'] for r in rs})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, d in enumerate(drives):
        s = [r for r in rs if r['drive_cmd'] == d]
        jitter = 1 + 0.06 * (i - (len(drives) - 1) / 2)
        ax.errorbar([r['tc_s'] * 1e3 * jitter for r in s], [r['tau_s'] for r in s],
                    yerr=[r['tau_se'] for r in s], ls='none', marker=MARKERS[i % len(MARKERS)], ms=7,
                    color=RAMP[round(i * (len(RAMP) - 1) / max(1, len(drives) - 1))], label=f'drive {d:g}')
    ax.axhline(expected_tau, color=INK2, lw=1, ls='--', label=f'config Q/(π f0) = {expected_tau:.4f} s')
    ax.set(xscale='log', xlabel='lock-in TC (ms)', ylabel='ring-down τ (s)',
           title='Ring-down time vs lock-in TC (each condition, ± fit SE)')
    ax.legend(title='drive (shade = level)', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def baseline_vs_tc(rows: list[dict], path: Path) -> None:
    plt = _plt()
    rs = [r for r in rows if math.isfinite(r.get('baseline_mean', math.nan)) and r.get('baseline_mean', 0) > 0]
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, key, lab in ((axes[0], 'baseline_mean', 'zero-drive QPlusAmpl mean (V)'),
                         (axes[1], 'baseline_std', 'zero-drive QPlusAmpl std (V)')):
        x = np.array([r['tc_s'] * 1e3 for r in rs])
        y = np.array([r[key] for r in rs])
        kinds = sorted({r['kind'] for r in rs})
        for i, k in enumerate(kinds):
            sel = np.array([r['kind'] == k for r in rs])
            ax.plot(x[sel], y[sel], ls='none', marker=MARKERS[i], ms=7, color=RAMP[2 + 2 * (i % 2)], label=k)
        ok = y > 0
        if np.sum(ok) > 1:
            x0 = np.exp(np.mean(np.log(x[ok])))
            y0 = np.exp(np.mean(np.log(y[ok])))
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, y0 * (xx / x0) ** -0.5, color=INK2, lw=1, ls='--', label='slope −1/2 (white noise)')
        ax.set(xscale='log', yscale='log', xlabel='lock-in TC (ms)', ylabel=lab)
        ax.legend(fontsize=8)
    fig.suptitle('Zero-drive baseline vs lock-in TC (after the discard window)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def anchors(rows: list[dict], path: Path) -> None:
    import datetime as dt
    plt = _plt()
    rs = [r for r in rows if r['kind'] == 'anchor' and r.get('start') and math.isfinite(r.get('amp_ss', math.nan))]
    if len(rs) < 2:
        return
    t0 = dt.datetime.fromisoformat(rs[0]['start'])
    h = [(dt.datetime.fromisoformat(r['start']) - t0).total_seconds() / 3600 for r in rs]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    axes[0].plot(h, [r['amp_ss'] for r in rs], marker='o', ms=7, color=RAMP[3])
    axes[0].set(ylabel='steady QPlusAmpl (V)', title='Anchor (reference condition) over the campaign')
    axes[1].errorbar(h, [r.get('tau_s', math.nan) for r in rs], yerr=[r.get('tau_se', 0) for r in rs],
                     marker='s', ms=7, color=RAMP[5])
    axes[1].set(ylabel='ring-down τ (s)', xlabel='hours since the first anchor')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pooled_ringdown(rp: dict, path: Path) -> None:
    """Averaged ring-downs per drive (sqrt of the averaged power) with their fits."""
    plt = _plt()
    groups = [(k, g) for k, g in (rp.get('by_drive') or {}).items() if g.get('ok')]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, (k, g) in enumerate(groups):
        col = RAMP[round(i * (len(RAMP) - 1) / max(1, len(groups) - 1))]
        c = g['curve']
        ax.plot(c['t'], np.sqrt(np.maximum(c['r2_mean'], 1e-30)), color=col, lw=1.2,
                marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(c['t']) // 12), ms=6,
                label=f'drive {k}: tau {g["tau_s"]:.3f} s, {g["n_traces"]} traces')
        ax.plot(c['t'], np.sqrt(np.maximum(c['fit'], 1e-30)), color=FIT, lw=1.5, ls='--')
    ax.set(yscale='log', xlabel='time after the drive step off (s)', ylabel='sqrt(<QPlusAmpl^2>) (V)',
           title='Averaged ring-down per drive (dashed: fit with the floor as offset)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def trace(t, data: dict, m: dict, title: str, path: Path) -> None:
    """One condition: amplitude (log) with the windows and the ring-down fit; Drive DAC below (shared time)."""
    plt = _plt()
    amp = np.asarray(data['QPlusAmpl'], float)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    a1.plot(t, np.maximum(amp, 1e-12), color=RAMP[3], lw=1, label='QPlusAmpl')
    t_off = m.get('t_off_dac')
    if m.get('steady_window'):
        a1.axvspan(*m['steady_window'], color=RAMP[0], alpha=0.25, lw=0, label='steady window')
    if t_off is not None and m.get('baseline_window'):
        a1.axvspan(t_off + m['baseline_window'][0], t_off + m['baseline_window'][1], color=GRID, alpha=0.8, lw=0,
                   label='baseline window')
    if m.get('ringdown_ok') and t_off is not None:
        w0, w1 = m['ringdown_window']
        tt = np.linspace(t_off + w0, t_off + w1, 200)
        fit = np.sqrt(np.maximum(m['a0_v'] ** 2 * np.exp(-2 * (tt - t_off - w0) / m['tau_s']) + m['noise_power_c'], 0))
        a1.plot(tt, fit, color=FIT, lw=2, ls='--', label=f'fit τ = {m["tau_s"]:.4f} ± {m["tau_se"]:.4f} s')
    pos = amp[amp > 0]
    if len(pos):
        a1.set_ylim(bottom=0.3 * float(np.percentile(pos, 1)))     # Rayleigh dips towards 0 hide the floor
    a1.set(yscale='log', ylabel='QPlusAmpl (V)', title=title)
    a1.legend(fontsize=7, loc='upper right')
    a2.plot(t, np.asarray(data['Drive'], float), color=INK2, lw=1)
    a2.set(ylabel='Drive DAC (V)', xlabel='time from record start (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
