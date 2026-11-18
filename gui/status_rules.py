"""What one read of SXM says about the sensor and the loops, as findings for the Status page.

Pure Python: ``assess(values, previous)`` takes ``SXMReadout.values`` (and the read before it) and returns a
list of ``{'level', 'title', 'evidence', 'action'}`` with level 'ok' | 'info' | 'warn' | 'bad'. Gains are
SXM's arbitrary units: they are only compared with the manual's starting rule (tuning.workflow.afl_start_values),
never interpreted.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..common import format_number
from ..tuning.workflow import afl_start_values

OK, INFO, WARN, BAD = 'ok', 'info', 'warn', 'bad'

# (group, [(key, label, unit)])
GROUPS = [
    ('Sensor', [('f_peak', 'fPeak (last sweep)', 'Hz'), ('q', 'Q', ''), ('ring_down_s', 'Ring-down', 's'),
                ('used_freq', '`use` frequency', 'Hz'), ('drive', 'Drive (DNC field)', '')]),
    ('Amplitude loop (AFL)', [('amp_ref', 'Ref', ''), ('amp_kp', 'Kp', ''), ('amp_ki', 'Ki', ''),
                              ('amp_tau_s', 'Tau', 's'), ('afl_output_gain', 'Output gain ±', 'V')]),
    ('PLL', [('pll_kp', 'Kp', ''), ('pll_ki', 'Ki', '')]),
    ('Detection (DNC)', [('dnc_time_constant_s', 'TimeConstant', 's'), ('dnc_rolloff', 'RollOff', ''),
                         ('input_gain_ina', 'Input gain InA ×', '')]),
    ('z feedback', [('feedback_off', 'Feedback', ''), ('feedback_mode', 'Mode', ''), ('topo_ref', 'Ref', ''),
                    ('topo_kp', 'Kp', ''), ('topo_ki', 'Ki', '')]),
]
LABELS = {key: f'{group.split(" (")[0]} {label}' for group, rows in GROUPS for key, label, _u in rows}


def show(key: str, value: Any, unit: str = '') -> str:
    if value is None:
        return 'n/a'
    if key == 'feedback_off':
        return 'OFF' if value else 'on'
    if isinstance(value, bool) or isinstance(value, str):
        return str(value)
    if unit == 's':
        v = float(value)
        return f'{format_number(v * 1e3, 4)} ms' if abs(v) < 1 else f'{format_number(v, 4)} s'
    return f'{format_number(float(value), 9)}{" " + unit if unit else ""}'


def _num(values: Dict[str, Any], key: str) -> Optional[float]:
    v = values.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def finding(level: str, title: str, evidence: str = '', action: str = '') -> dict:
    return {'level': level, 'title': title, 'evidence': evidence, 'action': action}


def changes(values: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> List[str]:
    """'AFL Ki 100 -> 300' for every value that differs from the read before."""
    if not previous:
        return []
    units = {key: unit for _g, rows in GROUPS for key, _l, unit in rows}
    out = []
    for key, label in LABELS.items():
        a, b = previous.get(key), values.get(key)
        if a is None or b is None or a == b:
            continue
        if isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
            continue
        out.append(f'{label} {show(key, a, units[key])} → {show(key, b, units[key])}')
    return out


def assess(values: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> List[dict]:
    if not any(v is not None for v in values.values()):
        return [finding(BAD, 'SXM is not readable', 'No value could be read from the SXM windows.',
                        'Start SXM (Femto), open the Dynamic Non-Contact window, then press Refresh.')]
    out: List[dict] = []
    f0, q, use = _num(values, 'f_peak'), _num(values, 'q'), _num(values, 'used_freq')
    akp, aki = _num(values, 'amp_kp'), _num(values, 'amp_ki')
    pkp, pki = _num(values, 'pll_kp'), _num(values, 'pll_ki')
    pll_on = None if pkp is None or pki is None else not (pkp == 0 and pki == 0)
    afl_on = None if akp is None or aki is None else not (akp == 0 and aki == 0)

    changed = changes(values, previous)
    if changed:
        out.append(finding(INFO, f'{len(changed)} value(s) changed since the last read', '; '.join(changed)))

    if not f0 or not q or f0 <= 0 or q <= 0:
        out.append(finding(WARN, 'Q and fPeak are not shown in the DNC status bar',
                           'The starting gains, the step timing and Tau all follow from Q.',
                           'Run a resonance sweep in the DNC window, then Refresh.'))
    else:
        half = f0 / (2 * q)
        ring = q / (math.pi * f0)
        if use is not None and pll_on is False:
            off = use - f0
            if abs(off) > half:
                out.append(finding(WARN, f'`use` is {off:+.3f} Hz from the peak ({abs(off) / half:.1f} half-widths)',
                                   f'Half-width f0/2Q = {half * 1e3:.3g} mHz at Q = {q:,.0f}; fPeak is from the last '
                                   'sweep. With the PLL off the sensor is driven off resonance: less amplitude per '
                                   'drive and a phase that is not 0.',
                                   'Set `use` to the peak, or sweep again if the resonance moved.'))
            else:
                out.append(finding(OK, '`use` is on the resonance',
                                   f'{off * 1e3:+.3g} mHz from fPeak, half-width {half * 1e3:.3g} mHz.'))
        out.append(finding(INFO, f'Sensor: Q {q:,.0f}, ring-down {ring:.3g} s, half-width {half * 1e3:.3g} mHz',
                           f'An amplitude step cannot settle faster than about {3 * ring:.2g} s without the loop '
                           'driving it; step tests wait in multiples of the ring-down.'))

    if afl_on is False:
        out.append(finding(INFO, 'Amplitude loop is off (Kp = Ki = 0)', 'Constant excitation: the amplitude is not held.'))
    elif afl_on and f0 and q and f0 > 0 and q > 0:
        og = _num(values, 'afl_output_gain')
        if og:
            rule = afl_start_values(q, f0, og)
            g = abs(aki) / rule.ki if rule.ki else float('nan')
            shape = abs(akp / aki) if aki else float('inf')
            ev = (f'Manual rule at Q {q:,.0f}, output gain ±{og:g} V: Kp {format_number(rule.kp, 3)}, '
                  f'Ki {format_number(rule.ki, 3)}, Tau {rule.tau_s * 1e3:.3g} ms. Now Kp:Ki = {shape:.3g} '
                  f'(rule 1e4).')
            if g > 3:
                out.append(finding(WARN, f'AFL Ki is {g:.3g}× the manual starting rule', ev,
                                   'The automatic search never goes above 3×; check with a step test before scanning.'))
            else:
                out.append(finding(INFO, f'AFL Ki is {g:.3g}× the manual starting rule', ev))
            tau = _num(values, 'amp_tau_s')
            if tau and (tau > 10 * rule.tau_s or tau < rule.tau_s / 10):
                out.append(finding(INFO, f'AFL Tau {tau * 1e3:.3g} ms is far from the rule ({rule.tau_s * 1e3:.3g} ms)',
                                   'Tau = Q / (100 f0) in the manual; Q has changed, or Tau was chosen for noise.'))
        else:
            out.append(finding(INFO, 'AFL output gain not readable', 'The manual rule depends on it (×10 per range).'))

    if pll_on is False:
        out.append(finding(INFO, 'PLL is off (Kp = Ki = 0)', 'df is not tracked; the drive stays at `use`.'))

    fb = values.get('feedback_off')
    if fb is False:
        out.append(finding(WARN, 'z feedback is ON', f'Mode: {values.get("feedback_mode") or "n/a"}. The tip may be '
                           'at the surface.', 'Tuning sheets and sensor characterisation need the tip retracted.'))
    elif fb is True:
        out.append(finding(OK, 'z feedback is off'))
    return out
