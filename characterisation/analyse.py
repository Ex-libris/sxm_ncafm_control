"""Analyse a campaign folder from its raw records. Writes a new derived/<time>/ folder; never touches raw/.

    python -m sxm_ncafm_control.characterisation.analyse <run folder> [--no-plots] [--discard 5] [--no-auto-discard]

Output: conditions.csv (one row per record), cells.csv (mean / SD / 95 % CI per drive x TC), summary.json
(linearity, TC dependence of tau, baseline vs TC, drift of the anchors, all with their methods),
report.md, plots/*.png.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from . import VERSION
from . import analysis as A
from .config import tc_seconds
from .runner import dumps, write_new


def analyse_run(run_dir: Path, params: A.AnalysisParams | None = None, plots: bool = True) -> dict:
    run_dir = Path(run_dir)
    params = params or A.AnalysisParams()
    camp = json.loads((run_dir / 'campaign.json').read_text(encoding='utf-8'))
    cfg = camp['config']
    rows, records = [], {}
    for js in sorted((run_dir / 'raw').glob('c*.json')):
        meta = json.loads(js.read_text(encoding='utf-8'))
        if not js.with_suffix('.npz').exists():
            rows.append({'idx': meta['idx'], 'kind': meta['kind'], 'block': meta['block'], 'drive_cmd': meta.get('drive'),
                         'tc': meta['tc'], 'tc_s': tc_seconds(meta['tc']), 'status': meta.get('status'),
                         'note': 'no record (aborted before capture)'})
            continue
        t, data, meta = A.load_condition(js)
        meta.setdefault('steady_s', cfg['timing']['steady_s'])
        r = A.analyse_condition(t, data, meta, params)
        r['file'] = js.with_suffix('.npz').name
        r['warnings'] = meta.get('warnings', [])
        r['aborts'] = meta.get('aborts', [])
        rows.append(r)
        records[meta['idx']] = (t, data, r)
    rows.sort(key=lambda r: r['idx'])
    expected_tau = cfg['expect']['q'] / (math.pi * cfg['expect']['f0_hz'])
    summary = {
        'run_dir': str(run_dir), 'analysed': datetime.datetime.now().astimezone().isoformat(), 'qplus_version': VERSION,
        'params': asdict(params), 'environment': camp.get('environment'), 'n_records': len(rows),
        'status_counts': {s: sum(1 for r in rows if r.get('status') == s) for s in {r.get('status') for r in rows}},
        'linearity': A.linearity(rows),
        'tau_vs_tc': A.tc_dependence(rows, 'tau_s'),
        'delay_vs_tc': A.tc_dependence(rows, 'ringdown_delay_s'),
        'amp_vs_tc': A.tc_dependence(rows, 'amp_per_drive'),
        'baseline_vs_tc': A.baseline_vs_tc(rows),
        'drift_amp': A.drift(rows, 'amp_ss'), 'drift_tau': A.drift(rows, 'tau_s'),
        'expected_tau_s': expected_tau,
    }
    taus = [r['tau_s'] for r in rows if r.get('ringdown_ok') and r.get('status') in ('ok', 'warn')]
    if taus:
        import numpy as np
        m, sd = float(np.mean(taus)), float(np.std(taus, ddof=1)) if len(taus) > 1 else math.nan
        f0 = next((r['ringdown_f0_used'] for r in rows if r.get('ringdown_ok')), cfg['expect']['f0_hz'])
        summary['tau_pooled'] = {'mean_s': m, 'sd_s': sd, 'n': len(taus), 'ci95_s': A.t_crit(len(taus)) * sd / math.sqrt(len(taus)),
                                 'q': math.pi * f0 * m, 'q_ci95': math.pi * f0 * A.t_crit(len(taus)) * sd / math.sqrt(len(taus)),
                                 'f0_used_hz': f0, 'method': 'mean of per-condition ring-down fits (ok/warn), '
                                 't-distribution CI; Q = pi f0 tau with f0 = the used (excitation) frequency'}
    for name in ('headroom.json', 'schedule.json'):
        if (run_dir / name).exists():
            summary[name.split('.')[0]] = json.loads((run_dir / name).read_text(encoding='utf-8'))
            summary[name.split('.')[0]].pop('entries', None)
    pfs = sorted(run_dir.glob('preflight_*.json'))
    if pfs:
        pf = json.loads(pfs[-1].read_text(encoding='utf-8'))
        summary['preflight'] = {k: pf.get(k) for k in ('ok', 'k_cal', 'rate', 'drive_dac_zero', 'echo', 'tc_options')}
        summary['preflight']['checks'] = [c for c in pf.get('checks', []) if c['level'] != 'ok']
        summary['preflight']['start_state'] = pf.get('start_state')
    cells = A.summarise_cells(rows)

    out = run_dir / 'derived' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    A.write_csv(out / 'conditions.csv', rows)
    A.write_csv(out / 'cells.csv', cells)
    write_new(out / 'summary.json', dumps(summary))
    write_new(out / 'report.md', report(cfg, summary, cells, rows))
    if plots:
        try:
            from . import plots as PL
            pd = out / 'plots'
            pd.mkdir()
            PL.amp_vs_drive(cells, pd / 'amp_vs_drive.png')
            PL.tau_vs_tc(rows, expected_tau, pd / 'tau_vs_tc.png')
            PL.baseline_vs_tc(rows, pd / 'baseline_vs_tc.png')
            PL.anchors(rows, pd / 'anchors.png')
            ref = cfg['reference']['drive']
            for tc in sorted({r['tc'] for r in rows if r['kind'] == 'matrix'}, key=tc_seconds):
                pick = next((i for i, (_, _, r) in records.items() if r['kind'] == 'matrix' and r['tc'] == tc
                             and r['drive_cmd'] == ref), None) or next(
                    (i for i, (_, _, r) in records.items() if r['kind'] == 'matrix' and r['tc'] == tc), None)
                if pick is not None:
                    t, data, r = records[pick]
                    PL.trace(t, data, r, f'c{pick:04d}: drive {r["drive_cmd"]:g}, TC {tc}',
                             pd / f'trace_c{pick:04d}_TC{tc.replace(" ", "")}.png')
        except ImportError as exc:
            summary['plots'] = f'skipped: {exc}'
    summary['derived_dir'] = str(out)
    return summary


def _f(x, fmt='.4g') -> str:
    return '–' if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def report(cfg: dict, s: dict, cells: list[dict], rows: list[dict]) -> str:
    env = s.get('environment') or {}
    L = [f'# qPlus open-loop characterisation: {cfg["name"]}', '',
         f'Run folder `{s["run_dir"]}`; analysed {s["analysed"]} (qplus {s["qplus_version"]}).',
         f'Instrument: {env.get("instrument")}; sxm_anfatec {env.get("sxm_anfatec")}.', '']
    if env.get('simulated'):
        L += ['> **SIMULATED DATA.** Numbers show the format only; they are not measurements.', '']
    L += ['QPlusAmpl is in volts as SXM outputs it. It is **not** a displacement: no voltage-to-motion '
          'calibration is applied.', '', '## Run', '',
          f'Records: {s["n_records"]}; status: ' + ', '.join(f'{k} {v}' for k, v in sorted(s['status_counts'].items(), key=str)), '']
    pf = s.get('preflight') or {}
    if pf:
        L += [f'Preflight: k_cal (Drive DAC V per drive unit) = {_f(pf.get("k_cal"), ".5g")}, capture rate '
              f'{_f(pf.get("rate"), ".0f")} sweeps/s, start state {pf.get("start_state")}.']
        for c in pf.get('checks', []):
            L.append(f'- {c["level"]}: {c["check"]}: {c["detail"]}')
        L.append('')
    hr = s.get('headroom')
    if hr:
        L += [f'Headroom pass at TC {hr["tc"]} ({hr["rule"]}): allowed drives {hr["allowed"]}.', '',
              '| drive | A_ss (V) | A/D vs lowest | InA / range | stop |', '|---|---|---|---|---|']
        for st in hr['steps']:
            comp = st.get('compression')
            L.append(f'| {st["drive"]:g} | {_f(st.get("amp_ss"))} | {_f(1 - comp if comp is not None else None, ".4f")} '
                     f'| {_f(st.get("ina_fraction"), ".1%")} | {st.get("stop") or ""} |')
        L.append('')
    warns = {}
    for r in rows:
        for w in r.get('warnings', []):
            key = w.split(':')[0][:70]
            warns[key] = warns.get(key, 0) + 1
    aborts = [(r['idx'], a) for r in rows for a in r.get('aborts', [])]
    if warns or aborts:
        L += ['Warnings (count): ' + '; '.join(f'{k} ({v})' for k, v in warns.items())] if warns else []
        L += [f'- ABORT c{i:04d} {a["kind"]}: {a["reason"]}' for i, a in aborts]
        L.append('')

    L += ['## 1. Steady response per drive × TC', '',
          '| drive | TC | n | A_ss (V) ± CI95 | phase (°) | A/D (V/V) | τ (s) ± CI95 | baseline mean (V) |',
          '|---|---|---|---|---|---|---|---|']
    for c in cells:
        if c['kind'] != 'matrix':
            continue
        L.append(f'| {c["drive_cmd"]:g} | {c["tc"]} | {c["n"]} | {_f(c["amp_ss_mean"])} ± {_f(c["amp_ss_ci95"], ".2g")} '
                 f'| {_f(c["phase_ss_mean"], ".2f")} | {_f(c["amp_per_drive_mean"])} | {_f(c["tau_s_mean"])} ± '
                 f'{_f(c["tau_s_ci95"], ".2g")} | {_f(c["baseline_mean_mean"], ".3g")} |')
    L += ['', '## 2. Linearity (A_ss vs measured drive, per TC)', '',
          '| TC | slope (V/V) ± SE | offset (V) ± SE | max rel. residual | A/D relative to the lowest drive |',
          '|---|---|---|---|---|']
    for tc, v in s['linearity'].items():
        rel = ', '.join(f'{d}: {x:.4f}' for d, x in v['amp_per_drive_rel_to_lowest'].items())
        L.append(f'| {tc} | {_f(v["slope_v_per_v"])} ± {_f(v["slope_se"], ".2g")} | {_f(v["offset_v"], ".2g")} ± '
                 f'{_f(v["offset_se"], ".2g")} | {_f(v["max_rel_resid"], ".2%")} | {rel} |')
    tp, tv = s.get('tau_pooled'), s.get('tau_vs_tc') or {}
    L += ['', '## 3. Ring-down: mechanical decay vs readout filter', '']
    if tp:
        L.append(f'Pooled τ = {tp["mean_s"]:.5f} ± {tp["ci95_s"]:.2g} s (95 % CI, n = {tp["n"]}) → '
                 f'Q = {tp["q"]:.0f} ± {tp["q_ci95"]:.0f} (f0 = {tp["f0_used_hz"]} Hz); configured expectation '
                 f'{s["expected_tau_s"]:.4f} s.')
    if tv:
        z = abs(tv['slope_over_se'])
        verdict = ('no detectable TC dependence (|slope/SE| < 2): the fitted decay is mechanical, not the readout filter.'
                   if z < 2 else
                   f'statistically detectable (|slope/SE| ≥ 2), size {tv["change_over_range_rel"]:+.2%} over the TC range: '
                   'compare with the accuracy you need; if it matters, start the fit later (fit_settle_accuracy).')
        L.append(f'τ vs TC: slope {tv["slope_per_s"]:.3g} s/s ± {_f(tv["slope_se"], ".2g")} '
                 f'(|slope/SE| = {_f(z, ".2f")}; change over the TC range {tv["change_over_range_rel"]:+.2%}): {verdict}')
        L += ['', '| TC | τ mean (s) | SD | n |', '|---|---|---|---|']
        L += [f'| {k} | {_f(v["mean"])} | {_f(v["sd"], ".2g")} | {v["n"]} |' for k, v in tv['per_tc'].items()]
    dv = s.get('delay_vs_tc') or {}
    if dv:
        L += ['', 'Apparent ring-down delay (extrapolated exponential meets A_ss) vs the filter group delay n·TC:', '',
              '| TC | delay mean (s) | expected n·TC (s) |', '|---|---|---|']
        exp = {r['tc']: r.get('ringdown_delay_expected_s') for r in rows if r.get('ringdown_delay_expected_s')}
        L += [f'| {k} | {_f(v["mean"], ".3g")} | {_f(exp.get(k), ".3g")} |' for k, v in dv['per_tc'].items()]
    bv = s.get('baseline_vs_tc') or {}
    L += ['', '## 4. Zero-drive baseline', '']
    if bv:
        L.append(f'log-log slope of the baseline mean vs TC = {bv["loglog_slope"]:.3f} (white noise: −0.5). '
                 f'Implied noise density per quadrature (Rayleigh model, in QPlusAmpl volts) median '
                 f'{_f(bv["noise_density_v_rthz_median"], ".3g")} V/√Hz, '
                 f'relative spread {_f(bv["noise_density_spread_rel"], ".1%")} across conditions.')
        bc = {}
        for r in rows:
            if math.isfinite(r.get('baseline_mean', math.nan)):
                bc.setdefault(r['tc'], []).append(r)
        L += ['', '| TC | n | mean (V) | std (V) | discard used (s) | noise density (V/√Hz) |', '|---|---|---|---|---|---|']
        import numpy as np
        for tc in sorted(bc, key=tc_seconds):
            rs = bc[tc]
            disc = [r['baseline_discard_s'] for r in rs if 'baseline_discard_s' in r]
            L.append(f'| {tc} | {len(rs)} | {_f(np.mean([r["baseline_mean"] for r in rs]), ".3g")} | '
                     f'{_f(np.mean([r["baseline_std"] for r in rs]), ".3g")} | '
                     f'{_f(float(np.mean(disc)) if disc else None, ".2f")} (driven) | '
                     f'{_f(float(np.nanmean([r.get("noise_density_v_rthz", math.nan) for r in rs])), ".3g")} |')
    da, dt_ = s.get('drift_amp') or {}, s.get('drift_tau') or {}
    L += ['', '## 5. Repeatability and drift (anchors)', '']
    if da:
        L.append(f'Anchor A_ss: n = {da["n"]}, spread {_f(da.get("spread_rel"), ".2%")}, trend '
                 f'{_f(da.get("slope_rel_per_h"), "+.2%")}/h. Anchor τ trend {_f(dt_.get("slope_per_h"), "+.3g")} s/h.')
    L += ['', '## Methods', '', f'- Ring-down: {A.MODEL_RINGDOWN}. Window from the Drive-DAC step + settle_time(TC, roll-off, '
          f'{s["params"]["fit_settle_accuracy"]:g}) to {s["params"]["fit_floor_factor"]:g}× the baseline mean or '
          f'{s["params"]["fit_max_tau"]:g} τ. SE inflated by sqrt(N/N_eff), N_eff = window × 2 ENBW.',
          f'- Ring-up (secondary): {A.MODEL_RINGUP}.',
          f'- Baseline: from ≥ {s["params"]["discard_s"]:g} s after the Drive-DAC step off (extended until the fitted '
          f'tail is < {s["params"]["residual_max"]:g} × std) to the record end; block SE ({s["params"]["blocks"]} blocks).',
          '- Steady window: the last steady_s before the step-off command, minus a 0.1 s margin.',
          '- Cells: mean, SD, t-distribution 95 % CI over repeats. Linearity, TC regressions: ordinary least squares.',
          '- Full parameters: summary.json → params; every per-condition window is in conditions.csv.']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--discard', type=float, default=5.0, help='minimum baseline discard after the step off, s')
    ap.add_argument('--no-auto-discard', action='store_true')
    a = ap.parse_args(argv)
    s = analyse_run(Path(a.run_dir), A.AnalysisParams(discard_s=a.discard, discard_auto=not a.no_auto_discard),
                    plots=not a.no_plots)
    print(f'Wrote {s["derived_dir"]} (report.md, summary.json, conditions.csv, cells.csv, plots/)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
