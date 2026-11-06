"""The campaign runner: preflight, headroom pass, randomized blocks, per-condition checks, safe stop.

Writes: only ``instrument.drive_param`` and ``instrument.tc_param`` (an allowlist enforced by the
instrument layer). It never writes z, the feedback mode, a feedback gain or a loop switch.

Before every condition the relevant settings are read back and logged; after it they are read again and
the state is verified. Each condition is one continuous record: zero drive (pre) -> drive step on (ring-up,
steady) -> drive step off (ring-down, baseline). The raw record (.npz) and its sidecar (.json: requested
and read-back settings, marks, checks, warnings, online metrics) are written to new files only.

Stops: ``stop()`` (Ctrl+C in the CLI, the GUI Stop button, or a file named STOP in the run folder) ends
the current record at once. Then, and after any abort or error: drive -> off_drive, TC -> its start value,
both read back. After a normal end the drive goes back to the value the run found (``end_drive: start``).

Abort classes: 'limit' (amplitude / input over a limit: the drive was set to off at once by the
record's trip), 'state' (the instrument is not in the state the run needs: a loop acting on drive or
frequency, a setting changed under the run, a write not taken), 'error' (communication, timeout).
In the headroom pass a 'limit' only ends the drive ladder; everywhere else any abort ends the campaign.
"""
from __future__ import annotations

import datetime
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import VERSION
from . import analysis as A
from . import config as C
from .instrument import INA_RANGE_V, Event, as_number


class Stopped(Exception):
    pass


class Abort(Exception):
    def __init__(self, kind: str, reason: str):
        super().__init__(f'{kind}: {reason}')
        self.kind, self.reason = kind, reason


def _fmt_drive(d) -> str:
    return 'D0' if d is None else f'D{d:.3g}'


def _fmt_tc(tc: str) -> str:
    return 'TC' + tc.replace(' ', '')


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_new(path: Path, text: str) -> None:
    """Create ``path``; never overwrite."""
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write(text)


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=_json_default, ensure_ascii=False)


class Campaign:
    def __init__(self, cfg: C.Config, inst, run_dir: Path, progress: Callable | None = None,
                 stop: Callable[[], bool] | None = None):
        self.cfg, self.inst, self.dir = cfg, inst, Path(run_dir)
        self.raw = self.dir / 'raw'
        self._progress = progress or (lambda event, data: None)
        self._stop = stop or (lambda: False)
        self.start_state: dict = {}
        self.k_cal: float | None = None
        self.dac_zero = 0.0
        self.f_exc_ref: float | None = None                    # excitation frequency at the preflight
        self.ina_range: float | None = None
        self.touched = False
        self.results: list[dict] = []

    # ------------------------------------------------------------------------------------- plumbing

    def stop(self) -> bool:
        return self._stop() or (self.dir / 'STOP').exists()

    def log(self, text: str) -> None:
        line = f'{datetime.datetime.now():%H:%M:%S} {text}'
        with open(self.dir / 'run.log', 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
        self._progress('log', line)

    def _numbered(self, stem: str, suffix: str) -> Path:
        n = 1
        while (self.dir / f'{stem}_{n}{suffix}').exists():
            n += 1
        return self.dir / f'{stem}_{n}{suffix}'

    def _environment(self) -> dict:
        return {'qplus_version': VERSION, 'instrument': self.inst.describe(),
                'sxm_anfatec': getattr(self.inst, 'version', None), 'simulated': self.inst.simulated,
                'python': sys.version.split()[0], 'platform': platform.platform(), 'host': platform.node(),
                'channels': {n: {'unit': u, 'scale': self.inst.channel_info([n])[1][n]}
                             for n, u in self.inst.channel_info(self.cfg.instrument.channels)[0].items()}}

    def open(self, resume: bool) -> None:
        if resume:
            if not (self.dir / 'campaign.json').exists():
                raise FileNotFoundError(f'{self.dir} is not a campaign folder (no campaign.json)')
            write_new(self._numbered('resume', '.json'),
                      dumps({'time': datetime.datetime.now().astimezone().isoformat(), **self._environment()}))
        else:
            self.dir.mkdir(parents=True, exist_ok=False)
            write_new(self.dir / 'campaign.json', dumps({'config': self.cfg.to_dict(), 'environment': self._environment(),
                                                         'created': datetime.datetime.now().astimezone().isoformat(),
                                                         'timing': C.describe_timing(self.cfg)}))
        self.raw.mkdir(exist_ok=True)

    def set(self, name: str, value) -> dict:
        self.touched = True
        return self.inst.set(name, value)

    # ------------------------------------------------------------------------------------- preflight

    def preflight(self) -> dict:
        """Read-backs, refusals, channel rate, and the drive echo test (the physical AFL/PLL check)."""
        cfg, ic, lim = self.cfg, self.cfg.instrument, self.cfg.limits
        self._progress('stage', 'preflight')
        snap = self.inst.snapshot()
        checks: list[dict] = []

        def check(name, ok, detail, level='fail'):
            checks.append({'check': name, 'ok': bool(ok), 'level': 'ok' if ok else level, 'detail': detail})
            self.log(f'preflight {"OK  " if ok else level.upper():5s} {name}: {detail}')
            return ok

        self.start_state = {'drive': as_number(snap.get(ic.drive_param)), 'tc': snap.get(ic.tc_param),
                            'used_freq': as_number(snap.get(ic.freq_param))}
        check('operator: AFL switched off', cfg.operator.afl_off_confirmed,
              'SXM "Amplitude feedback On/Off" cannot be read by Python: set operator.afl_off_confirmed after '
              'checking it in SXM (also tested physically by the drive echo below)')
        check('operator: no surface', cfg.operator.no_surface_confirmed,
              'tip retracted / no sample: set operator.no_surface_confirmed (this run never moves z)')
        kp, ki = as_number(snap.get('pll.kp')), as_number(snap.get('pll.ki'))
        check('PLL off (Kp = Ki = 0)', kp == 0 and ki == 0,
              f'read Kp = {snap.get("pll.kp")!r}, Ki = {snap.get("pll.ki")!r}; set both to 0 in SXM '
              '(manual: PLL off = gains 0). This run never writes them.')
        check('AFL gains (record only)', True,
              f'Ref {snap.get("amp.ref")!r}, Kp {snap.get("amp.kp")!r}, Ki {snap.get("amp.ki")!r}, '
              f'Tau {snap.get("amp.tau")!r}: saved gains, not proof of the loop state', level='info')
        check('drive readable', self.start_state['drive'] is not None,
              f'{ic.drive_param} = {snap.get(ic.drive_param)!r}')
        check('used frequency readable', self.start_state['used_freq'] is not None,
              f'{ic.freq_param} = {snap.get(ic.freq_param)!r} Hz (expected f0 {cfg.expect.f0_hz} Hz)')
        try:
            opts = self.inst.options(ic.tc_param) or []
        except Exception as exc:                                     # noqa: BLE001
            opts, why = [], f'{type(exc).__name__}: {exc}'
        else:
            why = ''
        wanted = sorted(set(cfg.factors.tcs + cfg.factors.baseline_tcs + [cfg.reference.tc]), key=C.tc_seconds)
        missing = [x for x in wanted if x not in opts]
        check('TC levels offered by SXM', not missing,
              f'{ic.tc_param} offers {opts}{" (" + why + ")" if why else ""}; missing: {missing}')
        roll = snap.get(ic.rolloff_param)
        check('roll-off (record only)', True, f'{ic.rolloff_param} = {roll!r} (not written)', level='info')
        og = snap.get('dnc.output_gain')
        if ic.output_gain_expected is not None:
            check('output gain', str(og).replace('+-', '±') == ic.output_gain_expected.replace('+-', '±'),
                  f'dnc.output_gain = {og!r}, expected {ic.output_gain_expected!r} (set it in SXM; not written)')
        gain = str(snap.get('dnc.input_gain', '')).strip()
        self.ina_range = INA_RANGE_V.get(gain)
        check('InA range known', self.ina_range is not None or 'InA' not in ic.channels,
              f'dnc.input_gain = {gain!r} -> InA range {self.ina_range} V', level='warn')
        check('feedback / z (record only)', True,
              f'feedback.mode {snap.get("feedback.mode")!r}, z.feedback_off {snap.get("z.feedback_off")!r}, '
              f'status {snap.get("stm.status")!r}: never written by this run', level='info')

        result = {'time': datetime.datetime.now().astimezone().isoformat(), 'snapshot': snap,
                  'start_state': self.start_state, 'tc_options': opts, 'checks': checks}
        if not all(c['ok'] for c in checks if c['level'] == 'fail'):
            result['ok'] = False
            write_new(self._numbered('preflight', '.json'), dumps(result))
            return result

        # channel rate
        rec = self.inst.record(ic.channels, 0.5, rate=ic.sample_rate_hz, stop=self.stop)
        if rec.stopped:
            raise Stopped()
        rate = rec.rate_achieved
        check('capture rate', rate >= 0.5 * ic.sample_rate_hz and not rec.error,
              f'{rate:.0f} sweeps/s of {len(ic.channels)} channels {ic.channels} (requested {ic.sample_rate_hz:g})'
              + (f'; capture error: {rec.error}' if rec.error else ''),
              level='fail' if rate < 0.5 * ic.sample_rate_hz else 'warn')
        if rate < 0.8 * ic.sample_rate_hz:
            checks[-1]['level'] = 'warn'
        result['rate'] = rate

        # drive echo: off -> d1 -> d2 -> off; the Drive DAC must follow in proportion and stay flat
        off = cfg.off_drive
        d1 = cfg.reference.drive
        d2 = 2 * d1 if 2 * d1 <= lim.drive_max else d1 / 2
        ev = [Event(0.5, ic.drive_param, off, 'e0'), Event(2.0, ic.drive_param, d1, 'e1'),
              Event(3.5, ic.drive_param, d2, 'e2'), Event(5.0, ic.drive_param, off, 'e3')]
        self.touched = True
        rec = self.inst.record(ic.channels, 6.0, ev, ic.sample_rate_hz, self.stop, self._trip(),
                               [(ic.drive_param, off)])
        n_echo = 1 + len(list(self.raw.glob('preflight_echo_*.json')))
        self._save(f'preflight_echo_{n_echo}', rec, {'kind': 'preflight_echo', 'events': [e.__dict__ for e in ev]})
        if rec.stopped:
            raise Stopped()
        if rec.error or rec.tripped:
            check('drive echo', False, f'record ended early: {rec.error or rec.tripped}')
            result['ok'] = False
            write_new(self._numbered('preflight', '.json'), dumps(result))
            return result
        m = {k['label']: k for k in rec.marks}
        drv = rec.data['Drive'].astype(float)

        def level(a, b):
            w = A.window(rec.t, m[a]['t_ack'] + 0.3, m[b]['t_cmd'] - 0.05)
            return A.stats(drv[w])

        s0, s1, s2 = level('e0', 'e1'), level('e1', 'e2'), level('e2', 'e3')
        k1 = (s1['mean'] - s0['mean']) / (d1 - off)
        k2 = (s2['mean'] - s0['mean']) / (d2 - off)
        flat = max(s1['std'] / abs(s1['mean']) if s1['mean'] else math.inf,
                   s2['std'] / abs(s2['mean']) if s2['mean'] else math.inf)
        echo_ok = k1 > 0 and k2 > 0 and abs(k2 / k1 - 1) < lim.drive_echo_rel_tol
        check('drive echo (Drive DAC follows the command)', echo_ok,
              f'k = DAC/command: {k1:.5g} at {d1:g}, {k2:.5g} at {d2:g} (ratio {k2 / k1 if k1 else math.nan:.4f}); '
              f'DAC at off {s0["mean"]:.4g} V. If the DAC does not follow: AFL on, or {ic.drive_param} is not the '
              'drive field')
        check('Drive DAC flat while driven', flat < lim.drive_flat_rel,
              f'std/level {flat:.2e} (limit {lim.drive_flat_rel:g}): a loop moving the drive shows here')
        self.k_cal = (k1 + k2) / 2 if echo_ok else None
        self.dac_zero = s0['mean']
        lsb = self.inst.channel_info(['Drive'])[1]['Drive']
        dmin = min([x for x in cfg.factors.drives if x > 0] + [d1])
        counts = dmin * (self.k_cal or 0) / lsb
        check('drive resolution', counts >= lim.min_dac_counts,
              f'smallest drive {dmin:g} = {counts:.0f} DAC counts (limit {lim.min_dac_counts})')
        fuse = self.start_state['used_freq'] or cfg.expect.f0_hz
        scale = self.inst.channel_info(['Frequency'])[1]['Frequency']
        problems, notes, fr = A.pll_off_check(rec.data['Frequency'], rec.data.get('df'), fuse, lim.freq_tol_hz, scale)
        check('PLL off: excitation frequency does not move', not problems,
              f'Frequency DAC std {fr["freq_dac_std"]:.1e} Hz (limit {lim.freq_tol_hz:g})'
              + ('. ' + '; '.join(problems) if problems else ''))
        check('excitation frequency (record only)', True,
              f'{fr["f_exc_hz"]:.4f} Hz (Frequency DAC {fr["freq_dac"]:.4f} Hz with the driver scale, '
              f'{fr["f_exc_hz"]:.4f} Hz with the exact DDS scale 1e7/2^32); use {fuse} Hz; '
              + (f'df {fr["df_dac"]:+.4f} Hz' if 'df_dac' in fr else 'df not recorded')
              + ('. ' + '; '.join(notes) if notes else ''), level='info')
        self.f_exc_ref = fr['f_exc_hz']
        if echo_ok and not problems:
            result['drive_levels_shown'] = self._drive_levels(check)
        result.update(k_cal=self.k_cal, drive_dac_zero=s0['mean'], echo={'k1': k1, 'k2': k2, 'flat': flat},
                      freq_dac=fr, ok=all(c['ok'] for c in checks if c['level'] == 'fail'))
        write_new(self._numbered('preflight', '.json'), dumps(result))
        return result

    def _drive_levels(self, check) -> dict:
        """Write every planned drive level once and read it back, then drive -> off and let the sensor ring down.

        SXM rounds a typed number to the digits its field shows: on the instrument (2026-11-06) the DNC Drive
        field showed 2.5e-06 as 0.0 while 1e-05 and 2e-05 were kept. A level SXM changes would run at the wrong
        drive (or at zero), so it is refused here, before any condition, with what SXM showed."""
        cfg, ic = self.cfg, self.cfg.instrument
        levels = sorted({x for x in cfg.factors.drives if x > 0} | {cfg.reference.drive})
        shown = {}
        for lvl in levels:
            if self.stop():
                raise Stopped()
            try:
                shown[lvl] = self.inst.set(ic.drive_param, lvl).get('after')
            except Exception as exc:                                 # noqa: BLE001 - reported in the check
                after = (getattr(exc, 'result', None) or {}).get('after')
                shown[lvl] = after if after is not None else f'{type(exc).__name__}: {exc}'
        self.inst.set(ic.drive_param, cfg.off_drive)
        bad = {lvl: v for lvl, v in shown.items()
               if not (isinstance(as_number(v), float) and math.isclose(as_number(v), lvl, rel_tol=1e-6))}
        ok = [f'{lvl:g}' for lvl in levels if lvl not in bad]
        check('drive levels SXM accepts', not bad,
              f'kept: {ok or "none"}' + ('; changed by SXM: ' + ', '.join(f'{lvl:g} -> {v!r}' for lvl, v in bad.items())
                                         + '. SXM rounds the Drive field to the digits it shows: use levels it keeps '
                                         '(edit factors.drives) or change the field\'s unit in SXM' if bad else ''))
        if not self.inst.wait(C.settle_s(cfg), self.stop):          # let the last level ring down
            raise Stopped()
        return {f'{k:g}': v for k, v in shown.items()}

    def _trip(self) -> dict:
        trip = {'QPlusAmpl': self.cfg.limits.amp_max_v}
        if 'InA' in self.cfg.instrument.channels and self.ina_range:
            trip['InA'] = self.cfg.limits.ina_max_fraction * self.ina_range
        return trip

    # ------------------------------------------------------------------------------------- conditions

    def done_indices(self) -> set:
        out = set()
        for p in self.raw.glob('c*.json'):
            try:
                out.add(int(p.name[1:5]))
            except ValueError:
                pass
        return out

    def _save(self, stem: str, rec, meta: dict) -> Path:
        # Stored losslessly as the driver's integer counts (read_scaled = count x scale, so value / scale is an
        # integer); value = count x scale on loading. Time as float32 (4 us resolution at 60 s).
        path = self.raw / f'{stem}.npz'
        counts = {f'cnt_{k}': np.clip(np.rint(np.asarray(v, float) / rec.scales[k]), -2**31, 2**31 - 1).astype(np.int32)
                  for k, v in rec.data.items()}
        with open(path, 'xb') as fh:
            np.savez_compressed(fh, t=rec.t.astype(np.float32), **counts,
                                **{f'scale_{k}': np.float64(rec.scales[k]) for k in rec.data})
        meta = {**meta, 'file': path.name, 'record': {
            'started': rec.started, 'duration_s': rec.duration, 'n': len(rec.t), 'rate_requested': rec.rate_requested,
            'rate_achieved': rec.rate_achieved, 'units': rec.units, 'scales': rec.scales, 'stopped': rec.stopped,
            'tripped': rec.tripped, 'error': rec.error}, 'marks': rec.marks}
        write_new(path.with_suffix('.json'), dumps(meta))
        return path

    def run_entry(self, e: dict) -> dict:
        cfg, ic, lim, tm = self.cfg, self.cfg.instrument, self.cfg.limits, self.cfg.timing
        if self.stop():
            raise Stopped()
        t_wall = time.monotonic()
        self._progress('condition_start', e)
        pre = self.inst.snapshot()
        meta = {**e, 'start': datetime.datetime.now().astimezone().isoformat(), 'expect': asdict(cfg.expect),
                'freq_tol_hz': lim.freq_tol_hz, 'scales': self.inst.channel_info(ic.channels)[1],
                'off_drive': cfg.off_drive, 'steady_s': tm.steady_s, 'k_cal': self.k_cal,
                'f_use': as_number(pre.get(ic.freq_param)), 'rolloff': pre.get(ic.rolloff_param),
                'loops': {'afl': 'off (operator-confirmed; switch not readable; drive echo checked)',
                          'pll': f'off (Kp {pre.get("pll.kp")!r}, Ki {pre.get("pll.ki")!r})'},
                'readback_pre': pre, 'writes': [], 'warnings': [], 'checks': []}
        warn, abort = meta['warnings'], []

        if as_number(pre.get('pll.kp')) != 0 or as_number(pre.get('pll.ki')) != 0:
            abort.append(('state', f'PLL gains are not 0 before the condition: {pre.get("pll.kp")!r} / {pre.get("pll.ki")!r}'))
        d_now = as_number(pre.get(ic.drive_param))
        if not abort and (d_now is None or abs(d_now - cfg.off_drive) > 1e-12):
            w = self.set(ic.drive_param, cfg.off_drive)            # start from rest
            meta['writes'].append({**w, 'why': 'drive to off before the condition'})
            if not self.inst.wait(C.settle_s(cfg), self.stop):
                raise Stopped()

        events = []
        if not abort and e['kind'] == 'baseline':
            events = [Event(tm.pre_s, ic.tc_param, e['tc'], 'tc')]
            duration = tm.pre_s + tm.baseline_s
        elif not abort:
            if pre.get(ic.tc_param) != e['tc']:
                w = self.set(ic.tc_param, e['tc'])
                meta['writes'].append({**w, 'why': 'time constant of the condition'})
                if not self.inst.wait(C.tc_wait(cfg, e['tc']), self.stop):
                    raise Stopped()
            t_off = tm.pre_s + C.settle_s(cfg) + tm.steady_s
            events = [Event(tm.pre_s, ic.drive_param, e['drive'], 'on'), Event(t_off, ic.drive_param, cfg.off_drive, 'off')]
            duration = t_off + tm.off_s

        rec = None
        if not abort:
            rec = self.inst.record(ic.channels, duration, events, ic.sample_rate_hz, self.stop, self._trip(),
                                   [(ic.drive_param, cfg.off_drive)])
            if rec.error:
                abort.append(('error', rec.error))
            if rec.tripped:
                abort.append(('limit', f'{rec.tripped}; drive set to {cfg.off_drive:g} at once'))
        post = self.inst.snapshot()
        meta['readback_post'] = post
        meta['tc_readback'] = post.get(ic.tc_param)

        metrics = {}
        if rec is not None and len(rec.t) > 10 and not rec.stopped:
            meta['marks'] = rec.marks
            try:
                metrics = A.analyse_condition(rec.t, rec.data, meta)
            except Exception as exc:                                 # noqa: BLE001 - recorded, raw data kept
                warn.append(f'online analysis failed: {type(exc).__name__}: {exc}')
            abort += self._checks(e, meta, metrics, pre, post, rec)
        if time.monotonic() - t_wall > lim.condition_timeout_s:
            abort.append(('error', f'condition took {time.monotonic() - t_wall:.0f} s (> {lim.condition_timeout_s:g} s)'))

        status = 'stopped' if rec is not None and rec.stopped else ('aborted' if abort else ('warn' if warn else 'ok'))
        meta.update(status=status, aborts=[{'kind': k, 'reason': r} for k, r in abort],
                    online={k: v for k, v in metrics.items() if k != 'params'}, analysis_params=metrics.get('params'))
        stem = f'c{e["idx"]:04d}_{e["kind"]}_b{e["block"]}_{_fmt_drive(e.get("drive"))}_{_fmt_tc(e["tc"])}'
        if rec is not None:
            self._save(stem, rec, meta)
        else:
            write_new(self.raw / f'{stem}.json', dumps(meta))
        summary = {'idx': e['idx'], 'kind': e['kind'], 'block': e['block'], 'drive': e.get('drive'), 'tc': e['tc'],
                   'status': status, 'amp_ss': metrics.get('amp_ss'), 'tau_s': metrics.get('tau_s'),
                   'baseline_mean': metrics.get('baseline_mean'), 'amp_per_drive': metrics.get('amp_per_drive'),
                   'drive_dac': metrics.get('drive_dac'), 'ina_peak': metrics.get('ina_peak'),
                   'warnings': list(warn), 'aborts': meta['aborts'], 'file': stem}
        self.results.append(summary)
        self._progress('condition_done', summary)
        for w_ in warn:
            self.log(f'  c{e["idx"]:04d} warning: {w_}')
        if status == 'stopped':
            raise Stopped()
        if abort:
            kind, reason = abort[0]
            self.log(f'  c{e["idx"]:04d} ABORT ({kind}): {reason}')
            raise Abort(kind, reason)
        return summary

    def _checks(self, e, meta, m, pre, post, rec) -> list[tuple[str, str]]:
        """Per-condition state checks. Hard failures are returned; soft ones are added to the warnings."""
        cfg, ic, lim = self.cfg, self.cfg.instrument, self.cfg.limits
        warn, out = meta['warnings'], []
        if post.get(ic.tc_param) != e['tc']:
            out.append(('state', f'TC reads {post.get(ic.tc_param)!r} after the condition, requested {e["tc"]!r}'))
        d_post = as_number(post.get(ic.drive_param))
        if d_post is None or abs(d_post - cfg.off_drive) > 1e-12:
            out.append(('state', f'drive reads {post.get(ic.drive_param)!r} after the condition, expected {cfg.off_drive:g}'))
        if as_number(post.get('pll.kp')) != 0 or as_number(post.get('pll.ki')) != 0:
            out.append(('state', f'PLL gains changed during the condition: {post.get("pll.kp")!r} / {post.get("pll.ki")!r}'))
        if pre.get('dnc.output_gain') != post.get('dnc.output_gain'):
            warn.append(f'dnc.output_gain changed {pre.get("dnc.output_gain")!r} -> {post.get("dnc.output_gain")!r} '
                        '(SXM auto-range: the drive DAC resolution changed)')
        for key in ('dnc.input_gain', ic.rolloff_param, ic.freq_param, 'amp.kp', 'amp.ki', 'feedback.mode'):
            if pre.get(key) != post.get(key):
                out.append(('state', f'{key} changed during the condition: {pre.get(key)!r} -> {post.get(key)!r}'))
        if rec.rate_achieved < 0.8 * ic.sample_rate_hz:
            warn.append(f'capture rate {rec.rate_achieved:.0f}/s below 80 % of {ic.sample_rate_hz:g}')
        for mk in rec.marks:
            slip = mk['t_cmd'] - mk['at']
            if slip > 0.5:
                warn.append(f'write "{mk["label"]}" {slip:.2f} s late')
        if not m or 'note' in m:
            if m.get('note'):
                out.append(('state', m['note']))
            return out
        for p in m.get('pll_off_problems', []):
            out.append(('state', p))
        if self.f_exc_ref is not None and math.isfinite(m.get('f_exc_hz', math.nan)):
            shift = m['f_exc_hz'] - self.f_exc_ref
            if abs(shift) > lim.freq_tol_hz:
                warn.append(f'excitation frequency {shift:+.4f} Hz from the preflight\'s {self.f_exc_ref:.4f} Hz '
                            '(use changed in SXM?): this cell is at a different detuning')
        if e['kind'] == 'baseline':
            z = self.dac_zero + cfg.off_drive * (self.k_cal or 0)
            if abs(m.get('drive_dac_baseline', 0) - z) > max(1e-9, 0.01 * cfg.reference.drive * (self.k_cal or 1)):
                out.append(('state', f'Drive DAC {m.get("drive_dac_baseline"):.4g} V at zero drive'))
            return out
        echo = m.get('drive_echo_rel', math.nan)
        if math.isfinite(echo) and abs(echo) > lim.drive_echo_rel_tol:
            out.append(('state', f'Drive DAC / command differs from the preflight calibration by {echo:+.1%}: '
                                 'AFL acting on the drive, or a different drive field'))
        if m.get('drive_dac_flat_rel', 0) > lim.drive_flat_rel:
            out.append(('state', f'Drive DAC not constant while driven (std/level {m["drive_dac_flat_rel"]:.2e}): loop active?'))
        if math.isfinite(m.get('amp_ss_drift_rel', math.nan)) and abs(m['amp_ss_drift_rel']) > 0.02:
            warn.append(f'steady amplitude drifts {m["amp_ss_drift_rel"]:+.1%} over the steady window (not settled?)')
        if not m.get('ringdown_ok'):
            warn.append(f'ring-down fit failed: {m.get("ringdown_note", "?")}')
        elif abs(m['tau_s'] / cfg.expect.tau_mech - 1) > 0.3:
            warn.append(f'ring-down tau {m["tau_s"]:.3f} s is >30 % from the expected {cfg.expect.tau_mech:.3f} s (Q changed?)')
        if m.get('residual_over_baseline_std', 0) > 0.2:          # the analysis extends the discard to 0.1 when it can
            warn.append(f'ring-down residual at the baseline start = {m["residual_over_baseline_std"]:.2f} x baseline std '
                        f'(discard {m.get("baseline_discard_s", math.nan):.1f} s): increase timing.off_s')
        return out

    # ------------------------------------------------------------------------------------- stages

    def headroom(self) -> list[float]:
        """Ascending drive ladder at the reference TC; stops at the first level that compresses or nears a limit."""
        path = self.dir / 'headroom.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))['allowed']
        self._progress('stage', 'headroom')
        cfg, lim = self.cfg, self.cfg.limits
        allowed, steps, r0 = [], [], None
        attempt = len({i // 100 for i in self.done_indices() if i >= 9000})     # a stopped pass is re-run whole
        for i, e in enumerate(C.headroom_entries(cfg), 1):
            e['idx'] = 9000 + 100 * attempt + i
            try:
                s = self.run_entry(e)
            except Abort as a:
                if a.kind != 'limit':
                    raise
                steps.append({'drive': e['drive'], 'stop': a.reason})
                break
            ratio = s.get('amp_per_drive')
            r0 = r0 or ratio
            comp = 1 - ratio / r0 if ratio and r0 else math.nan
            ina = (s.get('ina_peak') or 0) / self.ina_range if self.ina_range else math.nan
            amp = (s.get('amp_ss') or 0) / lim.amp_max_v
            step = {'drive': e['drive'], 'amp_ss': s.get('amp_ss'), 'amp_per_drive': ratio, 'compression': comp,
                    'ina_fraction': ina, 'amp_fraction_of_limit': amp}
            reasons = []
            if math.isfinite(comp) and comp > lim.compression_max:
                reasons.append(f'amplitude/drive {comp:.1%} below the lowest level (> {lim.compression_max:.0%})')
            if math.isfinite(ina) and ina > 0.8 * lim.ina_max_fraction:
                reasons.append(f'InA peak at {ina:.0%} of range (stop at 80 % of the {lim.ina_max_fraction:.0%} limit)')
            if amp > 0.8:
                reasons.append(f'QPlusAmpl at {amp:.0%} of limits.amp_max_v')
            steps.append({**step, 'stop': '; '.join(reasons) or None})
            self.log(f'headroom {e["drive"]:g}: A {s.get("amp_ss", math.nan):.4g} V, A/D rel {1 - comp if math.isfinite(comp) else math.nan:.4f}, '
                     f'InA {ina:.0%}' + (f' -> STOP: {"; ".join(reasons)}' if reasons else ''))
            if reasons:
                break
            allowed.append(e['drive'])
        allowed = [d for d in cfg.factors.drives if d in allowed]
        write_new(path, dumps({'allowed': allowed, 'steps': steps, 'tc': cfg.reference.tc,
                               'rule': f'stop at compression > {lim.compression_max:g}, InA > 80 % of '
                                       f'{lim.ina_max_fraction:g} x range, or QPlusAmpl > 80 % of {lim.amp_max_v:g} V'}))
        if not allowed:
            raise Abort('limit', 'no drive level passed the headroom pass')
        return allowed

    def schedule(self, drives: list[float]) -> list[dict]:
        path = self.dir / 'schedule.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))['entries']
        entries = C.build_schedule(self.cfg, drives)
        write_new(path, dumps({'drives': drives, 'seed': self.cfg.design.seed, 'order': self.cfg.design.order,
                               'estimate_min': C.estimate_minutes(self.cfg, entries), 'entries': entries}))
        lines = ['idx,block,kind,drive,tc,est_s'] + [
            f'{x["idx"]},{x["block"]},{x["kind"]},{"" if x["drive"] is None else x["drive"]},{x["tc"]},'
            f'{C.entry_seconds(self.cfg, x):.0f}' for x in entries]
        write_new(self.dir / 'schedule.csv', '\n'.join(lines) + '\n')
        return entries

    # ------------------------------------------------------------------------------------- run

    def run(self, *, resume: bool = False, preflight_only: bool = False, max_blocks: int | None = None,
            max_minutes: float | None = None) -> dict:
        out = {'run_dir': str(self.dir), 'status': 'error', 'message': ''}
        self.open(resume)
        self.log(f'{"resume" if resume else "start"}: {self.inst.describe()}; {C.describe_timing(self.cfg)}')
        try:
            pf = self.preflight()
            out['preflight_ok'] = pf['ok']
            if not pf['ok']:
                bad = [c for c in pf['checks'] if not c['ok'] and c['level'] == 'fail']
                out.update(status='refused', message='; '.join(f'{c["check"]}: {c["detail"]}' for c in bad))
                return out
            if preflight_only:
                out.update(status='preflight', message=f'preflight passed; k_cal {self.k_cal:.5g} V per drive unit')
                return out
            drives = [d for d in self.cfg.factors.drives if d > 0]
            if self.cfg.design.headroom and drives:
                drives = self.headroom()
            entries = self.schedule(drives)
            done = self.done_indices()
            todo = [x for x in entries if x['idx'] not in done]
            if max_blocks is not None and todo:
                first = todo[0]['block']
                todo = [x for x in todo if x['block'] < first + max_blocks]
            if max_minutes is not None:
                kept, acc = [], 0.0
                for x in todo:
                    acc += C.entry_seconds(self.cfg, x) / 60
                    if acc > max_minutes:
                        break
                    kept.append(x)
                todo = kept
            est = C.estimate_minutes(self.cfg, todo)
            if est > self.cfg.limits.max_total_min or len(todo) > self.cfg.limits.max_conditions:
                out.update(status='refused', message=(
                    f'this batch plans {len(todo)} conditions, {est:.0f} min: above limits.max_total_min '
                    f'{self.cfg.limits.max_total_min:g} / max_conditions {self.cfg.limits.max_conditions}. '
                    'Run it in parts with --blocks N or --max-minutes M (then --resume).'))
                return out
            self.log(f'batch: {len(todo)} of {len(entries)} conditions, estimate {est:.0f} min')
            t0 = time.monotonic()
            for n, x in enumerate(todo, 1):
                self._progress('progress', {'n': n, 'of': len(todo), 'elapsed_s': time.monotonic() - t0,
                                            'eta_s': C.estimate_minutes(self.cfg, todo[n - 1:]) * 60})
                self.run_entry(x)
            left = len([x for x in entries if x['idx'] not in self.done_indices()])
            out.update(status='complete' if left == 0 else 'batch done',
                       message=f'{len(todo)} conditions run; {left} left in the schedule')
        except Stopped:
            out.update(status='stopped', message='stopped by the operator')
        except Abort as a:
            out.update(status='aborted', message=f'{a.kind}: {a.reason}')
        except Exception as exc:                                     # noqa: BLE001 - always restore
            out.update(status='error', message=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        finally:
            out['restore'] = self.restore(out['status'])
            out['results'] = self.results
            write_new(self._numbered('end', '.json'), dumps(out))
            self.log(f'end: {out["status"]}: {out["message"]}')
        return out

    def restore(self, status: str) -> list[dict]:
        """Drive to off (after a stop/abort/error) or to the start value (normal end); TC to the start value."""
        if not self.touched:
            return []
        ic = self.cfg.instrument
        normal = status in ('complete', 'batch done', 'preflight')
        drive = self.start_state.get('drive') if (normal and self.cfg.end_drive == 'start') else self.cfg.off_drive
        steps = []
        for name, value in ((ic.drive_param, drive), (ic.tc_param, self.start_state.get('tc'))):
            if value is None:
                continue
            try:
                w = self.inst.set(name, value)
                steps.append({'name': name, 'value': value, 'after': w.get('after'), 'ok': True})
            except Exception as exc:                                 # noqa: BLE001 - tell the operator
                steps.append({'name': name, 'value': value, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
                self.log(f'RESTORE FAILED: set {name} to {value!r} by hand in SXM ({exc})')
        if not normal and self.start_state.get('drive') not in (None, self.cfg.off_drive):
            self.log(f'drive left at {self.cfg.off_drive:g} after "{status}"; it was {self.start_state["drive"]:g} '
                     'before the run')
        return steps
