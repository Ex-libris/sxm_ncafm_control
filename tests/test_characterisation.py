"""qPlus open-loop campaign: config, schedule, safety (refusals, aborts, trips, stop/resume, allowlist),
lossless storage, and the analysis recovering the simulated sensor. No hardware, no sleeping."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sxm_ncafm_control.characterisation import analysis as A
from sxm_ncafm_control.characterisation import config as C
from sxm_ncafm_control.characterisation.analyse import analyse_run
from sxm_ncafm_control.characterisation.instrument import SafetyError
from sxm_ncafm_control.characterisation.runner import Campaign, write_new
from sxm_ncafm_control.characterisation.sim import SimQPlus, SimSensor

DEFAULT = Path(__file__).resolve().parents[1] / 'characterisation' / 'configs' / 'open_loop_matrix.json'


def small_cfg(**over) -> C.Config:
    cfg = C.Config()
    cfg.operator.afl_off_confirmed = cfg.operator.no_surface_confirmed = True
    cfg.design.repeats = 1
    cfg.factors.drives = [5e-6, 1e-5]
    cfg.factors.tcs = ['1 ms', '20 ms']
    cfg.factors.baseline_tcs = ['1 ms']
    cfg.timing.off_s = 15.0
    cfg.timing.baseline_s = 10.0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class Recorder(SimQPlus):
    """Remembers every name written."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.written = []

    def set(self, name, value):
        self.written.append(name)
        return super().set(name, value)


def run(cfg, sensor=None, stop=None, tmp=None, start_drive=1e-5, **kw):
    inst = Recorder(sensor or SimSensor(), {cfg.instrument.drive_param, cfg.instrument.tc_param}, drive=start_drive)
    d = Path(tmp) / 'run'
    out = Campaign(cfg, inst, d, stop=stop).run(**kw)
    return out, inst, d


class Config(unittest.TestCase):
    def test_default_file_loads(self):
        cfg = C.load(DEFAULT)
        self.assertFalse(cfg.operator.afl_off_confirmed)          # the operator must set it
        self.assertEqual(cfg.reference.drive, 1e-5)
        self.assertIn(1e-5, cfg.factors.drives)

    def test_refusals(self):
        base = json.loads(DEFAULT.read_text(encoding='utf-8'))
        for path, value in ((('loops', 'afl'), 'on'), (('factors', 'drives'), [1e-5, 5e-3]),
                            (('factors', 'tcs'), ['1 msec']), (('timing', 'discard_s'), 99.0)):
            d = json.loads(json.dumps(base))
            d[path[0]][path[1]] = value
            with self.assertRaises(C.ConfigError, msg=str(path)):
                C.from_dict(d)
        d = json.loads(json.dumps(base))
        d['limits']['typo'] = 1
        with self.assertRaises(C.ConfigError):
            C.from_dict(d)

    def test_schedule_balanced_and_reproducible(self):
        cfg = C.load(DEFAULT)
        drives = cfg.factors.drives
        a, b = C.build_schedule(cfg, drives), C.build_schedule(cfg, drives)
        self.assertEqual(a, b)
        for blk in range(1, cfg.design.repeats + 1):
            cells = [(e['drive'], e['tc']) for e in a if e['block'] == blk and e['kind'] == 'matrix']
            self.assertEqual(sorted(cells), sorted((d, t) for d in drives for t in cfg.factors.tcs))
        self.assertEqual(a[0]['kind'], 'anchor')
        self.assertEqual(a[-1]['kind'], 'anchor')
        orders = [[(e['drive'], e['tc']) for e in a if e['block'] == blk and e['kind'] == 'matrix'] for blk in (1, 2)]
        self.assertNotEqual(orders[0], orders[1])                 # interleaved differently per block
        self.assertGreater(C.estimate_minutes(cfg, a), 30)


class Safety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowlist(self):
        inst = SimQPlus(writable={'dnc.drive', 'dnc.tc'})
        for name in ('z.dz', 'pll.kp', 'amp.ki', 'z.feedback_off', 'dnc.used_freq'):
            with self.assertRaises(SafetyError):
                inst.set(name, 0)

    def test_operator_not_confirmed_refuses_and_writes_nothing(self):
        cfg = small_cfg()
        cfg.operator.afl_off_confirmed = False
        out, inst, _ = run(cfg, tmp=self.tmp.name)
        self.assertEqual(out['status'], 'refused')
        self.assertEqual(inst.written, [])

    def test_pll_gains_refuse(self):
        out, inst, _ = run(small_cfg(), SimSensor(pll_gains=(-300.0, -3000.0)), tmp=self.tmp.name)
        self.assertEqual(out['status'], 'refused')
        self.assertIn('PLL', out['message'])
        self.assertEqual(inst.written, [])

    def test_tc_not_offered_refuses(self):
        cfg = small_cfg()
        cfg.factors.tcs = ['1 ms', '3 ms']
        out, _, _ = run(cfg, tmp=self.tmp.name)
        self.assertEqual(out['status'], 'refused')
        self.assertIn('3 ms', out['message'])

    def test_afl_on_fails_the_drive_echo(self):
        out, inst, _ = run(small_cfg(), SimSensor(afl_on=True), tmp=self.tmp.name)
        self.assertEqual(out['status'], 'refused')
        self.assertIn('drive echo', out['message'])
        self.assertEqual(inst.state['dnc.drive'], 0.0)              # left at off after a refusal that wrote

    def test_pll_moving_frequency_stops_the_run(self):
        """A slow wander may pass the 6 s preflight; the first condition's check catches it."""
        out, inst, _ = run(small_cfg(), SimSensor(pll_on=True), tmp=self.tmp.name)
        self.assertIn(out['status'], ('refused', 'aborted'))
        self.assertIn('Frequency DAC moves', out['message'])
        self.assertEqual(inst.state['dnc.drive'], 0.0)

    def test_only_drive_and_tc_written_and_restored(self):
        out, inst, _ = run(small_cfg(), tmp=self.tmp.name)
        self.assertEqual(out['status'], 'complete', out['message'])
        self.assertEqual(set(inst.written), {'dnc.drive', 'dnc.tc'})
        self.assertEqual(inst.state['dnc.drive'], 1e-5)            # end_drive 'start'
        self.assertEqual(inst.state['dnc.tc'], '2 ms')

    def test_compression_ends_the_headroom_ladder(self):
        cfg = small_cfg()
        cfg.factors.drives = [2.5e-6, 5e-6, 1e-5, 2e-5, 4e-5]
        cfg.factors.tcs = ['1 ms']
        cfg.factors.baseline_tcs = []
        out, _, d = run(cfg, SimSensor(amp_sat_v=0.004), tmp=self.tmp.name, preflight_only=False, max_blocks=1)
        allowed = json.loads((d / 'headroom.json').read_text(encoding='utf-8'))['allowed']
        self.assertLess(len(allowed), 5)
        self.assertIn(2.5e-6, allowed)

    def test_amplitude_trip_sets_drive_off_and_aborts(self):
        cfg = small_cfg()
        cfg.design.headroom = False
        cfg.reference.drive = 2.5e-6                                # echo and anchors stay below the limit
        cfg.limits.amp_max_v = 0.0007                               # the 1e-5 cells (1 mV) exceed it
        out, inst, d = run(cfg, tmp=self.tmp.name, start_drive=2.5e-6)
        self.assertEqual(out['status'], 'aborted', out['message'])
        self.assertIn('limit', out['message'])
        self.assertEqual(inst.state['dnc.drive'], 0.0)             # stays off after an abort
        metas = [json.loads(p.read_text(encoding='utf-8')) for p in (d / 'raw').glob('c0*.json')]
        tripped = [m for m in metas if m['record']['tripped']]
        self.assertEqual(len(tripped), 1)
        self.assertEqual(tripped[0]['drive'], 1e-5)
        self.assertTrue(any(m['label'] == 'safe' and m['value'] == 0.0 for m in tripped[0]['marks']))

    def test_trip_during_preflight_echo_refuses(self):
        cfg = small_cfg()
        cfg.limits.amp_max_v = 0.0007                               # the echo at 2e-5 exceeds it
        out, inst, _ = run(cfg, tmp=self.tmp.name)
        self.assertEqual(out['status'], 'refused')
        self.assertEqual(inst.state['dnc.drive'], 0.0)

    def test_stop_then_resume_without_overwriting(self):
        cfg = small_cfg()
        calls = {'n': 0}

        def stop():
            calls['n'] += 1
            return calls['n'] > 40
        out, inst, d = run(cfg, stop=stop, tmp=self.tmp.name)
        self.assertEqual(out['status'], 'stopped')
        self.assertEqual(inst.state['dnc.drive'], 0.0)
        self.assertEqual(inst.state['dnc.tc'], '2 ms')
        before = {p.name: p.stat().st_mtime_ns for p in (d / 'raw').iterdir()}
        inst2 = Recorder(SimSensor(), {'dnc.drive', 'dnc.tc'})
        out2 = Campaign(cfg, inst2, d).run(resume=True)
        self.assertEqual(out2['status'], 'complete', out2['message'])
        for name, mt in before.items():
            self.assertEqual((d / 'raw' / name).stat().st_mtime_ns, mt, name)
        n_sched = len(json.loads((d / 'schedule.json').read_text(encoding='utf-8'))['entries'])
        self.assertEqual(len([p for p in (d / 'raw').glob('c0*.json')]), n_sched)

    def test_batches_and_time_limit(self):
        cfg = small_cfg()
        cfg.design.repeats = 2
        out, _, d = run(cfg, tmp=self.tmp.name, max_blocks=1)
        self.assertEqual(out['status'], 'batch done')
        cfg2 = small_cfg()
        cfg2.limits.max_total_min = 1.0
        out2, _, _ = run(cfg2, tmp=tempfile.mkdtemp(dir=self.tmp.name))
        self.assertEqual(out2['status'], 'refused')
        self.assertIn('max_total_min', out2['message'])

    def test_write_new_never_overwrites(self):
        p = Path(self.tmp.name) / 'x.json'
        write_new(p, '1')
        with self.assertRaises(FileExistsError):
            write_new(p, '2')


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = small_cfg()
        cfg.design.repeats = 2
        cfg.factors.tcs = ['0.2 ms', '5 ms', '50 ms']
        cfg.factors.baseline_tcs = ['0.2 ms', '50 ms']
        cls.sensor = SimSensor()
        cls.out, _, cls.dir = run(cfg, cls.sensor, tmp=cls.tmp.name)
        cls.summary = analyse_run(cls.dir, plots=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_campaign_complete(self):
        self.assertEqual(self.out['status'], 'complete', self.out['message'])

    def test_tau_and_q_recovered_and_tc_independent(self):
        tp = self.summary['tau_pooled']
        self.assertAlmostEqual(tp['mean_s'] / self.sensor.tau_m, 1, delta=0.005)
        self.assertAlmostEqual(tp['q'] / self.sensor.q, 1, delta=0.005)
        self.assertLess(abs(self.summary['tau_vs_tc']['change_over_range_rel']), 0.01)

    def test_delay_follows_filter_group_delay(self):
        per = self.summary['delay_vs_tc']['per_tc']
        self.assertAlmostEqual(per['50 ms']['mean'], 0.1, delta=0.02)      # 2 stages x 50 ms

    def test_baseline_white_noise_slope(self):
        self.assertAlmostEqual(self.summary['baseline_vs_tc']['loglog_slope'], -0.5, delta=0.06)

    def test_linearity(self):
        for tc, v in self.summary['linearity'].items():
            self.assertAlmostEqual(v['slope_v_per_v'] / self.sensor.gain_v_per_unit, 1, delta=0.01, msg=tc)

    def test_outputs_are_new_files(self):
        d = Path(self.summary['derived_dir'])
        for name in ('conditions.csv', 'cells.csv', 'summary.json', 'report.md'):
            self.assertTrue((d / name).exists(), name)
        self.assertIn('SIMULATED', (d / 'report.md').read_text(encoding='utf-8'))

    def test_storage_is_lossless_counts(self):
        js = next((self.dir / 'raw').glob('c0001_*.json'))
        _, data, meta = A.load_condition(js)
        scale = meta['record']['scales']['Frequency']
        f = data['Frequency']
        self.assertTrue(np.allclose(f / scale, np.rint(f / scale)))
        self.assertLess(abs(np.mean(f) - self.sensor.f_use), scale)


class RingdownFit(unittest.TestCase):
    def test_fit_starts_after_the_filter(self):
        """An exponential through a 2-stage filter: the fitted tau must not depend on TC."""
        from scipy.signal import lfilter
        rate, tau = 2000.0, 0.55
        t = np.arange(0, 12, 1 / rate)
        rng = np.random.default_rng(3)
        for tc in (0.2e-3, 5e-3, 50e-3):
            x = np.where(t < 1, 1e-3, 1e-3 * np.exp(-(t - 1) / tau)) + 0j
            x = x + 3e-7 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
            a = 1 - math.exp(-1 / rate / tc)
            y = x
            for _ in range(2):
                y = lfilter([a], [1, -(1 - a)], y, zi=[(1 - a) * 1e-3])[0]
            amp = np.abs(y)
            base = amp[t > 9]
            r = A.fit_ringdown(t, amp, 1.0, tc, 12, 25563.8, base.mean(), base.std(), tau, A.AnalysisParams())
            self.assertTrue(r['ringdown_ok'])
            self.assertAlmostEqual(r['tau_s'] / tau, 1, delta=0.005, msg=f'TC {tc}')


if __name__ == '__main__':
    unittest.main()


class DriveLevels(unittest.TestCase):
    """SXM's Drive field keeps only the digits it shows (5 decimals on the instrument)."""

    def test_rounded_level_is_refused_at_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            cfg.factors.drives = [2.5e-6, 1e-5]
            out, inst, d = run(cfg, SimSensor(drive_decimals=5), tmp=tmp)
            self.assertEqual(out['status'], 'refused')
            self.assertIn('drive levels SXM accepts', out['message'])
            self.assertIn('2.5e-06 -> 0.0', out['message'])
            self.assertEqual(inst.state['dnc.drive'], 0.0)
            self.assertFalse(list((d / 'raw').glob('c0*.json')))       # no condition ran

    def test_default_levels_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = C.load(DEFAULT)
            cfg.operator.afl_off_confirmed = cfg.operator.no_surface_confirmed = True
            out, _, d = run(cfg, SimSensor(drive_decimals=5), tmp=tmp, preflight_only=True)
            self.assertEqual(out['status'], 'preflight', out['message'])
            pf = json.loads(next(d.glob('preflight_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(set(pf['drive_levels_shown']), {'1e-05', '3e-05', '0.0001', '0.0003', '0.001', '0.003'})


class PllOffCheck(unittest.TestCase):
    """Open loop only a MOVING excitation frequency is a problem; constant offsets are recorded and noted."""

    def test_instrument_reading_2026_11_06(self):
        # Preflight on the instrument: Frequency DAC 25563.7495 Hz flat at use = 25563.71 Hz, df not recorded.
        problems, notes, v = A.pll_off_check(np.full(1000, 25563.7495), None, 25563.71, 0.005, 0.00232831)
        self.assertEqual(problems, [])
        self.assertEqual(notes, [])
        self.assertAlmostEqual(v['f_exc_hz'], 25563.71, delta=0.002)          # exact 1e7/2^32 scale

    def test_only_moving_is_a_problem(self):
        moving = 25563.71 + np.cumsum(np.full(1000, 1e-4))
        self.assertTrue(A.pll_off_check(moving, None, 25563.71, 0.005)[0])
        problems, notes, _ = A.pll_off_check(np.full(100, 25563.75), np.full(100, 0.04), 25563.71, 0.005)
        self.assertEqual(problems, [])
        self.assertTrue(any('frozen' in n for n in notes))
        problems, notes, _ = A.pll_off_check(np.full(100, 25564.71), None, 25563.71, 0.005)
        self.assertEqual(problems, [])
        self.assertTrue(notes)

    def test_campaign_with_rounded_scale_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _, _ = run(small_cfg(), SimSensor(freq_rel_error=1.53e-6), tmp=tmp)
            self.assertEqual(out['status'], 'complete', out['message'])

    def test_frozen_pll_offset_is_recorded_not_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _, d = run(small_cfg(), SimSensor(df_frozen=0.04), tmp=tmp)
            self.assertEqual(out['status'], 'complete', out['message'])
            pf = json.loads(next(d.glob('preflight_*.json')).read_text(encoding='utf-8'))
            info = [c for c in pf['checks'] if c['check'] == 'excitation frequency (record only)'][0]
            self.assertIn('frozen', info['detail'])
            s = analyse_run(d, plots=False)
            self.assertAlmostEqual(s['excitation']['df_median_hz'], 0.04, delta=1e-3)

    def test_moving_frequency_still_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, inst, _ = run(small_cfg(), SimSensor(pll_on=True), tmp=tmp)
            self.assertIn(out['status'], ('refused', 'aborted'))
            self.assertIn('moves', out['message'])
            self.assertEqual(inst.state['dnc.drive'], 0.0)

    def test_detuning_effect_in_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            sensor = SimSensor(f_use=25563.8 + 0.05)                     # excited 50 mHz above f0
            out, _, d = run(small_cfg(), sensor, tmp=tmp)
            self.assertEqual(out['status'], 'complete', out['message'])
            fx = analyse_run(d, plots=False)['excitation']
            self.assertAlmostEqual(fx['detune_median_hz'], 0.05, delta=0.01)
            x = 2 * sensor.q * 0.05 / sensor.f0
            self.assertAlmostEqual(fx['amp_factor'], 1 / math.sqrt(1 + x * x), delta=0.01)


class Retries(unittest.TestCase):
    """QPlusSxm retries a transient read refusal of the strict bridge, never a write SXM rejected."""

    def setUp(self):
        from sxm_ncafm_control.characterisation import instrument as I
        self.I = I
        self._pause, I.RETRY_PAUSE_S = I.RETRY_PAUSE_S, 0.0

    def tearDown(self):
        self.I.RETRY_PAUSE_S = self._pause

    def make(self, failures):
        from sxm_anfatec.bridge import SXMBridgeError

        class Writer:
            def __init__(self):
                self.calls = 0

            def set(self, name, value):
                self.calls += 1
                if self.calls <= len(failures):
                    raise failures[self.calls - 1]
                return {'before': 1.0, 'after': value}

            def get(self, name):
                self.calls += 1
                if self.calls <= len(failures):
                    raise failures[self.calls - 1]
                return 0.0

        w = Writer()
        return self.I.QPlusSxm({'dnc.drive'}, driver=object(), writer=w), w, SXMBridgeError

    def test_transient_read_refusal_is_retried(self):
        from sxm_anfatec.bridge import SXMBridgeError
        empty = SXMBridgeError("Dynamic Non-Contact R / Phi: field 'Range' missing or empty")
        inst, w, _ = self.make([empty, empty])
        self.assertEqual(inst.set('dnc.drive', 0.0)['attempts'], 3)
        inst, w, _ = self.make([empty])
        self.assertEqual(inst.read('dnc.drive'), (0.0, None))

    def test_gives_up_after_three(self):
        from sxm_anfatec.bridge import SXMBridgeError
        empty = SXMBridgeError('empty')
        inst, w, _ = self.make([empty] * 3)
        with self.assertRaises(SXMBridgeError):
            inst.set('dnc.drive', 0.0)
        self.assertEqual(w.calls, 3)

    def test_rejected_write_is_not_retried(self):
        from sxm_anfatec.writer import SXMWriteError
        inst, w, _ = self.make([SXMWriteError('dnc.drive is disabled (greyed out) in SXM; nothing sent')])
        with self.assertRaises(SXMWriteError):
            inst.set('dnc.drive', 0.0)
        self.assertEqual(w.calls, 1)
