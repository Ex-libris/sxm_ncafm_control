"""A simulated qPlus sensor + DNC lock-in with the same methods as QPlusSxm, for tests and dry runs.

Complex envelope a(t) of the sensor response (QPlusAmpl volts), relative to the reference at f_use:

    da/dt = -(1/tau_m + j 2 pi (f0 - f_use)) a + (G / tau_m) (D(t) + w(t)),   tau_m = Q / (pi f0)

D = drive actually applied (command after a DAC latency), w = thermal force noise (gives a Lorentzian peak
of width f0/Q at f0). Detector noise (white, V/sqrt(Hz)) is added before the lock-in, whose output filter
is ``stages`` cascaded first-order sections with time constant TC on X + jY. QPlusAmpl = |X + jY|,
Phase = arg + offset. Everything is solved exactly per sample (scipy.signal.lfilter), so nothing sleeps.

Fault switches for the safety tests: ``afl_on`` (the Drive DAC no longer follows the command), ``pll_on``
(the Frequency DAC wanders), ``amp_sat_v`` (readout compression), ``pll_gains`` (non-zero PLL gains).
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.signal import lfilter

from . import filters as P
from .config import tc_seconds
from .instrument import SNAPSHOT_NAMES, Event, Record, SafetyError, now_iso

try:                                       # units and scales as the real driver table has them
    from sxm_anfatec.driver import CHANNELS
except Exception:                          # noqa: BLE001 - no sxm_anfatec / no pywin32: a fixed copy for the sim
    CHANNELS = {'Frequency': (-9, 'DAC9', 'Hz', 0.00232831), 'Drive': (-10, 'DAC10', 'V', 4.97789e-09),
                'QPlusAmpl': (-12, 'DAC12', 'V', 3.64244e-09), 'Phase': (-13, 'DAC13', '°', 0.001),
                'df': (-40, 'DAC40', 'Hz', 0.00232838), 'InA': (8, 'ADC4', 'V', 3.07991e-09)}

TC_OPTIONS = ['0.1 ms', '0.2 ms', '0.5 ms', '1 ms', '2 ms', '5 ms', '10 ms', '20 ms', '50 ms', '100 ms']


@dataclass
class SimSensor:
    f0: float = 25563.8
    q: float = 44451.0
    f_use: float = 25563.8
    gain_v_per_unit: float = 100.0         # steady QPlusAmpl (V) per drive unit, on resonance
    phase_offset_deg: float = 30.0
    noise_v_rthz: float = 1e-7             # detector noise density at the lock-in input
    thermal_v: float = 0.0                 # rms thermal amplitude of the envelope (V)
    amp_sat_v: float | None = None         # readout compression: A -> sat * tanh(A / sat)
    dac_v_per_unit: float = 1.0            # Drive DAC volts per drive unit
    dac_latency_s: float = 0.02
    write_latency_s: float = 0.05
    ina_v_per_v: float = 1.0
    afl_on: bool = False
    afl_level: float = 3e-5                # drive the 'AFL' holds when afl_on
    pll_on: bool = False
    pll_gains: tuple = (0.0, 0.0)
    seed: int = 1

    @property
    def tau_m(self) -> float:
        return self.q / (math.pi * self.f0)


class SimQPlus:
    simulated = True
    version = 'simulation'

    def __init__(self, sensor: SimSensor | None = None, writable=frozenset({'dnc.drive', 'dnc.tc'}),
                 drive: float = 1e-5, tc: str = '2 ms'):
        self.s = sensor or SimSensor()
        self.writable = frozenset(writable)
        self.channels = CHANNELS
        self.rng = np.random.default_rng(self.s.seed)
        self.state = {
            'dnc.drive': drive, 'dnc.used_freq': self.s.f_use, 'dnc.tc': tc, 'dnc.rolloff': '12 db/oct',
            'dnc.input_gain': '1', 'dnc.output_gain': '±0.1', 'dnc.q': 175175.0, 'dnc.f_peak': 25564.0,
            'dnc.tau_us': 2.18e6, 'dnc.sweep_start': 25560.0, 'dnc.sweep_stop': 25566.0,
            'amp.ref': 1.0, 'amp.kp': 1e6, 'amp.ki': 10.0, 'amp.tau': '10 ms',
            'pll.kp': self.s.pll_gains[0], 'pll.ki': self.s.pll_gains[1],
            'feedback.mode': 'STM + PLL AFM', 'z.feedback_off': True, 'stm.status': 'Piezo Retracted',
        }
        self.d_actual = self._d_eff(drive)
        self.a = self._steady(self.d_actual)       # it has been driven for a while
        self.filt = [self.a] * 4
        self._x_prev = complex(self.d_actual)
        self._f_walk = 0.0
        self.clock = 0.0

    def describe(self) -> str:
        return f'SIMULATION (f0 {self.s.f0} Hz, Q {self.s.q:g}, tau {self.s.tau_m:.3f} s)'

    # -- settings ----------------------------------------------------------------------------------------

    def read(self, name: str) -> tuple[Any, str | None]:
        if name in self.state:
            return self.state[name], None
        return None, f'KeyError: {name}'

    def snapshot(self, names=SNAPSHOT_NAMES) -> dict:
        out = {'time': now_iso()}
        for n in names:
            v, err = self.read(n)
            out[n] = v if err is None else {'error': err}
        return out

    def options(self, name: str) -> list | None:
        return {'dnc.tc': TC_OPTIONS, 'dnc.rolloff': ['6 db/oct', '12 db/oct', '24 db/oct'],
                'dnc.output_gain': ['±0.1', '±1', '±10']}.get(name)

    def set(self, name: str, value: Any) -> dict:
        if name not in self.writable:
            raise SafetyError(f'{name!r} is not writable by this campaign (allowed: {sorted(self.writable)})')
        before = self.state.get(name)
        if name == 'dnc.tc' and value not in TC_OPTIONS:
            raise ValueError(f'dnc.tc: {value!r} is not one of {TC_OPTIONS}')
        self.state[name] = value
        return {'name': name, 'before': before, 'requested': value, 'after': self.state[name]}

    def channel_info(self, names) -> tuple[dict, dict]:
        return {n: self.channels[n][2] for n in names}, {n: self.channels[n][3] for n in names}

    # -- physics -----------------------------------------------------------------------------------------

    def _d_eff(self, cmd: float) -> float:
        return self.s.afl_level if self.s.afl_on else float(cmd)

    def _lam(self) -> complex:
        return 1 / self.s.tau_m + 2j * math.pi * (self.s.f0 - self.s.f_use)

    def _steady(self, d: float) -> complex:
        return self.s.gain_v_per_unit * d / (self.s.tau_m * self._lam())

    def _segment(self, n: int, dt: float, d: float, tc_s: float, stages: int) -> tuple[np.ndarray, np.ndarray]:
        lam = self._lam()
        p = cmath.exp(-lam * dt)
        q = self.s.gain_v_per_unit / self.s.tau_m * (1 - p) / lam
        x = np.full(n, complex(d))
        if self.s.thermal_v > 0:
            sw = self.s.thermal_v * math.sqrt(1 - abs(p) ** 2) / abs(q) / math.sqrt(2)
            x = x + sw * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        a, _ = lfilter([0, q], [1, -p], x, zi=[p * self.a + q * self._x_prev])
        self.a, self._x_prev = complex(a[-1]), complex(x[-1])
        sn = self.s.noise_v_rthz * math.sqrt(0.5 / dt) / math.sqrt(2)
        y = a + sn * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        alpha = 1 - math.exp(-dt / tc_s)
        for k in range(stages):
            y, _ = lfilter([alpha], [1, -(1 - alpha)], y, zi=[(1 - alpha) * self.filt[k]])
            self.filt[k] = complex(y[-1])
        return a, y

    def _channels(self, names, a, y, n) -> dict:
        amp = np.abs(y)
        if self.s.amp_sat_v:
            amp = self.s.amp_sat_v * np.tanh(amp / self.s.amp_sat_v)
        out = {}
        for nm in names:
            if nm == 'QPlusAmpl':
                out[nm] = amp
            elif nm == 'Phase':
                out[nm] = (np.degrees(np.angle(y)) + self.s.phase_offset_deg + 180) % 360 - 180
            elif nm == 'Drive':
                lsb = self.channels['Drive'][3]
                level = self.s.dac_v_per_unit * self.d_actual
                if self.s.afl_on:
                    level = level * (1 + 0.01 * self.rng.standard_normal(n))
                out[nm] = np.round(np.full(n, level) / lsb) * lsb
            elif nm == 'Frequency':
                f = np.full(n, self.state['dnc.used_freq'], float)
                if self.s.pll_on:
                    f = f + self._f_walk + np.cumsum(self.rng.standard_normal(n)) * 2e-4
                    self._f_walk = float(f[-1] - self.state['dnc.used_freq'])
                out[nm] = f
            elif nm == 'InA':
                ph = self.rng.uniform(0, 2 * math.pi, n)
                out[nm] = math.sqrt(2) * self.s.ina_v_per_v * np.abs(a) * np.sin(ph)
            else:
                out[nm] = np.zeros(n)
        return out

    def record(self, names: list[str], duration: float, events: list[Event] = (), rate: float = 5000.0,
               stop: Callable[[], bool] = lambda: False, trip: dict | None = None,
               safe_events: list[tuple[str, Any]] = (), after_trip_s: float = 1.0) -> Record:
        units, scales = self.channel_info(names)
        rec = Record(np.array([]), {}, units, scales, rate_requested=rate, started=now_iso())
        dt = 1.0 / rate
        end_i = int(round(duration * rate)) + 1
        pending = sorted(events, key=lambda e: e.at)
        effects: list[tuple[float, str, Any]] = []          # (time, name, value): when the hardware follows
        chunks = {nm: [] for nm in names}
        i = 0

        def write(label, name, value, at):
            m = {'label': label, 'name': name, 'value': value, 'at': at, 't_cmd': at}
            try:
                res = self.set(name, value)
                m.update(before=res['before'], after=res['after'])
                lat = self.s.dac_latency_s if name == 'dnc.drive' else 0.0
                effects.append((at + lat, name, value))
            except Exception as exc:                                # noqa: BLE001 - as the real record does
                m['error'] = f'{type(exc).__name__}: {exc}'
            m['t_ack'] = at + self.s.write_latency_s
            rec.marks.append(m)
            return m

        while i < end_i:
            if stop():
                rec.stopped = True
                break
            t_i = i * dt
            if pending and pending[0].at <= t_i + 1e-12:
                ev = pending.pop(0)
                if 'error' in write(ev.label, ev.name, ev.value, ev.at):
                    rec.error = rec.marks[-1]['error']
                    break
                continue
            due = [e for e in effects if e[0] <= t_i + 1e-12]
            if due:
                for e in due:
                    effects.remove(e)
                    if e[1] == 'dnc.drive':
                        self.d_actual = self._d_eff(e[2])
                continue
            cands = ([pending[0].at] if pending else []) + [e[0] for e in effects]
            nxt = min(cands) if cands else math.inf
            j = min(end_i, max(i + 1, math.ceil(nxt / dt - 1e-9))) if math.isfinite(nxt) else end_i
            tc_s = tc_seconds(self.state['dnc.tc'])
            a, y = self._segment(j - i, dt, self.d_actual, tc_s, P.stages(int(self.state['dnc.rolloff'].split()[0])))
            seg = self._channels(names, a, y, j - i)
            if trip and rec.tripped is None:
                hit = [(int(np.argmax(np.abs(seg[n]) > lim)), n) for n, lim in trip.items()
                       if n in seg and np.any(np.abs(seg[n]) > lim)]
                if hit:
                    k, n = min(hit)
                    rec.tripped = f'{n} = {seg[n][k]:.4g} exceeded {trip[n]:.4g}'
                    seg = {nm: v[:k + 1] for nm, v in seg.items()}
                    j = i + k + 1
                    pending = []
                    for name, value in safe_events:
                        write('safe', name, value, j * dt)
                    end_i = min(end_i, j + int(after_trip_s * rate))
            for nm in names:
                chunks[nm].append(seg[nm])
            i = j
        n = sum(len(c) for c in chunks[names[0]]) if names else 0
        rec.t = np.arange(n) * dt
        rec.data = {nm: (np.concatenate(chunks[nm]) if chunks[nm] else np.zeros(0)) for nm in names}
        rec.duration = float(rec.t[-1]) if n else 0.0
        self.clock += rec.duration
        return rec

    def wait(self, seconds: float, stop: Callable[[], bool] = lambda: False) -> bool:
        """Advance the virtual clock: the envelope relaxes towards its steady state, the filters follow."""
        ss = self._steady(self.d_actual)
        self.a = ss + (self.a - ss) * cmath.exp(-self._lam() * seconds)
        self.filt = [self.a] * 4
        self._x_prev = complex(self.d_actual)
        self.clock += seconds
        return not stop()

    def close(self):
        pass
