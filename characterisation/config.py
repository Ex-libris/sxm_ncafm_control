"""Campaign configuration (JSON), its validation, the run schedule and the time estimate.

A campaign = an optional headroom pass (ascending drive at the reference time constant), then ``repeats``
blocks. Each block holds every drive x TC cell once (and the zero-drive baselines, one per TC) in a
shuffled order, opened by an anchor (the reference condition) for drift tracking; one closing anchor ends
the campaign. The seed is stored, so the order can be rebuilt.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from . import filters as P

KIND = 'qplus_open_loop_matrix'
VOLTAGE_LIMIT_ABS = 10.0         # common.VOLTAGE_LIMIT_ABS (not imported: common.py pulls in Qt)


class ConfigError(ValueError):
    pass


@dataclass
class Operator:
    """What Python cannot read from SXM and the operator must confirm before a run."""
    afl_off_confirmed: bool = False          # SXM 'Amplitude feedback On/Off' = off (the switch is not readable)
    no_surface_confirmed: bool = False       # tip retracted far / no sample: nothing to approach


@dataclass
class InstrumentCfg:
    drive_param: str = 'dnc.drive'           # VERIFY: writer name 'by position, unconfirmed'
    tc_param: str = 'dnc.tc'                 # VERIFY: the DNC (qPlus) lock-in TimeConstant
    freq_param: str = 'dnc.used_freq'        # read only in this stage
    rolloff_param: str = 'dnc.rolloff'       # read only
    output_gain_expected: str | None = None  # e.g. '±0.1': checked against read-back, never written
    channels: list = field(default_factory=lambda: ['QPlusAmpl', 'Phase', 'Drive', 'Frequency', 'df', 'InA'])
    sample_rate_hz: float = 2000.0           # polling sweeps per second (all channels per sweep); >= 4x the
                                             # -3 dB bandwidth of the shortest TC (0.2 ms, 12 dB/oct: 0.5 kHz)


@dataclass
class Expect:
    f0_hz: float = 25563.8
    q: float = 44451.0

    @property
    def tau_mech(self) -> float:
        """Amplitude ring-down time Q / (pi f0), s."""
        return self.q / (math.pi * self.f0_hz)


@dataclass
class Limits:
    drive_max: float = 3e-3                  # V rms (SXM drive units); levels above are refused
    amp_max_v: float = 0.05                  # QPlusAmpl above this: drive off at once, campaign aborted
    ina_max_fraction: float = 0.5            # |InA| peak over the InA input range (x1: 7 V, x10: 0.7 V)
    drive_echo_rel_tol: float = 0.03         # Drive DAC vs k_cal x command
    drive_flat_rel: float = 0.003            # Drive DAC std / level in the steady window (a loop moves it)
    freq_tol_hz: float = 0.005               # Frequency DAC vs used frequency (a PLL moves it)
    compression_max: float = 0.03            # amplitude/drive drop vs the lowest drive: stop the headroom pass
    min_dac_counts: int = 200                # smallest drive must be this many DAC counts (quantisation < 0.5 %)
    max_total_min: float = 120.0             # one invocation may not plan longer than this
    max_conditions: int = 250
    condition_timeout_s: float = 240.0


@dataclass
class Timing:
    pre_s: float = 2.0                       # zero-drive record before the drive step on
    settle_tau: float = 10.0                 # ring-up wait, in configured tau (covers tau up to ~0.7 s at 5 tau)
    steady_s: float = 5.0                    # driven steady window
    off_s: float = 25.0                      # after the step off: ring-down, then baseline
    discard_s: float = 5.0                   # baseline statistics start this long after the step off
    baseline_s: float = 30.0                 # zero-drive baseline conditions (stage 1)
    tc_settle_s: float = 1.0                 # wait after a TC change (at least 10 TC)
    overhead_s: float = 4.0                  # read-backs, writes, saving (for the estimate only)


@dataclass
class Design:
    repeats: int = 5
    seed: int = 20261106
    order: str = 'randomized_blocks'         # or 'sequential'
    anchors: bool = True
    headroom: bool = True


@dataclass
class Reference:
    drive: float = 1e-5
    tc: str = '1 ms'


@dataclass
class Factors:
    drives: list = field(default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4])   # V rms; 1e-5 = the field's step
    tcs: list = field(default_factory=lambda: ['0.2 ms', '1 ms', '10 ms', '50 ms'])
    baseline_tcs: list = field(default_factory=list)   # zero-drive baselines per block; [] = none


@dataclass
class Loops:
    afl: str = 'off'
    pll: str = 'off'


@dataclass
class Config:
    name: str = 'open_loop'
    notes: str = ''
    kind: str = KIND
    version: int = 1
    operator: Operator = field(default_factory=Operator)
    instrument: InstrumentCfg = field(default_factory=InstrumentCfg)
    expect: Expect = field(default_factory=Expect)
    limits: Limits = field(default_factory=Limits)
    timing: Timing = field(default_factory=Timing)
    design: Design = field(default_factory=Design)
    reference: Reference = field(default_factory=Reference)
    factors: Factors = field(default_factory=Factors)
    loops: Loops = field(default_factory=Loops)
    off_drive: float = 0.0
    end_drive: str = 'start'                 # after a normal end: 'start' (drive the run found) or 'off'

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data: Any, where: str):
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f'{where}: expected an object, got {data!r}')
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f'{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}')
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        target = f.default_factory() if f.default_factory is not MISSING else f.default
        kwargs[name] = _build(type(target), value, f'{where}.{name}') if is_dataclass(target) else value
    return cls(**kwargs)


def from_dict(data: dict) -> Config:
    if data.get('kind', KIND) != KIND:
        raise ConfigError(f'kind must be {KIND!r}, not {data.get("kind")!r}')
    cfg = _build(Config, data, 'config')
    validate(cfg)
    return cfg


def load(path: str | Path) -> Config:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ConfigError(f'{path}: not valid JSON ({exc})') from exc
    return from_dict(data)


def tc_seconds(text: str) -> float:
    s = str(text).strip().lower().replace('µ', 'u')
    for unit, f in (('ms', 1e-3), ('us', 1e-6), ('s', 1.0)):
        if s.endswith(unit):
            return float(s[:-len(unit)].strip()) * f
    raise ConfigError(f'time constant {text!r} must end in us, ms or s (as SXM shows it, e.g. "1 ms")')


def rolloff_db(text: Any) -> int:
    try:
        return int(str(text).split()[0])
    except (ValueError, IndexError):
        return 12


def validate(cfg: Config) -> None:
    """Static checks; the instrument-dependent ones (TC options, gains) are in the runner's preflight."""
    if cfg.loops.afl != 'off' or cfg.loops.pll != 'off':
        raise ConfigError('this stage runs open loop only: loops.afl and loops.pll must be "off" '
                          '(AFL / PLL characterisation are later, separate stages)')
    if cfg.limits.drive_max > VOLTAGE_LIMIT_ABS:
        raise ConfigError(f'limits.drive_max {cfg.limits.drive_max:g} exceeds the +-{VOLTAGE_LIMIT_ABS:g} V rule of the app')
    d = cfg.factors.drives
    if any(not isinstance(x, (int, float)) or x < 0 or not math.isfinite(x) for x in d):
        raise ConfigError(f'factors.drives must be non-negative numbers: {d}')
    if any(x > cfg.limits.drive_max for x in d + [cfg.reference.drive]):
        raise ConfigError(f'a drive level exceeds limits.drive_max = {cfg.limits.drive_max:g}: {d}')
    if len(set(d)) != len(d):
        raise ConfigError(f'factors.drives has duplicates: {d}')
    for tc in cfg.factors.tcs + cfg.factors.baseline_tcs + [cfg.reference.tc]:
        tc_seconds(tc)
    if cfg.reference.drive <= 0:
        raise ConfigError('reference.drive must be > 0')
    if not (0 <= cfg.off_drive < min([x for x in d if x > 0] or [cfg.reference.drive])):
        raise ConfigError('off_drive must be >= 0 and below the smallest drive level')
    if cfg.design.repeats < 1 or cfg.design.order not in ('randomized_blocks', 'sequential'):
        raise ConfigError('design.repeats >= 1; design.order "randomized_blocks" or "sequential"')
    t = cfg.timing
    if t.discard_s >= t.off_s:
        raise ConfigError('timing.discard_s must be shorter than timing.off_s (no baseline window left)')
    if t.discard_s >= t.baseline_s and cfg.factors.baseline_tcs:
        raise ConfigError('timing.discard_s must be shorter than timing.baseline_s')
    if cfg.end_drive not in ('start', 'off'):
        raise ConfigError('end_drive must be "start" or "off"')
    if cfg.instrument.sample_rate_hz <= 0 or cfg.instrument.sample_rate_hz > 50000:
        raise ConfigError('instrument.sample_rate_hz must be in (0, 50000]')
    for ch in ('QPlusAmpl', 'Drive', 'Frequency'):
        if ch not in cfg.instrument.channels:
            raise ConfigError(f'instrument.channels must include {ch} (used by the safety checks)')


# ----------------------------------------------------------------------------------------------- schedule

def entry(kind: str, drive: float | None, tc: str, block: int) -> dict:
    return {'kind': kind, 'drive': drive, 'tc': tc, 'block': block}


def headroom_entries(cfg: Config) -> list[dict]:
    return [entry('headroom', d, cfg.reference.tc, 0) for d in sorted(x for x in cfg.factors.drives if x > 0)]


def build_schedule(cfg: Config, drives: list[float]) -> list[dict]:
    """Blocks of every (drive, TC) cell and baseline once; shuffled per block with the stored seed."""
    rng = random.Random(cfg.design.seed)
    out = []
    for b in range(1, cfg.design.repeats + 1):
        cells = [entry('matrix', d, tc, b) for d in drives for tc in cfg.factors.tcs]
        cells += [entry('baseline', None, tc, b) for tc in cfg.factors.baseline_tcs]
        if cfg.design.order == 'randomized_blocks':
            rng.shuffle(cells)
        if cfg.design.anchors:
            out.append(entry('anchor', cfg.reference.drive, cfg.reference.tc, b))
        out += cells
    if cfg.design.anchors:
        out.append(entry('anchor', cfg.reference.drive, cfg.reference.tc, cfg.design.repeats))
    for i, e in enumerate(out, 1):
        e['idx'] = i
    return out


def settle_s(cfg: Config) -> float:
    return cfg.timing.settle_tau * cfg.expect.tau_mech


def tc_wait(cfg: Config, tc: str) -> float:
    return max(cfg.timing.tc_settle_s, 10 * tc_seconds(tc))


def driven_duration(cfg: Config) -> float:
    t = cfg.timing
    return t.pre_s + settle_s(cfg) + t.steady_s + t.off_s


def entry_seconds(cfg: Config, e: dict) -> float:
    record = cfg.timing.baseline_s if e['kind'] == 'baseline' else driven_duration(cfg)
    return tc_wait(cfg, e['tc']) + record + cfg.timing.overhead_s


def estimate_minutes(cfg: Config, entries: list[dict]) -> float:
    return sum(entry_seconds(cfg, e) for e in entries) / 60


def describe_timing(cfg: Config) -> str:
    tm = cfg.expect.tau_mech
    return (f'mechanical tau = Q/(pi f0) = {tm:.3f} s; ring-up wait {settle_s(cfg):.1f} s; driven condition '
            f'{driven_duration(cfg):.1f} s of record; lock-in TC levels {cfg.factors.tcs} '
            f'(1 % settling at 12 dB/oct: '
            + ', '.join(f'{P.settle_time(tc_seconds(x), 12) * 1e3:.1f} ms' for x in cfg.factors.tcs) + ')')
