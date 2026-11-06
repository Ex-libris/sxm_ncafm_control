"""The instrument as the qPlus campaign sees it. Only existing sxm_anfatec calls are used:

    AnfatecSXMWriter.get / .set / .options     settings by short name (every write is read back by the writer)
    SXMIOCTL.read_scaled                        the fast hardware channels (QPlusAmpl, Phase, Drive, ...)
    sxm_anfatec.CHANNELS                        units and scale factors, recorded with every capture

What this layer adds: a write allowlist (the campaign can only touch the drive and the lock-in time
constant; z, feedback gains and loop switches are never written) and ``record()``, a time-stamped
polling capture during which writes are made at scheduled offsets, so a step and its response are in
one record with the command time (t_cmd) and the moment the writer confirmed it (t_ack).

``SimQPlus`` (characterisation/sim.py) has the same methods for tests and for dry runs.
"""
from __future__ import annotations

import datetime
import threading
import time
from pathlib import Path
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Names read before and after every condition (read only; failures are recorded, not raised).
SNAPSHOT_NAMES = (
    'dnc.drive', 'dnc.used_freq', 'dnc.tc', 'dnc.rolloff', 'dnc.input_gain', 'dnc.output_gain',
    'dnc.q', 'dnc.f_peak', 'dnc.tau_us', 'dnc.sweep_start', 'dnc.sweep_stop',
    'amp.ref', 'amp.kp', 'amp.ki', 'amp.tau', 'pll.kp', 'pll.ki',
    'feedback.mode', 'z.feedback_off', 'stm.status',
)

RETRIES, RETRY_PAUSE_S = 3, 0.7        # transient read refusals (see QPlusSxm._transient)

# InA input range by DNC input gain (manual: Gain 1 = +-7 V, Gain 10 = +-0.7 V).
INA_RANGE_V = {'1': 7.0, '10': 0.7}


class SafetyError(RuntimeError):
    """A write outside the allowlist, or a refused precondition."""


@dataclass
class Event:
    at: float                 # s after the record start
    name: str
    value: Any
    label: str = ''


@dataclass
class Record:
    t: np.ndarray                              # s from record start (middle of each polling sweep)
    data: dict                                 # channel -> float64 array, scaled (saved as int32 counts)
    units: dict
    scales: dict
    marks: list = field(default_factory=list)  # one per event: label, name, value, at, t_cmd, t_ack, before, after
    rate_requested: float = 0.0
    started: str = ''
    duration: float = 0.0
    stopped: bool = False
    tripped: str | None = None
    error: str | None = None

    @property
    def rate_achieved(self) -> float:
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 and self.t[-1] > self.t[0] else 0.0

    def mark(self, label: str) -> dict | None:
        return next((m for m in self.marks if m['label'] == label), None)


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='milliseconds')


def as_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(',', '.'))
    except ValueError:
        return None


class QPlusSxm:
    """The real SXM through sxm_anfatec. Shares the app's driver handle, or opens its own."""
    simulated = False

    def __init__(self, writable: set[str], driver=None, writer=None):
        """``driver``: an open SXMIOCTL to share (the app's ``conn.driver``; it is not closed here). Without
        one a handle is opened, which fails while another program (the app) holds it."""
        import sxm_anfatec
        # 2026.11.06: an empty DNC status bar (no resonance sweep since SXM started) no longer makes the
        # strict bridge refuse the whole DNC window, i.e. dnc.drive / dnc.used_freq / dnc.tc. Only this tool
        # needs it, so the app's SXM_ANFATEC_MIN stays lower.
        sxm_anfatec.require('2026.11.06')
        from sxm_anfatec import CHANNELS
        from sxm_anfatec.bridge import AnfatecSXMBridge
        from sxm_anfatec.writer import AnfatecSXMWriter
        from sxm_anfatec import bridge as _bridge
        from sxm_anfatec.bridge import SXMBridgeError
        if not getattr(_bridge, 'DNC_RANGE_MAY_BE_UNSELECTED', False):
            # Same version date (2026.11.06) as the empty-status fix, so require() cannot tell them apart.
            raise RuntimeError(f'sxm_anfatec {sxm_anfatec.__version__} at {Path(_bridge.__file__).parent} lacks the '
                               '2026-11-06 DNC Range fix (at zero drive SXM can leave no Range button selected, and '
                               'this copy then refuses the whole DNC window). Copy the current sxm_anfatec folder.')
        from sxm_anfatec.writer import SXMWriteError
        self._bridge_error, self._write_error = SXMBridgeError, SXMWriteError
        self.version = sxm_anfatec.__version__
        self.channels = CHANNELS
        self.writer = writer or AnfatecSXMWriter(AnfatecSXMBridge(strict=True))
        self._own_driver = driver is None
        if driver is None:
            from sxm_anfatec.driver import SXMIOCTL
            driver = SXMIOCTL()
        self.driver = driver
        self.writable = frozenset(writable)

    def describe(self) -> str:
        return f'SXM instrument (sxm_anfatec {self.version})'

    # -- settings --------------------------------------------------------------------------------------

    def _transient(self, exc: Exception) -> bool:
        """A read refusal (the strict bridge found a field empty, e.g. the DNC Range radio group while SXM
        re-ranges the output gain after a drive change), not a write SXM rejected."""
        return isinstance(exc, self._bridge_error) and not isinstance(exc, self._write_error)

    def read(self, name: str) -> tuple[Any, str | None]:
        for attempt in range(1, RETRIES + 1):
            try:
                return self.writer.get(name), None
            except Exception as exc:                                 # noqa: BLE001 - recorded, never raised
                if attempt < RETRIES and self._transient(exc):
                    time.sleep(RETRY_PAUSE_S)
                    continue
                return None, f'{type(exc).__name__}: {exc}' + (f' (after {attempt} tries)' if attempt > 1 else '')

    def snapshot(self, names=SNAPSHOT_NAMES) -> dict:
        out = {'time': now_iso()}
        for n in names:
            v, err = self.read(n)
            out[n] = v if err is None else {'error': err}
        return out

    def options(self, name: str) -> list | None:
        return self.writer.options(name)

    def set(self, name: str, value: Any) -> dict:
        """Write and read back. A transient read refusal is retried (RETRIES x, RETRY_PAUSE_S apart): every
        write here sets an absolute value, so sending it again is harmless."""
        if name not in self.writable:
            raise SafetyError(f'{name!r} is not writable by this campaign (allowed: {sorted(self.writable)})')
        for attempt in range(1, RETRIES + 1):
            try:
                res = self.writer.set(name, value)
                res['attempts'] = attempt
                return res
            except Exception as exc:
                if attempt < RETRIES and self._transient(exc):
                    time.sleep(RETRY_PAUSE_S)
                    continue
                raise

    # -- capture -------------------------------------------------------------------------------------------

    def channel_info(self, names) -> tuple[dict, dict]:
        units = {n: self.channels[n][2] for n in names}
        scales = {n: self.channels[n][3] for n in names}
        return units, scales

    def record(self, names: list[str], duration: float, events: list[Event] = (), rate: float = 5000.0,
               stop: Callable[[], bool] = lambda: False, trip: dict | None = None,
               safe_events: list[tuple[str, Any]] = (), after_trip_s: float = 1.0) -> Record:
        """Poll ``names`` for ``duration`` s at up to ``rate`` sweeps/s; make ``events`` at their offsets.

        ``trip`` = {channel: max |value|}: when exceeded, the remaining events are dropped, ``safe_events``
        are written at once, and the record ends ``after_trip_s`` later. A write error or ``stop()`` ends
        the record early (the partial record is returned, flagged)."""
        units, scales = self.channel_info(names)
        for n in names:
            if n not in self.channels:
                raise KeyError(f'unknown IOCTL channel {n!r}')
        ts, cols = array('d'), [array('d') for _ in names]
        done, tripped, poll_err = threading.Event(), [], []
        limits = [(i, n, float(trip[n])) for i, n in enumerate(names) if trip and n in trip]
        read, period = self.driver.read_scaled, 1.0 / rate
        t0 = time.perf_counter()

        def poll():
            nxt = time.perf_counter()
            try:
                while not done.is_set():
                    now = time.perf_counter()
                    if now < nxt:
                        time.sleep(0)
                        continue
                    nxt = max(nxt + period, now)
                    vals = [read(n) for n in names]
                    ts.append((now + time.perf_counter()) / 2 - t0)
                    for c, v in zip(cols, vals):
                        c.append(v)
                    if limits and not tripped:
                        for i, n, lim in limits:
                            if abs(vals[i]) > lim:
                                tripped.append(f'{n} = {vals[i]:.4g} exceeded {lim:.4g}')
                                break
            except Exception as exc:                                 # noqa: BLE001 - reported in the record
                poll_err.append(f'{type(exc).__name__}: {exc}')

        rec = Record(np.array([]), {}, units, scales, rate_requested=rate, started=now_iso())
        th = threading.Thread(target=poll, name='qplus-capture', daemon=True)
        th.start()
        pending = sorted(events, key=lambda e: e.at)

        def write(ev_label, name, value, at):
            t_cmd = time.perf_counter() - t0
            m = {'label': ev_label, 'name': name, 'value': value, 'at': at, 't_cmd': t_cmd}
            try:
                res = self.set(name, value)
                m.update(before=res.get('before'), after=res.get('after'))
            except Exception as exc:                                 # noqa: BLE001 - ends the record
                m['error'] = f'{type(exc).__name__}: {exc}'
            m['t_ack'] = time.perf_counter() - t0
            rec.marks.append(m)
            return m

        end = duration
        try:
            while True:
                el = time.perf_counter() - t0
                if poll_err:
                    rec.error = 'capture: ' + poll_err[0]
                    break
                if tripped and rec.tripped is None:
                    rec.tripped = tripped[0]
                    pending = []
                    for name, value in safe_events:
                        write('safe', name, value, el)
                    end = min(end, time.perf_counter() - t0 + after_trip_s)
                if stop() and not rec.stopped:
                    rec.stopped = True
                    break
                if pending and el >= pending[0].at:
                    ev = pending.pop(0)
                    if 'error' in write(ev.label, ev.name, ev.value, ev.at):
                        rec.error = rec.marks[-1]['error']
                        break
                    continue
                if el >= end:
                    break
                time.sleep(min(0.01, max(0.0, (pending[0].at if pending else end) - el)))
        finally:
            done.set()
            th.join(2.0)
        n = min([len(ts)] + [len(c) for c in cols])
        rec.t = np.frombuffer(ts, dtype=np.float64)[:n].copy()
        rec.data = {nm: np.frombuffer(c, dtype=np.float64)[:n].copy() for nm, c in zip(names, cols)}
        rec.duration = float(rec.t[-1]) if n else 0.0
        return rec

    # -- time ----------------------------------------------------------------------------------------------

    def wait(self, seconds: float, stop: Callable[[], bool] = lambda: False) -> bool:
        """Sleep, checking ``stop`` every 50 ms. False if stopped."""
        end = time.perf_counter() + seconds
        while True:
            if stop():
                return False
            left = end - time.perf_counter()
            if left <= 0:
                return True
            time.sleep(min(0.05, left))

    def close(self):
        if not self._own_driver:                                    # the app's shared handle: the app closes it
            return
        try:
            self.driver.close()
        except Exception:                                            # noqa: BLE001
            pass
