"""
Noise of a quiet window: a stretch of recording with the loops running and nothing being stepped.

Step trains say how fast a loop is; only a quiet window says how noisy it is. Measured per channel: level,
drift-tolerant rms, rms in frequency bands, the noise left after averaging over one pixel dwell time (what ends
up in an image), and for Drive the share of time at its zero floor. Between channels: coherence, which tells
whether two noises share a source (e.g. Drive noise reaching the PLL's Phase).

Pure numpy/scipy, on plain ``(t, y)`` arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from . import metrics as M

BANDS: Tuple[Tuple[float, float], ...] = ((1.0, 10.0), (10.0, 50.0), (50.0, 200.0))
PIXEL_TIMES: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3)
COHERENCE_PAIRS: Tuple[Tuple[str, str], ...] = (("Drive", "Phase"), ("QPlusAmpl", "Phase"), ("Drive", "df"))
DRIVE_FLOOR_LEVEL = 0.1          # Drive below this fraction of its median counts as at the zero floor


def band_key(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} Hz"


@dataclass
class ChannelNoise:
    mean: float
    rms: float                                    # detrended standard deviation
    rel_rms: float                                # rms / |mean| (nan when the mean is ~0, e.g. df, Phase)
    bands: Dict[str, float] = field(default_factory=dict)
    pixel: Dict[float, float] = field(default_factory=dict)   # dwell time [s] -> std of pixel means
    floor_frac: float = math.nan                  # Drive only


@dataclass
class QuietMetrics:
    duration_s: float
    fs: float
    channels: Dict[str, ChannelNoise] = field(default_factory=dict)
    asd: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)   # for plotting, log-binned
    coherence: Dict[str, Dict[str, float]] = field(default_factory=dict)          # 'A~B' -> band -> mean coherence

    def get(self, channel: str, what: str = "rms", key=None) -> float:
        """One number, nan when not measured: ``what`` in rms / rel_rms / mean / floor_frac / band / pixel."""
        c = self.channels.get(channel)
        if c is None:
            return math.nan
        if what == "band":
            return c.bands.get(key, math.nan)
        if what == "pixel":
            return c.pixel.get(key, math.nan)
        return getattr(c, what)


def _uniform(t, y):
    """The capture's slices are close to regular: treat them as uniform at the median spacing. Interpolating
    onto a grid instead would low-pass the noise being measured."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    return t, y, 1.0 / float(np.median(np.diff(t)))


def _nperseg(n: int, fs: float, resolution_hz: float = 0.25) -> int:
    want = int(2 ** math.ceil(math.log2(max(fs / resolution_hz, 64))))
    return int(min(want, 2 ** int(math.log2(max(n // 2, 64)))))


def _log_bin(f, a, n_bins: int = 300):
    keep = f > 0
    f, a = f[keep], a[keep]
    if len(f) <= n_bins:
        return f, a
    edges = np.geomspace(f[0], f[-1] * 1.0001, n_bins + 1)
    idx = np.digitize(f, edges) - 1
    fb, ab = [], []
    for k in range(n_bins):
        m = idx == k
        if m.any():
            fb.append(float(np.mean(f[m])))
            ab.append(float(np.sqrt(np.mean(a[m] ** 2))))
    return np.asarray(fb), np.asarray(ab)


def quiet_metrics(t, channels: Dict[str, np.ndarray], *, bands: Sequence[Tuple[float, float]] = BANDS,
                  pixel_times: Sequence[float] = PIXEL_TIMES) -> Optional[QuietMetrics]:
    """Noise of every channel in ``channels`` over the samples ``t``. None if the window is too short."""
    t = np.asarray(t, float)
    if len(t) < 200 or t[-1] - t[0] < 1.0:
        return None
    out = None
    uni: Dict[str, np.ndarray] = {}
    for name, y in channels.items():
        tu, yu, fs = _uniform(t, y)
        if out is None:
            out = QuietMetrics(duration_s=float(t[-1] - t[0]), fs=fs)
        uni[name] = yu
        mean = float(np.mean(yu))
        rms = M.detrended_std(yu)
        scale = max(abs(mean), 1e-30)
        rel = rms / scale if abs(mean) > 5 * rms else math.nan
        cn = ChannelNoise(mean=mean, rms=rms, rel_rms=rel)
        for lo, hi in bands:
            if hi <= fs / 2:
                cn.bands[band_key(lo, hi)] = M.band_rms(yu, fs, lo, hi)
        for tp in pixel_times:
            cn.pixel[float(tp)] = M.pixel_noise(tu, yu, tp)
        if name == "Drive":
            med = float(np.median(yu))
            cn.floor_frac = float(np.mean(yu <= DRIVE_FLOOR_LEVEL * med)) if med > 0 else math.nan
        out.channels[name] = cn
        f, p = signal.welch(yu - mean, fs=fs, nperseg=_nperseg(len(yu), fs), detrend="linear")
        out.asd[name] = _log_bin(f, np.sqrt(p))
    if out is None:
        return None
    for a, b in COHERENCE_PAIRS:
        if a in uni and b in uni and np.std(uni[a]) > 0 and np.std(uni[b]) > 0:
            n = min(len(uni[a]), len(uni[b]))
            f, c = signal.coherence(uni[a][:n], uni[b][:n], fs=out.fs, nperseg=_nperseg(n, out.fs, 0.5),
                                    detrend="linear")
            out.coherence[f"{a}~{b}"] = {band_key(lo, hi): float(np.mean(c[(f >= lo) & (f < hi)]))
                                         for lo, hi in bands if hi <= out.fs / 2 and ((f >= lo) & (f < hi)).any()}
    return out
