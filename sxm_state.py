"""
Read-back of SXM parameters from the running SXM GUI.

DDE is effectively write-only (see dde_client.py), so this module reads what SXM *shows* instead,
through the passive AnfatecSXMBridge (Win32 query messages only: no clicks, no writes, no DDE,
no driver access). One ``SXMReader.read()`` reads every section once and returns an ``SXMReadout``
keyed like the app's parameter registry (``common.PARAMS_BASE``), plus a few extras the tabs use.

Values are in SXM's current GUI units, the same units DDE writes use.

Mapping notes
-------------
* Scan-parameter keys (amp_ref, amp_ki, amp_kp, pll_kp, pll_ki) come from named bridge fields.
* The four numeric edits of the DNC window have no captions. They are taken by position, top to
  bottom, as DNC 1..4 = sweep start, sweep stop, ``use`` frequency, Drive (layout of Femto_28_4:
  three frequency edits, the Output Gain radio group, then the Drive edit). A different count
  raises instead of guessing. Not yet confirmed on the hardware: with SXM running, send
  ``DNCPara(3, use + 0.01)`` (Parameters tab, Used Frequency), read back, check that only
  ``used_freq`` moved, then write the original value back.
* Anything else the bridge can read is in ``readout.raw`` (the bridge snapshot).
"""
from __future__ import annotations

import datetime
import re
from typing import Any, Dict, Optional

from .common import PARAMS_BASE

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ns": 1e-9}

# key -> (bridge section, field)
_NAMED = {
    "amp_ref": ("Amplitude feedback", "Ref"),
    "amp_ki": ("Amplitude feedback", "Ki"),
    "amp_kp": ("Amplitude feedback", "Kp"),
    "pll_kp": ("PLL", "Kp"),
    "pll_ki": ("PLL", "Ki"),
    "amp_pull_back": ("Amplitude feedback", "Pull back at"),
    "amp_pull_speed": ("Amplitude feedback", "Pull Speed"),
    "topo_ref": ("Topography feedback", "Ref"),
    "topo_ki": ("Topography feedback", "Ki"),
    "topo_kp": ("Topography feedback", "Kp"),
    "scan_range": ("Scan", "Range"),
    "scan_speed": ("Scan", "Speed"),
    "scan_pixel": ("Scan", "Pixel"),
    "slew_rate": ("zControl", "Slew Rate"),
}
# DNC numeric edits, top to bottom -> key (DNCPara index = position + 1)
DNC_EDIT_KEYS = ("dnc_sweep_start", "dnc_sweep_stop", "used_freq", "drive")

LABELS = {
    "dnc_sweep_start": "DNC sweep start (Hz)", "dnc_sweep_stop": "DNC sweep stop (Hz)",
    "afl_output_gain": "AFL output gain (+-V)", "input_gain_ina": "DNC input gain InA (x)",
    "dnc_time_constant_s": "DNC TimeConstant (s)",
    "dnc_rolloff": "DNC RollOff", "amp_tau_s": "Amplitude Tau (s)", "q": "Q (DNC status)",
    "f_peak": "fPeak (Hz, DNC status)", "ring_down_s": "Ring-down tau (s, DNC status)",
    "feedback_off": "zControl Feedback Off", "feedback_mode": "Feedback mode",
    "amp_pull_back": "Amplitude pull back at", "amp_pull_speed": "Amplitude pull speed",
    "topo_ref": "Topography Ref", "topo_ki": "Topography Ki", "topo_kp": "Topography Kp",
    "scan_range": "Scan range", "scan_speed": "Scan speed", "scan_pixel": "Scan pixels",
    "slew_rate": "zControl slew rate",
}
LABELS.update({k: lbl for (k, _t, _c, lbl, _v) in PARAMS_BASE})

# (ptype, pcode) as in PARAMS_BASE -> readout key
_CODE_TO_KEY = {(t, str(c)): k for (k, t, c, _l, _v) in PARAMS_BASE}


def parse_time_s(text: Any) -> Optional[float]:
    """'1 ms' -> 0.001, '100 µs' -> 1e-4, '2 s' -> 2.0; None when not a time."""
    m = re.fullmatch(rf"\s*({_NUM})\s*([^\d\s]+)\s*", str(text))
    if not m:
        return None
    unit = m.group(2).replace("μ", "µ").lower()
    scale = _TIME_UNITS.get(unit)
    return float(m.group(1)) * scale if scale is not None else None


def parse_gain_v(text: Any) -> Optional[float]:
    """Output-gain radio caption '±0.1' -> 0.1 (also the input-gain caption: '10' / 'x10' -> 10)."""
    m = re.search(_NUM, str(text))
    return abs(float(m.group(0))) if m else None


def _as_float(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class SXMReadout:
    """One read of the SXM GUI. ``values[key]`` is a float/str/bool or None; ``errors[key]`` says why not."""

    def __init__(self, values: Dict[str, Any], errors: Dict[str, str], raw: Dict[str, Any]):
        self.values = values
        self.errors = errors
        self.raw = raw
        self.timestamp = datetime.datetime.now()

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def by_code(self, ptype: str, pcode) -> Optional[float]:
        """Value for a PARAMS_BASE (ptype, pcode), e.g. ('EDIT', 'Edit24') or ('DNC', 3)."""
        key = _CODE_TO_KEY.get((str(ptype).upper(), str(pcode)))
        return self.values.get(key) if key else None

    @property
    def ok(self) -> bool:
        """At least one value was read (SXM is running and its windows are reachable)."""
        return any(v is not None for v in self.values.values())

    def summary(self) -> str:
        n = sum(v is not None for v in self.values.values())
        return f"{n}/{len(self.values)} values read from SXM at {self.timestamp:%H:%M:%S}"


class SXMReader:
    """Reads the SXM GUI through AnfatecSXMBridge. Stateless apart from the bridge; safe to call any time."""

    def __init__(self, bridge=None):
        if bridge is None:
            from .AnfatecSXMBridge import AnfatecSXMBridge
            bridge = AnfatecSXMBridge(strict=False)
        self.bridge = bridge

    def read(self) -> SXMReadout:
        raw = self.bridge.snapshot()["parameters"]      # strict=False: failing sections carry '_error'
        values: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        def put(key, fn):
            try:
                v = fn()
            except Exception as exc:          # a missing section or field: record, never guess
                v, errors[key] = None, str(exc)
            else:
                if v is None:
                    errors.setdefault(key, "not readable")
            values[key] = v

        def section(name):
            data = raw.get(name, {})
            if "_error" in data:
                raise RuntimeError(data["_error"])
            return data

        def field(sec, name):
            v = section(sec).get(name)
            if v is None:
                raise RuntimeError(f"{sec}.{name} missing")
            return v

        for key, (sec, name) in _NAMED.items():
            put(key, lambda s=sec, n=name: _as_float(field(s, n)))

        dnc = "Dynamic Non-Contact R / Phi"

        def dnc_edits():
            edits = sorted(field(dnc, "unmapped_numeric_controls"), key=lambda c: c["rect"][1])
            if len(edits) != len(DNC_EDIT_KEYS):
                raise RuntimeError(f"DNC window has {len(edits)} numeric edits, expected {len(DNC_EDIT_KEYS)}; "
                                   "layout changed, refusing to guess")
            return [_as_float(c["value"]) for c in edits]

        try:
            positional = dnc_edits()
        except Exception as exc:
            for k in DNC_EDIT_KEYS:
                values[k], errors[k] = None, str(exc)
        else:
            for k, v in zip(DNC_EDIT_KEYS, positional):
                put(k, lambda v=v: v)

        put("afl_output_gain", lambda: parse_gain_v(field(dnc, "Range")))
        put("input_gain_ina", lambda: parse_gain_v(field(dnc, "Input Gain InA")))
        put("dnc_time_constant_s", lambda: parse_time_s(field(dnc, "TimeConstant")))
        put("dnc_rolloff", lambda: str(field(dnc, "RollOff")))
        put("q", lambda: _as_float(field(dnc, "q")))
        put("f_peak", lambda: _as_float(field(dnc, "f_peak")))
        put("ring_down_s", lambda: float(field(dnc, "tau_us")) * 1e-6)
        put("amp_tau_s", lambda: parse_time_s(field("Amplitude feedback", "Tau")))
        put("feedback_off", lambda: bool(field("zControl", "Feedback Off")))
        put("feedback_mode", lambda: str(field("Feedback mode", "Mode")))
        return SXMReadout(values, errors, raw)


def make_reader() -> Optional[SXMReader]:
    """An SXMReader, or None when the bridge cannot be loaded (e.g. not on Windows)."""
    try:
        return SXMReader()
    except Exception as exc:
        print(f"[SXM read-back] unavailable: {exc}")
        return None

