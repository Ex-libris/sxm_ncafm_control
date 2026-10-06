"""
Run sheets: ordered lists of conditions, each measured the way a loop is tuned by hand (set the parameters, let
the loop settle, step it, record a quiet window, move on).

Gains are never typed as raw pairs. Every condition is a point relative to a known-good **anchor** (Kp0, Ki0):

    Kp = G * Kp0 * c
    Ki = G * rho * Ki0 * c

* ``G`` (common scale) sets the speed of the loop. Along one ratio the behaviour is monotonic in G: slow and
  quiet at low G, then overshoot / Drive at zero / lost at high G.
* ``rho`` (ratio factor) sets the shape at a given speed: below 1 a slow integral tail, above 1 overshoot and
  ringing. ``rho = 1`` keeps the anchor's Ki:Kp.
* ``c`` (compensation, amplitude loop only) keeps the loop gain constant when a condition changes the AFL output
  gain or the input gain InA: gains scale as 1 / output gain (manual: x10 going from +-1 V to +-0.1 V) and as
  1 / input gain (InA x10 makes the measured amplitude, and so the loop gain, ten times larger).

A run sheet is usually built from **ramps**: G at a fixed rho, rho at a fixed G, or one secondary setting (AFL
Tau, DNC TimeConstant / RollOff, output gain, input gain, Ref) at fixed G and rho. The orders of magnitude between
Kp and Ki never change by accident, and regimes known to be unstable are not visited: a G or rho ramp stops at its
first condition that rings, loses the loop or keeps Drive at zero (:func:`stop_rule`).

Pure Python; no Qt, no hardware.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..sxm_state import parse_gain_v, parse_time_s

# ---------------------------------------------------------------------------
# what a condition can change besides the gains
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    unit: str
    sxm_key: str                 # sxm_state readout key
    writer_name: Optional[str]   # AnfatecSXMWriter short name (GUI-only control); None = written over DDE
    to_sxm: float = 1.0          # value in this table x to_sxm = value as the readout reports it
    text: bool = False           # compared as text (RollOff)


SETTINGS: Dict[str, Setting] = {s.key: s for s in (
    Setting("amp_tau_ms", "AFL Tau", "ms", "amp_tau_s", "amp.tau", 1e-3),
    Setting("dnc_tc_ms", "DNC TimeConstant", "ms", "dnc_time_constant_s", "dnc.tc", 1e-3),
    Setting("dnc_rolloff", "DNC RollOff", "dB/oct", "dnc_rolloff", "dnc.rolloff", text=True),
    Setting("output_gain_v", "AFL output gain", "±V", "afl_output_gain", "dnc.output_gain"),
    Setting("input_gain", "Input gain InA", "x", "input_gain_ina", "dnc.input_gain"),
    Setting("amp_ref", "Amplitude Ref", "", "amp_ref", None),
)}

# what can be ramped, in menu order: the two gain axes, then the settings
RAMPABLE = ("g", "rho") + tuple(SETTINGS)
RAMP_LABEL = {"g": "G (common scale)", "rho": "ρ (Ki:Kp ratio factor)",
              **{k: f"{s.label} [{s.unit}]" if s.unit else s.label for k, s in SETTINGS.items()}}

# suggested ramp values per loop (from the 5 Oct 2026 step tests, Q = 343 561)
DEFAULT_VALUES = {
    "afl": {"g": [0.1, 0.2, 0.3, 0.5, 1, 2, 3], "rho": [0.3, 1, 3], "amp_tau_ms": [5, 10, 20, 50, 100],
            "dnc_tc_ms": [0.5, 1, 2, 5, 10], "dnc_rolloff": ["6", "12", "24"], "output_gain_v": [0.1, 1],
            "input_gain": [1, 10], "amp_ref": [0.5, 1, 2, 4]},
    "pll": {"g": [0.25, 0.4, 0.6, 1, 1.5, 2], "rho": [0.5, 1, 2], "amp_tau_ms": [5, 10, 20, 50],
            "dnc_tc_ms": [0.5, 1, 2, 5, 10], "dnc_rolloff": ["6", "12", "24"], "output_gain_v": [0.1, 1],
            "input_gain": [1, 10], "amp_ref": [0.5, 1, 2, 4]},
}

_LEAD_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _lead_number(text: Any) -> Optional[float]:
    m = _LEAD_NUM.search(str(text))
    return float(m.group(0).replace(",", ".")) if m else None


def parse_value(key: str, text: str) -> Any:
    """A ramp value as typed: a number, or text for RollOff ('12', '12 dB/oct')."""
    s = SETTINGS.get(key)
    if s is not None and s.text:
        v = _lead_number(text)
        if v is None:
            raise ValueError(f"{s.label}: '{text}' has no number")
        return f"{v:g}"
    return float(text)


def parse_values(key: str, text: str) -> List[Any]:
    """Comma- or semicolon-separated ramp values (decimal point); space-separated when there is neither."""
    text = text.strip()
    sep = r"[;,]" if re.search(r"[;,]", text) else r"\s+"
    parts = [p.strip() for p in re.split(sep, text) if p.strip()]
    if not parts:
        raise ValueError("no values")
    return [parse_value(key, p) for p in parts]


def sxm_value(key: str, readout_value: Any) -> Any:
    """A readout value in this table's units (Tau 0.02 s -> 20 ms; RollOff '12 db/oct' -> '12')."""
    s = SETTINGS[key]
    if readout_value is None:
        return None
    if s.text:
        v = _lead_number(readout_value)
        return None if v is None else f"{v:g}"
    if isinstance(readout_value, (int, float)) and not isinstance(readout_value, bool):
        return float(readout_value) / s.to_sxm
    return None


def same_setting(key: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if SETTINGS[key].text:
        return _lead_number(a) == _lead_number(b)
    a, b = float(a), float(b)
    return abs(a - b) <= 0.02 * max(abs(a), abs(b), 1e-12)


def option_for(key: str, options: Sequence[str], value: Any) -> Optional[str]:
    """The option text of a GUI-only control (combo / radio caption) that means ``value``."""
    s = SETTINGS[key]
    for opt in options or ():
        if s.text:
            got = _lead_number(opt)
            want = _lead_number(value)
        elif key in ("amp_tau_ms", "dnc_tc_ms"):
            t = parse_time_s(opt)
            got, want = (None if t is None else t * 1e3), float(value)
        else:
            got, want = parse_gain_v(opt), float(value)
        if got is not None and want is not None and abs(got - want) <= 0.02 * max(abs(want), 1e-12):
            return opt
    return None


def format_setting(key: str, value: Any) -> str:
    s = SETTINGS[key]
    if s.text:
        return f"{value} dB/oct"
    if key == "output_gain_v":
        return f"±{value:g} V"
    if key == "input_gain":
        return f"x{value:g}"
    return f"{value:g}" + (f" {s.unit}" if s.unit else "")


# ---------------------------------------------------------------------------
# anchor and conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """The known-good pair every condition is relative to, and the gains of the DNC it was found at."""

    loop: str                     # 'afl' | 'pll'
    kp: float
    ki: float
    output_gain_v: float = 0.1
    input_gain: float = 1.0

    def __post_init__(self):
        if self.loop not in ("afl", "pll"):
            raise ValueError("loop must be 'afl' or 'pll'")
        if self.kp == 0 or self.ki == 0:
            raise ValueError("the anchor needs Kp and Ki both non-zero")


@dataclass
class Condition:
    g: float = 1.0
    rho: float = 1.0
    settings: Dict[str, Any] = field(default_factory=dict)   # only what this condition sets; the rest stays as is
    group: str = ""                                          # conditions of one ramp
    ramp: str = ""                                           # what the group ramps ('g', 'rho', a setting key)
    role: str = "test"                                       # 'test' | 'reference'

    def label(self) -> str:
        parts = [f"G {self.g:g}", f"ρ {self.rho:g}"]
        parts += [f"{SETTINGS[k].label} {format_setting(k, v)}" for k, v in self.settings.items()]
        return ", ".join(parts)


def compensation(anchor: Anchor, output_gain_v: Optional[float], input_gain: Optional[float]) -> float:
    """Gain factor that keeps the amplitude loop's gain when the output / input gain differs from the anchor's."""
    if anchor.loop != "afl":
        return 1.0
    c = 1.0
    if output_gain_v:
        c *= anchor.output_gain_v / float(output_gain_v)
    if input_gain:
        c *= anchor.input_gain / float(input_gain)
    return c


def gains(anchor: Anchor, cond: Condition, current: Optional[Dict[str, Any]] = None,
          compensate: bool = True) -> Tuple[float, float]:
    """(Kp, Ki) to write for ``cond``. ``current``: settings in effect (this module's keys) where cond sets none."""
    cur = dict(current or {})
    cur.update(cond.settings)
    c = compensation(anchor, cur.get("output_gain_v"), cur.get("input_gain")) if compensate else 1.0
    return anchor.kp * cond.g * c, anchor.ki * cond.g * cond.rho * c


def ramp(param: str, values: Sequence[Any], *, g: float = 1.0, rho: float = 1.0,
         settings: Optional[Dict[str, Any]] = None, group: str = "", reference_every: int = 0) -> List[Condition]:
    """
    One ramp as conditions. G and rho ramps run from the gentlest value up (the stop rule depends on it).
    ``reference_every`` > 0 inserts the anchor condition (G = rho = 1, no settings) at the start and after every
    that many conditions, to separate drift from parameter effects.
    """
    if param not in RAMPABLE:
        raise ValueError(f"cannot ramp '{param}'")
    vals = list(values)
    if not vals:
        raise ValueError("no values to ramp")
    if param in ("g", "rho"):
        vals = sorted(float(v) for v in vals)
        if any(v <= 0 for v in vals):
            raise ValueError(f"{param} must be positive")
    group = group or f"{RAMP_LABEL[param].split(' [')[0]}: {', '.join(str(v) for v in vals)}"
    base = dict(settings or {})
    out: List[Condition] = []

    def ref():
        out.append(Condition(group=group, ramp=param, role="reference"))

    if reference_every > 0:
        ref()
    for i, v in enumerate(vals):
        if param == "g":
            c = Condition(g=float(v), rho=rho, settings=dict(base), group=group, ramp=param)
        elif param == "rho":
            c = Condition(g=g, rho=float(v), settings=dict(base), group=group, ramp=param)
        else:
            c = Condition(g=g, rho=rho, settings={**base, param: v}, group=group, ramp=param)
        out.append(c)
        if reference_every > 0 and (i + 1) % reference_every == 0 and i + 1 < len(vals):
            ref()
    return out


# ---------------------------------------------------------------------------
# stop rule
# ---------------------------------------------------------------------------
STOP_STATUSES = ("ringing", "lost", "floor")


def stop_rule(sheet: Sequence[Condition], failed: int, status: str) -> List[int]:
    """
    Indices of the conditions to skip after ``sheet[failed]`` ended with ``status``: in the same G (rho) ramp,
    every later test condition at the same rho (G) with at least the failed G (rho). Other ramps are not monotonic
    in their parameter and are not cut.
    """
    if status not in STOP_STATUSES:
        return []
    f = sheet[failed]
    if f.role != "test" or f.ramp not in ("g", "rho"):
        return []
    out = []
    for i in range(failed + 1, len(sheet)):
        c = sheet[i]
        if c.group != f.group or c.role != "test" or c.settings != f.settings:
            continue
        if f.ramp == "g" and math.isclose(c.rho, f.rho) and c.g >= f.g * (1 - 1e-9):
            out.append(i)
        elif f.ramp == "rho" and math.isclose(c.g, f.g) and c.rho >= f.rho * (1 - 1e-9):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# saving a sheet
# ---------------------------------------------------------------------------


def sheet_to_dict(anchor: Anchor, sheet: Sequence[Condition], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"kind": "ncafm_runsheet", "version": 1, "anchor": asdict(anchor),
            "conditions": [asdict(c) for c in sheet], **(extra or {})}


def sheet_from_dict(d: Dict[str, Any]) -> Tuple[Anchor, List[Condition], Dict[str, Any]]:
    if d.get("kind") != "ncafm_runsheet":
        raise ValueError("not a run sheet (kind != 'ncafm_runsheet')")
    anchor = Anchor(**d["anchor"])
    sheet = [Condition(**c) for c in d.get("conditions", [])]
    for c in sheet:
        unknown = set(c.settings) - set(SETTINGS)
        if unknown:
            raise ValueError(f"unknown settings in the sheet: {', '.join(sorted(unknown))}")
    extra = {k: v for k, v in d.items() if k not in ("kind", "version", "anchor", "conditions")}
    return anchor, sheet, extra


def save_sheet(path: str, anchor: Anchor, sheet: Sequence[Condition], extra: Optional[Dict[str, Any]] = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sheet_to_dict(anchor, sheet, extra), f, indent=2, ensure_ascii=False)


def load_sheet(path: str) -> Tuple[Anchor, List[Condition], Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return sheet_from_dict(json.load(f))
