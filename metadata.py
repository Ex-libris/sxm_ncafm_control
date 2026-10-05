"""
Instrument metadata for exported files: what SXM was set to when the data was taken.

One :class:`Metadata` holds an SXM read-back (``sxm_state.SXMReadout``) plus any app-side context the
exporting tab adds (capture channels, sample rate, step-test plan, ...). It renders three ways:

* :meth:`Metadata.header_lines` - an aligned, grouped text block for the top of a CSV (written by
  :func:`csv_preamble` as ``# `` lines, led by a line saying how to read the file:
  ``pandas.read_csv(path, comment='#')``, or ``numpy.loadtxt`` with the ``skiprows`` it states);
* :meth:`Metadata.to_dict` - the same groups as JSON (the ``.json`` sidecar written next to every export);
* :meth:`Metadata.filename_tag` - the key settings packed into a file name, e.g.
  ``AFL-Ref0.55-Kp1e7-Ki1200-Tau50ms_PLL-Kp-100-Ki-1e4_InA1_OG0.1V_TC2ms``.

Values are in SXM's GUI units, as read from its windows (see sxm_state.py). The bridge does not report
units for Ref, Drive or the topography Ref, so those are labelled "SXM units" rather than guessed.

Pure Python (no Qt): safe to import in tests and analysis scripts.
"""
from __future__ import annotations

import datetime
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# (readout key, label, unit, factor applied for display) per group. Display unit != SXM unit only for times
# (SXM read-back gives seconds; ms reads better for TimeConstant / Tau / ring-down).
SXM_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, str, str, float], ...]], ...] = (
    ("Amplitude feedback (AFL)", (
        ("amp_ref", "Ref (amplitude setpoint)", "SXM units", 1.0),
        ("amp_kp", "Kp", "", 1.0),
        ("amp_ki", "Ki", "", 1.0),
        ("amp_tau_s", "Tau (AFL input filter)", "ms", 1e3),
        ("amp_pull_back", "Pull back at", "%", 1.0),
        ("amp_pull_speed", "Pull speed", "nm/s", 1.0),
    )),
    ("PLL", (
        ("pll_kp", "Kp", "", 1.0),
        ("pll_ki", "Ki", "", 1.0),
    )),
    ("DNC (lock-in and excitation)", (
        ("used_freq", "use (PLL centre / excitation)", "Hz", 1.0),
        ("drive", "Drive", "SXM units", 1.0),
        ("input_gain_ina", "Input gain InA", "x", 1.0),
        ("afl_output_gain", "Output gain", "+-V", 1.0),
        ("dnc_time_constant_s", "TimeConstant t", "ms", 1e3),
        ("dnc_rolloff", "RollOff", "", 1.0),
        ("dnc_sweep_start", "Sweep from", "Hz", 1.0),
        ("dnc_sweep_stop", "Sweep to", "Hz", 1.0),
    )),
    ("Resonance (last DNC sweep)", (
        ("f_peak", "f peak", "Hz", 1.0),
        ("q", "Q", "", 1.0),
        ("ring_down_s", "Ring-down tau", "ms", 1e3),
    )),
    ("Topography / z feedback", (
        ("feedback_mode", "Mode", "", 1.0),
        ("feedback_off", "Feedback off", "", 1.0),
        ("topo_ref", "Ref", "SXM units", 1.0),
        ("topo_kp", "Kp", "", 1.0),
        ("topo_ki", "Ki", "", 1.0),
        ("slew_rate", "Slew rate", "", 1.0),
    )),
    ("Scan", (
        ("scan_range", "Range", "", 1.0),
        ("scan_speed", "Speed", "", 1.0),
        ("scan_pixel", "Pixels", "", 1.0),
    )),
)

_UNSAFE = re.compile(r'[<>:"/\\|?*\s]+')     # not allowed (or awkward) in Windows file names


def fmt_value(v: Any, sig: int = 8) -> str:
    """Human-readable value: up to ``sig`` significant digits (f = 25562.49 keeps its mHz), scientific from 1e7
    (raw gains: 1e7, 5e8) and below 1e-3."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isnan(x):
            return "n/a"
        if x != 0 and (abs(x) >= 1e7 or abs(x) < 1e-3):
            m, e = f"{x:.{sig - 1}e}".split("e")
            m = m.rstrip("0").rstrip(".")
            return f"{m}e{int(e)}"
        return f"{x:.{sig}g}"
    return str(v)


def short_value(v: Any) -> str:
    """Compact number for a file name: up to 4 significant digits ('1200', '0.55'), '1e7' / '2.5e8' from 1e4 on."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(float(v)):
        return str(v)
    x = float(v)
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        m, e = f"{x:.2e}".split("e")
        m = m.rstrip("0").rstrip(".")
        return f"{m}e{int(e)}"
    return f"{x:.4g}"


def safe_filename(text: str) -> str:
    """Make ``text`` usable as (part of) a Windows file name."""
    return _UNSAFE.sub("_", text).strip("._")


class Metadata:
    """
    The instrument state for one export: SXM read-back values (``values``, keyed like ``sxm_state``) plus
    app-side ``sections`` added by the exporting tab, each ``(title, [(label, value, unit), ...])``.
    """

    def __init__(self, values: Optional[Dict[str, Any]] = None, errors: Optional[Dict[str, str]] = None,
                 timestamp: Optional[datetime.datetime] = None, source: str = "SXM read-back"):
        self.values: Dict[str, Any] = dict(values or {})
        self.errors: Dict[str, str] = dict(errors or {})
        self.timestamp = timestamp or datetime.datetime.now()
        self.source = source
        self.sections: List[Tuple[str, List[Tuple[str, Any, str]]]] = []

    # -- building ---------------------------------------------------------------------------------
    @classmethod
    def from_readout(cls, readout) -> "Metadata":
        return cls(readout.values, readout.errors, readout.timestamp, "SXM read-back")

    @classmethod
    def unavailable(cls, why: str) -> "Metadata":
        return cls(source=f"not available ({why})")

    def add_section(self, title: str, rows: Iterable[Tuple[str, Any, str]]) -> "Metadata":
        self.sections.append((title, list(rows)))
        return self

    def get(self, key: str) -> Any:
        return self.values.get(key)

    @property
    def has_sxm(self) -> bool:
        return any(v is not None for v in self.values.values())

    # -- views ------------------------------------------------------------------------------------
    def groups(self) -> List[Tuple[str, List[Tuple[str, Any, str]]]]:
        """Every group to show, SXM groups first, as ``(title, [(label, display value, unit)])``."""
        out = []
        if self.has_sxm:
            for title, rows in SXM_GROUPS:
                shown = []
                for key, label, unit, factor in rows:
                    v = self.values.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and factor != 1.0:
                        v = v * factor
                    shown.append((label, v, unit))
                out.append((title, shown))
        out.extend(self.sections)
        return out

    def header_lines(self, title: str = "") -> List[str]:
        """Aligned plain-text block (no comment prefix): title, timestamp, source, then each group."""
        lines = []
        if title:
            lines += [title, "=" * len(title)]
        lines.append(f"Saved:  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        lines.append(f"SXM settings: {self.source}" +
                     (f", read at {self.timestamp:%Y-%m-%d %H:%M:%S}" if self.has_sxm else ""))
        if self.has_sxm:
            lines.append("Values as shown in SXM (GUI units); 'SXM units' = unit not reported by SXM.")
        groups = self.groups()
        width = max((len(lbl) for _t, rows in groups for lbl, _v, _u in rows), default=10)
        for gtitle, rows in groups:
            lines.append("")
            lines.append(f"[{gtitle}]")
            for label, v, unit in rows:
                u = "" if unit in ("", None) or v is None else f" {unit}"
                lines.append(f"  {label:<{width}}  {fmt_value(v)}{u}")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready: the grouped values (value + unit), the raw read-back keys, and what could not be read."""
        groups = {}
        for gtitle, rows in self.groups():
            groups[gtitle] = {label: {"value": _jsonable(v), "unit": unit} for label, v, unit in rows}
        return {
            "kind": "ncafm_metadata",
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "sxm_source": self.source,
            "sxm_read_at": self.timestamp.isoformat(timespec="seconds") if self.has_sxm else None,
            "units_note": "Values as shown in SXM (GUI units). 'SXM units' = the unit is not reported by SXM.",
            "groups": groups,
            "sxm_values": {k: _jsonable(v) for k, v in self.values.items()},
            "sxm_not_read": dict(self.errors),
        }

    def caption_lines(self) -> List[str]:
        """A few dense lines of the key settings, for under a plot image."""
        v = self.values

        def item(key, label, unit="", factor=1.0):
            x = v.get(key)
            if x is None:
                return None
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                x = x * factor
            return f"{label} {fmt_value(x, 5)}{(' ' + unit) if unit else ''}"

        def line(head, items):
            items = [i for i in items if i]
            return f"{head}: " + "  ·  ".join(items) if items else None

        out = [
            line("AFL", [item("amp_ref", "Ref"), item("amp_kp", "Kp"), item("amp_ki", "Ki"),
                         item("amp_tau_s", "Tau", "ms", 1e3)]),
            line("PLL", [item("pll_kp", "Kp"), item("pll_ki", "Ki")]),
            line("DNC", [item("used_freq", "use", "Hz"), item("drive", "Drive"), item("input_gain_ina", "InA x"),
                         item("afl_output_gain", "Output gain +-", "V"), item("dnc_time_constant_s", "TC", "ms", 1e3),
                         item("q", "Q")]),
        ]
        out = [o.replace("InA x ", "InA x") for o in out if o]
        for title, rows in self.sections:
            if title in ("Capture",):
                out.append(line(title, [f"{lbl} {fmt_value(val, 5)}{(' ' + u) if u else ''}" for lbl, val, u in rows]))
        stamp = f"SXM settings read {self.timestamp:%Y-%m-%d %H:%M:%S}" if self.has_sxm else f"SXM settings: {self.source}"
        out.append(stamp)
        return out

    # columns of the PNG legend: the loops work together, so each is shown in full, side by side
    LEGEND_SXM = (
        ("Amplitude (AFL)", (("amp_ref", "Ref", "", 1.0), ("amp_kp", "Kp", "", 1.0), ("amp_ki", "Ki", "", 1.0),
                             ("amp_tau_s", "Tau", "ms", 1e3), ("afl_output_gain", "Out gain", "+-V", 1.0))),
        ("PLL", (("pll_kp", "Kp", "", 1.0), ("pll_ki", "Ki", "", 1.0),
                 ("dnc_time_constant_s", "TC", "ms", 1e3), ("dnc_rolloff", "RollOff", "", 1.0))),
        ("DNC", (("used_freq", "use", "Hz", 1.0), ("drive", "Drive", "", 1.0),
                 ("input_gain_ina", "InA", "x", 1.0))),
        ("Resonance", (("f_peak", "f peak", "Hz", 1.0), ("q", "Q", "", 1.0), ("ring_down_s", "Ring-down", "ms", 1e3))),
    )
    LEGEND_SECTIONS = ("Capture", "Step Test")      # app sections shown in the legend (not the event list)

    def legend(self) -> Tuple[List[Tuple[str, List[Tuple[str, str]]]], str]:
        """
        The PNG legend: ``(columns, footer)``, columns as ``(title, [(label, value text)])``. Every loop column
        is listed in full (``n/a`` = not read), so a missing value is visible rather than silently dropped.
        """
        def text(x, unit="", sig=5):
            if x is None:
                return "n/a"
            v = fmt_value(x, sig)
            if unit == "x":
                return f"x{v}"
            if unit == "+-V":
                return f"+-{v} V"
            return f"{v} {unit}" if unit else v

        cols = []
        if self.has_sxm:
            for title, rows in self.LEGEND_SXM:
                items = []
                for key, label, unit, factor in rows:
                    x = self.values.get(key)
                    if isinstance(x, (int, float)) and not isinstance(x, bool):
                        x = x * factor
                    items.append((label, text(x, unit, 8 if unit == "Hz" or key == "q" else 5)))  # keep mHz, all of Q
                cols.append((title, items))
        for title, rows in self.sections:
            if title in self.LEGEND_SECTIONS:
                cols.append((title, [(lbl, text(v, u)) for lbl, v, u in rows]))
        footer = (f"SXM settings read {self.timestamp:%Y-%m-%d %H:%M:%S}, in SXM GUI units" if self.has_sxm
                  else f"SXM settings: {self.source}")
        return cols, footer

    def filename_tag(self, loops: Sequence[str] = ("afl", "pll")) -> str:
        """
        The key settings as a file-name fragment, in the order the loops are given. Values that could not
        be read are left out. Example: ``AFL-Ref0.55-Kp1e7-Ki1200-Tau50ms_PLL-Kp-100-Ki-1e4_InA1_OG0.1V_TC2ms``.
        """
        v = self.values
        parts = []

        def num(key, prefix, unit="", factor=1.0):
            x = v.get(key)
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return f"{prefix}{short_value(x * factor)}{unit}"
            return None

        for loop in loops:
            if loop == "afl":
                bits = [num("amp_ref", "Ref"), num("amp_kp", "Kp"), num("amp_ki", "Ki"), num("amp_tau_s", "Tau", "ms", 1e3)]
                bits = [b for b in bits if b]
                if bits:
                    parts.append("AFL-" + "-".join(bits))
            elif loop == "pll":
                bits = [b for b in (num("pll_kp", "Kp"), num("pll_ki", "Ki")) if b]
                if bits:
                    parts.append("PLL-" + "-".join(bits))
        for b in (num("input_gain_ina", "InA"), num("afl_output_gain", "OG", "V"),
                  num("dnc_time_constant_s", "TC", "ms", 1e3)):
            if b:
                parts.append(b)
        return safe_filename("_".join(parts))

    def filename(self, kind: str, detail: str = "", loops: Sequence[str] = ("afl", "pll")) -> str:
        """``YYYYmmdd-HHMMSS_<kind>[_<detail>][_<tag>]`` (no extension)."""
        stem = [f"{self.timestamp:%Y%m%d-%H%M%S}", kind]
        if detail:
            stem.append(detail)
        tag = self.filename_tag(loops)
        if tag:
            stem.append(tag)
        return safe_filename("_".join(stem))


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def collect(reader, sections: Iterable[Tuple[str, Iterable[Tuple[str, Any, str]]]] = ()) -> Metadata:
    """Read SXM now (``reader`` = sxm_state.SXMReader or None) and attach ``sections``. Never raises."""
    if reader is None:
        meta = Metadata.unavailable("no SXM read-back in this session")
    else:
        try:
            readout = reader.read()
        except Exception as exc:          # SXM closed, a window missing, ...: export without it, say why
            meta = Metadata.unavailable(f"read failed: {exc}")
        else:
            meta = Metadata.from_readout(readout) if readout.ok else Metadata.unavailable("SXM not readable")
    for title, rows in sections:
        meta.add_section(title, rows)
    return meta


def comment_block(lines: Sequence[str], prefix: str = "# ") -> str:
    """``lines`` as comment lines, ready to write before a CSV's column header."""
    return "".join(f"{prefix}{ln}".rstrip() + "\n" for ln in lines)


def csv_preamble(lines: Sequence[str], numeric: bool = True) -> str:
    """
    ``lines`` as a '#' block for the top of a CSV whose next line is the column header, led by a line saying
    how to read the file (numpy's readers cannot skip a comment block *and* take names from the line after it).
    ``numeric=False`` (text columns) leaves numpy out of the hint.
    """
    skip = len(lines) + 2           # this hint, the block, the column header
    hint = "Read with pandas.read_csv(path, comment='#')"
    if numeric:
        hint += f" or numpy.loadtxt(path, delimiter=',', skiprows={skip})"
    return comment_block([hint] + list(lines))


def write_sidecar(path: str, meta: Metadata, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write ``meta`` (plus ``extra`` top-level entries) as JSON to ``path``. Returns the path."""
    payload = meta.to_dict()
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
