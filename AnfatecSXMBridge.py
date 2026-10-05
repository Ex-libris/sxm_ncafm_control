"""AnfatecSXMBridge - read-only semantic Python API for a running Anfatec SXM (Femto_28_4.exe).

Self-contained: reads the Delphi/Win32 GUI through window enumeration and
read-only query messages (WM_GETTEXT, BM_GETCHECK, CB_GETCURSEL, ...). No
clicks, keystrokes, WM_SETTEXT, DDE, IOCTL, driver access, injection or
process-memory access.

    from AnfatecSXMBridge import AnfatecSXMBridge
    sxm = AnfatecSXMBridge()
    sxm.scan.range                      # live read of the Scan section
    sxm.amplitude.ki
    sxm.dynamic.q                       # parsed from the DNC status bar
    sxm.get("Amplitude feedback.Ki")    # == sxm["Amplitude feedback.Ki"]
    sxm.search("Ki"); sxm.paths(); sxm.snapshot(); sxm.save_snapshot()
    sxm.discovery.forms(); sxm.discovery.form("TdncForm"); sxm.raw("TdncForm")

Every section access is a fresh GUI read. To read several fields consistently
(and cheaply), hold on to the section: ``s = sxm.dynamic; s.q, s.f_peak``.

Controls are identified structurally (Form -> GroupBox -> control, ordered
top-to-bottom), never by HWND or current value. With ``strict=True``
(default) a missing form, a changed layout, an empty required field, or an
unparseable status raises SXMBridgeError instead of returning a guess.

Run ``python AnfatecSXMBridge.py`` to print and save a snapshot, or
``python AnfatecSXMBridgeMonitor.py`` for a live view.

``sxm.control(path)`` locates the control behind a path without touching it;
AnfatecSXMWriter uses it to write the same paths this module reads.

Master copy: anfatec_code/AnfatecSXMBridge.py in the author's development
folder. sxm_ncafm_control ships a copy, updated from the master; make changes
in the master, not in the copy.
"""
from __future__ import annotations

import copy
import ctypes
import ctypes.wintypes as wt
import datetime
import json
import os
import re
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

__all__ = ['AnfatecSXMBridge', 'Win32Backend', 'Control', 'Section', 'Discovery',
           'SXMBridgeError', 'SXMPathError', 'parse_dnc_status']

TARGET_EXE = 'femto_28_4.exe'
BACKEND = 'Win32 GUI (passive)'


class SXMBridgeError(RuntimeError):
    """The GUI could not be read unambiguously."""


class SXMPathError(SXMBridgeError, KeyError):
    """A path or field name does not resolve to exactly one value."""
    __str__ = RuntimeError.__str__


# =============================================================================
# Win32 primitives (read-only)
# =============================================================================

WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
BM_GETCHECK = 0x00F0
CB_GETCURSEL = 0x0147
TBM_GETPOS = 0x0400
PBM_GETPOS = 0x0408
SMTO_ABORTIFHUNG = 0x0002
QUERY_TIMEOUT_MS = 250
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_user32 = ctypes.WinDLL('user32', use_last_error=True)
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_LONG_PTR = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
_ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
_ENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

_user32.EnumWindows.argtypes = [_ENUMPROC, wt.LPARAM]
_user32.EnumChildWindows.argtypes = [wt.HWND, _ENUMPROC, wt.LPARAM]
_user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
_user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
_user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
_user32.GetParent.argtypes = [wt.HWND]
_user32.GetParent.restype = wt.HWND
_user32.SendMessageTimeoutW.argtypes = [wt.HWND, wt.UINT, _ULONG_PTR, _LONG_PTR, wt.UINT, wt.UINT,
                                        ctypes.POINTER(_ULONG_PTR)]
_user32.SendMessageTimeoutW.restype = _LONG_PTR
_kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_kernel32.OpenProcess.restype = wt.HANDLE
_kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
_kernel32.CloseHandle.argtypes = [wt.HANDLE]


def _pid(hwnd: int) -> int:
    pid = wt.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _process_name(pid: int) -> str:
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ''
    try:
        size = wt.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        ok = _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return os.path.basename(buf.value) if ok else ''
    finally:
        _kernel32.CloseHandle(handle)


def _cls(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, buf, len(buf))
    return buf.value


def _msg(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0) -> int:
    result = _ULONG_PTR()
    ok = _user32.SendMessageTimeoutW(hwnd, msg, wparam, lparam, SMTO_ABORTIFHUNG, QUERY_TIMEOUT_MS,
                                     ctypes.byref(result))
    if not ok:
        raise RuntimeError(f'Win32 query timed out/failed: HWND=0x{int(hwnd):X}, msg=0x{msg:X}')
    return int(result.value)


def _text(hwnd: int) -> str:
    cap = max(_msg(hwnd, WM_GETTEXTLENGTH) + 1, 256)
    buf = ctypes.create_unicode_buffer(cap)
    _msg(hwnd, WM_GETTEXT, cap, ctypes.addressof(buf))
    return buf.value.strip()


def _rect(hwnd: int) -> tuple:
    r = wt.RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError('GetWindowRect failed')
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def _top(hwnd: int) -> int:
    return _rect(hwnd)[1]


def _parent(hwnd: int) -> int:
    return int(_user32.GetParent(hwnd) or 0)


def _children(hwnd: int, direct: bool = False) -> list:
    found = []

    @_ENUMPROC
    def callback(child, _):
        if not direct or _parent(child) == int(hwnd):
            found.append(int(child))
        return True

    _user32.EnumChildWindows(hwnd, callback, 0)
    return found


def _top_windows() -> list:
    """Top-level windows owned by Femto_28_4.exe."""
    names, found = {}, []

    @_ENUMPROC
    def callback(hwnd, _):
        pid = _pid(hwnd)
        if pid:
            if pid not in names:
                names[pid] = _process_name(pid).lower()
            if names[pid] == TARGET_EXE:
                found.append(int(hwnd))
        return True

    _user32.EnumWindows(callback, 0)
    return found


def _form(form_class: str) -> int:
    forms = [h for h in _top_windows() if _cls(h) == form_class]
    if len(forms) != 1:
        raise RuntimeError(f'Expected one {form_class}, found {len(forms)}')
    return forms[0]


def _norm(s: Any) -> str:
    return ' '.join(str(s).strip().casefold().split())


def _group(parent: int, caption: str) -> int:
    groups = [h for h in _children(parent) if _cls(h) == 'TGroupBox' and _norm(_text(h)) == _norm(caption)]
    if len(groups) != 1:
        raise RuntimeError(f'Expected one group {caption!r}, found {len(groups)}')
    return groups[0]


def _direct(parent: int, control_class: str) -> list:
    """Direct children of one class, ordered top-to-bottom."""
    return sorted((h for h in _children(parent, True) if _cls(h) == control_class), key=_top)


def _num(s: Any) -> Any:
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _checked(hwnd: int) -> bool:
    return _msg(hwnd, BM_GETCHECK) != 0


def _control_value(hwnd: int) -> Any:
    c, t = _cls(hwnd), _text(hwnd)
    if c in ('TCheckBox', 'TRadioButton', 'TGroupButton'):
        return {'caption': t, 'checked': _checked(hwnd)}
    if c == 'TComboBox':
        try:
            index = _msg(hwnd, CB_GETCURSEL)
        except RuntimeError:
            index = None
        return {'text': t, 'selected_index': index}
    if c in ('TTrackBar', 'TProgressBar'):
        try:
            return _msg(hwnd, TBM_GETPOS if c == 'TTrackBar' else PBM_GETPOS)
        except RuntimeError:
            return None
    return _num(t) if c in ('TEdit', 'Edit') else t


_SKIP_FORMS = ('TPUtilWindow', 'DDEMLEvent', 'DDEMLMom', 'TApplication')


def _inventory(form: int) -> dict:
    """Semantic-neutral dump of one form. HWNDs are not used as identities."""
    rows = []
    for h in _children(form):
        c = _cls(h)
        if c == 'TPUtilWindow':
            continue
        try:
            value = _control_value(h)
        except Exception as exc:
            value = {'_error': str(exc)}
        p = _parent(h)
        rows.append({'class': c, 'value': value, 'rect': _rect(h), 'parent_class': _cls(p) if p else None})
    return {'form_class': _cls(form), 'caption': _text(form), 'rect': _rect(form), 'controls': rows}


# =============================================================================
# Backend: GUI structure -> named parameters
# =============================================================================

class Control(NamedTuple):
    """The GUI control behind one parameter, located structurally at one moment.

    ``kind`` is ``'edit'``, ``'combo'``, ``'check'`` or ``'choice'`` (a set of radio
    buttons whose value is the caption of the checked one). ``hwnds`` holds the one
    control, or the buttons of a choice. HWNDs are not identities: locate again for
    every access.
    """
    kind: str
    hwnds: tuple
    numeric: bool = False          # combo whose text is read as a number


def _edit(h: int) -> Control:
    return Control('edit', (h,))


def _combo(h: int, numeric: bool = False) -> Control:
    return Control('combo', (h,), numeric)


def _check(h: int) -> Control:
    return Control('check', (h,))


def _choice(hwnds: list) -> Control:
    return Control('choice', tuple(hwnds))


def _value(c: Control) -> Any:
    h = c.hwnds[0] if c.hwnds else 0
    if c.kind == 'edit':
        return _num(_text(h))
    if c.kind == 'combo':
        return _num(_text(h)) if c.numeric else _text(h)
    if c.kind == 'check':
        return _checked(h)
    return next((_text(b) for b in c.hwnds if _checked(b)), None)


def _resolve(tree: Any) -> Any:
    """Read a control tree into the same structure of values."""
    if isinstance(tree, Control):
        return _value(tree)
    if isinstance(tree, dict):
        return {k: _resolve(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_resolve(v) for v in tree]
    return tree


class Win32Backend:
    """Passive GUI reader. Each method raises RuntimeError when the layout is not as expected.

    Every section is located once, by ``_locate_<section>``, as a tree of Controls;
    the section read resolves that tree. ``controls(section)`` exposes the same tree,
    so anything that acts on a path acts on exactly the control that is read for it.
    """

    def controls(self, method: str) -> dict:
        """Control tree behind the read method ``method`` (same shape as its values)."""
        locate = getattr(self, f'_locate_{method}', None)
        if locate is None:
            raise RuntimeError(f'{method!r} has no control map')
        return locate()

    def scan(self) -> dict:
        return _resolve(self._locate_scan())

    def topography(self) -> dict:
        return _resolve(self._locate_topography())

    def amplitude(self) -> dict:
        return _resolve(self._locate_amplitude())

    def pll(self) -> dict:
        return _resolve(self._locate_pll())

    def zcontrol(self) -> dict:
        return _resolve(self._locate_zcontrol())

    def spectroscopy(self) -> dict:
        return _resolve(self._locate_spectroscopy())

    def lockin(self) -> dict:
        return _resolve(self._locate_lockin())

    def dynamic_non_contact(self) -> dict:
        return _resolve(self._locate_dynamic_non_contact())

    def feedback_mode(self) -> dict:
        return _resolve(self._locate_feedback_mode())

    def tip_conditioning(self) -> dict:
        return _resolve(self._locate_tip_conditioning())

    def oscilloscope_state(self) -> dict:
        return _resolve(self._locate_oscilloscope_state())

    # -- forms located directly by structure --

    def _locate_scan(self) -> dict:
        f = _form('TScanParaForm')
        left = _rect(f)[0]

        def column(control_class):
            # Scan column = direct form children in the left 120 px (verified from screenshot + live dump).
            return sorted((h for h in _children(f, True) if _cls(h) == control_class and _rect(h)[0] < left + 120),
                          key=_top)

        edits, combos = column('TEdit'), column('TComboBox')
        if len(edits) < 5:
            raise RuntimeError('Scan layout changed')
        rng, speed, xc, yc, angle = (_edit(h) for h in edits[:5])
        return {'Range': rng, 'Speed': speed, 'Pixel': _combo(combos[0], numeric=True) if combos else None,
                'x-Center': xc, 'y-Center': yc, 'Angle': angle}

    def _locate_topography(self) -> dict:
        edits = _direct(_group(_form('TScanParaForm'), 'Topography feedback'), 'TEdit')
        if len(edits) != 3:
            raise RuntimeError('Topography feedback layout changed')
        return dict(zip(('Ref', 'Ki', 'Kp'), map(_edit, edits)))

    def _locate_amplitude(self) -> dict:
        g = _group(_form('TScanParaForm'), 'Amplitude feedback')
        edits, combos = _direct(g, 'TEdit'), _direct(g, 'TComboBox')
        if len(edits) != 5 or len(combos) != 1:
            raise RuntimeError('Amplitude feedback layout changed')
        out = dict(zip(('Ref', 'Ki', 'Kp', 'Pull back at', 'Pull Speed'), map(_edit, edits)))
        out['Tau'] = _combo(combos[0])
        return out

    def _locate_pll(self) -> dict:
        edits = _direct(_group(_form('TScanParaForm'), 'PLL'), 'TEdit')
        if len(edits) != 2:
            raise RuntimeError('PLL layout changed')
        return dict(zip(('Kp', 'Ki'), map(_edit, edits)))

    def _locate_zcontrol(self) -> dict:
        f = _form('TzControlForm')
        edits, boxes = _direct(f, 'TEdit'), _direct(f, 'TCheckBox')
        if len(edits) != 3 or len(boxes) != 1:
            raise RuntimeError('zControl layout changed')
        dz, per_tick, slew = map(_edit, edits)
        return {'Feedback Off': _check(boxes[0]), 'dz': dz, 'dz per Mouse Tick': per_tick, 'Slew Rate': slew}

    def _locate_spectroscopy(self) -> dict:
        f = _form('TSpektForm')
        panels = [h for h in _children(f, True) if _cls(h) == 'TPanel' and _rect(h)[2] < 250]
        if not panels:
            raise RuntimeError('Spectroscopy parameter panel not found')
        p = min(panels, key=lambda h: _rect(h)[0])
        edits, combos = _direct(p, 'TEdit'), _direct(p, 'TComboBox')
        names = ('X', 'Y', 'Delay1', 'AguT', 'dz', 'U Start', 'U Stop')
        out = {n: _edit(h) for n, h in zip(names, edits)}
        out['Mode'] = _combo(combos[0]) if combos else None
        acquire = [h for h in _children(p) if _cls(h) == 'TGroupBox' and _norm(_text(h)) == 'acquire']
        if acquire:
            out['Acquire'] = [_combo(h) for h in _direct(acquire[0], 'TComboBox')]
        return out

    def _locate_lockin(self) -> dict:
        f = _form('TMultiLockInForm')
        out = {}
        for caption, key in (('TimeConstant t', 'TimeConstant'), ('RollOff', 'RollOff')):
            combos = _direct(_group(f, caption), 'TComboBox')
            out[key] = _combo(combos[0]) if combos else None
        for n in (1, 2, 3):
            g = _group(f, f'Lia {n}')
            edits, combos = _direct(g, 'TEdit'), _direct(g, 'TComboBox')
            d = {'Link': _combo(combos[0]) if combos else None}
            if edits:
                d['Value1'] = _edit(edits[0])
            # The phase edit sits in a nested 'Phase' group.
            phases = [h for h in _children(g) if _cls(h) == 'TGroupBox' and _norm(_text(h)) == 'phase']
            if phases:
                pe = _direct(phases[0], 'TEdit')
                d['Phase'] = _edit(pe[0]) if pe else None
            out[f'Lia{n}'] = d
        return out

    # -- forms located over all their controls, in enumeration order --

    @staticmethod
    def _all(form_class: str):
        f = _form(form_class)
        kids = [h for h in _children(f) if _cls(h) != 'TPUtilWindow']
        return f, lambda control_class: [h for h in kids if _cls(h) == control_class]

    def _locate_dynamic_non_contact(self) -> dict:
        _, of = self._all('TdncForm')
        groups = {_text(h): h for h in of('TGroupBox')}

        def combo_in(caption):
            if caption not in groups:
                raise RuntimeError(f'DNC group {caption!r} not found')
            x, y, w, h = _rect(groups[caption])
            hits = [c for c in of('TComboBox') if x <= _rect(c)[0] <= x + w and y <= _rect(c)[1] <= y + h]
            if len(hits) != 1:
                raise RuntimeError(f'Ambiguous DNC group {caption!r}')
            return _combo(hits[0])

        edits = sorted(of('TEdit'), key=_top)
        status = of('TStatusBar')
        return {
            'Input Gain InA': _choice(of('TRadioButton')),
            'TimeConstant': combo_in('TimeConstant t'),
            'RollOff': combo_in('RollOff'),
            'Range': _choice(of('TGroupButton')),
            'Status': _text(status[0]) if status else '',
            'unmapped_numeric_controls': [{'value': _edit(h), 'rect': _rect(h)} for h in edits],
        }

    def _locate_feedback_mode(self) -> dict:
        _, of = self._all('TfeedbackForm')
        combos = of('TComboBox')
        return {'Mode': _combo(combos[0]) if combos else None}

    def _locate_tip_conditioning(self) -> dict:
        _, of = self._all('TTipForm')
        edits = sorted(of('TEdit'), key=_top)
        return {'Mode': _choice(of('TGroupButton')),
                'unmapped_numeric_controls': [{'value': _edit(h), 'rect': _rect(h)} for h in edits]}

    def _locate_oscilloscope_state(self) -> dict:
        _, of = self._all('TOszi2Form')
        return {'Channels': [_combo(h) for h in of('TComboBox')],
                'x_axis_values': [_edit(h) for h in of('TEdit')],
                'selected_modes': [_text(h) for h in of('TRadioButton') if _checked(h)]}

    # -- forms read from their control inventory (read-only) --

    @staticmethod
    def _of(form: dict, control_class: str) -> list:
        return [c for c in form['controls'] if c['class'] == control_class]

    @staticmethod
    def _state(control: dict) -> dict:
        v = control['value']
        return v if isinstance(v, dict) else {'caption': str(v), 'checked': None}

    def scanner_state(self) -> dict:
        f = self.discovery_form('TScannerForm')
        buttons = [self._state(c) for c in self._of(f, 'TGroupButton')]
        radios = [self._state(c) for c in self._of(f, 'TRadioButton')]
        return {'caption': f['caption'],
                'selected_group_buttons': [s['caption'] for s in buttons if s.get('checked')],
                'selected_radio_buttons': [s['caption'] for s in radios if s.get('checked')],
                'combos': [c['value'] for c in self._of(f, 'TComboBox')]}

    # -- discovery --

    def discover_forms(self) -> list:
        return [_inventory(h) for h in _top_windows() if _cls(h) not in _SKIP_FORMS]

    def discovery_form(self, form_class: str) -> dict:
        return _inventory(_form(form_class))

    def raw_form(self, form_class: str) -> list:
        f = _form(form_class)
        return [{'class': _cls(h), 'text': _text(h), 'rect': _rect(h), 'parent': _parent(h), 'hwnd': h}
                for h in [f] + _children(f)]


# =============================================================================
# Semantic layer
# =============================================================================

def _snake(s: Any) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(s).casefold()).strip('_')


def _match_key(mapping: Mapping, key: str, where: str) -> str:
    if key in mapping:
        return key
    hits = [k for k in mapping if _snake(k) == _snake(key)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise SXMPathError(f'{where!r}: {key!r} is ambiguous between {hits}')
    raise SXMPathError(f'{where!r} has no field {key!r}; available: {list(mapping)}')


_NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_DNC_STATUS_FIELDS = (
    ('q', re.compile(rf'Q:\s*({_NUM})')),
    ('f_peak', re.compile(rf'fPeak:\s*({_NUM})\s*Hz')),
    ('tau_us', re.compile(rf'tau:\s*({_NUM})\s*[\u00b5\u03bcu]s')),  # micro sign, Greek mu, or ASCII u
)


def parse_dnc_status(status: str) -> dict:
    """Parse ``'Q: 148699 | fPeak: 25562 Hz | tau: 1851628 µs'`` into q, f_peak [Hz], tau_us [µs]."""
    parts = [p.strip() for p in str(status).split('|')]
    out = {}
    for key, rx in _DNC_STATUS_FIELDS:
        hits = [m for m in map(rx.fullmatch, parts) if m]
        if len(hits) != 1:
            raise SXMBridgeError(f'Cannot parse {key!r} from DNC status {status!r}; refusing to guess')
        out[key] = float(hits[0].group(1))
    return out


def _add_dnc_status(data: dict, strict: bool) -> dict:
    data = dict(data)
    try:
        data.update(parse_dnc_status(data.get('Status', '')))
    except SXMBridgeError:
        if strict:
            raise
        data.update(dict.fromkeys(('q', 'f_peak', 'tau_us')))
    return data


class _Spec(NamedTuple):
    name: str                      # canonical name, as used in paths and snapshots
    attr: str                      # bridge property name
    method: str                    # backend method
    required: tuple = ()           # dotted fields that must be present and not None
    post: Callable | None = None   # (data, strict) -> data
    aliases: tuple = ()


_LIA = tuple(f'Lia{n}.{f}' for n in (1, 2, 3) for f in ('Link', 'Value1', 'Phase'))

SECTIONS = (
    _Spec('Scan', 'scan', 'scan', ('Range', 'Speed', 'Pixel', 'x-Center', 'y-Center', 'Angle')),
    _Spec('Topography feedback', 'topography', 'topography', ('Ref', 'Ki', 'Kp')),
    _Spec('Amplitude feedback', 'amplitude', 'amplitude', ('Ref', 'Ki', 'Kp', 'Pull back at', 'Pull Speed', 'Tau')),
    _Spec('PLL', 'pll', 'pll', ('Kp', 'Ki')),
    _Spec('zControl', 'zcontrol', 'zcontrol', ('Feedback Off', 'dz', 'dz per Mouse Tick', 'Slew Rate')),
    _Spec('Multi Channel LockIn', 'lockin', 'lockin', ('TimeConstant', 'RollOff') + _LIA),
    _Spec('Spectroscopy', 'spectroscopy', 'spectroscopy',
          ('X', 'Y', 'Delay1', 'AguT', 'dz', 'U Start', 'U Stop', 'Mode', 'Acquire')),
    _Spec('Dynamic Non-Contact R / Phi', 'dynamic', 'dynamic_non_contact',
          ('Input Gain InA', 'TimeConstant', 'RollOff', 'Range', 'Status', 'unmapped_numeric_controls'),
          _add_dnc_status, ('dnc',)),
    _Spec('Feedback mode', 'feedback', 'feedback_mode', ('Mode',)),
    _Spec('Scanner', 'scanner', 'scanner_state', ('caption',)),
    _Spec('Tip Conditioning', 'tip', 'tip_conditioning', ('Mode', 'unmapped_numeric_controls')),
    _Spec('Oscilloscope', 'oscilloscope', 'oscilloscope_state', ('Channels', 'x_axis_values', 'selected_modes')),
)


def _find_section(name: str) -> _Spec:
    want = _snake(name)
    for spec in SECTIONS:
        if want in (_snake(spec.name), spec.attr, *spec.aliases):
            return spec
    raise SXMPathError(f'Unknown section {name!r}; available: {[s.name for s in SECTIONS]}')


def _walk(value: Any, segments: list, where: str) -> Any:
    for seg in segments:
        if isinstance(value, Mapping):
            key = _match_key(value, seg, where)
            value, where = value[key], f'{where}.{key}'
        elif isinstance(value, list):
            if not seg.isdigit() or int(seg) >= len(value):
                raise SXMPathError(f'{where!r}: invalid index {seg!r} (length {len(value)})')
            value, where = value[int(seg)], f'{where}.{seg}'
        else:
            raise SXMPathError(f'{where!r} is a value, cannot descend into {seg!r}')
    return value


def _leaves(prefix: str, value: Any):
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _leaves(f'{prefix}.{k}', v)
    elif isinstance(value, list) and any(isinstance(v, Mapping) for v in value):
        for i, v in enumerate(value):
            yield from _leaves(f'{prefix}.{i}', v)
    else:
        yield prefix, value


class Section(Mapping):
    """Immutable view of one GUI section as read at one moment.

    Fields are reachable by original caption (``s['x-Center']``) or by
    snake_case attribute (``s.x_center``). Nested groups are Sections too.
    """
    __slots__ = ('_name', '_data')

    def __init__(self, name: str, data: Mapping):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_data', data)

    def __getitem__(self, key: str) -> Any:
        k = _match_key(self._data, key, self._name)
        value = self._data[k]
        return Section(f'{self._name}.{k}', value) if isinstance(value, Mapping) else copy.deepcopy(value)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith('_'):
            raise AttributeError(attr)
        try:
            return self[attr]
        except SXMPathError as exc:
            raise AttributeError(str(exc)) from None

    def __setattr__(self, attr, value):
        raise AttributeError('SXM sections are read-only')

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __dir__(self):
        return sorted(set(super().__dir__()) | {_snake(k) for k in self._data})

    def __repr__(self):
        return f'Section({self._name!r}, {self._data!r})'

    @property
    def name(self) -> str:
        return self._name

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self._data))


class Discovery:
    """Semantic-neutral inventory of SXM forms."""

    def __init__(self, bridge: AnfatecSXMBridge):
        self._bridge = bridge

    def _call(self, method: str, *args):
        try:
            return getattr(self._bridge._backend, method)(*args)
        except RuntimeError as exc:
            raise SXMBridgeError(str(exc)) from exc

    def forms(self) -> list:
        return self._call('discover_forms')

    def form(self, form_class: str) -> dict:
        return self._call('discovery_form', form_class)

    def raw(self, form_class: str) -> list:
        return self._call('raw_form', form_class)

    def save(self, path: str | None = None) -> str:
        """Write mapped parameters plus the full control inventory, for diagnosing layout changes."""
        now = datetime.datetime.now()
        path = path or 'SXM_discovery_' + now.strftime('%Y%m%d_%H%M%S') + '.json'
        data = {'timestamp': now.astimezone().isoformat(timespec='seconds'), 'source': TARGET_EXE,
                'mapped': self._bridge._read_all(strict=False), 'forms': self.forms()}
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return os.path.abspath(path)


class AnfatecSXMBridge:
    """Read-only semantic access to the running Femto_28_4.exe GUI."""

    def __init__(self, strict: bool = True, backend: Any = None):
        self._backend = backend if backend is not None else Win32Backend()
        self.strict = strict
        self.discovery = Discovery(self)

    def __repr__(self):
        return f'AnfatecSXMBridge(backend={BACKEND!r}, strict={self.strict})'

    # -- reading --

    def _read(self, spec: _Spec) -> dict:
        try:
            data = getattr(self._backend, spec.method)()
        except Exception as exc:
            raise SXMBridgeError(f'{spec.name}: {exc}') from exc
        if spec.post:
            data = spec.post(data, self.strict)
        if self.strict:
            for field in spec.required:
                try:
                    value = _walk(data, field.split('.'), spec.name)
                except SXMPathError:
                    value = None
                if value is None:
                    raise SXMBridgeError(f'{spec.name}: field {field!r} missing or empty; '
                                         'GUI layout may have changed, refusing to guess')
        return data

    def _read_all(self, strict: bool) -> dict:
        out = {}
        for spec in SECTIONS:
            try:
                out[spec.name] = self._read(spec)
            except SXMBridgeError as exc:
                if strict:
                    raise
                out[spec.name] = {'_error': str(exc)}
        return out

    def section(self, name: str) -> Section:
        """Live read of one section by name, alias or attribute name."""
        spec = _find_section(name)
        return Section(spec.name, self._read(spec))

    scan = property(lambda self: self.section('Scan'))
    topography = property(lambda self: self.section('Topography feedback'))
    amplitude = property(lambda self: self.section('Amplitude feedback'))
    pll = property(lambda self: self.section('PLL'))
    zcontrol = property(lambda self: self.section('zControl'))
    lockin = property(lambda self: self.section('Multi Channel LockIn'))
    spectroscopy = property(lambda self: self.section('Spectroscopy'))
    dynamic = property(lambda self: self.section('Dynamic Non-Contact R / Phi'))
    feedback = property(lambda self: self.section('Feedback mode'))
    scanner = property(lambda self: self.section('Scanner'))
    tip = property(lambda self: self.section('Tip Conditioning'))
    oscilloscope = property(lambda self: self.section('Oscilloscope'))

    # -- generic access --

    def get(self, path: str) -> Any:
        """Resolve ``'Section.Field[.Sub...]'``; names match by caption or snake_case, list items by index."""
        head, *rest = path.split('.')
        spec = _find_section(head)
        value = _walk(self._read(spec), rest, spec.name)
        return Section('.'.join([spec.name, *rest]), value) if isinstance(value, Mapping) else copy.deepcopy(value)

    __getitem__ = get

    def control(self, path: str) -> Any:
        """Locate, without reading or sending anything, what ``get(path)`` reads.

        Returns a Control, or the subtree of Controls under ``path``. Used by
        AnfatecSXMWriter so that writing a path acts on exactly the control read for it.
        """
        head, *rest = path.split('.')
        spec = _find_section(head)
        try:
            tree = self._backend.controls(spec.method)
        except Exception as exc:
            raise SXMBridgeError(f'{spec.name}: {exc}') from exc
        return _walk(tree, rest, spec.name)

    def paths(self) -> list:
        """All leaf paths currently readable. Sections that fail to read are left out."""
        return [p for p, _ in self._readable_leaves()]

    def search(self, term: str) -> dict:
        """``{path: value}`` for readable leaf paths with a word starting with ``term`` (case-insensitive).

        ``'Ki'`` matches ``Amplitude feedback.Ki`` but not ``Multi Channel LockIn``.
        """
        rx = re.compile(rf'(?:^|_){re.escape(_snake(term))}')
        return {p: v for p, v in self._readable_leaves() if rx.search(_snake(p))}

    def _readable_leaves(self):
        for name, data in self._read_all(strict=False).items():
            if '_error' not in data:
                yield from _leaves(name, data)

    # -- snapshots and discovery --

    def snapshot(self) -> dict:
        """All sections at once. In strict mode any failing section raises; otherwise it is recorded as ``_error``."""
        return {'timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                'source': TARGET_EXE, 'backend': BACKEND,
                'parameters': self._read_all(self.strict)}

    def save_snapshot(self, path: str | None = None, snapshot: dict | None = None) -> str:
        """Write ``snapshot`` (or a fresh one) as JSON; returns the absolute path."""
        snap = snapshot if snapshot is not None else self.snapshot()
        if path is None:
            path = 'SXM_snapshot_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(snap, fh, indent=2, ensure_ascii=False)
        return os.path.abspath(path)

    def raw(self, form_class: str | None = None) -> list:
        """Unfiltered control dump of one form, or the full discovery inventory when no form is given."""
        return self.discovery.forms() if form_class is None else self.discovery.raw(form_class)


if __name__ == '__main__':
    sxm = AnfatecSXMBridge(strict=False)
    snap = sxm.snapshot()
    print(json.dumps(snap, indent=2, ensure_ascii=False))
    print('\nSnapshot:', sxm.save_snapshot(snapshot=snap))
