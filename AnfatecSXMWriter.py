"""AnfatecSXMWriter - set SXM GUI parameters, including those DDE does not expose, by the paths AnfatecSXMBridge reads.

Find a parameter:

    python AnfatecSXMWriter.py                 # every parameter: name, value, accepted values, meaning
    python AnfatecSXMWriter.py dnc             # one SXM window
    python AnfatecSXMWriter.py dnc.tc          # one parameter with all its options

    from AnfatecSXMWriter import AnfatecSXMWriter
    w = AnfatecSXMWriter()
    print(w.table())                           # same table; w.table('dnc') for one window
    w.dnc                                      # same, attribute style (tab-completes in IPython / Jupyter)
    w.dnc.tc                                   # description, SXM location, value, options
    w.dnc.tc.options                           # ['0.1 ms', '0.3 ms', ...]

Set it:

    w.set('dnc.tc', '3 ms')                    # dropdown, by item text
    w.dnc.output_gain.set('±1')                # radio buttons, by caption ('+-1' also accepted)
    w.set('z.feedback_off', True)              # tick box
    w.set('dnc.drive', 0.5, commit='enter')    # number field (see commit below)
    w.set('pll.ki', 2e4, dry_run=True)         # show what would be sent, send nothing

The short names (``PARAMS``, below, grouped by SXM window in ``GROUPS``) follow
sxm_ncafm_control/sxm_state.py. Any AnfatecSXMBridge path works too
(``'Amplitude feedback.Tau'``). The control behind a name comes from
``AnfatecSXMBridge.control(path)``, the same structural lookup the bridge reads with,
so a write always lands on the control that is read for that name. Derived values
(``dnc.q``, ``dnc.tau_us``, ``dnc.Status``, ...) have no control and cannot be written.

How a value reaches SXM: only the window messages Delphi's VCL itself reacts to,
sent to that one control. No mouse, keyboard input, focus or foreground change.

- combo:          CB_SETCURSEL, then CBN_SELCHANGE to the parent  -> VCL OnChange/OnSelect
- check, choice:  BN_CLICKED to the parent                        -> VCL toggles/checks it, OnClick
- edit:           WM_SETTEXT (EN_CHANGE)                          -> VCL OnChange
                  commit='enter' also posts an Enter key to the edit, for fields SXM only applies on Enter.

Which commit an edit needs is a property of SXM, established once per field on the
instrument and then recorded in EDIT_COMMIT. Until then ``set`` refuses an edit without
an explicit ``commit=``: never guessed.

Every write re-reads the path through the bridge and raises SXMWriteError unless the
GUI shows the requested value. That proves SXM's GUI took the value, not that the
hardware followed it; check that once per field on the instrument.

Master copy: anfatec_code/AnfatecSXMWriter.py in the author's development folder.
sxm_ncafm_control ships a copy next to its copy of AnfatecSXMBridge.py; make
changes in the master, not in the copy.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import math
import time
from collections.abc import Mapping
from typing import Any

try:                                   # inside a package (sxm_ncafm_control)
    from .AnfatecSXMBridge import (AnfatecSXMBridge, Control, SXMBridgeError, SXMPathError, SECTIONS, _ULONG_PTR,
                                   _find_section, _norm, _num, _parent, _snake, _text, _user32, _walk, SMTO_ABORTIFHUNG)
except ImportError:                    # stand-alone, next to AnfatecSXMBridge.py
    from AnfatecSXMBridge import (AnfatecSXMBridge, Control, SXMBridgeError, SXMPathError, SECTIONS, _ULONG_PTR,
                                  _find_section, _norm, _num, _parent, _snake, _text, _user32, _walk, SMTO_ABORTIFHUNG)

__all__ = ['AnfatecSXMWriter', 'SXMWriteError', 'PARAMS', 'GROUPS', 'EDIT_COMMIT', 'Param', 'Group', 'main']

WM_SETTEXT = 0x000C
WM_COMMAND = 0x0111
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_RETURN = 0x0D
BN_CLICKED = 0
CBN_SELCHANGE = 1
CB_GETCOUNT = 0x0146
CB_GETLBTEXT = 0x0148
CB_GETLBTEXTLEN = 0x0149
CB_SETCURSEL = 0x014E
ERROR_ACCESS_DENIED = 5
WRITE_TIMEOUT_MS = 5000         # SXM may talk to the controller inside its handler
SETTLE_S = 1.0                  # how long a write may take to show in the GUI

# =============================================================================
# Parameter names
# =============================================================================

# Group -> SXM window, in display order.
GROUPS = {
    'amp': 'Amplitude feedback (AFL)',
    'pll': 'PLL',
    'dnc': 'Dynamic Non-Contact R / Phi',
    'topo': 'Topography feedback',
    'scan': 'Scan',
    'z': 'zControl',
    'lockin': 'Multi Channel LockIn',
    'spec': 'Spectroscopy',
    'feedback': 'Feedback mode',
    'tip': 'Tip Conditioning',
    'scope': 'Oscilloscope',
}

_DNC = 'Dynamic Non-Contact R / Phi'
_DNC_EDIT = _DNC + '.unmapped_numeric_controls.{}.value'

# Short name -> (bridge path, description). Names follow sxm_ncafm_control/sxm_state.py.
# Values are in SXM's GUI units. "DDE ..." = the same parameter is also writable over DDE.
# "(status bar)" entries are computed by SXM: readable, not writable.
PARAMS: dict[str, tuple[str, str]] = {
    'amp.ref':           ('Amplitude feedback.Ref', 'amplitude setpoint (DDE ScanPara Edit23)'),
    'amp.ki':            ('Amplitude feedback.Ki', 'amplitude loop integral gain (DDE Edit24)'),
    'amp.kp':            ('Amplitude feedback.Kp', 'amplitude loop proportional gain (DDE Edit32)'),
    'amp.pull_back':     ('Amplitude feedback.Pull back at', 'pull back at'),
    'amp.pull_speed':    ('Amplitude feedback.Pull Speed', 'pull speed'),
    'amp.tau':           ('Amplitude feedback.Tau', 'amplitude loop time constant Tau'),

    'pll.kp':            ('PLL.Kp', 'PLL proportional gain (DDE Edit27)'),
    'pll.ki':            ('PLL.Ki', 'PLL integral gain (DDE Edit22)'),

    'dnc.input_gain':    (_DNC + '.Input Gain InA', 'input gain InA, x1 / x10'),
    'dnc.output_gain':   (_DNC + '.Range', 'AFL output gain, +-V (the "Range" radio group)'),
    'dnc.tc':            (_DNC + '.TimeConstant', 'PLL lock-in time constant t'),
    'dnc.rolloff':       (_DNC + '.RollOff', 'PLL lock-in roll-off'),
    'dnc.sweep_start':   (_DNC_EDIT.format(0), 'sweep start, Hz (DDE DNCPara 1; by position, unconfirmed)'),
    'dnc.sweep_stop':    (_DNC_EDIT.format(1), 'sweep stop, Hz (DDE DNCPara 2; by position, unconfirmed)'),
    'dnc.used_freq':     (_DNC_EDIT.format(2), 'used frequency f0, Hz (DDE DNCPara 3; by position, unconfirmed)'),
    'dnc.drive':         (_DNC_EDIT.format(3), 'drive (DDE DNCPara 4; by position, unconfirmed)'),
    'dnc.q':             (_DNC + '.q', 'quality factor Q (status bar)'),
    'dnc.f_peak':        (_DNC + '.f_peak', 'resonance peak frequency, Hz (status bar)'),
    'dnc.tau_us':        (_DNC + '.tau_us', 'ring-down time Q/(pi fPeak), us (status bar)'),

    'topo.ref':          ('Topography feedback.Ref', 'topography setpoint'),
    'topo.ki':           ('Topography feedback.Ki', 'topography integral gain'),
    'topo.kp':           ('Topography feedback.Kp', 'topography proportional gain'),

    'scan.range':        ('Scan.Range', 'scan range'),
    'scan.speed':        ('Scan.Speed', 'scan speed'),
    'scan.pixels':       ('Scan.Pixel', 'pixels per line'),
    'scan.x_center':     ('Scan.x-Center', 'x centre'),
    'scan.y_center':     ('Scan.y-Center', 'y centre'),
    'scan.angle':        ('Scan.Angle', 'scan angle'),

    'z.feedback_off':    ('zControl.Feedback Off', '"Feedback Off" tick box'),
    'z.dz':              ('zControl.dz', 'dz'),
    'z.dz_per_tick':     ('zControl.dz per Mouse Tick', 'dz per mouse tick'),
    'z.slew_rate':       ('zControl.Slew Rate', 'slew rate'),

    'lockin.tc':         ('Multi Channel LockIn.TimeConstant', 'lock-in time constant t'),
    'lockin.rolloff':    ('Multi Channel LockIn.RollOff', 'lock-in roll-off'),
    **{f'lockin.lia{n}_{key}': (f'Multi Channel LockIn.Lia{n}.{field}', f'Lia {n} {what}')
       for n in (1, 2, 3) for key, field, what in (('link', 'Link', 'link'), ('value', 'Value1', 'Value1'),
                                                    ('phase', 'Phase', 'phase'))},

    'spec.x':            ('Spectroscopy.X', 'X'),
    'spec.y':            ('Spectroscopy.Y', 'Y'),
    'spec.delay1':       ('Spectroscopy.Delay1', 'Delay1'),
    'spec.agu_t':        ('Spectroscopy.AguT', 'AguT'),
    'spec.dz':           ('Spectroscopy.dz', 'dz'),
    'spec.u_start':      ('Spectroscopy.U Start', 'bias sweep start'),
    'spec.u_stop':       ('Spectroscopy.U Stop', 'bias sweep stop'),
    'spec.mode':         ('Spectroscopy.Mode', 'spectroscopy mode'),
    **{f'spec.acquire{n}': (f'Spectroscopy.Acquire.{n - 1}', f'acquired channel {n}') for n in (1, 2, 3, 4)},

    'feedback.mode':     ('Feedback mode.Mode', 'feedback mode'),

    'tip.mode':          ('Tip Conditioning.Mode', 'tip conditioning mode'),
    **{f'tip.field{n}': (f'Tip Conditioning.unmapped_numeric_controls.{n - 1}.value',
                         f'numeric field {n} from the top (no caption)') for n in (1, 2, 3)},

    **{f'scope.channel{n}': (f'Oscilloscope.Channels.{n - 1}', f'channel {n}') for n in (1, 2, 3)},
    'scope.x_axis':      ('Oscilloscope.x_axis_values.0', 'x-axis value'),
}

# Edit fields whose commit is verified on the instrument: short name -> 'change' | 'enter'.
# e.g. 'dnc.drive': 'enter'
EDIT_COMMIT: dict[str, str] = {}


def _path_key(path: str) -> str:
    head, *rest = path.split('.')
    return '.'.join([_snake(_find_section(head).name), *map(_snake, rest)])


_BY_PATH = {_path_key(path): name for name, (path, _) in PARAMS.items()}

_user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
_user32.IsWindowEnabled.argtypes = [wt.HWND]
_user32.GetDlgCtrlID.argtypes = [wt.HWND]


class SXMWriteError(SXMBridgeError):
    """A write was refused, failed, or did not show up in the SXM GUI. ``result`` holds before/after when known."""

    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result


# =============================================================================
# Win32 sends (the only place that changes SXM)
# =============================================================================

def _send(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0) -> int:
    result = _ULONG_PTR()
    ctypes.set_last_error(0)
    if not _user32.SendMessageTimeoutW(hwnd, msg, wparam, lparam, SMTO_ABORTIFHUNG, WRITE_TIMEOUT_MS,
                                       ctypes.byref(result)):
        err = ctypes.get_last_error()
        if err == ERROR_ACCESS_DENIED:
            raise SXMWriteError('Windows blocked the message: SXM runs with higher rights than Python. '
                                'Run Python as administrator as well.')
        raise SXMWriteError(f'SXM did not answer message 0x{msg:X} within {WRITE_TIMEOUT_MS} ms (error {err}); '
                            'it may or may not have been applied. Re-read before retrying.')
    return ctypes.c_ssize_t(result.value).value


def _post(hwnd: int, msg: int, wparam: int, lparam: int) -> None:
    if not _user32.PostMessageW(hwnd, msg, wparam, lparam):
        raise SXMWriteError(f'Could not post message 0x{msg:X} (error {ctypes.get_last_error()})')


def _notify(hwnd: int, code: int) -> None:
    """Tell the parent that ``hwnd`` sent ``code``. VCL reflects it to the control's own handler."""
    ident = _user32.GetDlgCtrlID(hwnd) & 0xFFFF
    _send(_parent(hwnd), WM_COMMAND, (code << 16) | ident, hwnd)


def _combo_items(hwnd: int) -> list:
    items = []
    for i in range(max(_send(hwnd, CB_GETCOUNT), 0)):
        buf = ctypes.create_unicode_buffer(max(_send(hwnd, CB_GETLBTEXTLEN, i), 0) + 1)
        _send(hwnd, CB_GETLBTEXT, i, ctypes.addressof(buf))
        items.append(buf.value.strip())
    return items


# =============================================================================
# Value matching
# =============================================================================

def _caption_key(s: Any) -> str:
    return _norm(str(s).replace('+/-', '±').replace('+-', '±'))


def _pick(options: list, value: Any, path: str) -> int:
    hits = [i for i, o in enumerate(options) if _caption_key(o) == _caption_key(value)]
    if len(hits) != 1 and isinstance(value, (int, float)):
        hits = [i for i, o in enumerate(options) if _num(o) == value]
    if len(hits) != 1:
        raise SXMWriteError(f'{path}: {value!r} is not one of {options}')
    return hits[0]


def _edit_text(value: Any, current: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not isinstance(_num(text), float):
            raise SXMWriteError(f'{text!r} is not a number')
        return text
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SXMWriteError(f'{value!r} is not a finite number')
    v = float(value)
    text = str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    if ',' in current and '.' not in current:            # SXM shows a decimal comma: answer in kind
        text = text.replace('.', ',')
    return text


def _same(kind: str, shown: Any, wanted: Any) -> bool:
    if kind == 'edit':
        a, b = _num(str(shown).replace(',', '.')), _num(str(wanted).replace(',', '.'))
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-300)
        return str(shown) == str(wanted)
    if kind == 'check':
        return shown is wanted
    return _caption_key(shown) == _caption_key(wanted)


# =============================================================================
# Writer
# =============================================================================

def _controls(tree: Any, prefix: str):
    """(path, Control) for every writable leaf of a control tree."""
    if isinstance(tree, Control):
        yield prefix, tree
    elif isinstance(tree, Mapping):
        for k, v in tree.items():
            yield from _controls(v, f'{prefix}.{k}')
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _controls(v, f'{prefix}.{i}')


class AnfatecSXMWriter:
    """Write access to the running Femto_28_4.exe GUI, by short name (``dnc.tc``) or bridge path.

    Browse: ``print(w.table())``, ``print(w.table('dnc'))``, ``print(w.describe('dnc.tc'))``,
    or attribute style with tab completion: ``w.dnc``, ``w.dnc.tc``, ``w.dnc.tc.options``,
    ``w.dnc.tc.set('3 ms')``.
    """

    def __init__(self, bridge: AnfatecSXMBridge | None = None):
        self.bridge = bridge if bridge is not None else AnfatecSXMBridge(strict=True)

    def __repr__(self):
        return f'AnfatecSXMWriter(bridge={self.bridge!r})'

    def __getattr__(self, attr):
        if attr in GROUPS:
            return Group(self, attr)
        raise AttributeError(f'{type(self).__name__!r} has no attribute {attr!r}; groups: {", ".join(GROUPS)}')

    def __dir__(self):
        return [*super().__dir__(), *GROUPS]

    # -- names --

    @staticmethod
    def names(group: str | None = None) -> list:
        """Short names, all or of one group (``'dnc'``)."""
        if group is not None and group not in GROUPS:
            raise SXMPathError(f'Unknown group {group!r}; groups: {", ".join(GROUPS)}')
        return [n for n in PARAMS if group is None or n.split('.')[0] == group]

    @staticmethod
    def _path(name: str) -> str:
        """Bridge path for a short name; anything else is taken as a bridge path."""
        key = name.strip().lower()
        return PARAMS[key][0] if key in PARAMS else name

    @staticmethod
    def _short(path: str) -> str | None:
        try:
            return _BY_PATH.get(_path_key(path))
        except SXMBridgeError:
            return None

    def _control(self, name: str) -> tuple[str, Control]:
        """``(read_path, Control)``; read_path is what the bridge reads for this control."""
        path = self._path(name)
        try:
            node = self.bridge.control(path)
        except SXMPathError:
            try:
                self.bridge.get(path)
            except SXMPathError:
                group = name.split('.')[0].strip().lower()
                if group in GROUPS and path == name:
                    raise SXMPathError(f'{name!r}: unknown name. In {group!r}: '
                                       f'{", ".join(self.names(group))}') from None
                raise                               # not a path at all: keep the bridge's message
            raise SXMWriteError(f'{name!r} is derived by SXM (e.g. from the status bar), not a control; '
                                'it cannot be written') from None
        if isinstance(node, Mapping) and isinstance(node.get('value'), Control):  # unmapped_numeric_controls.N
            path, node = f'{path}.value', node['value']
        if not isinstance(node, Control):
            raise SXMWriteError(f'{name!r} is not a single control (derived, read-only or a group); '
                                'see AnfatecSXMWriter().table()')
        if not node.hwnds:
            raise SXMWriteError(f'{name!r}: no control found')
        return path, node

    # -- reading and browsing --

    def get(self, name: str) -> Any:
        """Current value as SXM shows it."""
        return self.bridge.get(self._path(name))

    def kind(self, name: str) -> str:
        """``'edit'`` (number), ``'combo'`` (dropdown), ``'check'`` (tick box) or ``'choice'`` (radio buttons)."""
        return self._control(name)[1].kind

    def options(self, name: str) -> list | None:
        """Dropdown items, radio captions, ``[False, True]`` for a tick box, None for a number field."""
        _, c = self._control(name)
        if c.kind == 'combo':
            return _combo_items(c.hwnds[0])
        if c.kind == 'choice':
            return [_text(h) for h in c.hwnds]
        return [False, True] if c.kind == 'check' else None

    def _commit(self, read_path: str) -> str | None:
        return EDIT_COMMIT.get(self._short(read_path) or '')

    def _how(self, read_path: str, c: Control, width: int | None = None) -> str:
        """What the control accepts, in words."""
        if c.kind == 'edit':
            commit = self._commit(read_path)
            return f'number (commit {commit!r})' if commit else "number, needs commit='change'|'enter'"
        if c.kind == 'check':
            return 'True / False'
        opts = ' | '.join(map(str, _combo_items(c.hwnds[0]) if c.kind == 'combo' else [_text(h) for h in c.hwnds]))
        return opts if width is None or len(opts) <= width else opts[:width - 3] + '...'

    def describe(self, name: str) -> str:
        """Everything about one parameter: description, SXM location, value, what it accepts."""
        try:
            read_path, c = self._control(name)
        except SXMWriteError:                               # derived value: readable, no control
            path = self._path(name)
            value, short = self.bridge.get(path), self._short(path)
            return '\n'.join([f'{short}: {PARAMS[short][1]}' if short else path,
                              f'  SXM path : {path}',
                              f'  value    : {value!r}',
                              '  read-only: computed by SXM, there is no control to set'])
        short = self._short(read_path)
        label = short or read_path
        lines = [f'{short}: {PARAMS[short][1]}' if short else read_path,
                 f'  SXM path : {read_path}',
                 f'  kind     : {c.kind} ({_KINDS[c.kind]})',
                 f'  value    : {self.bridge.get(read_path)!r}',
                 f'  accepts  : {self._how(read_path, c)}',
                 f'  set with : w.set({label!r}, ...)' + (f'  or  w.{short}.set(...)' if short else '')]
        return '\n'.join(lines)

    def table(self, group: str | None = None) -> str:
        """All parameters (or one group) with current value, what they accept and what they are."""
        reader = AnfatecSXMBridge(strict=False, backend=self.bridge._backend)
        out = []
        for g in ([group] if group else GROUPS):
            values, trees, rows = {}, {}, []
            for n in self.names(g):
                path = PARAMS[n][0]
                head, *rest = path.split('.')
                if head not in values:
                    try:
                        values[head], trees[head] = reader.section(head).to_dict(), reader.control(head)
                    except SXMBridgeError as exc:
                        values[head] = trees[head] = exc
                try:
                    if isinstance(values[head], Exception):
                        raise values[head]
                    value = _walk(values[head], rest, head)
                    try:
                        how = self._how(path, _walk(trees[head], rest, head), 44)
                    except SXMPathError:
                        how = 'read-only (derived by SXM)'
                    rows.append((n, repr(value), how, PARAMS[n][1]))
                except (SXMBridgeError, RuntimeError):
                    rows.append((n, 'n/a', '', PARAMS[n][1]))
            closed = [v for v in values.values() if isinstance(v, Exception)]
            out.append(f'{g:8} {GROUPS[g]}' + (f'   [not readable: {closed[0]}]' if closed else ''))
            out += [f'  {n:18} {v:>14}   {how:44}   {what}' for n, v, how, what in rows]
            out.append('')
        return '\n'.join(out).rstrip()

    def writable(self) -> dict:
        """``{name: kind}`` for every control of the open SXM windows (short name, else bridge path)."""
        out = {}
        for spec in SECTIONS:
            try:
                tree = self.bridge.control(spec.name)
            except SXMBridgeError:
                continue
            out.update((self._short(p) or p, c.kind) for p, c in _controls(tree, spec.name))
        return out

    # -- writing --

    def set(self, name: str, value: Any, *, commit: str | None = None, dry_run: bool = False,
            settle: float = SETTLE_S) -> dict:
        """Set ``name`` (short name or bridge path) to ``value`` and confirm it through the bridge.

        Returns ``{'name', 'path', 'kind', 'before', 'requested', 'after', 'sent'}``; with
        ``dry_run`` nothing is sent and ``after`` is None. Raises SXMWriteError if the
        value is invalid, the control is disabled, or the GUI does not show the value
        within ``settle`` seconds.
        """
        read_path, c = self._control(name)
        label = self._short(read_path) or read_path
        before = self.bridge.get(read_path)
        h, sent = c.hwnds[0], []

        if c.kind == 'edit':
            commit = commit or self._commit(read_path)
            if commit not in ('change', 'enter'):
                raise SXMWriteError(f"{label}: how SXM commits this edit is not established. Pass "
                                    f"commit='change' or commit='enter' (check on the instrument which one "
                                    f"takes effect), then record it in EDIT_COMMIT[{label!r}].")
            wanted = _edit_text(value, _text(h))
            sent.append(f'WM_SETTEXT {wanted!r}')
            if commit == 'enter':
                sent.append('Enter key (posted)')

            def act():
                buf = ctypes.create_unicode_buffer(wanted)
                _send(h, WM_SETTEXT, 0, ctypes.addressof(buf))
                if commit == 'enter':
                    _post(h, WM_KEYDOWN, VK_RETURN, 0x001C0001)       # TranslateMessage turns this into WM_CHAR
                    _post(h, WM_KEYUP, VK_RETURN, 0xC01C0001)

        elif c.kind == 'combo':
            items = _combo_items(h)
            index = _pick(items, value, label)
            wanted = items[index]
            sent += [f'CB_SETCURSEL {index} ({wanted!r})', 'CBN_SELCHANGE to parent']

            def act():
                _send(h, CB_SETCURSEL, index)
                _notify(h, CBN_SELCHANGE)

        elif c.kind == 'check':
            if not isinstance(value, bool):
                raise SXMWriteError(f'{label}: a tick box takes True or False, not {value!r}')
            wanted = value
            if before is not value:
                sent.append('BN_CLICKED to parent (toggle)')

            def act():
                if before is not value:
                    _notify(h, BN_CLICKED)

        else:  # choice
            captions = [_text(b) for b in c.hwnds]
            h = c.hwnds[_pick(captions, value, label)]
            wanted = _text(h)
            sent.append(f'BN_CLICKED {wanted!r} to parent')

            def act():
                _notify(h, BN_CLICKED)

        result = {'name': label, 'path': read_path, 'kind': c.kind, 'before': before, 'requested': wanted,
                  'after': None, 'sent': sent}
        if dry_run:
            return result
        if not _user32.IsWindowEnabled(h):
            raise SXMWriteError(f'{label} is disabled (greyed out) in SXM; nothing sent')

        act()
        deadline = time.monotonic() + settle
        while True:
            after = self.bridge.get(read_path)
            if _same(c.kind, after, wanted) or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        result['after'] = after
        if not _same(c.kind, after, wanted):
            raise SXMWriteError(f'{label}: sent {wanted!r} but SXM shows {after!r} (was {before!r}); '
                                f'SXM rejected or changed the value', result)
        return result



class Param:
    """One parameter by short name: ``.value``, ``.options``, ``.kind``, ``.set(value)``; repr describes it."""

    def __init__(self, writer: AnfatecSXMWriter, name: str):
        self._writer, self.name = writer, name

    path = property(lambda self: PARAMS[self.name][0])
    description = property(lambda self: PARAMS[self.name][1])
    value = property(lambda self: self._writer.get(self.name))
    options = property(lambda self: self._writer.options(self.name))
    kind = property(lambda self: self._writer.kind(self.name))

    def set(self, value: Any, **kwargs) -> dict:
        return self._writer.set(self.name, value, **kwargs)

    def __repr__(self):
        try:
            return self._writer.describe(self.name)
        except SXMBridgeError as exc:
            return f'{self.name}: {self.description}\n  not readable: {exc}'


class Group:
    """The parameters of one SXM window, as attributes (``w.dnc.tc``); repr lists them with values."""

    def __init__(self, writer: AnfatecSXMWriter, group: str):
        self._writer, self._group = writer, group

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError(attr)
        name = f'{self._group}.{attr}'
        if name not in PARAMS:
            raise AttributeError(f'{self._group!r} has no parameter {attr!r}; available: {", ".join(dir(self))}')
        return Param(self._writer, name)

    def __dir__(self):
        return [n.split('.', 1)[1] for n in AnfatecSXMWriter.names(self._group)]

    def __repr__(self):
        return self._writer.table(self._group)


_KINDS = {'edit': 'number field', 'combo': 'dropdown', 'check': 'tick box', 'choice': 'radio buttons'}

_USAGE = """examples:
  python AnfatecSXMWriter.py                       every parameter: value, accepted values, meaning
  python AnfatecSXMWriter.py dnc                   one window (groups: {groups})
  python AnfatecSXMWriter.py dnc.tc                one parameter, with all its options
  python AnfatecSXMWriter.py dnc.tc "3 ms"         show what would be sent (nothing is sent)
  python AnfatecSXMWriter.py dnc.tc "3 ms" --apply
  python AnfatecSXMWriter.py z.feedback_off on --apply
  python AnfatecSXMWriter.py dnc.drive 0.5 --commit enter --apply"""


def main(argv: list | None = None) -> None:
    """Command line: list, describe or set parameters (see ``python AnfatecSXMWriter.py -h``)."""
    import argparse
    import sys

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')        # captions with '±', 'µ' on any console
    ap = argparse.ArgumentParser(prog='AnfatecSXMWriter.py', description='List, inspect or set SXM GUI parameters.',
                                 epilog=_USAGE.format(groups=', '.join(GROUPS)),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', nargs='?', help='group (dnc) or parameter (dnc.tc); omit to list everything')
    ap.add_argument('value', nargs='?', help='new value; only shown, not sent, unless --apply')
    ap.add_argument('--apply', action='store_true', help='send the new value to SXM')
    ap.add_argument('--commit', choices=('change', 'enter'), help='for number fields')
    args = ap.parse_args(argv)

    w = AnfatecSXMWriter(AnfatecSXMBridge(strict=False))
    try:
        if args.name is None or args.name in GROUPS:
            print(w.table(args.name))
        elif args.value is None:
            print(w.describe(args.name))
        else:
            value: Any = args.value
            if w.kind(args.name) == 'check':
                truth = {'on': True, 'true': True, '1': True, 'yes': True,
                         'off': False, 'false': False, '0': False, 'no': False}
                if value.lower() not in truth:
                    raise SXMWriteError(f'{args.name}: give on/off (or true/false), not {value!r}')
                value = truth[value.lower()]
            r = w.set(args.name, value, commit=args.commit, dry_run=not args.apply)
            if args.apply:
                print(f'{r["name"]}: {r["before"]!r} -> {r["after"]!r}')
            else:
                print(f'{r["name"]}: {r["before"]!r} -> {r["requested"]!r}, would send {r["sent"]}\n'
                      'Nothing sent; add --apply.')
    except SXMBridgeError as exc:
        raise SystemExit(f'error: {exc}')


if __name__ == '__main__':
    main()
