"""AnfatecSXMWriter - set SXM GUI parameters, including those DDE does not expose, by the paths AnfatecSXMBridge reads.

    from AnfatecSXMWriter import AnfatecSXMWriter
    w = AnfatecSXMWriter()
    w.writable()                                  # {path: kind} for every control the bridge reads
    w.options('dnc.TimeConstant')                 # combo items / radio captions
    w.set('dnc.TimeConstant', '3 ms')             # select, re-read through the bridge, return the change
    w.set('dnc.Range', '±1')                      # radio button by caption ('+-1' also accepted)
    w.set('zControl.Feedback Off', True)          # checkbox
    w.set('dnc.unmapped_numeric_controls.3', 0.5, commit='enter')
    w.set('PLL.Ki', 2e4, dry_run=True)            # report what would be sent, send nothing

Paths are exactly ``AnfatecSXMBridge.paths()``. The control behind a path comes from
``AnfatecSXMBridge.control(path)``, the same structural lookup the bridge reads with,
so a write always lands on the control that is read for that path. Derived values
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
                                   _find_section, _norm, _num, _parent, _snake, _text, _user32, SMTO_ABORTIFHUNG)
except ImportError:                    # stand-alone, next to AnfatecSXMBridge.py
    from AnfatecSXMBridge import (AnfatecSXMBridge, Control, SXMBridgeError, SXMPathError, SECTIONS, _ULONG_PTR,
                                  _find_section, _norm, _num, _parent, _snake, _text, _user32, SMTO_ABORTIFHUNG)

__all__ = ['AnfatecSXMWriter', 'SXMWriteError', 'EDIT_COMMIT']

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

# Edit fields whose commit is verified on the instrument: snake_case path -> 'change' | 'enter'.
# e.g. 'dynamic_non_contact_r_phi.unmapped_numeric_controls.3': 'enter'
EDIT_COMMIT: dict[str, str] = {}

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
    """Write access to the running Femto_28_4.exe GUI, by AnfatecSXMBridge paths."""

    def __init__(self, bridge: AnfatecSXMBridge | None = None):
        self.bridge = bridge if bridge is not None else AnfatecSXMBridge(strict=True)

    def __repr__(self):
        return f'AnfatecSXMWriter(bridge={self.bridge!r})'

    # -- discovery --

    def _control(self, path: str) -> tuple[str, Control]:
        """``(read_path, Control)``; read_path is what the bridge reads for this control."""
        try:
            node = self.bridge.control(path)
        except SXMPathError:
            try:
                self.bridge.get(path)
            except SXMBridgeError:
                raise                               # not a path at all: keep the bridge's message
            raise SXMWriteError(f'{path!r} is derived by SXM (e.g. from the status bar), not a control; '
                                'it cannot be written') from None
        if isinstance(node, Mapping) and isinstance(node.get('value'), Control):  # unmapped_numeric_controls.N
            path, node = f'{path}.value', node['value']
        if not isinstance(node, Control):
            raise SXMWriteError(f'{path!r} is not a single control (derived, read-only or a group); '
                                f'writable paths: AnfatecSXMWriter().writable()')
        if not node.hwnds:
            raise SXMWriteError(f'{path!r}: no control found')
        return path, node

    def writable(self) -> dict:
        """``{path: kind}`` for every control of the forms that are open."""
        out = {}
        for spec in SECTIONS:
            try:
                tree = self.bridge.control(spec.name)
            except SXMBridgeError:
                continue
            out.update((p, c.kind) for p, c in _controls(tree, spec.name))
        return out

    def options(self, path: str) -> list | None:
        """Combo items, radio captions, ``[False, True]`` for a checkbox, None for an edit."""
        _, c = self._control(path)
        if c.kind == 'combo':
            return _combo_items(c.hwnds[0])
        if c.kind == 'choice':
            return [_text(h) for h in c.hwnds]
        return [False, True] if c.kind == 'check' else None

    # -- writing --

    def set(self, path: str, value: Any, *, commit: str | None = None, dry_run: bool = False,
            settle: float = SETTLE_S) -> dict:
        """Set ``path`` to ``value`` and confirm it through the bridge.

        Returns ``{'path', 'kind', 'before', 'requested', 'after', 'sent'}``; with
        ``dry_run`` nothing is sent and ``after`` is None. Raises SXMWriteError if the
        value is invalid, the control is disabled, or the GUI does not show the value
        within ``settle`` seconds.
        """
        read_path, c = self._control(path)
        before = self.bridge.get(read_path)
        h, sent = c.hwnds[0], []

        if c.kind == 'edit':
            head, *rest = read_path.split('.')
            commit = commit or EDIT_COMMIT.get('.'.join([_snake(_find_section(head).name), *map(_snake, rest)]))
            if commit not in ('change', 'enter'):
                raise SXMWriteError(f"{read_path}: how SXM commits this edit is not established. Pass "
                                    f"commit='change' or commit='enter' (check on the instrument which one "
                                    f"takes effect), then record it in EDIT_COMMIT.")
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
            index = _pick(items, value, read_path)
            wanted = items[index]
            sent += [f'CB_SETCURSEL {index} ({wanted!r})', 'CBN_SELCHANGE to parent']

            def act():
                _send(h, CB_SETCURSEL, index)
                _notify(h, CBN_SELCHANGE)

        elif c.kind == 'check':
            if not isinstance(value, bool):
                raise SXMWriteError(f'{read_path}: a checkbox takes True or False, not {value!r}')
            wanted = value
            if before is not value:
                sent.append('BN_CLICKED to parent (toggle)')

            def act():
                if before is not value:
                    _notify(h, BN_CLICKED)

        else:  # choice
            captions = [_text(b) for b in c.hwnds]
            h = c.hwnds[_pick(captions, value, read_path)]
            wanted = _text(h)
            sent.append(f'BN_CLICKED {wanted!r} to parent')

            def act():
                _notify(h, BN_CLICKED)

        result = {'path': read_path, 'kind': c.kind, 'before': before, 'requested': wanted,
                  'after': None, 'sent': sent}
        if dry_run:
            return result
        if not _user32.IsWindowEnabled(h):
            raise SXMWriteError(f'{read_path} is disabled (greyed out) in SXM; nothing sent')

        act()
        deadline = time.monotonic() + settle
        while True:
            after = self.bridge.get(read_path)
            if _same(c.kind, after, wanted) or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        result['after'] = after
        if not _same(c.kind, after, wanted):
            raise SXMWriteError(f'{read_path}: sent {wanted!r} but SXM shows {after!r} (was {before!r}); '
                                f'SXM rejected or changed the value', result)
        return result


if __name__ == '__main__':
    w = AnfatecSXMWriter(AnfatecSXMBridge(strict=False))
    for p, kind in w.writable().items():
        print(f'{kind:7} {p}')
