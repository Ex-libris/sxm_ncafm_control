"""One question every capture asks before it starts: may I use the driver now?

Scope, Live Scope, Single Step, a Tuning run and a characterisation campaign all read the same driver handle;
two of them at once spoil each other's timing. The shell window (gui/shell_window.py) installs an arbiter
that knows which of them is active. ``allow(widget, name)`` then stops an interruptible one (Live Scope) or
refuses with the name of the one that is running. Without an arbiter (classic window, tests) it allows.
"""
from __future__ import annotations

from PyQt5 import QtWidgets

_arbiter = None
warn = None            # tests: a callable(text) instead of the message box


def set_arbiter(arbiter) -> None:
    global _arbiter
    _arbiter = arbiter


def allow(widget, name: str) -> bool:
    if _arbiter is None:
        return True
    try:
        _arbiter.request(name)
    except RuntimeError as exc:                       # sxm_ui.session.Busy
        if warn is not None:
            warn(str(exc))
        else:
            QtWidgets.QMessageBox.warning(widget, "Driver in use", str(exc))
        return False
    return True
