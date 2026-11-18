"""Status page: what the sensor and the loops are doing now, read from SXM in the background.

One ``sxm_ui.state.StatePoller`` reads SXM off the GUI thread; this page shows the values by group, marks what
changed since the read before, and lists findings (gui/status_rules.py). Reads only.
"""
from __future__ import annotations

import datetime

from PyQt5 import QtCore, QtWidgets

from sxm_ui import theme as T
from sxm_ui.widgets import FindingList

from . import status_rules as R


class StatusPage(QtWidgets.QWidget):
    assessed = QtCore.pyqtSignal(str)                       # worst level of the findings

    def __init__(self, poller=None, parent=None):
        super().__init__(parent)
        self.poller = poller
        self.values: dict = {}
        self.previous: dict | None = None
        self.read_at: datetime.datetime | None = None
        self.labels: dict[str, QtWidgets.QLabel] = {}

        self.refresh_btn = QtWidgets.QPushButton('Refresh from SXM')
        self.auto = QtWidgets.QCheckBox('every 10 s')
        self.auto.setToolTip('Read SXM in the background every 10 s while this page is shown.')
        self.when = QtWidgets.QLabel('not read yet')
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.refresh_btn)
        top.addWidget(self.auto)
        top.addWidget(self.when, 1)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        for n, (group, rows) in enumerate(R.GROUPS):
            box = QtWidgets.QGroupBox(group)
            form = QtWidgets.QFormLayout(box)
            form.setHorizontalSpacing(14)
            for key, label, _unit in rows:
                value = QtWidgets.QLabel('n/a')
                value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                form.addRow(label, value)
                self.labels[key] = value
            grid.addWidget(box, n // 3, n % 3)
        self.findings = FindingList()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(grid)
        lay.addWidget(QtWidgets.QLabel('<b>What this means</b>'))
        lay.addWidget(self.findings, 1)

        self.refresh_btn.clicked.connect(self.refresh)
        self.auto.toggled.connect(self._set_auto)
        self._age = QtCore.QTimer(self)
        self._age.timeout.connect(self._show_age)
        self._age.start(1000)
        if poller is not None:
            poller.changed.connect(self.show_readout)
            poller.failed.connect(self.show_error)
            poller.busy.connect(lambda on: self.refresh_btn.setEnabled(not on))

    # -- reading --

    def refresh(self):
        if self.poller is not None:
            self.when.setText('reading SXM ...')
            self.poller.refresh()

    def _set_auto(self, on: bool):
        if self.poller is not None:
            self.poller.set_interval(10.0 if on and self.isVisible() else 0.0)

    def on_enter(self):
        self.refresh()
        self._set_auto(self.auto.isChecked())

    def on_leave(self):
        if self.poller is not None:
            self.poller.set_interval(0.0)

    # -- showing --

    def show_readout(self, readout):
        values = dict(getattr(readout, 'values', readout))
        if self.values:
            self.previous = self.values
        self.values = values
        self.read_at = getattr(readout, 'timestamp', None) or datetime.datetime.now()
        units = {key: unit for _g, rows in R.GROUPS for key, _l, unit in rows}
        accent = T.level_colors(T.WARN)[0]
        errors = getattr(readout, 'errors', {}) or {}
        for key, label in self.labels.items():
            v = values.get(key)
            text = R.show(key, v, units[key])
            was = None if self.previous is None else self.previous.get(key)
            if was is not None and v is not None and was != v:
                label.setText(f'<b style="color:{accent}">{text}</b> <span style="color:{accent}">'
                              f'(was {R.show(key, was, units[key])})</span>')
            else:
                label.setText(f'<b>{text}</b>')
            label.setToolTip(errors.get(key, ''))
        found = R.assess(values, self.previous)
        self.findings.set_findings(found)
        self._show_age()
        self.assessed.emit(self.findings.level())

    def show_error(self, text: str):
        self.when.setText(f'read failed: {text}')
        self.findings.set_findings([R.finding(R.BAD, 'SXM could not be read', text,
                                              'Is SXM running? Press Refresh to try again.')])
        self.assessed.emit(T.BAD)

    def _show_age(self):
        if self.read_at is None:
            return
        age = (datetime.datetime.now() - self.read_at).total_seconds()
        old = age > 120
        self.when.setText(f'read at {self.read_at:%H:%M:%S} ({age:.0f} s ago)' + (' - press Refresh' if old else ''))
        self.when.setStyleSheet(f'color: {T.level_colors(T.WARN)[0]};' if old else '')
