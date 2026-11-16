"""The window on the shared shell (sxm_ui): pages by task instead of ten flat tabs.

The pages are the tabs of the classic ``MainWindow``, which is still built (never shown) and keeps doing the
cross-tab wiring, reconnect, font handling and shutdown; this window moves its tabs into task groups and adds:

- **Status** (home): SXM read in the background, values by group, findings (gui/status_page.py);
- a status strip: connection, who is using the driver, sensor (f0, Q);
- a driver arbiter (gui/driver_gate.py): a capture that starts stops Live Scope, and is refused with the
  name of any other capture that is running;
- View menu for text size / dark / high contrast (the toolbar row is gone).

``python -m sxm_ncafm_control.app --classic`` opens the old tabbed window.
"""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from sxm_ui import theme as T
from sxm_ui.session import Arbiter
from sxm_ui.shell import Shell
from sxm_ui.state import StatePoller

from . import driver_gate
from . import status_rules as R
from .main_window import MainWindow
from .status_page import StatusPage

# (group, [(page title, MainWindow attribute, hint)])
LAYOUT = [
    ('Sensor', [('Characterisation', 'characterisation_tab', 'The qPlus by itself, open loop: drive x TimeConstant campaign.'),
                ('Suggested setup', 'suggest_tab', 'Starting gains from Q and f0; Lorentzian fit of a spectrum.'),
                ('Amplitude calibration', 'qplus_tab', 'pm per mV from an amplitude sweep.')]),
    ('Tune', [('Tuning', 'tuning_tab', 'Run sheets and the automatic search for the amplitude loop.'),
              ('Step test', 'step_tab', 'Square wave on one parameter, recorded by the Scope.'),
              ('Single step', 'single_step_tab', 'One step with long times on either side.')]),
    ('Look', [('Scope', 'scope_tab', 'One-shot capture of two channels.'),
              ('Live scope', 'live_scope_tab', 'Rolling view; stops by itself when a capture needs the driver.')]),
    ('Operate', [('Parameters', 'params_tab', 'Stage, apply and read back gains and setpoints.'),
                 ('Constant height', 'topo_hold_tab', 'z with the feedback off.')]),
]


class _TabJump:
    """Stands in for the old QTabWidget where a tab jumps to another one (Step Test -> Scope)."""

    def __init__(self, shell, titles):
        self.shell, self.titles = shell, titles

    def setCurrentIndex(self, index):
        self.shell.show_page(self.titles[index])


class ShellWindow(Shell):
    def __init__(self, conn, read_at_start: bool = True):
        classic = MainWindow(conn)                            # builds the tabs and their wiring; never shown
        super().__init__('NC-AFM Control Suite', on_scale=self._set_scale,
                         on_dark=classic.accessibility_manager.set_dark_mode)
        self.classic, self.manager = classic, classic.accessibility_manager
        self.conn = conn
        self.scale = self.manager.settings['font_scale']

        reader = conn.reader
        self.poller = StatePoller(reader.read) if reader is not None else None
        self.status_page = StatusPage(self.poller)
        self.add_page('Status', self.status_page, hint='Sensor and loops now, read from SXM.')
        self.status_page.assessed.connect(lambda level: self.set_badge('Status', '' if level in (T.OK, T.INFO) else level))
        if self.poller is not None:
            self.poller.changed.connect(self._show_sensor)

        c = self.classic
        for group, pages in LAYOUT:
            for title, attr, hint in pages:
                self.add_page(title, getattr(c, attr), group=group, hint=hint)
        c.step_tab.tabs_widget = _TabJump(self, {0: 'Scope'})
        c.step_tab.scope_tab_index = 0

        # who uses the driver; Live Scope yields, the others refuse a second capture
        self.arbiter = Arbiter(self)
        a = self.arbiter
        a.register('Live Scope', lambda: c.live_scope_tab.capture_thread is not None, c.live_scope_tab.stop_live)
        a.register('Scope capture', lambda: c.scope_tab.capture_thread is not None and c.scope_tab.capture_thread.isRunning())
        a.register('Single Step', c.single_step_tab.is_running)
        a.register('Tuning run', lambda: c.tuning_tab.runner_active() or c.tuning_tab.auto_active())
        a.register('Sensor characterisation', c.characterisation_tab.runner_active)
        driver_gate.set_arbiter(a)

        self.reconnect_btn = QtWidgets.QPushButton('Reconnect')
        self.reconnect_btn.setToolTip('Close and reopen the DDE connection and the driver handle (stale after a long '
                                      'session or an SXM restart). Running captures are stopped first.')
        self.reconnect_btn.clicked.connect(self.reconnect)
        self.add_strip_widget(self.reconnect_btn)

        view = self.view_menu
        self.contrast_action = view.addAction('High contrast')
        self.contrast_action.setCheckable(True)
        self.contrast_action.setChecked(bool(self.manager.settings.get('high_contrast')))
        self.contrast_action.toggled.connect(self.manager.set_high_contrast)
        self.dark_action.blockSignals(True)
        self.dark_action.setChecked(bool(self.manager.settings.get('dark_mode')))
        self.dark_action.blockSignals(False)
        self.dark = self.dark_action.isChecked()
        T._dark = self.dark
        help_menu = self.menuBar().addMenu('&Help')
        help_menu.addAction('Safety: units and the ±10 V limit', c.show_safety_details)
        help_menu.addAction('Accessibility help', c.show_accessibility_help).setShortcut(QtGui.QKeySequence('F1'))

        self.manager.settings_changed.connect(self._apply_settings)
        self._apply_settings()
        self._tick = QtCore.QTimer(self)
        self._tick.timeout.connect(self.show_status)
        self._tick.start(500)
        self.show_status()
        self.resize(c.size())
        if read_at_start and self.poller is not None:
            QtCore.QTimer.singleShot(0, self.status_page.refresh)

    # -- status strip --

    def show_status(self):
        if self.conn.is_offline:
            self.set_status('conn', 'OFFLINE (mock DDE / no driver)', T.WARN,
                            'Nothing is sent to the instrument. Start SXM, then Reconnect.')
        else:
            self.set_status('conn', 'ONLINE (real SXM)', T.OK)
        users = self.arbiter.active()
        self.set_status('driver', 'driver: ' + (', '.join(users) if users else 'free'), T.WARN if users else '',
                        'Captures share one driver handle; a new one stops Live Scope and waits for the others.')

    def _show_sensor(self, readout):
        f0, q = readout.get('f_peak'), readout.get('q')
        if f0 and q:
            self.set_status('sensor', f'f0 {f0:,.2f} Hz   Q {q:,.0f}', '', 'From the DNC status bar (last sweep).')
        else:
            self.set_status('sensor', 'f0 / Q: not shown by SXM', T.WARN, 'Run a resonance sweep in the DNC window.')

    def reconnect(self):
        self.reconnect_btn.setEnabled(False)
        try:
            self.classic.reconnect_hardware()
        finally:
            self.reconnect_btn.setEnabled(True)
        self.show_status()
        self.log('Reconnected: ' + ('offline' if self.conn.is_offline else 'online'))
        self.status_page.refresh()

    # -- fonts and colours: the program's own manager --

    def _set_scale(self, scale: float):
        options = self.manager.scale_options
        name = min(options, key=lambda n: abs(options[n] - scale))
        self.scale = options[name]
        self.manager.set_font_scale(name)

    def step_scale(self, direction: int):
        values = sorted(self.manager.scale_options.values())
        now = self.manager.settings['font_scale']
        i = min(range(len(values)), key=lambda k: abs(values[k] - now))
        self._set_scale(values[max(0, min(len(values) - 1, i + direction))])

    def _apply_settings(self, _settings=None):
        font = self.manager.get_scaled_font()
        for w in (self, self.nav, self.strip, self.menuBar(), self.log_view):
            w.setFont(font)
        for w in self.strip.findChildren(QtWidgets.QWidget):
            w.setFont(font)
        self.manager.apply_to_widget(self.status_page)
        self.dark = bool(self.manager.settings.get('dark_mode'))
        T._dark = self.dark
        self._style()
        for key, chip in self._chips.items():
            chip.setFont(font)
        if self.status_page.values:
            self.status_page.findings.set_findings(R.assess(self.status_page.values, self.status_page.previous))

    # -- shutdown --

    def closeEvent(self, event):
        self._tick.stop()
        if self.poller is not None:
            self.poller.shutdown()
        driver_gate.set_arbiter(None)
        self.classic.closeEvent(QtGui.QCloseEvent())           # stops captures, restores gains, closes the driver
        super().closeEvent(event)

