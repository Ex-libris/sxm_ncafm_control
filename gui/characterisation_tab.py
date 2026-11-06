"""Sensor Characterisation tab: the qPlus open-loop campaign (characterisation/) inside the app.

The campaign runs on a worker thread with the app's shared driver handle (``conn.driver``). It writes only the
DNC drive and the DNC lock-in TimeConstant, through AnfatecSXMWriter (thread-safe Win32 messages, read back);
no DDE, so nothing has to stay on the GUI thread. The two things Python cannot read from SXM (AFL switched
off, no surface) are confirmed here with tick boxes instead of editing the config file.

Buttons: Preflight (read-backs + drive echo, ~15 s), Run (headroom + the next N blocks), Resume run...
(continue a run folder), Stop (ends the record at once; drive -> 0, TC -> start value), Analyse run...
"""
from __future__ import annotations

import datetime
import json
import os
import traceback
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from ..characterisation import DEFAULT_CONFIG, RESULTS_DIR
from ..characterisation import config as C
from ..common import append_log_line

COLUMNS = ['#', 'kind', 'block', 'drive', 'TC', 'status', 'A_ss (V)', 'tau (s)', 'baseline (V)']
FINISHED_OK = ('complete', 'batch done', 'preflight')


class CampaignThread(QtCore.QThread):
    """Runs ``job(progress, stop)`` off the GUI thread; its events arrive through ``event``."""
    event = QtCore.pyqtSignal(str, object)

    def __init__(self, job):
        super().__init__()
        self.job, self._stop = job, False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.job(self.event.emit, lambda: self._stop)
        except BaseException:                      # noqa: BLE001 - an exception escaping run() kills PyQt
            self.event.emit('error', traceback.format_exc())


def _fmt(x, spec='.4g') -> str:
    return '' if x is None else format(x, spec)


class CharacterisationTab(QtWidgets.QWidget):
    def __init__(self, driver, parent=None):
        super().__init__(parent)
        self.driver = driver
        self.thread: CampaignThread | None = None
        self.cfg_path = DEFAULT_CONFIG
        self.cfg = C.load(DEFAULT_CONFIG)
        self.results_dir = RESULTS_DIR
        self.run_dir: Path | None = None
        self._build()
        self._show_config()
        self._update_enabled()

    # ------------------------------------------------------------------------------------------------ UI

    def _build(self):
        v = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "qPlus sensor by itself (no surface), open loop: zero-drive baselines and drive x DNC TimeConstant, "
            "in shuffled blocks with repeats. Writes only the DNC Drive and TimeConstant (read back); never z, "
            "gains or loop switches. Details: characterisation/README.md.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        g = QtWidgets.QGroupBox("Campaign")
        gl = QtWidgets.QGridLayout(g)
        self.cfg_label = QtWidgets.QLabel()
        self.cfg_label.setWordWrap(True)
        self.cfg_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        b_load = QtWidgets.QPushButton("Load config...")
        b_load.clicked.connect(self.load_config)
        b_default = QtWidgets.QPushButton("Default")
        b_default.clicked.connect(lambda: self.load_config(str(DEFAULT_CONFIG)))
        gl.addWidget(self.cfg_label, 0, 0, 1, 3)
        gl.addWidget(b_load, 0, 3)
        gl.addWidget(b_default, 0, 4)
        gl.addWidget(QtWidgets.QLabel("Run on:"), 1, 0)
        self.target = QtWidgets.QComboBox()
        self.target.addItems(["SXM instrument", "Simulation"])
        self.target.currentIndexChanged.connect(self._update_enabled)
        gl.addWidget(self.target, 1, 1)
        gl.addWidget(QtWidgets.QLabel("Blocks this run:"), 1, 2)
        self.blocks = QtWidgets.QSpinBox()
        self.blocks.setRange(1, 99)
        self.blocks.setToolTip("Each block is every drive x TC cell once (~25 min with the default config). "
                               "Run one, then 'Resume run...' for the next.")
        gl.addWidget(self.blocks, 1, 3)
        gl.addWidget(QtWidgets.QLabel("Results folder:"), 2, 0)
        self.folder_label = QtWidgets.QLabel()
        self.folder_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        gl.addWidget(self.folder_label, 2, 1, 1, 3)
        b_folder = QtWidgets.QPushButton("Choose...")
        b_folder.clicked.connect(self.choose_folder)
        gl.addWidget(b_folder, 2, 4)
        v.addWidget(g)

        c = QtWidgets.QGroupBox("Confirm in SXM (Python cannot read these)")
        cl = QtWidgets.QVBoxLayout(c)
        self.afl_check = QtWidgets.QCheckBox("Amplitude feedback is switched OFF")
        self.afl_check.setToolTip("The AFL On/Off switch is not readable. The preflight also checks it physically: "
                                  "the Drive DAC must follow the drive in proportion and stay flat.")
        self.surface_check = QtWidgets.QCheckBox("Tip retracted, no surface")
        for cb in (self.afl_check, self.surface_check):
            cb.toggled.connect(self._update_enabled)
            cl.addWidget(cb)
        note = QtWidgets.QLabel("PLL Kp = Ki = 0 is read back (the run is refused otherwise). Stop Scope, Live "
                                "Scope and Single Step captures first: they share the driver.")
        note.setWordWrap(True)
        cl.addWidget(note)
        v.addWidget(c)

        row = QtWidgets.QHBoxLayout()
        self.b_preflight = QtWidgets.QPushButton("Preflight")
        self.b_preflight.setToolTip("Read-backs, refusals, capture rate and the drive echo (~15 s); "
                                    "puts the drive and TC back.")
        self.b_run = QtWidgets.QPushButton("Run")
        self.b_resume = QtWidgets.QPushButton("Resume run...")
        self.b_stop = QtWidgets.QPushButton("Stop")
        self.b_analyse = QtWidgets.QPushButton("Analyse run...")
        self.b_open = QtWidgets.QPushButton("Open run folder")
        self.b_preflight.clicked.connect(lambda: self.start('preflight'))
        self.b_run.clicked.connect(lambda: self.start('run'))
        self.b_resume.clicked.connect(self.resume)
        self.b_stop.clicked.connect(self.stop)
        self.b_analyse.clicked.connect(self.analyse)
        self.b_open.clicked.connect(self.open_folder)
        for b in (self.b_preflight, self.b_run, self.b_resume, self.b_stop, self.b_analyse, self.b_open):
            row.addWidget(b)
        v.addLayout(row)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        v.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Idle.")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        split.addWidget(self.table)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        split.addWidget(self.log)
        split.setSizes([300, 200])
        v.addWidget(split, 1)

    def _show_config(self):
        cfg = self.cfg
        n_cells = len(cfg.factors.drives) * len(cfg.factors.tcs) + len(cfg.factors.baseline_tcs)
        block = C.build_schedule(cfg, cfg.factors.drives)
        per_block = C.estimate_minutes(cfg, [e for e in block if e['block'] == 1])
        self.cfg_label.setText(
            f"<b>{cfg.name}</b> ({self.cfg_path.name}): drives {cfg.factors.drives} (SXM units), "
            f"TC {cfg.factors.tcs}, baselines at {cfg.factors.baseline_tcs or 'none'}; {cfg.design.repeats} blocks of "
            f"{n_cells} conditions + anchors, ~{per_block:.0f} min per block; expected Q {cfg.expect.q:g} at "
            f"{cfg.expect.f0_hz} Hz (tau {cfg.expect.tau_mech:.3f} s).")
        self.blocks.setMaximum(max(1, cfg.design.repeats))
        self.folder_label.setText(str(self.results_dir))

    def _update_enabled(self, *_):
        busy = self.runner_active()
        sim = self.target.currentText() == "Simulation"
        ready = self.afl_check.isChecked() and self.surface_check.isChecked() and (sim or self.driver is not None)
        for b in (self.b_preflight, self.b_run, self.b_resume):
            b.setEnabled(ready and not busy)
        self.b_analyse.setEnabled(not busy)
        self.b_stop.setEnabled(busy)
        self.b_open.setEnabled(self.run_dir is not None)
        if not busy:
            why = []
            if not (self.afl_check.isChecked() and self.surface_check.isChecked()):
                why.append("tick both confirmations")
            if not sim and self.driver is None:
                why.append("the SXM driver is not open (offline): reconnect, or run on Simulation")
            if why:
                self.status.setText("To start: " + "; ".join(why) + ".")

    def _log(self, text: str):
        append_log_line(self.log, text)

    # -------------------------------------------------------------------------------------------- actions

    def load_config(self, path: str | None = None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Campaign config", str(self.cfg_path.parent),
                                                            "JSON (*.json)")
            if not path:
                return
        try:
            self.cfg = C.load(path)
        except (C.ConfigError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Config", f"{path}:\n{exc}")
            return
        self.cfg_path = Path(path)
        self._show_config()
        self._log(f"config {self.cfg_path}")

    def choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Results folder", str(self.results_dir))
        if d:
            self.results_dir = Path(d)
            self.folder_label.setText(d)

    def open_folder(self):
        if self.run_dir is not None and self.run_dir.exists():
            os.startfile(str(self.run_dir))            # noqa: S606 - Windows only, like the app

    def resume(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Run folder to continue", str(self.results_dir))
        if d:
            self.start('resume', Path(d))

    def analyse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Run folder to analyse", str(self.run_dir or self.results_dir))
        if d:
            self.start('analyse', Path(d))

    def _instrument(self, cfg: C.Config):
        writable = {cfg.instrument.drive_param, cfg.instrument.tc_param}
        if self.target.currentText() == "Simulation":
            from ..characterisation.sim import SimQPlus, SimSensor
            return SimQPlus(SimSensor(f0=cfg.expect.f0_hz, q=cfg.expect.q, f_use=cfg.expect.f0_hz), writable)
        from ..characterisation.instrument import QPlusSxm
        return QPlusSxm(writable, driver=self.driver)

    def start(self, mode: str, folder: Path | None = None):
        """mode: 'preflight' | 'run' | 'resume' (folder) | 'analyse' (folder)."""
        if self.runner_active():
            return
        from ..characterisation.analyse import analyse_run
        from ..characterisation.runner import Campaign

        if mode == 'analyse':
            def job(progress, stop):
                progress('stage', f'analysing {folder}')
                s = analyse_run(folder)
                progress('analysed', s)
            self.run_dir = folder
        else:
            try:
                if mode == 'resume':
                    data = json.loads((folder / 'campaign.json').read_text(encoding='utf-8'))
                    cfg = C.from_dict(data['config'])
                    run_dir = folder
                else:
                    cfg = C.from_dict(self.cfg.to_dict())
                    sim = 'sim_' if self.target.currentText() == "Simulation" else ''
                    run_dir = self.results_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{sim}{cfg.name}"
                cfg.operator.afl_off_confirmed = self.afl_check.isChecked()
                cfg.operator.no_surface_confirmed = self.surface_check.isChecked()
                inst = self._instrument(cfg)
            except Exception as exc:                                    # noqa: BLE001 - shown, not raised
                QtWidgets.QMessageBox.warning(self, "Cannot start", f"{type(exc).__name__}: {exc}")
                return
            self.run_dir = run_dir
            blocks = self.blocks.value()

            def job(progress, stop):
                try:
                    out = Campaign(cfg, inst, run_dir, progress, stop).run(
                        resume=(mode == 'resume'), preflight_only=(mode == 'preflight'),
                        max_blocks=None if mode == 'preflight' else blocks)
                finally:
                    inst.close()
                if out['status'] in ('complete', 'batch done'):
                    progress('stage', 'analysing the run')
                    try:
                        out['analysis'] = analyse_run(run_dir)['derived_dir']
                    except Exception as exc:                            # noqa: BLE001 - the run itself is saved
                        out['analysis_error'] = f'{type(exc).__name__}: {exc}'
                progress('finished', out)
            self.table.setRowCount(0)
            self.progress.setRange(0, 0)
        self._log(f"--- {mode} {self.run_dir} ---")
        self.thread = CampaignThread(job)
        self.thread.event.connect(self._on_event)
        self.thread.finished.connect(self._update_enabled)
        self.thread.start()
        self.status.setText(f"{mode}: running...")
        self._update_enabled()

    def runner_active(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def stop(self, wait_ms: int = 0):
        """Ask the campaign to stop: the record ends, drive -> off, TC -> start (on the worker thread).
        ``wait_ms`` > 0 blocks until it has (used when the window closes or the driver is reconnected)."""
        if self.runner_active():
            self.thread.stop()
            self.status.setText("Stopping: drive -> off, TC -> start value...")
            if wait_ms:
                self.thread.wait(wait_ms)

    # ---------------------------------------------------------------------------------------------- events

    def _on_event(self, event: str, data):
        if event == 'log':
            self._log(data)
        elif event == 'stage':
            self.status.setText(f"Stage: {data}")
        elif event == 'progress':
            self.progress.setRange(0, data['of'])
            self.progress.setValue(data['n'] - 1)
            self.status.setText(f"Condition {data['n']} of {data['of']}; about {data['eta_s'] / 60:.0f} min left "
                                "in this batch.")
        elif event == 'condition_done':
            d = data
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [d['idx'], d['kind'], d['block'], '0' if d['drive'] is None else f"{d['drive']:g}", d['tc'],
                    d['status'], _fmt(d.get('amp_ss')), _fmt(d.get('tau_s'), '.4f'), _fmt(d.get('baseline_mean'), '.3g')]
            for col, val in enumerate(vals):
                self.table.setItem(r, col, QtWidgets.QTableWidgetItem(str(val)))
            self.table.scrollToBottom()
            self.progress.setValue(self.progress.value() + 1)
        elif event == 'finished':
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            st = data['status']
            colour = '#2e8b57' if st in FINISHED_OK else '#b36b00' if st in ('stopped', 'refused') else '#c0392b'
            text = f"<b style='color:{colour}'>{st.upper()}</b>: {data['message']}"
            if data.get('analysis'):
                text += f"<br>Report: {Path(data['analysis']) / 'report.md'}"
            if data.get('analysis_error'):
                text += f"<br>Analysis failed: {data['analysis_error']}"
            if st in ('batch done', 'stopped', 'aborted'):
                text += "<br>Continue with 'Resume run...' on this run folder."
            self.status.setText(text)
            for r in data.get('restore', []):
                self._log(f"restore {r['name']} -> {r['value']!r}: {'ok' if r['ok'] else 'FAILED ' + r.get('error', '')}")
        elif event == 'analysed':
            self.progress.setRange(0, 1)
            self.status.setText(f"Analysed: {Path(data['derived_dir']) / 'report.md'}")
        elif event == 'error':
            self.progress.setRange(0, 1)
            self._log(data)
            self.status.setText("<b style='color:#c0392b'>ERROR</b>: see the log (the run folder keeps what was saved).")
