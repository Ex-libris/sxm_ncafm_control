"""Sensor Characterisation tab: gating, a simulated run end to end (with analysis), Stop. Offscreen, no hardware."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets                                       # noqa: E402

from sxm_ncafm_control.characterisation import DEFAULT_CONFIG     # noqa: E402
from sxm_ncafm_control.gui.characterisation_tab import CharacterisationTab   # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def small_config(folder: Path) -> Path:
    d = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    d['design']['repeats'] = 1
    d['factors'].update(drives=[5e-6, 1e-5], tcs=['1 ms', '20 ms'], baseline_tcs=['1 ms'])
    d['timing'].update(off_s=15.0, baseline_s=10.0)
    p = folder / 'small.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    return p


class Tab(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tab = CharacterisationTab(driver=None)
        self.tab.load_config(str(small_config(Path(self.tmp.name))))
        self.tab.results_dir = Path(self.tmp.name) / 'results'
        self.tab.target.setCurrentText("Simulation")

    def tearDown(self):
        if self.tab.thread is not None:
            self.tab.thread.wait(30000)
        self.tmp.cleanup()

    def finish(self):
        self.assertTrue(self.tab.thread.wait(120000))
        app.processEvents()

    def test_needs_both_confirmations_and_a_driver(self):
        self.assertFalse(self.tab.b_run.isEnabled())
        self.tab.afl_check.setChecked(True)
        self.assertFalse(self.tab.b_run.isEnabled())
        self.tab.surface_check.setChecked(True)
        self.assertTrue(self.tab.b_run.isEnabled())
        self.tab.target.setCurrentText("SXM instrument")          # offline: no driver
        self.assertFalse(self.tab.b_run.isEnabled())
        self.assertIn("driver", self.tab.status.text())

    def test_simulated_run_completes_and_is_analysed(self):
        self.tab.afl_check.setChecked(True)
        self.tab.surface_check.setChecked(True)
        self.tab.start('run')
        self.finish()
        self.assertIn("COMPLETE", self.tab.status.text())
        self.assertGreaterEqual(self.tab.table.rowCount(), 8)      # headroom + block 1
        reports = list(self.tab.run_dir.glob('derived/*/report.md'))
        self.assertEqual(len(reports), 1)
        self.assertTrue(self.tab.b_run.isEnabled())
        self.assertTrue(str(self.tab.run_dir).startswith(str(self.tab.results_dir)))

    def test_stop_restores(self):
        self.tab.afl_check.setChecked(True)
        self.tab.surface_check.setChecked(True)
        self.tab.start('run')
        self.tab.stop()
        self.finish()
        self.assertIn("STOPPED", self.tab.status.text())


if __name__ == '__main__':
    unittest.main()
