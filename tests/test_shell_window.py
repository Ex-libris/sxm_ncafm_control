"""Status rules, the Status page, the driver gate and the shell window (offline, SXM closed)."""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from sxm_ncafm_control.gui import driver_gate  # noqa: E402
from sxm_ncafm_control.gui import status_rules as R  # noqa: E402
from sxm_ncafm_control.sxm_state import SXMReadout  # noqa: E402

try:
    import sxm_ui  # noqa: F401
    HAVE_UI = True
except ImportError:
    HAVE_UI = False

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

GOOD = {"f_peak": 30000.0, "q": 50000.0, "ring_down_s": 0.53, "used_freq": 30000.05, "drive": 1e-4,
        "amp_ref": 1.0, "amp_kp": 1e9, "amp_ki": 1e5, "amp_tau_s": 0.0167, "afl_output_gain": 0.1,
        "pll_kp": 0.0, "pll_ki": 0.0, "dnc_time_constant_s": 1e-3, "dnc_rolloff": "12 dB", "input_gain_ina": 1.0,
        "feedback_off": True, "feedback_mode": "AFM", "topo_ref": 1.0, "topo_kp": 1.0, "topo_ki": 1.0}


def titles(found, level=None):
    return [f["title"] for f in found if level is None or f["level"] == level]


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        APP.processEvents()
        QtCore.QThread.msleep(5)
    return cond()


class Rules(unittest.TestCase):
    def test_nothing_readable(self):
        found = R.assess({k: None for k in GOOD})
        self.assertEqual((len(found), found[0]["level"]), (1, R.BAD))

    def test_healthy_state_has_no_warnings(self):
        found = R.assess(dict(GOOD))
        self.assertEqual(titles(found, R.WARN) + titles(found, R.BAD), [])
        self.assertIn("`use` is on the resonance", titles(found, R.OK))
        self.assertTrue(any("1× the manual" in t for t in titles(found)), titles(found))   # Ki = 5e8/Q/0.1

    def test_use_off_resonance_with_the_pll_off(self):
        found = R.assess(dict(GOOD, used_freq=30002.0))
        self.assertTrue(any("+2.000 Hz from the peak" in t for t in titles(found, R.WARN)), titles(found))
        on = R.assess(dict(GOOD, used_freq=30002.0, pll_kp=-200.0, pll_ki=-1.65e4))
        self.assertFalse(any("from the peak" in t for t in titles(on)))            # the PLL moves the drive itself

    def test_gains_far_above_the_rule_and_missing_q(self):
        self.assertTrue(any("10×" in t for t in titles(R.assess(dict(GOOD, amp_ki=1e6)), R.WARN)))
        found = R.assess(dict(GOOD, q=None, f_peak=None))
        self.assertIn("Q and fPeak are not shown in the DNC status bar", titles(found, R.WARN))

    def test_feedback_on_and_loops_off(self):
        found = R.assess(dict(GOOD, feedback_off=False, amp_kp=0.0, amp_ki=0.0))
        self.assertIn("z feedback is ON", titles(found, R.WARN))
        self.assertIn("Amplitude loop is off (Kp = Ki = 0)", titles(found, R.INFO))
        self.assertIn("PLL is off (Kp = Ki = 0)", titles(found, R.INFO))

    def test_changes_since_the_last_read(self):
        found = R.assess(dict(GOOD, amp_ki=3e5), GOOD)
        self.assertIn("Amplitude loop Ki 100000 → 300000", found[0]["evidence"])
        self.assertEqual(R.changes(GOOD, dict(GOOD)), [])

    def test_show(self):
        self.assertEqual(R.show("amp_tau_s", 0.0167, "s"), "16.7 ms")
        self.assertEqual(R.show("feedback_off", True), "OFF")
        self.assertEqual(R.show("q", None), "n/a")


class Gate(unittest.TestCase):
    def tearDown(self):
        driver_gate.set_arbiter(None)
        driver_gate.warn = None

    def test_allows_without_an_arbiter(self):
        self.assertTrue(driver_gate.allow(None, "Scope capture"))

    @unittest.skipUnless(HAVE_UI, "sxm_ui not next to the package")
    def test_refuses_with_the_holders_name(self):
        from sxm_ui.session import Arbiter
        a = Arbiter()
        a.register("Tuning run", lambda: True)
        driver_gate.set_arbiter(a)
        said = []
        driver_gate.warn = said.append
        self.assertFalse(driver_gate.allow(None, "Scope capture"))
        self.assertIn("Tuning run", said[0])
        self.assertTrue(driver_gate.allow(None, "Tuning run"))


class FakeReader:
    def __init__(self):
        self.values = dict(GOOD)

    def read(self, sections=None):
        return SXMReadout(dict(self.values), {}, {})


@unittest.skipUnless(HAVE_UI, "sxm_ui not next to the package")
class Window(unittest.TestCase):
    def test_pages_status_and_gate(self):
        from sxm_ncafm_control.connection import SXMConnection
        from sxm_ncafm_control.gui.shell_window import LAYOUT, ShellWindow
        conn = SXMConnection()
        conn.reader = FakeReader()
        win = ShellWindow(conn, read_at_start=False)
        try:
            self.assertEqual(win.current_page(), "Status")
            self.assertEqual(len(win.pages), 1 + sum(len(p) for _g, p in LAYOUT))
            self.assertEqual(win.classic.tabs.count(), 0)                      # every tab became a page
            self.assertIn("OFFLINE", win.status_text("conn"))
            self.assertEqual(win.status_text("driver"), "driver: free")

            win.status_page.refresh()
            self.assertTrue(wait_for(lambda: win.status_page.values))
            self.assertIn("50,000", win.status_text("sensor"))
            self.assertIn("50000", win.status_page.labels["q"].text())
            conn.reader.values["amp_ki"] = 3e5
            win.status_page.refresh()
            self.assertTrue(wait_for(lambda: "was" in win.status_page.labels["amp_ki"].text()))
            self.assertIn("was 100000", win.status_page.labels["amp_ki"].text())
            self.assertTrue(any("changed since" in t for t in win.status_page.findings.texts()))

            win.classic.step_tab.tabs_widget.setCurrentIndex(win.classic.step_tab.scope_tab_index)
            self.assertEqual(win.current_page(), "Scope")                      # Step Test still jumps to the Scope

            said = []
            driver_gate.warn = said.append
            tuning = win.classic.tuning_tab
            tuning.runner_active = lambda: True                                 # as if a sheet were running
            win.show_status()
            self.assertEqual(win.status_text("driver"), "driver: Tuning run")
            before = win.classic.scope_tab.capture_thread
            win.classic.scope_tab.start_capture()
            self.assertIs(win.classic.scope_tab.capture_thread, before)        # refused, nothing started
            self.assertIn("Tuning run", said[0])
            del tuning.runner_active

            win.step_scale(+1)
            win.step_scale(-1)
        finally:
            driver_gate.warn = None
            win.close()
        self.assertTrue(driver_gate.allow(None, "Scope capture"))              # the gate went with the window


if __name__ == "__main__":
    unittest.main()
