"""
SXM read-back (sxm_state.py) and the tabs that use it. The Win32 GUI is replaced by a fake bridge backend
whose values are a real discovery dump of Femto_28_4 (2026-09-30). Offscreen Qt; no hardware.
"""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets                                              # noqa: E402

from sxm_anfatec.bridge import AnfatecSXMBridge                    # noqa: E402
from sxm_ncafm_control.sxm_state import SXMReader, parse_gain_v, parse_time_s   # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _edits(*values):
    return [{"value": v, "rect": [-1134, 1435 + 25 * i, 66, 21]} for i, v in enumerate(values)]


GUI = {
    "scan": {"Range": 50.0, "Speed": 2.4, "Pixel": 180.0, "x-Center": -957.61, "y-Center": 128.23, "Angle": 0.0},
    "topography": {"Ref": 0.0, "Ki": 0.0, "Kp": 0.0},
    "amplitude": {"Ref": 0.5, "Ki": 5e4, "Kp": 5e8, "Pull back at": 0.0, "Pull Speed": 80.0, "Tau": "2 ms"},
    "pll": {"Kp": -100.0, "Ki": -1e4},
    "zcontrol": {"Feedback Off": False, "dz": 0.0, "dz per Mouse Tick": 10.0, "Slew Rate": 0.155},
    "lockin": {"TimeConstant": "1 ms", "RollOff": "6 db/oct",
               **{f"Lia{n}": {"Link": "no link", "Value1": 0.0, "Phase": 0.0} for n in (1, 2, 3)}},
    "spectroscopy": {"X": "?", "Y": "?", "Delay1": 2000.0, "AguT": 120.01, "dz": 0.0, "U Start": -1500.0,
                     "U Stop": 1000.0, "Mode": "X(U)", "Acquire": ["It_to_PC", "Topo"]},
    "dynamic_non_contact": {"Input Gain InA": "1", "TimeConstant": "1 ms", "RollOff": "6 db/oct",
                            "Range": "±0.1", "Status": "Q: 148699 | fPeak: 25562 Hz | tau: 1851628 µs",
                            # stored bottom-up on purpose: the reader must order by screen position
                            "unmapped_numeric_controls": list(reversed(_edits(25560.0, 25566.0, 25562.49, 0.25)))},
    "feedback_mode": {"Mode": "STM + PLL AFM"},
    "scanner_state": {"caption": "Scanner"},
    "tip_conditioning": {"Mode": "Tip Dive", "unmapped_numeric_controls": []},
    "oscilloscope_state": {"Channels": ["QPlusAmpl"], "x_axis_values": [2.0], "selected_modes": ["time"]},
}


class FakeGUI:
    """Bridge backend: one method per section, as Win32Backend. ``broken`` sections raise like a missing form."""

    def __init__(self, gui=None, broken=()):
        self.gui = copy.deepcopy(gui or GUI)
        self.broken = set(broken)

    def __getattr__(self, method):
        if method not in self.gui:
            raise AttributeError(method)

        def read():
            if method in self.broken:
                raise RuntimeError(f"Expected one form for {method}, found 0")
            return copy.deepcopy(self.gui[method])
        return read


def settle(tab):
    """Wait for the Suggested tab's background SXM read and let its result be applied."""
    from PyQt5 import QtWidgets
    for _ in range(200):
        thread = tab._read_thread
        if thread is None:
            return
        thread.join(5)
        QtWidgets.QApplication.processEvents()
    raise AssertionError("the SXM read did not finish")


def reader(gui=None, broken=()):
    fake = FakeGUI(gui, broken)
    return SXMReader(AnfatecSXMBridge(strict=False, backend=fake)), fake


class Parsing(unittest.TestCase):
    def test_time(self):
        self.assertAlmostEqual(parse_time_s("1 ms"), 1e-3)
        self.assertAlmostEqual(parse_time_s("100 µs"), 1e-4)
        self.assertAlmostEqual(parse_time_s("100 μs"), 1e-4)
        self.assertAlmostEqual(parse_time_s("2 s"), 2.0)
        self.assertIsNone(parse_time_s("6 db/oct"))
        self.assertIsNone(parse_time_s(""))

    def test_gain(self):
        self.assertEqual(parse_gain_v("±0.1"), 0.1)
        self.assertEqual(parse_gain_v("±10"), 10.0)
        self.assertIsNone(parse_gain_v("off"))


class Reader(unittest.TestCase):
    def test_maps_every_registry_parameter(self):
        r = reader()[0].read()
        self.assertTrue(r.ok)
        expect = {"amp_ref": 0.5, "amp_ki": 5e4, "amp_kp": 5e8, "pll_kp": -100.0, "pll_ki": -1e4,
                  "used_freq": 25562.49, "drive": 0.25, "dnc_sweep_start": 25560.0, "dnc_sweep_stop": 25566.0}
        for k, v in expect.items():
            self.assertEqual(r.get(k), v, k)
        self.assertEqual(r.by_code("EDIT", "Edit24"), 5e4)
        self.assertEqual(r.by_code("DNC", 3), 25562.49)
        self.assertEqual(r.by_code("dnc", "4"), 0.25)
        self.assertIsNone(r.by_code("EDIT", "Edit99"))                   # custom EditXX: no GUI mapping

    def test_extras(self):
        r = reader()[0].read()
        self.assertEqual(r.get("afl_output_gain"), 0.1)
        self.assertEqual(r.get("input_gain_ina"), 1.0)
        self.assertAlmostEqual(r.get("dnc_time_constant_s"), 1e-3)
        self.assertAlmostEqual(r.get("amp_tau_s"), 2e-3)
        self.assertEqual(r.get("q"), 148699.0)
        self.assertEqual(r.get("f_peak"), 25562.0)
        self.assertAlmostEqual(r.get("ring_down_s"), 1.851628)
        self.assertIs(r.get("feedback_off"), False)
        self.assertEqual(r.get("feedback_mode"), "STM + PLL AFM")
        self.assertIn("Multi Channel LockIn", r.raw)
        self.assertEqual(r.errors, {})

    def test_dnc_layout_change_refuses_to_guess(self):
        gui = copy.deepcopy(GUI)
        gui["dynamic_non_contact"]["unmapped_numeric_controls"] = _edits(25560.0, 25566.0, 25562.49)
        r = reader(gui)[0].read()
        for k in ("used_freq", "drive", "dnc_sweep_start", "dnc_sweep_stop"):
            self.assertIsNone(r.get(k))
            self.assertIn("refusing to guess", r.errors[k])
        self.assertEqual(r.get("amp_ki"), 5e4)                             # the rest still reads
        self.assertEqual(r.get("q"), 148699.0)

    def test_closed_window_only_loses_its_own_values(self):
        r = reader(broken={"dynamic_non_contact"})[0].read()
        self.assertTrue(r.ok)
        self.assertIsNone(r.get("used_freq"))
        self.assertIsNone(r.get("q"))
        self.assertIn("found 0", r.errors["q"])
        self.assertEqual(r.get("pll_kp"), -100.0)

    def test_sxm_not_running(self):
        r = reader(broken=set(GUI))[0].read()
        self.assertFalse(r.ok)
        self.assertTrue(all(v is None for v in r.values.values()))

    def test_each_read_is_live(self):
        rd, fake = reader()
        self.assertEqual(rd.read().get("pll_ki"), -1e4)
        fake.gui["pll"]["Ki"] = -2e4
        self.assertEqual(rd.read().get("pll_ki"), -2e4)


class Tabs(unittest.TestCase):
    def test_params_tab_fills_current_and_flags_a_write_that_did_not_land(self):
        from sxm_ncafm_control.common import PARAMS_BASE
        from sxm_ncafm_control.gui.params_tab import ParamsTab
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        rd, fake = reader()
        tab = ParamsTab(FakeInstrument(), reader=rd)
        rows = {k: i for i, (k, *_r) in enumerate(PARAMS_BASE)}
        tab.read_from_sxm(quiet=True)
        self.assertEqual(tab.table.item(rows["used_freq"], 3).text(), "25562.49")
        self.assertEqual(tab.table.item(rows["amp_kp"], 3).text(), "5e8")
        # a write SXM took, and one it did not
        tab._pending_verify = {rows["pll_kp"]: -100.0, rows["pll_ki"]: -3e4}
        tab.read_from_sxm(quiet=True)
        log = tab.log_widget.toPlainText()
        self.assertIn("PLL Ki: sent -30000.0, but SXM shows -10000.0", log)
        self.assertNotIn("PLL Kp: sent", log)
        self.assertEqual(tab._pending_verify, {})

    def test_tuning_tab_takes_the_anchor_from_sxm_without_rescaling_it(self):
        from sxm_ncafm_control.gui import runsheet_tab as T
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        rd, fake = reader()
        tab = T.RunSheetTab(FakeInstrument(), None, reader=rd)
        self.assertEqual(tab.anchor_og.currentData(), 0.1)
        self.assertEqual((tab.kp0.value(), tab.ki0.value()), (5e8, 5e4))            # as read, not x10
        self.assertEqual(tab.current_settings()["amp_tau_ms"], 2.0)
        self.assertEqual(tab.current_settings()["dnc_rolloff"], "6")
        fake.gui["amplitude"]["Ki"] = 7e4                                           # someone edits SXM by hand
        self.assertTrue(tab.refresh_from_sxm(quiet=True))
        self.assertEqual(tab.ki0.value(), 7e4)                                      # an empty sheet follows SXM
        tab.add_ramp()
        fake.gui["amplitude"]["Ki"] = 9e4
        self.assertTrue(tab.refresh_from_sxm(quiet=True))
        self.assertEqual(tab.ki0.value(), 7e4)                                      # a built sheet keeps its anchor

    def test_tuning_tab_without_sxm(self):
        from sxm_ncafm_control.gui import runsheet_tab as T
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        tab = T.RunSheetTab(FakeInstrument(), None, reader=reader(broken=set(GUI))[0])
        kp = tab.kp0.value()
        self.assertFalse(tab.refresh_from_sxm(quiet=True))
        self.assertEqual(tab.kp0.value(), kp)
        self.assertTrue(any("SXM not read" in t for _, t in tab.checks()))

    def test_step_test_base_and_suggested_q_f0(self):
        from sxm_ncafm_control.gui.step_test_tab import StepTestTab
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        rd, _ = reader()
        st = StepTestTab(FakeInstrument(), reader=rd)
        st.param.setCurrentIndex(st.param.findText("Drive"))
        self.assertTrue(st.base_from_sxm())
        self.assertEqual(st.base.value(), 0.25)
        st.low.setValue(1.0)                                 # a base read from SXM no longer follows the midpoint
        self.assertEqual(st.base.value(), 0.25)

        sg = SuggestedTab(FakeInstrument(), None, reader=rd)
        self.assertTrue(sg.read_from_sxm(quiet=True))
        self.assertEqual((sg.q_val.value(), sg.f0_val.value(), sg.out_gain.currentData()), (148699.0, 25562.0, 0.1))

    def test_suggested_takes_its_inputs_from_sxm_or_by_hand(self):
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        rd, _ = reader()
        sg = SuggestedTab(FakeInstrument(), None, reader=rd)
        self.assertTrue(sg.chk_from_sxm.isChecked())                      # the default when SXM can be read
        sg.show()                                                         # opening the tab reads SXM
        self.assertIn("Reading SXM", sg.sxm_status.text())                # in the background
        self.assertFalse(sg.btn_read_sxm.isEnabled())
        settle(sg)
        self.assertTrue(sg.btn_read_sxm.isEnabled())
        self.assertEqual((sg.q_val.value(), sg.f0_val.value(), sg.out_gain.currentData()), (148699.0, 25562.0, 0.1))
        self.assertFalse(sg.q_val.isEnabled() or sg.f0_val.isEnabled() or sg.out_gain.isEnabled())
        self.assertTrue(sg.bw_pll.isEnabled())                            # a target: always editable
        self.assertEqual(sg.bw_pll.value(), 100.0)                        # 1 / (10 * TimeConstant), TimeConstant 1 ms
        self.assertIn("From SXM", sg.sxm_status.text())
        # what SXM is set to now, next to the suggestions (Tau 2 ms, TimeConstant 1 ms, ring-down 1851.628 ms)
        self.assertEqual({k: w.text() for k, w in sg.sxm_now.items()},
                         {"ki": "5e4", "kp": "5e8", "tau": "2", "tpll": "1", "ring": "1851.63"})
        self.assertAlmostEqual(float(sg.ki_out.text()), 5e8 / 148699.0 * 10, delta=1.0)   # recalculated: +-0.1 V range

        sg.chk_from_sxm.setChecked(False)                                 # by hand
        self.assertTrue(sg.q_val.isEnabled() and sg.f0_val.isEnabled() and sg.out_gain.isEnabled())
        sg.q_val.setValue(1000.0)
        for w in sg.sxm_now.values():
            w.clear()
        sg.hide(); sg.show(); settle(sg)                                  # reopening does not read over it
        self.assertEqual(sg.q_val.value(), 1000.0)
        self.assertIn("typed by hand", sg.sxm_status.text())
        self.assertEqual(sg.sxm_now["tau"].text(), "2")                   # but 'SXM now' is refreshed

        sg.chk_from_sxm.setChecked(True); settle(sg)                      # back to SXM: read at once
        self.assertEqual(sg.q_val.value(), 148699.0)

        sg.bw_pll.setValue(30.0)                                          # a typed target is not read over
        sg.hide(); sg.show(); settle(sg)
        self.assertEqual(sg.bw_pll.value(), 30.0)
        self.assertEqual(sg.tpll_out.text(), f"{1000.0 / 300.0:.6g}")
        self.assertNotIn("PLL bandwidth", sg.sxm_status.text())
        sg.close()

    def test_suggested_reads_only_its_own_windows_and_off_the_gui_thread(self):
        import threading
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        rd, fake = reader()
        asked, threads = [], set()
        plain = type(fake).__getattr__

        class Watching(type(fake)):
            def __getattr__(self, method):
                read = plain(self, method)

                def watched():
                    asked.append(method)
                    threads.add(threading.current_thread())
                    return read()
                return watched
        fake.__class__ = Watching
        sg = SuggestedTab(FakeInstrument(), None, reader=rd)
        sg.show(); settle(sg)
        self.assertEqual(sorted(set(asked)), ["amplitude", "dynamic_non_contact"])
        self.assertNotIn(threading.main_thread(), threads)
        self.assertEqual(sg.q_val.value(), 148699.0)
        sg.btn_read_sxm.click(); sg.btn_read_sxm.click()                  # a second request joins the first
        settle(sg)
        self.assertEqual(sg.sxm_now["tau"].text(), "2")
        sg.close()

    def test_a_partial_read_says_what_it_left_out(self):
        rd, _ = reader()
        ro = rd.read(sections=("Amplitude feedback",))
        self.assertEqual(ro.get("amp_tau_s"), 0.002)
        self.assertIsNone(ro.get("q"))
        self.assertIn("section not read", ro.errors["q"])
        self.assertEqual(rd.read().get("q"), 148699.0)

    def test_suggested_keeps_what_sxm_does_not_give_editable(self):
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        gui = copy.deepcopy(GUI)
        gui["dynamic_non_contact"]["Status"] = ""                         # no sweep done: no Q, no fPeak
        rd, _ = reader(gui)
        sg = SuggestedTab(FakeInstrument(), None, reader=rd)
        self.assertFalse(sg.read_from_sxm(quiet=True))
        self.assertEqual(sg.out_gain.currentData(), 0.1)                  # the output gain is still taken
        self.assertFalse(sg.out_gain.isEnabled())
        self.assertTrue(sg.q_val.isEnabled() and sg.f0_val.isEnabled())
        self.assertIn("Not read (type by hand): Q, f", sg.sxm_status.text())

        none = SuggestedTab(FakeInstrument(), None)                       # no read-back in this session
        self.assertFalse(none.chk_from_sxm.isChecked() or none.chk_from_sxm.isEnabled())
        self.assertTrue(none.q_val.isEnabled())

    def test_suggested_spectrum_values_switch_to_by_hand(self):
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        from sxm_ncafm_control.tests.fake_instrument import FakeInstrument
        rd, _ = reader()
        sg = SuggestedTab(FakeInstrument(), None, reader=rd)
        sg.read_from_sxm(quiet=True)
        sg._last_fit = (5000.0, 30000.0, 0.0)
        sg._apply_from_spectrum()
        self.assertFalse(sg.chk_from_sxm.isChecked())
        self.assertEqual((sg.q_val.value(), sg.f0_val.value()), (5000.0, 30000.0))
        self.assertTrue(sg.q_val.isEnabled())


if __name__ == "__main__":
    unittest.main()
