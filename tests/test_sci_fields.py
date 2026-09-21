"""
Scientific notation in the number fields: gains such as Kp / Ki span many decades (2.5e8 instead of 250000000.0000).
Offscreen Qt; no hardware.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtGui, QtWidgets                                       # noqa: E402

from sxm_ncafm_control.common import NumericItemDelegate, format_number  # noqa: E402
from sxm_ncafm_control.gui import tuning_tab as T                        # noqa: E402
from sxm_ncafm_control.gui.sci_spinbox import SciDoubleSpinBox           # noqa: E402
from sxm_ncafm_control.tests.fake_instrument import FakeInstrument       # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
V = QtGui.QValidator


class FormatNumber(unittest.TestCase):
    def test_examples(self):
        cases = {
            2e9: "2e9", 8.9e7: "8.9e7", -2.5e8: "-2.5e8", 2.5e-4: "2.5e-4", 1234567.0: "1.234567e6",
            25000.0: "25000", 25000.123456: "25000.123456", 100.0: "100", -100.0: "-100", 0.5: "0.5", 0.0015: "0.0015", 0.0: "0",
            279999999.99999994: "2.8e8",                                  # float noise is not shown
        }
        for v, text in cases.items():
            self.assertEqual(format_number(v), text, v)

    def test_threshold_is_adjustable(self):
        self.assertEqual(format_number(-1e4), "-10000")
        self.assertEqual(format_number(-1e4, sci_above=1e4), "-1e4")
        self.assertEqual(format_number(9999.0, sci_above=1e4), "9999")
        self.assertEqual(format_number(2e4, 6, 1e4), "2e4")

    def test_it_reads_back_exactly_enough_to_be_sent_to_the_instrument(self):
        for v in (2e9, 8.9e7, -2.5e8, 2.5e-4, 25000.123456, 3.7123456789e8, 0.1 + 0.2, 1e-7):
            self.assertAlmostEqual(float(format_number(v)) / v, 1.0, places=13)

    def test_non_finite_values_do_not_raise(self):
        self.assertEqual(format_number(float("inf")), "inf")


class SciSpin(unittest.TestCase):
    @staticmethod
    def spin(**kw):
        s = SciDoubleSpinBox(**kw)
        s.setDecimals(4)
        s.setRange(-1e12, 1e12)
        return s

    def test_display(self):
        s = self.spin()
        for v, text in ((2e9, "2e9"), (-100.0, "-100"), (0.5, "0.5"), (25000.0, "25000")):
            s.setValue(v)
            self.assertEqual(s.text(), text)
        s.setSuffix(" Hz")
        s.setValue(25000.0)
        self.assertEqual(s.text(), "25000 Hz")

    def test_scientific_input_is_acceptable(self):
        s = self.spin()
        for txt, val in (("2.5e8", 2.5e8), ("2.5E8", 2.5e8), ("1.23456789e8", 1.23456789e8), ("-1e4", -1e4),
                         ("2,5e8", 2.5e8), ("25000", 25000.0)):
            self.assertEqual(s.validate(txt, 0)[0], V.Acceptable, txt)
            self.assertEqual(s.valueFromText(txt), val, txt)

    def test_partial_out_of_range_and_bad_input(self):
        s = self.spin()
        for txt in ("", "-", "2e", "2e-", ".", "1e13"):
            self.assertEqual(s.validate(txt, 0)[0], V.Intermediate, txt)
        for txt in ("abc", "2e8x", "1e5e5", "nan"):
            self.assertEqual(s.validate(txt, 0)[0], V.Invalid, txt)

    def test_typed_text_becomes_the_value(self):
        s = self.spin()
        s.lineEdit().setText("2.5e8")
        s.interpretText()
        self.assertEqual(s.value(), 2.5e8)
        s.setSuffix(" ms")
        s.lineEdit().setText("2e3 ms")
        s.interpretText()
        self.assertEqual(s.value(), 2000.0)
        s.lineEdit().setText("1e13")                                        # out of range: Qt puts the previous value back
        s.interpretText()
        self.assertEqual(s.value(), 2000.0)

    def test_arrow_steps_are_relative_for_values_shown_in_e_notation(self):
        s = self.spin()
        s.setValue(2e9)
        s.stepBy(1)
        self.assertAlmostEqual(s.value() / 2e9, 1.1)
        self.assertEqual(s.text(), "2.2e9")
        s.stepBy(-1)
        self.assertAlmostEqual(s.value() / 2e9, 1.0)
        s.setSingleStep(1.0)
        s.setValue(5.0)
        s.stepBy(1)
        self.assertEqual(s.value(), 6.0)                                    # ordinary values keep the ordinary step

    def test_a_value_survives_being_re_read_from_its_text(self):
        s = self.spin()
        s.setDecimals(6)
        for v in (3.7e8, 25000.123456, -1e4, 2.5e-3):
            s.setValue(v)
            s.interpretText()
            self.assertEqual(s.value(), v)


class FieldsInTheTabs(unittest.TestCase):
    def test_tuning_tab_gain_fields(self):
        tab = T.TuningTab(FakeInstrument(), FakeInstrument())
        self.assertEqual((tab.kp_spin.text(), tab.ki_spin.text()), ("-100", "-1e4"))
        self.assertEqual(tab.base_spin.text(), "25000")
        tab.loop_combo.setCurrentIndex(tab.loop_combo.findData("afl"))
        self.assertEqual((tab.kp_spin.text(), tab.ki_spin.text()), ("2e8", "2e4"))
        tab.gain_combo.setCurrentIndex(tab.gain_combo.findData(0.1))
        tab.btn_fill.click()
        self.assertEqual((tab.kp_spin.text(), tab.ki_spin.text()), ("2e9", "2e5"))
        self.assertIn("Kp = 2e9", tab.start_label.text())
        tab.kp_spin.lineEdit().setText("3.5e9")
        tab.kp_spin.interpretText()
        self.assertEqual(tab.kp_spin.value(), 3.5e9)
        self.assertEqual(tab.range_spin.text(), "1000 x")

    def test_every_number_field_of_the_tuning_tab_takes_scientific_input(self):
        tab = T.TuningTab(FakeInstrument(), FakeInstrument())
        spins = [w for w in tab.findChildren(QtWidgets.QDoubleSpinBox)]
        self.assertGreater(len(spins), 10)
        self.assertTrue(all(isinstance(w, SciDoubleSpinBox) for w in spins))

    def test_step_test_fields(self):
        from sxm_ncafm_control.gui.step_test_tab import StepTestTab
        tab = StepTestTab(FakeInstrument())
        for w in (tab.low, tab.high, tab.base):
            self.assertIsInstance(w, SciDoubleSpinBox)
        tab.low.setValue(8.9e7)
        self.assertEqual(tab.low.text(), "8.9e7")
        tab.high.lineEdit().setText("1.2e8")
        tab.high.interpretText()
        self.assertEqual(tab.high.value(), 1.2e8)

    def test_suggested_setup_fields(self):
        from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
        tab = SuggestedTab(FakeInstrument(), None)
        tab.q_val.setValue(25000.0)
        tab.f0_val.setValue(25000.0)
        tab.out_gain.setCurrentIndex(tab.out_gain.findData(1.0))
        tab._recalc()
        self.assertEqual((tab.ki_out.text(), tab.kp_out.text()), ("2e4", "2e8"))
        tab.q_val.lineEdit().setText("1e5")
        tab.q_val.interpretText()
        self.assertEqual(tab.q_val.value(), 1e5)
        self.assertEqual(tab.ki_out.text(), "5000")                        # recalculated by itself (5e8 / 1e5)

    def test_params_table_accepts_and_shows_scientific_notation(self):
        editor = NumericItemDelegate().createEditor(None, None, None)
        for txt in ("2.5e8", "-1e4", "25000.5", "1.23456789e8"):
            self.assertEqual(editor.validator().validate(txt, 0)[0], V.Acceptable, txt)
        self.assertEqual(editor.validator().validate("abc", 0)[0], V.Invalid)
        from sxm_ncafm_control.gui.params_tab import ParamsTab

        class Dde:
            def __getattr__(self, name):
                return lambda *a, **k: None
        tab = ParamsTab(Dde())
        self.assertTrue(tab.stage_value("EDIT", "Edit24", 2e8))            # Amp Ki
        rows = [r for r in range(tab.table.rowCount()) if tab.table.item(r, 4).text()]
        self.assertEqual([tab.table.item(r, 4).text() for r in rows], ["2e8"])


if __name__ == "__main__":
    unittest.main()
