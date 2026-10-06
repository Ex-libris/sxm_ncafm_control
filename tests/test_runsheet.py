"""tuning/runsheet.py: gains from G and rho, compensation, ramps, the stop rule, settings, saving. Pure Python."""
import math
import os
import tempfile
import unittest

from sxm_ncafm_control.tuning import runsheet as R

AFL = R.Anchor("afl", 1e6, 100.0, output_gain_v=0.1, input_gain=1.0)
PLL = R.Anchor("pll", -200.0, -1.65e4)


class Gains(unittest.TestCase):
    def test_g_scales_both_and_rho_only_ki(self):
        self.assertEqual(R.gains(AFL, R.Condition()), (1e6, 100.0))
        kp, ki = R.gains(AFL, R.Condition(g=0.3))
        self.assertAlmostEqual(kp, 3e5)
        self.assertAlmostEqual(ki, 30.0)
        kp, ki = R.gains(AFL, R.Condition(g=0.3, rho=3))
        self.assertAlmostEqual(kp, 3e5)
        self.assertAlmostEqual(ki, 90.0)
        self.assertAlmostEqual(ki / kp, 3 * 100.0 / 1e6)

    def test_pll_gains_keep_their_sign(self):
        kp, ki = R.gains(PLL, R.Condition(g=0.4, rho=2))
        self.assertAlmostEqual(kp, -80.0)
        self.assertAlmostEqual(ki, -1.32e4)

    def test_compensation_keeps_the_amplitude_loop_gain(self):
        # +-1 V instead of +-0.1 V: gains /10 (manual: x10 going down a range); InA x10: gains /10
        kp, ki = R.gains(AFL, R.Condition(settings={"output_gain_v": 1.0}))
        self.assertAlmostEqual(kp, 1e5)
        self.assertAlmostEqual(ki, 10.0)
        kp, _ = R.gains(AFL, R.Condition(), current={"input_gain": 10.0})
        self.assertAlmostEqual(kp, 1e5)
        kp, _ = R.gains(AFL, R.Condition(settings={"output_gain_v": 1.0}), compensate=False)
        self.assertEqual(kp, 1e6)
        self.assertEqual(R.compensation(PLL, 1.0, 10.0), 1.0)                 # the PLL is not rescaled

    def test_anchor_needs_both_gains(self):
        with self.assertRaises(ValueError):
            R.Anchor("afl", 1e6, 0.0)


class Ramps(unittest.TestCase):
    def test_g_ramp_runs_from_the_gentlest_value(self):
        sheet = R.ramp("g", [3, 0.3, 1], rho=0.5)
        self.assertEqual([c.g for c in sheet], [0.3, 1.0, 3.0])
        self.assertTrue(all(c.rho == 0.5 and c.ramp == "g" for c in sheet))
        self.assertEqual(len({c.group for c in sheet}), 1)

    def test_setting_ramp_and_reference_repeats(self):
        sheet = R.ramp("amp_tau_ms", [5, 10, 20, 50], g=0.5, reference_every=2)
        self.assertEqual([c.role for c in sheet], ["reference", "test", "test", "reference", "test", "test"])
        tests = [c for c in sheet if c.role == "test"]
        self.assertEqual([c.settings["amp_tau_ms"] for c in tests], [5, 10, 20, 50])
        self.assertTrue(all(c.g == 0.5 for c in tests))
        ref = sheet[0]
        self.assertEqual((ref.g, ref.rho, ref.settings), (1.0, 1.0, {}))

    def test_bad_ramps(self):
        with self.assertRaises(ValueError):
            R.ramp("g", [0, 1])
        with self.assertRaises(ValueError):
            R.ramp("kp", [1])
        with self.assertRaises(ValueError):
            R.ramp("g", [])

    def test_parse_values(self):
        self.assertEqual(R.parse_values("g", "0.1, 0.3; 1,3"), [0.1, 0.3, 1.0, 3.0])
        self.assertEqual(R.parse_values("g", "0.1 0.3 1"), [0.1, 0.3, 1.0])
        self.assertEqual(R.parse_values("dnc_rolloff", "6, 12 dB/oct, 24"), ["6", "12", "24"])
        with self.assertRaises(ValueError):
            R.parse_values("g", "fast")


class StopRule(unittest.TestCase):
    def test_a_failed_g_skips_the_higher_g_of_its_ramp_only(self):
        sheet = R.ramp("g", [0.3, 1, 3, 10]) + R.ramp("g", [0.3, 1, 3], rho=3)
        self.assertEqual(R.stop_rule(sheet, 1, "ringing"), [2, 3])
        self.assertEqual(R.stop_rule(sheet, 1, "overshoot"), [])               # overshoot is not a stop
        self.assertEqual(R.stop_rule(sheet, 5, "lost"), [6])

    def test_rho_ramp_and_setting_ramps(self):
        sheet = R.ramp("rho", [0.3, 1, 3], g=1)
        self.assertEqual(R.stop_rule(sheet, 1, "floor"), [2])
        sheet = R.ramp("amp_tau_ms", [5, 10, 20])
        self.assertEqual(R.stop_rule(sheet, 0, "lost"), [])                    # not monotonic: not cut

    def test_references_are_never_skipped(self):
        sheet = R.ramp("g", [0.3, 1, 3, 10], reference_every=2)
        failed = next(i for i, c in enumerate(sheet) if c.role == "test" and c.g == 1.0)
        skipped = R.stop_rule(sheet, failed, "lost")
        self.assertTrue(all(sheet[i].role == "test" for i in skipped))
        self.assertEqual(sorted(sheet[i].g for i in skipped), [3.0, 10.0])


class Settings(unittest.TestCase):
    def test_readout_values_in_table_units(self):
        self.assertAlmostEqual(R.sxm_value("amp_tau_ms", 0.02), 20.0)
        self.assertEqual(R.sxm_value("dnc_rolloff", "12 db/oct"), "12")
        self.assertIsNone(R.sxm_value("amp_tau_ms", None))
        self.assertTrue(R.same_setting("dnc_tc_ms", 5.0, 5.04))
        self.assertFalse(R.same_setting("dnc_tc_ms", 5.0, 10.0))
        self.assertTrue(R.same_setting("dnc_rolloff", "12", "12 db/oct"))

    def test_option_for_gui_controls(self):
        self.assertEqual(R.option_for("amp_tau_ms", ["1 ms", "2 ms", "20 ms", "0.5 s"], 20), "20 ms")
        self.assertEqual(R.option_for("dnc_tc_ms", ["100 µs", "1 ms"], 0.1), "100 µs")
        self.assertEqual(R.option_for("output_gain_v", ["±0.1", "±1", "±10"], 1.0), "±1")
        self.assertEqual(R.option_for("input_gain", ["1", "10"], 10), "10")
        self.assertEqual(R.option_for("dnc_rolloff", ["6 db/oct", "12 db/oct", "24 db/oct"], "12"), "12 db/oct")
        self.assertIsNone(R.option_for("amp_tau_ms", ["1 ms"], 7))


class Saving(unittest.TestCase):
    def test_round_trip(self):
        sheet = R.ramp("g", [0.3, 1]) + R.ramp("dnc_tc_ms", [1, 5], g=0.5, rho=2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sheet.json")
            R.save_sheet(path, AFL, sheet, {"train": {"hold_s": 4.0}})
            anchor, back, extra = R.load_sheet(path)
        self.assertEqual(anchor, AFL)
        self.assertEqual(back, sheet)
        self.assertEqual(extra["train"]["hold_s"], 4.0)

    def test_rejects_other_files(self):
        with self.assertRaises(ValueError):
            R.sheet_from_dict({"kind": "ncafm_tune"})
        bad = R.sheet_to_dict(AFL, [R.Condition(settings={"warp": 9})])
        with self.assertRaises(ValueError):
            R.sheet_from_dict(bad)

    def test_labels(self):
        c = R.Condition(g=0.5, rho=2, settings={"output_gain_v": 1.0, "dnc_rolloff": "12"})
        self.assertEqual(c.label(), "G 0.5, ρ 2, AFL output gain ±1 V, DNC RollOff 12 dB/oct")
        self.assertTrue(math.isclose(R.gains(AFL, c)[0], 0.5e5))


if __name__ == "__main__":
    unittest.main()
