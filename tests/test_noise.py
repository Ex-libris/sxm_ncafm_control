"""tuning/noise.py on synthetic quiet windows with known noise. Pure numpy."""
import math
import unittest

import numpy as np

from sxm_ncafm_control.tuning import noise as N


def window(seconds=20.0, fs=2000.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / fs)
    t = t + rng.uniform(-0.1, 0.1, len(t)) / fs                 # slices are not perfectly regular
    return t, rng


class QuietMetrics(unittest.TestCase):
    def test_white_noise_levels_bands_and_pixels(self):
        t, rng = window()
        fs = 2000.0
        sigma = 0.01                                             # 10 mHz rms df, white up to fs/2
        df = rng.normal(0.0, sigma, len(t))
        amp = 1e-3 * (1 + rng.normal(0.0, 0.01, len(t)))         # 1 mV, 1 % rms
        q = N.quiet_metrics(t, {"df": df, "QPlusAmpl": amp})
        self.assertAlmostEqual(q.duration_s, 20.0, delta=0.01)
        self.assertAlmostEqual(q.fs, fs, delta=5)
        self.assertAlmostEqual(q.get("df"), sigma, delta=0.05 * sigma)
        self.assertTrue(math.isnan(q.get("df", "rel_rms")))       # df has no level to be relative to
        self.assertAlmostEqual(q.get("QPlusAmpl", "rel_rms"), 0.01, delta=0.001)
        # white noise: band rms ~ sigma * sqrt(bandwidth / (fs/2))
        b = q.get("df", "band", N.band_key(10, 50))
        self.assertAlmostEqual(b, sigma * math.sqrt(40 / 1000), delta=0.2 * sigma * math.sqrt(40 / 1000))
        # pixel averaging: sigma / sqrt(samples per pixel)
        p = q.get("df", "pixel", 0.1)
        self.assertAlmostEqual(p, sigma / math.sqrt(200), delta=0.35 * sigma / math.sqrt(200))
        f, a = q.asd["df"]
        self.assertLessEqual(len(f), 300)
        self.assertAlmostEqual(float(np.median(a[(f > 20) & (f < 500)])), sigma / math.sqrt(1000), delta=3e-5)

    def test_coherence_finds_a_shared_source(self):
        t, rng = window()
        common = rng.normal(0.0, 1.0, len(t))
        drive = 10e-6 * (1 + 0.1 * common + 0.02 * rng.normal(size=len(t)))
        phase = 0.5 * common + 0.05 * rng.normal(size=len(t))
        q = N.quiet_metrics(t, {"Drive": drive, "Phase": phase, "QPlusAmpl": 1e-3 + 1e-5 * rng.normal(size=len(t))})
        self.assertGreater(q.coherence["Drive~Phase"][N.band_key(1, 10)], 0.9)
        self.assertLess(q.coherence["QPlusAmpl~Phase"][N.band_key(1, 10)], 0.2)

    def test_drive_floor(self):
        t, rng = window(seconds=5)
        drive = np.full(len(t), 10e-6)
        drive[: len(t) // 4] = 0.0
        q = N.quiet_metrics(t, {"Drive": drive})
        self.assertAlmostEqual(q.get("Drive", "floor_frac"), 0.25, delta=0.01)

    def test_too_short(self):
        t = np.linspace(0, 0.5, 100)
        self.assertIsNone(N.quiet_metrics(t, {"df": np.zeros(100)}))


if __name__ == "__main__":
    unittest.main()
