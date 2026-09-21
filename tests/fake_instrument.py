"""
TEST FIXTURE: stands in for BOTH the DDE client and the IOCTL driver of the PLL, running the loop in real time.

Writes to Edit27/Edit22 (PLL Kp/Ki) and DNC 3 (`use`) change the virtual instrument; ``read_raw`` returns the
``df`` / ``Phase`` channels in raw counts, exactly as the driver would.
"""
import math
import threading
import time

import numpy as np

from sxm_ncafm_control.device_driver import CHANNELS
from sxm_ncafm_control.tests.sim import PLLSetup, SXMScale


class FakeInstrument:
    def __init__(self, f_res=25000.0, setup=None, scale=None, seed=0):
        self.setup = setup or PLLSetup(dt=5e-4)
        self.scale = scale or SXMScale()
        self.f_res = f_res
        self.kp_raw, self.ki_raw, self.use = -100.0, -1e4, f_res
        self.writes = []
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        self._phi = 0.0
        self._lp1 = self._lp2 = 0.0
        self._integ = 0.0
        self._t = time.perf_counter()
        self._df_idx, _, _, self._df_scale = CHANNELS["df"]
        self._ph_idx, _, _, self._ph_scale = CHANNELS["Phase"]

    # -- DDE-like ---------------------------------------------------------------------------------
    # No lock here: attribute writes are atomic under the GIL, and the real DDE path never waits for the driver.
    # (A lock made the GUI thread starve behind the capture thread's tight read loop.)
    def send_scanpara(self, code, value):
        self.writes.append((code, float(value)))
        if code == "Edit27":
            self.kp_raw = float(value)
        elif code == "Edit22":
            self.ki_raw = float(value)

    def send_dncpara(self, index, value):
        self.writes.append((f"DNC{index}", float(value)))
        if index == 3:
            self.use = float(value)

    # -- driver-like ---------------------------------------------------------------------------------
    def read_raw(self, idx):
        with self._lock:
            self._advance()
            if idx == self._df_idx:
                return int(round((self._kp * self._lp2 + self._integ) / self._df_scale))
            if idx == self._ph_idx:
                return int(round(self._lp2 / self._ph_scale))
            return 0

    @property
    def _kp(self):
        return -self.kp_raw * self.scale.kp_hz_per_deg

    def _advance(self):
        s = self.setup
        dt = s.dt
        now = time.perf_counter()
        n = min(int((now - self._t) / dt), 4000)
        if n <= 0:
            return
        self._t = now if n == 4000 else self._t + n * dt
        kp, ki = self._kp, -self.ki_raw * self.scale.ki_hz_per_deg_s
        a = 1.0 - math.exp(-dt / s.lockin_tau)
        sigma = s.phase_noise_deg_rthz / math.sqrt(2.0 * dt)
        noise = self._rng.normal(0.0, sigma, n)
        dF = self.f_res - self.use
        phi, lp1, lp2, integ = self._phi, self._lp1, self._lp2, self._integ
        for k in range(n):
            delta = kp * lp2 + integ
            phi += dt * (2.0 * math.pi * (dF - delta) - s.gamma * math.tan(phi))
            phi = max(-1.45, min(1.45, phi))            # saturate: the phase reading pins at ~83 deg when lock is lost
            lp1 += a * (phi * 180.0 / math.pi + noise[k] - lp1)
            lp2 += a * (lp1 - lp2)
            integ += ki * lp2 * dt
        self._phi, self._lp1, self._lp2, self._integ = phi, lp1, lp2, integ
