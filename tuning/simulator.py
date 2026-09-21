"""
Virtual qPlus + lock-in + PLL, for developing and testing the tuner offline.

Physics (all standard, ideal amplitude loop assumed)
----------------------------------------------------
* The sensor is a resonator with f0 and Q. Its phase relative to the drive
  obeys  dphi/dt = Delta - gamma*tan(phi),  gamma = pi*f0/Q,
  Delta = 2*pi*(f_res - f_drive). The phase therefore lags a frequency change
  with the ring-down time 1/gamma = Q/(pi*f0) (0.32 s for Q = f0 = 25k) and
  saturates like atan for offsets beyond the half-width f0/(2Q).
* The lock-in (DNC ``TimeConstant``) low-passes the measured phase
  (``lockin_stages`` first-order stages, 12 dB/oct for 2).
* A PI controller turns the filtered phase into the drive-frequency offset:
  ``delta = kp*phase + integral(ki*phase)``. The ``df`` channel is delta.
* White phase noise at the detector reaches df through the lock-in filter and
  the controller. With the proportional term dominating (as it does here),
  the df noise scales ~linearly with the gain (about bandwidth**1.1, see
  tests) and falls as ~1/sqrt(lockin_tau). The exponent is model-dependent:
  the real instrument's scaling has to be *measured*, which is what the tuner
  does.
* Against the sensor's slow pole (ring-down ~0.3 s) the proportional term does
  the fast tracking of df; the integral term only removes the residual, which
  shows up as the tail of the *phase* transient. That is why the manual
  watches both df and Phase.

What is NOT known
-----------------
How SXM's arbitrary Kp/Ki map onto physical gains. :class:`SXMScale` holds an
*assumed* mapping, chosen (loop crossover ~100 rad/s, PI zero ~8x below it at
Ki=-1e4) so that Kp=-100 with Ki=-1e3 / -1e4 / -5e4 behaves like the manual's
figure: slow phase tail / fast with ~10 % overshoot / strong overshoot. Its
absolute time scale is a guess. It is a placeholder to be replaced by
:mod:`tuning.identify` once real captures exist; the tuner never relies on it.
"""

import math
from dataclasses import dataclass

import numpy as np

from .trial import StepProtocol, TrialResult


@dataclass(frozen=True)
class SXMScale:
    """Assumed raw SXM gain -> physical gain mapping (see module docstring)."""

    kp_hz_per_deg: float = 0.0028     # physical Kp [Hz/deg] per unit of |raw Kp|
    ki_hz_per_deg_s: float = 0.00035  # physical Ki [Hz/(deg*s)] per unit of |raw Ki|


@dataclass(frozen=True)
class PLLSetup:
    """The virtual sensor and lock-in."""

    f0: float = 25e3                  # free resonance [Hz]
    q: float = 25e3
    lockin_tau: float = 2e-3          # DNC TimeConstant [s] (manual: ~1/(10*BW_PLL))
    lockin_stages: int = 2
    phase_noise_deg_rthz: float = 0.01  # white detector phase noise [deg/sqrt(Hz)]
    dt: float = 2.5e-4                # integration step [s]

    @property
    def gamma(self) -> float:
        """Amplitude/phase decay rate of the resonator [1/s]."""
        return math.pi * self.f0 / self.q


def simulate_pll(kp_raw, ki_raw, dF, setup=PLLSetup(), scale=SXMScale(), seed=0):
    """
    Integrate the loop.

    Parameters
    ----------
    kp_raw, ki_raw : float
        Raw SXM gains. The manual requires both to be negative; positive values
        are positive feedback and the loop runs away (lock is lost).
    dF : array
        Apparent resonance offset from the PLL centre frequency, per time step [Hz].
    seed : int
        Seed of the detector noise.

    Returns
    -------
    df, phase : arrays (Hz, deg) per time step, and ``locked`` (bool).
    """
    dF = np.asarray(dF, dtype=float)
    n = len(dF)
    dt = setup.dt
    gamma = setup.gamma
    kp = -kp_raw * scale.kp_hz_per_deg
    ki = -ki_raw * scale.ki_hz_per_deg_s
    a = 1.0 - math.exp(-dt / setup.lockin_tau)
    stages = setup.lockin_stages
    sigma = setup.phase_noise_deg_rthz / math.sqrt(2.0 * dt)
    noise = np.random.default_rng(seed).normal(0.0, sigma, n) if sigma > 0 else np.zeros(n)

    df = np.empty(n)
    ph = np.empty(n)
    phi = 0.0                          # true resonator phase [rad]
    lp1 = lp2 = 0.0                    # lock-in stages [deg]
    integ = float(dF[0])               # start locked: delta = dF[0]
    two_pi = 2.0 * math.pi
    to_deg = 180.0 / math.pi
    limit = 1.45                       # ~83 deg: beyond this the PLL has lost the resonance
    tan = math.tan
    locked = True
    k = 0
    while k < n:
        delta = kp * lp2 + integ
        phi += dt * (two_pi * (dF[k] - delta) - gamma * tan(phi))
        if not (-limit < phi < limit):  # also catches nan
            locked = False
            df[k:] = delta
            ph[k:] = lp2
            break
        meas = phi * to_deg + noise[k]
        lp1 += a * (meas - lp1)
        lp2 = lp1 if stages < 2 else lp2 + a * (lp1 - lp2)
        integ += ki * lp2 * dt
        df[k] = delta
        ph[k] = lp2
        k += 1
    return df, ph, locked


def _decimate(x, k):
    n = (len(x) // k) * k
    return x[:n].reshape(-1, k).mean(axis=1)


def run_step_trial(kp_raw, ki_raw, protocol=StepProtocol(), setup=PLLSetup(), scale=SXMScale(),
                   seed=0, out_fs=2000.0) -> TrialResult:
    """Simulate the manual's +-1 Hz PLL step test and return what a scope would record."""
    n_per_hold = int(round(protocol.hold_s / setup.dt))
    dF = np.repeat(np.array(protocol.levels), n_per_hold)
    df, ph, locked = simulate_pll(kp_raw, ki_raw, dF, setup, scale, seed)
    k = max(1, int(round(1.0 / (out_fs * setup.dt))))
    df_d, ph_d = _decimate(df, k), _decimate(ph, k)
    t = (np.arange(len(df_d)) + 0.5) * k * setup.dt
    return TrialResult(kp=kp_raw, ki=ki_raw, t=t, df=df_d, phase=ph_d, protocol=protocol,
                       locked=locked, meta={"simulated": True, "seed": seed})


class SimulatedPLLBackend:
    """Trial backend that runs the simulator. Each trial gets an independent noise seed."""

    def __init__(self, protocol=StepProtocol(), setup=PLLSetup(), scale=SXMScale(), seed=0):
        self.protocol = protocol
        self.setup = setup
        self.scale = scale
        self._seed = seed
        self.n_trials = 0

    def run_trial(self, kp_raw, ki_raw) -> TrialResult:
        self.n_trials += 1
        self._seed += 1
        return run_step_trial(kp_raw, ki_raw, self.protocol, self.setup, self.scale, self._seed)
