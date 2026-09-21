"""
Universal identification of a PI loop from ONE recorded train of steps.

Both loops that are tuned on the SXM have the same structure::

    controller   u = Kp*e + Ki*integral(e)            (PI)
    plant        y' = -gamma*y + b*(input)            (first order: the sensor's ring-down)
    filter       F(s) between the plant and the controller input

============  ===============================  ===========================  ===============
loop          controller input ``e``           controller output ``u``      commanded input
============  ===============================  ===========================  ===============
PLL           ``Phase`` (setpoint 0)           ``df``                       ``use`` steps -> plant input is (dF - u)
amplitude     kappa*Ref - Tau-filtered ampl.   ``Drive``                     ``Ref`` steps -> plant input is u
============  ===============================  ===========================  ===============

Because both channels of the controller are recorded, its physical gains follow
from a *linear regression on the traces* (``u = Kp*e + Ki*int(e) + c``): no
transient has to settle, so trains of 0.2 s steps work, and no simulation is
needed. The plant follows from a second regression in integral form. Two
details make it robust:

* the recorded channels are lock-in filtered, so the plant equation is fitted
  in *filtered* variables (the same known filter is applied to its input terms);
* the delay between the commanded step and its effect (DDE latency) is found by
  a 1-D search and refined.

The result, :class:`LoopModel`, maps raw SXM gains to physical ones and can
predict step metrics, ringing, stability margins and noise for gains that were
never tried (see :meth:`LoopModel.predict`).

Assumptions to verify on real data: ``df`` is the PI output relative to ``use``
and ``Phase`` is the PI input (PLL); the recorded ``QPlusAmpl`` is lock-in
filtered but *not* passed through ``Tau``; the amplitude loop is fast compared
with the PLL when the PLL is tested (its ``Drive`` action is neglected).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import optimize, signal
from scipy.integrate import cumulative_trapezoid

from . import metrics as M
from .trial import LoopCapture, StepTrain


# ---------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------
def _cumint(x, t):
    return cumulative_trapezoid(x, t, initial=0.0)


def lowpass_cascade(x, dt, tau, stages):
    """``stages`` identical first-order low-passes (exact discretisation), starting in equilibrium with x[0]."""
    y = np.asarray(x, dtype=float)
    if tau <= 0 or stages <= 0:
        return y.copy()
    a = math.exp(-dt / tau)
    for _ in range(stages):
        zi = signal.lfiltic([1.0 - a], [1.0, -a], [y[0]], [y[0]])
        y, _ = signal.lfilter([1.0 - a], [1.0, -a], y, zi=zi)
    return y


def _uniform(t, *arrays):
    """Resample onto a uniform grid if the time base is not (already) uniform."""
    t = np.asarray(t, dtype=float)
    dts = np.diff(t)
    dt = float(np.median(dts))
    if np.max(np.abs(dts - dt)) <= 0.02 * dt:
        return t, dt, [np.asarray(a, dtype=float) for a in arrays]
    tu = np.arange(t[0], t[-1], dt)
    return tu, dt, [np.interp(tu, t, a) for a in arrays]


def _lstsq(X, target):
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    res = target - X @ coef
    var = float(np.var(target)) or 1.0
    return coef, float(np.mean(res ** 2)), 1.0 - float(np.var(res)) / var


def _best_delay(cost, lo=-0.010, hi=0.050, step=0.0025):
    """Minimise ``cost(delay)``: coarse grid, then a bounded refinement around the best cell."""
    grid = np.arange(lo, hi + 1e-12, step)
    costs = [cost(d) for d in grid]
    j = int(np.argmin(costs))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    return float(optimize.minimize_scalar(cost, bounds=(a, b), method="bounded", options={"xatol": 2e-5}).x)


def estimate_noise_density(t, y, li_tau, li_stages=2, block_s=0.0, band=None) -> float:
    """
    White noise density [y-units/sqrt(Hz)] at the detector, from the high-frequency part of a
    lock-in filtered channel (where the loop has no influence): median-averaged Welch spectrum,
    divided by the known filter response (and the block-averaging of the recording).
    """
    t, dt, (y,) = _uniform(t, y)
    fs = 1.0 / dt
    f, pxx = signal.welch(y - np.mean(y), fs=fs, nperseg=int(min(len(y), 1024)), average="median", detrend="linear")
    lo, hi = band if band else (max(3.0 / (2 * math.pi * li_tau), 0.08 * fs), 0.35 * fs)
    m = (f >= lo) & (f <= hi)
    if m.sum() < 3:
        return math.nan
    filt2 = (1.0 / (1.0 + (2 * math.pi * f[m] * li_tau) ** 2)) ** li_stages
    box2 = np.sinc(f[m] * block_s) ** 2 if block_s > 0 else 1.0
    return float(math.sqrt(np.median(pxx[m] / filt2 / box2)))


# ---------------------------------------------------------------------------
# the fitted model
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    """What the identified model says about one (Kp, Ki)."""

    kp_raw: float
    ki_raw: float
    stable: bool
    primary: Optional[M.StepMetrics] = None      # PLL: df step response; amplitude loop: amplitude response
    secondary: Optional[M.StepMetrics] = None    # amplitude loop: Drive step response
    error: Optional[M.ErrorMetrics] = None       # PLL: Phase transient
    noise_u: float = math.nan                    # RMS noise of the controller output (df / Drive)
    noise_pixel_u: float = math.nan              # same after averaging over one pixel dwell
    crossover_rad_s: float = math.nan
    phase_margin_deg: float = math.nan
    max_stable_scale: float = math.nan           # largest common multiplier on (Kp, Ki) that stays stable


@dataclass(frozen=True)
class LoopModel:
    """Identified PI loop; ``scale_p``/``scale_i`` map raw SXM gains to physical gains (signed)."""

    kind: str
    scale_p: float
    scale_i: float
    gamma: float                 # plant pole [1/s]
    b: float                     # plant input gain
    kappa: float                 # Ref units -> amplitude units (amplitude loop; 1 for the PLL)
    li_tau: float                # lock-in time constant of the recorded channels
    li_stages: int
    ctrl_tau: float              # amplitude loop 'Tau' (0 for the PLL)
    delay_s: float               # command -> effect latency found in the data
    noise_density: float         # detector noise [unit/sqrt(Hz)], nan if unknown
    diagnostics: Dict[str, float] = field(default_factory=dict)
    warnings: tuple = ()

    # -- transfer functions -----------------------------------------------------------
    def _parts(self, kp_raw, ki_raw, scale=1.0):
        kp, ki = self.scale_p * kp_raw * scale, self.scale_i * ki_raw * scale
        num_c, den_c = np.array([kp, ki]), np.array([1.0, 0.0])
        num_p, den_p = np.array([self.b]), np.array([1.0, self.gamma])
        if self.kind == "pll":
            tau, n = self.li_tau, self.li_stages
        else:
            tau, n = self.ctrl_tau, 1
        num_f, den_f = np.array([1.0]), np.array([1.0])
        for _ in range(n):
            den_f = np.polymul(den_f, [tau, 1.0])
        num_l = np.polymul(num_c, np.polymul(num_f, num_p))
        den_l = np.polymul(den_c, np.polymul(den_f, den_p))
        return dict(num_c=num_c, den_c=den_c, num_p=num_p, den_p=den_p, num_f=num_f, den_f=den_f,
                    num_l=num_l, den_l=den_l, den=np.polyadd(den_l, num_l))

    def transfer(self, kp_raw, ki_raw, name, scale=1.0):
        """(num, den) of the closed-loop transfer function ``name`` from the commanded input r."""
        p = self._parts(kp_raw, ki_raw, scale)
        pm = np.polymul
        if self.kind == "pll":            # plant input is (r - u)
            tf = {"u": (p["num_l"], p["den"]),
                  "y": (pm(p["num_p"], pm(p["den_c"], p["den_f"])), p["den"]),
                  "e": (pm(p["num_f"], pm(p["num_p"], p["den_c"])), p["den"])}
        else:                             # plant input is u; controller sees r - F*y
            tf = {"u": (pm(p["num_c"], pm(p["den_f"], p["den_p"])), p["den"]),
                  "y": (pm(p["num_p"], pm(p["num_c"], p["den_f"])), p["den"])}
        if name == "noise_u":             # measurement noise -> controller output (both loops)
            return pm(p["num_c"], pm(p["num_f"], p["den_p"])), p["den"]
        if name == "noise_y":
            return p["num_l"], p["den"]
        return tf[name]

    def is_stable(self, kp_raw, ki_raw, scale=1.0) -> bool:
        return bool(np.all(np.roots(self._parts(kp_raw, ki_raw, scale)["den"]).real < 0))

    def max_stable_scale(self, kp_raw, ki_raw, limit=64.0) -> float:
        """Largest common multiplier on (Kp, Ki) that stays stable (``limit`` if none below it)."""
        if not self.is_stable(kp_raw, ki_raw):
            return 0.0
        lo, hi = 1.0, limit
        if self.is_stable(kp_raw, ki_raw, hi):
            return limit
        for _ in range(30):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if self.is_stable(kp_raw, ki_raw, mid) else (lo, mid)
        return lo

    def margins(self, kp_raw, ki_raw):
        """(crossover [rad/s], phase margin [deg]) of the open loop L = C*F*P."""
        p = self._parts(kp_raw, ki_raw)
        w = np.logspace(-2, 5, 6000)
        L = np.polyval(p["num_l"], 1j * w) / np.polyval(p["den_l"], 1j * w)
        mag = np.abs(L)
        below = np.flatnonzero(mag < 1.0)
        if len(below) == 0 or below[0] == 0:
            return math.nan, math.nan
        i = below[0]
        phase = np.degrees(np.unwrap(np.angle(L)))
        return float(w[i]), float(180.0 + phase[i])

    # -- predictions ----------------------------------------------------------------------
    def _step_curve(self, num, den, pre=0.5):
        poles = np.roots(den)
        tau_slow = -1.0 / float(np.max(poles.real)) if np.all(poles.real < 0) else 1.0
        t_end = float(np.clip(9.0 * tau_slow, 0.4, 60.0))
        t = np.linspace(0.0, t_end, 6000)
        _, y = signal.step((num, den), T=t)
        dt = t[1] - t[0]
        t_pre = -dt * np.arange(int(pre / dt), 0, -1)
        return np.concatenate([t_pre, t]), np.concatenate([np.zeros_like(t_pre), y])

    def noise_rms(self, kp_raw, ki_raw, out="noise_u", t_px=None) -> float:
        """RMS noise at the controller output ('noise_u') or the plant output ('noise_y')."""
        if not (self.noise_density > 0):
            return math.nan
        num, den = self.transfer(kp_raw, ki_raw, out)
        f = np.logspace(-2, 5, 8000)
        h2 = np.abs(np.polyval(num, 2j * np.pi * f) / np.polyval(den, 2j * np.pi * f)) ** 2
        if t_px:
            h2 = h2 * np.sinc(f * t_px) ** 2
        return float(math.sqrt(self.noise_density ** 2 * np.trapezoid(h2, f)
                               if hasattr(np, "trapezoid") else self.noise_density ** 2 * np.trapz(h2, f)))

    def predict(self, kp_raw, ki_raw, t_px=None) -> Prediction:
        """Predicted response of this loop for raw gains (Kp, Ki) - no hardware involved."""
        pred = Prediction(kp_raw=kp_raw, ki_raw=ki_raw, stable=self.is_stable(kp_raw, ki_raw))
        if not pred.stable:
            return pred
        pred.max_stable_scale = self.max_stable_scale(kp_raw, ki_raw)
        pred.crossover_rad_s, pred.phase_margin_deg = self.margins(kp_raw, ki_raw)
        prim, sec = ("u", None) if self.kind == "pll" else ("y", "u")
        try:
            t, y = self._step_curve(*self.transfer(kp_raw, ki_raw, prim))
            pred.primary = M.step_response_metrics(t, y, 0.0)
            if sec:
                t, y = self._step_curve(*self.transfer(kp_raw, ki_raw, sec))
                pred.secondary = M.step_response_metrics(t, y, 0.0)
            if self.kind == "pll":
                t, y = self._step_curve(*self.transfer(kp_raw, ki_raw, "e"))
                pred.error = M.error_transient_metrics(t, y, 0.0)
        except M.StepNotDetectable:
            pass
        pred.noise_u = self.noise_rms(kp_raw, ki_raw, "noise_u")
        if t_px:
            pred.noise_pixel_u = self.noise_rms(kp_raw, ki_raw, "noise_u", t_px=t_px)
        return pred

    def simulate(self, kp_raw, ki_raw, train: StepTrain, t) -> Dict[str, np.ndarray]:
        """Noise-free response ('u' and 'y') of the linear model to a step train."""
        t = np.asarray(t, dtype=float)
        r_scale = self.kappa if self.kind == "afl" else 1.0
        r = train.value_at(t, self.delay_s) * r_scale
        out = {}
        for name in ("u", "y"):
            num, den = self.transfer(kp_raw, ki_raw, name)
            # response to the deviation from the initial level, plus that level's DC response
            _, y, _ = signal.lsim((num, den), r - r[0], t, interp=False)
            out[name] = y + r[0] * float(np.polyval(num, 0.0) / np.polyval(den, 0.0))
        return out


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------
def _fit_pll(t, dt, y, u, train, li_tau, li_stages, delay):
    i_y = _cumint(y, t)
    coef_c, _, r2_c = _lstsq(np.column_stack([y, i_y, np.ones_like(t)]), u)

    def plant(d):
        drive = lowpass_cascade(train.value_at(t, d) - u, dt, li_tau, li_stages)
        X = np.column_stack([np.ones_like(t), i_y, _cumint(drive, t), t])
        return _lstsq(X, y)

    d = _best_delay(lambda x: plant(x)[1]) if delay is None else float(delay)
    coef_p, _, r2_p = plant(d)
    return dict(kp=coef_c[0], ki=coef_c[1], r2_ctrl=r2_c, gamma=-coef_p[1], b=coef_p[2], r2_plant=r2_p,
                delay=d, kappa=1.0)


def _fit_afl(t, dt, y, u, train, li_tau, li_stages, ctrl_tau, kappa, delay):
    y_tau = lowpass_cascade(y, dt, ctrl_tau, 1)             # what the controller sees (F_tau of the channel)
    target = lowpass_cascade(u, dt, li_tau, li_stages)      # F_LI(u): filter both sides of the controller equation

    def ctrl(k, d):
        r = lowpass_cascade(train.value_at(t, d), dt, li_tau, li_stages)
        e = k * r - y_tau
        return _lstsq(np.column_stack([e, _cumint(e, t), np.ones_like(t)]), target)

    # kappa (Ref units -> amplitude units) is not identifiable together with Kp and Ki (Ref and the
    # amplitude are nearly collinear). But before the first step the loop has been running at a constant
    # Ref, so its integral action has converged there and amplitude = kappa * Ref.
    if kappa is None:
        first = t < train.step_times[0]
        settled = first & (t > t[0] + 0.4 * (train.step_times[0] - t[0]))
        if settled.sum() < 20:
            raise ValueError("need a settled hold before the first step to calibrate Ref -> amplitude; "
                             "pass kappa explicitly")
        kappa = float(np.median(y[settled]) / train.levels[0])
    k = float(kappa)
    d = _best_delay(lambda x: ctrl(k, x)[1]) if delay is None else float(delay)
    coef_c, _, r2_c = ctrl(k, d)
    drive = lowpass_cascade(u, dt, li_tau, li_stages)
    coef_p, _, r2_p = _lstsq(np.column_stack([np.ones_like(t), _cumint(y, t), _cumint(drive, t), t]), y)
    return dict(kp=coef_c[0], ki=coef_c[1], r2_ctrl=r2_c, gamma=-coef_p[1], b=coef_p[2], r2_plant=r2_p,
                delay=d, kappa=k)


def identify_loop(cap: LoopCapture, *, li_tau: float, li_stages: int = 2, ctrl_tau: Optional[float] = None,
                  kappa: Optional[float] = None, delay_s: Optional[float] = None,
                  f0: Optional[float] = None, q: Optional[float] = None) -> LoopModel:
    """
    Fit a :class:`LoopModel` to one recorded train.

    Parameters
    ----------
    cap : the recording (see :class:`~tuning.trial.LoopCapture` for which channel is which).
    li_tau, li_stages : the lock-in ``TimeConstant`` and roll-off (12 dB/oct = 2 stages) of the recording.
    ctrl_tau : amplitude loop only - the AFL ``Tau`` (the manual's dropdown, in seconds).
    kappa : amplitude loop only - amplitude reached per unit of ``Ref`` if known (else fitted).
    delay_s : command latency if known (else found in the data).
    f0, q : optional resonance-sweep values; ``gamma = pi*f0/q`` is used as a sanity check and
        replaces the fitted value if that is off by more than a factor of two.
    """
    if cap.kind not in ("pll", "afl"):
        raise ValueError("kind must be 'pll' or 'afl'")
    if cap.kind == "afl" and not ctrl_tau:
        raise ValueError("the amplitude loop needs ctrl_tau (its Tau)")
    if len(cap.t) < 200 or cap.kp_raw == 0 or cap.ki_raw == 0:
        raise ValueError("need a recording of at least 200 samples taken at non-zero Kp and Ki")
    t, dt, (y, u) = _uniform(cap.t, cap.y, cap.u)

    if cap.kind == "pll":
        fit = _fit_pll(t, dt, y, u, cap.train, li_tau, li_stages, delay_s)
    else:
        fit = _fit_afl(t, dt, y, u, cap.train, li_tau, li_stages, ctrl_tau, kappa, delay_s)

    warnings: List[str] = []
    gamma = fit["gamma"]
    diag = {"r2_controller": fit["r2_ctrl"], "r2_plant": fit["r2_plant"], "gamma_fit": gamma,
            "b_fit": fit["b"], "kp_physical": fit["kp"], "ki_physical": fit["ki"]}
    if f0 and q:
        g0 = math.pi * f0 / q
        diag["gamma_prior"] = g0
        if not (0.5 * g0 <= gamma <= 2.0 * g0):
            warnings.append(f"fitted sensor pole {gamma:.3g} 1/s disagrees with pi*f0/Q = {g0:.3g} 1/s; using the latter")
            gamma = g0
    if gamma <= 0:
        raise ValueError("the plant fit gave a non-physical (negative) pole; the train probably excites too little")
    if fit["r2_ctrl"] < 0.98:
        warnings.append(f"controller regression explains only {fit['r2_ctrl']:.3f} of the variance: "
                        "the channels may not be the PI input/output assumed")
    if fit["kp"] * fit["b"] <= 0:
        warnings.append("Kp and the plant gain have opposite signs: the identified loop has positive feedback")

    block = float(cap.meta.get("block_s", 0.0))
    noise = estimate_noise_density(t, y, li_tau, li_stages, block_s=block)
    return LoopModel(kind=cap.kind, scale_p=fit["kp"] / cap.kp_raw, scale_i=fit["ki"] / cap.ki_raw, gamma=gamma,
                     b=fit["b"], kappa=fit["kappa"], li_tau=li_tau, li_stages=li_stages,
                     ctrl_tau=float(ctrl_tau or 0.0), delay_s=fit["delay"], noise_density=noise,
                     diagnostics=diag, warnings=tuple(warnings))
