"""Test fixtures: recordings of the virtual instrument in the shape the real workflow receives them."""
import numpy as np

from sxm_ncafm_control.tests.sim import (AFLScale, AFLSetup, PLLSetup, SXMScale, run_afl_capture, run_pll_capture)
from sxm_ncafm_control.tuning.trial import StepTrain
from sxm_ncafm_control.tuning.workflow import CapturedTest, StepTestPlan

PLL_SETUP, PLL_SCALE = PLLSetup(), SXMScale()
AFL_SETUP, AFL_SCALE = AFLSetup(), AFLScale()


def pll_plan(**kw):
    return StepTestPlan(**{**dict(loop="pll", base=25000.0, step=1.0, hold_s=0.5), **kw})


def afl_plan(**kw):
    return StepTestPlan(**{**dict(loop="afl", base=6.0, step=0.10, hold_s=1.0, lead_s=1.5), **kw})


def record_pll(plan, kp, ki, seed=1, latency_s=0.008, offset_hz=0.3, jitter_s=0.0, setup=PLL_SETUP, scale=PLL_SCALE):
    """A PLL test as the workflow gets it: channels by SXM name, event times as the host clock saw them."""
    train = StepTrain(tuple(plan.event_times), tuple(l - plan.base for l in plan.levels))   # offsets about f_res
    cap = run_pll_capture(kp, ki, train, setup, scale, seed=seed, latency_s=latency_s, offset_hz=offset_hz, tail_s=plan.hold_s + plan.tail_s)
    rng = np.random.default_rng(seed + 100)
    ev = [t + (rng.uniform(-jitter_s, jitter_s) if jitter_s else 0.0) for t in plan.event_times]
    return CapturedTest(plan=plan, t=cap.t, channels={"df": cap.u, "Phase": cap.y}, event_times=ev, kp=kp, ki=ki), cap


def record_afl(plan, kp, ki, seed=1, latency_s=0.005, jitter_s=0.0, setup=AFL_SETUP, scale=AFL_SCALE):
    train = StepTrain(tuple(plan.event_times), tuple(plan.levels))
    cap = run_afl_capture(kp, ki, train, setup, scale, seed=seed, latency_s=latency_s, tail_s=plan.hold_s + plan.tail_s)
    rng = np.random.default_rng(seed + 100)
    ev = [t + (rng.uniform(-jitter_s, jitter_s) if jitter_s else 0.0) for t in plan.event_times]
    return CapturedTest(plan=plan, t=cap.t, channels={"QPlusAmpl": cap.y, "Drive": cap.u}, event_times=ev, kp=kp, ki=ki), cap
