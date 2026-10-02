"""
Loop-tuning toolkit for the SXM nc-AFM controller (PLL and amplitude feedback).

Pure numpy/scipy: nothing here imports Qt or talks to hardware, so it can be tested offline.
The instrument I/O is ``gui/tuning_runner.py``, the GUI ``gui/tuning_tab.py``.

Design rule: SXM's Kp/Ki are in arbitrary internal units, so nothing assumes what a raw gain means.
Loops are judged from *measured responses* to step trains, every one started from the same verified baseline.

Modules
-------
trial     StepTrain (a step train with arbitrary timing) and LoopCapture (a recording of one loop)
metrics   step-response, error-transient and noise metrics from (t, y) arrays
workflow  one step test: the plan, channel detection, and the analysis of a recorded train
explore   judging a test per step direction, baseline-recovery checks, and the staged exploration of
          the Kp-Ki plane (decade grid -> edge along g -> ratio search -> repeats) with its trade-off
loopid    identify a PI loop (physical Kp/Ki, sensor pole, latency, noise) from ONE recorded train
          (a diagnostic: the explorer judges measured responses only)
"""
