"""
Loop-tuning toolkit for the SXM nc-AFM controller (PLL and amplitude feedback).

Pure numpy/scipy: nothing here imports Qt or talks to hardware, so it can be tested offline.
The instrument I/O and the GUI live in ``gui/tuning_tab.py``.

Design rule: SXM's Kp/Ki are in arbitrary internal units, so nothing assumes what a raw gain means.
Loops are judged from *measured responses* to step trains; the physical loop is identified from the
same recording by regression (loopid), which also lets the tool predict untested gains.

Modules
-------
trial     StepTrain (a step train with arbitrary timing) and LoopCapture (a recording of one loop)
metrics   step-response, error-transient and noise metrics from (t, y) arrays
loopid    identify a PI loop (physical Kp/Ki, sensor pole, latency, noise) from ONE recorded train,
          and predict step metrics / stability / noise for other gains
workflow  the tuning workflow: define the test, detect the loop from the channels, analyse, classify,
          advise, and screen a (Kp, Ki) grid into a 2D map with islands of good response
"""
