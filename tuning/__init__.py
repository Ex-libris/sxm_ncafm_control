"""
Loop-tuning toolkit for the SXM nc-AFM controller (PLL and amplitude feedback).

Pure numpy/scipy: nothing here imports Qt or talks to hardware, so everything
can be developed and tested offline.

Design rule: SXM's Kp/Ki are in arbitrary internal units, so nothing in this
package assumes what a raw gain means physically. Gains are judged only from
*measured responses* (see metrics.py) and searched relative to a known-good
baseline (see planner.py). The simulator's raw->physical mapping is an
assumption that identify.py can recover from real step responses.

Modules
-------
metrics    step-response and noise metrics from (t, y) arrays
simulator  virtual qPlus + lock-in + PLL for offline development
planner    scan-speed requirements, trial analysis and the guided search
identify   fit the unknown raw->physical gain scales from measured trials
"""
