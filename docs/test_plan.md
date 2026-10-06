# qPlus loop and noise test plan

Test plan for noise reduction and loop parameter discovery on the Infinity qPlus (SXM controller), with
submolecular constant-height nc-AFM as the goal. Written 6 Oct 2026 from the step tests of 5 Oct 2026
(`Tuning-PLL-AFM/Low_KiKp`, `2026-October-Amplitude`, `2026-October-PLL`).

## Where we are

| Item | Value |
|---|---|
| Sensor | f₀ = 25 564 Hz, Q = 343 561, ring-down τ = 4.28 s, linewidth f₀/2Q = 0.037 Hz, sample at 8.5 K |
| Amplitude loop (AFL) | Kp 1×10⁶, Ki 100, Tau 20 ms, InA ×1, output gain ±0.1 V. Steady Drive ≈ 10 µV at Ref 1 mV |
| PLL | Kp −200, Ki −1.65×10⁴, TC 1–5 ms. df rise ≈ 40 ms (≈ 9 Hz bandwidth), overshoot 2–3 % |
| df noise, both loops on | 81 mHz rms (≈ 14 mHz/√Hz at 2–10 Hz), too high for bond contrast |
| df noise, AFL off | 1.2 mHz rms, but probably at a much larger amplitude (DNC Drive 1 mV) |
| Amplitude noise | 0.9–1.2 % rms at every AFL gain tested: a detection floor |

**Open question that decides everything else:** is the df noise set by the signal-to-noise ratio of the deflection
signal (fix: larger calibrated amplitude, cleaner detection), by the AFL feeding Drive noise into the phase
(fix: slower AFL), or by excitation crosstalk (fix: wiring, compensation)? Tests A3, B1 and B2 answer it.

## Conventions

### Gains as G and ρ

Every gain pair is written relative to a known-good **anchor** (Kp₀, Ki₀):

```
Kp = G · Kp₀
Ki = G · ρ · Ki₀
```

* **G** (common scale) sets the loop speed. Within a sensible ratio the behaviour is monotonic in G: slow and
  quiet at low G, then overshoot, Drive at zero or loss of lock at high G.
* **ρ** (ratio factor) sets the shape at a given speed: ρ < 1 gives a slow integral tail, ρ > 1 overshoot and
  ringing. ρ = 1 keeps the anchor ratio.
* The orders of magnitude between Kp and Ki never change by accident, and regimes known to be unstable are
  never visited.

| Loop | Anchor (Kp₀, Ki₀) | Anchor ratio | Known limits |
|---|---|---|---|
| AFL | 1×10⁶, 100 | Kp/Ki = 10⁴ | G ≥ 10 pulses Drive (at 0 for 6–37 % of the time). Kp 1×10⁵ with Ki 80 (G 0.1, ρ 8) rings. |
| PLL | −200, −1.65×10⁴ | Ki/Kp = 82.5 | Kp −50 with Ki −2×10⁴ (G 0.25, ρ 4.8) rings with 20 % overshoot. |

At a new output gain or input gain the anchor moves with it: ±1 V instead of ±0.1 V multiplies both gains by 10,
InA ×10 instead of ×1 divides them by 10. G and ρ keep their meaning.

### A run

A **run** is one ordered list of conditions, measured exactly as by hand: set the parameters, wait until the loop
settles, run the step train, record a quiet window, move on. Typically one run holds ρ fixed and ramps G, or holds
G and ρ fixed and ramps one secondary parameter (Tau, TC, RollOff, output gain, InA, Ref).

Each condition records:

1. **Settle:** write the parameters, wait until the loop is steady (Drive or df flat), at most 30 s.
2. **Step train**, captured on the loop's own channels:
   * AFL: Ref ±5 %, period 8 s, 4 cycles (8 steps). Down-steps cannot be faster than the 4.3 s ring-down.
   * PLL: `use` ±1 Hz, period 3 s, 4 cycles.
3. **Quiet window:** 30 s, no steps, capturing df + Phase and QPlusAmpl + Drive (all four together once
   four-channel capture exists).
4. **Reference repeat:** the anchor condition again every 5 conditions, to separate drift from parameter effects.

**Stop rule:** a G ramp ends at its first condition that rings, loses lock or keeps Drive at 0 for more than 5 %
of the quiet window. Higher G at the same ρ is never tried.

### Metrics, every condition

| From | Metric |
|---|---|
| Step train | rise time 10–90 %, overshoot, settling to 2 %, per direction (up / down) |
| Step train | Drive at zero (fraction of time), Phase peak and decay τ (PLL) |
| Quiet window | df rms in 1–10, 10–50, 50–200 Hz; df spectrum; df noise per pixel at 10, 30, 100, 300 ms |
| Quiet window | Phase rms; QPlusAmpl rms relative to Ref; Drive mean, σ, Drive at zero |
| Quiet window | coherence Drive↔Phase, QPlusAmpl↔Phase, Drive↔df |

The decision metric is **df noise per pixel at the scan's pixel time**, subject to clean steps.
Speed alone never decides.

### General rules

* Tip retracted (≥ 5 nm) for parts A–C. Tip protection on.
* Re-sweep the resonance and Auto-phase at the start of each session. The retracted df offset was −12 mHz, a third
  of the linewidth.
* Record every capture with the standard export (CSV + JSON + PNG). Name runs `RUN-<id>` in the step-test label.
* Randomise the order of conditions within a run except along a G ramp (needed for the stop rule).

## A. Measurement chain (once, tip retracted)

### A1. Channel timing and resolution (5 min)

* **Do:** capture 10 s of each channel with nothing changing: QPlusAmpl, Drive, Phase, df, Frequency. Then toggle
  `use` ±1 Hz 50 times at random intervals while capturing `Frequency` (DAC9).
* **Get:** true update rate and value step of each channel (df: 2.3 mHz); latency and jitter from DDE write to
  visible change.
* **Decide:** event-time correction for all step analyses (old captures showed up to 0.1 s jitter); the real
  bandwidth each channel can show.

### A2. Detection noise floor (5 min)

* **Do:** AFL off, PLL off, Drive = 0. Record QPlusAmpl and Phase for 60 s with `use` = f₀, f₀ ± 50 Hz,
  f₀ ± 500 Hz.
* **Get:** detector noise density near f₀ (V/√Hz). With the pm/mV calibration (A5), the predicted
  detector-limited df noise for any amplitude and bandwidth.
* **Decide:** the noise target. When measured df noise reaches this prediction, loop tuning is finished.
* **Optional:** 10 min at f₀ with the excitation off: the thermal peak (≈ 0.25 pm rms at 8.5 K, k ≈ 1800 N/m)
  gives an independent amplitude calibration.

### A3. Excitation crosstalk (5 min)

* **Do:** AFL off, PLL off. `use` = f₀ ± 200 Hz and f₀ ± 2 kHz, where the sensor barely responds. Drive = 0, 10, 30,
  100 µV, 1 mV. Read QPlusAmpl and Phase at each level.
* **Get:** direct feedthrough from Drive to the detected signal (V/V and its phase), compared with the
  on-resonance gain (≈ 1 mV per 10 µV).
* **Cross-check:** fit the last resonance sweep with a Lorentzian plus a constant complex background
  (Fano-type). The background is the same feedthrough.
* **Decide:** if the feedthrough is above about 1 % of the signal, AFL Drive noise can appear directly in Phase.
  Then slow the AFL (part C) and look at excitation wiring and shielding.

### A4. Ring-down and Q (30 s; also daily)

* **Do:** PLL locked, AFL off, Drive set to 0. Fit the decay of QPlusAmpl. Repeat at Ref 1 and 4 mV.
* **Get:** Q = π f₀ τ; whether damping depends on amplitude.
* **Decide:** tip and sensor health (part E).

### A5. Amplitude calibration, pm per mV (10 min)

* **Do:** automate the calibration tab: STM constant current, Ref 0.5 → 4 → 0.5 mV in 8 steps, at two current
  setpoints. Fit z against Ref.
* **Get:** pm per mV, with up/down hysteresis as a check. Compare with A2's thermal estimate if available.
* **Decide:** the Ref that gives the target amplitude for constant-height work (tens of pm to ≈ 100 pm).

## B. Noise sources (tip retracted)

### B1. Two-loop noise matrix (≈ 45 min)

* **Do:** PLL at its anchor. For each combination, settle and record a 30 s quiet window (no steps):

| Factor | Levels |
|---|---|
| AFL | off at matched Drive (≈ 10 µV per mV of Ref); on at G = 0.3, 1, 3 (ρ = 1) |
| Ref | 0.5, 1, 2, 4 mV |

* **Get:** df noise (1–200 Hz) and Phase rms against Ref and AFL G.
* **Decide:**
  * df noise ∝ 1/Ref and no dependence on G or on AFL on/off → signal-to-noise limited. Priority goes to A5
    (largest usable amplitude) and to the detection chain (B5); part C changes little.
  * df noise rises with AFL G, or AFL-off is clearly quieter at matched amplitude → the AFL injects noise. Run C1
    at low G, and check A3.

### B2. Coherence (from B1 data, or 4 × 30 s)

* **Do:** with both loops on at the anchors, record simultaneous pairs: Drive + Phase, QPlusAmpl + Phase,
  Drive + df.
* **Get:** magnitude-squared coherence against frequency.
* **Decide:** strong Drive↔Phase coherence → Drive noise reaches the PLL (crosstalk or phase-reference error).
  Additive detector noise gives uncorrelated amplitude and phase, so QPlusAmpl↔Phase coherence points to a
  common electronic source instead.

### B3. Amplitude–frequency coupling (5 min)

* **Do:** PLL locked, AFL at anchor. Ref steps of ±10 % and ±50 % (period 10 s). Capture df + Phase.
* **Get:** df shift per % amplitude far from the surface (should be ≈ 0).
* **Decide:** coupling × amplitude noise = its share of the df noise.

### B4. Allan deviation of df (30 min)

* **Do:** both loops on at anchors, tip retracted, record df continuously for 30 min (Live Scope or chunked
  captures).
* **Get:** Allan deviation against averaging time: white noise, flicker floor, drift.
* **Decide:** best averaging time for spectroscopy; longest constant-height frame before drift dominates.

### B5. Spectral-line hunt (≈ 30 min)

* **Do:** 5 min captures of df and QPlusAmpl while switching sources one at a time: turbo / ion pumps, cryostat
  state and boil-off, lights, nearby instruments, ground straps, preamp cable routing.
* **Get:** automatic peak list per capture compared with a reference capture.
* **Targets:** peaks seen on 5 Oct at 10.5, 14.7, 18.9, 27.3 Hz (15:58) and 4.2, 9.2, 11.7, 16.7 Hz (16:02), and
  a possible 2.1 Hz comb.

### B6. Bias, z and scanner crosstalk (10 min)

* **Do:** tip 5–10 nm from the surface. Ramp bias (±1 V), ramp z (±2 nm), run a 10 nm x/y raster. Record df and
  Phase.
* **Get / decide:** all should be flat. Anything that follows bias, z or the scan line is pickup that will show up
  in constant-height images.

## C. Parameter discovery: G / ρ runs (tip retracted)

Run C1 and C2 in the configuration you will image with: the other loop on at its current best values, and Ref
at the value chosen in A5.

### C1. Amplitude loop

| Run | Fixed | Ramped | Purpose |
|---|---|---|---|
| C1-a | ρ = 1, Tau 20 ms, OG ±0.1 V, InA ×1 | G = 0.1, 0.2, 0.3, 0.5, 1, 2, 3 | speed vs Drive noise vs df noise along the anchor ratio |
| C1-b | ρ = 0.3 | same G list | less integral: slower tail, possibly quieter |
| C1-c | ρ = 3 | same G list | more integral: faster recovery, risk of ringing at low G |
| C1-d | best (G, ρ) from a–c | Tau = 5, 10, 20, 50, 100 ms | input filtering; expected small effect because most Drive noise is at 1–10 Hz |
| C1-e | best (G, ρ, Tau) | OG ±0.1 V and ±1 V (gains ×10 at ±1 V) | does output resolution matter at 10 µV of Drive? |
| C1-f | best (G, ρ, Tau) | InA ×1, ×10 (gains ÷10 at ×10) | input gain vs noise; explain the 6–8 % amplitude offset at ×10 |

Expected outcome: G between 0.3 and 1 at ρ = 1 (Kp 3×10⁵–1×10⁶, Ki 30–100), the lowest G whose recovery
(≈ 0.5–1.5 s) is still acceptable for constant height.

### C2. PLL

| Run | Fixed | Ramped | Purpose |
|---|---|---|---|
| C2-a | ρ = 1, TC 5 ms | G = 0.25, 0.4, 0.6, 1, 1.5, 2 | bandwidth vs df noise along the anchor ratio |
| C2-b | ρ = 0.5 | same G list | less overshoot, slower phase recovery |
| C2-c | ρ = 2 | same G list, stop rule applies | faster phase recovery, more overshoot |
| C2-d | best (G, ρ) | TC = 0.5, 1, 2, 5, 10 ms | demodulator filtering vs loop phase margin |
| C2-e | best (G, ρ, TC) | RollOff 6, 12, 24 dB/oct | same |
| C2-f | best | step size ±0.1, ±1, ±5 Hz | linearity and lock robustness for jumps near molecules |

Expected outcome: a choice between the current ≈ 9 Hz loop (G = 1) for pixels of ≈ 0.1 s, and a ≈ 4 Hz loop
(G ≈ 0.4) for slow constant-height frames, decided by df noise per pixel.

### C3. Cross-coupling (10 min)

* **Do:** at the chosen settings, a PLL step train recording QPlusAmpl + Drive, and an AFL step train recording
  df + Phase.
* **Get:** how much each loop disturbs the other, and on what timescale.

## D. Validation at the surface

### D1. Noise per pixel against pixel time
From C2's quiet windows, tabulate df noise per pixel for 10, 30, 100, 300 ms. This sets the scan speed.

### D2. Repeated line scan
At constant height over a molecule, scan one line 30–50 times. Line-to-line spread = noise + drift; average = real
contrast. Their ratio is the figure of merit.

### D3. Amplitude choice
df(z) over a molecule at three calibrated amplitudes (from A5). Pick the one with the best bond contrast relative
to noise.

### D4. Drift at a fixed point
Hold one point at constant height for 5–10 min with B4's settings. Gives the allowed frame time and how often drift
correction must update.

## E. Daily fingerprint (≈ 5 min, unattended)

Resonance sweep (f₀, Q) → ring-down (A4) → detection floor (A2, 30 s) → 60 s of all four channels at standard
settings → one PLL and one AFL step train at the anchors. Store as JSON with the export metadata and trend it.
Alerts: Q down by > 10 %, retracted Drive up by > 20 %, df noise up by > 30 % against the last week.

## Order of work

| Session | Tests | Time | Answers |
|---|---|---|---|
| 1 | A1, A3, B1, B2, B4 | ≈ 1.5 h | where the df noise comes from |
| 2 | A5, A2, B3, B5 (if lines matter) | ≈ 1.5 h | amplitude in pm, noise target, pickup |
| 3 | C1 a–c, C2 a–c | ≈ 2 h | G and ρ for both loops |
| 4 | C1 d–f, C2 d–f, C3 | ≈ 1.5 h | secondary parameters, interaction |
| 5 | D1–D4 | at the surface | scan settings for constant height |
| daily | E | 5 min | health |

## Software needed

In order of payoff:

1. **Run sheet runner.** Done: the Tuning tab (`gui/runsheet_tab.py`, `gui/sheet_runner.py`). It builds sheets from
   G / ρ / setting ramps, runs set → settle → step train → quiet window per condition, applies the stop rule and
   reference repeats, and shows one plot set per condition plus ramp trends.
2. **Four-channel capture.** Done inside the Tuning tab (one round-robin capture of df, Phase, QPlusAmpl, Drive
   per condition). The Scope tab still captures two channels.
3. **Quiet-window metrics.** Done in `tuning/noise.py` (spectrum, band rms, coherence, noise per pixel). Allan
   deviation (B4) is still to add.
4. **`AnfatecSXMWriter` verified on the real SXM** for Tau, TimeConstant, RollOff, InA and output range, so C1-d–f
   and C2-d–e run without hand changes.
5. **Event times from echo channels** (`Frequency` for `use` steps; `align_events` for Ref steps).
