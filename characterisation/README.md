# qPlus sensor characterisation without a surface

Automated, repeatable characterisation of the qPlus sensor **by itself** (tip retracted, no sample), built on
`sxm_anfatec`: the pure-logic package `characterisation/` and the **Sensor Characterisation** tab
(`gui/characterisation_tab.py`). It never moves z, never writes a feedback gain and never switches a loop on.

**Implemented now:** stage 0 (preflight), stage 1 (zero-drive baselines), stage 2 (drive × lock-in TC
matrix with repeats, interleaving and drift anchors) and stage 4 (ring-down analysis of the stage-2
records). Stages 3 and 5–8 are designed below and come later, as separate stages.

**In the app** (`python -m sxm_ncafm_control.app`), tab *Sensor Characterisation*:

1. In SXM: tip retracted, Amplitude feedback OFF, PLL Kp = Ki = 0, DNC Range (output gain) fixed (e.g. ±0.1).
2. Run on: *SXM instrument*. Tick *Amplitude feedback is switched OFF* and *Tip retracted, no surface*.
3. *Preflight* (~15 s). If it says REFUSED, the reason is in the status line and the run folder.
4. *Blocks this run* = 1, *Run* (headroom + block 1, ~29 min). Then *Resume run...* on the same folder for
   blocks 2 and 3 (~25 min each). Each finished batch is analysed automatically (`derived/<time>/report.md`).
5. *Stop* at any time: the record ends, the drive goes to 0 and the TimeConstant back to its start value.

The tab shares the app's driver handle: stop Scope, Live Scope and Single Step captures first. Results go
to `qplus_characterisation_results/` **next to** the `sxm_ncafm_control` folder (never into this repository);
*Choose...* picks another folder.

**Without the app** (it holds the only driver handle, so close it first), from the folder holding
`sxm_ncafm_control` and `sxm_anfatec`:

```
python -m sxm_ncafm_control.characterisation.run     sxm_ncafm_control/characterisation/configs/open_loop_matrix.json --preflight
python -m sxm_ncafm_control.characterisation.run     <config.json> --blocks 1        # STOP file or Ctrl+C to stop
python -m sxm_ncafm_control.characterisation.run     --resume <run folder> --blocks 1
python -m sxm_ncafm_control.characterisation.analyse <run folder>                     # new derived/<time>/ each call
python -m sxm_ncafm_control.characterisation.run     <config.json> --sim [--sim-fault afl|pll|compress|pll-gains]
```

The CLI reads the operator confirmations from the config (`operator.*`, false in the shipped file). Tests (no
hardware): `tests/test_characterisation.py` (logic, safety, analysis on the simulator) and
`tests/test_characterisation_tab.py` (the tab, offscreen). Plots need matplotlib (not in requirements.txt:
without it the report and tables are written and the plots are skipped).

---

## 1. Discovered interface (what the campaign calls)

The campaign does not add any communication code. It uses only these existing calls:

| Purpose | Real API | Name / channel | Status on the instrument |
|---|---|---|---|
| Set drive | `AnfatecSXMWriter.set` (types, Enter, reads back) | `dnc.drive` (DNC window, 4th number field) | **name by position, unconfirmed** → verified physically by the drive echo (§5 V1) |
| Set lock-in TC | `AnfatecSXMWriter.set` (combo) | `dnc.tc` | combo writes work (`dnc.output_gain`, `lockin.tc` confirmed); `dnc.tc` itself untested → V2 |
| List TC levels | `AnfatecSXMWriter.options` | `dnc.tc` | read at preflight; levels not offered are refused (`lockin.tc` has no 3 ms, for example) |
| Read settings | `AnfatecSXMWriter.get` → `AnfatecSXMBridge.get` | `dnc.*`, `amp.*`, `pll.*`, `feedback.mode`, `z.feedback_off`, `stm.status` | read-only, safe during measurement |
| Fast signals | `SXMIOCTL.read_scaled` | `QPlusAmpl` (DAC12, V), `Phase` (DAC13, °), `Drive` (DAC10, V), `Frequency` (DAC9, Hz), `InA` (ADC4, V) | polled; **no fixed sample rate** (each sweep is time-stamped) |
| Units / scales | `sxm_anfatec.CHANNELS` | per channel | recorded with every capture; the scales are SXM's, not independently checked |
| DDE (not used) | `RealDDEClient.send_dncpara(4, v)` | drive over DDE | a fallback if `dnc.drive` turns out wrong |

**Gaps that shape the design**

- **The AFL On/Off switch cannot be read.** The bridge maps Ref, Ki, Kp, Pull back, Pull speed and Tau, but
  not the switch. The saved gains do not show whether the loop runs (in your earlier files AFL was off with
  non-zero gains). So "AFL off" means the operator confirms it (`operator.afl_off_confirmed`) **and** the
  runner checks it physically. With AFL off, the Drive DAC follows `dnc.drive` in proportion and stays flat.
  A running AFL breaks either property.
- **"PLL off" means Kp = Ki = 0** (the manual), read back before and after every condition. Gains of zero can
  still freeze an old frequency offset, so the runner also checks that the Frequency DAC equals the used
  frequency and does not move.
- **Units of the `dnc.drive` field are not readable.** All drive levels are in "SXM units". The preflight
  measures k_cal = Drive-DAC volts per SXM unit and checks that the smallest drive is ≥ 200 DAC counts.
- **QPlusAmpl is in volts.** Nothing here converts it to displacement.
- **Output gain auto-ranges with the drive** (manual), which changes the DAC resolution. The runner records
  `dnc.output_gain` before and after each condition and warns when it changes. Fix it in SXM (e.g. ±0.1).

## 2. What the layer adds

```
config.py      JSON config, validation, schedule (seeded randomized blocks + anchors), time estimate
instrument.py  QPlusSxm: write ALLOWLIST (drive, TC only), snapshot(), record() = polled capture with
               writes at scheduled offsets (t_cmd / t_ack marks) and a hardware-limit trip
sim.py         SimQPlus: same methods; driven damped oscillator + n-stage lock-in, fault switches
runner.py      Campaign: preflight → headroom → blocks; per-condition read-back, checks, save, safe stop
analysis.py    per-condition metrics and campaign statistics (pure numpy/scipy)
analyse.py     CLI: derived/<time>/ report.md, summary.json, conditions.csv, cells.csv, plots/
filters.py     lock-in filter relations (n RC stages): ENBW, settling time, group delay
plots.py       diagnostic PNGs (matplotlib, optional)
run.py         CLI with Ctrl+C / STOP-file safe stop, batches, resume
gui/characterisation_tab.py   the tab: CampaignThread runs Campaign.run(progress, stop) with conn.driver
```

The writes go through `AnfatecSXMWriter` (Win32 messages, safe from the worker thread), not DDE, which must
stay on the GUI thread. `MainWindow` stops a running campaign (and waits for the drive / TC restore) before a
reconnect and on close.

**Safety rules (enforced in code, tested)**

| Rule | How |
|---|---|
| Only drive and TC are written | allowlist in the instrument layer: any other name raises `SafetyError` |
| No z / approach / surface feedback | never written; `z.feedback_off`, `stm.status`, `feedback.mode` only logged |
| No silent AFL/PLL | config must say `loops: {afl: off, pll: off}` (anything else is refused); PLL gains read back; physical echo / frequency checks |
| Bounded | `limits.max_total_min`, `max_conditions` per invocation (refused if exceeded); `--blocks`, `--max-minutes` |
| Timeouts | each SXM message 5 s (writer); each condition ≤ `condition_timeout_s`; the record length is fixed |
| Hardware limits | during every record: QPlusAmpl > `amp_max_v` or \|InA\| > `ina_max_fraction` × range → drive set to off at once, campaign aborted (in the headroom pass: the ladder stops) |
| Safe stop | Ctrl+C, a `STOP` file in the run folder, or the GUI stop: the record ends, drive → `off_drive`, TC → its start value, both read back; a failed restore says what to set by hand |
| Raw data kept | every file is created with exclusive mode (`x`): nothing is ever overwritten; analysis writes a new `derived/<time>/` |

## 3. Factor matrix (all stages)

| Factor | Levels | Stage | Responses compared |
|---|---|---|---|
| Drive amplitude (SXM units) | 2.5e-6, 5e-6, **1e-5 (reference)**, 2e-5, 4e-5. The headroom pass decides how far up to go | 2 | A_ss, phase, Drive DAC, A/D slope and offset, compression |
| DNC lock-in TC | 0.2, 1, 5, 10, 20, 50 ms (checked against SXM's list) | 1, 2 | baseline mean/std/ASD, ring-down τ and delay, steady noise |
| Drive state | zero (baseline conditions) vs driven | 1 | noise floor vs TC (white noise: R ∝ TC^-½) |
| Excitation frequency | coarse ±5 Hz / 0.25 Hz, fine ±1 Hz / 0.05 Hz around f0 | 3 (later) | f0, Q (Lorentzian), phase curve, drive-dependent shift |
| AFL gains | one at a time from conservative values | 5 (later) | setpoint error, settling, overshoot, drive effort |
| PLL gains | one at a time, AFL off | 6 (later) | frequency error, settling, overshoot, phase |
| Repetition / order | 3 blocks, shuffled per block (seed stored), anchor at each block start + end | 1–2 | CI per cell, drift (%/h) |

## 4. Run schedule (default config, `configs/open_loop_matrix.json`)

One **driven condition** is one continuous record (36.4 s at Q = 44 451):

```
pre 2 s (zero drive) | drive step on: ring-up 8τ = 4.4 s | steady 5 s | drive step off: ring-down + baseline 25 s
```

TC is written before the record (wait max(1 s, 10 TC)). One **baseline condition** records 2 s at the old TC,
writes the new TC inside the record (so the transition is kept), then 30 s at zero drive.

| Part | Conditions | Time |
|---|---|---|
| Preflight (read-backs, rate, drive echo off→1e-5→2e-5→off) | – | ~15 s |
| Headroom pass at TC 1 ms, ascending drive, stops at compression > 3 %, InA > 40 % of range, or QPlusAmpl > 80 % of the limit | 5 | ~3.5 min |
| Block 1: anchor + 30 drive × TC cells + 6 baselines, shuffled | 37 | ~25 min |
| Block 2, block 3 (3 also closes with an anchor) | 37 + 38 | ~25 + 26 min |
| **Total** | 117 records | **~79 min** (run as 3 batches with `--blocks 1`) |

Raw data: ~0.5 MB per condition (lossless int32 DAC counts + float32 time), ~60 MB per campaign.

## 5. Instrument-specific items to verify (preflight does V1, V3–V6 itself)

| # | Item | How it is checked | If it fails |
|---|---|---|---|
| V1 | `dnc.drive` is the drive field and AFL is off | drive echo: Drive DAC follows off→d→2d in proportion (±3 %), flat (std/level < 0.3 %) | refused: check the AFL switch; if AFL is off, the field mapping is wrong (fallback: DDE `DNCPara(4, v)`) |
| V2 | `dnc.tc` writes reach the lock-in | analysis: baseline ∝ TC^-½ and ring-down delay ≈ 2·TC (12 dB/oct) | if the baseline does not change with TC, the write only changed the GUI |
| V3 | TC levels exist (0.2 ms and 5 ms in particular) | `options('dnc.tc')` at preflight | refused, with SXM's list printed: edit `factors.tcs` |
| V4 | PLL off | Kp = Ki = 0 read back; Frequency DAC = used frequency ± 5 mHz, flat | refused / aborted |
| V5 | drive resolution | smallest drive ≥ 200 DAC counts | refused: raise the drives or fix the output gain |
| V6 | polling rate | ≥ 50 % of 2000 sweeps/s (warning below 80 %) | lower `sample_rate_hz` |
| V7 | Drive DAC 0 at zero drive | baseline conditions | aborted (state) |
| V8 | QPlusAmpl / InA headroom | headroom pass | the ladder stops; only the passing levels are used |
| V9 | `amp_max_v` = 50 mV is a guess | — | set it from the first headroom result |

## 6. Configuration format

`configs/open_loop_matrix.json` (all keys; unknown keys are refused):

```jsonc
{
  "kind": "qplus_open_loop_matrix", "version": 1, "name": "open_loop_matrix", "notes": "...",
  "operator":   {"afl_off_confirmed": false,          // set true after checking SXM (switch not readable)
                 "no_surface_confirmed": false},
  "instrument": {"drive_param": "dnc.drive",          // VERIFY (V1)
                 "tc_param": "dnc.tc",                // VERIFY (V2)
                 "freq_param": "dnc.used_freq", "rolloff_param": "dnc.rolloff",   // read only
                 "output_gain_expected": null,        // e.g. "±0.1": refused if SXM shows another range
                 "channels": ["QPlusAmpl","Phase","Drive","Frequency","InA"],
                 "sample_rate_hz": 2000},
  "expect":     {"f0_hz": 25563.8, "q": 44451},      // sets the waits (tau = Q/(pi f0)) and a sanity check
  "limits":     {"drive_max": 1e-4, "amp_max_v": 0.05, "ina_max_fraction": 0.5, "drive_echo_rel_tol": 0.03,
                 "drive_flat_rel": 0.003, "freq_tol_hz": 0.005, "compression_max": 0.03, "min_dac_counts": 200,
                 "max_total_min": 120, "max_conditions": 250, "condition_timeout_s": 240},
  "timing":     {"pre_s": 2, "settle_tau": 8, "steady_s": 5, "off_s": 25, "discard_s": 5, "baseline_s": 30,
                 "tc_settle_s": 1, "overhead_s": 4},
  "design":     {"repeats": 3, "seed": 20261106, "order": "randomized_blocks", "anchors": true, "headroom": true},
  "reference":  {"drive": 1e-5, "tc": "1 ms"},
  "factors":    {"drives": [2.5e-6, 5e-6, 1e-5, 2e-5, 4e-5],
                 "tcs": ["0.2 ms","1 ms","5 ms","10 ms","20 ms","50 ms"],
                 "baseline_tcs": ["0.2 ms","1 ms","5 ms","10 ms","20 ms","50 ms"]},   // [] = no stage 1
  "loops":      {"afl": "off", "pll": "off"},         // the only values this stage accepts
  "off_drive": 0.0, "end_drive": "start"              // after a normal end: the drive the run found
}
```

A **stage-1-only** run uses `"drives": []`: only baselines and the reference anchors run.

## 7. What is recorded

```
qplus_characterisation_results/<YYYYmmdd-HHMMSS>_<name>/
  campaign.json        config as run, environment (versions, host, channel units/scales), timing summary
  preflight_N.json     every read-back and check, k_cal, DAC zero, rate, TC options, start state
  raw/preflight_echo_N.npz/.json
  headroom.json        ladder results, allowed drives, the stopping rule
  schedule.json/.csv   every planned condition (idx, block, kind, drive, TC, estimate), seed
  raw/cNNNN_<kind>_b<block>_D<drive>_TC<tc>.npz    t (float32 s), cnt_<ch> (int32 counts), scale_<ch>
  raw/cNNNN_....json   see below
  run.log, end_N.json, resume_N.json
  derived/<time>/      report.md, summary.json, conditions.csv, cells.csv, plots/*.png
```

Each condition's sidecar holds the requested items: timestamp and file name, requested settings, read-back
snapshots **before and after** (drive, used frequency, TC, roll-off, input/output gain, Q/fPeak status,
AFL Ref/Kp/Ki/Tau, PLL Kp/Ki, feedback mode, z status), the writes made (before/after), marks per event
(`at`, `t_cmd`, `t_ack`, value, read-back), initial and final drive, measured Drive DAC and Frequency DAC,
loop states (with how each was established), requested and achieved sampling rate, channel units and
scales, run order (idx, block), status, aborts, warnings, online metrics and the analysis parameters.

## 8. Analysis (every number carries its method)

| Metric | Window / method |
|---|---|
| Step times | from the **Drive DAC** crossing its midpoint; command→DAC delay reported separately |
| Steady A_ss, phase, Drive DAC | last `steady_s` before the step-off command minus 0.1 s; block SE; circular mean for phase; drift over the window |
| Ring-down τ, Q | R² = P0·exp(−2(t−t_s)/τ) + c, weighted LSQ, c = noise power (E[R²] = A² + 2σ²: subtract power, not amplitude). t_s = DAC step + filter settling to 0.1 %; end at 5× the floor or 6 τ. SE × √(N/N_eff), N_eff = window × 2·ENBW. Cross-check: log-linear fit where R > 10× the floor |
| Ring-down delay | where the fitted exponential meets A_ss, compared with the group delay n·TC (not resolved when TC is shorter than the sample period) |
| Ring-up (secondary) | \|A_ss(1 − e^{−(1/τ + j2πδ)t})\| → τ and \|detuning\| δ (beats when the drive is off resonance) |
| Baseline | from ≥ 5 s after the DAC step off, **extended automatically** until the fitted tail is < 0.1 × the baseline std (the 5 s you used is too short at the larger drives and longer TCs; the value used is reported); mean, std, block SE, median, slope, ASD in 0.5–5 / 5–50 / 50–500 Hz |
| Noise density | Rayleigh model: E[R] = σ√(π/2), σ² = S·ENBW per quadrature |
| Cells | mean, SD, t-distribution 95 % CI over repeats |
| Linearity | per TC: A_ss = k·(Drive DAC − zero) + b, max relative residual, A/D relative to the lowest drive |
| Readout vs mechanics | τ regressed on TC: \|slope/SE\| < 2 and a negligible change → the decay is mechanical |
| Drift | anchors: trend in %/h of A_ss and τ |

## 9. Example output (SIMULATED: format only, not a measurement)

Default config on `--sim` (f0 25 563.8 Hz, Q 44 451, 1e-7 V/√Hz detector noise), 117 records, all `ok`:

```
## 3. Ring-down: mechanical decay vs readout filter
Pooled τ = 0.55353 ± 4.4e-05 s (95 % CI, n = 99) → Q = 44454 ± 4 (f0 = 25563.8 Hz)
τ vs TC: slope 0.00258 s/s ± 0.0014 (|slope/SE| = 1.85; change over the TC range +0.02%):
  no detectable TC dependence: the fitted decay is mechanical, not the readout filter.
| TC     | delay mean (s) | expected n·TC (s) |
| 1 ms   | 0.00157        | 0.002             |
| 10 ms  | 0.0197         | 0.02              |
| 50 ms  | 0.104          | 0.1               |
## 4. Zero-drive baseline
log-log slope of the baseline mean vs TC = -0.516 (white noise: −0.5)
| TC     | n  | mean (V) | std (V)  | discard used (s) |
| 0.2 ms | 18 | 2.39e-06 | 1.25e-06 | 5.22             |
| 50 ms  | 18 | 1.37e-07 | 7.14e-08 | 6.65             |
## 2. Linearity: slope 99.99 ± 0.0015 V/V per TC, max residual 0.03 %
```

The fault switches show the safety paths. `--sim-fault afl` is refused at the drive echo. `pll` is refused or
aborted on the Frequency DAC check. `pll-gains` is refused at preflight. `compress` stops the headroom ladder.

## 10. Later stages (designed, not built)

**Stage 3, open-loop frequency response.** This needs writes to `dnc.used_freq` (by position, unconfirmed:
verify like V1) and **`Auto 0°` off** (when it is on, every change of `use` re-phases, and the phase curve
is lost). The switch is not mapped either, so the operator confirms it, and a phase that snaps to 0 after
each step aborts the sweep. Stepped sweep: the coarse sweep is ±5 Hz in 0.25 Hz steps, the fine sweep
±1 Hz in 0.05 Hz steps (the linewidth f0/Q ≈ 0.58 Hz). Each point waits 6τ (3.3 s) and averages 1 s, so a
fine sweep takes about 3 min. Sweep up and down to expose drift and hysteresis. Fit amplitude and phase
together: z(f) = A0·e^{jφ0}/(1 + 2jQ(f−f0)/f0) + background. Repeat the fine sweep at 3 drive levels for
the amplitude-dependent shift. SXM's own sweep stays a cross-check (`dnc.q` and `dnc.f_peak` are already
logged).

**Stages 5–7 (AFL alone, PLL alone, then both).** `sxm_ncafm_control` already has the machinery for these:
the run-sheet Tuning tab, step tests, quiet-window noise and `tuning/loopid.py`, all in this app. Reuse
them and do not duplicate them here. Two inputs come from this tool: the open-loop plant (gain A/D and τ, now known with
CIs) feeds `loopid`; and Q has moved (175 k on 7 Oct, ≈44 k now), so the gains in `TUNING_STATUS.md` scale
as 1/Q and must be re-derived, not reused. Before any AFL stage, **the AFL On/Off switch must be mapped
in `sxm_anfatec`** (bridge + writer, PR), so the loop state can be read back as the brief requires.

**Stage 8, stiffness by thermal noise.** It needs (a) a voltage-to-displacement sensitivity S from an
independent calibration (the amplitude calibration tab needs a surface; otherwise the qPlus charge
sensitivity and the preamp gain), (b) the sensor temperature, and (c) a model of the thermal peak (a
Lorentzian of width f0/Q at f0) on the detector floor. Then k = k_B·T / ⟨z²⟩. Stage 1 already tests
whether it is feasible. Detector noise gives a baseline ∝ TC^-½. Thermal motion of the sensor is narrower
than the ENBW at every TC here, so it adds a TC-independent floor. A flattening of the log-log slope means
the thermal peak is visible. As an order of magnitude only (k ≈ 1800 N/m, 8.5 K, and the 6 Oct
"1 mV ≈ 60 pm"): z_rms ≈ 0.26 pm ≈ 4 µV of QPlusAmpl. If the detector floor at long TC is below that, the
thermal peak would dominate there, so it is worth checking. A stage-1b run at an off-resonance excitation frequency (f0 + 50 Hz, again needing
`used_freq` writes) separates the two directly.
