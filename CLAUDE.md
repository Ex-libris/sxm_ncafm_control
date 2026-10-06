# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyQt5 desktop GUI for tuning NC-AFM (qPlus) loops on an **Anfatec SXM** SPM controller. It talks to the instrument over **two independent channels**, and reads SXM's parameters back through a third, passive one:

| Channel | Module | Transport | Used for |
|---|---|---|---|
| **DDE** | `SXMRemote.py` → `dde_client.py` | Win32 DDEML (`user32`) to the running SXM software (service `SXM`, topic `Remote`) | Writing parameters (`EditNN`, DNC), setting Z, toggling feedback |
| **IOCTL** | `device_driver.py` | `DeviceIoControl` on `\\.\SXM` (the Anfatec kernel driver), bypassing SXM software | Fast reads of any hardware channel (scope, live Z) |
| **GUI read-back** | `sxm_state.py` → `AnfatecSXMBridge.py` | Read-only Win32 queries (`WM_GETTEXT`, `BM_GETCHECK`, ...) of the running `Femto_28_4.exe` windows | Reading the parameter values SXM shows (gains, DNC frequencies/Drive, Q, f0, time constants, ...) |

Windows-only (pywin32 + ctypes on `user32`). `device_driver.py` imports `win32file` at module top, so even offline mode needs Windows + pywin32.

## Commands

There is no build step or linter. Tests are stdlib `unittest`, no hardware (Qt ones run offscreen; `test_sxm_state` fakes the SXM windows with a bridge backend). `test_hidpi` currently crashes the interpreter (PenScaling), and the real-time tuning tests (`test_sheet_runner`, `test_runsheet_tab`) take about a minute. They run the loops on pure-Python fakes that compete with the GUI thread for the GIL, so they check what the runner does, not how clean the steps look: real performance is judged on the instrument. Run from the **parent** folder:

```bash
PYTHONPATH=. python -m unittest discover -s sxm_ncafm_control/tests -v      # all
PYTHONPATH=. python -m unittest sxm_ncafm_control.tests.test_metrics         # one module
```

Beyond the tab tests, the full `MainWindow` is smoke-tested offscreen (`QT_QPA_PLATFORM=offscreen`) with SXM closed.

```bash
# Setup (Python 3.11; conda recommended)
conda env update -f environment.yml        # or: pip install -r requirements.txt

# Run — MUST be run from the PARENT of this folder, as a module
cd ..
python -m sxm_ncafm_control.app
```

The checkout directory must be named `sxm_ncafm_control`: modules use absolute imports (`from sxm_ncafm_control.gui...`) *and* relative ones (`from ..common import ...`), so running `python app.py` from inside the folder fails.

`measure_driver_throughput.py` is a read-only diagnostic for the achievable IOCTL/DDE read rates. Run it **on the hardware PC, from inside this folder** (it uses bare `import device_driver`): `python measure_driver_throughput.py`. Measured so far: ~150–175 kHz combined IOCTL reads.

To exercise the GUI without hardware, just launch with SXM closed: the app falls back to offline mode (see below). Verifying real DDE/IOCTL behaviour requires the SXM software and driver on the measurement PC.

## Architecture

### Startup and the connection object
`app.py` builds one `SXMConnection` (`connection.py`) and passes it to `MainWindow`. This is the only place connection logic lives:
- `conn.dde` — `RealDDEClient`, or `MockDDEClient` if anything raises
- `conn.driver` — `SXMIOCTL`, or `None` if the driver can't be opened
- `conn.is_offline` — true if either fell back
- `conn.reader` — `sxm_state.SXMReader` (holds no handle; each `read()` queries the SXM windows), or `None` if the bridge cannot be imported. Independent of `is_offline`: it works whenever SXM is running.

Fallback messages are printed by `common.offline_message()` (root `common.py`). Offline, the mock DDE client prints `[MOCK] ...` lines instead of sending, and tabs synthesize data when `driver is None` (Live Scope has no mock and just refuses to start).

`conn.reconnect()` tears both handles down and re-creates them (the DDE conversation and the exclusive IOCTL handle can go stale in long sessions). **Tabs each hold their own `dde`/`driver` reference**, so `MainWindow.reconnect_hardware()` first pauses anything using the old handles (scope capture, live scope, step test, Z timer) and then `_push_connection_to_tabs()` re-injects the new ones. A new tab that stores `dde`/`driver` must be added to that method. The shared driver is closed once, in `MainWindow.closeEvent` — individual tabs must not close it.

### DDE layer (two files, easy to confuse)
- **`SXMRemote.py`** — Anfatec's vendor DDEML wrapper (ctypes), lightly modified. Keep edits minimal. **Importing it opens a DDE conversation** (`MySXM = DDEClient("SXM","Remote")` at module bottom); if SXM isn't running the import raises `DDEError`, which `SXMConnection` catches and turns into the mock. `RealDDEClient` then creates its *own* `SXMRemote.DDEClient`. `dde_client.py` does a bare `import SXMRemote`, which only resolves because `app.main()` inserts the package dir into `sys.path`.
- **`dde_client.py`** — the app-facing API. `BaseDDE` / `RealDDEClient` / `MockDDEClient` share this contract, and **the mock must be kept in sync with the real client** when adding a method:
  `send_scanpara("EditNN", v)`, `send_dncpara(i, v)`, `read_channel(i)`, `set_channel(i, v)`, `read_topography()`, `feed_para(name, v)`, `last_written(ptype, pcode)`.

How DDE commands work: `execute()` wraps the command in a Pascal program (`begin ... end.`) and sends `XTYP_EXECUTE`. SXM speaks Pascal, e.g. `ScanPara('Edit23', 0.08);`, `DNCPara(4, v);`, `SetChannel(0, v);`, `FeedPara('enable', 1);`, `a:=GetChannel(0); writeln(a);`. Replies arrive asynchronously through the **advise callback** (`Scan`, `Command`, `SaveFileName`, `ScanLine`, `MicState`, `SpectSave` items are subscribed); `SendWait`/`GetChannel` set `NotGotAnswer=True` and spin on `loop()` (a blocking `GetMessageW`) until the callback clears it. This runs **on the calling thread with no overall timeout** — a missing SXM reply hangs the GUI. Numeric replies use decimal commas and the value is on the second line (`BackStr[1]`). `StartMsgLoop`/`MyMsgClass` exist but are unused.

SXM parameters are **write-only over DDE**; `BaseDDE._last` caches the last value written per `(ptype, pcode)`. Reading goes through `sxm_state.py` instead (below).

### GUI read-back (`sxm_state.py`)
- `AnfatecSXMBridge.py` is a **copy**: the master is `dev/anfatec_code/AnfatecSXMBridge.py` and the copy is overwritten from it by the sync script. Don't edit it here; app-level mapping belongs in `sxm_state.py`.
- `AnfatecSXMWriter.py` is a copy too (master `dev/anfatec_code/AnfatecSXMWriter.py`; overwritten by the sync script like the bridge). It writes GUI-only parameters by short name (`PARAMS`: `amp.tau`, `dnc.tc`, `dnc.output_gain`, `z.feedback_off`, ...; names follow `sxm_state.py`) or bridge path, through the messages Delphi reacts to (combo select, button click, `WM_SETTEXT` + optional Enter), and confirms each write by reading it back through the bridge. Browse with `python -m sxm_ncafm_control.AnfatecSXMWriter [group|name]`, `w.table('dnc')`, `w.describe('dnc.tc')` or `w.dnc.tc` (tab-completes). Edits need `commit='change'|'enter'` until recorded in `EDIT_COMMIT` (keyed by short name). Not used by the app yet. On the instrument, a `dnc.output_gain` (radio) write is confirmed to take effect; the other kinds are tested only against a stand-in.
- `SXMReader.read()` reads every bridge section once (bridge `strict=False`) and returns an `SXMReadout`: `values[key]` (None when not readable, reason in `errors[key]`), `by_code(ptype, pcode)` for `PARAMS_BASE` codes, `raw` (the whole bridge snapshot: scan, lock-in, spectroscopy, ...), `ok` (anything read at all). Keys are the `PARAMS_BASE` keys plus extras (`afl_output_gain`, `dnc_time_constant_s`, `amp_tau_s`, `q`, `f_peak`, `ring_down_s`, `feedback_off`, ...; see `LABELS`).
- The DNC window's four numeric edits have no captions and are mapped **by screen position** as DNC 1..4 = sweep start, sweep stop, `use`, Drive (any other count -> error, never a guess). This order matches the layout and values but is **not yet confirmed on the hardware** (write `DNCPara(3, ...)` and see that only `used_freq` moves).
- Custom `EditXX` rows cannot be read: the Delphi component names are not visible through Win32.
- `input_gain_ina` (DNC `Input Gain InA`, x1 / x10) is read too. `QPlusAmpl` / `Drive` are converted with **fixed** scale factors in `device_driver.CHANNELS`, so values taken at different input gains are on different scales; the Tuning tab rescales the amplitude-loop gains for an input / output gain different from the anchor's (unless 'Rescale gains' is off).
- Values are in SXM's GUI units, same as DDE writes. Reads run on the GUI thread (each query has a 250 ms timeout).
- Used by: `ParamsTab` ("Read from SXM" fills *Current*; each apply is read back ~400 ms later and a value SXM did not take is marked red and logged; "All SXM Values..." dialog; one quiet read at startup), `RunSheetTab` (anchor gains, setpoints, Tau / TimeConstant / RollOff / output and input gain, f0/Q, the other loop's state; the runner re-reads SXM to confirm settings the operator set), `StepTestTab` ("= SXM value" for Base), `SuggestedTab` (Q/f0/output gain from the DNC status bar).

### Export metadata (`metadata.py`)
Every export records the SXM settings it was taken at. `metadata.collect(reader, sections)` reads SXM (never raises; without SXM it says why) and returns a `Metadata`: SXM values grouped as AFL / PLL / DNC / Resonance / Topography / Scan (`SXM_GROUPS`; times shown in ms, Ref / Drive labelled "SXM units" because the bridge reports no unit), plus app-side sections the tab adds. It renders as a `#` block for CSVs (`csv_preamble`, whose first line gives the pandas / `numpy.loadtxt(skiprows=N)` call), a `.json` sidecar (`write_sidecar`), a PNG legend (`legend`: one boxed column per loop, AFL / PLL / DNC / Resonance, plus Capture / Step Test, `n/a` for values not read; drawn by `gui/export_image.save_scene_png`) and a file name (`filename`: `YYYYmmdd-HHMMSS_<kind>_<detail>_AFL-Ref..-Kp..-Ki..-Tau..ms_PLL-Kp..-Ki.._InA.._OG..V_TC..ms`). Used by: Scope export (CSV/NPY + JSON + PNG with legend; SXM read when the capture *finishes*; a Step Test hands its events and settings via `set_event_markers(events, step_test=...)`, which adds a "Step Test" section and names the file `steptest_<param><low>-<high>_<channels>_...` instead of `scope_...`; a new capture clears both), Tuning tab *Export results* (one CSV row per condition + JSON with the sheet and the quiet-window noise; each condition's `meta` holds the settings read when it started) and its per-condition recordings (`runsheet_...csv` + JSON, all four channels), Parameters tab tune JSON (`sxm_settings`), calibration CSV. Pure Python, no Qt.

### IOCTL layer (`device_driver.py`)
- `SXMIOCTL` opens `\\.\SXM` (share mode 0, so **one handle for the whole app** — pass `conn.driver` around, never open a second). One instance is used concurrently by capture threads and GUI timers, and reuses a single input buffer, so `read_raw`/`write_raw` are serialized by `_io_lock`; keep any new driver call inside it. `read_raw(index)` → signed 32-bit via `IOCTL_GET_KANAL` (0xF0D); `read_scaled(name)` multiplies by the scale from `CHANNELS`. `write_raw` uses `IOCTL_SET_CHANNEL` (0xF18); `write_unit` currently hard-codes DAC 0 (Topo) regardless of the name passed.
- `CHANNELS` = `{name: (driver_index, short_label, unit, scale)}` is the single source for channel names/units/scales; the scope and constant-height tabs populate their combo boxes from it. Negative indices are DACs/outputs, positive are ADC inputs. Channel keys are e.g. `'QPlusAmpl'`, `'Drive'`, `'Phase'`, `'df'`, `'Topo'`; `Topo` is in **nm**.
- `io_reader.py` is dead code (imports an external `SXMOscilloscope` module that isn't in the repo, nothing imports it); `device_driver.py` replaced it.

### Parameter registry
`PARAMS_BASE` in **root `common.py`** defines the tunable parameters as `(key, ptype, pcode, label, voltage_guarded)`:
Amp Ref `Edit23`, Amp Ki `Edit24`, Amp Kp `Edit32`, PLL Kp `Edit27`, PLL Ki `Edit22` (ptype `EDIT` → `send_scanpara`), and Used Frequency `DNC 3`, Drive `DNC 4` (ptype `DNC` → `send_dncpara`). Tabs import it via `from ..common import ...`.

`gui/common.py` is a stale near-duplicate (different row order) that nothing imports — edit root `common.py`. The ±10 V guard (`confirm_high_voltage`, `VOLTAGE_LIMIT_ABS`) applies to `voltage_guarded` params (Amp Ref, Drive). `StepTestTab._tick` and `ParamsTab._add_custom_editxx` re-derive "voltage-like" by hard-coding `Edit23` / `DNC 4` rather than reading the flag, so change all three together.

### GUI (`gui/`)
`MainWindow` (a `QWidget`, not `QMainWindow`) creates the tabs and does all cross-tab wiring:
- `ParamsTab(dde, reader)` — table Previous/Current/New; stage → apply; save/load "tune" JSON (`kind: "ncafm_tune"`); user-added `EditXX` rows. Emits `custom_params_changed` → `StepTestTab.set_custom_params`.
- `StepTestTab(dde)` — QTimer-driven low/high square wave on one parameter. Holds direct refs to the scope and tab widget; triggers `ScopeTab.start_capture(npoints_override=...)` sized from the test duration, and hands it `(QDateTime, label)` events via `set_event_markers()` for overlay. `ScopeTab.set_test_tab_reference()` links back for "Repeat Test". Optional "Return to base value": `Base` is an explicit field that follows the Low/High midpoint until edited or read from SXM ("= SXM value"). With it on, the last step is held one full period (an extra final tick writes the base), and `stop()` writes the base if at least one step was sent. All sends go through `_send_value()` (voltage guard, log, scope event).
- `ScopeTab(driver, reader)` — one-shot capture: `CaptureThread` (QThread) reads two channels back-to-back for N samples as fast as the driver allows; there is no fixed sample rate, it is measured afterwards (samples / wall time) and shown in the status line. Plot is downsampled above 100k points, export keeps the full data. `estimate_capture_npoints(duration_s)` sizes a capture from the last measured rate. Events past the end of the capture are skipped, not clamped. Step-test overlays are dashed lines only by default (value in the hover tooltip); "Marker labels" / "Step markers" checkboxes toggle text and lines. Synthetic signals if `driver is None`.
- `LiveScopeTab(driver)` — separate tab (not a mode of `ScopeTab`): `LiveCaptureThread` never stores raw samples: it reduces the read stream, per ~1 ms of wall time, to one `(t, min, max)` record per channel in a fixed ring (24 MB = 10 min, `MAX_WINDOW_S`). The plot draws min/max envelopes (`_envelope`, ≤1500 columns), so spikes survive any window, and record timestamps give an exact time axis. The window (1 s–10 min) is only a *view* — changeable while running; changing a channel restarts the thread. The GUI reads the ring without a lock, by design (display only); the writer publishes `rec_count` last.
- `SuggestedTab(dde, params_tab)` — calculator from Q, f₀, PLL bandwidth, plus Lorentzian fit (`scipy.curve_fit`) of a loaded spectrum; can stage into `ParamsTab.stage_value()` or send directly. Only amplitude Ki (Edit24) / Kp (Edit32) are staged/sent. **Derived** (physics): resonator bandwidth f₀/Q and ring-down Q/(π·f₀). **Rules of thumb from the Scienta Omicron QPlus NC-AFM manual** (local PDF in `manuals/`, not tracked; gains are in arbitrary SXM units): amplitude Ki = 5e8/Q at ±1 V AFL output gain (×10 per decade lower gain), Kp = 1e4·Ki; AFL `Tau` = Q/(100·f₀) s = 10·Q/f₀ **ms** (SXM's Tau is a discrete dropdown); PLL `DNC TimeConstant` = 1/(10·BW_PLL). The manual also gives PLL starting values (Kp ≈ −50…−200, Ki ≈ 100·Kp, both negative) that this tab does not suggest yet. Spectrum files (`_read_spectrum_file`) may be tab/space/`;`/`,`-separated, with decimal commas and title/`#` lines.
- `QplusCalibrationTab(dde)` — sweeps `Edit23`, reads topography via DDE `read_topography()` (constructed without a driver, so the IOCTL fallback is unused), fits pm/mV with `scipy.stats.linregress`.
- `ZConstAcquisition(dde, driver)` — 100 ms `QTimer` polls `Topo` via the driver for live Z (plain lists trimmed to a time window). "Disable Feedback" calls `feed_para("enable", 1)` (note the inverted-looking sense: 1 = feedback off) and re-enabling sends `0`. Manual mode is in **absolute Z (nm)** but the write goes to DDE channel 0, so it maps through a reference captured at disable time: `CH0 = ch0_base + ch0_sign * (z_target - abs_ref_z)`. `ch0_sign` (+1) is a manual knob if the piezo direction is inverted. Re-enabling feedback first presets CH0 to the spinbox target. Has its own font-scale combo, independent of the global accessibility manager.
- `gui_accessibility_manager.py` — `AccessibilityManager` is stashed on the `QApplication` instance (`app.accessibility_manager`), persists to `~/.scientific_gui_accessibility.json`, and emits `settings_changed` (font scale, high contrast, dark mode). `MainWindow.apply_accessibility_to_all_tabs()` iterates an explicit tab list, so **a new tab must be added there** (and to `addTab`).

### Tuning (`tuning/` pure numpy/scipy; `gui/sheet_runner.py`, `gui/runsheet_tab.py`)
Goal: tune the PLL or the amplitude loop the way it is done by hand - set parameters, settle, step, look at the noise - but from a **run sheet**, so the conditions are reproducible and recorded. **SXM's Kp/Ki are arbitrary units: nothing assumes what a raw gain means**; loops are judged only from measured responses. Gains are never typed as raw pairs: every condition is relative to a known-good **anchor** (Kp0, Ki0), `Kp = G Kp0`, `Ki = G rho Ki0` (G sets the speed, rho the Ki:Kp shape), so regimes known to be unstable are not visited. The test plan behind it: `docs/test_plan.md`.
- `tuning/runsheet.py` - `Anchor`, `Condition` (G, rho, settings, group, ramp, role), `gains` (with `compensation`: amplitude-loop gains x 1/output gain and x 1/input gain relative to the anchor), `ramp` (G / rho / one setting; G and rho ramps sorted gentlest first; optional anchor repeats), `stop_rule` (a G or rho ramp stops at its first ringing / lost / Drive-at-zero condition), the GUI-only `SETTINGS` (AFL Tau, DNC TimeConstant / RollOff, output gain, input gain; plus Ref over DDE) with read-back keys and `AnfatecSXMWriter` names, `option_for` (combo / radio caption for a value), sheet JSON (`kind: "ncafm_runsheet"`).
- `tuning/noise.py` - `quiet_metrics`: per channel level, detrended rms, rms in 1-10 / 10-50 / 50-200 Hz, noise per pixel dwell (10-300 ms), Drive-at-zero fraction, log-binned ASD for plotting, coherence Drive~Phase / QPlusAmpl~Phase / Drive~df.
- `tuning/workflow.py` - one step test: `StepTestPlan` (amplitude step +-5 % by default: Drive cannot go below zero and the manual's 10 % pins it at small amplitudes), `detect_loop`, `analyze_test` (folded and per-direction step metrics, Drive-at-zero fraction `drive_floor_frac`, failure messages that distinguish too slow / oscillating / ran away), `runaway_reason`, `afl_start_values` (manual: Ki = 5e8/Q, Kp = 1e4 Ki at +-1 V, x10 per lower output-gain range; Tau = Q/(100 f0)).
- `tuning/explore.py` - `assess` (status per direction: clean / slow / overshoot / floor / ringing / lost; a condition is as good as its worse direction; amplitude down-steps at the ring-down rate are flagged as a sensor limit) and the steadiness checks (`steady`, `settled_at`, ...) used by the runner. Its `Explorer` / `SearchRegion` / `trade_off` (decade-grid exploration of the Kp-Ki plane) belonged to the old Tuning tab and are no longer used by the app.
- `gui/sheet_runner.py` - `SheetRunner`, a QTimer state machine per condition: GUI-only settings (the run pauses with `operator_needed` and continues on `continue_()` once SXM's read-back shows them; or written through `AnfatecSXMWriter` when 'automatically' is ticked) -> gains + first level over DDE -> settle (`steady`) -> step train -> back to base, settle, **quiet window** -> analyse (`analyze_test` on the loop's two channels, `quiet_metrics` on all) -> `stop_rule`. One `TrainCaptureThread` per condition records QPlusAmpl, Drive, df and Phase together (it yields the GIL once per slice so the GUI-thread timers that send the steps stay on time). A lost condition (own loop runaway, or the other loop losing lock / amplitude) releases the PLL (0, 0), writes the anchor and waits until steady before the next one; no recovery aborts the run. At the end, on Stop and on abort: anchor gains, start Ref / `use` and (with the writer) the start settings are written back; otherwise the log lists the settings to put back by hand. Outcomes keep a 5 ms display copy of the recording; the full one goes to disk when 'Save every recording' is on.
- `gui/runsheet_tab.py` - `RunSheetTab` (the "Tuning" tab): loop and SXM read-back, anchor (= SXM gains while the sheet is empty), train settings, *Add a ramp* (values prefilled per loop from `runsheet.DEFAULT_VALUES`), the sheet table (status colour, rise up / down, overshoot, Drive sigma or Phase peak, amplitude noise, df per pixel; best = clean with the lowest df per pixel, starred), plots of the selected condition (averaged step up / down, controller output, quiet-window spectrum, recording with marks) and *Ramp trends* (speed and noise against the ramped value), sheet save / load, *Stage gains in Parameters*, export. Checks block Run when offline, without a confirmed retracted tip, with an empty sheet or a bad anchor.
- `tuning/loopid.py`, `trial.py` - physical identification of a PI loop from one train; a diagnostic, not used for decisions. Test fixtures: `tests/sim.py` (simulators), `tests/fake_instrument.py` (`FakeInstrument` PLL, `FakeAFLInstrument` amplitude loop with Drive clipped at zero, both real time).

### Gotchas
- `MainWindow` sets `step_tab.scope_tab_index = 2` (hard-coded); reordering tabs breaks the "jump to Scope" behaviour.
- Don't `print()` non-ASCII (e.g. `→`): with a cp1252 stdout it raises `UnicodeEncodeError`, and an exception inside a Qt slot aborts PyQt5. Use `->`.
- Long-running `QTextEdit` logs should go through `common.append_log_line()` (bounded to `LOG_MAX_BLOCKS`), not `.append()`.
- **Known bug:** `ZConstAcquisition.toggle_feedback` calls `self.dde.get_channel(0)`, but the DDE clients only define `read_channel()`. The `AttributeError` is swallowed by its `try/except`, so `ch0_base` stays `0.0` and absolute-Z writes use the wrong base. Fix by calling `read_channel(0)`.
- Values sent are in **SXM's current GUI units**; the app does no unit conversion for DDE parameters.
- `__pycache__/*.pyc` files are tracked in git despite `.gitignore`, so they show up as modified/deleted after any run. Don't stage them.
