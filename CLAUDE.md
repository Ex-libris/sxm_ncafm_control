# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyQt5 desktop GUI for tuning NC-AFM (qPlus) loops on an **Anfatec SXM** SPM controller. It talks to the instrument over **two independent channels**:

| Channel | Module | Transport | Used for |
|---|---|---|---|
| **DDE** | `SXMRemote.py` → `dde_client.py` | Win32 DDEML (`user32`) to the running SXM software (service `SXM`, topic `Remote`) | Writing parameters (`EditNN`, DNC), setting Z, toggling feedback |
| **IOCTL** | `device_driver.py` | `DeviceIoControl` on `\\.\SXM` (the Anfatec kernel driver), bypassing SXM software | Fast reads of any hardware channel (scope, live Z) |

Windows-only (pywin32 + ctypes on `user32`). `device_driver.py` imports `win32file` at module top, so even offline mode needs Windows + pywin32.

## Commands

There is no build step or linter. The only tests are unit tests of the pure-logic `tuning/` package (stdlib `unittest`, no Qt or hardware; 58 tests, ~20 s). Run from the **parent** folder:

```bash
PYTHONPATH=. python -m unittest discover -s sxm_ncafm_control/tests -v      # all
PYTHONPATH=. python -m unittest sxm_ncafm_control.tests.test_metrics         # one module
```

The GUI has no automated tests; it was smoke-tested offscreen (`QT_QPA_PLATFORM=offscreen`) with fake drivers.

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

Fallback messages are printed by `common.offline_message()` (root `common.py`). Offline, the mock DDE client prints `[MOCK] ...` lines instead of sending, and tabs synthesize data when `driver is None` (Live Scope has no mock and just refuses to start).

`conn.reconnect()` tears both handles down and re-creates them (the DDE conversation and the exclusive IOCTL handle can go stale in long sessions). **Tabs each hold their own `dde`/`driver` reference**, so `MainWindow.reconnect_hardware()` first pauses anything using the old handles (scope capture, live scope, step test, Z timer) and then `_push_connection_to_tabs()` re-injects the new ones. A new tab that stores `dde`/`driver` must be added to that method. The shared driver is closed once, in `MainWindow.closeEvent` — individual tabs must not close it.

### DDE layer (two files, easy to confuse)
- **`SXMRemote.py`** — Anfatec's vendor DDEML wrapper (ctypes), lightly modified. Keep edits minimal. **Importing it opens a DDE conversation** (`MySXM = DDEClient("SXM","Remote")` at module bottom); if SXM isn't running the import raises `DDEError`, which `SXMConnection` catches and turns into the mock. `RealDDEClient` then creates its *own* `SXMRemote.DDEClient`. `dde_client.py` does a bare `import SXMRemote`, which only resolves because `app.main()` inserts the package dir into `sys.path`.
- **`dde_client.py`** — the app-facing API. `BaseDDE` / `RealDDEClient` / `MockDDEClient` share this contract, and **the mock must be kept in sync with the real client** when adding a method:
  `send_scanpara("EditNN", v)`, `send_dncpara(i, v)`, `read_channel(i)`, `set_channel(i, v)`, `read_topography()`, `feed_para(name, v)`, `last_written(ptype, pcode)`.

How DDE commands work: `execute()` wraps the command in a Pascal program (`begin ... end.`) and sends `XTYP_EXECUTE`. SXM speaks Pascal, e.g. `ScanPara('Edit23', 0.08);`, `DNCPara(4, v);`, `SetChannel(0, v);`, `FeedPara('enable', 1);`, `a:=GetChannel(0); writeln(a);`. Replies arrive asynchronously through the **advise callback** (`Scan`, `Command`, `SaveFileName`, `ScanLine`, `MicState`, `SpectSave` items are subscribed); `SendWait`/`GetChannel` set `NotGotAnswer=True` and spin on `loop()` (a blocking `GetMessageW`) until the callback clears it. This runs **on the calling thread with no overall timeout** — a missing SXM reply hangs the GUI. Numeric replies use decimal commas and the value is on the second line (`BackStr[1]`). `StartMsgLoop`/`MyMsgClass` exist but are unused.

SXM parameters are effectively **write-only**; `BaseDDE._last` caches the last value written per `(ptype, pcode)` so the UI can show something.

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
- `ParamsTab(dde)` — table Previous/Current/New; stage → apply; save/load "tune" JSON (`kind: "ncafm_tune"`); user-added `EditXX` rows. Emits `custom_params_changed` → `StepTestTab.set_custom_params`.
- `StepTestTab(dde)` — QTimer-driven low/high square wave on one parameter. Holds direct refs to the scope and tab widget; triggers `ScopeTab.start_capture(npoints_override=...)` sized from the test duration, and hands it `(QDateTime, label)` events via `set_event_markers()` for overlay. `ScopeTab.set_test_tab_reference()` links back for "Repeat Test". Optional "Return to base value": since SXM values can't be read back, `Base` is an explicit field that follows the Low/High midpoint until edited. With it on, the last step is held one full period (an extra final tick writes the base), and `stop()` writes the base if at least one step was sent. All sends go through `_send_value()` (voltage guard, log, scope event).
- `ScopeTab(driver)` — one-shot capture: `CaptureThread` (QThread) reads two channels back-to-back for N samples as fast as the driver allows; there is no fixed sample rate, it is measured afterwards (samples / wall time) and shown in the status line. Plot is downsampled above 100k points, export keeps the full data. `estimate_capture_npoints(duration_s)` sizes a capture from the last measured rate. Events past the end of the capture are skipped, not clamped. Step-test overlays are dashed lines only by default (value in the hover tooltip); "Marker labels" / "Step markers" checkboxes toggle text and lines. Synthetic signals if `driver is None`.
- `LiveScopeTab(driver)` — separate tab (not a mode of `ScopeTab`): `LiveCaptureThread` never stores raw samples: it reduces the read stream, per ~1 ms of wall time, to one `(t, min, max)` record per channel in a fixed ring (24 MB = 10 min, `MAX_WINDOW_S`). The plot draws min/max envelopes (`_envelope`, ≤1500 columns), so spikes survive any window, and record timestamps give an exact time axis. The window (1 s–10 min) is only a *view* — changeable while running; changing a channel restarts the thread. The GUI reads the ring without a lock, by design (display only); the writer publishes `rec_count` last.
- `SuggestedTab(dde, params_tab)` — calculator from Q, f₀, PLL bandwidth, plus Lorentzian fit (`scipy.curve_fit`) of a loaded spectrum; can stage into `ParamsTab.stage_value()` or send directly. Only amplitude Ki (Edit24) / Kp (Edit32) are staged/sent. **Derived** (physics): resonator bandwidth f₀/Q and ring-down Q/(π·f₀). **Rules of thumb from the Scienta Omicron QPlus NC-AFM manual** (local PDF in `manuals/`, not tracked; gains are in arbitrary SXM units): amplitude Ki = 5e8/Q at ±1 V AFL output gain (×10 per decade lower gain), Kp = 1e4·Ki; AFL `Tau` = Q/(100·f₀) s = 10·Q/f₀ **ms** (SXM's Tau is a discrete dropdown); PLL `DNC TimeConstant` = 1/(10·BW_PLL). The manual also gives PLL starting values (Kp ≈ −50…−200, Ki ≈ 100·Kp, both negative) that this tab does not suggest yet. Spectrum files (`_read_spectrum_file`) may be tab/space/`;`/`,`-separated, with decimal commas and title/`#` lines.
- `QplusCalibrationTab(dde)` — sweeps `Edit23`, reads topography via DDE `read_topography()` (constructed without a driver, so the IOCTL fallback is unused), fits pm/mV with `scipy.stats.linregress`.
- `ZConstAcquisition(dde, driver)` — 100 ms `QTimer` polls `Topo` via the driver for live Z (plain lists trimmed to a time window). "Disable Feedback" calls `feed_para("enable", 1)` (note the inverted-looking sense: 1 = feedback off) and re-enabling sends `0`. Manual mode is in **absolute Z (nm)** but the write goes to DDE channel 0, so it maps through a reference captured at disable time: `CH0 = ch0_base + ch0_sign * (z_target - abs_ref_z)`. `ch0_sign` (+1) is a manual knob if the piezo direction is inverted. Re-enabling feedback first presets CH0 to the spinbox target. Has its own font-scale combo, independent of the global accessibility manager.
- `gui_accessibility_manager.py` — `AccessibilityManager` is stashed on the `QApplication` instance (`app.accessibility_manager`), persists to `~/.scientific_gui_accessibility.json`, and emits `settings_changed` (font scale, high contrast, dark mode). `MainWindow.apply_accessibility_to_all_tabs()` iterates an explicit tab list, so **a new tab must be added there** (and to `addTab`).

### Loop-tuning toolkit (`tuning/`, pure numpy/scipy, no Qt/hardware)
Groundwork for a guided PLL / amplitude-loop autotuner (goal and manual protocols: the manual in `manuals/`). **Design rule: SXM's Kp/Ki are arbitrary units, so nothing here assumes what a raw gain means** — gains are judged only from measured responses and searched as multiples of the user's known-good baseline. One loop is tuned per session; the user approves each trial ("guided").
- `trial.py` — `StepProtocol` (the manual's ±1 Hz toggle of `DNC use`: offsets alternate ∓`step_hz`, steps are ±2·`step_hz`) and `TrialResult` (t, df, phase, raw Kp/Ki). Every backend returns these.
- `metrics.py` — step metrics (10–90 % rise, 5 % settling, overshoot, ringing extrema, damping), `average_steps` (folds ± steps together; pass commanded `signs` for signals that return to rest, e.g. Phase), `error_transient_metrics` (Phase peak/decay/IAE), noise metrics. Noise-aware: overshoot counts only above 5σ, settling band ≥ 4σ; `StepNotDetectable` is raised when the step is buried or the loop rings/runs away.
- `simulator.py` — virtual qPlus + 2-stage lock-in + PI (`simulate_pll`, `run_step_trial`, `SimulatedPLLBackend`). **`SXMScale` (raw → physical gain) is an assumption** tuned so Kp=-100 with Ki=-1e3/-1e4/-5e4 reproduces the manual's slow/good/overshoot figure; its absolute time scale is a guess. Assumes Q≈f₀≈25k. In this model Kp does the fast df tracking against the sensor's slow pole (ring-down Q/(π·f₀) ≈ 0.3 s) while Ki only removes the residual — visible as the *Phase* tail — so both channels must be scored; df noise ∝ gain^~1.0 and ∝ 1/√(lock-in τ), i.e. the DNC `TimeConstant` is a third noise knob (not yet searched).
- `identify.py` — `identify_scale` fits `SXMScale` from real trials at known gains (sign-convention independent); the identified model then predicts untested gains. Verified only on simulator data so far.
- `planner.py` — `ScanSpec` (pixel dwell = t_line/n_px; rise ≤ ½ dwell, 5 % settle ≤ 1 dwell, overshoot ≤ 10 %, Phase decay ≤ 10 % of a line — all adjustable rules of thumb), `analyze_trial`, and `GuidedTuner`: a generator (`proposals()`; call `record()`/`skip()` after each) running baseline → Ki:Kp ratio scan → scale along the ratio → geometric bisection → verify, bounded by `Limits`. It picks the lowest-noise feasible gains; when none are feasible it reports the cleanest response and which constraint fails. Offline demo: `run_guided(GuidedTuner(-100, -1e4, ScanSpec(...)), SimulatedPLLBackend())`.
- **Not written yet:** the hardware backend (`run_trial(kp, ki) -> TrialResult` via DDE writes + IOCTL capture) and the GUI tab. DDE calls must stay on the GUI thread (they block in `GetMessage`), so the tuner has to be driven by a `QTimer` state machine; only capture may use a worker thread. SXM parameters cannot be read back (`GetScanPara` does not work that way), so the baseline must be typed in / remembered by the app.

### Gotchas
- `MainWindow` sets `step_tab.scope_tab_index = 2` (hard-coded); reordering tabs breaks the "jump to Scope" behaviour.
- Don't `print()` non-ASCII (e.g. `→`): with a cp1252 stdout it raises `UnicodeEncodeError`, and an exception inside a Qt slot aborts PyQt5. Use `->`.
- Long-running `QTextEdit` logs should go through `common.append_log_line()` (bounded to `LOG_MAX_BLOCKS`), not `.append()`.
- **Known bug:** `ZConstAcquisition.toggle_feedback` calls `self.dde.get_channel(0)`, but the DDE clients only define `read_channel()`. The `AttributeError` is swallowed by its `try/except`, so `ch0_base` stays `0.0` and absolute-Z writes use the wrong base. Fix by calling `read_channel(0)`.
- Values sent are in **SXM's current GUI units**; the app does no unit conversion for DDE parameters.
- `__pycache__/*.pyc` files are tracked in git despite `.gitignore`, so they show up as modified/deleted after any run. Don't stage them.
