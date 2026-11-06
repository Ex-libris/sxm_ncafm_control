"""Run a qPlus open-loop campaign (or only its preflight). From the folder holding sxm_ncafm_control and sxm_anfatec:

    python -m sxm_ncafm_control.characterisation.run sxm_ncafm_control/characterisation/configs/open_loop_matrix.json --preflight
    python -m sxm_ncafm_control.characterisation.run sxm_ncafm_control/characterisation/configs/open_loop_matrix.json --blocks 1
    python -m sxm_ncafm_control.characterisation.run --resume qplus_characterisation_results/<run folder> --blocks 1
    python -m sxm_ncafm_control.characterisation.run <config> --sim              # simulated sensor, no instrument

Stop safely: Ctrl+C, or create a file named STOP in the run folder. The current record ends at once, the
drive goes to off_drive and the time constant back to its start value (both read back).
"""
from __future__ import annotations

import argparse
import datetime
import json
import signal
import sys
import tempfile
from pathlib import Path

from . import config as C
from .runner import Campaign

from . import RESULTS_DIR as RESULTS


def _printer(verbose: bool):
    def progress(event, data):
        if event == 'log' and (verbose or 'preflight' in data or 'ABORT' in data or 'end:' in data
                               or 'headroom' in data or 'batch' in data or 'RESTORE' in data or 'warning' in data):
            print(data, flush=True)
        elif event == 'progress':
            print(f'--- {data["n"]}/{data["of"]}  elapsed {data["elapsed_s"] / 60:.1f} min, '
                  f'left ~{data["eta_s"] / 60:.1f} min', flush=True)
        elif event == 'condition_done':
            d = data
            drive = '0' if d['drive'] is None else f'{d["drive"]:g}'

            def f(x, fmt):
                return '-' if x is None else format(x, fmt)
            print(f'  c{d["idx"]:04d} {d["kind"]:8s} b{d["block"]} drive {drive:>8s} TC {d["tc"]:>7s}  '
                  f'{d["status"]:7s} A {f(d["amp_ss"], ".4g")} V  tau {f(d["tau_s"], ".4f")} s  '
                  f'base {f(d["baseline_mean"], ".3g")} V', flush=True)
    return progress


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='qPlus open-loop drive x lock-in TC campaign')
    ap.add_argument('config', nargs='?', help='campaign JSON (not needed with --resume)')
    ap.add_argument('--resume', metavar='RUN_DIR', help='continue a campaign folder (its own config is used)')
    ap.add_argument('--preflight', action='store_true', help='read-backs + drive echo only, then restore')
    ap.add_argument('--blocks', type=int, help='run at most N blocks in this invocation')
    ap.add_argument('--max-minutes', type=float, help='plan at most M minutes in this invocation')
    ap.add_argument('--sim', action='store_true', help='simulated sensor (no instrument)')
    ap.add_argument('--sim-fault', choices=['afl', 'pll', 'compress', 'pll-gains'], help='simulate a fault')
    ap.add_argument('--out', help='parent folder for the run folder')
    ap.add_argument('--no-analyse', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args(argv)

    if a.resume:
        run_dir = Path(a.resume)
        cfg = C.from_dict(json.loads((run_dir / 'campaign.json').read_text(encoding='utf-8'))['config'])
    elif a.config:
        cfg = C.load(a.config)
        parent = Path(a.out) if a.out else (Path(tempfile.gettempdir()) / 'sxm_qplus_sim' if a.sim else RESULTS)
        run_dir = parent / f'{datetime.datetime.now():%Y%m%d-%H%M%S}_{"sim_" if a.sim else ""}{cfg.name}'
    else:
        ap.error('give a config file or --resume RUN_DIR')

    writable = {cfg.instrument.drive_param, cfg.instrument.tc_param}
    if a.sim:
        from .sim import SimQPlus, SimSensor
        s = SimSensor(f0=cfg.expect.f0_hz, q=cfg.expect.q, f_use=cfg.expect.f0_hz)
        if a.sim_fault == 'afl':
            s.afl_on = True
        elif a.sim_fault == 'pll':
            s.pll_on = True
        elif a.sim_fault == 'compress':
            s.amp_sat_v = 0.004
        elif a.sim_fault == 'pll-gains':
            s.pll_gains = (-300.0, -3000.0)
        inst = SimQPlus(s, writable)
    else:
        from .instrument import QPlusSxm
        inst = QPlusSxm(writable)

    stop = {'n': 0}

    def on_sigint(signum, frame):
        stop['n'] += 1
        print('\n*** stopping safely: ending the record, drive -> off, TC -> start value. '
              + ('Still stopping, please wait.' if stop['n'] > 1 else ''), flush=True)

    signal.signal(signal.SIGINT, on_sigint)
    print(f'Run folder: {run_dir}\nStop: Ctrl+C, or create {run_dir / "STOP"}', flush=True)
    camp = Campaign(cfg, inst, run_dir, _printer(a.verbose), lambda: stop['n'] > 0)
    try:
        out = camp.run(resume=bool(a.resume), preflight_only=a.preflight, max_blocks=a.blocks,
                       max_minutes=a.max_minutes)
    finally:
        inst.close()
    print(f'\n{out["status"].upper()}: {out["message"]}')
    for r in out.get('restore', []):
        print(f'  restore {r["name"]} -> {r["value"]!r}: {"ok" if r["ok"] else "FAILED " + r.get("error", "")}')
    if out['status'] in ('complete', 'batch done') and not a.no_analyse:
        from .analyse import analyse_run
        s = analyse_run(run_dir)
        print(f'Analysis: {Path(s["derived_dir"]) / "report.md"}')
    if out['status'] in ('batch done', 'stopped', 'aborted'):
        print(f'Continue later with:  python -m sxm_ncafm_control.characterisation.run --resume "{run_dir}" --blocks 1')
    return 0 if out['status'] in ('complete', 'batch done', 'preflight') else 1


if __name__ == '__main__':
    sys.exit(main())
