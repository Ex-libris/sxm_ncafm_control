"""qPlus sensor characterisation without a surface: open-loop drive x lock-in time-constant campaigns.

Pure logic (no Qt): an orchestration layer on top of sxm_anfatec (writer / bridge / IOCTL driver). It never
moves z, never writes a feedback gain and never switches a loop on. See characterisation/README.md.

In the app: the "Sensor Characterisation" tab (gui/characterisation_tab.py), which shares the app's driver.
Without the app (it holds the only driver handle, so close it first):

    python -m sxm_ncafm_control.characterisation.run     <config.json> [--sim] [--preflight] [--resume DIR] [--blocks N]
    python -m sxm_ncafm_control.characterisation.analyse <run folder>
"""
from pathlib import Path

VERSION = '2026.11.06'

# Results go next to the checkout, never into this (public) repository.
RESULTS_DIR = Path(__file__).resolve().parents[2] / 'qplus_characterisation_results'
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'open_loop_matrix.json'
