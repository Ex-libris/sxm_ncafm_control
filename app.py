# app.py
"""
Entry point for the NC-AFM control GUI.

This script is intentionally kept small:
- It chooses the appropriate DDE backend (real connection or mock fallback).
- It launches the main GUI window.
"""

import sys
from PyQt5 import QtWidgets
from sxm_ncafm_control.dde_client import RealDDEClient, MockDDEClient
from sxm_ncafm_control.gui.main_window import MainWindow


def main() -> int:
    """
    Main entry point of the application.

    Inputs:
        None (uses system arguments from sys.argv implicitly).

    Process:
        1. Initializes a Qt application.
        2. Attempts to connect to the real SXM DDE interface.
           - If successful, uses `RealDDEClient`.
           - If it fails, falls back to `MockDDEClient` for offline testing.
        3. Creates and shows the main GUI window.
        4. Starts the Qt event loop.

    Outputs:
        int: Exit code from the Qt application event loop.
             - Typically `0` for a clean exit.
    """

    # Create the Qt application instance (required by PyQt5).
    app = QtWidgets.QApplication(sys.argv)

    # Attempt to connect to the real SXM DDE backend.
    try:
        dde_client = RealDDEClient(app_name="SXM", topic="Remote")
        print("[INFO] Connected to SXM via DDE.")
    except Exception as error:
        # If connection fails, switch to a mock client for offline testing.
        print(f"[WARN] Could not connect to SXM DDE: {error}")
        print("[INFO] Using MockDDEClient (offline).")
        dde_client = MockDDEClient()

    # Create and show the main application window, passing in the DDE client.
    main_window = MainWindow(dde_client=dde_client)
    main_window.show()

    # Start the Qt event loop (keeps the GUI running).
    return app.exec_()


# Standard Python entry-point check:
# Ensures this script only runs when executed directly,
# not when imported as a module.
if __name__ == "__main__":
    raise SystemExit(main())
