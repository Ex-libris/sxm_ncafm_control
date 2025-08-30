# app.py
"""
Entry point for the NC-AFM control GUI.

This module is intentionally thin. It:
1) Creates the Qt application
2) Builds a single, centralized SXMConnection (DDE + IOCTL)
3) Passes that connection to the main window
4) Shows a concise connection status dialog
5) Starts the Qt event loop
"""

import os
import sys
from PyQt5 import QtWidgets

from sxm_ncafm_control.connection import SXMConnection
from sxm_ncafm_control.gui.main_window import MainWindow


def show_connection_status_dialog(parent: QtWidgets.QWidget, conn: SXMConnection) -> None:
    """
    Show a single startup dialog that summarizes the current connection state.

    The dialog reports DDE connectivity (real vs. mock) and IOCTL availability.
    Detailed low-level errors are printed by the connection layer to the console;
    here we present a short, user-friendly summary.

    Parameters
    ----------
    parent : QtWidgets.QWidget
        Parent widget used for dialog positioning.
    conn : SXMConnection
        The centralized connection object holding DDE and IOCTL handles.
    """
    # Determine states without importing backend classes here.
    dde_online = getattr(conn, "dde", None) is not None and not getattr(conn, "is_offline", False)
    driver_online = getattr(conn, "driver", None) is not None

    # If either DDE fell back to mock or the driver is missing, call it "offline mode"
    # for user purposes. (You can refine this if you want a tri-state later.)
    is_fully_online = dde_online and driver_online

    # Build concise lines for the dialog.
    dde_line = "✅ DDE: connected (real SXM)" if dde_online else "⚠️ DDE: offline (using mock)"
    drv_line = "✅ Driver: available (IOCTL OK)" if driver_online else "⚠️ Driver: not available"

    dialog = QtWidgets.QMessageBox(parent)
    dialog.setWindowTitle("SXM Connection Status")

    if is_fully_online:
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setText("Connected to SXM (ONLINE mode).")
        dialog.setInformativeText(
            "Commands will be sent to the real hardware.\n"
            f"{dde_line}\n{drv_line}"
        )
    else:
        dialog.setIcon(QtWidgets.QMessageBox.Warning)
        dialog.setText("Running in OFFLINE mode.")
        dialog.setInformativeText(
            "You can use the GUI for testing and planning, but no commands are sent to hardware.\n"
            f"{dde_line}\n{drv_line}\n\n"
            "If you expect to be online:\n"
            "• Start the SXM software on the measurement PC\n"
            "• Ensure the SXM driver/device is present\n"
            "• Restart this application"
        )

    dialog.exec_()


def main() -> int:
    """
    Main entry point of the application.

    Returns
    -------
    int
        Qt exit code (0 on normal termination).
    """
    # Create the Qt application instance first.
    app = QtWidgets.QApplication(sys.argv)

    # Ensure the package directory is in sys.path so local modules (e.g., SXMRemote.py)
    # can be found when the connection layer tries to import them.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Build a single, centralized connection manager.
    # SXMConnection is responsible for:
    #   - Establishing DDE (real or mock fallback)
    #   - Opening IOCTL driver (or None if unavailable)
    #   - Printing unified "OFFLINE MODE" messages to the console when needed
    conn = SXMConnection()

    # Create and show the main window. Pass the connection object in, so all
    # tabs/widgets share the same handles and no one re-implements connection logic.
    win = MainWindow(conn)
    win.show()

    # Append mode tag to the window title for constant visual feedback.
    # Consider "PARTIAL" if you later differentiate cases (e.g., DDE real, driver None).
    mode_suffix = " - ONLINE" if (getattr(conn, "driver", None) is not None and not getattr(conn, "is_offline", False)) else " - OFFLINE"
    if not win.windowTitle().endswith(mode_suffix):
        win.setWindowTitle(win.windowTitle() + mode_suffix)

    # Show a concise connection dialog once at startup.
    show_connection_status_dialog(win, conn)

    # Enter the Qt event loop.
    return app.exec_()


if __name__ == "__main__":
    # Use SystemExit for clean termination.
    raise SystemExit(main())
