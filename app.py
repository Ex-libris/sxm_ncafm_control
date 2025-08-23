# app.py
"""
Entry point for the NC-AFM control GUI.

This script is intentionally kept small:
- It chooses the appropriate DDE backend (real connection or mock fallback).
- It launches the main GUI window.
"""

import sys
import os
from PyQt5 import QtWidgets
from sxm_ncafm_control.dde_client import RealDDEClient, MockDDEClient
from sxm_ncafm_control.gui.main_window import MainWindow


def interpret_dde_error(error_msg: str) -> str:
    """
    Convert technical DDE error messages into user-friendly explanations.
    """
    error_lower = str(error_msg).lower()
    
    if "0x400a" in error_lower:
        return (
            "SXM software is not running or not responding.\n"
            "→ Make sure the SXM application is open and fully loaded.\n"
            "→ Check that DDE communication is enabled in SXM settings."
        )
    elif "unable to establish a conversation" in error_lower:
        return (
            "Cannot communicate with SXM software.\n"
            "→ Ensure SXM application is running.\n"
            "→ Try restarting both SXM and this application."
        )
    elif "no module named 'sxmremote'" in error_lower:
        return (
            "SXMRemote.py library not found.\n"
            "→ Copy SXMRemote.py to the project directory.\n"
            "→ This file should be provided by Anfatec with your SXM software."
        )
    elif "unable to register with ddeml" in error_lower:
        return (
            "Windows DDE system error.\n"
            "→ Try running as Administrator.\n"
            "→ Restart Windows if the problem persists."
        )
    else:
        return f"Connection failed: {error_msg}"


def show_connection_status_dialog(parent, is_connected: bool, error_details: str = ""):
    """
    Show a dialog explaining the connection status to the user.
    """
    dialog = QtWidgets.QMessageBox(parent)
    
    if is_connected:
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setWindowTitle("SXM Connection - Online Mode")
        dialog.setText("✅ Successfully connected to SXM software!")
        dialog.setInformativeText(
            "The application is running in ONLINE mode.\n"
            "Commands will be sent directly to your SXM controller."
        )
    else:
        dialog.setIcon(QtWidgets.QMessageBox.Warning)
        dialog.setWindowTitle("SXM Connection - Offline Mode")
        dialog.setText("⚠️ Running in OFFLINE mode")
        dialog.setDetailedText(error_details)
        dialog.setInformativeText(
            "The application will run in simulation mode.\n"
            "• All controls remain functional for testing\n"
            "• Commands will be logged but not sent to hardware\n"
            "• Perfect for parameter planning and GUI development\n\n"
            "To connect to real hardware:\n"
            "1. Start the SXM software on the measurement computer\n"
            "2. Restart this application"
        )
    
    # Add a "Don't show again" checkbox for offline mode
    if not is_connected:
        checkbox = QtWidgets.QCheckBox("Don't show this dialog on startup")
        dialog.setCheckBox(checkbox)
    
    dialog.exec_()
    
    # Return whether user wants to suppress future dialogs
    if not is_connected and dialog.checkBox():
        return dialog.checkBox().isChecked()
    return False


def main() -> int:
    """
    Main entry point of the application.
    """
    # Create the Qt application instance (required by PyQt5).
    app = QtWidgets.QApplication(sys.argv)

    # Add current directory to Python path to help find SXMRemote.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Attempt to connect to the real SXM DDE backend.
    dde_client = None
    connection_successful = False
    error_explanation = ""
    
    try:
        dde_client = RealDDEClient(app_name="SXM", topic="Remote")
        connection_successful = True
        print("[INFO] ✅ Connected to SXM via DDE - ONLINE mode")
        
    except ImportError as e:
        # SXMRemote module not found
        error_explanation = interpret_dde_error(str(e))
        print(f"[WARN] ❌ SXMRemote module not found")
        print(f"[INFO] 🔄 Switching to OFFLINE mode (MockDDEClient)")
        
    except Exception as error:
        # Connection failed (hardware/software not available)
        error_explanation = interpret_dde_error(str(error))
        print(f"[WARN] ❌ Cannot connect to SXM: {error}")
        print(f"[INFO] 🔄 Switching to OFFLINE mode (MockDDEClient)")
    
    # Create mock client if real connection failed
    if not connection_successful:
        dde_client = MockDDEClient()

    # Create the main application window
    main_window = MainWindow(dde_client=dde_client)
    
    # Show connection status dialog (unless user previously opted out)
    # You could add a settings file to remember user preference
    show_connection_status_dialog(main_window, connection_successful, error_explanation)
    
    # Show the main window
    main_window.show()
    
    # Update window title to reflect mode
    mode_suffix = " - ONLINE" if connection_successful else " - OFFLINE"
    main_window.setWindowTitle(main_window.windowTitle() + mode_suffix)

    # Start the Qt event loop (keeps the GUI running).
    return app.exec_()


# Standard Python entry-point check
if __name__ == "__main__":
    raise SystemExit(main())