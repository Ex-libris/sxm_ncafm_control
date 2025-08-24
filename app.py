# app.py
"""
Entry point for the NC-AFM control GUI.

This script is intentionally kept small:
- It chooses the appropriate DDE backend (real connection or mock fallback).
- It launches the main GUI window.

This is the file you run to start the program.
"""

import sys
import os
from PyQt5 import QtWidgets
from sxm_ncafm_control.dde_client import RealDDEClient, MockDDEClient
from sxm_ncafm_control.gui.main_window import MainWindow


def interpret_dde_error(error_msg: str) -> str:
    """
    Convert technical DDE error messages into user-friendly explanations.
    
    DDE (Dynamic Data Exchange) is how Windows programs communicate with each other.
    In our case, this GUI talks to the SXM software through DDE messages.
    When something goes wrong, Windows gives cryptic error codes - this function
    translates them into actionable advice.
    
    Args:
        error_msg: The raw error message from Windows/DDE system
        
    Returns:
        A human-readable explanation with troubleshooting steps
    """
    # Convert to lowercase for easier pattern matching
    error_lower = str(error_msg).lower()
    
    # Check for common DDE error patterns and provide specific guidance
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
        # Fallback for unknown errors
        return f"Connection failed: {error_msg}"


def show_connection_status_dialog(parent, is_connected: bool, error_details: str = ""):
    """
    Show a dialog explaining the connection status to the user.
    
    This is important to understand whether you're connected to real hardware 
    or running in simulation mode. Offline mode is useful for planning 
    experiments, testing parameters, or learning the interface without 
    needing the actual SXM setup.
    
    Args:
        parent: The main window (needed for dialog positioning)
        is_connected: True if connected to real SXM, False if in offline mode
        error_details: Technical error info (shown in expandable details)
    
    Returns:
        bool: True if user wants to suppress future offline dialogs
    """
    # Create a message box dialog
    dialog = QtWidgets.QMessageBox(parent)
    
    if is_connected:
        # Online mode - connected to real hardware
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setWindowTitle("SXM Connection - Online Mode")
        dialog.setText("✅ Successfully connected to SXM software!")
        dialog.setInformativeText(
            "The application is running in ONLINE mode.\n"
            "Commands will be sent directly to your SXM controller."
        )
    else:
        # Offline mode - using simulation
        dialog.setIcon(QtWidgets.QMessageBox.Warning)
        dialog.setWindowTitle("SXM Connection - Offline Mode")
        dialog.setText("⚠️ Running in OFFLINE mode")
        dialog.setDetailedText(error_details)  # Technical details (expandable)
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
    # (Users might want to work offline frequently during development)
    if not is_connected:
        checkbox = QtWidgets.QCheckBox("Don't show this dialog on startup")
        dialog.setCheckBox(checkbox)
    
    # Show the dialog and wait for user to close it
    dialog.exec_()
    
    # Return whether user wants to suppress future dialogs
    if not is_connected and dialog.checkBox():
        return dialog.checkBox().isChecked()
    return False


def main() -> int:
    """
    Main entry point of the application.
    
    This function handles the complete startup process:
    1. Creates the GUI framework
    2. Attempts to connect to SXM software
    3. Falls back to offline mode if needed
    4. Shows the main window
    5. Starts the event loop (keeps the program running)
    
    Returns:
        int: Exit code (0 = success, non-zero = error)
    """
    # Create the Qt application instance
    # This is required by PyQt5 before any GUI elements can be created
    app = QtWidgets.QApplication(sys.argv)

    # Add current directory to Python path
    # This helps Python find the SXMRemote.py file that Anfatec provides
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Initialize connection variables
    dde_client = None
    connection_successful = False
    error_explanation = ""
    
    # Attempt to connect to the real SXM software via DDE
    try:
        # Try to establish communication with SXM software
        # "SXM" is the application name, "Remote" is the communication topic
        dde_client = RealDDEClient(app_name="SXM", topic="Remote")
        connection_successful = True
        print("[INFO] ✅ Connected to SXM via DDE - ONLINE mode")
        
    except ImportError as e:
        # SXMRemote module not found - this is a common issue
        error_explanation = interpret_dde_error(str(e))
        print(f"[WARN] ❌ SXMRemote module not found")
        print(f"[INFO] 🔄 Switching to OFFLINE mode (MockDDEClient)")
        
    except Exception as error:
        # Any other connection failure (SXM not running, DDE disabled, etc.)
        error_explanation = interpret_dde_error(str(error))
        print(f"[WARN] ❌ Cannot connect to SXM: {error}")
        print(f"[INFO] 🔄 Switching to OFFLINE mode (MockDDEClient)")
    
    # Create mock client if real connection failed
    # The mock client simulates SXM responses, allowing development/testing
    if not connection_successful:
        dde_client = MockDDEClient()

    # Create the main application window
    # Pass the DDE client (real or mock) to the GUI
    main_window = MainWindow(dde_client=dde_client)
    
    # Show connection status dialog to inform the user
    # This is important so users know if they're working with real hardware
    show_connection_status_dialog(main_window, connection_successful, error_explanation)
    
    # Display the main window on screen
    main_window.show()
    
    # Update window title to clearly show the current mode
    # This provides constant visual feedback about the connection status
    mode_suffix = " - ONLINE" if connection_successful else " - OFFLINE"
    main_window.setWindowTitle(main_window.windowTitle() + mode_suffix)

    # Start the Qt event loop
    # This keeps the GUI running and responsive to user interactions
    # The program will stay here until the user closes the window
    return app.exec_()


# Standard Python entry-point check
# This ensures the main() function only runs when this file is executed directly,
# not when it's imported as a module by another script
if __name__ == "__main__":
    # Use SystemExit for clean program termination
    # This ensures proper cleanup of resources
    raise SystemExit(main())