# dde_client.py
"""
DDE client bridge for SXM control.

This module acts as a thin bridge between the GUI and the SXM software,
translating Python method calls into Pascal-like commands that SXM expects.

DDE (Dynamic Data Exchange) is a Windows protocol that allows programs to
communicate with each other. The SXM software acts as a DDE server, accepting
commands in Pascal syntax (like "ScanPara('Edit23', 0.080);").

The module also maintains a "last written" cache so the GUI can show the most recent
values, even in cases where the underlying SXM system is write-only (you can send
parameters but not read them back).

Public API:
    - send_scanpara(edit_code: str, value: float) -> None
    - send_dncpara(index: int, value: float) -> None
    - last_written(ptype: str, pcode: str) -> float | None
"""

from typing import Dict, Tuple


class BaseDDE:
    """
    Base class providing shared functionality for both real and mock DDE clients.

    This class implements the common caching functionality that both the real
    SXM connection and the offline simulator need. It uses the template pattern:
    the base class handles caching, while subclasses implement the actual
    communication (either real DDE or console logging).

    Responsibilities:
        - Store a cache of the last parameters written
        - Provide a consistent interface for parameter recall
        - Define the API that subclasses must implement
    """

    def __init__(self) -> None:
        """
        Initialize the cache for storing last-written parameter values.

        The cache uses a dictionary with tuple keys to organize parameters by
        type and code. This allows the GUI to retrieve the last value sent to
        any parameter, which is essential since SXM parameters are typically
        write-only.

        Cache key format:
            (ptype, pcode) where:
            - ptype: "EDIT" for scan parameters, "DNC" for NC-AFM parameters
            - pcode: "Edit23" for scan params, "4" for DNC index

        Example cache entries:
            ("EDIT", "Edit23"): 0.080  # Amplitude reference
            ("DNC", "4"): 0.050        # Drive amplitude
        """
        self._last: Dict[Tuple[str, str], float] = {}

    def _remember(self, ptype: str, pcode: str, value: float) -> None:
        """
        Save the most recently written value in the cache.

        This is called internally after every successful parameter write.
        The cache allows the GUI to display current values and implement
        features like "restore previous settings."

        Args:
            ptype: Parameter type ("EDIT" for scan parameters, "DNC" for NC-AFM)
            pcode: Parameter code (e.g., "Edit23" for scan, "4" for DNC index)
            value: Value that was written to the parameter

        Returns:
            None (updates internal cache only)
        """
        self._last[(ptype, pcode)] = float(value)

    def last_written(self, ptype: str, pcode: str) -> float | None:
        """
        Retrieve the last value written for a given parameter.

        This is the main way for the GUI to check what value was last sent
        to any parameter. Returns None if no value has been written yet.

        Args:
            ptype: Parameter type ("EDIT" or "DNC")
            pcode: Parameter code ("Edit23", "4", etc.)

        Returns:
            The last written value as float, or None if never written

        Example:
            last_amp = client.last_written("EDIT", "Edit23")
            if last_amp is not None:
                print(f"Amplitude reference was last set to {last_amp}")
        """
        return self._last.get((ptype, pcode), None)

    # Abstract API methods that subclasses must implement
    # These define the interface the GUI will use

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """
        Send a scan parameter command (must be implemented in subclass).
        
        Scan parameters control general SPM operation like setpoints,
        gains, and measurement conditions.
        """
        raise NotImplementedError

    def send_dncpara(self, index: int, value: float) -> None:
        """
        Send a DNC parameter command (must be implemented in subclass).
        
        DNC (Dynamic Non-Contact) parameters specifically control
        NC-AFM operation like frequency, amplitude, and feedback.
        """
        raise NotImplementedError


class RealDDEClient(BaseDDE):
    """
    Real implementation that communicates with actual SXM software via DDE.

    This class wraps the SXMRemote.DDEClient provided by Anfatec and translates
    our Python API calls into the Pascal commands that SXM expects.

    Requirements:
        - SXMRemote.py file (provided by Anfatec) must be in Python path
        - SXM software must be running and have DDE enabled
        - Windows operating system (DDE is Windows-specific)
    """

    def __init__(self, app_name: str = "SXM", topic: str = "Remote") -> None:
        """
        Initialize connection to real SXM software via DDE.

        The DDE connection requires two identifiers:
        - app_name: The DDE application name (always "SXM" for Anfatec software)
        - topic: The communication topic (always "Remote" for remote control)

        Args:
            app_name: DDE application name, should be "SXM"
            topic: DDE topic name, should be "Remote"

        Raises:
            ImportError: If SXMRemote.py is not found
            Exception: If DDE connection fails (SXM not running, DDE disabled, etc.)
        """
        # Initialize the base class (sets up caching)
        super().__init__()
        
        # Import SXMRemote dynamically to provide better error messages
        # if the file is missing (which is a common setup issue)
        import importlib
        SXMRemote = importlib.import_module("SXMRemote")
        
        # Create the actual DDE connection to SXM
        # This will throw an exception if SXM is not running or DDE is disabled
        self._dde = SXMRemote.DDEClient(app_name, topic)

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """
        Send a scan parameter command to SXM software.

        Scan parameters control general microscope operation. Each Edit code
        corresponds to a specific parameter in SXM's interface. For example:
        - Edit1 = I_t setpoint (current setpoint for STM)
        - Edit23 = Amplitude reference (target amplitude for NC-AFM)
        - Edit24 = Amplitude Ki (integral gain for amplitude feedback)

        The complete mapping of Edit codes to parameters can be found in:
        - SXM Language Description document (provided with your system)
        - Parameter mapping files in this repository

        Technical note: This sends a Pascal command like "ScanPara('Edit23', 0.080);"
        to the SXM software via DDE.

        Args:
            edit_code: Must start with "Edit" followed by digits (e.g., "Edit23")
            value: The value to assign to this parameter

        Raises:
            ValueError: If edit_code format is invalid
            Exception: If DDE communication fails

        Example:
            client.send_scanpara("Edit23", 0.080)  # Set amplitude reference to 80 mV
        """
        # Validate the Edit code format before sending
        # Edit codes must be "Edit" followed by one or more digits
        if not (edit_code.startswith("Edit") and edit_code[4:].isdigit()):
            raise ValueError(f"Invalid Edit code: {edit_code!r}")
        
        # Send the Pascal command to SXM via DDE
        # SendWait() sends the command and waits for SXM to acknowledge it
        self._dde.SendWait(f"ScanPara('{edit_code}', {value});")
        
        # Cache this value so the GUI can retrieve it later
        self._remember("EDIT", edit_code, value)

    def send_dncpara(self, index: int, value: float) -> None:
        """
        Send a DNC (Dynamic Non-Contact) parameter command to SXM.

        DNC parameters specifically control NC-AFM (Non-Contact Atomic Force
        Microscopy) operation. These include frequency settings, drive amplitudes,
        and feedback parameters specific to cantilever-based measurements.

        Common DNC parameter indices:
        - Index 3: Used frequency (f₀) - the resonant frequency for oscillation
        - Index 4: Drive amplitude - voltage applied to drive the cantilever

        The complete list of DNC parameters is in your SXM documentation.

        Technical note: This sends a Pascal command like "DNCPara(4, 0.050);"
        to the SXM software via DDE.

        Args:
            index: DNC parameter index (must be non-negative integer)
            value: The value to assign to this parameter

        Raises:
            ValueError: If index is negative
            Exception: If DDE communication fails

        Example:
            client.send_dncpara(4, 0.050)  # Set drive amplitude to 50 mV
        """
        # Validate the index (must be non-negative)
        if index < 0:
            raise ValueError("DNC index must be non-negative.")
        
        # Send the Pascal command to SXM via DDE
        self._dde.SendWait(f"DNCPara({index}, {value});")
        
        # Cache this value using string representation of index as the code
        self._remember("DNC", str(index), value)


class MockDDEClient(BaseDDE):
    """
    Mock implementation for offline development and testing.

    This class simulates SXM communication without requiring the actual
    hardware or software to be present. It's useful for:
    - Developing and testing the GUI without hardware
    - Training users on the interface
    - Planning measurement sequences
    - Debugging parameter sequences

    Instead of sending real commands, it logs detailed information to the
    console and maintains the same cache as the real client, so the GUI
    behaves identically in both modes.
    """

    def __init__(self) -> None:
        """
        Initialize mock client with enhanced logging capabilities.
        
        The mock client provides rich debugging information including:
        - Command numbering for sequence tracking
        - Parameter context (what each code controls)
        - Clear indication that commands are simulated
        """
        # Initialize the base class (sets up caching)
        super().__init__()
        
        # Track command sequence for debugging
        self._command_count = 0
        
        # Let user know they're in simulation mode
        print("[MOCK] MockDDEClient initialized - all commands will be simulated")

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """
        Simulate sending a scan parameter command.

        Logs the command with contextual information about what the parameter
        controls. This helps users understand what each Edit code does and
        verify their parameter sequences are correct.

        Args:
            edit_code: Scan parameter code (e.g., "Edit23")
            value: Value to assign

        Example output:
            [MOCK] #001 ScanPara('Edit23', 0.080); // Amplitude Reference
        """
        # Increment command counter for sequence tracking
        self._command_count += 1
        
        # Provide context about what common parameters control
        # This helps users understand the meaning of Edit codes
        param_context = {
            "Edit1": "I_t setpoint",
            "Edit23": "Amplitude Reference",
            "Edit24": "Amplitude Ki", 
            "Edit32": "Amplitude Kp",
            "Edit22": "PLL Ki",
            "Edit27": "PLL Kp"
        }.get(edit_code, "Unknown parameter")
        
        # Log the simulated command with context
        print(f"[MOCK] #{self._command_count:03d} ScanPara('{edit_code}', {value}); // {param_context}")
        
        # Update cache just like the real client would
        self._remember("EDIT", edit_code, value)

    def send_dncpara(self, index: int, value: float) -> None:
        """
        Simulate sending a DNC parameter command.

        Logs the command with contextual information about what the DNC
        parameter controls. This is especially helpful for NC-AFM parameters
        which are less familiar to many users.

        Args:
            index: DNC parameter index
            value: Value to assign

        Example output:
            [MOCK] #002 DNCPara(4, 0.050); // Drive amplitude
        """
        # Increment command counter for sequence tracking
        self._command_count += 1
        
        # Provide context about what common DNC parameters control
        # This helps users understand NC-AFM parameter meanings
        dnc_context = {
            3: "Used Frequency (f₀)",
            4: "Drive amplitude"
        }.get(index, f"DNC parameter {index}")
        
        # Log the simulated command with context
        print(f"[MOCK] #{self._command_count:03d} DNCPara({index}, {value}); // {dnc_context}")
        
        # Update cache just like the real client would
        self._remember("DNC", str(index), value)