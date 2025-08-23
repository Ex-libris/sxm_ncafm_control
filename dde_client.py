# dde_client.py
"""
DDE client bridge for SXM control.

This module acts as a thin bridge between the GUI and the SXM software,
translating Python method calls into Pascal-like commands that SXM expects.

It also maintains a "last written" cache so the GUI can show the most recent
values, even in cases where the underlying SXM system is write-only.

Public API:
    - send_scanpara(edit_code: str, value: float) -> None
    - send_dncpara(index: int, value: float) -> None
    - last_written(ptype: str, pcode: str) -> float | None
"""

from typing import Dict, Tuple


class BaseDDE:
    """
    Base class providing shared functionality for both real and mock DDE clients.

    Responsibilities:
        - Store a cache of the last parameters written.
        - Provide an interface that RealDDEClient and MockDDEClient must implement.
    """

    def __init__(self) -> None:
        """
        Initialize the cache for storing last-written parameter values.

        Cache key format:
            (ptype, pcode), e.g. ("EDIT", "Edit23") or ("DNC", "4")
        """
        self._last: Dict[Tuple[str, str], float] = {}

    def _remember(self, ptype: str, pcode: str, value: float) -> None:
        """
        Save the most recently written value in the cache.

        Inputs:
            ptype (str): Parameter type, e.g. "EDIT" or "DNC".
            pcode (str): Parameter code (e.g., "Edit23" or index string).
            value (float): Value assigned to the parameter.

        Outputs:
            None (updates internal cache).
        """
        self._last[(ptype, pcode)] = float(value)

    def last_written(self, ptype: str, pcode: str) -> float | None:
        """
        Retrieve the last value written for a given parameter.

        Inputs:
            ptype (str): Parameter type ("EDIT" or "DNC").
            pcode (str): Parameter code ("Edit23", "4", etc.).

        Outputs:
            float | None: Last written value if available, else None.
        """
        return self._last.get((ptype, pcode), None)

    # Abstract API methods the GUI will use (must be implemented by subclasses)
    def send_scanpara(self, edit_code: str, value: float) -> None:
        """Send a scan parameter command (must be implemented in subclass)."""
        raise NotImplementedError

    def send_dncpara(self, index: int, value: float) -> None:
        """Send a DNC parameter command (must be implemented in subclass)."""
        raise NotImplementedError


class RealDDEClient(BaseDDE):
    """
    Real implementation that wraps SXMRemote.DDEClient
    and sends commands directly to SXM.

    This requires the SXMRemote library to be available in your environment.
    """

    def __init__(self, app_name: str = "SXM", topic: str = "Remote") -> None:
        """
        Initialize the real DDE client.

        Inputs:
            app_name (str): DDE application name (default: "SXM").
            topic (str): DDE topic name (default: "Remote").

        Outputs:
            None (creates and stores a DDE connection).
        """
        super().__init__()
        import importlib
        SXMRemote = importlib.import_module("SXMRemote")
        self._dde = SXMRemote.DDEClient(app_name, topic)

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """
        Send a scan parameter command to SXM. There is a 1:1 mapping between edit codes and a parameter in SXM. 
        For instancte, Edit1 = I_t setpoint in the parameter window. The edit codes can be found in the SXM Language Descrion document of your machine, but also somewhere in this repository.

        Example SXM Pascal command:
            ScanPara('Edit23', 0.080);

        Inputs:
            edit_code (str): Must start with "Edit" followed by digits (e.g. "Edit23").
            value (float): Value to assign.

        Outputs:
            None (sends command to SXM, updates cache).

        Raises:
            ValueError: If edit_code is invalid.
        """
        if not (edit_code.startswith("Edit") and edit_code[4:].isdigit()):
            raise ValueError(f"Invalid Edit code: {edit_code!r}")
        self._dde.SendWait(f"ScanPara('{edit_code}', {value});")
        self._remember("EDIT", edit_code, value)

    def send_dncpara(self, index: int, value: float) -> None:
        """
        Send a DNC (Dynamic Non-contact, the module for nc AFM) parameter command to SXM.

        Example SXM Pascal command:
            DNCPara(4, 0.050);

        Inputs:
            index (int): Must be non-negative, indicates DNC parameter index.
            value (float): Value to assign.

        Outputs:
            None (sends command to SXM, updates cache).

        Raises:
            ValueError: If index < 0.
        """
        if index < 0:
            raise ValueError("DNC index must be non-negative.")
        self._dde.SendWait(f"DNCPara({index}, {value});")
        self._remember("DNC", str(index), value)


class MockDDEClient(BaseDDE):
    """
    Mock implementation for offline development and testing.

    Instead of sending real commands, it prints them to stdout and
    updates the cache so the GUI behaves normally.
    """

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """
        Simulate sending a scan parameter.

        Inputs:
            edit_code (str): Scan parameter code (e.g., "Edit23").
            value (float): Value to assign.

        Outputs:
            None (prints simulated command, updates cache).
        """
        print(f"[MOCK] ScanPara('{edit_code}', {value});")
        self._remember("EDIT", edit_code, value)

    def send_dncpara(self, index: int, value: float) -> None:
        """
        Simulate sending a DNC parameter.

        Inputs:
            index (int): DNC parameter index.
            value (float): Value to assign.

        Outputs:
            None (prints simulated command, updates cache).
        """
        print(f"[MOCK] DNCPara({index}, {value});")
        self._remember("DNC", str(index), value)
