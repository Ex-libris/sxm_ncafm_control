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
    - read_channel(index: int) -> float
    - read_topography() -> float
    - last_written(ptype: str, pcode: str) -> float | None
"""

from typing import Dict, Tuple


class BaseDDE:
    """Base class for DDE clients with last-written value caching."""

    def __init__(self) -> None:
        """Initializes the DDE client's internal write cache."""
        self._last: Dict[Tuple[str, str], float] = {}

    def _remember(self, ptype: str, pcode: str, value: float) -> None:
        """Stores the last written value for a parameter.

        Args:
            ptype (str): Parameter type (e.g., "EDIT", "DNC").
            pcode (str): Parameter code or identifier.
            value (float): Value written to the parameter.
        """
        self._last[(ptype.upper(), str(pcode))] = float(value)

    def last_written(self, ptype: str, pcode: str):
        """Returns the most recently written value for a parameter.

        Args:
            ptype (str): Parameter type.
            pcode (str): Parameter code or identifier.

        Returns:
            float | None: Last written value, or None if not found.
        """
        return self._last.get((ptype.upper(), str(pcode)))


class RealDDEClient(BaseDDE):
    """Real DDE client that communicates with SXM via SXMRemote.DDEClient (Windows only)."""

    def __init__(self, app_name: str = "SXM", topic: str = "Remote") -> None:
        """Initializes the DDE client and connects to the SXM application.

        Args:
            app_name (str): Name of the DDE application. Defaults to "SXM".
            topic (str): DDE topic. Defaults to "Remote".

        Raises:
            ImportError: If SXMRemote.py is not found or importable.
        """
        super().__init__()
        try:
            import SXMRemote  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SXMRemote.py not found. Put it next to this module or in PYTHONPATH."
            ) from e

        self._dde = SXMRemote.DDEClient(app_name, topic)
        self._command_count = 0

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """Sends a ScanPara command (e.g., Edit23) to SXM.

        Args:
            edit_code (str): Edit code name (e.g., "Edit23").
            value (float): Value to assign.

        Raises:
            ValueError: If edit_code is not a valid EditNN format.
        """
        if not (isinstance(edit_code, str) and edit_code.startswith("Edit")):
            raise ValueError("edit_code must look like 'EditNN' (e.g., 'Edit23').")
        self._dde.SendWait(f"ScanPara('{edit_code}', {value});")
        self._remember("EDIT", edit_code, value)
        self._command_count += 1

    def send_dncpara(self, index: int, value: float) -> None:
        """Sends a DNCPara command (by index) to SXM.

        Args:
            index (int): Parameter index.
            value (float): Value to assign.

        Raises:
            ValueError: If index is negative.
        """
        if index < 0:
            raise ValueError("DNC index must be non-negative.")
        self._dde.SendWait(f"DNCPara({index}, {value});")
        self._remember("DNC", str(index), value)

    def read_channel(self, index: int) -> float:
        """Reads the value of a specified channel via DDE.

        Args:
            index (int): Channel index (e.g., 0 for topography).

        Returns:
            float: Current value from the SXM channel.
        """
        return float(self._dde.GetChannel(int(index)))

    def read_topography(self) -> float:
        """Reads topography (channel 0) in nanometers.

        Returns:
            float: Topography value.
        """
        return self.read_channel(0)


class MockDDEClient(BaseDDE):
    """Offline mock client for development and testing without SXM software."""

    def __init__(self) -> None:
        """Initializes the mock client with internal command tracking."""
        super().__init__()
        self._command_count = 0

    def send_scanpara(self, edit_code: str, value: float) -> None:
        """Simulates sending a ScanPara command and logs it.

        Args:
            edit_code (str): Edit code name (e.g., "Edit23").
            value (float): Value to assign.

        Raises:
            ValueError: If edit_code is not a valid EditNN format.
        """
        if not (isinstance(edit_code, str) and edit_code.startswith("Edit")):
            raise ValueError("edit_code must look like 'EditNN' (e.g., 'Edit23').")
        self._command_count += 1
        print(f"[MOCK] #{self._command_count:03d} ScanPara('{edit_code}', {value});")
        self._remember("EDIT", edit_code, value)

    def send_dncpara(self, index: int, value: float) -> None:
        """Simulates sending a DNCPara command and logs it.

        Args:
            index (int): Parameter index.
            value (float): Value to assign.

        Raises:
            ValueError: If index is negative.
        """
        if index < 0:
            raise ValueError("DNC index must be non-negative.")
        self._command_count += 1
        desc = {
            3: "Used Frequency (f0)",
            4: "Drive amplitude",
        }.get(index, f"DNC parameter {index}")
        print(f"[MOCK] #{self._command_count:03d} DNCPara({index}, {value}); // {desc}")
        self._remember("DNC", str(index), value)

    def read_channel(self, index: int) -> float:
        """Simulates reading a channel value.

        Channel 0 (topography) increases slowly to mimic drift.

        Args:
            index (int): Channel index.

        Returns:
            float: Simulated value.
        """
        base = getattr(self, "_sim_base", 0.0)
        base += 0.001
        self._sim_base = base
        if int(index) == 0:
            return base
        return 0.0

    def read_topography(self) -> float:
        """Returns simulated topography from channel 0.

        Returns:
            float: Simulated topography value.
        """
        return self.read_channel(0)
