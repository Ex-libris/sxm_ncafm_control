# io_reader.py
"""
Wrapper for SXMOscilloscope.py.

Purpose:
    - Provides a simple API to read scaled channel values from the SXM driver.
    - Shields the rest of the app from Windows driver plumbing (win32 API).
"""

from typing import Optional, Callable


def make_reader() -> tuple[Optional[Callable[[str], float]], dict]:
    """
    Create a reader function for SXM oscilloscope channels.

    Process:
        - Attempts to connect to the SXM driver via Windows handle.
        - If successful:
            • Returns a `read_scaled(name)` function.
            • This reads one sample from the given channel and applies scaling.
        - If driver is unavailable:
            • Returns (None, channels_dict) so the GUI can still list channels.

    Inputs:
        None (uses system-installed SXMOscilloscope + win32 APIs).

    Outputs:
        tuple:
            (reader_function_or_None, channels_dict)

            reader_function_or_None (Callable | None):
                Function with signature:
                    read_scaled(name: str) -> float
                Returns one scaled sample in physical units.
                None if driver cannot be opened.

            channels_dict (dict):
                Channel metadata from SXMOscilloscope (index, short name, unit, scale).

    Example:
        reader, channels = make_reader()
        if reader:
            print(reader("Zsensor"))  # prints scaled float value
    """
    try:
        # Import from SXMOscilloscope (user-provided driver module)
        from SXMOscilloscope import channels as SXM_CHANNELS, DriverDataSource
        import win32file, win32con

        # Open handle to the SXM driver (Windows device API)
        handle = win32file.CreateFile(
            r"\\.\SXM",
            win32con.GENERIC_READ | win32con.GENERIC_WRITE,
            0, None,
            win32con.OPEN_EXISTING,
            win32con.FILE_ATTRIBUTE_NORMAL,
            None,
        )

        # Instantiate driver data source using open handle
        src = DriverDataSource(handle)

        def read_scaled(name: str) -> float:
            """
            Read one sample from a given channel.

            Inputs:
                name (str): Channel name from SXM_CHANNELS dict.

            Outputs:
                float: Scaled value in physical units (raw * scale).
            """
            idx, _short, _unit, scale = SXM_CHANNELS[name]
            raw = src.read_value(idx)  # returns raw integer from device
            return float(raw) * float(scale)

        return read_scaled, SXM_CHANNELS

    except Exception:
        # Driver not available → return channel list only (no reader function)
        try:
            from SXMOscilloscope import channels as SXM_CHANNELS
        except Exception:
            SXM_CHANNELS = {}
        return None, SXM_CHANNELS
