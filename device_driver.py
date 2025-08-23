# ioctl_client.py
"""
Windows IOCTL client for the SXM driver, matching your SXMOscilloscope.py.

- Opens SXM driver
- Uses the same CTL_CODE computation and IOCTL (0xF0D function code)
- Reads one channel at a time: returns the *scaled* value in the documented unit

Dependencies: pywin32 (win32file, win32con), ctypes.
"""

import ctypes
from ctypes import c_long
import win32file, win32con

# ---- CTL_CODE helper (same as in your scope) ----
FILE_DEVICE_UNKNOWN = 0x00000022
METHOD_BUFFERED     = 0x00000000
FILE_ANY_ACCESS     = 0x0000

def CTL_CODE(DeviceType, Access, Function_code, Method):
    return (DeviceType << 16) | (Access << 14) | (Function_code << 2) | Method

IOCTL_GET_KANAL = CTL_CODE(FILE_DEVICE_UNKNOWN, FILE_ANY_ACCESS, 0xF0D, METHOD_BUFFERED)

# ---- Device path (same as in your scope) ----
DEVICE_PATH = r"\\.\SXM"

# ---- Channels map (name -> (driver_index, short_label, unit, scale)) ----
# Copied from your SXMOscilloscope.py so values match your running system.
CHANNELS = {
    'Topo':       (  0, 'DAC0',    'nm', -2.60914e-07),
    'Bias':       ( -1, 'DAC1',    'mV',  9.4e-06),
    'x-direction':( -2, 'DAC2',    'nm',  1.34e-06),
    'y-direction':( -3, 'DAC3',    'nm',  1.34e-06),
    'DA1':        ( -4, 'DAC4',     'V', -9.41e-09),
    'Frequency':  ( -9, 'DAC9',    'Hz',  0.00232831),
    'Drive':      (-10, 'DAC10',    'V',  4.97789e-09),
    'QPlusAmpl':  (-12, 'DAC12',    'V',  3.64244e-09),
    'Phase':      (-13, 'DAC13',    '°',  0.001),
    'Lia1X':      (-14, 'DAC14',    'A',  1.56618e-19),
    'Lia1Y':      (-15, 'DAC15',    'A',  1.56618e-19),
    'Lia2X':      (-16, 'DAC16',    'A',  1.56618e-19),
    'Lia2Y':      (-17, 'DAC17',    'A',  1.56618e-19),
    'Lia3X':      (-18, 'DAC18',    'A',  1.56618e-19),
    'Lia3Y':      (-19, 'DAC19',    'A',  1.56618e-19),
    'Lia1R':      (-22, 'DAC22',    'A',  9.51067e-20),
    'Lia2R':      (-23, 'DAC23',    'A',  9.51067e-20),
    'Lia3R':      (-24, 'DAC24',    'A',  9.51067e-20),
    'Lia1Phi':    (-27, 'DAC27',    '*',  0.001),
    'Lia2Phi':    (-28, 'DAC28',    '*',  0.001),
    'Lia3Phi':    (-29, 'DAC29',    '*',  0.001),
}

class SXMIOCTL:
    """
    Simple, explicit Windows IOCTL client for SXM.

    Usage:
        sxm = SXMIOCTL()  # opens the device
        f0_hz = sxm.read_scaled('Frequency')
        drive_v = sxm.read_scaled('Drive')
        amp_v = sxm.read_scaled('QPlusAmpl')
        phase_deg = sxm.read_scaled('Phase')
    """
    def __init__(self, device_path: str = DEVICE_PATH):
        self.handle = win32file.CreateFile(
            device_path,
            win32con.GENERIC_READ | win32con.GENERIC_WRITE,
            0, None, win32con.OPEN_EXISTING,
            win32con.FILE_ATTRIBUTE_NORMAL, None
        )
        # preallocate an input buffer for channel index (LONG)
        self._inbuf = ctypes.create_string_buffer(ctypes.sizeof(c_long))

    def close(self):
        if self.handle:
            win32file.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_raw(self, channel_index: int) -> int:
        """
        Issue DeviceIoControl to read one channel's raw integer from the driver.
        Matches your oscilloscope code exactly.
        """
        ctypes.memmove(self._inbuf, ctypes.byref(c_long(channel_index)), ctypes.sizeof(c_long))
        out = win32file.DeviceIoControl(self.handle, IOCTL_GET_KANAL, self._inbuf, ctypes.sizeof(c_long))
        return c_long.from_buffer_copy(out).value

    def read_scaled(self, name: str) -> float:
        """
        Read by human-friendly name and return scaled physical units.
        For example:
            'Frequency'  -> Hz (float)
            'Drive'      -> V  (float)
            'QPlusAmpl'  -> V  (float)
            'Phase'      -> degrees (float)
        """
        if name not in CHANNELS:
            raise KeyError(f"Unknown channel {name!r}. Known: {list(CHANNELS.keys())}")
        idx, _short, _unit, scale = CHANNELS[name]
        raw = self.read_raw(idx)
        return float(raw) * float(scale)
