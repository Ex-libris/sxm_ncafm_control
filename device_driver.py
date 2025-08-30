# ioctl_client.py
"""
Windows IOCTL client for direct communication with the SXM driver.

This module provides low-level access to the Anfatec SXM controller driver,
bypassing the SXM software to read channel data directly from the hardware.
This approach offers several advantages:

1. Lower latency - no intermediate software layers
2. Higher sampling rates - direct driver access
3. Real-time monitoring - immediate channel readings
4. Independence - works even if SXM GUI is busy

IOCTL (Input/Output Control) is a Windows system call that allows programs
to communicate directly with device drivers. The SXM driver exposes channel
data through a specific IOCTL interface that this module wraps.

Dependencies: 
    - pywin32 (win32file, win32con) for Windows API access
    - ctypes for low-level memory operations
    - Windows operating system (IOCTL is Windows-specific)

Technical note: This implementation matches the protocol used by SXMOscilloscope.py
to ensure compatibility with existing Anfatec software.
"""

import ctypes
from ctypes import c_long
import win32file, win32con

# ---- Windows IOCTL Control Code Generation ----
# These constants define how to construct the control code for the IOCTL call
# They follow the Windows DDK (Driver Development Kit) standard format

FILE_DEVICE_UNKNOWN = 0x00000022  # Device type for custom/unknown devices
METHOD_BUFFERED     = 0x00000000  # Buffered I/O method (most common)
FILE_ANY_ACCESS     = 0x0000      # Access level (read/write permissions)


def CTL_CODE(DeviceType, Access, Function_code, Method):
    """
    Generate a Windows IOCTL control code.
    
    Windows uses a 32-bit control code to identify specific driver operations.
    The code encodes the device type, access requirements, function number,
    and I/O method in specific bit positions.
    
    This function follows the standard Windows CTL_CODE macro used in C/C++
    driver development.
    
    Args:
        DeviceType: Type of device (FILE_DEVICE_UNKNOWN for custom drivers)
        Access: Access level required (FILE_ANY_ACCESS for general use)
        Function_code: Specific function number (0xF0D for SXM channel read)
        Method: I/O method (METHOD_BUFFERED for safe data transfer)
    
    Returns:
        32-bit integer control code for use with DeviceIoControl
    
    Technical details:
        Bit layout: [DeviceType:16][Access:2][Function:12][Method:2]
    """
    return (DeviceType << 16) | (Access << 14) | (Function_code << 2) | Method


# Generate the specific IOCTL code for reading SXM channels
# Function code 0xF0D is defined by Anfatec's driver specification
IOCTL_GET_KANAL = CTL_CODE(FILE_DEVICE_UNKNOWN, FILE_ANY_ACCESS, 0xF0D, METHOD_BUFFERED)
IOCTL_SET_CHANNEL = CTL_CODE(FILE_DEVICE_UNKNOWN, FILE_ANY_ACCESS, 0xF18, METHOD_BUFFERED)
# ---- SXM Driver Device Path ----
# This is the Windows device path for the SXM driver
# DEVICE_PATH is the symbolic link created by the Anfatec driver
DEVICE_PATH = r"\\.\SXM"

# ---- Channel Definition Map ----
# This comprehensive map defines all available channels on the SXM system
# Format: 'channel_name': (driver_index, short_label, unit, scale_factor)
#
# driver_index: The hardware channel index used by the driver
#               Negative values typically indicate output channels (DACs)
#               Positive values typically indicate input channels (ADCs)
#
# short_label:  Abbreviated name used in hardware documentation
# unit:         Physical unit of the measurement
# scale_factor: Multiplication factor to convert raw integer to physical units
#
# These values are system-specific and should match your SXM configuration.
# They are typically determined during system calibration.

CHANNELS = {
    # ---- Scan Control Channels (DAC outputs) ----
    'Topo':       (  0, 'DAC0',    'nm', -2.60914e-07),  # Topography feedback
    'Bias':       ( -1, 'DAC1',    'mV',  9.4e-06),      # Sample bias voltage
    'x-direction':( -2, 'DAC2',    'nm',  1.34e-06),     # X scan position
    'y-direction':( -3, 'DAC3',    'nm',  1.34e-06),     # Y scan position
    'DA1':        ( -4, 'DAC4',     'V', -9.41e-09),     # General purpose DAC
    
    # ---- NC-AFM Control Channels ----
    'Frequency':  ( -9, 'DAC9',    'Hz',  0.00232831),   # Cantilever drive frequency
    'Drive':      (-10, 'DAC10',    'V',  4.97789e-09),  # Cantilever drive amplitude
    'QPlusAmpl':  (-12, 'DAC12',    'V',  3.64244e-09),  # qPlus amplitude signal
    'Phase':      (-13, 'DAC13',    '°',  0.001),        # Phase between drive and response
    
    # ---- Lock-in Amplifier Channels (X/Y components) ----
    'Lia1X':      (-14, 'DAC14',    'A',  1.56618e-19),  # Lock-in 1 X component
    'Lia1Y':      (-15, 'DAC15',    'A',  1.56618e-19),  # Lock-in 1 Y component
    'Lia2X':      (-16, 'DAC16',    'A',  1.56618e-19),  # Lock-in 2 X component
    'Lia2Y':      (-17, 'DAC17',    'A',  1.56618e-19),  # Lock-in 2 Y component
    'Lia3X':      (-18, 'DAC18',    'A',  1.56618e-19),  # Lock-in 3 X component
    'Lia3Y':      (-19, 'DAC19',    'A',  1.56618e-19),  # Lock-in 3 Y component
    
    # ---- Lock-in Amplifier Channels (R/Phi polar form) ----
    'Lia1R':      (-22, 'DAC22',    'A',  9.51067e-20),  # Lock-in 1 magnitude
    'Lia2R':      (-23, 'DAC23',    'A',  9.51067e-20),  # Lock-in 2 magnitude
    'Lia3R':      (-24, 'DAC24',    'A',  9.51067e-20),  # Lock-in 3 magnitude
    'Lia1Phi':    (-27, 'DAC27',    '*',  0.001),        # Lock-in 1 phase
    'Lia2Phi':    (-28, 'DAC28',    '*',  0.001),        # Lock-in 2 phase
    'Lia3Phi':    (-29, 'DAC29',    '*',  0.001),        # Lock-in 3 phase
    
    # ---- Derived Measurements ----
    'df':         (-40, 'DAC40',    'Hz', 0.00232838),   # Frequency shift (f - f₀)
    
    # ---- Analog Input Channels (ADC inputs) ----
    'It_ext':     ( 32, 'ADC0',     'A',  1.011e-19),    # External current measurement
    'QPlus_ext':  ( 33, 'ADC1',    'mV',  1.008e-05),    # Digitized qPlus experimental signal
    'AD1':        ( 34, 'ADC2',     'V',  1.012e-08),    # General purpose ADC 1
    'AD2':        ( 35, 'ADC3',    'mV',  1.011e-05),    # General purpose ADC 2
    'InA':        (  8, 'ADC4',     'V',  3.07991e-09),  # Galvanically isolated qplus signal. Used as input for the amplitude lock-in and the PLL ('LiaO')
    'It_to_PC':   ( 12, 'ADC9',     'A',  8.04187e-20),  # Current to PC (for logging)
    'Zeit':       ( 23, 'ADC12',    's',  0.001),        # Time channel (from german "Zeit" = time)
    'AD3':        ( 36, 'ADC13',   'mV',  1.01e-05),     # General purpose ADC 3
    'AD4':        ( 37, 'ADC14',   'mV',  1.013e-05),    # General purpose ADC 4
    'AD5':        ( 38, 'ADC15',   'mV',  1.01e-05),     # General purpose ADC 5
    'AD6':        ( 39, 'ADC16',   'mV',  1.009e-05),    # General purpose ADC 6
    'minmax':     ( 47, 'ADC21',    'A',  3.35e-07)      # Min/max detector output
}


class SXMIOCTL:
    """
    Direct Windows IOCTL interface to the SXM controller driver.
    
    This class provides a simple Python interface to read channel data directly
    from the Anfatec SXM driver without going through the SXM software. This
    enables real-time monitoring with minimal latency.
    
    The class handles:
    - Opening/closing the Windows device driver
    - Formatting IOCTL requests properly
    - Converting raw integer values to physical units
    - Providing human-readable channel names
    
    Typical workflow:
        1. Create SXMIOCTL instance (opens driver connection)
        2. Read channels by name using read_scaled()
        3. Instance automatically closes driver on cleanup
    """
    
    def __init__(self, device_path: str = DEVICE_PATH):
        """
        Initialize connection to the SXM driver.
        
        Opens a Windows file handle to the SXM device driver. This handle
        will be used for all subsequent IOCTL operations to read channel data.
        
        Args:
            device_path: Windows device path (default: "backslash SXM")
            
        Raises:
            Exception: If driver cannot be opened (common causes:
                      - SXM driver not installed
                      - Driver not started  
                      - Insufficient permissions
                      - Hardware not connected)
        
        Technical note: Uses CreateFile with GENERIC_READ|WRITE and OPEN_EXISTING
        flags, which is the standard approach for device driver access.
        """
        # Open a Windows file handle to the SXM device driver
        # GENERIC_READ|WRITE: Request both read and write access
        # OPEN_EXISTING: Device must already exist (driver must be loaded)
        # FILE_ATTRIBUTE_NORMAL: Standard file attributes
        self.handle = win32file.CreateFile(
            device_path,
            win32con.GENERIC_READ | win32con.GENERIC_WRITE,
            0,                                    # No sharing
            None,                                 # Default security
            win32con.OPEN_EXISTING,               # Must exist
            win32con.FILE_ATTRIBUTE_NORMAL,       # Normal attributes
            None                                  # No template file
        )
        
        # Pre-allocate input buffer for channel index
        # IOCTL calls require a buffer to pass the channel index to the driver
        # We allocate this once and reuse it for efficiency
        self._inbuf = ctypes.create_string_buffer(ctypes.sizeof(c_long))

    def close(self):
        """
        Explicitly close the driver connection.
        
        While the destructor will also close the handle, it's good practice
        to explicitly close resources when you're done with them.
        """
        if self.handle:
            win32file.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        """
        Destructor ensures driver handle is closed during garbage collection.
        
        This provides automatic cleanup if the user forgets to call close()
        explicitly. The try/except ensures no exceptions during cleanup.
        """
        try:
            self.close()
        except Exception:
            # Ignore cleanup errors - object is being destroyed anyway
            pass

    def read_raw(self, channel_index: int) -> int:
        """
        Read raw integer value from a specific driver channel.
        
        This is the low-level interface that communicates directly with the
        SXM driver using Windows IOCTL calls. The driver returns raw integer
        values that need to be scaled to physical units.
        
        Args:
            channel_index: Hardware channel index (can be negative for DACs)
            
        Returns:
            Raw integer value from the driver (unscaled)
            
        Raises:
            Exception: If IOCTL call fails (driver error, invalid channel, etc.)
            
        Technical details:
            1. Puts channel_index into input buffer as a C long integer
            2. Calls DeviceIoControl with IOCTL_GET_KANAL control code
            3. Driver returns 4 bytes containing the raw channel value
            4. Converts bytes back to C long integer
        """
        # Copy channel index into the pre-allocated input buffer
        # This converts Python int to C long in the buffer format expected by driver
        ctypes.memmove(
            self._inbuf, 
            ctypes.byref(c_long(channel_index)), 
            ctypes.sizeof(c_long)
        )
        
        # Execute the IOCTL call to read channel data
        # DeviceIoControl sends control code + input buffer to driver
        # Driver processes request and returns output buffer with channel value
        out = win32file.DeviceIoControl(
            self.handle,              # Device handle
            IOCTL_GET_KANAL,         # Control code (what operation to perform)
            self._inbuf,             # Input buffer (channel index)
            ctypes.sizeof(c_long)    # Expected output size (4 bytes for long)
        )
        
        # Convert output bytes back to a C long integer
        return c_long.from_buffer_copy(out).value

    def read_scaled(self, name: str) -> float:
        """
        Read channel data by human-friendly name and return in physical units.
        
        This is the main interface users should use. It handles the complete
        process of reading a channel and converting to meaningful units.
        
        Args:
            name: Human-readable channel name (e.g., 'Frequency', 'Drive', 'Phase')
                 Must be a key in the CHANNELS dictionary
                 
        Returns:
            Channel value in physical units (Hz, V, degrees, etc.)
            
        Raises:
            KeyError: If channel name is not recognized
            Exception: If driver read fails
            
        Examples:
            frequency_hz = sxm.read_scaled('Frequency')      # Returns Hz
            drive_volts = sxm.read_scaled('Drive')           # Returns V  
            amplitude_volts = sxm.read_scaled('QPlusAmpl')   # Returns V
            phase_degrees = sxm.read_scaled('Phase')         # Returns degrees
        """
        # Look up channel configuration
        if name not in CHANNELS:
            available = list(CHANNELS.keys())
            raise KeyError(f"Unknown channel '{name}'. Available channels: {available}")
        
        # Extract channel parameters
        idx, _short, _unit, scale = CHANNELS[name]
        
        # Read raw value from driver
        raw = self.read_raw(idx)
        
        # Apply calibration scaling to get physical units
        # Raw driver values are integers; scaling converts to physical units
        return float(raw) * float(scale)
        
    def write_raw(self, write_index: int, counts: int) -> None:
        """
        Write a 32-bit signed integer to a driver channel (DAC).
        """
        import struct
        buf = struct.pack("<ll", int(write_index), int(counts))
        win32file.DeviceIoControl(self.handle, IOCTL_SET_CHANNEL, buf, 0)

    def write_unit(self, name: str, value: float) -> int:
        """
        Write a value in physical units to the matching DAC.
        For 'Topo' (read_id=0).
        Returns the counts sent.
        """
        if name not in CHANNELS:
            raise KeyError(name)
        read_id, _short, _unit, scale = CHANNELS[name]
        # read_id -> write_id mapping (same rule used in your earlier code)
        write_id = 0
        counts = int(round(float(value) / float(scale)))
        self.write_raw(write_id, counts)
        return counts