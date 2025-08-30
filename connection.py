# sxm_ncafm_control/connection.py
from .dde_client import RealDDEClient, MockDDEClient
from .device_driver import SXMIOCTL
from . import common
class SXMConnection:
    """
    Holds both DDE and IOCTL handles.
    If offline, provides mock fallbacks.
    """
    def __init__(self):
        # DDE
        try:
            self.dde = RealDDEClient()
        except Exception as e:
            common.offline_message("DDE connection", e, "MockDDEClient")
            self.dde = MockDDEClient()

        # IOCTL
        try:
            self.driver = SXMIOCTL()
        except Exception as e:
            common.offline_message("Microscope driver", e, "mock driver")
            self.driver = None

    @property
    def is_offline(self):
        return isinstance(self.dde, MockDDEClient) or self.driver is None
