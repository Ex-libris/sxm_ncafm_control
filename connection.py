# sxm_ncafm_control/connection.py
from .dde_client import RealDDEClient, MockDDEClient
from .device_driver import SXMIOCTL
from . import common
class SXMConnection:
    """
    Holds both DDE and IOCTL handles.
    If offline, provides mock fallbacks.

    Both underlying resources are Windows handles that can go stale over a
    long-running session (SXM host restarted, driver reset, USB hiccup)
    without the Python side noticing until the next call fails. `reconnect()`
    tears both down and re-establishes them from scratch.
    """
    def __init__(self):
        self.dde = None
        self.driver = None
        self._connect()

    def _connect(self):
        """(Re)establish DDE and IOCTL, closing any existing driver handle first."""
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass

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

    def reconnect(self) -> bool:
        """
        Tear down and re-establish both connections.

        The caller is responsible for pausing anything actively using the
        old `dde`/`driver` objects beforehand, and for pushing the fresh
        `self.dde`/`self.driver` out to anything holding its own reference
        to the old ones afterward (this object does not track who else
        holds those references).

        Returns
        -------
        bool
            True if now fully online (real DDE + driver), False if still
            (or newly) offline.
        """
        self._connect()
        return not self.is_offline

    @property
    def is_offline(self):
        return isinstance(self.dde, MockDDEClient) or self.driver is None
