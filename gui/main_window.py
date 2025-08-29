"""
Main application window for NC-AFM control suite.

This module defines the MainWindow class, which hosts a tabbed interface for
interacting with a scanning probe microscope. It integrates multiple functional
tabs for parameter tuning, signal monitoring, test automation, and calibration.

Tabs:
    - Parameters: Adjust feedback gains, amplitude reference, and other settings.
    - Step Test: Automatically switch parameters to test system response.
    - Scope: View real-time signals from the microscope.
    - Suggested Setup: View recommended starting values.
    - QPlus Calibration: Sweep amplitude and calibrate delta-topography response.
    - Constant Height: Control Z in constant-height mode.

Safety:
    Displays a footer warning if voltage-like parameters exceed ±10 V.
"""

from PyQt5 import QtWidgets

from sxm_ncafm_control.gui.params_tab import ParamsTab
from sxm_ncafm_control.gui.step_test_tab import StepTestTab
from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
from sxm_ncafm_control.gui.scope_tab import ScopeTab
from sxm_ncafm_control.gui.qplus_calibration_tab import QplusCalibrationTab
from sxm_ncafm_control.gui.z_const_acquisition import ZConstAcquisitionTab


class MainWindow(QtWidgets.QWidget):
    """
    Main window containing all NC-AFM control tabs and safety footer.

    This widget sets up and displays a tabbed interface for different aspects of
    microscope configuration and monitoring. All tabs are connected to a shared
    DDE client and IOCTL driver, which are provided via SXMConnection.

    Attributes
    ----------
    params_tab : ParamsTab
        Tab for setting SXM parameters.
    step_tab : StepTestTab
        Tab for executing step tests.
    scope_tab : ScopeTab
        Tab for plotting real-time signal traces.
    suggest_tab : SuggestedTab
        Tab displaying recommended initial settings.
    qplus_tab : QplusCalibrationTab
        Tab for running QPlus amplitude calibration.
    topo_hold_tab : ZConstAcquisitionTab
        Tab for Z-constant / constant height control.
    tabs : QTabWidget
        Main tab widget containing all sub-tabs.
    """

    def __init__(self, conn):
        """
        Initialize the main window and all functional tabs.

        Parameters
        ----------
        conn : SXMConnection
            Centralized connection object providing:
            - conn.dde: DDE client (real or mock)
            - conn.driver: IOCTL driver (or None if unavailable)
        """
        super().__init__()
        self.setWindowTitle("NC-AFM Control Suite")
        self.resize(1000, 700)

        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()

        # Create all tabs, sharing the same connection handles
        self.params_tab = ParamsTab(conn.dde)
        self.step_tab = StepTestTab(conn.dde)
        self.scope_tab = ScopeTab(conn.dde, conn.driver)
        self.suggest_tab = SuggestedTab(conn.dde, self.params_tab)
        self.qplus_tab = QplusCalibrationTab(conn.dde)
        self.topo_hold_tab = ZConstAcquisitionTab(conn.dde, conn.driver)

        # Link StepTest to Scope and Tabs
        self.step_tab.scope_tab = self.scope_tab
        self.step_tab.tabs_widget = self.tabs
        self.step_tab.scope_tab_index = 2  # Tab order: 0=Parameters, 1=Step Test, 2=Scope, ...

        # Add tabs to UI
        self.tabs.addTab(self.params_tab, "Parameters")
        self.tabs.addTab(self.step_tab, "Step Test")
        self.tabs.addTab(self.scope_tab, "Scope")
        self.tabs.addTab(self.suggest_tab, "Suggested Setup")
        self.tabs.addTab(self.qplus_tab, "QPlus Amplitude calibration")
        self.tabs.addTab(self.topo_hold_tab, "Constant Height")

        # Connect custom parameters from ParamsTab to StepTestTab
        self.params_tab.custom_params_changed.connect(self.step_tab.set_custom_params)

        layout.addWidget(self.tabs)

        # Safety footer
        footer = QtWidgets.QLabel(
            "⚠ Check SXM units & never exceed ±10 V without attenuation. <a href='#'>Details</a>"
        )
        footer.setOpenExternalLinks(False)
        footer.linkActivated.connect(
            lambda _: QtWidgets.QMessageBox.information(
                self,
                "Safety details",
                "Units follow the current SXM GUI (Hz/kHz, V/mV/µV...).\n"
                "Voltage-like channels (Amplitude Ref, Drive) should never exceed ±10 V "
                "unless a hardware divider is in place.\n"
                "Step Test sends values as entered – verify LOW/HIGH against the GUI units."
            )
        )
        footer.setStyleSheet(
            "color:#6b5900; background:#fff7da; border:1px solid #e6d9a2; "
            "border-radius:6px; padding:6px;"
        )
        layout.addWidget(footer)
