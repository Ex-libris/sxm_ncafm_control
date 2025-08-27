# gui/main_window.py
from PyQt5 import QtWidgets
from sxm_ncafm_control.gui.params_tab import ParamsTab
from sxm_ncafm_control.gui.step_test_tab import StepTestTab
from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
from sxm_ncafm_control.gui.scope_tab import ScopeTab

class MainWindow(QtWidgets.QWidget):
    """
    Main application window for NC-AFM control suite.

    Provides a tabbed interface for controlling and monitoring the scanning probe microscope.
    Tabs include parameter adjustment, signal monitoring, step testing, and suggested setup.

    Tabs:
        - Parameters: Adjust feedback gains, amplitude reference, and other settings.
        - Scope: View real-time signals from the microscope.
        - Step Test: Automatically switch parameters to test system response.
        - Suggested Setup: View recommended starting values.

    Safety:
        - Displays a warning if voltage-like parameters exceed ±10 V.

    Usage:
        - Select the appropriate tab for your task.
        - Enter values or use controls as needed.
        - Communication with the SXM system is handled automatically.
    """

    def __init__(self, dde_client):
        """
        Initialize the main window and all tabs.

        Parameters
        ----------
        dde_client : object
            DDE client for communication with the SXM system.
        """
        super().__init__()
        self.setWindowTitle("NC-AFM Control Suite")
        self.resize(1000, 700)

        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()

        # Create all tabs
        self.params_tab = ParamsTab(dde_client)
        self.step_tab = StepTestTab(dde_client)
        self.scope_tab = ScopeTab()
        self.suggest_tab = SuggestedTab(dde_client, self.params_tab)

        # Link StepTest to Scope and Tabs
        self.step_tab.scope_tab = self.scope_tab
        self.step_tab.tabs_widget = self.tabs
        self.step_tab.scope_tab_index = 1  # Tab order: 0=Parameters, 1=Scope, 2=StepTest, 3=Suggested

        # Add tabs to the widget
        self.tabs.addTab(self.params_tab, "Parameters")
        self.tabs.addTab(self.step_tab, "Step Test")
        self.tabs.addTab(self.scope_tab, "Scope")
        self.tabs.addTab(self.suggest_tab, "Suggested Setup")

        # Connect custom parameters signal from params_tab to step_test_tab
        self.params_tab.custom_params_changed.connect(self.step_tab.set_custom_params)

        layout.addWidget(self.tabs)

        # Footer
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