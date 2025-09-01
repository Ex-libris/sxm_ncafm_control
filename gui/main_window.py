"""
Main application window for NC-AFM control suite with accessibility features.

This module defines the MainWindow class, which hosts a tabbed interface for
interacting with a scanning probe microscope. It integrates multiple functional
tabs for parameter tuning, signal monitoring, test automation, and calibration.

Enhanced with comprehensive accessibility features for users with visual impairments.

Tabs:
    - Parameters: Adjust feedback gains, amplitude reference, and other settings.
    - Step Test: Automatically switch parameters to test system response.
    - Scope: View real-time signals from the microscope.
    - Suggested Setup: View recommended starting values.
    - QPlus Calibration: Sweep amplitude and calibrate delta-topography response.
    - Constant Height: Control Z in constant-height mode.

Safety:
    Displays a footer warning if voltage-like parameters exceed ±10 V.

Accessibility:
    - Global font scaling (70% to 220%)
    - High contrast mode
    - Keyboard shortcuts for quick adjustments
    - Persistent user settings
    - Comprehensive UI and plot scaling
"""

from PyQt5 import QtWidgets, QtCore, QtGui

from sxm_ncafm_control.gui.params_tab import ParamsTab
from sxm_ncafm_control.gui.step_test_tab import StepTestTab
from sxm_ncafm_control.gui.suggested_tab import SuggestedTab
from sxm_ncafm_control.gui.scope_tab import ScopeTab
from sxm_ncafm_control.gui.qplus_calibration_tab import QplusCalibrationTab
from sxm_ncafm_control.gui.z_const_acquisition import ZConstAcquisition
from sxm_ncafm_control.gui.gui_accessibility_manager import (
    AccessibilityManager, 
    AccessibilityToolbar,
    AccessibilityShortcuts
)


class MainWindow(QtWidgets.QWidget):
    """
    Main window containing all NC-AFM control tabs, safety footer, and accessibility features.

    This widget sets up and displays a tabbed interface for different aspects of
    microscope configuration and monitoring. All tabs are connected to a shared
    DDE client and IOCTL driver, which are provided via SXMConnection.

    Enhanced with comprehensive accessibility support including font scaling,
    high contrast mode, and keyboard shortcuts.

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
    accessibility_manager : AccessibilityManager
        Global accessibility settings manager.
    accessibility_toolbar : AccessibilityToolbar
        Toolbar with accessibility controls.
    """

    def __init__(self, conn):
        """
        Initialize the main window and all functional tabs with accessibility support.

        Parameters
        ----------
        conn : SXMConnection
            Centralized connection object providing:
            - conn.dde: DDE client (real or mock)
            - conn.driver: IOCTL driver (or None if unavailable)
        """
        super().__init__()
        self.setWindowTitle("NC-AFM Control Suite")
        self.resize(768,768)

        # Initialize accessibility manager
        self.setup_accessibility()

        # Create main layout
        layout = QtWidgets.QVBoxLayout(self)

        # Add accessibility toolbar at the top
        self.accessibility_toolbar = AccessibilityToolbar(self.accessibility_manager, self)
        layout.addWidget(self.accessibility_toolbar)

        # Add visual separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(separator)

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()

        # Create all tabs, sharing the same connection handles
        self.params_tab = ParamsTab(conn.dde)
        self.step_tab = StepTestTab(conn.dde)
        self.scope_tab = ScopeTab(conn.dde, conn.driver)
        self.suggest_tab = SuggestedTab(conn.dde, self.params_tab)
        self.qplus_tab = QplusCalibrationTab(conn.dde)
        self.topo_hold_tab = ZConstAcquisition(conn.dde, conn.driver)

        # Link StepTest to Scope and Tabs
        self.step_tab.scope_tab = self.scope_tab
        self.step_tab.tabs_widget = self.tabs
        self.step_tab.scope_tab_index = 2  # Tab order: 0=Parameters, 1=Step Test, 2=Scope, ...
        self.scope_tab.set_test_tab_reference(self.step_tab)
        
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

        # Safety footer with enhanced accessibility styling
        self.footer = QtWidgets.QLabel(
            "⚠ Check SXM units & never exceed ±10 V without attenuation. <a href='#'>Details</a>"
        )
        self.footer.setOpenExternalLinks(False)
        self.footer.linkActivated.connect(self.show_safety_details)
        self.footer.setStyleSheet(
            "color:#6b5900; background:#fff7da; border:1px solid #e6d9a2; "
            "border-radius:6px; padding:8px; font-weight: bold;"
        )
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

        # Apply initial accessibility settings
        self.apply_accessibility_to_all_tabs()

    def setup_accessibility(self):
        """Initialize accessibility features"""
        # Create or get global accessibility manager
        app = QtWidgets.QApplication.instance()
        if not hasattr(app, 'accessibility_manager'):
            app.accessibility_manager = AccessibilityManager()
            app.accessibility_shortcuts = AccessibilityShortcuts(app.accessibility_manager)
        
        self.accessibility_manager = app.accessibility_manager
        
        # Connect to accessibility changes
        self.accessibility_manager.settings_changed.connect(self.on_accessibility_changed)
        
        # Setup window-specific keyboard shortcuts
        self.setup_accessibility_shortcuts()
        
        # Create accessibility menu
        self.create_accessibility_menu()

    def setup_accessibility_shortcuts(self):
        """Setup accessibility keyboard shortcuts"""
        # Font size shortcuts
        self.shortcut_increase = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self)
        self.shortcut_increase.activated.connect(self.increase_font_size)
        
        self.shortcut_decrease = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+-"), self)
        self.shortcut_decrease.activated.connect(self.decrease_font_size)
        
        self.shortcut_reset = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        self.shortcut_reset.activated.connect(self.reset_font_size)
                
        # Help shortcut
        self.shortcut_help = QtWidgets.QShortcut(QtGui.QKeySequence("F1"), self)
        self.shortcut_help.activated.connect(self.show_accessibility_help)

    def create_accessibility_menu(self):
        """Create accessibility context menu"""
        # Right-click context menu for accessibility
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_accessibility_context_menu)

    def show_accessibility_context_menu(self, position):
        """Show accessibility options in context menu"""
        menu = QtWidgets.QMenu(self)
        
        # Font size submenu
        font_menu = menu.addMenu("♿ Font Size")
        font_group = QtWidgets.QActionGroup(self)
        
        current_scale = self.accessibility_manager.settings['font_scale']
        for name, scale in self.accessibility_manager.scale_options.items():
            action = QtWidgets.QAction(name, self)
            action.setCheckable(True)
            action.setActionGroup(font_group)
            if abs(scale - current_scale) < 0.01:
                action.setChecked(True)
            action.triggered.connect(lambda checked, n=name: self.set_font_scale(n))
            font_menu.addAction(action)
        
        menu.addSeparator()
        
        # High contrast toggle
        contrast_action = QtWidgets.QAction("High Contrast Mode", self)
        contrast_action.setCheckable(True)
        contrast_action.setChecked(self.accessibility_manager.settings['high_contrast'])
        contrast_action.toggled.connect(self.accessibility_manager.set_high_contrast)
        menu.addAction(contrast_action)
        
        menu.addSeparator()
        
        # Help
        help_action = QtWidgets.QAction("Accessibility Help (F1)", self)
        help_action.triggered.connect(self.show_accessibility_help)
        menu.addAction(help_action)
        
        # Show menu
        menu.exec_(self.mapToGlobal(position))

    def on_accessibility_changed(self, settings):
        """Handle accessibility settings changes"""
        # Apply to this window
        self.accessibility_manager.apply_to_widget(self)
        
        # Apply to all tabs
        self.apply_accessibility_to_all_tabs()

    def apply_accessibility_to_all_tabs(self):
        """Apply accessibility settings to all tabs"""
        tabs_to_update = [
            self.params_tab,
            self.step_tab,
            self.scope_tab,
            self.suggest_tab,
            self.qplus_tab,
            self.topo_hold_tab
        ]
        
        for tab in tabs_to_update:
            if tab is not None:
                self.accessibility_manager.apply_to_widget(tab)

    def set_font_scale(self, scale_name):
        """Set font scale and apply to all components"""
        self.accessibility_manager.set_font_scale(scale_name)

    def increase_font_size(self):
        """Increase font size to next level"""
        current_scale = self.accessibility_manager.settings['font_scale']
        scales = list(self.accessibility_manager.scale_options.values())
        scales.sort()
        
        for scale in scales:
            if scale > current_scale + 0.01:
                for name, s in self.accessibility_manager.scale_options.items():
                    if abs(s - scale) < 0.01:
                        self.set_font_scale(name)
                        self.show_font_change_feedback(name)
                        return
        
        # Already at maximum
        self.show_font_change_feedback("Already at maximum size")

    def decrease_font_size(self):
        """Decrease font size to previous level"""
        current_scale = self.accessibility_manager.settings['font_scale']
        scales = list(self.accessibility_manager.scale_options.values())
        scales.sort(reverse=True)
        
        for scale in scales:
            if scale < current_scale - 0.01:
                for name, s in self.accessibility_manager.scale_options.items():
                    if abs(s - scale) < 0.01:
                        self.set_font_scale(name)
                        self.show_font_change_feedback(name)
                        return
        
        # Already at minimum
        self.show_font_change_feedback("Already at minimum size")

    def reset_font_size(self):
        """Reset font to normal size"""
        self.set_font_scale("Normal")
        self.show_font_change_feedback("Normal")

    def toggle_high_contrast(self):
        """Toggle high contrast mode"""
        current = self.accessibility_manager.settings['high_contrast']
        new_state = not current
        self.accessibility_manager.set_high_contrast(new_state)
        
        mode_text = "ON" if new_state else "OFF"
        self.show_font_change_feedback(f"High contrast: {mode_text}")

    def show_font_change_feedback(self, message):
        # QLabel has no showMessage; just set text and clear later.
        bar = self.statusBar()
        bar.setText(f"Accessibility: {message}")
        # Clear after 2 s
        QtCore.QTimer.singleShot(2000, lambda: bar.setText(""))


    def statusBar(self):
        """Create a simple status bar if it doesn't exist"""
        if not hasattr(self, '_status_bar'):
            self._status_bar = QtWidgets.QLabel("")
            self._status_bar.setStyleSheet(
                "QLabel { color: #666; font-size: 11px; padding: 2px 8px; }"
            )
            self.layout().addWidget(self._status_bar)
        return self._status_bar

    def show_safety_details(self):
        """Show enhanced safety details dialog"""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Safety Details")
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        
        safety_text = """
        <h3>⚠ Safety Information</h3>
        
        <h4>Units and Measurements:</h4>
        <ul>
        <li>Units follow the current SXM GUI (Hz/kHz, V/mV/µV...)</li>
        <li>Always verify unit scaling before making changes</li>
        </ul>
        
        <h4>Voltage Limits:</h4>
        <ul>
        <li><b>Never exceed ±10 V</b> on voltage channels without proper attenuation</li>
        <li>Amplitude Reference and Drive channels are particularly sensitive</li>
        <li>Hardware damage may occur if limits are exceeded</li>
        </ul>
        
        <h4>Step Test Safety:</h4>
        <ul>
        <li>Step Test sends values exactly as entered</li>
        <li>Always verify LOW/HIGH values against GUI units</li>
        <li>Start with small parameter changes</li>
        </ul>
        
        <h4>Emergency Procedures:</h4>
        <ul>
        <li>Use the STOP button immediately if unusual behavior occurs</li>
        <li>Check all parameter values before starting measurements</li>
        <li>Keep hardware documentation readily available</li>
        </ul>
        """
        
        msg.setText(safety_text)
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        # Make dialog accessible
        self.accessibility_manager.apply_to_widget(msg)
        
        msg.exec_()

    def show_accessibility_help(self):
        """Show comprehensive accessibility help dialog"""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Accessibility Help")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        
        help_text = """
        <h3>♿ Accessibility Features</h3>
        
        <h4>Keyboard Shortcuts:</h4>
        <ul>
        <li><b>Ctrl + Plus (+)</b>: Increase font size</li>
        <li><b>Ctrl + Minus (-)</b>: Decrease font size</li>
        <li><b>Ctrl + 0</b>: Reset to normal size</li>
        <li><b>Ctrl + H</b>: Toggle high contrast mode</li>
        <li><b>F1</b>: Show this help dialog</li>
        <li><b>Right-click</b>: Accessibility context menu</li>
        </ul>
        
        <h4>Font Size Options:</h4>
        <ul>
        <li><b>Tiny</b> (70%) - Compact view</li>
        <li><b>Small</b> (85%) - Slightly reduced</li>
        <li><b>Normal</b> (100%) - Default size</li>
        <li><b>Large</b> (125%) - Enhanced readability</li>
        <li><b>Extra Large</b> (150%) - High visibility</li>
        <li><b>Huge</b> (180%) - Very large text</li>
        <li><b>Maximum</b> (220%) - Highest magnification</li>
        </ul>
        
        <h4>Visual Enhancements:</h4>
        <ul>
        <li><b>High Contrast Mode</b>: Enhanced borders and text contrast</li>
        <li><b>Plot Scaling</b>: Axis labels and tick marks scale with text</li>
        <li><b>Consistent Scaling</b>: All tabs and windows scale together</li>
        </ul>
        
        <h4>Persistent Settings:</h4>
        <ul>
        <li>Your accessibility preferences are automatically saved</li>
        <li>Settings persist between application sessions</li>
        <li>Changes apply immediately to all open windows</li>
        </ul>
        
        <h4>Scientific Plot Features:</h4>
        <ul>
        <li>Axis labels and tick marks scale proportionally</li>
        <li>Grid lines become more prominent in high contrast mode</li>
        <li>Plot markers and annotations scale with font size</li>
        </ul>
        
        <p><b>Need more help?</b> Contact your system administrator or refer to the user manual for additional accessibility options.</p>
        """
        
        msg.setText(help_text)
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        # Make dialog accessible
        self.accessibility_manager.apply_to_widget(msg)
        
        msg.exec_()

    def showEvent(self, event):
        """Apply accessibility settings when window is shown"""
        super().showEvent(event)
        # Small delay to ensure all widgets are fully initialized
        QtCore.QTimer.singleShot(100, self.apply_accessibility_to_all_tabs)

    def closeEvent(self, event):
        """Handle window closing"""
        # Accessibility settings are automatically saved by the manager
        super().closeEvent(event)