"""
gui_accessibility_manager.py
============================

Global accessibility manager for the GUI of the SPM microscope control software.

This module provides a centralized system for managing accessibility features
across all Qt widgets and pyqtgraph components. It controls font scaling,
dark mode, and high contrast mode, ensuring consistent readability and usability.

Accessibility preferences are stored in a JSON file in the user home directory
so they persist across sessions. This is especially useful in long-term
scientific projects where different users or setups may require different
visual configurations.

Main features
-------------
- Adjustable font scaling with predefined levels.
- High-contrast and dark mode display options.
- Application of styles to Qt widgets, tables, and pyqtgraph plots.
- Toolbar for interactive accessibility controls.
- Keyboard shortcuts for quick toggling of settings.

Target audience
---------------
Graduate students and researchers working with microscopy GUIs. Focus is on
clarity, usability, and reproducibility, with minimal software abstraction.
"""

import json
import os
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class AccessibilityManager(QtCore.QObject):
    """
    Manage accessibility settings across the application.

    Provides methods to save, load, and apply accessibility preferences,
    including font scaling, high-contrast mode, and dark mode.

    Attributes
    ----------
    settings_changed : QtCore.pyqtSignal(dict)
        Signal emitted when settings are updated.

    settings_file : str
        Path to the JSON file for persistent storage.

    settings : dict
        Current accessibility parameters.

    scale_options : dict
        Mapping of scale names to numeric font scale factors.
    """

    settings_changed = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.settings_file = os.path.join(
            os.path.expanduser("~"), ".scientific_gui_accessibility.json"
        )

        # Default settings
        self.settings = {
            "font_scale": 1.0,
            "font_family": "default",
            "high_contrast": False,
            "dark_mode": False,
            "plot_line_width": 2,
            "grid_alpha": 0.3,
            "button_height": "default",
        }

        # Preset scaling factors for fonts
        self.scale_options = {
            "Tiny": 0.7,
            "Small": 0.85,
            "Normal": 1.0,
            "Large": 1.25,
            "Extra Large": 1.5,
            "Huge": 1.8,
            "Maximum": 2.2,
        }

        self.load_settings()

    # ---------------- Settings I/O ----------------
    def load_settings(self):
        """
        Load accessibility settings from JSON file.

        Keeps defaults if file is missing or corrupted.
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    saved = json.load(f)
                    self.settings.update(saved)
        except Exception as e:
            print(f"Could not load accessibility settings: {e}")

    def save_settings(self):
        """
        Save current accessibility settings to JSON file.
        """
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Could not save accessibility settings: {e}")

    # ---------------- Setters ----------------
    def set_font_scale(self, scale_name: str) -> bool:
        """
        Set font scaling factor.

        Parameters
        ----------
        scale_name : str
            One of the predefined scale options.

        Returns
        -------
        bool
            True if applied successfully, False if invalid name.
        """
        if scale_name in self.scale_options:
            self.settings["font_scale"] = self.scale_options[scale_name]
            self.save_settings()
            self.settings_changed.emit(self.settings)
            return True
        return False

    def set_high_contrast(self, enabled: bool):
        """
        Enable or disable high contrast mode.

        Parameters
        ----------
        enabled : bool
            Whether high contrast mode should be enabled.
        """
        self.settings["high_contrast"] = bool(enabled)
        self.save_settings()
        self.settings_changed.emit(self.settings)

    def set_dark_mode(self, enabled: bool):
        """
        Enable or disable dark mode.

        Parameters
        ----------
        enabled : bool
            Whether dark mode should be enabled.
        """
        self.settings["dark_mode"] = bool(enabled)
        self.save_settings()
        self.settings_changed.emit(self.settings)

    # ---------------- Helpers ----------------
    def get_scaled_font(self, base_size: int = None) -> QtGui.QFont:
        """
        Return a QFont object scaled by current font factor.

        Parameters
        ----------
        base_size : int, optional
            Base font size. Uses application default if None.

        Returns
        -------
        QtGui.QFont
            Scaled font.
        """
        app = QtWidgets.QApplication.instance()
        if base_size is None:
            base_size = app.font().pointSize() or 10
        scaled = max(6, int(base_size * self.settings["font_scale"]))
        font = QtGui.QFont()
        font.setPointSize(scaled)
        return font

    def get_scaled_size(self, base_size: int) -> int:
        """
        Return a scaled integer size (e.g., for labels).

        Parameters
        ----------
        base_size : int
            Reference size.

        Returns
        -------
        int
            Scaled size, minimum of 1.
        """
        return max(1, int(base_size * self.settings["font_scale"]))

    # ---------------- Apply to widgets ----------------
    def apply_to_widget(self, widget: QtWidgets.QWidget):
        """
        Apply current accessibility settings to a widget and its children.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            Target widget.
        """
        # Update fonts
        widget.setFont(self.get_scaled_font())
        for child in widget.findChildren(QtWidgets.QWidget):
            if not isinstance(child, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
                child.setFont(self.get_scaled_font())

        # Update plots
        for plot_widget in widget.findChildren(pg.PlotWidget):
            self.apply_to_plot(plot_widget)

        for glw in widget.findChildren(pg.GraphicsLayoutWidget):
            for item in glw.ci.items.keys():
                if isinstance(item, pg.PlotItem):
                    self.apply_to_plotitem(item)
            glw.setBackground("k" if self.settings.get("dark_mode", False) else "w")

        # Update tables
        for table in widget.findChildren(QtWidgets.QTableWidget):
            self.apply_table_colors(table)

        # Apply global styles
        if self.settings.get("dark_mode", False):
            self.apply_dark_mode_style(widget)
        elif self.settings.get("high_contrast", False):
            self.apply_high_contrast_style(widget)
        else:
            self.clear_styles(widget)

    def apply_to_plot(self, plot_widget: pg.PlotWidget):
        """
        Apply accessibility settings to a pyqtgraph PlotWidget.

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            Target plot widget.
        """
        tick_font = QtGui.QFont()
        tick_font.setPointSize(self.get_scaled_size(9))
        plot_widget.getAxis("bottom").setTickFont(tick_font)
        plot_widget.getAxis("left").setTickFont(tick_font)

        grid_alpha = (
            0.5 if self.settings.get("high_contrast") else self.settings["grid_alpha"]
        )

        if self.settings.get("dark_mode", False):
            plot_widget.setBackground("k")
            plot_widget.getAxis("left").setTextPen("w")
            plot_widget.getAxis("left").setPen(pg.mkPen("w"))
            plot_widget.getAxis("bottom").setTextPen("w")
            plot_widget.getAxis("bottom").setPen(pg.mkPen("w"))
            plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)
        else:
            plot_widget.setBackground("w")
            plot_widget.getAxis("left").setTextPen("k")
            plot_widget.getAxis("left").setPen(pg.mkPen("k"))
            plot_widget.getAxis("bottom").setTextPen("k")
            plot_widget.getAxis("bottom").setPen(pg.mkPen("k"))
            plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)

    def apply_to_plotitem(self, plot_item: pg.PlotItem):
        """
        Apply accessibility settings to a pyqtgraph PlotItem.

        Parameters
        ----------
        plot_item : pg.PlotItem
            Target plot item.
        """
        tick_font = QtGui.QFont()
        tick_font.setPointSize(self.get_scaled_size(9))
        plot_item.getAxis("bottom").setTickFont(tick_font)
        plot_item.getAxis("left").setTickFont(tick_font)

        grid_alpha = (
            0.5 if self.settings.get("high_contrast") else self.settings["grid_alpha"]
        )

        if self.settings.get("dark_mode", False):
            plot_item.getViewBox().setBackgroundColor("k")
            plot_item.getAxis("left").setTextPen("w")
            plot_item.getAxis("left").setPen(pg.mkPen("w"))
            plot_item.getAxis("bottom").setTextPen("w")
            plot_item.getAxis("bottom").setPen(pg.mkPen("w"))
            plot_item.showGrid(x=True, y=True, alpha=grid_alpha)
        else:
            plot_item.getViewBox().setBackgroundColor("w")
            plot_item.getAxis("left").setTextPen("k")
            plot_item.getAxis("left").setPen(pg.mkPen("k"))
            plot_item.getAxis("bottom").setTextPen("k")
            plot_item.getAxis("bottom").setPen(pg.mkPen("k"))
            plot_item.showGrid(x=True, y=True, alpha=grid_alpha)

    # ---------------- Table helpers ----------------
    def apply_table_colors(self, table: QtWidgets.QTableWidget):
        """
        Recolor 'Current' and 'New Value' columns according to mode.

        Parameters
        ----------
        table : QtWidgets.QTableWidget
            Table to update.
        """
        if table.columnCount() == 0:
            return

        headers = [
            table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else ""
            for c in range(table.columnCount())
        ]
        try:
            col_current = headers.index("Current")
        except ValueError:
            col_current = None
        try:
            col_newvalue = headers.index("New Value")
        except ValueError:
            col_newvalue = None

        if self.settings.get("dark_mode", False):
            current_bg = QtGui.QColor("#2a2a2a")
            new_bg = QtGui.QColor("#665500")
            fg = QtGui.QColor("#f0f0f0")
        else:
            current_bg = QtGui.QColor("#dcdcdc")
            new_bg = QtGui.QColor("#ffffcc")
            fg = QtGui.QColor("#000000")

        for r in range(table.rowCount()):
            if col_current is not None:
                item = table.item(r, col_current)
                if item:
                    item.setBackground(current_bg)
                    item.setForeground(fg)
            if col_newvalue is not None:
                item = table.item(r, col_newvalue)
                if item:
                    item.setBackground(new_bg)
                    item.setForeground(fg)

    # ---------------- Style helpers ----------------
    def apply_high_contrast_style(self, widget: QtWidgets.QWidget):
        """
        Apply a high contrast stylesheet to a widget.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            Target widget.
        """
        style = """
            QWidget { background-color: white; color: black; }
            QPushButton { background-color: #f0f0f0; border: 2px solid #333; padding: 6px; font-weight: bold; }
            QPushButton:pressed { background-color: #e0e0e0; }
            QLabel { color: black; font-weight: bold; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: white; border: 2px solid #333; font-weight: bold;
            }
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #f0f0f0; color: black; padding: 6px; }
            QTabBar::tab:selected { background: #ddd; }
            QTabBar::tab:hover { background: #eee; }
            QTableWidget, QTableView {
                background-color: white; alternate-background-color: #f9f9f9;
                color: black; gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #f0f0f0; color: black; border: 1px solid #333;
            }
        """
        if widget.property("_a11y_prev_stylesheet") is None:
            widget.setProperty("_a11y_prev_stylesheet", widget.styleSheet() or "")
        widget.setStyleSheet(style)

    def apply_dark_mode_style(self, widget: QtWidgets.QWidget):
        """
        Apply a dark mode stylesheet to a widget.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            Target widget.
        """
        style = """
            QWidget { background-color: #121212; color: #f0f0f0; }
            QPushButton { background-color: #333; border: 1px solid #555; padding: 6px; }
            QPushButton:pressed { background-color: #444; }
            QLabel { color: #f0f0f0; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #222; color: #f0f0f0; border: 1px solid #555;
            }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #f0f0f0; padding: 6px; }
            QTabBar::tab:selected { background: #555; }
            QTabBar::tab:hover { background: #444; }
            QTableWidget, QTableView {
                background-color: #121212; alternate-background-color: #1e1e1e;
                color: #f0f0f0; gridline-color: #444;
                selection-background-color: #333366; selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #222; color: #f0f0f0; border: 1px solid #444;
            }
            QGroupBox {
                border: 1px solid #444; margin-top: 6px;
            }
            QGroupBox:title {
                subcontrol-origin: margin; left: 7px; padding: 0 3px 0 3px;
                color: #f0f0f0;
            }
            QGraphicsView {
                background-color: #121212; border: 1px solid #444;
            }
        """
        if widget.property("_a11y_prev_stylesheet") is None:
            widget.setProperty("_a11y_prev_stylesheet", widget.styleSheet() or "")
        widget.setStyleSheet(style)

    def clear_styles(self, widget: QtWidgets.QWidget):
        """
        Restore original widget stylesheet.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            Target widget.
        """
        prev = widget.property("_a11y_prev_stylesheet")
        if prev is not None:
            widget.setStyleSheet(prev)
            widget.setProperty("_a11y_prev_stylesheet", None)
        else:
            if widget.styleSheet():
                widget.setStyleSheet("")


class AccessibilityToolbar(QtWidgets.QWidget):
    """
    Toolbar with controls for accessibility settings.

    Provides combo box for font scaling and toggle buttons for
    high contrast and dark mode. Updates settings globally.

    Parameters
    ----------
    accessibility_manager : AccessibilityManager
        Reference to global manager handling the settings.

    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, accessibility_manager: AccessibilityManager, parent=None):
        super().__init__(parent)
        self.accessibility_manager = accessibility_manager
        self.setup_ui()
        self.accessibility_manager.settings_changed.connect(self.update_from_settings)

    def setup_ui(self):
        """Initialize toolbar layout and controls."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        acc_frame = QtWidgets.QFrame()
        acc_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        acc_layout = QtWidgets.QHBoxLayout(acc_frame)

        # Font scale selector
        acc_layout.addWidget(QtWidgets.QLabel("🔍 Font:"))
        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.addItems(list(self.accessibility_manager.scale_options.keys()))
        current_scale = self.accessibility_manager.settings["font_scale"]
        for name, scale in self.accessibility_manager.scale_options.items():
            if abs(scale - current_scale) < 0.01:
                self.font_combo.setCurrentText(name)
                break
        self.font_combo.currentTextChanged.connect(self.on_font_scale_changed)
        acc_layout.addWidget(self.font_combo)

        # High contrast toggle
        self.high_contrast_btn = QtWidgets.QPushButton("Accessibility Mode")
        self.high_contrast_btn.setCheckable(True)
        self.high_contrast_btn.setChecked(
            self.accessibility_manager.settings["high_contrast"]
        )
        self.high_contrast_btn.toggled.connect(self.on_high_contrast_toggled)
        acc_layout.addWidget(self.high_contrast_btn)

        # Dark mode toggle
        self.dark_mode_btn = QtWidgets.QPushButton("Dark Mode")
        self.dark_mode_btn.setCheckable(True)
        self.dark_mode_btn.setChecked(self.accessibility_manager.settings["dark_mode"])
        self.dark_mode_btn.toggled.connect(self.on_dark_mode_toggled)
        acc_layout.addWidget(self.dark_mode_btn)

        layout.addWidget(QtWidgets.QLabel("♿ Accessibility:"))
        layout.addWidget(acc_frame)
        layout.addStretch()

    def on_font_scale_changed(self, scale_name: str):
        """Handle change in font scale selection."""
        self.accessibility_manager.set_font_scale(scale_name)
        self.apply_to_current_window()

    def on_high_contrast_toggled(self, enabled: bool):
        """Handle toggle of high contrast button."""
        self.accessibility_manager.set_high_contrast(enabled)
        self.apply_to_current_window()

    def on_dark_mode_toggled(self, enabled: bool):
        """Handle toggle of dark mode button."""
        self.accessibility_manager.set_dark_mode(enabled)
        self.apply_to_current_window()

    def apply_to_current_window(self):
        """Reapply accessibility settings to parent window."""
        if self.parent():
            self.accessibility_manager.apply_to_widget(self.parent())

    def update_from_settings(self, settings: dict):
        """Update toolbar state from current settings."""
        for name, scale in self.accessibility_manager.scale_options.items():
            if abs(scale - settings["font_scale"]) < 0.01:
                self.font_combo.setCurrentText(name)
                break
        self.high_contrast_btn.setChecked(settings["high_contrast"])
        self.dark_mode_btn.setChecked(settings["dark_mode"])


class AccessibleWidget(QtWidgets.QWidget):
    """
    Base widget class with automatic accessibility support.

    Connects to the global AccessibilityManager and reapplies settings
    on changes or when the widget is shown.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        app = QtWidgets.QApplication.instance()
        if not hasattr(app, "accessibility_manager"):
            app.accessibility_manager = AccessibilityManager()
        self.accessibility_manager = app.accessibility_manager
        self.accessibility_manager.settings_changed.connect(self.on_accessibility_changed)

    def on_accessibility_changed(self, settings: dict):
        """Reapply accessibility settings when updated globally."""
        self.accessibility_manager.apply_to_widget(self)

    def add_accessibility_toolbar(self, layout: QtWidgets.QLayout):
        """
        Add toolbar for accessibility controls to a given layout.

        Parameters
        ----------
        layout : QtWidgets.QLayout
            Target layout to add toolbar to.

        Returns
        -------
        AccessibilityToolbar
            The created toolbar instance.
        """
        toolbar = AccessibilityToolbar(self.accessibility_manager, self)
        layout.addWidget(toolbar)
        return toolbar

    def showEvent(self, event):
        """Apply accessibility settings after widget is shown."""
        super().showEvent(event)
        QtCore.QTimer.singleShot(
            50, lambda: self.accessibility_manager.apply_to_widget(self)
        )


def make_accessible(widget: QtWidgets.QWidget) -> AccessibilityManager:
    """
    Apply accessibility settings to an existing widget.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        Target widget.

    Returns
    -------
    AccessibilityManager
        Reference to the global manager.
    """
    app = QtWidgets.QApplication.instance()
    if not hasattr(app, "accessibility_manager"):
        app.accessibility_manager = AccessibilityManager()
    mgr = app.accessibility_manager
    mgr.apply_to_widget(widget)
    mgr.settings_changed.connect(lambda _s: mgr.apply_to_widget(widget))
    return mgr


class AccessibilityShortcuts(QtWidgets.QWidget):
    """
    Global keyboard shortcuts for accessibility control.

    Provides key sequences for increasing/decreasing font,
    resetting font scale, and toggling high contrast or dark mode.
    """

    def __init__(self, accessibility_manager: AccessibilityManager):
        super().__init__()
        self.accessibility_manager = accessibility_manager
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Initialize all shortcuts and connect them to actions."""
        self.shortcut_increase = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl++"), self
        )
        self.shortcut_increase.activated.connect(self.increase_font)

        self.shortcut_decrease = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+-"), self
        )
        self.shortcut_decrease.activated.connect(self.decrease_font)

        self.shortcut_reset = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        self.shortcut_reset.activated.connect(self.reset_font)

        self.shortcut_contrast = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+H"), self)
        self.shortcut_contrast.activated.connect(self.toggle_high_contrast)

        self.shortcut_dark = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+D"), self)
        self.shortcut_dark.activated.connect(self.toggle_dark_mode)

    def increase_font(self):
        """Increase font size to the next larger scale."""
        current = self.accessibility_manager.settings["font_scale"]
        scales = sorted(self.accessibility_manager.scale_options.values())
        for s in scales:
            if s > current + 0.01:
                for name, val in self.accessibility_manager.scale_options.items():
                    if abs(val - s) < 0.01:
                        self.accessibility_manager.set_font_scale(name)
                        return

    def decrease_font(self):
        """Decrease font size to the next smaller scale."""
        current = self.accessibility_manager.settings["font_scale"]
        scales = sorted(
            self.accessibility_manager.scale_options.values(), reverse=True
        )
        for s in scales:
            if s < current - 0.01:
                for name, val in self.accessibility_manager.scale_options.items():
                    if abs(val - s) < 0.01:
                        self.accessibility_manager.set_font_scale(name)
                        return

    def reset_font(self):
        """Reset font scale to 'Normal'."""
        self.accessibility_manager.set_font_scale("Normal")

    def toggle_high_contrast(self):
        """Toggle high contrast mode."""
        cur = self.accessibility_manager.settings["high_contrast"]
        self.accessibility_manager.set_high_contrast(not cur)

    def toggle_dark_mode(self):
        """Toggle dark mode."""
        cur = self.accessibility_manager.settings["dark_mode"]
        self.accessibility_manager.set_dark_mode(not cur)


def setup_global_accessibility(app: QtWidgets.QApplication) -> AccessibilityManager:
    """
    Initialize global accessibility manager and shortcuts for an application.

    Parameters
    ----------
    app : QtWidgets.QApplication
        Application instance.

    Returns
    -------
    AccessibilityManager
        Global accessibility manager.
    """
    if not hasattr(app, "accessibility_manager"):
        app.accessibility_manager = AccessibilityManager()
        app.accessibility_shortcuts = AccessibilityShortcuts(app.accessibility_manager)
    return app.accessibility_manager


def add_accessibility_to_existing_widget(widget: QtWidgets.QWidget) -> AccessibilityManager:
    """
    Add accessibility support to an already existing widget.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        Target widget.

    Returns
    -------
    AccessibilityManager
        Global manager reference.
    """
    app = QtWidgets.QApplication.instance()
    if not hasattr(app, "accessibility_manager"):
        setup_global_accessibility(app)
    app.accessibility_manager.apply_to_widget(widget)
    app.accessibility_manager.settings_changed.connect(
        lambda _s: app.accessibility_manager.apply_to_widget(widget)
    )
    return app.accessibility_manager
