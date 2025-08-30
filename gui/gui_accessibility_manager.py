# gui_accessibility_manager.py
# Global accessibility manager for your application

import json
import os
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class AccessibilityManager(QtCore.QObject):
    """Global manager for accessibility settings across the app"""

    settings_changed = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.settings_file = os.path.join(os.path.expanduser("~"), 
                                          ".scientific_gui_accessibility.json")

        # Defaults
        self.settings = {
            "font_scale": 1.0,
            "font_family": "default",
            "high_contrast": False,
            "plot_line_width": 2,
            "grid_alpha": 0.3,
            "button_height": "default",
        }

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
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    saved = json.load(f)
                    self.settings.update(saved)
        except Exception as e:
            print(f"Could not load accessibility settings: {e}")

    def save_settings(self):
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Could not save accessibility settings: {e}")

    # ---------------- Setters ----------------
    def set_font_scale(self, scale_name: str):
        if scale_name in self.scale_options:
            self.settings["font_scale"] = self.scale_options[scale_name]
            self.save_settings()
            self.settings_changed.emit(self.settings)
            return True
        return False

    def set_high_contrast(self, enabled: bool):
        self.settings["high_contrast"] = bool(enabled)
        self.save_settings()
        self.settings_changed.emit(self.settings)

    # ---------------- Helpers ----------------
    def get_scaled_font(self, base_size: int = None) -> QtGui.QFont:
        app = QtWidgets.QApplication.instance()
        if base_size is None:
            base_size = app.font().pointSize() or 10
        scaled = max(6, int(base_size * self.settings["font_scale"]))
        font = QtGui.QFont()
        font.setPointSize(scaled)
        return font

    def get_scaled_size(self, base_size: int) -> int:
        return max(1, int(base_size * self.settings["font_scale"]))

    # ---------------- Apply to widgets ----------------
    def apply_to_widget(self, widget: QtWidgets.QWidget):
        # Fonts
        widget.setFont(self.get_scaled_font())
        for child in widget.findChildren(QtWidgets.QWidget):
            if not isinstance(child, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
                child.setFont(self.get_scaled_font())

        # Plots
        for plot_widget in widget.findChildren(pg.PlotWidget):
            self.apply_to_plot(plot_widget)

        # Contrast mode
        if self.settings["high_contrast"]:
            self.apply_high_contrast_style(widget)
        else:
            self.clear_high_contrast_style(widget)

    def apply_to_plot(self, plot_widget: pg.PlotWidget):
        tick_font = QtGui.QFont()
        tick_font.setPointSize(self.get_scaled_size(9))
        plot_widget.getAxis("bottom").setTickFont(tick_font)
        plot_widget.getAxis("left").setTickFont(tick_font)

        grid_alpha = 0.5 if self.settings["high_contrast"] else self.settings["grid_alpha"]
        plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)

    # ---------------- Contrast styles ----------------
    def apply_high_contrast_style(self, widget: QtWidgets.QWidget):
        """Apply HC style and remember old sheet once."""
        style = (
            """
            QWidget { background-color: white; color: black; }
            QPushButton { background-color: #f0f0f0; border: 2px solid #333; padding: 6px; font-weight: bold; }
            QPushButton:pressed { background-color: #e0e0e0; }
            QLabel { color: black; font-weight: bold; }
            QComboBox { background-color: white; border: 2px solid #333; font-weight: bold; }
            QDoubleSpinBox { background-color: white; border: 2px solid #333; font-weight: bold; }
            """
        )
        if widget.property("_a11y_prev_stylesheet") is None:
            widget.setProperty("_a11y_prev_stylesheet", widget.styleSheet() or "")
        widget.setStyleSheet(style)

    def clear_high_contrast_style(self, widget: QtWidgets.QWidget):
        """Restore old sheet, or clear if none stored."""
        prev = widget.property("_a11y_prev_stylesheet")
        if prev is not None:
            widget.setStyleSheet(prev)
            widget.setProperty("_a11y_prev_stylesheet", None)
        else:
            if widget.styleSheet():
                widget.setStyleSheet("")


class AccessibilityToolbar(QtWidgets.QWidget):
    """Toolbar with font scale and contrast toggle"""

    def __init__(self, accessibility_manager: AccessibilityManager, parent=None):
        super().__init__(parent)
        self.accessibility_manager = accessibility_manager
        self.setup_ui()
        self.accessibility_manager.settings_changed.connect(self.update_from_settings)

    def setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        acc_frame = QtWidgets.QFrame()
        acc_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        acc_layout = QtWidgets.QHBoxLayout(acc_frame)

        # Font control
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

        # Contrast toggle
        self.high_contrast_btn = QtWidgets.QPushButton("High Contrast")
        self.high_contrast_btn.setCheckable(True)
        self.high_contrast_btn.setChecked(self.accessibility_manager.settings["high_contrast"])
        self.high_contrast_btn.toggled.connect(self.on_high_contrast_toggled)
        acc_layout.addWidget(self.high_contrast_btn)

        layout.addWidget(QtWidgets.QLabel("♿ Accessibility:"))
        layout.addWidget(acc_frame)
        layout.addStretch()

    def on_font_scale_changed(self, scale_name: str):
        self.accessibility_manager.set_font_scale(scale_name)
        self.apply_to_current_window()

    def on_high_contrast_toggled(self, enabled: bool):
        self.accessibility_manager.set_high_contrast(enabled)
        self.apply_to_current_window()

    def apply_to_current_window(self):
        if self.parent():
            self.accessibility_manager.apply_to_widget(self.parent())

    def update_from_settings(self, settings: dict):
        # Sync combo
        for name, scale in self.accessibility_manager.scale_options.items():
            if abs(scale - settings["font_scale"]) < 0.01:
                self.font_combo.setCurrentText(name)
                break
        # Sync button
        self.high_contrast_btn.setChecked(settings["high_contrast"])


class AccessibleWidget(QtWidgets.QWidget):
    """Mixin widget that auto-applies accessibility"""

    def __init__(self, parent=None):
        super().__init__(parent)
        app = QtWidgets.QApplication.instance()
        if not hasattr(app, "accessibility_manager"):
            app.accessibility_manager = AccessibilityManager()
        self.accessibility_manager = app.accessibility_manager
        self.accessibility_manager.settings_changed.connect(self.on_accessibility_changed)

    def on_accessibility_changed(self, settings: dict):
        self.accessibility_manager.apply_to_widget(self)

    def add_accessibility_toolbar(self, layout: QtWidgets.QLayout):
        toolbar = AccessibilityToolbar(self.accessibility_manager, self)
        layout.addWidget(toolbar)
        return toolbar

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(50, lambda: self.accessibility_manager.apply_to_widget(self))


def make_accessible(widget: QtWidgets.QWidget):
    """Apply accessibility to any widget and connect future updates"""
    app = QtWidgets.QApplication.instance()
    if not hasattr(app, "accessibility_manager"):
        app.accessibility_manager = AccessibilityManager()
    mgr = app.accessibility_manager
    mgr.apply_to_widget(widget)
    mgr.settings_changed.connect(lambda _s: mgr.apply_to_widget(widget))
    return mgr


class AccessibilityShortcuts(QtWidgets.QWidget):
    """Global keyboard shortcuts"""

    def __init__(self, accessibility_manager: AccessibilityManager):
        super().__init__()
        self.accessibility_manager = accessibility_manager
        self.setup_shortcuts()

    def setup_shortcuts(self):
        self.shortcut_increase = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self)
        self.shortcut_increase.activated.connect(self.increase_font)

        self.shortcut_decrease = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+-"), self)
        self.shortcut_decrease.activated.connect(self.decrease_font)

        self.shortcut_reset = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        self.shortcut_reset.activated.connect(self.reset_font)

        self.shortcut_contrast = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+H"), self)
        self.shortcut_contrast.activated.connect(self.toggle_high_contrast)

    def increase_font(self):
        current = self.accessibility_manager.settings["font_scale"]
        scales = sorted(self.accessibility_manager.scale_options.values())
        for s in scales:
            if s > current + 0.01:
                for name, val in self.accessibility_manager.scale_options.items():
                    if abs(val - s) < 0.01:
                        self.accessibility_manager.set_font_scale(name)
                        return

    def decrease_font(self):
        current = self.accessibility_manager.settings["font_scale"]
        scales = sorted(self.accessibility_manager.scale_options.values(), reverse=True)
        for s in scales:
            if s < current - 0.01:
                for name, val in self.accessibility_manager.scale_options.items():
                    if abs(val - s) < 0.01:
                        self.accessibility_manager.set_font_scale(name)
                        return

    def reset_font(self):
        self.accessibility_manager.set_font_scale("Normal")

    def toggle_high_contrast(self):
        cur = self.accessibility_manager.settings["high_contrast"]
        self.accessibility_manager.set_high_contrast(not cur)


def setup_global_accessibility(app: QtWidgets.QApplication) -> AccessibilityManager:
    if not hasattr(app, "accessibility_manager"):
        app.accessibility_manager = AccessibilityManager()
        app.accessibility_shortcuts = AccessibilityShortcuts(app.accessibility_manager)
    return app.accessibility_manager


def add_accessibility_to_existing_widget(widget: QtWidgets.QWidget) -> AccessibilityManager:
    app = QtWidgets.QApplication.instance()
    if not hasattr(app, "accessibility_manager"):
        setup_global_accessibility(app)
    app.accessibility_manager.apply_to_widget(widget)
    app.accessibility_manager.settings_changed.connect(
        lambda _s: app.accessibility_manager.apply_to_widget(widget)
    )
    return app.accessibility_manager
