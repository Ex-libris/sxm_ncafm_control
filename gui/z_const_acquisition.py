# z_const_acquisition.py
# Full script with elegant change overlay and color-coded feedback

import sys
import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Uses your mapping. CHANNELS[name] -> (idx, short, unit, scale) or similar.
from ..device_driver import CHANNELS


class FlexibleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)  # Only emit valueChanged when editing is finished
        self.custom_step_enabled = True
        
    def stepBy(self, steps):
        if not self.custom_step_enabled:
            super().stepBy(steps)
            return
            
        # Get current cursor position in the line edit
        line_edit = self.lineEdit()
        cursor_pos = line_edit.cursorPosition()
        text = line_edit.text()
        
        # Remove suffix for position calculation
        suffix = self.suffix()
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
        
        # Find decimal point position
        decimal_pos = text.find('.')
        
        # Determine step size based on cursor position
        step_size = self.determine_step_size(cursor_pos, text, decimal_pos)
        
        # Apply the step with strict boundary checking
        current_value = self.value()
        new_value = current_value + (steps * step_size)
        
        # CRITICAL: Prevent wraparound by strictly enforcing boundaries
        if new_value > self.maximum():
            new_value = self.maximum()
            print(f"Warning: Clamped to maximum value {self.maximum()}")
        elif new_value < self.minimum():
            new_value = self.minimum()
            print(f"Warning: Clamped to minimum value {self.minimum()}")
        
        # Only set value if it actually changed to prevent unnecessary signals
        if abs(new_value - current_value) > 1e-10:  # Account for floating point precision
            self.setValue(new_value)
        
        # Restore cursor position (approximately)
        QtCore.QTimer.singleShot(0, lambda: line_edit.setCursorPosition(cursor_pos))
    
    def determine_step_size(self, cursor_pos, text, decimal_pos):
        """Determine step size based on cursor position"""
        if decimal_pos == -1:  # No decimal point
            # Count digits from right to cursor position
            digits_from_right = len(text) - cursor_pos
            if digits_from_right <= 0:
                return 1.0
            return 10 ** (digits_from_right - 1)
        else:
            if cursor_pos <= decimal_pos:
                # Before decimal point
                digits_from_right = decimal_pos - cursor_pos
                return 10 ** digits_from_right if digits_from_right > 0 else 1.0
            else:
                # After decimal point
                decimal_places = cursor_pos - decimal_pos - 1
                return 10 ** (-decimal_places - 1)
    
    def wheelEvent(self, event):
        """Enhanced wheel event with position-aware stepping"""
        if not self.custom_step_enabled:
            super().wheelEvent(event)
            return
            
        # Get the number of steps (usually +1 or -1)
        steps = event.angleDelta().y() // 120
        
        # Check for modifier keys for different behaviors
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        
        if modifiers == QtCore.Qt.ControlModifier:
            # Ctrl+wheel: use the default single step
            self.custom_step_enabled = False
            super().wheelEvent(event)
            self.custom_step_enabled = True
        elif modifiers == QtCore.Qt.ShiftModifier:
            # Shift+wheel: larger steps (10x current position step)
            self.stepBy(steps * 10)
        else:
            # Normal wheel: position-aware stepping
            self.stepBy(steps)
            
        event.accept()
    
    def keyPressEvent(self, event):
        """Enhanced key press event for better navigation"""
        if event.key() == QtCore.Qt.Key_Up:
            self.stepBy(1)
            event.accept()
        elif event.key() == QtCore.Qt.Key_Down:
            self.stepBy(-1)
            event.accept()
        else:
            super().keyPressEvent(event)


class ChangeOverlay(QtWidgets.QLabel):
    """Elegant text overlay for displaying z-position changes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                border: 2px solid rgba(100, 100, 100, 150);
                border-radius: 15px;
                padding: 8px 16px;
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
        """)
        
        # Animation for fade in/out
        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        
        # Timer for auto-hide
        self.hide_timer = QtCore.QTimer()
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.fade_out)
        
        self.hide()
    
    def show_change(self, change_value, current_value):
        """Display the change value with appropriate color coding"""
        if abs(change_value) < 0.001:  # Threshold for significant change
            return
            
        # Format the change value
        if abs(change_value) >= 1.0:
            change_text = f"{change_value:+.2f}"
        elif abs(change_value) >= 0.01:
            change_text = f"{change_value:+.3f}"
        else:
            change_text = f"{change_value:+.4f}"
        
        # Create the display text
        display_text = f"Δz: {change_text} nm\nz: {current_value:.3f} nm"
        
        # Color coding
        if change_value > 0:
            color = "#2E8B57"  # Sea green for positive changes
            bg_color = "rgba(200, 255, 200, 220)"
            border_color = "rgba(46, 139, 87, 180)"
        else:
            color = "#DC143C"  # Crimson for negative changes
            bg_color = "rgba(255, 200, 200, 220)"
            border_color = "rgba(220, 20, 60, 180)"
        
        # Background with no color, only text colored
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(255, 255, 255, 240);
                border: 1px solid rgba(150, 150, 150, 120);
                border-radius: 12px;
                padding: 6px 12px;
                font-size: 14px;
                font-weight: bold;
                color: {color};
            }}
        """)
        
        self.setText(display_text)
        self.adjustSize()
        
        # Position overlay in bottom-left corner of parent
        if self.parent():
            parent_rect = self.parent().rect()
            overlay_rect = self.rect()
            x = 20
            y = parent_rect.height() - overlay_rect.height() - 20
            self.move(x, y)
        
        # Show with fade in
        self.fade_in()
        
        # Auto-hide after 4 seconds (increased duration)
        self.hide_timer.stop()
        self.hide_timer.start(4000)
    
    def fade_in(self):
        """Fade in animation"""
        self.show()
        self.fade_animation.stop()
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()
    
    def fade_out(self):
        """Fade out animation"""
        self.fade_animation.stop()
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.hide)
        self.fade_animation.start()


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde=None, driver=None):
        super().__init__()
        self.setWindowTitle("Z-Const Acquisition")
        self.resize(1000, 700)

        self.dde = dde
        self.driver = driver

        # state
        self.live_mode = True
        self.feedback_enabled = True
        self.last_z = 0.0
        self.previous_z = 0.0  # For change calculation
        self.window_seconds = 10
        self.change_threshold = 0.001  # Minimum change to display overlay

        # Change event markers
        self.change_markers = []  # Store reference lines for changes
        
        # Font scaling for accessibility
        self.font_scale = 1.0
        self.base_font_size = QtWidgets.QApplication.font().pointSize() or 10

        # topography trace
        self.t_stamps = []
        self.z_hist = []

        # aux trace
        default_aux = "df" if "df" in CHANNELS else list(CHANNELS.keys())[0]
        self.aux_channel = default_aux
        self.aux_unit = CHANNELS[self.aux_channel][2]
        self.aux_hist = []

        # ui
        main = QtWidgets.QVBoxLayout(self)

        # --- top controls ---
        ctrl = QtWidgets.QHBoxLayout()

        self.btn_toggle = QtWidgets.QPushButton("Disable Feedback")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self.toggle_feedback)
        ctrl.addWidget(self.btn_toggle)

        ctrl.addWidget(QtWidgets.QLabel("Z Position:"))
        self.z_spin = FlexibleDoubleSpinBox()
        self.z_spin.setRange(-250.0, 250.0)
        self.z_spin.setDecimals(3)            
        self.z_spin.setSuffix(" nm")
        self.z_spin.setEnabled(False)  # enabled only if manual
        self.z_spin.valueChanged.connect(self.manual_update)
        ctrl.addWidget(self.z_spin)

        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Window (s):"))
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(["2", "5", "10", "30", "60", "120"])
        self.combo_window.setCurrentText("10")
        self.combo_window.currentTextChanged.connect(
            lambda v: setattr(self, "window_seconds", int(v))
        )
        ctrl.addWidget(self.combo_window)

        self.btn_clear = QtWidgets.QPushButton("Clear Trace")
        self.btn_clear.clicked.connect(self.clear_trace)
        ctrl.addWidget(self.btn_clear)

        self.btn_clear_markers = QtWidgets.QPushButton("Clear Markers")
        self.btn_clear_markers.clicked.connect(self.clear_markers_only)
        self.btn_clear_markers.setToolTip("Clear change markers and overlays without affecting trace data")
        ctrl.addWidget(self.btn_clear_markers)

        ctrl.addSpacing(20)
        
        # Font scaling controls for accessibility
        ctrl.addWidget(QtWidgets.QLabel("Font Size:"))
        self.font_scale_combo = QtWidgets.QComboBox()
        self.font_scale_combo.addItems(["Small", "Normal", "Large", "Extra Large"])
        self.font_scale_combo.setCurrentText("Normal")
        self.font_scale_combo.currentTextChanged.connect(self.change_font_scale)
        self.font_scale_combo.setToolTip("Adjust font size for better accessibility")
        ctrl.addWidget(self.font_scale_combo)

        ctrl.addStretch(1)
        main.addLayout(ctrl)

        # help row
        help_row = QtWidgets.QHBoxLayout()
        help_lbl = QtWidgets.QLabel(
            "Wheel = pos-aware step | Shift+wheel = 10× | Ctrl+wheel = fixed step | ↑↓ = pos-aware step"
        )
        help_lbl.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        help_lbl.setWordWrap(True)
        help_row.addWidget(help_lbl)
        main.addLayout(help_row)

        # status row
        status = QtWidgets.QHBoxLayout()
        self.status_lbl = QtWidgets.QLabel("Status: Live mode, Feedback ON")
        self.status_lbl.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status.addWidget(self.status_lbl)
        status.addStretch(1)
        main.addLayout(status)

        # --- topography plot with overlay ---
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_topo = pg.PlotWidget()
        self.plot_topo.setBackground("w")
        self.curve_topo = self.plot_topo.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot_topo.setLabel("bottom", "Time", units="s")
        self.plot_topo.setLabel("left", "Z Position", units="nm")
        self.plot_topo.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.plot_topo)
        
        # Create change overlay
        self.change_overlay = ChangeOverlay(self.plot_topo)
        
        main.addWidget(plot_container)

        # --- aux controls + plot ---
        aux_ctrl = QtWidgets.QHBoxLayout()
        aux_ctrl.addWidget(QtWidgets.QLabel("Aux channel:"))

        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.addItems(sorted(CHANNELS.keys()))
        i_def = self.combo_channel.findText(self.aux_channel)
        if i_def >= 0:
            self.combo_channel.setCurrentIndex(i_def)
        self.combo_channel.currentTextChanged.connect(self.on_channel_changed)
        aux_ctrl.addWidget(self.combo_channel)

        self.lbl_aux_unit = QtWidgets.QLabel(f"[{self.aux_unit}]")
        self.lbl_aux_unit.setStyleSheet("QLabel { color: #444; }")
        aux_ctrl.addWidget(self.lbl_aux_unit)
        aux_ctrl.addStretch(1)
        main.addLayout(aux_ctrl)

        self.plot_aux = pg.PlotWidget()
        self.plot_aux.setBackground("w")
        self.curve_aux = self.plot_aux.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot_aux.setLabel("bottom", "Time", units="s")
        self.plot_aux.setLabel("left", self.aux_channel, units=self.aux_unit)
        self.plot_aux.showGrid(x=True, y=True, alpha=0.3)
        main.addWidget(self.plot_aux)

        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

        # initial sync and font setup
        self.initialize_z_position()
        self.apply_font_scaling()  # Apply initial font scaling

    # ---------- actions ----------
    def initialize_z_position(self):
        try:
            if self.driver:
                z = self.driver.read_scaled("Topo")
                self.last_z = float(z)
                self.previous_z = self.last_z
                self.z_spin.setValue(self.last_z)
        except Exception as e:
            print(f"Init error: {e}")

    def toggle_feedback(self, checked: bool):
        # checked means button is pressed -> we *disable* feedback
        self.feedback_enabled = not checked
        if self.feedback_enabled:
            self.btn_toggle.setText("Disable Feedback")
            self.status_lbl.setText("Status: Live mode, Feedback ON")
            self.status_lbl.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.z_spin.setEnabled(False)
        else:
            self.btn_toggle.setText("Enable Feedback")
            self.status_lbl.setText("Status: Manual mode, Feedback OFF")
            self.status_lbl.setStyleSheet("QLabel { color: #c77; font-weight: bold; }")
            self.z_spin.setEnabled(True)

    def manual_update(self, val: float):
        if self.feedback_enabled:
            return
        
        # Calculate change for manual updates
        change = val - self.last_z
        self.previous_z = self.last_z
        self.last_z = float(val)
        
        # Show overlay for manual changes
        if abs(change) >= self.change_threshold:
            self.change_overlay.show_change(change, self.last_z)
            # Add marker at current time if we have timestamp data
            if self.t_stamps:
                current_time = self.t_stamps[-1] if self.t_stamps else 0
                self.add_change_marker(current_time, self.last_z, change)

    def change_font_scale(self, scale_text):
        """Change font scale for accessibility"""
        scale_map = {
            "Small": 0.8,
            "Normal": 1.0,
            "Large": 1.3,
            "Extra Large": 1.6
        }
        
        self.font_scale = scale_map.get(scale_text, 1.0)
        self.apply_font_scaling()
        
        print(f"Font scale changed to: {scale_text} ({self.font_scale}x)")
    
    def apply_font_scaling(self):
        """Apply font scaling to all UI elements"""
        scaled_size = int(self.base_font_size * self.font_scale)
        font = QtGui.QFont()
        font.setPointSize(scaled_size)
        
        # Apply to main widget and all children
        self.setFont(font)
        for child in self.findChildren(QtWidgets.QWidget):
            if not isinstance(child, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
                child.setFont(font)
        
        # Scale plot labels and axes
        self.scale_plot_fonts()
        
        # Update overlay font size
        self.update_overlay_font_size()
        
        # Update change marker text sizes
        self.update_marker_text_sizes()
    
    def scale_plot_fonts(self):
        """Scale fonts for plot elements"""
        label_font_size = max(8, int(10 * self.font_scale))
        tick_font_size = max(7, int(9 * self.font_scale))
        
        # Style for plot labels and ticks
        label_style = {'font-size': f'{label_font_size}pt', 'color': 'black'}
        tick_style = {'font-size': f'{tick_font_size}pt', 'color': 'black'}
        
        # Update topography plot
        self.plot_topo.setLabel('bottom', 'Time', units='s', **label_style)
        self.plot_topo.setLabel('left', 'Z Position', units='nm', **label_style)
        self.plot_topo.getAxis('bottom').setTickFont(QtGui.QFont('', tick_font_size))
        self.plot_topo.getAxis('left').setTickFont(QtGui.QFont('', tick_font_size))
        
        # Update aux plot
        self.plot_aux.setLabel('bottom', 'Time', units='s', **label_style)
        self.plot_aux.setLabel('left', self.aux_channel, units=self.aux_unit, **label_style)
        self.plot_aux.getAxis('bottom').setTickFont(QtGui.QFont('', tick_font_size))
        self.plot_aux.getAxis('left').setTickFont(QtGui.QFont('', tick_font_size))
    
    def update_overlay_font_size(self):
        """Update the change overlay font size"""
        overlay_font_size = max(12, int(14 * self.font_scale))
        current_style = self.change_overlay.styleSheet()
        
        # Update font-size in the stylesheet
        import re
        new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {overlay_font_size}px', current_style)
        self.change_overlay.setStyleSheet(new_style)
    
    def update_marker_text_sizes(self):
        """Update existing marker text sizes"""
        marker_font_size = max(8, int(10 * self.font_scale))
        
        for line, text_item, timestamp in self.change_markers:
            if hasattr(text_item, 'setFont'):
                font = QtGui.QFont()
                font.setPointSize(marker_font_size)
                text_item.setFont(font)

    def add_change_marker(self, time_stamp, z_value, change_value):
        """Add a vertical line marker at the change event"""
        if abs(change_value) < self.change_threshold:
            return
        
        # Color based on change direction
        if change_value > 0:
            color = (46, 139, 87, 150)  # Sea green with transparency
        else:
            color = (220, 20, 60, 150)  # Crimson with transparency
        
        # Create vertical line
        line = pg.InfiniteLine(
            pos=time_stamp,
            angle=90,
            pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine)
        )
        
        # Create text label with the change value and proper font size
        marker_font_size = max(8, int(10 * self.font_scale))
        if abs(change_value) >= 1.0:
            change_text = f"{change_value:+.2f}"
        elif abs(change_value) >= 0.01:
            change_text = f"{change_value:+.3f}"
        else:
            change_text = f"{change_value:+.4f}"
        
        # Position text label slightly offset from line
        text_item = pg.TextItem(
            text=change_text,
            color=color[:3],  # RGB only for text
            anchor=(0.5, 1.1)  # Center horizontally, above the line
        )
        
        # Set font size for the text item
        font = QtGui.QFont()
        font.setPointSize(marker_font_size)
        text_item.setFont(font)
        
        text_item.setPos(time_stamp, z_value)
        
        # Add to plot
        self.plot_topo.addItem(line)
        self.plot_topo.addItem(text_item)
        
        # Store references for cleanup
        self.change_markers.append((line, text_item, time_stamp))
        
        # Clean up old markers outside the window
        self.cleanup_old_markers(time_stamp)
    
    def cleanup_old_markers(self, current_time):
        """Remove markers that are outside the current time window"""
        win = self.window_seconds
        markers_to_remove = []
        
        for line, text, timestamp in self.change_markers:
            if current_time - timestamp > win:
                self.plot_topo.removeItem(line)
                self.plot_topo.removeItem(text)
                markers_to_remove.append((line, text, timestamp))
        
        # Remove from list
        for marker in markers_to_remove:
            self.change_markers.remove(marker)
    def clear_markers_only(self):
        """Clear only the change markers and overlay, keep trace data"""
        # Clear existing markers
        for line, text, _ in self.change_markers:
            self.plot_topo.removeItem(line)
            self.plot_topo.removeItem(text)
        self.change_markers.clear()
        
        # Hide the overlay
        self.change_overlay.hide()
        
        print("Change markers and overlay cleared")

    def clear_trace(self):
        # Clear existing markers
        for line, text, _ in self.change_markers:
            self.plot_topo.removeItem(line)
            self.plot_topo.removeItem(text)
        self.change_markers.clear()
        
        self.t_stamps.clear()
        self.z_hist.clear()
        self.curve_topo.setData([], [])
        self.aux_hist.clear()
        self.curve_aux.setData([], [])
        print("Trace cleared")

    def on_channel_changed(self, name: str):
        try:
            unit = CHANNELS[name][2]
        except Exception:
            # keep previous if missing
            name = self.aux_channel
            unit = self.aux_unit
        self.aux_channel = name
        self.aux_unit = unit
        self.lbl_aux_unit.setText(f"[{unit}]")
        self.plot_aux.setLabel("left", name, units=unit)
        self.scale_plot_fonts()  # Reapply font scaling to new label
        self.aux_hist.clear()
        self.curve_aux.setData([], [])

    # ---------- polling ----------
    def poll(self):
        now = datetime.datetime.now()
        # time in seconds since midnight; stable increasing
        t_now = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        # read topo
        z = self.last_z
        try:
            if self.live_mode and self.driver:
                z = float(self.driver.read_scaled("Topo"))
                
                # Calculate change from previous reading
                change = z - self.previous_z
                
                # Update positions
                self.previous_z = self.last_z
                self.last_z = z
                
                # Show overlay if change is significant
                if abs(change) >= self.change_threshold:
                    self.change_overlay.show_change(change, z)
                    # Add vertical marker at this time point
                    self.add_change_marker(t_now, z, change)
                
                if self.feedback_enabled:
                    self.z_spin.blockSignals(True)
                    self.z_spin.setValue(z)
                    self.z_spin.blockSignals(False)
        except Exception as e:
            print(f"Polling Topo error: {e}")

        # read aux
        aux_val = None
        try:
            if self.driver:
                aux_val = float(self.driver.read_scaled(self.aux_channel))
        except Exception as e:
            print(f"Polling Aux '{self.aux_channel}' error: {e}")

        # push
        self.t_stamps.append(t_now)
        self.z_hist.append(z)
        if aux_val is not None:
            self.aux_hist.append(aux_val)

        # trim to window
        win = self.window_seconds
        while (
            self.t_stamps
            and len(self.t_stamps) > 1
            and self.t_stamps[-1] - self.t_stamps[0] > win
        ):
            self.t_stamps.pop(0)
            self.z_hist.pop(0)
            if self.aux_hist:
                self.aux_hist.pop(0)

        # redraw topo
        if self.t_stamps:
            self.curve_topo.setData(self.t_stamps, self.z_hist)
            x_min = max(0.0, t_now - win)
            x_max = t_now
            self.plot_topo.setXRange(x_min, x_max, padding=0.02)
            y_min = min(self.z_hist)
            y_max = max(self.z_hist)
            if y_max > y_min:
                pad = 0.1 * (y_max - y_min)
                self.plot_topo.setYRange(y_min - pad, y_max + pad)

        # redraw aux
        if self.t_stamps and self.aux_hist:
            # align lengths if aux started later
            n_aux = len(self.aux_hist)
            self.curve_aux.setData(self.t_stamps[-n_aux:], self.aux_hist)
            x_min = max(0.0, t_now - win)
            x_max = t_now
            self.plot_aux.setXRange(x_min, x_max, padding=0.02)
            y_min = min(self.aux_hist)
            y_max = max(self.aux_hist)
            if y_max > y_min:
                pad = 0.1 * (y_max - y_min)
                self.plot_aux.setYRange(y_min - pad, y_max + pad)

    def resizeEvent(self, event):
        """Handle window resize to reposition overlay"""
        super().resizeEvent(event)
        # The overlay will reposition itself when next shown

    # ---------- Qt ----------
    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        event.accept()


# def run(dde=None, driver=None):
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     w = ZConstAcquisition(dde=dde, driver=driver)
#     w.show()
#     return app.exec_()


# if __name__ == "__main__":
#     # Expect caller to pass a real driver with read_scaled(name).
#     # If you want to test without hardware, define a stub here.
#     class _StubDriver:
#         def __init__(self):
#             self._t0 = datetime.datetime.now()

#         def read_scaled(self, name):
#             dt = (datetime.datetime.now() - self._t0).total_seconds()
#             if name == "Topo":
#                 return 100.0 + 2.0 * pg.np.sin(0.8 * dt)
#             # generic aux
#             return 1.0 * pg.np.sin(2.0 * dt) + 5.0

#     # Comment out the stub when wiring to your stack.
#     sys.exit(run(driver=_StubDriver()))