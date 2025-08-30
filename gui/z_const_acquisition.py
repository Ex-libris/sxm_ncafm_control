import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


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
        
        # CRITICAL: Additional safety check - prevent large jumps
        max_single_jump = 100.0  # Maximum allowed change in one step
        if abs(new_value - current_value) > max_single_jump:
            if steps > 0:
                new_value = current_value + max_single_jump
            else:
                new_value = current_value - max_single_jump
            print(f"Warning: Large step prevented. Limited change to ±{max_single_jump}")
        
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
            print(f"Step: {current_value:.3f} → {new_value:.3f} (step size: {step_size}, change: {new_value-current_value:+.3f})")
        
        # Restore cursor position (approximately)
        QtCore.QTimer.singleShot(0, lambda: line_edit.setCursorPosition(cursor_pos))
    
    def determine_step_size(self, cursor_pos, text, decimal_pos):
        """Determine step size based on cursor position with safety limits"""
        # Remove any minus sign for position calculation
        clean_text = text.lstrip('-')
        sign_offset = len(text) - len(clean_text)
        adjusted_cursor = cursor_pos - sign_offset
        adjusted_decimal = (decimal_pos - sign_offset) if decimal_pos != -1 else -1
        
        # Ensure cursor is within valid range
        if adjusted_cursor < 0:
            adjusted_cursor = 0
        
        if adjusted_decimal == -1:  # No decimal point
            # Count digits from right to cursor position
            digits_from_right = len(clean_text) - adjusted_cursor
            if digits_from_right <= 0:
                return 1.0
            step_power = digits_from_right - 1
        else:
            if adjusted_cursor <= adjusted_decimal:
                # Before decimal point
                digits_from_right = adjusted_decimal - adjusted_cursor
                step_power = digits_from_right
            else:
                # After decimal point
                decimal_places = adjusted_cursor - adjusted_decimal - 1
                step_power = -decimal_places - 1
        
        # CRITICAL SAFETY: Limit maximum step size to prevent dangerous jumps
        max_safe_step_power = 2  # Maximum step of 100
        min_safe_step_power = -3  # Minimum step of 0.001 (matching spinbox decimals)
        
        step_power = max(min_safe_step_power, min(max_safe_step_power, step_power))
        step_size = 10 ** step_power
        
        # Additional safety: never allow steps larger than 1/10 of the total range
        total_range = abs(self.maximum() - self.minimum())
        max_allowed_step = total_range / 10.0
        
        return min(step_size, max_allowed_step)
    
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


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde, driver):
        super().__init__()
        self.dde = dde  # For DDE read/write
        self.driver = driver  # For IOCTL read
        self.live_mode = True
        self.last_z = 0.0
        self.base_z = 0.0
        self.z_history = []
        self.timestamps = []
        self.window_seconds = 10
        self.feedback_enabled = True

        layout = QtWidgets.QVBoxLayout(self)

        # ---- Top Controls ----
        ctrl = QtWidgets.QHBoxLayout()

        # Feedback toggle
        self.btn_toggle = QtWidgets.QPushButton("Disable Feedback")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self.toggle_feedback)
        ctrl.addWidget(self.btn_toggle)

        # Enhanced Z position spinbox
        self.z_spin = FlexibleDoubleSpinBox()
        self.z_spin.setDecimals(3)
        self.z_spin.setSingleStep(0.001)  # This is now the fallback step
        self.z_spin.setRange(-1000.0, 1000.0)
        self.z_spin.setSuffix(" nm")
        self.z_spin.setEnabled(False)
        self.z_spin.valueChanged.connect(self.manual_update)
        
        # Set up the spinbox for better editing
        self.z_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.z_spin.setAccelerated(True)  # Faster stepping when holding buttons
        
        # CRITICAL SAFETY: Disable wraparound behavior
        self.z_spin.setWrapping(False)  # Prevent wraparound from max to min
        
        ctrl.addWidget(QtWidgets.QLabel("Z Position:"))
        ctrl.addWidget(self.z_spin)

        # Add usage hint label
        hint_label = QtWidgets.QLabel("Tip: Click on digit → Arrow keys/Mouse wheel to adjust")
        hint_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        ctrl.addWidget(hint_label)

        # Time window selection
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(["2", "5", "10", "30", "60", "120"])
        self.combo_window.setCurrentText("10")
        self.combo_window.currentTextChanged.connect(
            lambda val: setattr(self, "window_seconds", int(val))
        )
        ctrl.addWidget(QtWidgets.QLabel("Window (s):"))
        ctrl.addWidget(self.combo_window)

        # Clear trace button
        self.btn_clear = QtWidgets.QPushButton("Clear Trace")
        self.btn_clear.clicked.connect(self.clear_trace)
        ctrl.addWidget(self.btn_clear)

        layout.addLayout(ctrl)

        # ---- Controls Help ----
        help_layout = QtWidgets.QHBoxLayout()
        help_text = QtWidgets.QLabel(
            "Controls: Mouse wheel = position-aware step | Shift+wheel = 10x step | "
            "Ctrl+wheel = fixed step | ↑↓ arrows = position-aware step"
        )
        help_text.setStyleSheet("QLabel { color: #666; font-size: 9px; }")
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        layout.addLayout(help_layout)

        # ---- Status Display ----
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Live mode, Feedback ON")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # ---- Plot ----
        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.curve = self.plot.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Z Position", units="nm")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        # ---- Timer for polling ----
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

        self.initialize_z_position()

    def initialize_z_position(self):
        try:
            if self.driver:
                current_z = self.driver.read_scaled("Topo")
                self.last_z = current_z
                self.z_spin.setValue(current_z)
                print(f"Initialized Z position: {current_z:.3f} nm")
            else:
                print("No driver available - using mock initialization")
                self.last_z = 0.0
        except Exception as e:
            print(f"Error initializing Z position: {e}")
            self.last_z = 0.0

    def toggle_feedback(self, checked: bool):
        if checked:
            # Disabling feedback
            self.btn_toggle.setText("Enable Feedback")
            try:
                if self.driver:
                    current_z = self.driver.read_scaled("Topo")
                    self.base_z = current_z
                    self.last_z = current_z
                    self.z_spin.setValue(current_z)
                    print(f"Feedback disabled. Base Z: {current_z:.3f} nm")

                # Disable feedback via DDE
                self.dde.feed_para("enable", 1)
                self.feedback_enabled = False

            except Exception as e:
                print(f"Error reading Z before disabling feedback: {e}")
                self.z_spin.setValue(self.last_z)

            self.live_mode = False
            self.z_spin.setEnabled(True)
            self.status_label.setText("Status: Manual mode, Feedback OFF")
            self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

        else:
            # Enabling feedback
            self.btn_toggle.setText("Disable Feedback")
            self.dde.feed_para("enable", 0)
            self.feedback_enabled = True
            self.live_mode = True
            self.z_spin.setEnabled(False)
            self.status_label.setText("Status: Live mode, Feedback ON")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            print("Feedback enabled - returning to live mode")

    def manual_update(self, value: float):
        if self.live_mode:
            return
        try:
            delta = self.base_z - value 
            self.dde.set_channel(0, delta)
            self.last_z = value
            print(f"Manual Z update: target {value:.3f} nm (Δ {delta:+.3f} nm from base)")
        except Exception as e:
            print(f"Manual Z write error: {e}")
            self.z_spin.setValue(self.last_z)

    def poll(self):
        now = datetime.datetime.now()
        elapsed = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        try:
            if self.live_mode and self.driver:
                z = self.driver.read_scaled("Topo")
                self.z_spin.setValue(z)
                self.last_z = z
            else:
                z = self.last_z
        except Exception as e:
            print(f"Polling error: {e}")
            z = self.last_z if hasattr(self, 'last_z') else 0.0

        self.timestamps.append(elapsed)
        self.z_history.append(z)

        while (self.timestamps and 
               len(self.timestamps) > 1 and 
               self.timestamps[-1] - self.timestamps[0] > self.window_seconds):
            self.timestamps.pop(0)
            self.z_history.pop(0)

        if len(self.timestamps) > 0:
            self.curve.setData(self.timestamps, self.z_history)
            x_min = max(0, elapsed - self.window_seconds)
            x_max = elapsed
            self.plot.setXRange(x_min, x_max, padding=0.02)
            if len(self.z_history) > 0:
                y_min = min(self.z_history)
                y_max = max(self.z_history)
                y_range = y_max - y_min
                if y_range > 0:
                    padding = y_range * 0.1
                    self.plot.setYRange(y_min - padding, y_max + padding)

    def clear_trace(self):
        self.z_history.clear()
        self.timestamps.clear()
        self.curve.setData([], [])
        print("Trace cleared")

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
        event.accept()