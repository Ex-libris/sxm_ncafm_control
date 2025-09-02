import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class FlexibleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Enhanced spinbox with position-aware stepping from new version"""
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
    """Elegant text overlay for displaying z-position changes from new version"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 240);
                border: 1px solid rgba(150, 150, 150, 120);
                border-radius: 12px;
                padding: 6px 12px;
                font-size: 14px;
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
        else:
            color = "#DC143C"  # Crimson for negative changes
        
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
        
        # Auto-hide after 4 seconds
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
    def __init__(self, dde, driver):
        super().__init__()
        self.setWindowTitle("Z-Const Acquisition")
        self.resize(1000, 700)
        
        # KEEP OLD WORKING CORE VARIABLES AND LOGIC
        self.dde = dde  # For DDE read/write
        self.driver = driver  # For IOCTL read
        self.live_mode = True
        self.last_z = 0.0
        self.previous_z = 0.0  # For change calculation (NEW)
        self.base_z = 0.0
        self.z_history = []
        self.timestamps = []
        self.window_seconds = 10
        self.feedback_enabled = True
        self.change_threshold = 0.001  # Minimum change to display overlay (NEW)
        
        # NEW: Change event markers
        self.change_markers = []  # Store reference lines for changes
        
        # NEW: Font scaling for accessibility
        self.font_scale = 1.0
        self.base_font_size = QtWidgets.QApplication.font().pointSize() or 10

        layout = QtWidgets.QVBoxLayout(self)

        # ---- Top Controls (ENHANCED UI) ----
        ctrl = QtWidgets.QHBoxLayout()

        # Feedback toggle (KEEP OLD LOGIC)
        self.btn_toggle = QtWidgets.QPushButton("Disable Feedback")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self.toggle_feedback)
        ctrl.addWidget(self.btn_toggle)

        # Z position spinbox (NEW ENHANCED SPINBOX)
        ctrl.addWidget(QtWidgets.QLabel("Z Position:"))
        self.z_spin = FlexibleDoubleSpinBox()  # NEW: Enhanced spinbox
        self.z_spin.setDecimals(6)  # Keep old precision
        self.z_spin.setSingleStep(0.001)  # Keep old step
        self.z_spin.setRange(-1000.0, 1000.0)  # Keep old range
        self.z_spin.setSuffix(" nm")
        self.z_spin.setEnabled(False)
        self.z_spin.valueChanged.connect(self.manual_update)  # KEEP OLD LOGIC
        ctrl.addWidget(self.z_spin)

        ctrl.addSpacing(12)  # NEW: Better spacing
        
        # Time window selection
        ctrl.addWidget(QtWidgets.QLabel("Window (s):"))
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(["2", "5", "10", "30", "60", "120"])
        self.combo_window.setCurrentText("10")
        self.combo_window.currentTextChanged.connect(
            lambda val: setattr(self, "window_seconds", int(val))
        )
        ctrl.addWidget(self.combo_window)

        # Clear trace button
        self.btn_clear = QtWidgets.QPushButton("Clear Trace")
        self.btn_clear.clicked.connect(self.clear_trace)
        ctrl.addWidget(self.btn_clear)

        # NEW: Clear markers button
        self.btn_clear_markers = QtWidgets.QPushButton("Clear Markers")
        self.btn_clear_markers.clicked.connect(self.clear_markers_only)
        self.btn_clear_markers.setToolTip("Clear change markers and overlays without affecting trace data")
        ctrl.addWidget(self.btn_clear_markers)

        ctrl.addSpacing(20)
        
        # NEW: Font scaling controls for accessibility
        ctrl.addWidget(QtWidgets.QLabel("Font Size:"))
        self.font_scale_combo = QtWidgets.QComboBox()
        self.font_scale_combo.addItems(["Small", "Normal", "Large", "Extra Large"])
        self.font_scale_combo.setCurrentText("Normal")
        self.font_scale_combo.currentTextChanged.connect(self.change_font_scale)
        self.font_scale_combo.setToolTip("Adjust font size for better accessibility")
        ctrl.addWidget(self.font_scale_combo)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # NEW: Help row with keyboard shortcuts
        help_row = QtWidgets.QHBoxLayout()
        help_lbl = QtWidgets.QLabel(
            "Wheel = pos-aware step | Shift+wheel = 10× | Ctrl+wheel = fixed step | ↑↓ = pos-aware step"
        )
        help_lbl.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        help_lbl.setWordWrap(True)
        help_row.addWidget(help_lbl)
        layout.addLayout(help_row)

        # ---- Status Display ----
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Live mode, Feedback ON")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # ---- Plot with Change Overlay (NEW ENHANCED PLOT) ----
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.curve = self.plot.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Z Position", units="nm")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.plot)
        
        # NEW: Create change overlay
        self.change_overlay = ChangeOverlay(self.plot)
        
        layout.addWidget(plot_container)

        # ---- Timer for polling (KEEP OLD LOGIC) ----
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

        # Initialize (KEEP OLD LOGIC)
        self.initialize_z_position()
        # NEW: Apply initial font scaling
        self.apply_font_scaling()

    # KEEP OLD WORKING INITIALIZATION LOGIC
    def initialize_z_position(self):
        try:
            if self.driver:
                current_z = self.driver.read_scaled("Topo")
                self.last_z = current_z
                self.previous_z = current_z  # NEW: Track for changes
                self.z_spin.setValue(current_z)
                print(f"Initialized Z position: {current_z:.6f} nm")
            else:
                print("No driver available - using mock initialization")
                self.last_z = 0.0
                self.previous_z = 0.0
        except Exception as e:
            print(f"Error initializing Z position: {e}")
            self.last_z = 0.0
            self.previous_z = 0.0

    # KEEP OLD WORKING FEEDBACK TOGGLE LOGIC
    def toggle_feedback(self, checked: bool):
        if checked:
            # Disabling feedback
            self.btn_toggle.setText("Enable Feedback")
            try:
                if self.driver:
                    current_z = self.driver.read_scaled("Topo")
                    self.base_z = current_z
                    self.last_z = current_z
                    self.previous_z = current_z  # NEW: Track for changes
                    self.z_spin.setValue(current_z)
                    print(f"Feedback disabled. Base Z: {current_z:.6f} nm")

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

    # KEEP OLD WORKING MANUAL UPDATE LOGIC but add visual feedback
    def manual_update(self, value: float):
        if self.live_mode:
            return
            
        # NEW: Calculate change for overlay
        change = value - self.last_z
        self.previous_z = self.last_z
        
        try:
            delta = self.base_z - value 
            self.dde.set_channel(0, delta)
            self.last_z = value
            
            # NEW: Show overlay for manual changes
            if abs(change) >= self.change_threshold:
                self.change_overlay.show_change(change, self.last_z)
                # Add marker at current time if we have timestamp data
                if self.timestamps:
                    current_time = self.timestamps[-1] if self.timestamps else 0
                    self.add_change_marker(current_time, self.last_z, change)
            
            print(f"Manual Z update: target {value:.6f} nm (Δ {delta:+.6f} nm from base)")
        except Exception as e:
            print(f"Manual Z write error: {e}")
            self.z_spin.setValue(self.last_z)

    # KEEP OLD WORKING POLL LOGIC but add change detection
    def poll(self):
        now = datetime.datetime.now()
        elapsed = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        try:
            if self.live_mode and self.driver:
                z = self.driver.read_scaled("Topo")
                
                # NEW: Calculate change from previous reading for overlay
                change = z - self.previous_z
                self.previous_z = self.last_z
                self.last_z = z
                
                # NEW: Show overlay if change is significant and in live mode
                if abs(change) >= self.change_threshold:
                    self.change_overlay.show_change(change, z)
                    self.add_change_marker(elapsed, z, change)
                
                self.z_spin.setValue(z)
            else:
                z = self.last_z
        except Exception as e:
            print(f"Polling error: {e}")
            z = self.last_z if hasattr(self, 'last_z') else 0.0

        self.timestamps.append(elapsed)
        self.z_history.append(z)

        # KEEP OLD WORKING WINDOW TRIMMING LOGIC
        while (self.timestamps and 
               len(self.timestamps) > 1 and 
               self.timestamps[-1] - self.timestamps[0] > self.window_seconds):
            self.timestamps.pop(0)
            self.z_history.pop(0)

        # KEEP OLD WORKING PLOT UPDATE LOGIC
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
        
        # NEW: Clean up old markers
        self.cleanup_old_markers(elapsed)

    # KEEP OLD WORKING CLEAR LOGIC
    def clear_trace(self):
        # NEW: Clear existing markers
        for line, text, _ in self.change_markers:
            self.plot.removeItem(line)
            self.plot.removeItem(text)
        self.change_markers.clear()
        
        self.z_history.clear()
        self.timestamps.clear()
        self.curve.setData([], [])
        print("Trace cleared")

    # NEW VISUAL ENHANCEMENT METHODS FROM NEW VERSION
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
    
    def scale_plot_fonts(self):
        """Scale fonts for plot elements"""
        label_font_size = max(8, int(10 * self.font_scale))
        tick_font_size = max(7, int(9 * self.font_scale))
        
        # Style for plot labels and ticks
        label_style = {'font-size': f'{label_font_size}pt', 'color': 'black'}
        tick_style = {'font-size': f'{tick_font_size}pt', 'color': 'black'}
        
        # Update plot
        self.plot.setLabel('bottom', 'Time', units='s', **label_style)
        self.plot.setLabel('left', 'Z Position', units='nm', **label_style)
        self.plot.getAxis('bottom').setTickFont(QtGui.QFont('', tick_font_size))
        self.plot.getAxis('left').setTickFont(QtGui.QFont('', tick_font_size))
    
    def update_overlay_font_size(self):
        """Update the change overlay font size"""
        overlay_font_size = max(12, int(14 * self.font_scale))
        current_style = self.change_overlay.styleSheet()
        
        # Update font-size in the stylesheet
        import re
        new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {overlay_font_size}px', current_style)
        self.change_overlay.setStyleSheet(new_style)

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
        
        # Create text label with the change value
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
        self.plot.addItem(line)
        self.plot.addItem(text_item)
        
        # Store references for cleanup
        self.change_markers.append((line, text_item, time_stamp))
    
    def cleanup_old_markers(self, current_time):
        """Remove markers that are outside the current time window"""
        win = self.window_seconds
        markers_to_remove = []
        
        for line, text, timestamp in self.change_markers:
            if current_time - timestamp > win:
                self.plot.removeItem(line)
                self.plot.removeItem(text)
                markers_to_remove.append((line, text, timestamp))
        
        # Remove from list
        for marker in markers_to_remove:
            self.change_markers.remove(marker)
    
    def clear_markers_only(self):
        """Clear only the change markers and overlay, keep trace data"""
        # Clear existing markers
        for line, text, _ in self.change_markers:
            self.plot.removeItem(line)
            self.plot.removeItem(text)
        self.change_markers.clear()
        
        # Hide the overlay
        self.change_overlay.hide()
        
        print("Change markers and overlay cleared")

    # KEEP OLD WORKING CLOSE EVENT
    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
        event.accept()