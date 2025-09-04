# z_const_acquisition.py
"""
Z-Constant acquisition tab with ring buffer for stable memory usage.
"""

import datetime
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from sxm_ncafm_control.device_driver import CHANNELS

<<<<<<< HEAD

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
            
        line_edit = self.lineEdit()
        cursor_pos = line_edit.cursorPosition()
        text = line_edit.text()
        
        suffix = self.suffix()
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
        
        decimal_pos = text.find('.')
        step_size = self.determine_step_size(cursor_pos, text, decimal_pos)
        
        current_value = self.value()
        new_value = current_value + (steps * step_size)
        
        if new_value > self.maximum():
            new_value = self.maximum()
            print(f"Warning: Clamped to maximum value {self.maximum()}")
        elif new_value < self.minimum():
            new_value = self.minimum()
            print(f"Warning: Clamped to minimum value {self.minimum()}")
        
        if abs(new_value - current_value) > 1e-10:
            self.setValue(new_value)
        
        QtCore.QTimer.singleShot(0, lambda: line_edit.setCursorPosition(cursor_pos))
    
    def determine_step_size(self, cursor_pos, text, decimal_pos):
        if decimal_pos == -1:
            digits_from_right = len(text) - cursor_pos
            if digits_from_right <= 0:
                return 1.0
            return 10 ** (digits_from_right - 1)
        else:
            if cursor_pos <= decimal_pos:
                digits_from_right = decimal_pos - cursor_pos
                return 10 ** digits_from_right if digits_from_right > 0 else 1.0
            else:
                decimal_places = cursor_pos - decimal_pos - 1
                return 10 ** (-decimal_places - 1)
    
    def wheelEvent(self, event):
        if not self.custom_step_enabled:
            super().wheelEvent(event)
            return
            
        steps = event.angleDelta().y() // 120
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        
        if modifiers == QtCore.Qt.ControlModifier:
            self.custom_step_enabled = False
            super().wheelEvent(event)
            self.custom_step_enabled = True
        elif modifiers == QtCore.Qt.ShiftModifier:
            self.stepBy(steps * 10)
        else:
            self.stepBy(steps)
            
        event.accept()
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up:
            self.stepBy(1)
            event.accept()
        elif event.key() == QtCore.Qt.Key_Down:
            self.stepBy(-1)
            event.accept()
        else:
            super().keyPressEvent(event)


class ChangeOverlay(QtWidgets.QLabel):
    """Overlay for displaying z-position changes"""
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
        
        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        
        self.hide_timer = QtCore.QTimer()
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.fade_out)
        
        self.hide()
    
    def show_change(self, change_value, current_value):
        if abs(change_value) < 0.001:
            return
            
        if abs(change_value) >= 1.0:
            change_text = f"{change_value:+.2f}"
        elif abs(change_value) >= 0.01:
            change_text = f"{change_value:+.3f}"
        else:
            change_text = f"{change_value:+.4f}"
        
        display_text = f"Δz: {change_text} nm\nz: {current_value:.3f} nm"
        
        if change_value > 0:
            color = "#2E8B57"
        else:
            color = "#DC143C"
        
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
        
        if self.parent():
            parent_rect = self.parent().rect()
            overlay_rect = self.rect()
            x = 20
            y = parent_rect.height() - overlay_rect.height() - 20
            self.move(x, y)
        
        self.fade_in()
        
        self.hide_timer.stop()
        self.hide_timer.start(4000)
    
    def fade_in(self):
        self.show()
        self.fade_animation.stop()
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()
    
    def fade_out(self):
        self.fade_animation.stop()
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.hide)
        self.fade_animation.start()


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde, driver):
        super().__init__()
        self.setWindowTitle("Z-Const Acquisition")
        self.resize(1000, 900)

        self.dde = dde
        self.driver = driver
        self.live_mode = True
        self.last_z = 0.0
        self.previous_z = 0.0
        self.base_z = 0.0
        self.z_history = []
        self.timestamps = []
        self.window_seconds = 10
        self.feedback_enabled = True
        self.change_threshold = 0.001

        self.change_markers = []
        self.font_scale = 1.0
        self.base_font_size = QtWidgets.QApplication.font().pointSize() or 10
=======
from sxm_ncafm_control.device_driver import CHANNELS


class RingBuffer:
    """Fixed-size circular buffer for numeric data."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=float)
        self.index = 0
        self.full = False

    def append(self, value):
        self.data[self.index] = value
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def get_all(self):
        if not self.full:
            return self.data[:self.index]
        return np.concatenate((self.data[self.index:], self.data[:self.index]))


class ZConstAcquisition(QtWidgets.QWidget):
    def __init__(self, dde, driver, parent=None):
        super().__init__(parent)
        self.dde = dde
        self.driver = driver

        self.timer_interval_ms = 100      # poll interval
        self.window_seconds = 60          # visible window

        # Ring buffers sized for maximum samples in window
        max_samples = int(self.window_seconds * 1000 / self.timer_interval_ms)
        self.timestamps = RingBuffer(max_samples)
        self.z_history = RingBuffer(max_samples)

        self.last_z = 0.0
>>>>>>> 7124424d1e6f2e4e448eb69ba4b41202465d3b32

        # ---- UI ----
        vbox = QtWidgets.QVBoxLayout(self)

<<<<<<< HEAD
        # ---- Controls ----
        ctrl = QtWidgets.QHBoxLayout()

        self.btn_toggle = QtWidgets.QPushButton("Disable Feedback")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self.toggle_feedback)
        ctrl.addWidget(self.btn_toggle)

        ctrl.addWidget(QtWidgets.QLabel("Z Position:"))
        self.z_spin = FlexibleDoubleSpinBox()
        self.z_spin.setDecimals(6)
        self.z_spin.setSingleStep(0.001)
        self.z_spin.setRange(-280.0, 280.0)
        self.z_spin.setSuffix(" nm")
        self.z_spin.setEnabled(False)
        self.z_spin.valueChanged.connect(self.manual_update)
        ctrl.addWidget(self.z_spin)

        ctrl.addSpacing(12)

        ctrl.addWidget(QtWidgets.QLabel("Window (s):"))
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(["2", "5", "10", "30", "60", "120"])
        self.combo_window.setCurrentText("10")
        self.combo_window.currentTextChanged.connect(
            lambda val: setattr(self, "window_seconds", int(val))
        )
        ctrl.addWidget(self.combo_window)

        self.btn_clear = QtWidgets.QPushButton("Clear Trace")
        self.btn_clear.clicked.connect(self.clear_trace)
        ctrl.addWidget(self.btn_clear)

        self.btn_clear_markers = QtWidgets.QPushButton("Clear Markers")
        self.btn_clear_markers.clicked.connect(self.clear_markers_only)
        ctrl.addWidget(self.btn_clear_markers)

        ctrl.addSpacing(20)

        ctrl.addWidget(QtWidgets.QLabel("Font Size:"))
        self.font_scale_combo = QtWidgets.QComboBox()
        self.font_scale_combo.addItems(["Small", "Normal", "Large", "Extra Large"])
        self.font_scale_combo.setCurrentText("Normal")
        self.font_scale_combo.currentTextChanged.connect(self.change_font_scale)
        ctrl.addWidget(self.font_scale_combo)

        ctrl.addSpacing(20)
        ctrl.addWidget(QtWidgets.QLabel("Extra Plot Channel:"))
        self.extra_chan_combo = QtWidgets.QComboBox()
        self.extra_chan_combo.addItems(list(CHANNELS.keys()))
        self.extra_chan_combo.currentTextChanged.connect(self.update_extra_plot_label)
        ctrl.addWidget(self.extra_chan_combo)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # ---- Status ----
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Live mode, Feedback ON")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # ---- Z plot ----
        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.curve = self.plot.plot([], [], pen=pg.mkPen('b', width=2))
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Z Position", units="nm")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        self.change_overlay = ChangeOverlay(self.plot)

        # ---- Extra channel plot ----
        self.extra_plot = pg.PlotWidget()
        self.extra_plot.setBackground('w')
        self.extra_curve = self.extra_plot.plot([], [], pen=pg.mkPen('g', width=2))
        self.extra_plot.setLabel("bottom", "Time", units="s")
        self.extra_plot.setLabel("left", "Extra Channel")
        self.extra_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.extra_plot)

        self.extra_history = []

        # ---- Timer ----
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

        self.initialize_z_position()
        self.apply_font_scaling()
        self.update_extra_plot_label(self.extra_chan_combo.currentText())

    def update_extra_plot_label(self, chan_name):
        if chan_name in CHANNELS:
            _, _, unit, _ = CHANNELS[chan_name]
            self.extra_plot.setLabel("left", f"{chan_name} ({unit})")
        else:
            self.extra_plot.setLabel("left", "Extra Channel")

    def initialize_z_position(self):
        try:
            if self.driver:
                current_z = self.driver.read_scaled("Topo")
                self.last_z = current_z
                self.previous_z = current_z
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

    def toggle_feedback(self, checked: bool):
        if checked:
            self.btn_toggle.setText("Enable Feedback")
            try:
                if self.driver:
                    current_z = self.driver.read_scaled("Topo")
                    self.base_z = current_z
                    self.last_z = current_z
                    self.previous_z = current_z
                    self.z_spin.setValue(current_z)
                    print(f"Feedback disabled. Base Z: {current_z:.6f} nm")

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
            
        change = value - self.last_z
        self.previous_z = self.last_z
        
        try:
            delta = self.base_z - value 
            self.dde.set_channel(0, delta)
            self.last_z = value
            
            if abs(change) >= self.change_threshold:
                self.change_overlay.show_change(change, self.last_z)
                if self.timestamps:
                    current_time = self.timestamps[-1] if self.timestamps else 0
                    self.add_change_marker(current_time, self.last_z, change)
            
            print(f"Manual Z update: target {value:.6f} nm (Δ {delta:+.6f} nm from base)")
        except Exception as e:
            print(f"Manual Z write error: {e}")
            self.z_spin.setValue(self.last_z)

=======
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot = self.plot_widget.plot(pen=pg.mkPen("b", width=1))
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Topo (nm)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_widget)

        # Status
        self.status_label = QtWidgets.QLabel("Ready")
        vbox.addWidget(self.status_label)

        # Timer
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.poll)

    def start(self):
        self._timer.start(self.timer_interval_ms)
        self.status_label.setText("Running...")

    def stop(self):
        self._timer.stop()
        self.status_label.setText("Stopped")

>>>>>>> 7124424d1e6f2e4e448eb69ba4b41202465d3b32
    def poll(self):
        """Poll driver for Z value and update plot."""
        now = datetime.datetime.now()
        try:
<<<<<<< HEAD
            if self.live_mode and self.driver:
                z = self.driver.read_scaled("Topo")
                change = z - self.previous_z
                self.previous_z = self.last_z
                self.last_z = z
                if abs(change) >= self.change_threshold:
                    self.change_overlay.show_change(change, z)
                    self.add_change_marker(elapsed, z, change)
                self.z_spin.setValue(z)
            else:
                z = self.last_z
=======
            z = self.driver.read_scaled("Topo")
            self.last_z = z
>>>>>>> 7124424d1e6f2e4e448eb69ba4b41202465d3b32
        except Exception as e:
            print(f"Driver read error: {e}")
            z = self.last_z

        # Append to ring buffers
        self.timestamps.append(now.timestamp())
        self.z_history.append(z)

<<<<<<< HEAD
        # Extra channel
        chan_name = self.extra_chan_combo.currentText()
        if self.driver and chan_name in CHANNELS:
            chan_idx, _, _, scale = CHANNELS[chan_name]
            try:
                val = self.driver.read_raw(chan_idx) * scale
            except Exception:
                val = 0.0
        else:
            val = 0.0
        self.extra_history.append(val)

        # Trim
        while self.timestamps and self.timestamps[-1] - self.timestamps[0] > self.window_seconds:
            self.timestamps.pop(0)
            self.z_history.pop(0)
            self.extra_history.pop(0)

        # Update Z plot
        if self.timestamps:
            self.curve.setData(self.timestamps, self.z_history)
            x_min = max(0, elapsed - self.window_seconds)
            x_max = elapsed
            self.plot.setXRange(x_min, x_max, padding=0.02)
            if self.z_history:
                y_min, y_max = min(self.z_history), max(self.z_history)
                if y_max > y_min:
                    pad = (y_max - y_min) * 0.1
                    self.plot.setYRange(y_min - pad, y_max + pad)

            self.extra_curve.setData(self.timestamps, self.extra_history)
            self.extra_plot.setXRange(x_min, x_max, padding=0.02)
            if self.extra_history:
                ymin, ymax = min(self.extra_history), max(self.extra_history)
                if ymax > ymin:
                    pad = (ymax - ymin) * 0.1
                    self.extra_plot.setYRange(ymin - pad, ymax + pad)

        self.cleanup_old_markers(elapsed)

    def clear_trace(self):
        for line, text, _ in self.change_markers:
            self.plot.removeItem(line)
            self.plot.removeItem(text)
        self.change_markers.clear()
        self.z_history.clear()
        self.timestamps.clear()
        self.curve.setData([], [])
        self.extra_history.clear()
        self.extra_curve.setData([], [])
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
=======
        # Retrieve all valid samples
        t_all = self.timestamps.get_all()
        z_vals = self.z_history.get_all()

        if len(t_all) == 0:
>>>>>>> 7124424d1e6f2e4e448eb69ba4b41202465d3b32
            return

        # Relative time axis
        t_rel = t_all - t_all[0]

        # Update plot
        self.plot.setData(t_rel, z_vals)

        self.status_label.setText(
            f"Latest z = {z:.3f} nm at {now.strftime('%H:%M:%S')}"
        )
