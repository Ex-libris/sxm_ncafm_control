import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from sxm_ncafm_control.device_driver import CHANNELS


class FlexibleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """
    Right-side digit stepping:
    - Put the caret to the RIGHT of the digit you want to change.
    - First step locks that digit place (no order jumps).
    - Caret is restored to the RIGHT of that digit after each change.
    - Minus sign is ignored for caret math.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self.custom_step_enabled = True

        # lock state
        self._lock_active = False
        self._locked_step = None
        self._locked_anchor = None  # ('L', k) for k digits left of '.', or ('R', k) for k digits right of '.'

        self._lock_timer = QtCore.QTimer(self)
        self._lock_timer.setSingleShot(True)
        self._lock_timer.timeout.connect(self._unlock)
        self._lock_idle_ms = 800

        self.editingFinished.connect(self._unlock)

    # ---------- helpers ----------
    def _unlock(self):
        self._lock_active = False
        self._locked_step = None
        self._locked_anchor = None

    def _split_core(self, text):
        """Return (sign_len, core_digits) stripping suffix and leading sign."""
        suffix = self.suffix()
        core = text[:-len(suffix)] if (suffix and text.endswith(suffix)) else text
        if core.startswith(('+', '-')):
            return 1, core[1:]
        return 0, core

    def _effective_pos_right(self, cursor_pos, text):
        """
        Convert absolute cursor_pos to position in the digits string,
        then shift one left if possible so caret is to the RIGHT of the digit we change.
        """
        sign_len, digits = self._split_core(text)
        pos = max(0, cursor_pos - sign_len)
        if pos > 0 and pos <= len(digits) and digits[pos - 1].isdigit():
            return pos - 1  # digit index whose RIGHT edge the caret is at
        return max(0, min(len(digits) - 1, pos)) if digits else 0

    def _anchor_from_pos_right(self, pos_right, digits):
        """Make an anchor relative to the decimal using pos_right (a digit index)."""
        dec = digits.find('.')
        if dec == -1:
            dec = len(digits)

        if pos_right <= dec - 1:
            # left side digit: k digits to the left of '.'
            k = dec - pos_right
            return ('L', k)
        else:
            # right side digit: k digits to the right of '.'
            k = pos_right - dec - 1
            return ('R', k)

    def _restore_caret_right(self, line_edit, anchor):
        """Place caret to the RIGHT of the anchored digit."""
        text = line_edit.text()
        sign_len, digits = self._split_core(text)
        dec = digits.find('.')
        if dec == -1:
            dec = len(digits)

        side, k = anchor
        if side == 'L':
            di = max(0, dec - k)
            caret_pos = di + 1  # right of that digit
        else:
            di = dec + 1 + k
            caret_pos = min(len(digits), di + 1)  # right of that digit

        abs_pos = sign_len + caret_pos
        QtCore.QTimer.singleShot(0, lambda: line_edit.setCursorPosition(abs_pos))

    def _step_size_from_pos_right(self, pos_right, digits):
        """Compute step size for the digit at pos_right (RIGHT-side model)."""
        dec = digits.find('.')
        if dec == -1:
            dec = len(digits)

        if pos_right <= dec - 1:
            # left of decimal: ones -> 10^0, tens -> 10^1, etc.
            power = dec - pos_right - 1
            return 10 ** power
        else:
            # right of decimal: tenths -> 10^-1, hundredths -> 10^-2, etc.
            power = -(pos_right - dec)
            return 10 ** power

    # ---------- main stepping ----------
    def stepBy(self, steps):
        if not self.custom_step_enabled:
            super().stepBy(steps)
            return

        le = self.lineEdit()
        cur_pos_abs = le.cursorPosition()
        raw_text = le.text()
        _, digits = self._split_core(raw_text)

        # first step in a session: lock step and anchor
        if not self._lock_active:
            pos_right = self._effective_pos_right(cur_pos_abs, raw_text)
            step_size = self._step_size_from_pos_right(pos_right, digits)

            self._lock_active = True
            self._locked_step = step_size
            self._locked_anchor = self._anchor_from_pos_right(pos_right, digits)

        # always use the locked step
        step_size = self._locked_step or self.singleStep()
        new_val = self.value() + steps * step_size

        # clamp
        new_val = min(max(new_val, self.minimum()), self.maximum())

        # set and restore caret to RIGHT of same digit
        self.setValue(new_val)
        if self._locked_anchor is not None:
            self._restore_caret_right(le, self._locked_anchor)

        # keep lock alive while stepping
        self._lock_timer.start(self._lock_idle_ms)

    # wheel and arrows go through the same logic
    def wheelEvent(self, event):
        if not self.custom_step_enabled:
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() // 120
        if steps:
            self.stepBy(steps)
            event.accept()
        else:
            super().wheelEvent(event)

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
        color = "#2E8B57" if change_value > 0 else "#DC143C"
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

        # state
        self.last_z = 0.0
        self.previous_z = 0.0

        # absolute manual mode mapping
        self.ch0_sign = +1   # set to -1 if direction is inverted
        self.abs_ref_z = 0.0 # Z at disable
        self.ch0_base = 0.0  # CH0 at disable

        self.z_history = []
        self.timestamps = []
        self.window_seconds = 10
        self.feedback_enabled = True
        self.change_threshold = 0.001

        self.change_markers = []
        self.font_scale = 1.0
        self.base_font_size = QtWidgets.QApplication.font().pointSize() or 10

        layout = QtWidgets.QVBoxLayout(self)

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
        self.curve = self.plot.plot([], [], pen=pg.mkPen((200,50,50), width=2))
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Z Position", units="nm")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        self.change_overlay = ChangeOverlay(self.plot)

        # ---- Extra channel plot ----
        self.extra_plot = pg.PlotWidget()
        self.extra_plot.setBackground('w')
        self.extra_curve = self.extra_plot.plot([], [], pen=pg.mkPen((50,100,200), width=2))
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
                print(f"Initialized Z: {current_z:.6f} nm")
            else:
                print("No driver; mock init")
                self.last_z = 0.0
                self.previous_z = 0.0
        except Exception as e:
            print(f"Init error: {e}")
            self.last_z = 0.0
            self.previous_z = 0.0

    def toggle_feedback(self, checked: bool):
        if checked:
            # DISABLE feedback → manual absolute mode
            self.btn_toggle.setText("Enable Feedback")
            try:
                if self.driver:
                    current_z = self.driver.read_scaled("Topo")
                else:
                    current_z = self.last_z

                self.abs_ref_z = current_z
                self.last_z = current_z
                self.previous_z = current_z
                print(f"FB OFF. Z_ref = {current_z:.6f} nm")

                try:
                    self.ch0_base = self.dde.get_channel(0)  # read-only is fine
                    print(f"CH0_base = {self.ch0_base:.6f}")
                except Exception as e:
                    print(f"CH0 readback failed: {e} (keeping CH0_base={self.ch0_base:.6f})")

                self.dde.feed_para("enable", 1)
                self.feedback_enabled = False
                self.live_mode = False

                # Spinbox shows ABSOLUTE Z, editable
                self.z_spin.blockSignals(True)
                self.z_spin.setEnabled(True)
                self.z_spin.setSuffix(" nm (absolute)")
                self.z_spin.setValue(current_z)
                self.z_spin.blockSignals(False)

                self.status_label.setText("Status: Manual mode, Feedback OFF (Abs Z)")
                self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

            except Exception as e:
                print(f"Error disabling FB: {e}")

        else:
            # ENABLE feedback
            self.btn_toggle.setText("Disable Feedback")
            try:
                z_target = self.z_spin.value()  # absolute target
                dz_cmd = z_target - self.abs_ref_z
                final_ch0 = self.ch0_base + self.ch0_sign * dz_cmd
                self.dde.set_channel(0, final_ch0)
                print(f"Restore before FB ON: Z_target={z_target:.6f} → CH0={final_ch0:.6f}")
            except Exception as e:
                print(f"Warn: cannot preset CH0: {e}")

            self.dde.feed_para("enable", 0)
            self.feedback_enabled = True
            self.live_mode = True
            self.z_spin.setEnabled(False)
            self.z_spin.setSuffix(" nm")
            self.status_label.setText("Status: Live mode, Feedback ON")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            print("FB ON")

    def manual_update(self, abs_target: float):
        """Manual Z in ABSOLUTE nm when feedback is disabled."""
        if self.live_mode:
            return

        dz_cmd = abs_target - self.abs_ref_z
        ch0_target = self.ch0_base + self.ch0_sign * dz_cmd

        try:
            self.dde.set_channel(0, ch0_target)
        except Exception as e:
            print(f"Manual write error: {e}")
            return

        change = abs_target - self.last_z
        self.previous_z = self.last_z
        self.last_z = abs_target

        if abs(change) >= self.change_threshold:
            self.change_overlay.show_change(change, self.last_z)
            # marker ONLY on manual input
            now = datetime.datetime.now()
            elapsed = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            self.add_change_marker(elapsed, self.last_z, change)

        print(f"Manual ABS Z: target={abs_target:.6f} nm, dz_cmd={dz_cmd:+.6f} nm → CH0={ch0_target:.6f}")

    def poll(self):
        now = datetime.datetime.now()
        elapsed = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        try:
            if self.live_mode and self.driver:
                z = self.driver.read_scaled("Topo")
                change = z - self.previous_z
                self.previous_z = self.last_z
                self.last_z = z
                if abs(change) >= self.change_threshold:
                    self.change_overlay.show_change(change, z)
                    # no marker in live mode
                # show absolute Z in live mode
                self.z_spin.blockSignals(True)
                self.z_spin.setValue(z)
                self.z_spin.blockSignals(False)
            else:
                # manual mode: read for plot only; do not touch spinbox
                if self.driver:
                    z = self.driver.read_scaled("Topo")
                    self.last_z = z
                else:
                    z = self.last_z
        except Exception as e:
            print(f"Poll error: {e}")
            z = self.last_z if hasattr(self, 'last_z') else 0.0

        self.timestamps.append(elapsed)
        self.z_history.append(z)

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

        # Update plots
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

    def change_font_scale(self, scale_text):
        scale_map = {"Small": 0.8, "Normal": 1.0, "Large": 1.3, "Extra Large": 1.6}
        self.font_scale = scale_map.get(scale_text, 1.0)
        self.apply_font_scaling()
        print(f"Font scale: {scale_text} ({self.font_scale}x)")

    def apply_font_scaling(self):
        scaled_size = int(self.base_font_size * self.font_scale)
        font = QtGui.QFont()
        font.setPointSize(scaled_size)
        self.setFont(font)
        for child in self.findChildren(QtWidgets.QWidget):
            if not isinstance(child, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
                child.setFont(font)
        self.scale_plot_fonts()
        self.update_overlay_font_size()

    def scale_plot_fonts(self):
        label_font_size = max(8, int(10 * self.font_scale))
        tick_font_size = max(7, int(9 * self.font_scale))
        label_style = {'font-size': f'{label_font_size}pt', 'color': 'black'}
        self.plot.setLabel('bottom', 'Time', units='s', **label_style)
        self.plot.setLabel('left', 'Z Position', units='nm', **label_style)
        self.plot.getAxis('bottom').setTickFont(QtGui.QFont('', tick_font_size))
        self.plot.getAxis('left').setTickFont(QtGui.QFont('', tick_font_size))

    def update_overlay_font_size(self):
        overlay_font_size = max(12, int(14 * self.font_scale))
        current_style = self.change_overlay.styleSheet()
        import re
        new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {overlay_font_size}px', current_style)
        self.change_overlay.setStyleSheet(new_style)

    def add_change_marker(self, time_stamp, z_value, change_value):
        if abs(change_value) < self.change_threshold:
            return
        color = (46, 139, 87, 150) if change_value > 0 else (220, 20, 60, 150)
        line = pg.InfiniteLine(pos=time_stamp, angle=90, pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine))
        marker_font_size = max(8, int(10 * self.font_scale))
        if abs(change_value) >= 1.0:
            change_text = f"{change_value:+.2f}"
        elif abs(change_value) >= 0.01:
            change_text = f"{change_value:+.3f}"
        else:
            change_text = f"{change_value:+.4f}"
        text_item = pg.TextItem(text=change_text, color=color[:3], anchor=(0.5, 1.1))
        font = QtGui.QFont()
        font.setPointSize(marker_font_size)
        text_item.setFont(font)
        text_item.setPos(time_stamp, z_value)
        self.plot.addItem(line)
        self.plot.addItem(text_item)
        self.change_markers.append((line, text_item, time_stamp))

    def cleanup_old_markers(self, current_time):
        win = self.window_seconds
        to_remove = []
        for line, text, timestamp in self.change_markers:
            if current_time - timestamp > win:
                self.plot.removeItem(line)
                self.plot.removeItem(text)
                to_remove.append((line, text, timestamp))
        for m in to_remove:
            self.change_markers.remove(m)

    def clear_markers_only(self):
        for line, text, _ in self.change_markers:
            self.plot.removeItem(line)
            self.plot.removeItem(text)
        self.change_markers.clear()
        self.change_overlay.hide()
        print("Markers cleared")

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
        event.accept()
