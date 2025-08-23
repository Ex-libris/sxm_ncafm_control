# gui/params_tab.py
import json
import datetime
from typing import List, Tuple, Dict, Any
from PyQt5 import QtWidgets, QtCore, QtGui

from .common import (
    PARAMS_BASE,
    PARAM_TOOLTIPS,
    _to_float,
    confirm_high_voltage,
    NumericItemDelegate,
    VOLTAGE_LIMIT_ABS,
)


class ParamsTab(QtWidgets.QWidget):
    """
    Parameters tab: editable table of NC-AFM tuning parameters.

    Features:
        - Displays a table with "Previous | Current | New Value" columns.
        - Supports adding custom EditXX parameters.
        - Allows saving/loading/staging of parameter sets ("tunes").
        - Provides Apply, Auto-Send, and logging functionality.

    Signals:
        custom_params_changed (list):
            Emitted whenever custom parameters are added.
            Carries a list of (ptype, pcode, label) tuples for Step Test tab.
    """

    custom_params_changed = QtCore.pyqtSignal(list)

    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self._custom_params: List[Tuple[str, str, object, str, bool]] = []

        # --- Layout ---
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)

        # Toolbar buttons
        self.btn_apply = QtWidgets.QPushButton("Apply Selected")
        toolbar.addWidget(self.btn_apply)

        self.chk_auto = QtWidgets.QCheckBox("Auto-Send on Edit")
        toolbar.addWidget(self.chk_auto)

        self.btn_add_custom = QtWidgets.QPushButton("Add Custom EditXX…")
        self.btn_add_custom.clicked.connect(self._add_custom_editxx)
        toolbar.addWidget(self.btn_add_custom)

        toolbar.addWidget(QtWidgets.QLabel())  # spacer

        self.btn_show_log = QtWidgets.QPushButton("Show Change Log")
        self.btn_show_log.setCheckable(True)
        toolbar.addWidget(self.btn_show_log)

        self.btn_save_tune = QtWidgets.QPushButton("Save Tune…")
        toolbar.addWidget(self.btn_save_tune)

        self.btn_load_tune = QtWidgets.QPushButton("Load Tune (Preview)…")
        toolbar.addWidget(self.btn_load_tune)

        self.btn_apply_prev = QtWidgets.QPushButton("Apply Tune Preview")
        toolbar.addWidget(self.btn_apply_prev)

        self.btn_clear_prev = QtWidgets.QPushButton("Clear Preview")
        toolbar.addWidget(self.btn_clear_prev)

        toolbar.addStretch(1)

        # Table setup
        self.table = QtWidgets.QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Code", "Previous", "Current", "New Value"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setItemDelegateForColumn(4, NumericItemDelegate(self.table))
        layout.addWidget(self.table)

        # Build rows
        self._rebuild_table()

        # Connect signals
        self.btn_apply.clicked.connect(self.apply_selected)
        self.table.installEventFilter(self)
        self.table.cellChanged.connect(self._maybe_auto_send)
        self.btn_show_log.toggled.connect(self._toggle_log)
        self.btn_save_tune.clicked.connect(self.save_tune)
        self.btn_load_tune.clicked.connect(self.load_tune_preview)
        self.btn_apply_prev.clicked.connect(self.apply_all_preview)
        self.btn_clear_prev.clicked.connect(self.clear_preview)

        # Log
        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.hide()
        layout.addWidget(self.log_widget)

    # ---------- internal helpers ----------
    def _all_params(self) -> List[Tuple[str, str, object, str, bool]]:
        """Return base + custom params."""
        return PARAMS_BASE + self._custom_params

    def _rebuild_table(self) -> None:
        """Rebuild table from current parameter list."""
        rows = self._all_params()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))

        for row, (key, ptype, pcode, label, voltage_like) in enumerate(rows):
            # Parameter label
            it = QtWidgets.QTableWidgetItem(label)
            it.setToolTip(PARAM_TOOLTIPS.get(key, ""))
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, it)

            # Code
            code_txt = pcode if ptype == "EDIT" else f"DNC{pcode}"
            it = QtWidgets.QTableWidgetItem(str(code_txt))
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 1, it)

            # Previous, Current
            for col in (2, 3):
                it = QtWidgets.QTableWidgetItem("—")
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                it.setBackground(QtGui.QColor("#e0e0e0"))
                self.table.setItem(row, col, it)

            # New Value
            it = QtWidgets.QTableWidgetItem("")
            it.setBackground(QtGui.QColor("#fff8dc"))
            self.table.setItem(row, 4, it)

        self.table.blockSignals(False)

        # Notify Step Test of updated params
        customs_for_combo = [
            (ptype, pcode, label) for (_k, ptype, pcode, label, _v) in self._custom_params
        ]
        self.custom_params_changed.emit(customs_for_combo)

    def eventFilter(self, widget, event):
        if widget is self.table and event.type() == QtCore.QEvent.KeyPress:
            if (
                event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)
                and self.chk_auto.isChecked()
            ):
                self.apply_selected()
                return True
        return super().eventFilter(widget, event)

    def _maybe_auto_send(self, row: int, col: int) -> None:
        if col == 4 and self.chk_auto.isChecked():
            self.apply_row(row)

    def _toggle_log(self, show: bool) -> None:
        self.log_widget.setVisible(show)
        self.btn_show_log.setText("Hide Change Log" if show else "Show Change Log")

    def _append_log(self, text: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_widget.append(f"[{ts}] {text}")

    # ---------- staging ----------
    def stage_value(self, ptype: str, pcode: str, value: float) -> bool:
        """Stage value in 'New Value' cell for given param."""
        mapping = {(t, str(c)): r for r, (_k, t, c, _l, _v) in enumerate(self._all_params())}
        row = mapping.get((ptype, str(pcode)))
        if row is None:
            return False
        self.table.item(row, 4).setText(str(float(value)))
        self.table.item(row, 4).setBackground(QtGui.QColor("#e6ffe6"))
        return True

    # ---------- add custom ----------
    def _add_custom_editxx(self) -> None:
        """Dialog to add custom EditXX parameter."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add Custom EditXX")
        form = QtWidgets.QFormLayout(dlg)

        spin = QtWidgets.QSpinBox(dlg)
        spin.setRange(1, 999)
        spin.setValue(37)
        lbl = QtWidgets.QLineEdit(dlg)
        lbl.setPlaceholderText("Optional label (e.g., Gate Voltage)")

        form.addRow("Edit number:", spin)
        form.addRow("Label:", lbl)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg
        )
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        n = spin.value()
        edit_code = f"Edit{n}"
        label = lbl.text().strip() or f"{edit_code} (user)"
        voltage_like = (edit_code.lower() == "edit23")
        key = f"user_{edit_code.lower()}"

        # Avoid duplicates
        for (_k, _t, pc, _l, _v) in self._custom_params:
            if _t == "EDIT" and pc == edit_code:
                QtWidgets.QMessageBox.information(
                    self, "Already added", f"{edit_code} is already in the table."
                )
                return

        self._custom_params.append((key, "EDIT", edit_code, label, voltage_like))
        self._rebuild_table()

    # ---------- apply operations ----------
    def apply_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        for r in rows:
            self.apply_row(r)

    def apply_row(self, row: int) -> None:
        all_params = self._all_params()
        key, ptype, pcode, label, voltage_like = all_params[row]
        txt = self.table.item(row, 4).text().strip()
        if not txt:
            return
        value = _to_float(txt)
        if value is None:
            QtWidgets.QMessageBox.warning(self, "Invalid number", f"'{txt}' is not a number.")
            return

        # Current → Previous
        prev_text = self.table.item(row, 3).text()
        self.table.item(row, 2).setText(prev_text)

        # Voltage guard
        if voltage_like and not confirm_high_voltage(self, label, value):
            return

        try:
            if ptype == "EDIT":
                self.dde.send_scanpara(str(pcode), value)
            else:
                self.dde.send_dncpara(int(pcode), value)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "DDE error", str(e))
            return

        # Update Current
        self.table.item(row, 3).setText(str(value))
        self.table.item(row, 3).setBackground(QtGui.QColor("#fff2b3"))
        QtCore.QTimer.singleShot(
            700, lambda: self.table.item(row, 3).setBackground(QtGui.QColor("#e0e0e0"))
        )

        if self.log_widget.isVisible():
            code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
            self._append_log(f"{label} ({code_text}) ← {value}")

    # ---------- tune management ----------
    def _row_lookup_by_code(self) -> Dict[Tuple[str, str], int]:
        return {
            (ptype, str(pcode)): row
            for row, (_key, ptype, pcode, _label, _vlike) in enumerate(self._all_params())
        }

    def _collect_params_snapshot(self) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        for row, (_key, ptype, pcode, label, _vlike) in enumerate(self._all_params()):
            cur_txt = self.table.item(row, 3).text().strip()
            cur_val = _to_float(cur_txt)
            records.append({"ptype": ptype, "pcode": pcode, "label": label, "value": cur_val})
        return {
            "kind": "ncafm_tune",
            "params": records,
            "note": "Values correspond to the CURRENT column in the ncAFM panel.",
        }

    def save_tune(self) -> None:
        # Default filename: YYYYMMDD_HH_MM_SS_NameSensor_tune.json
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        default_name = f"{timestamp}_NameSensor_tune.json"

        # Use last directory if available, else current dir
        start_dir = getattr(self, "_last_dir", "")
        start_path = str(QtCore.QDir(start_dir).filePath(default_name)) if start_dir else default_name

        # Ask for save path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Tune",
            start_path,
            "Tune Files (*.json);;All Files (*)"
        )
        if not path:
            return

        # Remember last used directory
        self._last_dir = str(QtCore.QFileInfo(path).absolutePath())

        # Ask for optional comments
        comments, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Add Comments",
            "Optional notes or comments for this tune:",
            ""
        )
        if not ok:
            comments = ""

        # Build payload
        payload = self._collect_params_snapshot()
        payload["comments"] = comments
        payload["saved_at"] = timestamp

        # Write to file
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "I/O error", f"Could not save:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Saved", f"Tune saved to:\n{path}")


    def load_tune_preview(self) -> None:
        """
        Load JSON and stage values into 'New Value' (no send).
        Accepts:
        1) {"kind":"ncafm_tune","params":[{ptype,pcode,label,value},...], "comments": "..."}
        2) A simple dict: {"amp_ki": 5.0, "amp_kp": 7.0, ...} — base keys only.
        Custom EditXX entries in (1) will be matched by (ptype,pcode) and will
        stage correctly if the custom row exists in the table.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Tune (Preview)", getattr(self, "_last_dir", ""), "Tune Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "I/O error", f"Could not read:\n{e}")
            return

        # If the file has comments, display them
        comments = payload.get("comments")
        if comments:
            QtWidgets.QMessageBox.information(
                self,
                "Tune Comments",
                f"Comments stored with this tune:\n\n{comments}"
            )

        staged = 0
        self.chk_auto.setChecked(False)  # do not auto-send while staging
        lookup = self._row_lookup_by_code()

        def stage_row(row: int, val: float):
            self.table.item(row, 4).setText(str(val))
            self.table.item(row, 4).setBackground(QtGui.QColor("#e6ffe6"))

        # Case 1: original schema (also supports customs if present in table)
        if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], list):
            for rec in payload["params"]:
                ptype = rec.get("ptype")
                pcode = str(rec.get("pcode"))
                val = rec.get("value", None)
                if isinstance(val, (int, float)):
                    row = lookup.get((ptype, pcode))
                    if row is not None:
                        stage_row(row, float(val))
                        staged += 1

        # Case 2: mapping by base keys only (customs don’t have stable keys)
        elif isinstance(payload, dict):
            base_index = {k: i for i, (k, *_r) in enumerate(PARAMS_BASE)}
            for key, val in payload.items():
                if key in base_index and isinstance(val, (int, float)):
                    row = base_index[key]
                    stage_row(row, float(val))
                    staged += 1

        if staged == 0:
            QtWidgets.QMessageBox.information(self, "Nothing staged", "No matching numeric values found.")
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Preview loaded",
                f"Staged {staged} value(s). Review and click 'Apply Tune Preview'."
            )


        def stage_row(row: int, val: float):
            self.table.item(row, 4).setText(str(val))
            self.table.item(row, 4).setBackground(QtGui.QColor("#e6ffe6"))

        if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], list):
            # Schema format
            for rec in payload["params"]:
                ptype = rec.get("ptype")
                pcode = str(rec.get("pcode"))
                val = rec.get("value", None)
                if isinstance(val, (int, float)):
                    row = lookup.get((ptype, pcode))
                    if row is not None:
                        stage_row(row, float(val))
                        staged += 1
        elif isinstance(payload, dict):
            # Simple mapping format
            base_index = {k: i for i, (k, *_r) in enumerate(PARAMS_BASE)}
            for key, val in payload.items():
                if key in base_index and isinstance(val, (int, float)):
                    row = base_index[key]
                    stage_row(row, float(val))
                    staged += 1

        if staged == 0:
            QtWidgets.QMessageBox.information(self, "Nothing staged", "No matching numeric values found.")
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Preview loaded",
                f"Staged {staged} value(s). Review and click 'Apply Tune Preview'.",
            )

    def clear_preview(self) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 4).setText("")
            self.table.item(row, 4).setBackground(QtGui.QColor("#fff8dc"))

    def apply_all_preview(self) -> None:
        rows_to_apply: List[Tuple[int, float]] = []
        for row in range(self.table.rowCount()):
            txt = self.table.item(row, 4).text().strip()
            if txt:
                val = _to_float(txt)
                if val is not None:
                    rows_to_apply.append((row, val))
        if not rows_to_apply:
            QtWidgets.QMessageBox.information(self, "Nothing to apply", "No staged numeric values.")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Apply Tune Preview",
            f"Apply {len(rows_to_apply)} staged value(s)?\n\n"
            "Safety:\n"
            "• Values are interpreted in SXM’s current GUI units.\n"
            f"• Amplitude Ref and Drive over ±{VOLTAGE_LIMIT_ABS} V still prompt individually.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        for row, _ in rows_to_apply:
            self.apply_row(row)
            self.table.item(row, 4).setBackground(QtGui.QColor("#fff8dc"))
