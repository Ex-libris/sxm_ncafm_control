# gui/common.py
from typing import Optional
from PyQt5 import QtWidgets, QtGui

# ---------------- Parameter registry ----------------
PARAMS_BASE = [
    ("amp_ki", "EDIT", "Edit24", "Amplitude Ki", False),
    ("amp_kp", "EDIT", "Edit32", "Amplitude Kp", False),
    ("pll_kp", "EDIT", "Edit27", "PLL Kp", False),
    ("pll_ki", "EDIT", "Edit22", "PLL Ki", False),
    ("amp_ref", "EDIT", "Edit23", "Amplitude Ref", True),   # guarded
    ("used_freq", "DNC", 3, "Used Frequency (f₀)", False),
    ("drive", "DNC", 4, "Drive", True),                    # guarded
]

PARAM_TOOLTIPS = {
    "amp_ki": "Amplitude loop integral gain...",
    "amp_kp": "Amplitude loop proportional gain...",
    "pll_kp": "PLL proportional gain...",
    "pll_ki": "PLL integral gain...",
    "amp_ref": "Target oscillation amplitude (units follow SXM).",
    "used_freq": "PLL used frequency (actual f₀, SXM units).",
    "drive": "Drive amplitude to sustain oscillation (SXM units).",
}

# Voltage safety limit
VOLTAGE_LIMIT_ABS = 10.0


def _to_float(text: str) -> Optional[float]:
    """Safely convert text to float or return None if invalid."""
    try:
        return float(text)
    except Exception:
        return None


def confirm_high_voltage(parent: QtWidgets.QWidget, label: str, value: float) -> bool:
    """Ask user for confirmation if exceeding ±10 V on voltage-like channels."""
    if abs(value) <= VOLTAGE_LIMIT_ABS:
        return True
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle("Confirm High Voltage")
    box.setText(
        f"You are about to set <b>{label}</b> to {value}.\n\n"
        f"⚠ Do not exceed ±{VOLTAGE_LIMIT_ABS} V unless a divider is installed.\n\n"
        "Proceed?"
    )
    box.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
    box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
    return box.exec_() == QtWidgets.QMessageBox.Ok


class NumericItemDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate that only allows numeric input in QTableWidget cells."""

    def __init__(self, parent=None, lo=-1e12, hi=1e12, decimals=9):
        super().__init__(parent)
        self._lo, self._hi, self._dec = lo, hi, decimals

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        validator = QtGui.QDoubleValidator(self._lo, self._hi, self._dec, editor)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor
