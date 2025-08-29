# gui/common.py

from typing import Optional
from PyQt5 import QtWidgets, QtGui

# ---------------- Parameter registry ----------------
PARAMS_BASE = [
    ("amp_ref", "EDIT", "Edit23", "Amplitude Ref", True), #guarded
    ("amp_ki", "EDIT", "Edit24", "Amplitude Ki", False),
    ("amp_kp", "EDIT", "Edit32", "Amplitude Kp", False),
    ("pll_kp", "EDIT", "Edit27", "PLL Kp", False),
    ("pll_ki", "EDIT", "Edit22", "PLL Ki", False),
    ("used_freq", "DNC", 3, "Used Frequency (f₀)", False),
    ("drive", "DNC", 4, "Drive", True),                    # guarded
]

PARAM_TOOLTIPS = {
    "amp_ref": "Target oscillation amplitude (units follow SXM).",
    "amp_ki": "Amplitude loop integral gain...",
    "amp_kp": "Amplitude loop proportional gain...",
    "pll_kp": "PLL proportional gain...",
    "pll_ki": "PLL integral gain...",
    "used_freq": "PLL used frequency (actual f₀, SXM units).",
    "drive": "Drive amplitude to sustain oscillation (SXM units).",
}

# Voltage safety limit
VOLTAGE_LIMIT_ABS = 10.0


def _to_float(text: str) -> Optional[float]:
    """
    Convert text to a floating-point number.

    This function tries to interpret the input text as a number (for example, '3.14').
    If the conversion fails (for example, if the text is 'abc'), it returns None.

    This is useful for safely handling user input in scientific applications,
    where parameters must be numeric.

    Parameters
    ----------
    text : str
        The input text to convert.

    Returns
    -------
    float or None
        The numeric value if conversion succeeds, or None if it fails.
    """
    try:
        return float(text)
    except Exception:
        return None


def confirm_high_voltage(parent: QtWidgets.QWidget, label: str, value: float) -> bool:
    """
    Display a warning dialog if a voltage-like parameter exceeds ±10 V.

    In many scientific instruments, applying a voltage above a certain threshold
    can damage equipment or produce unreliable results. This function checks if
    the value is above the safe limit (±10 V). If so, it asks the user to confirm
    before proceeding.

    Parameters
    ----------
    parent : QtWidgets.QWidget
        The parent window for the dialog (usually the main application window).
    label : str
        The name of the parameter being set (for display).
    value : float
        The value the user wants to set.

    Returns
    -------
    bool
        True if the user confirms, False if they cancel.
    """
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
    """
    Table cell editor that only allows numeric input.

    When editing values in a table (for example, parameter lists), this class
    ensures that only numbers can be entered. This prevents accidental entry of
    invalid data (such as letters or symbols) in scientific workflows.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.
    lo : float, optional
        Minimum allowed value (default: -1e12).
    hi : float, optional
        Maximum allowed value (default: 1e12).
    decimals : int, optional
        Number of decimal places allowed (default: 9).
    """

    def __init__(self, parent=None, lo=-1e12, hi=1e12, decimals=9):
        super().__init__(parent)
        self._lo, self._hi, self._dec = lo, hi, decimals

    def createEditor(self, parent, option, index):
        """
        Create a line editor for table cells that only accepts numbers.

        This is automatically called by the table when a cell is edited.

        Parameters
        ----------
        parent : QWidget
            The parent widget.
        option : QStyleOptionViewItem
            Style options for the editor.
        index : QModelIndex
            The index of the cell being edited.

        Returns
        -------
        QLineEdit
            An editor widget with numeric validation.
        """
        editor = QtWidgets.QLineEdit(parent)
        validator = QtGui.QDoubleValidator(self._lo, self._hi, self._dec, editor)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor
    
def offline_message(component: str, error: Exception, mock_name: str):
    """
    Print a clear, consistent offline-mode message.

    Parameters
    ----------
    component : str
        Which subsystem failed ("Microscope driver", "DDE connection", etc.)
    error : Exception
        The caught exception object (or string)
    mock_name : str
        What we will fall back to ("mock driver", "MockDDEClient", etc.)
    """
    err_msg = str(error)
    print(
        f"\n[OFFLINE MODE] {component} is not available.\n"
        f"→ Cause: {err_msg}\n"
        f"→ Action: Switching to {mock_name} (no hardware connected).\n"
    )
