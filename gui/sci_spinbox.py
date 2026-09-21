"""
A QDoubleSpinBox for values that span many orders of magnitude (SXM's Kp / Ki, Q, ...).

It shows large and small numbers in scientific notation ('2.5e8' instead of '250000000.0000'), trims
trailing zeros, and accepts scientific notation (and a decimal comma) as input. Range, decimals,
prefix / suffix and stepping are the ordinary QDoubleSpinBox ones.
"""
import re

from PyQt5 import QtGui, QtWidgets

from ..common import format_number

_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d*)?([eE][+-]?\d*)?$")     # a number, or the start of one ('2e', '-', '.')


class SciDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None, sci_above: float = 1e6, sci_below: float = 1e-3, sig: int = 15, plain: bool = False):
        """``plain=True``: ordinary fixed-decimals display, never e-notation (frequencies such as f0 = 25132.457 Hz).
        Scientific notation is still accepted as input."""
        super().__init__(parent)
        self.sci_above, self.sci_below, self.sig, self.plain = sci_above, sci_below, sig, plain

    def _bare(self, text: str) -> str:
        t = text.strip()
        prefix, suffix = self.prefix().strip(), self.suffix().strip()
        if prefix and t.startswith(prefix):
            t = t[len(prefix):]
        if suffix and t.endswith(suffix):
            t = t[:-len(suffix)]
        return t.strip().replace(",", ".")

    def textFromValue(self, value: float) -> str:
        if self.plain:
            return super().textFromValue(value)
        return format_number(value, self.sig, self.sci_above, self.sci_below)

    def stepBy(self, steps: int):
        """Arrow keys / buttons: a fixed step of 1 means nothing at 2e9, so values shown in e-notation move by 10 %."""
        v = self.value()
        if not self.plain and v != 0 and (abs(v) >= self.sci_above or abs(v) < self.sci_below):
            self.setValue(v * 1.1 ** steps)
        else:
            super().stepBy(steps)

    def valueFromText(self, text: str) -> float:
        try:
            return float(self._bare(text))
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        t = self._bare(text)
        if t == "":
            return QtGui.QValidator.Intermediate, text, pos
        if not _NUMBER.match(t):
            return QtGui.QValidator.Invalid, text, pos
        try:
            v = float(t)
        except ValueError:                                  # '2e', '2e-', '-': still being typed
            return QtGui.QValidator.Intermediate, text, pos
        if self.minimum() <= v <= self.maximum():
            return QtGui.QValidator.Acceptable, text, pos
        return QtGui.QValidator.Intermediate, text, pos
