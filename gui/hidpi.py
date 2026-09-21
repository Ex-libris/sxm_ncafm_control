"""
High-DPI awareness for a 4K / scaled Windows display.

Three things go wrong without it (measured, see tests/test_hidpi.py):

* Qt 5 does not scale the interface at all unless ``AA_EnableHighDpiScaling`` is set before the
  ``QApplication`` exists, so pixel sizes (stylesheets, minimum widths, tick lengths) stay tiny.
  Windows scale factors such as 125 % / 150 % are also rounded to 100 % / 200 % unless the rounding
  policy is ``PassThrough``.
* pyqtgraph draws lines with *cosmetic* pens, whose width is in device pixels: a width-2 pen is a
  2-pixel line at any scale, i.e. 1 logical pixel on a 200 % display. Pen widths therefore have to be
  multiplied by the device pixel ratio (``install_pen_scaling``).
* The window was created at a fixed size, whatever the screen.

Set ``SXM_PLOT_LINE_SCALE`` (a number, e.g. 2) to force the line-width multiplier when Qt's detection
is wrong (remote desktop, mixed-DPI monitors).
"""
import os

from PyQt5 import QtCore, QtGui

ENV_LINE_SCALE = "SXM_PLOT_LINE_SCALE"


def enable_high_dpi() -> None:
    """Switch Qt's high-DPI scaling on. Must be called BEFORE the QApplication is created."""
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    policy = getattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy", None)          # Qt >= 5.14
    if policy is not None and "QT_SCALE_FACTOR_ROUNDING_POLICY" not in os.environ:
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(policy.PassThrough)   # 150 % stays 1.5, not 2


def device_pixel_ratio(widget=None) -> float:
    """Device pixels per logical pixel of the widget's screen (primary screen if there is no widget)."""
    screen = None
    if widget is not None and hasattr(widget, "screen"):
        screen = widget.screen()
    if screen is None:
        screen = QtGui.QGuiApplication.primaryScreen()
    return max(1.0, float(screen.devicePixelRatio())) if screen is not None else 1.0


def pen_scale(widget=None) -> float:
    """Multiplier for plot line widths: the environment override, else the device pixel ratio."""
    raw = os.environ.get(ENV_LINE_SCALE, "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return device_pixel_ratio(widget)


def install_pen_scaling() -> None:
    """
    Make every pyqtgraph pen (ours and pyqtgraph's own: curves, axes, grids) scale with the display.
    Idempotent. Pens passed in as ready-made ``QPen`` objects, non-cosmetic pens and ``None`` are left alone.
    """
    from pyqtgraph import functions as fn
    import pyqtgraph as pg

    original = fn.mkPen
    if getattr(original, "_sxm_hidpi", False):
        return

    def mkPen(*args, **kargs):
        pen = original(*args, **kargs)
        if args and isinstance(args[0], QtGui.QPen) and not kargs:
            return pen                                            # already a finished pen: do not scale twice
        if pen.style() == QtCore.Qt.NoPen or not pen.isCosmetic() or pen.widthF() <= 0:
            return pen
        pen.setWidthF(pen.widthF() * pen_scale())
        return pen

    mkPen._sxm_hidpi = True
    mkPen._original = original
    fn.mkPen = mkPen
    pg.mkPen = mkPen


def uninstall_pen_scaling() -> None:
    """Undo :func:`install_pen_scaling` (used by the tests)."""
    from pyqtgraph import functions as fn
    import pyqtgraph as pg

    current = fn.mkPen
    if getattr(current, "_sxm_hidpi", False):
        fn.mkPen = pg.mkPen = current._original


def initial_window_size(widget=None, wanted=(1500, 950), fraction=0.95) -> QtCore.QSize:
    """A comfortable window size that still fits the available area of the screen it is on."""
    screen = widget.screen() if widget is not None and hasattr(widget, "screen") else None
    screen = screen or QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        return QtCore.QSize(*wanted)
    avail = screen.availableGeometry()
    return QtCore.QSize(min(wanted[0], int(avail.width() * fraction)), min(wanted[1], int(avail.height() * fraction)))


def describe_display() -> str:
    """One ASCII line saying what Qt detected, for the console / tooltips."""
    screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        return "Display: unknown"
    g = screen.geometry()
    dpr = float(screen.devicePixelRatio())
    return (f"Display: {g.width()}x{g.height()} logical px, scale x{dpr:g} "
            f"({round(g.width() * dpr)}x{round(g.height() * dpr)} device px), logical DPI {screen.logicalDotsPerInch():.0f}; "
            f"plot line widths x{pen_scale():g}")
