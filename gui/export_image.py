"""Save a pyqtgraph scene as PNG with a caption of the instrument settings underneath."""
from typing import Sequence

import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtGui


def save_scene_png(scene, path: str, caption_lines: Sequence[str] = (), width: int = 1200) -> None:
    """Render ``scene`` at ``width`` px and write it to ``path``, with ``caption_lines`` as a text band below."""
    exporter = pg.exporters.ImageExporter(scene)
    exporter.parameters()["width"] = width
    plot = exporter.export(toBytes=True)
    if not caption_lines:
        plot.save(path)
        return
    font = QtGui.QFont("Consolas")
    font.setStyleHint(QtGui.QFont.Monospace)
    font.setPixelSize(max(12, plot.width() // 90))
    metrics = QtGui.QFontMetrics(font)
    pad = metrics.height() // 2
    band = pad * 2 + metrics.lineSpacing() * len(caption_lines)
    out = QtGui.QImage(plot.width(), plot.height() + band, QtGui.QImage.Format_ARGB32)
    out.fill(QtCore.Qt.white)
    p = QtGui.QPainter(out)
    try:
        p.drawImage(0, 0, plot)
        p.setPen(QtGui.QColor(30, 30, 30))
        p.setFont(font)
        y = plot.height() + pad + metrics.ascent()
        for line in caption_lines:
            p.drawText(pad, y, metrics.elidedText(line, QtCore.Qt.ElideRight, out.width() - 2 * pad))
            y += metrics.lineSpacing()
    finally:
        p.end()
    out.save(path)
