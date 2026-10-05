"""Save a pyqtgraph scene as PNG with the instrument settings underneath (caption lines or a column legend)."""
from typing import List, Optional, Sequence, Tuple

import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtGui

LegendColumns = Sequence[Tuple[str, Sequence[Tuple[str, str]]]]


def _font(plot_width: int) -> QtGui.QFont:
    font = QtGui.QFont("Consolas")
    font.setStyleHint(QtGui.QFont.Monospace)
    font.setPixelSize(max(12, plot_width // 90))
    return font


def save_scene_png(scene, path: str, caption_lines: Sequence[str] = (), width: int = 1200,
                   legend: Optional[Tuple[LegendColumns, str]] = None) -> None:
    """
    Render ``scene`` at ``width`` px and write it to ``path``. Underneath: ``legend`` (``(columns, footer)``
    from ``metadata.Metadata.legend``) if given, else ``caption_lines`` as a text band.
    """
    exporter = pg.exporters.ImageExporter(scene)
    exporter.parameters()["width"] = width
    plot = exporter.export(toBytes=True)
    if legend is not None and (legend[0] or legend[1]):
        out = _with_legend(plot, *legend)
    elif caption_lines:
        out = _with_caption(plot, caption_lines)
    else:
        out = plot
    out.save(path)


def _with_caption(plot: QtGui.QImage, caption_lines: Sequence[str]) -> QtGui.QImage:
    font = _font(plot.width())
    metrics = QtGui.QFontMetrics(font)
    pad = metrics.height() // 2
    band = pad * 2 + metrics.lineSpacing() * len(caption_lines)
    out = QtGui.QImage(plot.width(), plot.height() + band, QtGui.QImage.Format_ARGB32)
    out.fill(QtGui.QColor(QtCore.Qt.white))   # QColor: a bare Qt.white is taken as a pixel value (transparent)
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
    return out


def _with_legend(plot: QtGui.QImage, columns: LegendColumns, footer: str) -> QtGui.QImage:
    """A boxed legend: one column per group (title, then ``label  value`` rows), wrapped to the image width."""
    font = _font(plot.width())
    font.setPixelSize(max(11, font.pixelSize() - 1))
    bold = QtGui.QFont(font)
    bold.setBold(True)
    fm, fb = QtGui.QFontMetrics(font), QtGui.QFontMetrics(bold)
    line = fm.lineSpacing()
    pad = fm.height() // 2
    gap = fm.horizontalAdvance("    ")
    avail = plot.width() - 4 * pad                           # inside the box

    # column geometry: label width, value width, total
    geo = []
    for title, rows in columns:
        lw = max((fm.horizontalAdvance(l) for l, _v in rows), default=0)
        vw = max((fm.horizontalAdvance(v) for _l, v in rows), default=0)
        w = max(fb.horizontalAdvance(title), lw + fm.horizontalAdvance("  ") + vw)
        geo.append((title, rows, lw, min(w, avail)))

    # greedy wrap into rows of columns
    bands: List[List[tuple]] = [[]]
    used = 0
    for g in geo:
        if bands[-1] and used + gap + g[3] > avail:
            bands.append([])
            used = 0
        used += (gap if bands[-1] else 0) + g[3]
        bands[-1].append(g)
    band_h = [line * (1 + max((len(g[1]) for g in b), default=0)) for b in bands]
    box_h = pad + sum(band_h) + pad * (len(bands) - 1) + pad
    footer_h = line if footer else 0
    total = pad + box_h + (pad // 2 + footer_h if footer else 0) + pad

    out = QtGui.QImage(plot.width(), plot.height() + total, QtGui.QImage.Format_ARGB32)
    out.fill(QtGui.QColor(QtCore.Qt.white))   # QColor: a bare Qt.white is taken as a pixel value (transparent)
    p = QtGui.QPainter(out)
    try:
        p.drawImage(0, 0, plot)
        box = QtCore.QRect(pad, plot.height() + pad, plot.width() - 2 * pad, box_h)
        p.setPen(QtGui.QPen(QtGui.QColor(190, 190, 190), 1))
        p.setBrush(QtGui.QColor(250, 250, 250))
        p.drawRect(box)
        y0 = box.top() + pad
        for b, h in zip(bands, band_h):
            x = box.left() + pad
            for title, rows, lw, w in b:
                y = y0 + fb.ascent()
                p.setFont(bold)
                p.setPen(QtGui.QColor(20, 20, 20))
                p.drawText(x, y, fb.elidedText(title, QtCore.Qt.ElideRight, w))
                p.setFont(font)
                for label, value in rows:
                    y += line
                    p.setPen(QtGui.QColor(110, 110, 110))
                    p.drawText(x, y, label)
                    p.setPen(QtGui.QColor(20, 20, 20))
                    vx = x + lw + fm.horizontalAdvance("  ")
                    p.drawText(vx, y, fm.elidedText(value, QtCore.Qt.ElideRight, max(0, x + w - vx)))
                x += w + gap
            y0 += h + pad
        if footer:
            p.setFont(font)
            p.setPen(QtGui.QColor(110, 110, 110))
            p.drawText(pad, box.bottom() + pad // 2 + fm.ascent(),
                       fm.elidedText(footer, QtCore.Qt.ElideRight, plot.width() - 2 * pad))
    finally:
        p.end()
    return out
