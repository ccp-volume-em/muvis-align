from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog, QPushButton


def color_to_tuple(color):
    if not isinstance(color, QColor) or not color.isValid():
        return None
    return color.redF(), color.greenF(), color.blueF()


def tuple_to_color(value):
    if isinstance(value, QColor):
        return value
    if isinstance(value, str):
        return QColor(value)
    if value is None:
        return QColor()
    r, g, b = (list(value) + [0, 0, 0])[:3]
    return QColor.fromRgbF(r, g, b)


class QColorSwatchButton(QPushButton):
    """A small flat button showing a single color; clicking it opens QColorDialog."""

    colorChanged = Signal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor()
        self.setFlat(True)
        self.clicked.connect(self._pick_color)
        self._update_style()

    def color(self):
        return self._color

    def setColor(self, color):
        color = QColor(color)
        if color != self._color:
            self._color = color
            self._update_style()
            self.colorChanged.emit(self._color)

    def _update_style(self):
        r, g, b = round(self._color.redF(), 3), round(self._color.greenF(), 3), round(self._color.blueF(), 3)
        self.setText(f'{r}, {g}, {b}')
        text_color = 'black' if self._color.lightnessF() > 0.5 else 'white'
        self.setStyleSheet(
            f'background-color: {self._color.name()}; color: {text_color}; border: 1px solid gray;')

    def _pick_color(self):
        # parent on the top-level window, not this button, so the dialog doesn't inherit
        # this button's own background-color stylesheet
        dialog = QColorDialog(self._color, self.window())
        if dialog.exec():
            color = dialog.currentColor()
            if color.isValid():
                self.setColor(color)


class _QColorSwatchWidget(QBaseValueWidget):
    """magicgui backend adapting QColorSwatchButton to the ValueWidgetProtocol."""

    _qwidget: QColorSwatchButton

    def __init__(self, **kwargs):
        super().__init__(QColorSwatchButton, 'color', 'setColor', 'colorChanged', **kwargs)

    def _pre_set_hook(self, value):
        return tuple_to_color(value)

    def _post_get_hook(self, value):
        return color_to_tuple(value)

    def _mgui_bind_change_callback(self, callback) -> None:
        # the native colorChanged signal carries a raw QColor - convert it to this widget's
        # (r, g, b) tuple value before it reaches ValueWidget's changed signal, so listeners
        # connected to `.changed` see the same type as `.value`
        self._qwidget.colorChanged.connect(lambda color: callback(color_to_tuple(color)))


class MagicColorPicker(ValueWidget):
    """A magicgui color-picker widget: a single swatch button that opens QColorDialog.

    Value is an (r, g, b) float tuple with each component in [0, 1].
    """

    def __init__(self, **kwargs):
        # when instantiated via create_widget(widget_type="...MagicColorPicker") magicgui
        # leaves its own (string) 'widget_type' in kwargs alongside the one set below
        kwargs.pop('widget_type', None)
        super().__init__(widget_type=_QColorSwatchWidget, **kwargs)
