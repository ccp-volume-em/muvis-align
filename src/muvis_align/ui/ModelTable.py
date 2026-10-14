from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from qtpy.QtWidgets import QTableView


class ArrayTableModel(QAbstractTableModel):
    """Rows of values with row and column labels, read-only: text and colour made only for the cells shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows, self.row_labels, self.column_labels = [], [], []
        self.background = None
        self._backgrounds = {}

    def set_rows(self, rows, row_labels, column_labels):
        self.beginResetModel()
        self.rows, self.row_labels, self.column_labels = list(rows), list(row_labels), list(column_labels)
        self._backgrounds = {}
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.column_labels)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == Qt.DisplayRole:
            return '' if value is None else str(value)
        if role == Qt.BackgroundRole and self.background is not None and value is not None:
            key = (index.row(), index.column())
            if key not in self._backgrounds:
                self._backgrounds[key] = self.background(value)
            return self._backgrounds[key]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        labels = self.column_labels if orientation == Qt.Horizontal else self.row_labels
        return str(labels[section]) if section < len(labels) else None


class QModelTableView(QTableView):
    """A table view over ArrayTableModel, with the QTableWidget calls the plugin uses."""

    valueChanged = Signal(object)
    itemSelectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table_model = ArrayTableModel(self)
        self.setModel(self.table_model)
        # a bound method, not a lambda holding the view: Qt keeps a lambda alive, and through it everything connected
        self.selectionModel().selectionChanged.connect(self._selection_changed)

    def _selection_changed(self, *_):
        self.itemSelectionChanged.emit()

    def value(self):
        model = self.table_model
        return {'data': model.rows, 'index': model.row_labels, 'columns': model.column_labels}

    def setValue(self, value):
        # (rows, row labels, column labels), or magicgui Table's {'data', 'index', 'columns'}
        if isinstance(value, dict):
            value = (value.get('data', []), value.get('index', []), value.get('columns', []))
        rows, row_labels, column_labels = value if value else ([], [], [])
        self.table_model.set_rows(rows, row_labels, column_labels)
        self.valueChanged.emit(self.value())

    def set_background(self, background):
        """`background(value)` gives a cell's QColor, asked only for the cells shown."""
        self.table_model.background = background
        self.table_model._backgrounds = {}

    def currentRow(self):
        return self.currentIndex().row()

    def row_label(self, row):
        labels = self.table_model.row_labels
        return labels[row] if 0 <= row < len(labels) else None


class _QModelTableWidget(QBaseValueWidget):
    _qwidget: QModelTableView

    def __init__(self, **kwargs):
        super().__init__(QModelTableView, 'value', 'setValue', 'valueChanged', **kwargs)


class ModelTable(ValueWidget):
    """A read-only magicgui table for many rows: value is {'data', 'index', 'columns'}, as magicgui's Table."""

    def __init__(self, **kwargs):
        # create_widget(widget_type="...ModelTable") leaves its string 'widget_type' in kwargs
        kwargs.pop('widget_type', None)
        kwargs.pop('options', None)
        super().__init__(widget_type=_QModelTableWidget, **kwargs)
