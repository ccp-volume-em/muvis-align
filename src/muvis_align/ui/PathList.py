import os

from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QListWidget, QListWidgetItem,
                            QPushButton, QVBoxLayout, QWidget)

from muvis_align.util import (path_param_to_list, pattern_base_dir, relativize_to_project_dir,
                              resolve_to_project_dir)


class QPathList(QWidget):
    """An editable list of paths or globs, each row typed in place or added with a file or folder dialog."""

    pathsChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        # the directory stored paths are relative to; set by the owner, as it changes with the project
        self.base_dir = lambda: None
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.list_widget.itemChanged.connect(self._item_edited)
        buttons = QHBoxLayout()
        for label, function in (('Add files...', self._add_files), ('Add folder...', self._add_folder),
                                ('Remove', self._remove), ('Up', lambda: self._move(-1)),
                                ('Down', lambda: self._move(1))):
            button = QPushButton(label)
            button.clicked.connect(function)
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list_widget)
        layout.addLayout(buttons)

    def paths(self):
        return [self.list_widget.item(row).text() for row in range(self.list_widget.count())]

    def setPaths(self, paths):
        # not emitted: only a user's edit is a change to store
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for path in paths:
            self._append(path)
        self.list_widget.blockSignals(False)

    def _append(self, path):
        item = QListWidgetItem(path)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.list_widget.addItem(item)

    def _item_edited(self, item):
        if not item.text().strip():
            self.list_widget.takeItem(self.list_widget.row(item))
        self.pathsChanged.emit(self.paths())

    def _start_dir(self):
        base_dir = self.base_dir()
        item = self.list_widget.currentItem()
        if item is None and self.list_widget.count():
            item = self.list_widget.item(self.list_widget.count() - 1)
        if item is not None:
            path = resolve_to_project_dir(item.text(), base_dir)
            directory = path if os.path.isdir(path) else pattern_base_dir(path)
            if os.path.isdir(directory):
                return directory
        return base_dir or ''

    def _add_paths(self, paths):
        for path in paths:
            # shown as stored: relative to the project, as the dialogs report absolute paths
            self._append(relativize_to_project_dir(path, self.base_dir()))
        if paths:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)
            self.pathsChanged.emit(self.paths())

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add input files', self._start_dir())
        self._add_paths(paths)

    def _add_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Add input folder', self._start_dir())
        self._add_paths([path] if path else [])

    def _remove(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            self.pathsChanged.emit(self.paths())

    def _move(self, step):
        row = self.list_widget.currentRow()
        target = row + step
        if row >= 0 and 0 <= target < self.list_widget.count():
            self.list_widget.blockSignals(True)
            self.list_widget.insertItem(target, self.list_widget.takeItem(row))
            self.list_widget.blockSignals(False)
            self.list_widget.setCurrentRow(target)
            self.pathsChanged.emit(self.paths())


class _QPathListWidget(QBaseValueWidget):
    _qwidget: QPathList

    def __init__(self, **kwargs):
        super().__init__(QPathList, 'paths', 'setPaths', 'pathsChanged', **kwargs)

    def _pre_set_hook(self, value):
        return path_param_to_list(value)


class PathList(ValueWidget):
    """A magicgui widget for several input paths. Value is a list of path or glob strings."""

    def __init__(self, **kwargs):
        # create_widget(widget_type="...PathList") leaves its string 'widget_type' in kwargs
        kwargs.pop('widget_type', None)
        super().__init__(widget_type=_QPathListWidget, **kwargs)
