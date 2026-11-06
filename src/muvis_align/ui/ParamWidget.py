from magicgui.widgets import Table
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHeaderView

from muvis_align.ui.bilayers_util import to_magicgui_choices
from muvis_align.util import to_posix_path


class ParamWidget:
    def __init__(self, param_name, widget, interface, to_str=False, number_type=None):
        self.param_name = param_name
        self.widget = widget
        self.interface = interface
        self.to_str = to_str
        self.number_type = number_type
        self.is_invalid_number = False

    def get_value(self):
        return self.widget.get_value()

    def set_value(self, value, choices=None):
        if choices is not None:
            self.set_choices(choices)
        if self.number_type is not None:
            self.widget.set_value('' if value is None else str(value))
        elif isinstance(self.widget, Table):
            self._set_table_value(value)
        else:
            self.widget.set_value(value)

    def _set_table_value(self, value):
        # signals blocked: the header view handles a signal per row label, quadratically; told once at the end
        model = self.widget.native.model()
        model.blockSignals(True)
        try:
            self.widget.set_value(value)
        finally:
            model.blockSignals(False)
        model.layoutChanged.emit()
        for orientation, count in ((Qt.Vertical, model.rowCount()), (Qt.Horizontal, model.columnCount())):
            if count:
                model.headerDataChanged.emit(orientation, 0, count - 1)

    def set_choices(self, choices):
        self.widget.choices = to_magicgui_choices(choices)

    def value_changed(self, value):
        if self.number_type is not None:
            text = str(value).strip()
            try:
                value = None if not text else (int(text) if self.number_type == 'integer' else float(text))
            except ValueError:
                # warned once until valid again, as this runs on every keystroke; the last valid value is kept
                if not self.is_invalid_number:
                    show_warning(f'{self.param_name}: {text!r} is not a{"n integer" if self.number_type == "integer" else " number"}')
                self.is_invalid_number = True
                return
            self.is_invalid_number = False
        elif isinstance(value, dict):
            value0 = self.get_value()
            if isinstance(value0, dict):
                value = update_dict_value(value0, value)
        elif isinstance(value, list):
            value = [to_posix_path(str(item)) for item in value]
        elif self.to_str:
            # the widget's text is left as typed: rewriting it live would erase a trailing '/' mid-typing
            value = to_posix_path(str(value))
        self.interface.change_param(self.param_name, value)

    def set_table_column_resize_mode(self, mode=QHeaderView.Stretch):
        self.widget.native.horizontalHeader().setSectionResizeMode(mode)


def update_dict_value(old_value, new_value):
    columns = old_value.get('columns', [])
    data = old_value.get('data', [[]])
    data[new_value['row']][new_value['column']] = new_value['data']
    dict_of_lists = create_dict_of_lists(data, columns)
    return dict_of_lists


def create_dict_of_lists(data, columns):
    return {column: [x[columni] for x in data] for columni, column in enumerate(columns)}
