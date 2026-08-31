from qtpy.QtWidgets import QHeaderView

from muvis_align.ui.bilayers_util import to_magicgui_choices


class ParamWidget:
    def __init__(self, param_name, widget, interface, to_str=False):
        self.param_name = param_name
        self.widget = widget
        self.interface = interface
        self.to_str = to_str

    def get_value(self):
        return self.widget.get_value()

    def get_native_item(self, rowi, coli):
        return self.widget.native.item(rowi, coli)

    def set_value(self, value, choices=None):
        if choices is not None:
            self.set_choices(choices)
        self.widget.set_value(value)

    def set_choices(self, choices):
        self.widget.choices = to_magicgui_choices(choices)

    def value_changed(self, value):
        if isinstance(value, dict):
            value0 = self.get_value()
            if isinstance(value0, dict):
                value = update_dict_value(value0, value)
        elif self.to_str:
            # pathlib always stringifies with backslashes on Windows regardless of the input
            # separators (e.g. after a file dialog selection or FileEdit's own internal
            # Path(...).absolute() call) - normalise to forward slashes here so both the
            # widget's displayed text and the stored/persisted param stay consistent
            value = str(value).replace('\\', '/')
            line_edit = getattr(self.widget, 'line_edit', None)
            if line_edit is not None and line_edit.value != value:
                line_edit.value = value
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
