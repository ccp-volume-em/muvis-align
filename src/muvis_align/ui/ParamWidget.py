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
        if self.to_str:
            value = str(value)
        self.interface.change_param(self.param_name, value)

    def set_table_column_resize_mode(self, mode=QHeaderView.Stretch):
        self.widget.native.horizontalHeader().setSectionResizeMode(mode)
