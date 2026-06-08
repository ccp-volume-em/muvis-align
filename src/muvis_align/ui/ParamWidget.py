class ParamWidget:
    def __init__(self, param_name, widget, interface, to_str):
        self.param_name = param_name
        self.widget = widget
        self.interface = interface
        self.to_str = to_str

    def get_value(self):
        return self.widget.get_value()

    def set_value(self, value, choices=None):
        if choices is not None:
            self.widget.choices = choices
        self.widget.set_value(value)

    def value_changed(self, value):
        if self.to_str:
            value = str(value)
        self.interface.change_param(self.param_name, value)
