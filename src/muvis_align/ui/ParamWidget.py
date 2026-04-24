class ParamWidget:
    def __init__(self, param_name, widget, interface, to_str):
        self.param_name = param_name
        self.widget = widget
        self.interface = interface
        self.to_str = to_str

    def set_value(self, value):
        self.widget.set_value(value)

    def value_changed(self, value):
        if self.to_str:
            value = str(value)
        self.interface.change_param(self.param_name, value)

    def create_choices(self, choices):
        # When setting choices with a dict, the dict must have keys 'choices' (Iterable),
        # and 'key' (callable that takes each value in `choices` and returns a string
        self.choices = choices
        return {
            'choices': choices.keys(),
            'key': self.get_choice_label
        }

    def get_choice_label(self, choice):
        return self.choices[choice]