from src.muvis_align.ui.create_widgets import create_section_container
from qtpy.QtWidgets import QApplication


class MinimalInterface:
    def __init__(self):
        self.params = {
            'input_output': {
                'pairing': 'overlay',
                'input_path': 'data/*.zarr',
            }
        }
        self.param_widgets = {}

    def get_function(self, function_label):
        return None


def test_create_section_container_end_to_end_dropdown_and_fileedit():
    _ = QApplication.instance() or QApplication([])
    interface = MinimalInterface()
    section_template = [
        {
            'name': 'pairing',
            'type': 'dropdown',
            'label': 'Pairing',
            'section_key': 'parameters',
            'options': [
                {'label': 'Default', 'value': 'default'},
                {'label': 'Overlay', 'value': 'overlay'},
            ],
            'default': 'default',
        },
        {
            'name': 'input_path',
            'type': 'image',
            'label': 'Input data',
            'section_key': 'inputs',
            'default': 'data/*.zarr',
            'file_count': 'multiple',
        },
    ]

    container = create_section_container(
        'input_output',
        section_template,
        interface,
        connect_changed=False,
        add_button=False,
    )

    pairing_widget = interface.param_widgets['input_output.pairing'].widget
    input_widget = interface.param_widgets['input_output.input_path'].widget

    assert pairing_widget.widget_type in {'Dropdown', 'ComboBox'}
    assert pairing_widget.value == 'overlay'
    assert tuple(pairing_widget.choices) == ('default', 'overlay')

    assert input_widget.widget_type == 'FileEdit'
    assert input_widget.filter == '*.zarr'
    assert 'directory' in str(input_widget.mode).lower()

