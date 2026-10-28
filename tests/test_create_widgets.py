import numpy as np
from muvis_align.ui.create_widgets import create_section_container
from qtpy.QtWidgets import QApplication


class MinimalInterface:
    def __init__(self):
        self.params = {
            'input_output': {
                'pairing': 'stack',
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
                {'label': 'Stack', 'value': 'stack'},
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
    assert pairing_widget.value == 'stack'
    assert tuple(pairing_widget.choices) == ('default', 'stack')

    assert input_widget.widget_type == 'FileEdit'
    assert input_widget.filter == '*.zarr'
    assert 'directory' in str(input_widget.mode).lower()



def test_a_table_row_selection_calls_the_interfaces_selected_function():
    _ = QApplication.instance() or QApplication([])
    selections = []

    class SelectingInterface(MinimalInterface):
        def get_function(self, function_label):
            return (lambda: selections.append(table.native.currentRow())) if function_label == 'metrics_selected' else None

    interface = SelectingInterface()
    create_section_container('registration', [{'name': 'metrics', 'type': 'table', 'label': ' ',
                                               'section_key': 'display_only', 'default': []}],
                             interface, connect_changed=False, add_button=False)
    table = interface.param_widgets['registration.metrics'].widget
    table.value = {'data': [[1], [2]], 'index': ['a - b', 'b - c'], 'columns': ['quality']}

    table.native.selectRow(1)

    assert selections == [1]


def test_a_table_shows_floats_at_four_decimals():
    _ = QApplication.instance() or QApplication([])
    interface = MinimalInterface()
    create_section_container('registration', [{'name': 'metrics', 'type': 'table', 'label': ' ',
                                               'section_key': 'display_only', 'default': []}],
                             interface, connect_changed=False, add_button=False)
    table = interface.param_widgets['registration.metrics'].widget
    table.value = {'data': [[0.25621754998563356, np.float32(0.5), 'z: 0', None]], 'index': ['summary'],
                   'columns': ['quality', 'ncc', 'position', 'ssim']}
    model = table.native.model()

    assert [model.data(model.index(0, column)) for column in range(4)] == ['0.2562', '0.5000', 'z: 0', '']
