from pathlib import Path

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


def test_create_section_container_end_to_end_dropdown_and_path_list():
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

    assert type(input_widget).__name__ == 'PathList'
    assert input_widget.value == ['data/*.zarr']


def test_a_path_list_adds_to_moves_and_removes_from_its_paths(monkeypatch, tmp_path):
    from qtpy.QtWidgets import QFileDialog
    from muvis_align.ui.PathList import PathList

    _ = QApplication.instance() or QApplication([])
    widget = PathList(value='tiles/*.tif, overviews')
    widget.native.base_dir = lambda: str(tmp_path)
    changes = []
    widget.changed.connect(changes.append)
    start_dirs = []

    def get_existing_directory(parent, caption, start_dir):
        start_dirs.append(start_dir)
        return (tmp_path / 'channel2').as_posix()

    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', get_existing_directory)
    widget.native._add_folder()
    widget.native._move(-1)
    widget.native.list_widget.setCurrentRow(0)
    widget.native._remove()

    # nothing selected: the dialog opens where the last path is, resolved against the project folder
    assert [Path(start_dir) for start_dir in start_dirs] == [tmp_path]
    # the dialog's absolute path is shown relative to the project
    assert changes == [['tiles/*.tif', 'overviews', 'channel2'],
                       ['tiles/*.tif', 'channel2', 'overviews'],
                       ['channel2', 'overviews']]



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


def test_a_file_dialog_starts_from_the_project_and_shows_its_choice_relative(monkeypatch, tmp_path):
    import magicgui.backends._qtpy as qt_backend

    _ = QApplication.instance() or QApplication([])
    interface = MinimalInterface()
    interface.params['input_output']['output_path'] = 'results/run1'
    interface.get_project_dir = lambda: str(tmp_path)
    (tmp_path / 'results').mkdir()
    start_paths = []

    def show_file_dialog(mode, caption=None, start_path=None, filter=None):
        start_paths.append(start_path)
        return str(tmp_path / 'stitched')

    monkeypatch.setattr(qt_backend, 'show_file_dialog', show_file_dialog)
    create_section_container('input_output', [{'name': 'output_path', 'type': 'image', 'label': 'Output data',
                                               'section_key': 'outputs', 'default': 'output',
                                               'file_count': 'multiple'}],
                             interface, connect_changed=False, add_button=False)
    widget = interface.param_widgets['input_output.output_path'].widget

    widget._on_choose_clicked()

    # results/run1 does not exist yet: its nearest existing folder
    assert [Path(start_path) for start_path in start_paths] == [tmp_path / 'results']
    assert widget.line_edit.value == 'stitched'
