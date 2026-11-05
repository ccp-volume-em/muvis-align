from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from muvis_align.ui.ParamWidget import ParamWidget


class _FakeLineEdit:
    def __init__(self, value=''):
        self.value = value


class _FakeFileEdit:
    def __init__(self, value=''):
        self.line_edit = _FakeLineEdit(value)


@pytest.mark.parametrize(
    ("typed", "stored"),
    [
        ('C:\\proj\\data\\input', 'C:/proj/data/input'),
        ('data/other', 'data/other'),
        # trimming the '/' just typed would stop it becoming a subdirectory
        ('data/input/', 'data/input/'),
    ],
    ids=["backslashes", "plain", "trailing-separator"],
)
def test_value_changed_normalizes_a_path_but_leaves_the_line_edit_as_typed(typed, stored):
    """The widget owns its text: writing it from here re-triggers textChanged, looping back into
    value_changed(), and the display re-syncs from the stored value when Process runs."""
    interface = SimpleNamespace(change_param=MagicMock())
    widget = _FakeFileEdit(value='as typed')
    param_widget = ParamWidget('input_output.input_path', widget, interface, to_str=True)

    param_widget.value_changed(typed)

    interface.change_param.assert_called_once_with('input_output.input_path', stored)
    assert widget.line_edit.value == 'as typed'


def test_value_changed_ignores_non_file_type_params():
    interface = SimpleNamespace(change_param=MagicMock())
    param_widget = ParamWidget('registration.method', MagicMock(), interface, to_str=False)

    param_widget.value_changed('phase')

    interface.change_param.assert_called_once_with('registration.method', 'phase')


@pytest.mark.parametrize('number_type, typed, stored', [
    ('float', '1.5', 1.5), ('float', ' 2 ', 2.0), ('float', '', None), ('integer', '4', 4), ('integer', '', None),
])
def test_a_number_text_field_stores_a_number_or_none(number_type, typed, stored):
    interface = SimpleNamespace(change_param=MagicMock())
    param_widget = ParamWidget('pre_processing.gaussian_sigma', MagicMock(), interface, number_type=number_type)

    param_widget.value_changed(typed)

    interface.change_param.assert_called_once_with('pre_processing.gaussian_sigma', stored)
    assert type(interface.change_param.call_args.args[1]) is type(stored)


def test_invalid_number_text_warns_once_and_keeps_the_last_value(monkeypatch):
    import muvis_align.ui.ParamWidget as param_widget_module
    warnings = []
    monkeypatch.setattr(param_widget_module, 'show_warning', warnings.append)
    interface = SimpleNamespace(change_param=MagicMock())
    param_widget = ParamWidget('registration.n_parallel_pairwise_regs', MagicMock(), interface, number_type='integer')

    for typed in ('1.5', 'x', '8'):
        param_widget.value_changed(typed)

    assert len(warnings) == 1
    interface.change_param.assert_called_once_with('registration.n_parallel_pairwise_regs', 8)


def test_a_table_filled_with_its_signals_held_matches_a_plain_fill(qapp):
    from magicgui.widgets import Table
    from qtpy.QtCore import Qt

    value = ([[0.9, None], [0.5, 0.4], [0.1, 0.2]], ['summary', 'a - b', 'b - c'], ['quality', 'ncc'])
    plain, held = Table(), Table()
    plain.set_value(value)

    ParamWidget('registration.metrics_table', held, interface=None).set_value(value)

    assert held.value == plain.value
    for orientation, count in ((Qt.Vertical, 3), (Qt.Horizontal, 2)):
        assert ([held.native.model().headerData(index, orientation) for index in range(count)]
                == [plain.native.model().headerData(index, orientation) for index in range(count)])
    assert held.native.verticalHeader().count() == 3


def test_a_cell_background_is_asked_with_its_column_label(qapp):
    from qtpy.QtCore import Qt
    from muvis_align.ui.ModelTable import QModelTableView

    table = QModelTableView()
    table.setValue(([[0.9, 0.1]], ['summary'], ['quality', 'ncc']))
    asked = []
    table.set_background(lambda value, column_label: asked.append((value, column_label)) or 'colour')

    model = table.table_model
    backgrounds = [model.data(model.index(0, column), Qt.BackgroundRole) for column in range(2)]

    assert backgrounds == ['colour', 'colour'] and asked == [(0.9, 'quality'), (0.1, 'ncc')]


def test_selecting_a_row_by_label_makes_it_current_before_the_selection_signal(qapp):
    from muvis_align.ui.ModelTable import QModelTableView

    table = QModelTableView()
    table.setValue(([[0.9], [0.5], [0.1]], ['summary', 'a - b', 'b - c'], ['quality']))
    rows_on_signal = []
    table.itemSelectionChanged.connect(lambda: rows_on_signal.append(table.currentRow()))

    table.select_row_label('b - c')

    assert rows_on_signal == [2]
    assert [index.row() for index in table.selectionModel().selectedRows()] == [2]
