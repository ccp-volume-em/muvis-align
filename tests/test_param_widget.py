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
