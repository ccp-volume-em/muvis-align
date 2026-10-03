from types import SimpleNamespace
from unittest.mock import MagicMock

from muvis_align.ui.ParamWidget import ParamWidget


class _FakeLineEdit:
    def __init__(self, value=''):
        self.value = value


class _FakeFileEdit:
    def __init__(self, value=''):
        self.line_edit = _FakeLineEdit(value)


def test_value_changed_normalizes_backslashes_for_file_type_params():
    """A file dialog (or FileEdit's own internal Path(...).absolute() call) always reports
    backslash-separated paths on Windows - value_changed() must normalise to forward slashes
    for the value handed to the interface, which is what gets stored and persisted.

    The widget's own text is deliberately left as typed: rewriting it live erases a trailing
    '/' mid-typing, and the display re-syncs from the stored value when Process runs instead
    (see Interface.update_input_output_path).
    """
    interface = SimpleNamespace(change_param=MagicMock())
    widget = _FakeFileEdit(value='C:\\proj\\data\\input')
    param_widget = ParamWidget('input_output.input_path', widget, interface, to_str=True)

    param_widget.value_changed('C:\\proj\\data\\input')

    interface.change_param.assert_called_once_with(
        'input_output.input_path', 'C:/proj/data/input'
    )
    assert widget.line_edit.value == 'C:\\proj\\data\\input'


def test_value_changed_never_writes_back_to_the_line_edit():
    """Writing the line edit from here would re-trigger Qt's textChanged signal, looping
    straight back into value_changed() - the widget owns its own text."""
    interface = SimpleNamespace(change_param=MagicMock())
    widget = _FakeFileEdit(value='data/input')
    param_widget = ParamWidget('input_output.input_path', widget, interface, to_str=True)

    param_widget.value_changed('data/other')

    interface.change_param.assert_called_once_with('input_output.input_path', 'data/other')
    assert widget.line_edit.value == 'data/input'


def test_value_changed_ignores_non_file_type_params():
    interface = SimpleNamespace(change_param=MagicMock())
    param_widget = ParamWidget('registration.method', MagicMock(), interface, to_str=False)

    param_widget.value_changed('phase')

    interface.change_param.assert_called_once_with('registration.method', 'phase')


def test_value_changed_keeps_a_trailing_separator_being_typed():
    """The regression that removing the live line-edit rewrite fixed: normalising separators
    must not also trim the '/' the user has just typed, on its way to being a subdirectory."""
    interface = SimpleNamespace(change_param=MagicMock())
    widget = _FakeFileEdit(value='data/input/')
    param_widget = ParamWidget('input_output.input_path', widget, interface, to_str=True)

    param_widget.value_changed('data/input/')

    interface.change_param.assert_called_once_with('input_output.input_path', 'data/input/')
    assert widget.line_edit.value == 'data/input/'


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
