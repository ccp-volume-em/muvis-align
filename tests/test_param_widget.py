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
    both for the value handed to the interface and for the widget's own displayed text."""
    interface = SimpleNamespace(change_param=MagicMock())
    widget = _FakeFileEdit(value='C:\\proj\\data\\input')
    param_widget = ParamWidget('input_output.input_path', widget, interface, to_str=True)

    param_widget.value_changed('C:\\proj\\data\\input')

    interface.change_param.assert_called_once_with(
        'input_output.input_path', 'C:/proj/data/input'
    )
    assert widget.line_edit.value == 'C:/proj/data/input'


def test_value_changed_skips_line_edit_write_when_already_normalized():
    """Re-writing the line edit with an unchanged value would re-trigger Qt's textChanged
    signal, looping back into value_changed() forever - skip the write once the displayed
    text already matches the normalised value."""
    interface = SimpleNamespace(change_param=MagicMock())
    widget = _FakeFileEdit(value='data/input')
    param_widget = ParamWidget('input_output.input_path', widget, interface, to_str=True)

    param_widget.value_changed('data/input')

    assert widget.line_edit.value == 'data/input'


def test_value_changed_ignores_non_file_type_params():
    interface = SimpleNamespace(change_param=MagicMock())
    param_widget = ParamWidget('registration.method', MagicMock(), interface, to_str=False)

    param_widget.value_changed('phase')

    interface.change_param.assert_called_once_with('registration.method', 'phase')
