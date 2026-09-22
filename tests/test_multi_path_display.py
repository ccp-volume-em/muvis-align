"""An input path may name several globs at once (e.g. 'tiles/**/*.tif, overviews/**/*.tif').

That value is a valid project setting - reg.init() splits it via eval_path() - so the path
widgets must show it as stored. They used to be left untouched (and so blank) whenever the
value was not a single path, which made a loaded project look like it had no input folder.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import muvis_align.ui.Interface as interface_module
from muvis_align.util import path_param_to_text

Interface = interface_module.Interface


class _FakeLineEdit:
    def __init__(self):
        self.value = ''


class _FakeParamWidget:
    def __init__(self):
        self.widget = SimpleNamespace(line_edit=_FakeLineEdit())

    @property
    def text(self):
        return self.widget.line_edit.value


def _interface_with_paths(input_path, output_path, monkeypatch):
    interface = Interface.__new__(Interface)
    interface.verbose = False
    interface.params_path = None
    interface.params = {'input_output': {'input_path': input_path, 'output_path': output_path}}
    interface.param_widgets = {
        'input_output.input_path': _FakeParamWidget(),
        'input_output.output_path': _FakeParamWidget(),
    }
    monkeypatch.setattr(interface_module, 'init_logging', MagicMock())
    return interface


def test_multiple_globs_are_shown_in_the_path_widget(monkeypatch):
    interface = _interface_with_paths('tiles/**/*.tif, overviews/**/*.tif', 'out/', monkeypatch)

    interface.update_input_output_path()

    assert interface.param_widgets['input_output.input_path'].text == \
        'tiles/**/*.tif, overviews/**/*.tif'
    assert interface.param_widgets['input_output.output_path'].text == 'out/'


def test_path_stored_as_a_yaml_list_is_shown_as_one_line(monkeypatch):
    interface = _interface_with_paths(['tiles/**/*.tif', 'overviews/**/*.tif'], 'out/', monkeypatch)

    interface.update_input_output_path()

    assert interface.param_widgets['input_output.input_path'].text == \
        'tiles/**/*.tif, overviews/**/*.tif'


def test_path_param_to_text_forms():
    assert path_param_to_text('a/*.tif') == 'a/*.tif'
    assert path_param_to_text(['a/*.tif', ' b/*.tif']) == 'a/*.tif, b/*.tif'
    assert path_param_to_text(None) == ''
