"""Tests of ui/Interface.py; the test project files are loaded by test_project_config_loading."""

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch, call
import pytest
import yaml
import numpy as np
from qtpy.QtWidgets import QMessageBox

from muvis_align.util import print_hbytes
from muvis_align._widget import MainWidget
from muvis_align.logging import close_fault_log
from muvis_align.ui.Interface import Interface, ViewMode
import muvis_align.ui.Interface as interface_module
from muvis_align.MVSRegistration import RegState
from tests.data_builders import DATA_DIR, TIFF_FILES, ZARR_FILES, FakeBar, make_phase_factory, \
    prepared_registration, recording_phase_factory


@pytest.fixture(autouse=True)
def suppress_completion_dialogs():
    """Keep workflow completion messages from blocking test execution."""
    with patch(
        'muvis_align.ui.Interface.QMessageBox.information'
    ) as mock_information:
        yield mock_information


@pytest.fixture(autouse=True)
def suppress_confirmation_dialogs():
    """Answer No to a confirmation a test did not patch itself, so none opens a real modal dialog."""
    with patch(
        'muvis_align.ui.Interface.QMessageBox.question',
        return_value=QMessageBox.No,
    ) as mock_question:
        yield mock_question


def get_project_configs():
    """Discover all muvis_align_project*.yml files in tests directory."""
    test_dir = Path(__file__).parent
    configs = sorted(test_dir.glob('muvis_align_project*.yml'))
    if not configs:
        raise FileNotFoundError(
            f"No project config files found in {test_dir}. "
            "Expected muvis_align_project*.yml files."
        )
    return configs


def template_options(name):
    """The dropdown values the plugin's project template offers for a parameter."""
    template_path = Path(__file__).parent.parent / 'src' / 'muvis_align' / 'ui' / 'project_template.yaml'
    with open(template_path, 'r') as file:
        template = yaml.safe_load(file)
    parameter = next(parameter for parameter in template['parameters'] if parameter['name'] == name)
    return [option['value'] for option in parameter['options']]


@pytest.fixture(params=get_project_configs(), ids=lambda path: path.name)
def project_config(request):
    """Fixture that provides path to each discovered project configuration file."""
    config_path = request.param
    assert config_path.exists(), f"Project config not found: {config_path}"
    return config_path


@pytest.fixture
def config_data(project_config):
    """Load and parse project configuration YAML."""
    with open(project_config, 'r') as file:
        return yaml.safe_load(file)


def test_project_config_loading(make_napari_viewer, project_config, tmp_path):
    viewer = make_napari_viewer()
    with patch('muvis_align._widget.ViewerWidget'), patch.object(viewer.window, 'add_dock_widget'):
        interface = MainWidget(viewer).interface
    config_copy = tmp_path / project_config.name
    config_copy.write_text(project_config.read_text())

    interface.project_path(str(config_copy))

    assert interface.params_path == str(config_copy)
    assert {'registration', 'fusion', 'input_output', 'pre_processing'} <= set(interface.params)
    # init_logging() opened a log file in tmp_path: Windows cannot delete it while open
    for handler in logging.getLogger().handlers[:]:
        handler.close()
        logging.getLogger().removeHandler(handler)
    close_fault_log()


def test_project_configs_match_the_template(config_data):
    """The test projects only hold values the plugin's template offers."""
    from muvis_align.util import parse_scale, pixel_size_to_um

    registration = config_data['registration']
    assert registration['method'] in template_options('method')
    assert registration['pairing'] in ['orthogonal', 'all']
    assert registration['transform_type'] in ['rigid', 'affine']
    assert registration['operation'] == 'register'
    assert isinstance(registration['max_keypoints'], int)
    assert isinstance(registration['ransac_iterations'], int)

    fusion = config_data['fusion']
    assert fusion['method'] in ['average', 'max', 'min']
    assert fusion['spacing'] in ['mean', 'min']
    assert isinstance(fusion['tile_size'], str)
    assert fusion['ome_version'] in ['0.4', '0.5']

    input_output = config_data['input_output']
    assert isinstance(input_output['input_path'], str)
    assert isinstance(input_output['output_path'], str)
    assert isinstance(input_output['overwrite'], bool)

    # a factor (a number, or its text as a text field saves it) or a pixel size such as '10um'
    scale = parse_scale(config_data['pre_processing']['scale'])
    assert scale > 0 if isinstance(scale, (int, float)) else pixel_size_to_um(scale) > 0


def test_interface_reset(make_napari_viewer):
    with patch('muvis_align._widget.ViewerWidget'):
        interface = Interface(make_napari_viewer(), MagicMock(), MagicMock(), MagicMock())
    interface.source_metadata = {'rotation': 1}
    interface.view_mode = ViewMode.PAIRS
    interface._preview_overlap_cache = 'cached'

    interface.reset()

    assert interface.source_metadata == {}
    assert interface.view_mode is None
    assert interface.extra_metadata == {}
    assert interface.output_channels == []
    assert interface._preview_overlap_cache is None
    assert interface.reg.state is RegState.UNINIT


def test_update_registered_refreshes_the_tables_and_the_view(make_napari_viewer):
    with patch('muvis_align._widget.ViewerWidget'):
        interface = Interface(make_napari_viewer(), MagicMock(), MagicMock(), MagicMock())

    with patch.object(interface, 'populate_coordinate_systems') as coordinate_systems,             patch.object(interface, 'populate_metadata_table') as metadata_table,             patch.object(interface, 'populate_metrics_table') as metrics_table,             patch.object(interface, 'update_views') as update_views:
        interface.update_registered()

    assert coordinate_systems.called and metadata_table.called and metrics_table.called
    update_views.assert_called_once_with(transform_key=None, progress_factory=None)


@pytest.fixture
def bare_interface():
    interface = Interface.__new__(Interface)
    interface.reg = MagicMock()
    interface.verbose = False
    interface.enable_plugin_widget = None
    interface.extra_metadata = {}
    interface.param_widgets = {}
    interface.reg.file_labels = ["image-0"]
    interface.view_msims = [
        SimpleNamespace(dims=("z", "y", "x"), sizes={"z": 2})
    ]
    return interface


@pytest.mark.parametrize(
    ("param", "value", "stored"),
    [
        ("registration.method", "phase", "phase"),
        # a file dialog reports an absolute path: stored relative to the project, to stay portable
        ("input_output.input_path", "{project}/data/input", "data/input"),
        ("registration.method", "{project}/not-a-path-param", "{project_posix}/not-a-path-param"),
    ],
    ids=["value", "path-relativized", "other-param-as-is"],
)
def test_change_param_stores_the_value_and_writes(bare_interface, tmp_path, param, value, stored):
    bare_interface.params = {}
    bare_interface.write_params = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")

    bare_interface.change_param(param, value.format(project=str(tmp_path)))

    section, name = param.split(".")
    assert bare_interface.params == {section: {name: stored.format(project_posix=tmp_path.as_posix())}}
    bare_interface.write_params.assert_called_once_with()


def test_get_project_dir_returns_none_before_project_loaded(bare_interface):
    assert not hasattr(bare_interface, "params_path")
    assert bare_interface.get_project_dir() is None


@pytest.mark.parametrize(
    ("input_path", "shown"),
    [
        ("data/input", "data/input"),
        # several globs are one valid path value, and were left blank
        ("tiles/**/*.tif, overviews/**/*.tif", "tiles/**/*.tif, overviews/**/*.tif"),
        (["tiles/**/*.tif", "overviews/**/*.tif"], "tiles/**/*.tif, overviews/**/*.tif"),
    ],
    ids=["relative", "globs", "yaml-list"],
)
def test_update_input_output_path_displays_stored_value_as_is(bare_interface, tmp_path, input_path, shown):
    """Shown as stored (relative to the project directory): FileEdit.set_value() would force it
    absolute, so the text goes straight into the widget's inner line edit."""
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {"input_path": input_path, "output_path": "results"}
    }
    input_widget = MagicMock()
    output_widget = MagicMock()
    bare_interface.param_widgets = {
        "input_output.input_path": input_widget,
        "input_output.output_path": output_widget,
    }

    bare_interface.update_input_output_path()

    assert input_widget.widget.line_edit.value == shown
    assert output_widget.widget.line_edit.value == "results"
    input_widget.set_value.assert_not_called()
    output_widget.set_value.assert_not_called()


def test_update_input_output_path_falls_back_to_set_value_without_line_edit(bare_interface, tmp_path):
    """A widget without an inner line_edit (e.g. not a FileEdit) falls back to the normal
    set_value() path instead of erroring."""
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {"input_path": "data/input", "output_path": ""}
    }
    input_widget = MagicMock()
    input_widget.widget = SimpleNamespace()  # no line_edit attribute
    bare_interface.param_widgets = {
        "input_output.input_path": input_widget,
        "input_output.output_path": MagicMock(),
    }

    bare_interface.update_input_output_path()

    input_widget.set_value.assert_called_once_with("data/input")


def test_input_output_process_resolves_relative_paths_before_reg_init(
    bare_interface, tmp_path, mocked_activity_contexts
):
    """input_path/output_path are stored relative to the project directory - MVSRegistration
    resolves a relative path against the process's cwd (not the project dir), so
    input_output_process() must resolve them to absolute paths before calling reg.init()."""
    bare_interface.viewer = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {
            "input_path": "data/input",
            "output_path": "results",
            "overwrite": True,
        },
        "registration": {"pairing": "stack"},
    }
    bare_interface.reg.is_initialised.return_value = False
    bare_interface.need_source_reinit = False
    bare_interface.reg.init.return_value = True
    bare_interface.reg.middle_section_indices.return_value = None
    bare_interface.update_metadata_source = MagicMock(return_value=True)
    bare_interface.populate_image_selection = MagicMock()
    bare_interface._load_saved_progress = MagicMock()
    bare_interface._show_loaded_project = MagicMock()

    bare_interface.input_output_process()

    expected_input = str(tmp_path / "data" / "input").replace('\\', '/')
    expected_output = str(tmp_path / "results").replace('\\', '/') + '/'
    bare_interface.reg.init.assert_called_once_with(
        input_path=expected_input,
        output_path=expected_output,
        overwrite=True,
        # is_stack is read from the pairing, so reg has to be initialised with it
        pairing="stack",
        verbose=False,
    )


def test_input_output_process_cancelled_while_reading_sources_reads_them_again_next_time(
    bare_interface, tmp_path, mocked_activity_contexts
):
    from muvis_align.util import OperationCancelled
    bare_interface.viewer = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {"input_path": "data/input", "output_path": "results", "overwrite": True},
        "registration": {"pairing": "orthogonal"},
    }
    bare_interface.reg.is_initialised.return_value = False
    bare_interface.need_source_reinit = False
    bare_interface.reg.init.return_value = True
    bare_interface.reg.middle_section_indices.return_value = None
    bare_interface.update_metadata_source = MagicMock(side_effect=OperationCancelled('Cancelled'))
    bare_interface._show_loaded_project = MagicMock()

    with patch.object(interface_module, 'show_info') as show_info:
        bare_interface.input_output_process()

    show_info.assert_called_once_with('Cancelled')
    bare_interface._show_loaded_project.assert_not_called()
    assert bare_interface.need_source_reinit
    assert bare_interface.reg.state is RegState.UNINIT


def test_get_all_widgets_excludes_widgets_on_disabled_tabs(bare_interface):
    """A widget on a currently disabled tab always reads .enabled == False, so if
    modify_pair_registration snapshotted and restored it via get_all_widgets, it would stay
    disabled even after its tab becomes enabled later. get_all_widgets must exclude it instead."""
    bare_interface.param_widgets = {
        "registration.method": SimpleNamespace(widget="reg-widget"),
        "fusion.method": SimpleNamespace(widget="fusion-widget"),
    }
    bare_interface.is_tab_enabled = lambda section_id: section_id != "fusion"

    all_widgets = bare_interface.get_all_widgets()

    assert all_widgets == {"registration.method": "reg-widget"}


@pytest.mark.parametrize("store", [True, False], ids=["store", "discard"])
def test_modify_pair_registration_disables_other_tabs_and_restores_them(
    bare_interface, monkeypatch, store
):
    """Entering pair-modification mode disables every other tab (registration's own widgets are
    disabled via get_all_widgets) and leaving restores them, saving the pair only when asked to."""
    import xarray as xr

    bare_interface.view_mode = None
    bare_interface.viewer = MagicMock()
    bare_interface.template = {
        "input_output": [], "pre_processing": [], "registration": [], "fusion": []
    }
    tab_states = {
        "project": True, "input_output": True, "pre_processing": True,
        "registration": True, "fusion": False,
    }
    bare_interface.is_tab_enabled = lambda section_id: tab_states[section_id]
    bare_interface.enable_tab = MagicMock(
        side_effect=lambda section_id, enabled: tab_states.__setitem__(section_id, enabled)
    )
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.param_widgets = {
        "registration.reg_preview_image1": SimpleNamespace(get_value=lambda: "image-0"),
        "registration.reg_preview_image2": SimpleNamespace(get_value=lambda: "image-0"),
    }
    bare_interface.reg.file_labels = ["image-0"]
    bare_interface.reg.register_msims = ["msim"]
    transform = xr.DataArray(
        np.eye(3).reshape(1, 3, 3), dims=["t", "x_in", "x_out"], coords={"t": [0]}
    )
    bbox = xr.DataArray([[1, 2], [3, 4]], dims=["x_in", "x_out"])
    monkeypatch.setattr(
        interface_module.nx, "get_edge_attributes",
        lambda _graph, key: {(0, 0): bbox if key == "bbox" else transform}
    )
    monkeypatch.setattr(interface_module.nx, "set_edge_attributes", MagicMock())
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_image = MagicMock()
    bare_interface.update_pair_metrics = MagicMock()
    _stub_source_affines(monkeypatch, {"msim": np.eye(3)})

    bare_interface.modify_pair_registration()

    disabled_ids = [call.args[0] for call in bare_interface.enable_tab.call_args_list]
    assert set(disabled_ids) == {"project", "input_output", "pre_processing", "fusion"}
    assert tab_states == {
        "project": False, "input_output": False, "pre_processing": False,
        "registration": True, "fusion": False,
    }

    bare_interface.enable_tab.reset_mock()
    bare_interface.update_registered = MagicMock()
    bare_interface.calc_mod_pair_transform = MagicMock(return_value=transform.sel(t=0))
    reply = interface_module.QMessageBox.Yes if store else interface_module.QMessageBox.No
    monkeypatch.setattr(interface_module.QMessageBox, "question", lambda *_: reply)

    bare_interface.modify_pair_registration()

    assert tab_states == {
        "project": True, "input_output": True, "pre_processing": True,
        "registration": True, "fusion": False,
    }
    assert bare_interface.reg.save_pair_mappings.called is store
    if store:
        # a bbox without a 't' dim is saved as it is
        assert bare_interface.reg.save_pair_mappings.call_args.args[2] == {(0, 0): [[1, 2], [3, 4]]}


def _stub_source_affines(monkeypatch, affines):
    """Each msim (a plain key here) reads back its own source transform."""
    monkeypatch.setattr(interface_module.msi_utils, "get_transform_from_msim",
                        lambda msim, transform_key=None: affines[msim])


def _rotation(degrees, translation=(0, 0)):
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), -np.sin(angle), translation[0]],
                     [np.sin(angle), np.cos(angle), translation[1]],
                     [0, 0, 1]])


def test_modify_pair_registration_shows_and_reads_back_through_source_transforms(
    bare_interface, monkeypatch
):
    """Each layer carries its source's own (rotated) transform, the fixed one the pair transform
    on top; reading the layers back gives the pair transform alone, unchanged or after a move."""
    import xarray as xr

    _arm_pair_modify_entry(bare_interface, monkeypatch)
    bare_interface.param_widgets["registration.reg_preview_image2"] = SimpleNamespace(
        get_value=lambda: "image-1")
    bare_interface.reg.file_labels = ["image-0", "image-1"]
    bare_interface.reg.register_msims = ["fixed", "moving"]
    source_affines = {"fixed": _rotation(30, (5, 7)), "moving": _rotation(-20, (40, 3))}
    _stub_source_affines(monkeypatch, source_affines)
    pair_transform = _rotation(2, (1.5, -0.5))
    monkeypatch.setattr(interface_module.nx, "get_edge_attributes", lambda *_: {(0, 1): xr.DataArray(
        pair_transform.reshape(1, 3, 3), dims=["t", "x_in", "x_out"], coords={"t": [0]})})

    bare_interface.modify_pair_registration()

    layer_affines = [call.args[3] for call in bare_interface._napari_view_add_image.call_args_list]
    np.testing.assert_allclose(layer_affines[0], pair_transform @ source_affines["fixed"])
    np.testing.assert_allclose(layer_affines[1], source_affines["moving"])

    def read_back(affines):
        bare_interface.viewer.layers = [SimpleNamespace(affine=SimpleNamespace(affine_matrix=affine))
                                        for affine in affines]
        return np.asarray(bare_interface.calc_mod_pair_transform())

    np.testing.assert_allclose(read_back(layer_affines), pair_transform, atol=1e-12)
    # moving the moving layer by a shift is the pair transform moved by the opposite shift
    shift = _rotation(0, (3, 4))
    np.testing.assert_allclose(read_back([layer_affines[0], shift @ layer_affines[1]]),
                               np.linalg.inv(shift) @ pair_transform, atol=1e-12)


def _arm_pair_modify_entry(bare_interface, monkeypatch):
    """Set a bare_interface up so modify_pair_registration() enters pair-modification mode."""
    import xarray as xr

    bare_interface.view_mode = None
    bare_interface.viewer = MagicMock()
    bare_interface.template = {"input_output": [], "registration": [], "fusion": []}
    tab_states = {
        "project": True, "input_output": True, "registration": True, "fusion": True
    }
    bare_interface.is_tab_enabled = lambda section_id: tab_states[section_id]
    bare_interface.enable_tab = MagicMock(
        side_effect=lambda section_id, enabled: tab_states.__setitem__(section_id, enabled)
    )
    widget = SimpleNamespace(enabled=True)
    bare_interface.get_all_widgets = MagicMock(
        return_value={"registration.metrics": widget}
    )
    bare_interface.param_widgets = {
        "registration.reg_preview_image1": SimpleNamespace(get_value=lambda: "image-0"),
        "registration.reg_preview_image2": SimpleNamespace(get_value=lambda: "image-0"),
    }
    bare_interface.reg.file_labels = ["image-0"]
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_image = MagicMock()
    bare_interface.update_pair_metrics = MagicMock()
    transform = xr.DataArray(
        np.eye(3).reshape(1, 3, 3), dims=["t", "x_in", "x_out"], coords={"t": [0]}
    )
    monkeypatch.setattr(
        interface_module.nx, "get_edge_attributes", lambda *_: {(0, 0): transform}
    )
    return widget, tab_states


def test_modify_pair_registration_restores_state_when_pre_processing_fails(
    bare_interface, monkeypatch
):
    """Bailing out of entering pair-modification mode has to undo the disabling applied on the
    way in - leaving it in place would strand the user with every widget and tab dead."""
    widget, tab_states = _arm_pair_modify_entry(bare_interface, monkeypatch)
    bare_interface.reg.register_msims = []
    bare_interface.run_pre_processing = MagicMock(return_value=None)

    bare_interface.modify_pair_registration()

    assert widget.enabled is True
    assert tab_states == {
        "project": True, "input_output": True, "registration": True, "fusion": True
    }
    assert bare_interface.view_mode is ViewMode.OVERVIEW


def test_modify_pair_registration_restores_state_when_entering_raises(
    bare_interface, monkeypatch
):
    """The same applies to a failure part-way through building the pair view."""
    widget, tab_states = _arm_pair_modify_entry(bare_interface, monkeypatch)
    bare_interface.reg.register_msims = ["msim"]
    _stub_source_affines(monkeypatch, {"msim": np.eye(3)})
    bare_interface._napari_view_add_image = MagicMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError):
        bare_interface.modify_pair_registration()

    assert widget.enabled is True
    assert tab_states == {
        "project": True, "input_output": True, "registration": True, "fusion": True
    }
    assert bare_interface.view_mode is ViewMode.OVERVIEW


def test_tab_changed_clears_feature_view_and_stops_timer(bare_interface):
    bare_interface.viewer = MagicMock()
    bare_interface.view_mode = ViewMode.FEATURES
    bare_interface.pair_metrics_timer = MagicMock()
    bare_interface._clear_napari_view = MagicMock()

    bare_interface.tab_changed("fusion")

    bare_interface._clear_napari_view.assert_called_once_with(
        bare_interface.viewer
    )
    bare_interface.pair_metrics_timer.stop.assert_called_once_with()
    assert bare_interface.view_mode is None


@pytest.mark.parametrize(
    ("method_name", "expected", "cleared"),
    [
        ("source_position_z", {"position": {"z": 2.5}}, {"position": {}}),
        ("source_position_y", {"position": {"y": 2.5}}, {"position": {}}),
        ("source_position_x", {"position": {"x": 2.5}}, {"position": {}}),
        ("source_scale_z", {"scale": {"z": 2.5}}, {"scale": {}}),
        ("source_scale_y", {"scale": {"y": 2.5}}, {"scale": {}}),
        ("source_scale_x", {"scale": {"x": 2.5}}, {"scale": {}}),
        ("source_rotation", {"rotation": 2.5}, {}),
    ],
)
def test_source_metadata_setters(bare_interface, method_name, expected, cleared):
    bare_interface.source_metadata = {}

    getattr(bare_interface, method_name)(2.5)
    assert bare_interface.source_metadata == expected

    # emptying the field drops the override rather than keeping the last value
    bare_interface.need_source_reinit = False
    getattr(bare_interface, method_name)('')
    assert bare_interface.source_metadata == cleared
    assert bare_interface.need_source_reinit is True


@pytest.mark.parametrize("exists", [True, False], ids=["existing", "new"])
def test_project_path_handles_existing_and_new_projects(
    bare_interface, monkeypatch, exists
):
    bare_interface.template = {"template": True}
    bare_interface.reset = MagicMock()
    bare_interface.update_widgets = MagicMock()
    bare_interface.write_params = MagicMock()
    bare_interface.update_input_output_path = MagicMock()
    bare_interface.resolve_output_settings = MagicMock()
    monkeypatch.setattr(interface_module.os.path, "exists", lambda _: exists)
    monkeypatch.setattr(
        interface_module,
        "get_template_params",
        lambda _: {"input_output": {}},
    )
    monkeypatch.setattr(
        interface_module, "read_params", lambda _: {"registration": {}}
    )
    monkeypatch.setattr(
        interface_module,
        "update_params",
        lambda defaults, loaded: defaults | loaded,
    )

    bare_interface.project_path("project.yml")

    bare_interface.reset.assert_called_once_with()
    assert bare_interface.params_path == "project.yml"
    # for both: it shows the stored relative paths, which update_widgets() skips
    bare_interface.update_input_output_path.assert_called_once_with()
    bare_interface.resolve_output_settings.assert_called_once_with()
    if exists:
        bare_interface.update_widgets.assert_called_once_with()
        bare_interface.write_params.assert_not_called()
    else:
        bare_interface.write_params.assert_called_once_with()
        bare_interface.update_widgets.assert_not_called()


@pytest.mark.parametrize(
    "previous_norm, reply, expect_question, expect_norm, expect_saved",
    [
        (False, None, False, False, True),
        (True, QMessageBox.Yes, True, True, True),
        (True, QMessageBox.Discard, True, False, False),
    ],
    ids=["unchanged", "changed-reload", "changed-discard"],
)
def test_resolve_output_settings_reloads_or_discards_on_changed_source_metadata(
    bare_interface, tmp_path, previous_norm, reply, expect_question, expect_norm, expect_saved
):
    from muvis_align.file.project_yaml import read_params, write_params
    from muvis_align.MVSRegistration import MVSRegistration
    output = tmp_path / "output"
    output.mkdir()
    for name in ("pair_mappings.json", "mappings.json", "metrics.json"):
        (output / name).write_text("{}")
    (output / "registered.ome.zarr").mkdir()
    bare_interface.reg = MVSRegistration()
    bare_interface.update_widgets = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {"output_path": "output", "normalise_rotated_positions": False,
                         "source_position_x": "fn[-2]*24"},
        "registration": {"operation": "register"},
    }
    write_params(bare_interface.params_path, bare_interface.params)
    previous = {"input_output": dict(bare_interface.params["input_output"], normalise_rotated_positions=previous_norm),
                "registration": {"operation": "register"}}
    write_params(str(output / "project.yml"), previous)

    with patch.object(interface_module.QMessageBox, "question", return_value=reply) as question:
        bare_interface.copy_params_to_output()

    assert question.called is expect_question
    assert read_params(bare_interface.params_path)["input_output"]["normalise_rotated_positions"] is expect_norm
    assert (output / "mappings.json").exists() is expect_saved
    assert (output / "registered.ome.zarr").exists() is expect_saved
    # either way the settings now match the output, so the question is not asked again
    assert read_params(str(output / "project.yml"))["input_output"] == \
        read_params(bare_interface.params_path)["input_output"]


@pytest.mark.parametrize("discarded, initialised, expect_read", [
    (True, True, True), (True, False, False), (False, True, False)],
    ids=["discarded", "discarded-before-open", "kept"])
def test_copy_params_to_output_reads_sources_again_once_registration_is_discarded(
    bare_interface, discarded, initialised, expect_read
):
    bare_interface.resolve_output_settings = MagicMock(return_value=discarded)
    bare_interface.reg.is_initialised.return_value = initialised
    bare_interface._input_output_process = MagicMock()

    bare_interface.copy_params_to_output()

    assert bare_interface._input_output_process.called is expect_read


def test_populate_choices_and_image_selection(bare_interface):
    channel_widget = MagicMock()
    coordinate_widget = MagicMock()
    image1_widget = MagicMock()
    image2_widget = MagicMock()
    bare_interface.param_widgets = {
        "registration.channel": channel_widget,
        "input_output.coordinate_system": coordinate_widget,
        "registration.reg_preview_image1": image1_widget,
        "registration.reg_preview_image2": image2_widget,
    }
    bare_interface.reg.sources = [
        SimpleNamespace(
            get_channels=lambda: [{"label": "red"}, {"label": "green"}]
        )
    ]
    bare_interface.reg.file_labels = ["b", "a", "c"]
    bare_interface.reg.positions = [{"z": 0, "y": 5, "x": 0}, {"z": 0, "y": 0, "x": 0}, {"z": 1, "y": 0, "x": 0}]

    bare_interface.populate_channels()
    bare_interface.populate_coordinate_systems(
        ["source_metadata", "registered"]
    )
    bare_interface.populate_image_selection()

    assert channel_widget.set_choices.call_args.args[0] == {
        "red": "red",
        "green": "green",
    }
    assert coordinate_widget.set_choices.call_args.args[0] == {
        "source_metadata": "Source metadata",
        "registered": "Registered",
    }
    # sorted by position (z, y, x), as the metadata and metrics tables are, not by file order
    image1_widget.set_value.assert_called_once_with("a", choices=["a", "b", "c"])
    image2_widget.set_value.assert_called_once_with("b", choices=["a", "b", "c"])


@pytest.mark.parametrize(
        ("transforms", "expected"),
        [
            (["source_metadata"], "source_metadata"),
            (["source_metadata", "transform"], "transform"),
            (["registered", "transform"], "registered"),
            ([], None),
        ],
)
def test_get_best_transform_key(
    bare_interface, monkeypatch, transforms, expected
):
    bare_interface.reg.reg_transform_key = "registered"
    bare_interface.reg.source_transform_key = "source_metadata"
    monkeypatch.setattr(
        interface_module, "get_transforms", lambda _: transforms
    )

    assert bare_interface.get_best_transform_key() == expected


def test_update_views_adds_enabled_preview_layers(
    bare_interface, monkeypatch
):
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"z": 2})]
    bare_interface.reg.positions = [{"z": 0}]
    bare_interface.reg.fileset_label = "sample"
    bare_interface.get_best_transform_key = MagicMock(
        return_value="registered"
    )
    bare_interface._clear_napari_view = MagicMock()
    shapes = [np.zeros((4, 2))]
    shape_data = (shapes, ["0"], ["image-0"], [(1, 1, 1)])
    image_data = object()
    # no overview built: what this checks happens around it
    bare_interface._create_lazy_overview = MagicMock(return_value=None)
    bare_interface._create_napari_shapes = MagicMock(
        return_value=shape_data
    )
    bare_interface._create_napari_data = MagicMock(
        return_value=image_data
    )
    bare_interface._napari_view_add_fused_data = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda _: {"z": 0},
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )

    bare_interface.update_views(show_preprocessed=True)

    assert bare_interface._clear_napari_view.call_args_list == [
        call(bare_interface.viewer),
        call(bare_interface.overview),
    ]
    # the viewer's shapes are built off the Qt thread, reporting per source into the refresh's
    # own bar; the overview's are rebuilt flattened (see _refresh_overview_shapes)
    assert bare_interface._create_napari_shapes.call_args_list == [
        call("registered", force_2d=False, progress_factory=ANY, weight=3),
        call("registered", force_2d=True),
    ]
    assert bare_interface._create_napari_data.call_count == 1
    args, kwargs = bare_interface._create_napari_data.call_args
    assert args == ("registered",)
    assert kwargs["show_preprocessed"] is True
    assert kwargs["composite"] is True
    # the longest step of a refresh reports from the inside rather than being one silent block,
    # so it is handed the operation's progress factory and the share of the bar it is worth
    assert kwargs["progress_factory"] is not None
    assert kwargs["weight"] > 1
    bare_interface._napari_view_add_fused_data.assert_called_once_with(
        bare_interface.viewer, image_data, "sample data", cheap=True
    )
    expected_shape_call = (
        shapes, ["0"], ["image-0"], [(1, 1, 1)], "sample shapes"
    )
    bare_interface._update_view_add_shapes.assert_any_call(
        bare_interface.viewer,
        *expected_shape_call,
    )
    # the overview reuses the viewer's own shapes, drawn much smaller - a label per shape is
    # unreadable there and covers the layout it is there to show
    bare_interface._update_view_add_shapes.assert_any_call(
        bare_interface.overview,
        *expected_shape_call,
        show_labels=False,
    )
    assert bare_interface._update_view_add_shapes.call_count == 2
    assert bare_interface.view_mode is ViewMode.OVERVIEW


def test_update_views_detects_multi_z_from_view_msims(
    bare_interface, monkeypatch
):
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}, {"z": 1}]
    # no overview built: what this checks happens around it
    bare_interface._create_lazy_overview = MagicMock(return_value=None)
    bare_interface._create_napari_shapes = MagicMock(
        return_value=([], [], [], [])
    )
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()

    bare_interface.update_views(transform_key="source_metadata", show_images=False)

    bare_interface._create_napari_shapes.assert_called_once_with(
        "source_metadata", force_2d=True, progress_factory=ANY, weight=3
    )


def test_registered_shapes_of_2d_view_msims_sit_at_their_sections_z(bare_interface):
    """The view msims stay 2D: each source's shape takes its section's z from its position."""
    from multiview_stitcher import msi_utils, spatial_image_utils as si_utils

    def flat_msim(origin_x):
        sim = si_utils.get_sim_from_array(np.zeros((8, 8), dtype=np.uint8), dims=["y", "x"],
                                          scale={"y": 1.0, "x": 1.0}, translation={"y": 0.0, "x": origin_x},
                                          transform_key="registered")
        return msi_utils.get_msim_from_sim(sim, scale_factors=[])

    bare_interface.view_msims = [flat_msim(0.0), flat_msim(4.0)]
    bare_interface.reg.positions = [{"z": 0.0}, {"z": 5.0}]
    bare_interface.reg.source_transform_key = "source_metadata"
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.file_labels = ["a", "b"]

    shapes, _, _, _ = bare_interface._create_napari_shapes("registered", force_2d=True)

    assert [sorted({point[0] for point in shape}) for shape in shapes] == [[0.0], [5.0]]
    assert min(point[2] for point in shapes[1]) == 4.0


def test_create_napari_shapes_reports_per_source(bare_interface, monkeypatch):
    """Building the shapes (a geometry per source, then every overlapping pair) takes a share of the
    refresh's bar and reports its own sub-steps, the per-source build counting each source."""
    sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10}) for _ in range(3)]
    bare_interface.reg.sources = sources
    bare_interface.reg.positions = [{"z": 0}] * 3
    bare_interface.reg._msim_transforms = [None] * 3
    bare_interface.reg._msim_output_order = "yx"
    bare_interface.reg._msim_z_scale = 1
    bare_interface.reg.source_transform_key = "source_metadata"
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.file_labels = ["a", "b", "c"]
    monkeypatch.setattr(
        interface_module, "build_source_stack_props", lambda *_, **__: "props"
    )
    monkeypatch.setattr(
        interface_module, "create_image_shapes",
        lambda msims, **__: [np.zeros((4, 2)) for _ in msims],
    )
    monkeypatch.setattr(
        interface_module, "create_overlap_shapes", lambda *_, **__: ([], [])
    )

    phases = []

    class _Phase:
        def __init__(self, total, weight):
            self.total = total
            self.weight = weight
            self.updates = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, n=1):
            self.updates += n

    def factory(total=None, desc=None, weight=1, **_):
        phase = _Phase(total, weight)
        phases.append(phase)
        return phase

    bare_interface._create_napari_shapes(
        "source_metadata", progress_factory=factory, weight=5
    )

    # the per-source build, then create_image_shapes and create_overlap_shapes
    assert [(phase.total, phase.updates) for phase in phases] == [(3, 3), (1, 1), (1, 1)]
    # and it is the per-source build that gets most of what the step was allowed
    assert phases[0].weight > phases[1].weight


def test_update_napari_shapes_adds_3d_box_with_overlap_metadata(
    bare_interface, monkeypatch
):
    """A 3D box is 6 quality-colored quad faces plus an edge-only 'path' wireframe; distinct
    per-corner values so a face/wire indexing mistake is caught."""
    viewer = MagicMock()
    image_shape = np.arange(24, dtype=float).reshape(8, 3)
    overlap_shape = np.arange(24, 48, dtype=float).reshape(8, 3)
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"z": 2})]
    bare_interface.reg.positions = [{"z": 0}]
    bare_interface.reg.get_metrics.return_value = 0.75
    create_shapes = MagicMock(return_value=[image_shape])
    create_overlaps = MagicMock(
        return_value=([overlap_shape], [np.array([0, 0])])
    )
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda _: {"z": 0},
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )
    monkeypatch.setattr(interface_module, "create_image_shapes", create_shapes)
    monkeypatch.setattr(
        interface_module, "create_overlap_shapes", create_overlaps
    )
    monkeypatch.setattr(
        interface_module, "metric_to_rgb", lambda _: (0.1, 0.2, 0.3)
    )

    shape_data = bare_interface._create_napari_shapes("registered")
    bare_interface._update_view_add_shapes(viewer, *shape_data, "boxes")

    assert viewer.add_shapes.call_count == 1
    args, kwargs = viewer.add_shapes.call_args
    shape_data_out = args[0]
    edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
    expected_wire_width = np.ptp(
        np.concatenate([image_shape, overlap_shape]), axis=0
    ).max() * 0.005

    assert kwargs["shape_type"] == ["polygon"] * 12 + ["path"] * 2
    # napari/napari#6860: faces come from each box's axis-aligned bounding box, so check their extent
    image_faces = np.concatenate([np.asarray(quad) for quad in shape_data_out[:6]])
    overlap_faces = np.concatenate([np.asarray(quad) for quad in shape_data_out[6:12]])
    np.testing.assert_allclose(image_faces.min(axis=0), image_shape.min(axis=0))
    np.testing.assert_allclose(image_faces.max(axis=0), image_shape.max(axis=0))
    np.testing.assert_allclose(overlap_faces.min(axis=0), overlap_shape.min(axis=0))
    np.testing.assert_allclose(overlap_faces.max(axis=0), overlap_shape.max(axis=0))
    # the wireframe keeps the true corners: edges render in 3D whatever their orientation
    np.testing.assert_allclose(shape_data_out[12], image_shape[edge_path])
    np.testing.assert_allclose(shape_data_out[13], overlap_shape[edge_path])

    # each box's quality color is on all its faces; a 'path' has no face, so gets a placeholder
    face_color = kwargs["face_color"]
    for face in range(6):
        np.testing.assert_allclose(face_color[face], (1, 1, 1))
        np.testing.assert_allclose(face_color[6 + face], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(face_color[12], (0, 0, 0))
    np.testing.assert_allclose(face_color[13], (0, 0, 0))
    # mixed 3- and 4-tuples make napari fall back to plain white for the whole layer
    assert len({len(color) for color in face_color}) == 1
    edge_color = kwargs["edge_color"]
    assert len({len(color) for color in edge_color}) == 1
    np.testing.assert_allclose(edge_color[:12], [(0, 0, 0)] * 12)
    np.testing.assert_allclose(edge_color[12:], [(0, 1, 1)] * 2)
    np.testing.assert_allclose(
        kwargs["edge_width"], [0] * 12 + [expected_wire_width] * 2
    )
    assert kwargs["features"]["refs"] == ["0"] * 6 + ["0 0"] * 6 + ["0", "0 0"]
    assert kwargs["features"]["labels"] == [""] * 12 + ["image-0", ""]


def test_update_napari_shapes_3d_faces_are_axis_aligned_and_wind_outward(
    bare_interface, monkeypatch
):
    """napari/napari#6860: a 3D face only fills when axis-orthogonal, so a rotated box's faces come
    from its axis-aligned bounding box - and each must wind outward, or it is backface-culled."""
    viewer = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"z": 2})]
    bare_interface.reg.positions = [{"z": 0}]
    unit_cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    theta = np.pi / 6
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    rotated_box = unit_cube @ rotation.T + np.array([10.0, 20.0, 30.0])

    monkeypatch.setattr(
        interface_module.si_utils, "get_origin_from_sim", lambda _: {"z": 0}
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )
    monkeypatch.setattr(
        interface_module, "create_image_shapes", lambda *_, **__: [rotated_box]
    )
    monkeypatch.setattr(
        interface_module, "create_overlap_shapes", lambda *_, **__: ([], [])
    )

    shape_data = bare_interface._create_napari_shapes("registered")
    bare_interface._update_view_add_shapes(viewer, *shape_data, "boxes")

    assert viewer.add_shapes.call_count == 1
    args, kwargs = viewer.add_shapes.call_args
    shape_data_out = args[0]
    face_quads = [np.asarray(shape) for shape, shape_type in zip(shape_data_out, kwargs["shape_type"])
                  if shape_type == "polygon"]
    assert len(face_quads) == 6

    box_min, box_max = rotated_box.min(axis=0), rotated_box.max(axis=0)
    box_center = (box_min + box_max) / 2
    for quad in face_quads:
        normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        nonzero_axes = np.sum(~np.isclose(normal, 0))
        assert nonzero_axes == 1, f"face normal {normal} is not axis-orthogonal"
        outward = quad.mean(axis=0) - box_center
        assert np.dot(normal, outward) >= 0, "face wound inward - would be backface-culled"

    # the fill's overall extent must still bound the true (rotated) box, not some
    # unrelated or degenerate region
    all_face_points = np.concatenate(face_quads)
    np.testing.assert_allclose(all_face_points.min(axis=0), box_min)
    np.testing.assert_allclose(all_face_points.max(axis=0), box_max)


def test_update_napari_shapes_labels_only_where_asked(
    bare_interface, monkeypatch
):
    """The overview shows the viewer's own shapes at a fraction of the size, where one label
    per shape is unreadable and covers the layout the overview is there to show - so it takes
    the same shapes untexted, and without the per-shape column only that text ever read."""
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}]
    monkeypatch.setattr(
        interface_module.si_utils, "get_origin_from_sim", lambda _: {}
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )
    shapes = [np.zeros((4, 2))]

    viewer = MagicMock()
    bare_interface._update_view_add_shapes(
        viewer, shapes, ["0"], ["image-0"], [(1, 1, 1)], "boxes"
    )
    _, kwargs = viewer.add_shapes.call_args
    assert kwargs["shape_type"] == "polygon"
    assert kwargs["edge_width"] == 0.1
    assert kwargs["text"] == {"string": "{labels}", "size": 6}
    assert kwargs["features"]["labels"] == ["image-0"]

    overview = MagicMock()
    bare_interface._update_view_add_shapes(
        overview, shapes, ["0"], ["image-0"], [(1, 1, 1)], "boxes", show_labels=False
    )
    args, kwargs = overview.add_shapes.call_args
    assert kwargs["text"] is None
    assert "labels" not in kwargs["features"]
    assert kwargs["features"]["refs"] == ["0"]
    # the shapes themselves are unchanged - the same ones the viewer was given
    np.testing.assert_allclose(args[0], [shapes[0]])


def test_update_napari_features_dispatches_all_layer_types(
    bare_interface, monkeypatch
):
    viewer = MagicMock()
    viewer.layers.__len__.return_value = 1
    layers = [
        ("image", {"name": "image"}, "image"),
        ("points", {"name": "points"}, "points"),
        ("shapes", {"name": "shapes"}, "shapes"),
    ]
    monkeypatch.setattr(
        interface_module,
        "draw_keypoints_matches_napari",
        lambda *_, **__: layers,
    )

    bare_interface._napari_view_show_features(
        viewer, None, None, None, None, None, None
    )

    viewer.layers.clear.assert_called_once_with()
    viewer.add_image.assert_called_once_with("image", name="image")
    viewer.add_points.assert_called_once_with("points", name="points")
    viewer.add_shapes.assert_called_once_with("shapes", name="shapes")


def test_add_napari_image_applies_color_and_affine_callback(
    bare_interface, monkeypatch
):
    viewer = MagicMock()
    layer = viewer.add_image.return_value
    data = object()
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_spacing_from_sim",
        lambda *_args, **_kwargs: [2, 3],
    )
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda *_args, **_kwargs: [4, 5],
    )

    result = bare_interface._napari_view_add_image(
        viewer,
        data,
        "image",
        transform="affine",
        color="red",
        affine_event=True,
    )

    assert result is layer
    assert layer.colormap == "red"
    layer.events.affine.connect.assert_called_once_with(
        bare_interface.on_image_data_changed
    )


def test_on_image_data_changed_restarts_metrics_timer(bare_interface):
    bare_interface.pair_metrics_timer = MagicMock()

    bare_interface.on_image_data_changed(object())

    bare_interface.pair_metrics_timer.stop.assert_called_once_with()
    bare_interface.pair_metrics_timer.start.assert_called_once_with()


@pytest.fixture
def mocked_activity_contexts(monkeypatch):
    monkeypatch.setattr(
        interface_module, "NapariMVSProgress", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        interface_module, "NapariDaskProgress", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        Interface,
        "_operation_widgets",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(
        interface_module, "VisibleActivityDock", lambda _: nullcontext()
    )


@pytest.mark.parametrize("with_t", [True, False], ids=["bbox-with-t", "bbox-without-t"])
def test_run_pair_registration_serializes_quality_and_time_bbox(
    bare_interface, monkeypatch, mocked_activity_contexts, with_t
):
    import xarray as xr

    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.metrics_methods = ["ncc"]
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.reg.register_msims = ["register-msim"]
    bare_interface.reg.pairs_graph = object()
    results = {
        "pair_mappings": {(0, 1): "mapping"},
        "metrics": {
            "pairs": {
                (0, 1): {
                    interface_module.default_transform_key: {
                        interface_module.default_quality_key: 0.9
                    }
                }
            }
        },
    }
    bare_interface.reg.register_pairs.return_value = results
    bbox = xr.DataArray([[1, 2], [3, 4]], dims=("corner", "axis"))
    if with_t:
        bbox = bbox.expand_dims({"t": [0]})
    monkeypatch.setattr(
        interface_module.nx,
        "get_edge_attributes",
        lambda *_: {(0, 1): bbox},
    )

    actual = bare_interface.run_pair_registration()

    assert actual is results
    bare_interface.reg.register_pairs.assert_called_once_with(
        ["register-msim"],
        params={"method": "phase", "metrics": ["ncc"]},
        progress_factory=ANY,
    )
    bare_interface.reg.save_pair_mappings.assert_called_once_with(
        {(0, 1): "mapping"},
        {(0, 1): 0.9},
        {(0, 1): [[1, 2], [3, 4]]},
    )


def test_run_global_registration_persists_all_results(
    bare_interface, mocked_activity_contexts, monkeypatch
):
    # the sources here are stand-ins with no transforms to snapshot
    monkeypatch.setattr(interface_module, 'snapshot_msims_transform', lambda msims, key: None)
    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.reg.pair_msims = ["msim"]
    bare_interface.reg.register_indices = [0]
    results = {
        "mappings": {"image-0": "mapping"},
        "metrics": {"summary": {}},
    }
    bare_interface.reg.register_global.return_value = results

    actual = bare_interface.run_global_registration()

    assert actual is results
    _, global_kwargs = bare_interface.reg.register_global.call_args
    assert global_kwargs["register_indices"] == [0]
    assert global_kwargs["params"] == {"method": "phase"}
    # register_global() reports its own stage boundaries: most of it is one blocking call, so
    # without them the bar would not move until dask work near the end
    assert global_kwargs["progress_factory"] is not None
    bare_interface.reg.save_mappings.assert_called_once_with(
        results["mappings"]
    )
    bare_interface.reg.save_mappings_csv.assert_called_once_with(
        results["mappings"]
    )
    bare_interface.reg.save_metrics.assert_called_once_with(
        results["metrics"]
    )


def _stub_preview_registration_deps(bare_interface, monkeypatch, label1="image-0", label2="image-1"):
    bare_interface.viewer = MagicMock()
    bare_interface.param_widgets = {
        "registration.reg_preview_image1": SimpleNamespace(get_value=lambda: label1),
        "registration.reg_preview_image2": SimpleNamespace(get_value=lambda: label2),
    }
    bare_interface.reg.file_labels = ["image-0", "image-1", "image-2"]
    bare_interface.reg.register_msims = ["msim-0", "msim-1", "msim-2"]
    bare_interface.metrics_methods = []
    bare_interface._preview_overlap_cache = None
    overlap1 = SimpleNamespace(compute=lambda: "overlap1-computed")
    overlap2 = SimpleNamespace(compute=lambda: "overlap2-computed")
    bare_interface.reg.select_pair_overlap.return_value = (overlap1, overlap2, "pixel-space")
    bare_interface.reg.register_overlap.return_value = ("transform", 0.5, {"fixed_points": []})
    monkeypatch.setattr(interface_module, "calc_msims_metrics", lambda *a, **k: {"metrics": True})
    bare_interface.params = {"registration": {"method": "orb"}}


@pytest.mark.parametrize(("change", "selections"), [("method", 1), ("pair", 2), ("register_msims", 2)])
def test_run_preview_registration_reuses_the_overlap_until_its_data_or_pair_changes(
    bare_interface, monkeypatch, mocked_activity_contexts, change, selections
):
    """The overlap crop depends only on the source data (a new register_msims list whenever
    pre-processing changes something) and the selected pair, never on the registration method."""
    _stub_preview_registration_deps(bare_interface, monkeypatch)
    assert bare_interface.run_preview_registration() is not None

    if change == "method":
        bare_interface.params = {"registration": {"method": "sift"}}
    elif change == "pair":
        bare_interface.param_widgets["registration.reg_preview_image2"] = SimpleNamespace(get_value=lambda: "image-2")
    else:
        bare_interface.reg.register_msims = ["msim-0-reprocessed", "msim-1-reprocessed", "msim-2-reprocessed"]
    assert bare_interface.run_preview_registration() is not None

    assert bare_interface.reg.select_pair_overlap.call_count == selections
    assert bare_interface.reg.register_overlap.call_count == 2


@pytest.mark.parametrize("overlaps", [True, False], ids=["failure", "no-overlap"])
def test_a_failed_preview_registration_returns_none(bare_interface, monkeypatch, mocked_activity_contexts, overlaps):
    """None lets preview_registration() bail out; two images that do not overlap are a warning
    naming them rather than a failure."""
    from muvis_align.image.util import NoOverlapError

    _stub_preview_registration_deps(bare_interface, monkeypatch)
    bare_interface.reg.select_pair_overlap.side_effect = (
        ValueError("boom") if overlaps else NoOverlapError("the images do not overlap"))
    report_failure = MagicMock()
    monkeypatch.setattr("muvis_align.ui._utils.report_failure", report_failure)

    with patch.object(interface_module, "show_warning") as show_warning:
        assert bare_interface.run_preview_registration() is None

    assert report_failure.called is overlaps
    if overlaps:
        show_warning.assert_not_called()
    else:
        show_warning.assert_called_once_with("image-0 and image-1 do not overlap: choose two images that do")


@pytest.mark.parametrize(
    ("global_registered", "pairs_registered", "reply", "runs", "prefix"),
    [
        (True, False, "Yes", True, "Global registration was already performed. "),
        (True, False, "No", False, "Global registration was already performed. "),
        (False, True, "Yes", True, "Pair registration was already performed. "),
        (False, False, "Yes", True, ""),
        (False, False, "No", False, ""),
    ],
)
def test_pair_registration_confirmation_paths(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    global_registered,
    pairs_registered,
    reply,
    runs,
    prefix,
):
    """Pair registration is offered whatever has run before: an earlier global (or pair)
    registration only prepends a note to the confirmation, it no longer blocks the operation."""
    bare_interface.viewer = MagicMock()
    bare_interface.reg.is_global_registered.return_value = global_registered
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.source_transform_key = "source_metadata"
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.update_registered = MagicMock()
    messages = []

    def question(_parent, _title, message, *_):
        messages.append(message)
        return getattr(interface_module.QMessageBox, reply)

    monkeypatch.setattr(interface_module.QMessageBox, "question", question)

    bare_interface.pair_registration()

    assert messages == [prefix + "Run pair registration?"]
    assert bare_interface.run_pair_registration.called is runs
    assert bare_interface.update_registered.called is runs


def test_registration_process_merge_goes_to_fusion_without_registering(
    bare_interface, monkeypatch
):
    """merge fuses at the source positions, so there is nothing to register or ask here."""
    bare_interface.params = {"registration": {"operation": "merge"}}
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.run_global_registration = MagicMock()
    bare_interface.select_tab = MagicMock()
    question = MagicMock()
    monkeypatch.setattr(interface_module.QMessageBox, "question", question)

    bare_interface.registration_process()

    assert not question.called
    assert not bare_interface.run_pair_registration.called
    assert not bare_interface.run_global_registration.called
    bare_interface.select_tab.assert_called_once_with(4)


def test_run_fusion_fuses_by_best_transform_key(bare_interface, monkeypatch):
    """A merge never writes a 'registered' transform, so fusion must not assume one."""
    bare_interface.params = {
        "registration": {"operation": "merge"},
        "fusion": {"method": "average", "spacing": "mean",
                   "tile_size": "", "ome_version": "0.5"},
        "input_output": {"registration_dimension": "space"},
    }
    bare_interface.reg.reg_transform_key = "registered"
    bare_interface.get_best_transform_key = MagicMock(
        return_value="source_metadata"
    )
    bare_interface.reg.fuse.return_value = ("fused", True)
    bare_interface._run_off_thread = lambda func, factory: func(MagicMock())
    monkeypatch.setattr(
        interface_module, "NapariMVSProgress", lambda **_: nullcontext()
    )
    bare_interface._operation_progress = lambda *a, **k: nullcontext(MagicMock())

    assert bare_interface.run_fusion() == "fused"

    _, fuse_kwargs = bare_interface.reg.fuse.call_args
    assert fuse_kwargs["transform_key"] == "source_metadata"
    # is_saved was True, so the separate save() path is not taken
    assert not bare_interface.reg.save.called


@pytest.mark.parametrize(
    ("pairs_registered", "reply", "run_pair", "run_global"),
    [
        (False, "No", False, False),
        (False, "Yes", True, True),
        (True, "Yes", False, True),
    ],
)
def test_registration_process_confirmation_and_prerequisites(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    pairs_registered,
    reply,
    run_pair,
    run_global,
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"operation": "register"}}
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.msims = ["sim"]
    bare_interface.reg.reg_transform_key = "registered"
    bare_interface.view_msims = ["preview"]
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.run_global_registration = MagicMock()
    bare_interface.enable_tabs = MagicMock()
    bare_interface.update_registered = MagicMock()
    copy = MagicMock()
    monkeypatch.setattr(interface_module, "copy_transforms_to_msims", copy)
    monkeypatch.setattr(
        interface_module.QMessageBox,
        "question",
        lambda *_: getattr(interface_module.QMessageBox, reply),
    )

    bare_interface.registration_process()

    assert bare_interface.run_pair_registration.called is run_pair
    assert bare_interface.run_global_registration.called is run_global
    if run_global:
        # pair and global registration each report their own bar, one after the other - sharing
        # one left the global registration reporting into a bar the pair phases had filled
        bare_interface.run_global_registration.assert_called_once_with()
        if run_pair:
            bare_interface.run_pair_registration.assert_called_once_with()
        copy.assert_called_once_with(["sim"], ["preview"], "registered")
        # the refresh is a phase of the registration operation's bar, not a bar of its own
        _, refresh_kwargs = bare_interface.update_registered.call_args
        assert refresh_kwargs['view_transform_key'] == "registered"
        assert refresh_kwargs['progress_factory'] is not None


@pytest.mark.parametrize(
    ("tile_size", "expected", "reply"),
    [("1024", 1024, "Yes"), ("512, 1024", [512, 1024], "Yes"), ("1024", None, "No")],
)
def test_fusion_process_parses_tile_size_and_updates_state(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    tile_size,
    expected,
    reply,
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {
        "registration": {"operation": "register"},
        "input_output": {"registration_dimension": "all"},
        "fusion": {
            "method": "average",
            "spacing": "mean",
            "tile_size": tile_size,
            "ome_version": "0.5",
        },
    }
    bare_interface.reg.is_fused.return_value = False
    bare_interface.reg.fuse.return_value = ("fused", None)
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_fused_data = MagicMock()
    bare_interface.reg.estimate_fusion_size.return_value = {
        "levels": [{"spacing": 0.01, "bytes": 1500}, {"spacing": 0.02, "bytes": 500}], "bytes": 2000, "native": True}
    messages = []

    def question(_parent, _title, message, *_):
        messages.append(message)
        return getattr(interface_module.QMessageBox, reply)

    monkeypatch.setattr(interface_module.QMessageBox, "question", question)
    # fuse() always returns real msims in production - this test only cares about tile_size
    # parsing and state transitions, so stand in a passthrough for the msim->sim extraction step
    monkeypatch.setattr(interface_module, "extract_sims_from_fused", lambda result: result)

    bare_interface.fusion_process()

    # the question carries the estimate, made for the fusion about to run
    assert messages[0].endswith(f"Estimated output: {print_hbytes(2000)}")
    if reply == "No":
        assert not bare_interface.reg.fuse.called
        assert not bare_interface._napari_view_add_fused_data.called
        return
    assert bare_interface.reg.estimate_fusion_size.call_args.kwargs["tile_size"] == expected
    assert bare_interface.reg.fuse.call_args.kwargs["tile_size"] == expected
    assert (
        bare_interface.reg.fuse.call_args.kwargs["output_filename"]
        == "registered"
    )
    bare_interface._napari_view_add_fused_data.assert_called_once_with(
        bare_interface.viewer, "fused", "Fused"
    )
    assert bare_interface.reg.state is RegState.FUSED
    assert bare_interface.view_mode is ViewMode.FUSED


@pytest.mark.parametrize(
    ("operation", "pairs_registered", "global_registered", "prefix", "run_pair", "run_global"),
    [
        ("register", False, False, "Registration not performed yet. ", True, True),
        ("register", True, False, "Global registration not performed yet. ", False, True),
        ("register", True, True, "", False, False),
        ("merge", False, False, "", False, False),
    ],
)
def test_fusion_process_runs_missing_registration_first(
    bare_interface, monkeypatch, mocked_activity_contexts,
    operation, pairs_registered, global_registered, prefix, run_pair, run_global,
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"operation": operation}}
    bare_interface.reg.is_fused.return_value = False
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.is_global_registered.return_value = global_registered
    bare_interface._fusion_size_text = MagicMock(return_value="")
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_fused_data = MagicMock()
    calls = []
    bare_interface.run_pair_registration = MagicMock(side_effect=lambda: calls.append("pair") or {"pairs": 1})
    bare_interface.run_global_registration = MagicMock(side_effect=lambda: calls.append("global") or {"global": 1})
    bare_interface.run_fusion = MagicMock(side_effect=lambda: calls.append("fusion") or "fused")
    messages = []

    def question(_parent, _title, message, *_):
        messages.append(message)
        return interface_module.QMessageBox.Yes

    monkeypatch.setattr(interface_module.QMessageBox, "question", question)
    monkeypatch.setattr(interface_module.QMessageBox, "information", MagicMock())

    bare_interface.fusion_process()

    assert messages[0].startswith(prefix)
    assert calls == ["pair"] * run_pair + ["global"] * run_global + ["fusion"]


def test_fusion_process_stops_when_registration_fails(bare_interface, monkeypatch):
    bare_interface.params = {"registration": {"operation": "register"}}
    bare_interface.reg.is_fused.return_value = False
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface._fusion_size_text = MagicMock(return_value="")
    bare_interface.run_pair_registration = MagicMock(return_value=None)
    bare_interface.run_global_registration = MagicMock()
    bare_interface.run_fusion = MagicMock()
    monkeypatch.setattr(interface_module.QMessageBox, "question",
                        lambda *_: interface_module.QMessageBox.Yes)

    bare_interface.fusion_process()

    assert not bare_interface.run_global_registration.called
    assert not bare_interface.run_fusion.called


@pytest.mark.parametrize(
    ("operation", "global_registered", "enabled"),
    [("register", False, False), ("register", True, True), ("merge", False, True)],
)
def test_fusion_preview_enabled_only_with_something_to_fuse(
    bare_interface, operation, global_registered, enabled
):
    """Unregistered, only a merge has positions worth previewing a fusion at."""
    bare_interface.params = {"registration": {"operation": "register"}}
    bare_interface.write_params = MagicMock()
    bare_interface.reg.is_global_registered.return_value = global_registered
    preview = SimpleNamespace(widget=SimpleNamespace(enabled=None))
    bare_interface.param_widgets = {"fusion.preview_fusion": preview}

    bare_interface.change_param("registration.operation", operation)

    assert preview.widget.enabled is enabled


@pytest.mark.parametrize(
    ("pairs_registered", "selected"), [(False, None), (True, 3)],
)
def test_show_loaded_project_opens_every_tab(bare_interface, pairs_registered, selected):
    """Every step is reachable once the sources are read: a later one runs what it needs first."""
    bare_interface.reg.is_fused.return_value = False
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.update_views = MagicMock()
    bare_interface.update_registered = MagicMock()
    bare_interface.enable_tabs = MagicMock()
    bare_interface.select_tab = MagicMock()

    bare_interface._show_loaded_project()

    bare_interface.enable_tabs.assert_called_once_with(True, 4)
    if selected is None:
        assert not bare_interface.select_tab.called
    else:
        bare_interface.select_tab.assert_called_once_with(selected)


def test_build_view_msims_downscales_large_single_resolution_source():
    """A single-resolution source (no native pyramid to pick a coarser level from) larger than
    1000px on its largest spatial dimension must be downscaled by one constant factor so that
    dimension becomes ~1000px; a source already at or under 1000px is left untouched."""
    from multiview_stitcher import spatial_image_utils as si_utils
    from muvis_align.image.util import get_msim_image0, wrap_sims_as_msims

    def make_source_and_msim(size):
        sim = si_utils.get_sim_from_array(
            np.zeros((size, size), dtype=np.uint8),
            dims=['y', 'x'],
            scale={'y': 1, 'x': 1},
            translation={'y': 0, 'x': 0},
            transform_key='source_metadata',
        )
        msim = wrap_sims_as_msims([sim])[0]
        source = SimpleNamespace(
            shapes=[(size, size)],
            scale_factors=[{'y': 1.0, 'x': 1.0}],
            get_pixel_size=lambda: {'y': 1.0, 'x': 1.0},
        )
        return source, msim

    large_source, large_msim = make_source_and_msim(2000)
    small_source, small_msim = make_source_and_msim(500)

    interface = Interface.__new__(Interface)
    interface.reg = SimpleNamespace(
        sources=[large_source, small_source],
        msims=[large_msim, small_msim],
        source_transform_key='source_metadata',
    )

    view_msims = interface._build_view_msims()

    large_image0 = get_msim_image0(view_msims[0])
    assert large_image0.sizes['x'] == 1000
    assert large_image0.sizes['y'] == 1000

    small_image0 = get_msim_image0(view_msims[1])
    assert small_image0.sizes['x'] == 500
    assert small_image0.sizes['y'] == 500


def test_the_preprocessed_preview_is_3d_without_building_or_changing_the_msims(tmp_path):
    """fuse() promotes 2D msims at several z positions to 3D itself, so the chunk size it is handed
    must already have z. The preview must neither build the full-resolution msims nor write into
    reg.register_msims, which the steps in between hand on as they are when they change nothing."""
    from multiview_stitcher import msi_utils
    from muvis_align.MVSRegistration import MVSRegistration

    source_metadata = {
        'position': {'z': 'fn[-2]', 'y': 0.0, 'x': 'fn[-2]*30'},
        'scale': {'z': '1', 'y': '0.032', 'x': '0.032'},
    }
    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[(DATA_DIR / name).as_posix() for name in TIFF_FILES[:2]],
        output_path=tmp_path.as_posix() + '/',
        source_metadata=source_metadata,
    )
    reg.init_data(source_metadata=source_metadata)
    assert len(set(position.get('z') for position in reg.positions)) > 1
    reg.preprocess(reg.msims, scale=None, flatfield_quantiles='', normalisation='none',
                   filter_foreground=False)
    assert 'z' not in reg.register_msims[0]['scale0'].ds['image'].dims
    # deferred, as in a real run: pre-processing works off msims built for its own scale
    reg._msims = None

    def stored_transforms():
        return [np.asarray(msi_utils.get_transform_from_msim(msim, reg.source_transform_key))
                for msim in reg.register_msims]

    before = stored_transforms()
    interface = Interface.__new__(Interface)
    interface.reg = reg
    interface.params = {'input_output': {'registration_dimension': 'space'}}
    interface.extra_metadata = {}

    fused_msim = interface._create_napari_data(reg.source_transform_key, show_preprocessed=True)

    assert fused_msim['scale0'].ds['image'].sizes['z'] == 2
    assert reg._msims is None
    for original, current in zip(before, stored_transforms()):
        np.testing.assert_array_equal(original, current)


def test_preview_data_layer_is_real_multiscale_pyramid(make_napari_viewer, tmp_path):
    """The preview's data layer (_create_napari_data -> _napari_view_add_fused_data) is a genuine
    napari multiscale layer from msims end to end, as the fused export's is."""
    from multiview_stitcher import msi_utils

    reg = prepared_registration([(DATA_DIR / name).as_posix() for name in ZARR_FILES[:2]], tmp_path,
                                preprocess=False)

    interface = Interface.__new__(Interface)
    interface.reg = reg
    interface.params = {'input_output': {'registration_dimension': 'all'}}
    interface.extra_metadata = {}
    interface.view_msims = interface._build_view_msims()

    fused_msim = interface._create_napari_data(reg.source_transform_key, fusion_method='')
    n_levels = len(msi_utils.get_sorted_scale_keys(fused_msim))
    assert n_levels > 1  # sanity check: fusion produced a real multiscale pyramid, not one level

    viewer = make_napari_viewer()
    interface._napari_view_add_fused_data(viewer, fused_msim, 'data')

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.multiscale is True
    assert len(layer.data) == n_levels
    for level_data, next_level_data in zip(layer.data, layer.data[1:]):
        assert next_level_data.shape[-1] <= level_data.shape[-1]
        assert next_level_data.shape[-2] <= level_data.shape[-2]


@pytest.mark.parametrize('with_factory', [True, False], ids=['factory', 'no-factory'])
def test_build_view_msims_keeps_native_pyramids_and_reports_per_source(with_factory):
    """A source with a native pyramid is shown as it is; building them reports one step per
    source when given a factory, and needs none."""
    interface = Interface.__new__(Interface)
    interface.reg = MagicMock()
    interface.reg.sources = [SimpleNamespace(shapes=[0, 1]), SimpleNamespace(shapes=[0, 1])]
    interface.reg.msims = ['msim-0', 'msim-1']
    factory, records = recording_phase_factory()

    view_msims = interface._build_view_msims(progress_factory=factory if with_factory else None)

    assert view_msims == ['msim-0', 'msim-1']
    assert records == ([{'desc': 'Building views', 'total': 2, 'done': 2}] if with_factory else [])


def test_init_progress_reports_saved_project_load(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    """reg.init_progress() (the msim build and the saved registration) reports into the load's bar;
    drawing the view it ends on is a separate operation with its own."""
    bare_interface.viewer = MagicMock()
    bare_interface.params = {'registration': {'operation': 'register'}}
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface.reg.is_fused.return_value = False
    bare_interface.update_views = MagicMock()
    bare_interface.enable_tabs = MagicMock()

    bare_interface.init_progress()

    _, kwargs = bare_interface.reg.init_progress.call_args
    assert callable(kwargs['progress_factory'])
    # the refresh owns its own bar (update_views() opens one when given no factory)
    bare_interface.update_views.assert_called_once_with(show_images=False)


def test_update_views_reports_into_the_callers_bar(bare_interface, monkeypatch):
    """A refresh that is part of a larger operation (loading a saved project) must report as
    phases of that operation's bar, and leave the activity dock / widget state to it."""
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}]
    # no overview built: what this checks happens around it
    bare_interface._create_lazy_overview = MagicMock(return_value=None)
    bare_interface._create_napari_shapes = MagicMock(return_value=([], [], [], []))
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()
    dock = MagicMock()
    monkeypatch.setattr(interface_module, "VisibleActivityDock", dock)

    FakeBar.instances.clear()
    factory = make_phase_factory(desc='Loading project')
    with factory:
        with factory(total=1, desc='Loading pair registration') as pbar:
            pbar.update(1)
        bare_interface.update_views(transform_key="source_metadata", show_images=False,
                                    progress_factory=factory)

    dock.assert_not_called()
    # one bar, still the caller's - the refresh opened none of its own, and did not rename it
    assert len(FakeBar.instances) == 1
    bar = FakeBar.instances[0]
    assert bar.total == factory.ticks
    assert bar.descriptions == ['Loading project']


def test_pre_processing_process_reports_work_and_view_separately(bare_interface):
    """Pre-processing and the view refresh it triggers are two operations, in that order: the
    work reports its own bar, then building and drawing the view reports a second one - neither
    is handed the other's factory."""
    bare_interface.viewer = MagicMock()
    bare_interface.run_pre_processing = MagicMock(return_value=True)
    bare_interface.update_views = MagicMock()
    bare_interface.enable_tabs = MagicMock()
    bare_interface.enable_modify_pair_registration = MagicMock()
    bare_interface.select_tab = MagicMock()

    bare_interface.pre_processing_process()

    bare_interface.run_pre_processing.assert_called_once_with()
    bare_interface.update_views.assert_called_once_with(show_preprocessed=True)


def test_pre_processing_after_registration_views_without_the_registration(bare_interface):
    """The pre-processed msims have no registered transform, so the registration is undone
    before the view picks its transform key, not after."""
    bare_interface.viewer = MagicMock()
    bare_interface.run_pre_processing = MagicMock(return_value=True)
    bare_interface.enable_tabs = MagicMock()
    bare_interface.enable_modify_pair_registration = MagicMock()
    bare_interface.select_tab = MagicMock()
    bare_interface.reg.state = RegState.PAIRS_REG
    bare_interface.reg.is_pairs_registered.side_effect = (
        lambda: bare_interface.reg.state.value >= RegState.PAIRS_REG.value)
    states_viewed = []
    bare_interface.update_views = MagicMock(
        side_effect=lambda **kwargs: states_viewed.append(bare_interface.reg.state))

    bare_interface.pre_processing_process()

    assert states_viewed == [RegState.SIMS_INIT]


def test_only_one_operation_bar_at_a_time(bare_interface, monkeypatch):
    """Two bars must never be on screen together: an operation started while another is running
    reports into the running one, and leaves it the activity dock and the widget state."""
    dock = MagicMock()
    widgets = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(interface_module, "VisibleActivityDock", dock)
    monkeypatch.setattr(bare_interface, "_operation_widgets", widgets)
    bare_interface.viewer = MagicMock()

    with bare_interface._operation_progress('Initialising sources') as outer:
        with bare_interface._operation_progress('Refreshing view') as inner:
            assert inner is outer

    dock.assert_called_once_with(bare_interface.viewer)
    widgets.assert_called_once_with()
    # ...and the next operation, once this one has finished, opens a bar of its own again
    assert bare_interface._running_operation is None


def test_update_metadata_source_refresh_reports_its_own_bar(bare_interface, monkeypatch):
    """The refresh at the end of update_metadata_source() must open its own bar. It used to be
    handed the source-initialisation factory, which by then belonged to a finished operation -
    so re-running input/output showed no bar for the refresh at all."""
    bare_interface.viewer = MagicMock()
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.is_initialised.return_value = True
    bare_interface.reg.source_transform_key = 'source_metadata'
    bare_interface.source_metadata = {}
    bare_interface.update_views = MagicMock()
    for name in ['populate_channels', 'populate_coordinate_systems', 'populate_channels_table',
                 'populate_metadata_table', 'check_3d_view']:
        setattr(bare_interface, name, MagicMock())
    bare_interface.update_output_channels = MagicMock(return_value=False)

    assert bare_interface.update_metadata_source() is True

    bare_interface.update_views.assert_called_once_with(show_images=False)


def test_run_off_thread_runs_the_work_elsewhere_and_keeps_qt_free(make_napari_viewer):
    """The heavy calls must not run on the Qt thread: that is what froze the window for as long
    as a registration or fusion took. _run_off_thread() runs them on a worker while a nested
    event loop keeps Qt going, and hands back what they returned."""
    import threading

    viewer = make_napari_viewer()
    interface = Interface.__new__(Interface)
    interface.viewer = viewer
    interface.enable_plugin_widget = None

    ticks = []
    timer = interface_module.QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))

    qt_thread = threading.current_thread().ident
    work_thread = {}

    def work(worker_factory):
        work_thread['ident'] = threading.current_thread().ident
        with worker_factory(total=4) as pbar:
            for _ in range(4):
                pbar.update(1)
                time.sleep(0.02)
        return 'result'

    with interface._operation_progress('Work') as factory:
        timer.start()
        result = interface._run_off_thread(work, factory)
        timer.stop()

    assert result == 'result'
    assert work_thread['ident'] != qt_thread
    # the event loop kept running while the work did - a frozen window ticks not at all
    assert ticks


def test_run_off_thread_propagates_failures_to_the_caller(make_napari_viewer):
    """A failure on the worker must surface where the call was made (and so reach
    @catch_run_errors), not be re-raised inside the Qt event loop where nothing catches it."""
    viewer = make_napari_viewer()
    interface = Interface.__new__(Interface)
    interface.viewer = viewer
    interface.enable_plugin_widget = None

    def work(_):
        raise ValueError('boom')

    with interface._operation_progress('Work') as factory:
        with pytest.raises(ValueError, match='boom'):
            interface._run_off_thread(work, factory)


def test_run_off_thread_runs_inline_without_a_qt_application(bare_interface, monkeypatch):
    """Headless (tests, any non-GUI caller): there is no event loop to keep alive, so the work
    simply runs here rather than going near a worker."""
    monkeypatch.setattr(interface_module.QApplication, 'instance', staticmethod(lambda: None))

    assert bare_interface._run_off_thread(lambda factory: (factory, 'ran'), 'the-factory') == (
        'the-factory', 'ran')


def _registered_view_interface(bare_interface, lazy_overview):
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}]
    bare_interface.reg.fileset_label = "sample"
    bare_interface._create_napari_shapes = MagicMock(return_value=([], [], [], []))
    bare_interface._create_lazy_overview = MagicMock(return_value=lazy_overview)
    bare_interface._create_napari_data = MagicMock(return_value="overview")
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()
    bare_interface._napari_view_add_fused_data = MagicMock()


def test_update_views_draws_the_registered_sections_lazily(bare_interface):
    """After registration the main view reads each section when viewed, as before it, rather than pasting every source."""
    _registered_view_interface(bare_interface, lazy_overview="lazy")

    bare_interface.update_views(transform_key="registered")

    assert bare_interface._create_lazy_overview.call_args.args == ("registered",)
    bare_interface._create_napari_data.assert_not_called()
    assert bare_interface._napari_view_add_fused_data.call_args.args[1] == "lazy"


@pytest.mark.parametrize('running, reply, expect_process, expect_cancel', [
    (False, None, True, False),
    (True, QMessageBox.Yes, False, True),
    (True, QMessageBox.No, False, False),
])
def test_process_button_runs_or_after_confirmation_cancels(bare_interface, monkeypatch, running, reply,
                                                             expect_process, expect_cancel):
    """While an operation runs the Process button reads Cancel: it asks first, and cancels only on yes."""
    process = MagicMock()
    cancelled = MagicMock()
    monkeypatch.setattr(interface_module, 'request_cancel', cancelled)
    monkeypatch.setattr(interface_module.QMessageBox, 'question', MagicMock(return_value=reply))
    bare_interface._running_operation = SimpleNamespace(desc='Pair registration') if running else None

    bare_interface.process_or_cancel(process)

    assert process.called == expect_process
    assert cancelled.called == expect_cancel


def test_a_process_failure_is_logged_and_reported(bare_interface, monkeypatch, caplog):
    """A failure outside the run_*() steps (e.g. refreshing the view after registration) still reaches the log."""
    reported = MagicMock()
    monkeypatch.setattr(interface_module, 'report_failure', reported)
    bare_interface._running_operation = None
    process = MagicMock(side_effect=KeyError('missing'), __name__='registration_process')

    with caplog.at_level(logging.ERROR):
        bare_interface.process_or_cancel(process)

    assert 'registration_process failed' in caplog.text
    assert reported.call_args.args[0] == 'Registration process'
    assert isinstance(reported.call_args.args[1], KeyError)


def test_a_cancelled_global_registration_puts_back_the_sources_transforms(bare_interface, monkeypatch,
                                                                           mocked_activity_contexts):
    """It writes the registered transform onto the sources before its metrics: cancelled, they get back
    what they had, and nothing is saved."""
    from muvis_align.util import OperationCancelled

    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.reg.reg_transform_key = 'registered'
    bare_interface.reg.pair_msims = ['pair-msim']
    bare_interface.reg.msims = ['msim']
    monkeypatch.setattr(interface_module, 'snapshot_msims_transform', lambda msims, key: f'snapshot of {msims}')
    restored = []
    monkeypatch.setattr(interface_module, 'restore_msims_transform',
                        lambda msims, key, snapshot: restored.append((msims, key, snapshot)))
    bare_interface.reg.register_global.side_effect = OperationCancelled('Cancelled')

    assert bare_interface.run_global_registration() is None
    assert restored == [(['pair-msim'], 'registered', "snapshot of ['pair-msim']"),
                        (['msim'], 'registered', "snapshot of ['msim']")]
    bare_interface.reg.save_mappings.assert_not_called()


def test_a_cancelled_fusion_removes_its_partial_output(bare_interface, monkeypatch, mocked_activity_contexts,
                                                        tmp_path):
    from muvis_align.util import OperationCancelled

    bare_interface.viewer = MagicMock()
    bare_interface.params = {'registration': {'operation': 'register'},
                             'fusion': {'tile_size': '', 'method': 'average', 'spacing': 'mean', 'ome_version': '0.5'},
                             'input_output': {'registration_dimension': 'space'}}
    bare_interface.reg.output = str(tmp_path) + '/'
    partial = tmp_path / ('registered' + interface_module.zarr_extension)
    partial.mkdir()
    (partial / 'chunk').write_text('partial')
    bare_interface.get_best_transform_key = MagicMock(return_value='registered')
    bare_interface.reg.fuse.side_effect = OperationCancelled('Cancelled')

    assert bare_interface.run_fusion() is None
    assert not partial.exists()


def test_metrics_table_lists_split_group_pairs_after_the_tile_pairs(bare_interface):
    bare_interface.reg.file_labels = ['S000_000', 'S000_001']
    bare_interface.reg.positions = [{'y': 0, 'x': 0}, {'y': 0, 'x': 1}]
    table = MagicMock()
    bare_interface.param_widgets = {'registration.metrics_table': table}
    metrics = {'summary': {'registered': {'quality': 0.5, 'ncc': 0.4}},
               'pairs': {(0, 1): {'registered': {'quality': 0.9, 'ncc': 0.8}}},
               'group_pairs': {('S000', 'S001'): {'registered': {'quality': 0.3}}}}

    bare_interface.populate_metrics_table(metrics)

    values, rows, columns = table.set_value.call_args[0][0]
    assert rows == ['summary', 'S000_000 - S000_001', 'S000 - S001']
    assert columns == ['quality', 'ncc']
    assert values[2] == [0.3, None]


@pytest.mark.parametrize(("row", "selected"), [(0, None), (1, ('S000_001', 'S000_000')), (2, None), (-1, None)],
                         ids=["summary", "tile-pair", "group-pair", "none"])
def test_selecting_a_metrics_row_picks_its_pair_for_preview(bare_interface, row, selected):
    """Only a tile pair row names two images; the summary and section pair rows leave the selection."""
    bare_interface.reg.file_labels = ['S000_000', 'S000_001']
    bare_interface.reg.positions = [{'y': 0, 'x': 0}, {'y': 0, 'x': 1}]
    table = MagicMock()
    image1, image2 = MagicMock(), MagicMock()
    bare_interface.param_widgets = {'registration.metrics_table': table,
                                    'registration.reg_preview_image1': image1,
                                    'registration.reg_preview_image2': image2}
    bare_interface.populate_metrics_table(
        {'summary': {'registered': {'quality': 0.5}},
         'pairs': {(1, 0): {'registered': {'quality': 0.9}}},
         'group_pairs': {('S000', 'S001'): {'registered': {'quality': 0.3}}}})
    _, rows, _ = table.set_value.call_args[0][0]
    table.widget.native.currentRow.return_value = row
    table.widget.native.verticalHeaderItem.side_effect = (
        lambda index: SimpleNamespace(text=lambda: rows[index]) if index >= 0 else None)

    bare_interface.metrics_table_selected()

    if selected is None:
        assert not image1.set_value.called and not image2.set_value.called
    else:
        image1.set_value.assert_called_once_with(selected[0])
        image2.set_value.assert_called_once_with(selected[1])


def test_the_project_file_is_copied_into_the_output_folder_as_an_action_starts(bare_interface, tmp_path):
    project = tmp_path / "project.yml"
    project.write_text("input_output:\n  output_path: output\n")
    bare_interface.params_path = str(project)
    bare_interface.params = {"input_output": {"output_path": "output"}}

    bare_interface.copy_params_to_output()

    assert (tmp_path / "output" / "project.yml").read_text() == project.read_text()


@pytest.mark.parametrize(
    ("size", "positions", "shown", "expected_z"),
    [
        ({'y': 10, 'x': 10}, [0.0, 2.5, 2.5], [1, 2], 2.5),
        ({'y': 10, 'x': 10}, [0.0, 2.5, 2.5], None, None),
        ({'z': 5, 'y': 10, 'x': 10}, [2.5], [0], None),
    ],
    ids=["section-shown", "no-section-shown", "z-stack"],
)
def test_the_view_stays_on_the_section_shown_while_the_rest_were_read(bare_interface, size, positions, shown,
                                                                      expected_z):
    """Without a section shown, or for z-stacks, the view is left where napari put it."""
    bare_interface.viewer = MagicMock()
    bare_interface.viewer.dims.ndim = 3
    bare_interface.reg.sources = [MagicMock()] * len(positions)
    bare_interface.reg.sources[0].get_size.return_value = size
    bare_interface.reg.positions = [{'z': z_position} for z_position in positions]
    bare_interface._shown_section_indices = shown

    bare_interface._go_to_shown_section()

    if expected_z is None:
        bare_interface.viewer.dims.set_point.assert_not_called()
    else:
        bare_interface.viewer.dims.set_point.assert_called_once_with(0, expected_z)


def test_replacing_the_main_view_clears_napari_dask_cache_but_the_overview_does_not(bare_interface):
    bare_interface.viewer = MagicMock()
    bare_interface.viewer.layers = [object()]
    overview = MagicMock()
    overview.layers = [object()]

    with patch.object(interface_module, 'clear_napari_dask_cache') as clear_cache:
        bare_interface._clear_napari_view(overview)
        clear_cache.assert_not_called()
        bare_interface._clear_napari_view(bare_interface.viewer)
        clear_cache.assert_called_once_with()
    assert not bare_interface.viewer.layers


# pre-processing's bar is sized by the phases that will actually report
def _run_pre_processing(build_pending, eager=False):
    """Returns (phases reserved, weight the msim build was given)."""
    from contextlib import contextmanager

    interface = interface_module.Interface.__new__(interface_module.Interface)
    interface.params = {'pre_processing': {'scale': 2}}
    interface.reg = MagicMock()
    interface.reg.msims_build_pending.return_value = build_pending
    interface.reg.has_eager_pre_processing.return_value = eager
    interface.reg.preprocess.return_value = (None, None, True)
    interface._timing_verbose = lambda: False
    interface._run_off_thread = lambda work, factory: work(factory)

    declared = {}

    class _Factory:
        def __call__(self, *args, **kwargs):
            raise AssertionError('no phase should be opened by the mocked work')

    @contextmanager
    def operation_progress(desc, progress_factory=None, phases=1):
        declared['phases'] = phases
        yield _Factory()

    interface._operation_progress = operation_progress
    interface.run_pre_processing()
    return declared['phases'], interface.reg.ensure_msims.call_args.kwargs['weight']


def test_only_the_phases_that_will_run_are_reserved():
    assert _run_pre_processing(build_pending=False)[0] == 1


def test_the_build_takes_the_bar_in_proportion_to_what_it_costs():
    # opening every source is nearly all of a run whose only step is scaling - splitting the
    # bar evenly with it left the work finishing at the halfway mark
    phases, weight = _run_pre_processing(build_pending=True)
    assert (phases, weight) == (9, 8)

    # a step that computes over the data makes the rest of the run real work again
    phases, weight = _run_pre_processing(build_pending=True, eager=True)
    assert (phases, weight) == (3, 2)
