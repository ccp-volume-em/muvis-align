"""
Parameterized napari integration tests for the Interface registration workflow.

Tests load project configuration files and simulate the full registration workflow:
1. Project initialization
2. Input/output setup
3. Pair registration
4. Global registration
5. Image fusion

Each test is parameterized to run with different project configurations
(muvis_align_project.yml, muvis_align_project2.yml, etc.).
"""

import os
import tempfile
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import pytest
import yaml
import numpy as np
from qtpy.QtWidgets import QMessageBox

from src.muvis_align._widget import MainWidget
from src.muvis_align.ui.Interface import Interface, ViewMode
from src.muvis_align.MVSRegistration import RegState


@pytest.fixture(autouse=True)
def suppress_completion_dialogs():
    """Keep workflow completion messages from blocking test execution."""
    with patch(
        'src.muvis_align.ui.Interface.QMessageBox.information'
    ) as mock_information:
        yield mock_information


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


@pytest.fixture(params=get_project_configs(), ids=lambda p: p.name)
def project_config(request):
    """Fixture that provides path to each discovered project configuration file."""
    config_path = request.param
    assert config_path.exists(), f"Project config not found: {config_path}"
    return config_path


@pytest.fixture
def config_data(project_config):
    """Load and parse project configuration YAML."""
    with open(project_config, 'r') as f:
        data = yaml.safe_load(f)
    return data


class TestNapariInterfaceRegistration:
    """Test suite for napari Interface registration workflow with different configs."""

    def test_napari_interface_instantiation(self, make_napari_viewer, project_config):
        """Test that MainWidget and Interface can be instantiated with project config."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget') as mock_viewer_widget:
            with patch.object(viewer.window, 'add_dock_widget'):
                # Mock the overview with necessary attributes
                mock_overview = MagicMock()
                mock_viewer_widget.return_value = mock_overview
                
                # Mock viewer layers
                viewer.layers.clear = MagicMock()
                
                # Mock widget creation functions to avoid magicgui complexity
                with patch('src.muvis_align.ui.create_widgets.create_project_widget') as mock_proj:
                    with patch('src.muvis_align.ui.create_widgets.create_template_widgets') as mock_tmpl:
                        mock_proj.return_value = MagicMock()
                        mock_tmpl.return_value = {}
                        
                        try:
                            main_widget = MainWidget(viewer)
                            assert main_widget is not None, "MainWidget should be created"
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            assert False, f"Failed: {type(e).__name__}: {str(e)}\n{tb[:200]}"

    def test_project_config_loading(self, make_napari_viewer, project_config, config_data):
        """Test that project configuration can be loaded into Interface."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                main_widget = MainWidget(viewer)
                interface = main_widget.interface
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_copy = Path(tmpdir) / project_config.name
            config_copy.write_text(project_config.read_text())
            
            interface.project_path(str(config_copy))
            
            assert interface.params_path == str(config_copy)
            assert interface.params is not None
            assert 'registration' in interface.params
            assert 'fusion' in interface.params
            assert 'input_output' in interface.params
            assert 'pre_processing' in interface.params

    def test_project_params_structure(self, config_data):
        """Test that project configuration has expected structure."""
        assert isinstance(config_data, dict), "Config should be a dictionary"
        
        required_sections = ['registration', 'fusion', 'input_output']
        for section in required_sections:
            assert section in config_data, f"Missing required section: {section}"
        
        registration = config_data['registration']
        assert 'method' in registration
        assert 'pairing' in registration
        assert 'transform_type' in registration
        assert 'operation' in registration
        
        fusion = config_data['fusion']
        assert 'method' in fusion
        assert 'spacing' in fusion
        assert 'tile_size' in fusion
        
        input_output = config_data['input_output']
        assert 'input_path' in input_output
        assert 'output_path' in input_output
        assert 'preview_scale' in input_output

    def test_interface_reset(self, make_napari_viewer, project_config):
        """Test that Interface reset clears state properly."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                main_widget = MainWidget(viewer)
                interface = main_widget.interface
        
        interface.reset()
        
        assert hasattr(interface, 'source_metadata'), "Should have source_metadata attribute"
        assert interface.source_metadata == {}, "source_metadata should be empty after reset"
        assert interface.view_mode is None, "view_mode should be None after reset"
        assert interface.selected_shape_index is None, "selected_shape_index should be None after reset"
        assert hasattr(interface.reg, 'state'), "reg should have state attribute"

    def test_interface_tab_management(self, make_napari_viewer, project_config):
        """Test tab enabling/disabling based on registration state."""
        viewer = make_napari_viewer()
        enable_tabs_mock = MagicMock()
        select_tab_mock = MagicMock()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(
                viewer,
                MagicMock(),
                enable_tabs_mock,
                select_tab_mock,
                verbose=False
            )
        
        # Initial state: all tabs disabled except first
        interface.enable_tabs(False, 1)
        enable_tabs_mock.assert_called_with(False, 1)
        
        # After pair registration: enable up to tab 3
        interface.enable_tabs(True, 3)
        enable_tabs_mock.assert_called_with(True, 3)
        
        # Select specific tab
        interface.select_tab(2)
        select_tab_mock.assert_called_with(2)

    def test_interface_viewer_management(self, make_napari_viewer, project_config):
        """Test napari viewer layer management (clear, add)."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        # Add some test layers
        viewer.add_image(np.random.random((100, 100)), name='test1')
        viewer.add_image(np.random.random((100, 100)), name='test2')
        assert len(viewer.layers) == 2
        
        # Test clear
        interface._clear_napari_view(viewer)
        assert len(viewer.layers) == 0

    def test_interface_view_modes(self, make_napari_viewer, project_config):
        """Test view mode transitions."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        assert interface.view_mode is None
        
        interface.view_mode = ViewMode.OVERVIEW
        assert interface.view_mode == ViewMode.OVERVIEW
        
        interface.view_mode = ViewMode.PAIRS
        assert interface.view_mode == ViewMode.PAIRS
        
        interface.view_mode = ViewMode.FEATURES
        assert interface.view_mode == ViewMode.FEATURES
        
        interface.view_mode = ViewMode.FUSED
        assert interface.view_mode == ViewMode.FUSED

    def test_registration_params_validation(self, config_data):
        """Validate registration parameters in config."""
        registration = config_data['registration']
        
        assert registration['method'] in ['sift', 'orb', 'akaze']
        assert registration['pairing'] in ['orthogonal', 'all']
        assert registration['transform_type'] in ['rigid', 'affine']
        assert registration['operation'] == 'register'
        assert isinstance(registration['max_keypoints'], int)
        assert isinstance(registration['ransac_iterations'], int)

    def test_fusion_params_validation(self, config_data):
        """Validate fusion parameters in config."""
        fusion = config_data['fusion']
        
        assert fusion['method'] in ['average', 'max', 'min']
        assert fusion['spacing'] in ['mean', 'min']
        assert isinstance(fusion['tile_size'], str)
        assert fusion['ome_version'] in ['0.4', '0.5']

    def test_input_output_params_validation(self, config_data):
        """Validate input/output parameters in config."""
        input_output = config_data['input_output']
        
        assert isinstance(input_output['input_path'], str)
        assert isinstance(input_output['output_path'], str)
        assert isinstance(input_output['preview_scale'], (int, float))
        assert isinstance(input_output['overwrite'], bool)
        assert isinstance(input_output['preview_images'], bool)
        assert isinstance(input_output['preview_shapes'], bool)

    def test_preprocessing_params_validation(self, config_data):
        """Validate pre-processing parameters in config."""
        preprocessing = config_data.get('pre_processing', {})
        
        assert 'scale' in preprocessing
        assert isinstance(preprocessing['scale'], (int, float))
        assert preprocessing['scale'] > 0

    @patch('src.muvis_align.ui.Interface.QMessageBox.question')
    def test_interface_pair_registration_mock(
        self, mock_question, make_napari_viewer, project_config
    ):
        """Test pair_registration method with mocked UI components and bbox handling."""
        viewer = make_napari_viewer()
        mock_question.return_value = True  # Simulate "Yes" click
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        with patch.object(interface.reg, 'is_global_registered', return_value=False):
            with patch.object(interface.reg, 'is_pairs_registered', return_value=False):
                with patch('src.muvis_align.ui.Interface.TqdmCallback'):
                    with patch('src.muvis_align.ui.Interface.TemporarilyDisabledWidgets'):
                        with patch('src.muvis_align.ui.Interface.VisibleActivityDock'):
                            with patch.object(interface, 'get_all_widgets', return_value={}):
                                # Create mock bbox DataArray WITHOUT 't' dimension (this is the key test)
                                import xarray as xr
                                import networkx as nx
                                mock_bbox = xr.DataArray(
                                    [[1, 2], [3, 4]],
                                    dims=['x_in', 'x_out'],
                                    coords={'x_in': [0, 1], 'x_out': [0, 1]}
                                )
                                
                                with patch.object(interface.reg, 'register_pairs', return_value={
                                    'pair_mappings': {},
                                    'metrics': {'pairs': {}}
                                }):
                                    # Mock nx.get_edge_attributes to return bbox without 't' dimension
                                    with patch('networkx.get_edge_attributes') as mock_get_attrs:
                                        mock_get_attrs.return_value = {('key1', 'key2'): mock_bbox}
                                        
                                        with patch.object(interface.reg, 'save_pair_mappings'):
                                            with patch.object(interface, 'update_registered'):
                                                # This should not raise KeyError
                                                interface.pair_registration()

    @patch('src.muvis_align.ui.Interface.QMessageBox.question')
    def test_interface_registration_process_mock(
        self, mock_question, make_napari_viewer, project_config
    ):
        """Test registration_process method with mocked UI components."""
        viewer = make_napari_viewer()
        mock_question.return_value = True  # Simulate "Yes" click
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
            interface.params = {'registration': {}}
        
        with patch.object(interface.reg, 'is_pairs_registered', return_value=True):
            with patch.object(interface.reg, 'is_global_registered', return_value=False):
                with patch.object(interface.reg, 'register_global', return_value={
                    'mappings': {},
                    'metrics': {}
                }):
                    with patch('src.muvis_align.ui.Interface.TqdmCallback'):
                        with patch('src.muvis_align.ui.Interface.TemporarilyDisabledWidgets'):
                            with patch('src.muvis_align.ui.Interface.VisibleActivityDock'):
                                with patch.object(interface, 'get_all_widgets', return_value={}):
                                    with patch.object(interface.reg, 'save_mappings'):
                                        with patch.object(interface.reg, 'save_metrics'):
                                            with patch.object(interface, 'enable_tabs'):
                                                with patch.object(interface, 'update_registered'):
                                                    interface.registration_process()

    @patch('src.muvis_align.ui.Interface.QMessageBox.question')
    def test_interface_fusion_process_mock(
        self, mock_question, make_napari_viewer, project_config
    ):
        """Test fusion_process method with mocked UI components."""
        viewer = make_napari_viewer()
        mock_question.return_value = True  # Simulate "Yes" click
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
            interface.params = {
                'registration': {'operation': 'register'},
                'fusion': {
                    'method': 'average',
                    'spacing': 'mean',
                    'tile_size': '1024,1024',
                    'ome_version': '0.5'
                }
            }
        
        with patch.object(interface.reg, 'is_fused', return_value=False):
            with patch('src.muvis_align.ui.Interface.TqdmCallback'):
                with patch('src.muvis_align.ui.Interface.TemporarilyDisabledWidgets'):
                    with patch('src.muvis_align.ui.Interface.VisibleActivityDock'):
                        with patch.object(interface, 'get_all_widgets', return_value={}):
                            with patch.object(interface.reg, 'fuse', return_value=(MagicMock(), None)):
                                with patch.object(interface, '_clear_napari_view'):
                                    with patch.object(interface, '_napari_view_add_image'):
                                        interface.fusion_process()

    def test_registration_state_transitions(self, make_napari_viewer, project_config):
        """Test valid registration state transitions."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        # Verify state progression methods
        assert not interface.reg.is_pairs_registered()
        assert not interface.reg.is_global_registered()
        assert not interface.reg.is_fused()
        
        # Simulate state transitions
        interface.reg.state = RegState.PAIRS_REG
        assert interface.reg.is_pairs_registered()
        assert not interface.reg.is_global_registered()
        
        interface.reg.state = RegState.GLOBAL_REG
        assert interface.reg.is_pairs_registered()
        assert interface.reg.is_global_registered()
        
        interface.reg.state = RegState.FUSED
        assert interface.reg.is_pairs_registered()
        assert interface.reg.is_global_registered()
        assert interface.reg.is_fused()

    def test_metrics_methods_available(self, make_napari_viewer, project_config):
        """Test that expected metrics methods are available."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        assert interface.metrics_methods == ['ncc', 'ssim', 'onmi']
        assert len(interface.metrics_methods) > 0
        for metric in interface.metrics_methods:
            assert isinstance(metric, str)

    def test_interface_initialization_with_template(self, make_napari_viewer, project_config):
        """Test that Interface properly initializes with project template."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        assert interface.raw_template is not None
        assert interface.template is not None
        assert isinstance(interface.template, dict)

    @patch('src.muvis_align.ui.Interface.QMessageBox.question')
    def test_modify_pair_registration_with_bbox(self, mock_question, make_napari_viewer, project_config):
        """Test modify_pair_registration with bbox handling (no 't' dimension)."""
        viewer = make_napari_viewer()
        mock_question.return_value = QMessageBox.Yes  # Simulate "Yes" click

        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())

        interface.view_mode = ViewMode.PAIRS
        interface.pair_indices = ('key1', 'key2')
        interface.reg.pairs_graph = object()
        interface.reg.source_transform_key = 'source_metadata'
        
        # Mock the temp_widget_state that gets called in modify_pair_registration
        interface.temp_widget_state = MagicMock()

        with patch.object(interface, 'calc_mod_pair_transform') as mock_calc:
            with patch('networkx.get_edge_attributes') as mock_get_attrs:
                # Create mock bbox DataArray WITHOUT 't' dimension
                import xarray as xr
                mock_bbox = xr.DataArray(
                    [[1, 2], [3, 4]],
                    dims=['x_in', 'x_out'],
                    coords={'x_in': [0, 1], 'x_out': [0, 1]}
                )
                
                # Mock transform with 't' dimension for the pair_transforms
                mock_transform_with_t = xr.DataArray(
                    [[[1, 0], [0, 1], [0, 0]]],
                    dims=['t', 'rows', 'cols'],
                    coords={'t': [0]}
                )
                
                # Set up return values - first call gets pair_transforms, second gets qualities, third gets bboxes
                mock_get_attrs.side_effect = [
                    {interface.pair_indices: mock_transform_with_t},  # pair_transforms
                    {interface.pair_indices: 0.95},  # qualities
                    {interface.pair_indices: mock_bbox}  # bboxes (without 't' dimension)
                ]
                
                mock_calc.return_value = mock_transform_with_t.sel(t=0)
                
                with patch('networkx.set_edge_attributes'):
                    with patch.object(interface.reg, 'save_pair_mappings') as mock_save:
                        with patch.object(interface, 'update_registered'):
                            # This should not raise KeyError
                            interface.modify_pair_registration()
                            
                            # Verify save_pair_mappings was called
                            assert mock_save.called

    def test_global_registration_with_dimension_mismatch(self, make_napari_viewer, project_config):
        """Test update_registered handles sims with transforms that include a t dimension."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        # Create mock sims with different transform dimensions
        import xarray as xr
        import numpy as np
        
        # Source sims with 't' dimension in transform
        mock_sim_with_t = MagicMock()
        mock_sim_with_t.dims = ('t', 'c', 'y', 'x')
        mock_sim_with_t.attrs = {
            'transforms': {
                'registered': xr.DataArray(
                    np.eye(3).reshape(1, 3, 3),
                    dims=['t', 'x_in', 'x_out'],
                    coords={'t': [0], 'x_in': ['y', 'x', '1'], 'x_out': ['y', 'x', '1']}
                )
            }
        }
        
        interface.reg.sims = [mock_sim_with_t]
        interface.preview_sims = [MagicMock()]
        
        # Mock the missing reg_transform_key attribute on MVSRegistration
        interface.reg.reg_transform_key = 'registered'
        
        with patch('src.muvis_align.ui.Interface.si_utils.get_tranform_keys_from_sim', return_value=['registered']):
            with patch.object(interface, 'populate_coordinate_systems') as mock_pop_coord:
                with patch.object(interface, 'populate_metadata_table') as mock_pop_meta:
                    with patch.object(interface, 'populate_metrics_table') as mock_pop_metrics:
                        with patch.object(interface, 'update_views') as mock_views:
                            # This should not raise KeyError about dimension mismatch
                            interface.update_registered()

                            assert mock_pop_coord.called
                            assert mock_pop_meta.called
                            assert mock_pop_metrics.called
                            mock_views.assert_called_once_with(transform_key=None)


class TestProjectConfigurationFiles:
    """Test suite for project configuration file validation."""

    def test_all_configs_valid_yaml(self, project_config):
        """Test that all project configs are valid YAML."""
        with open(project_config, 'r') as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_config_file_exists(self, project_config):
        """Test that config file exists and is readable."""
        assert project_config.exists()
        assert project_config.is_file()
        assert os.access(project_config, os.R_OK)

    def test_config_has_required_keys(self, config_data):
        """Test that config has all required top-level keys."""
        required_keys = ['registration', 'fusion', 'input_output']
        for key in required_keys:
            assert key in config_data, f"Missing required key: {key}"


class TestMainWidgetIntegration:
    """Test suite for MainWidget integration with different configs."""

    def test_main_widget_creation(self, make_napari_viewer, project_config):
        """Test MainWidget creation with napari viewer."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                widget = MainWidget(viewer)
        
        assert widget is not None
        assert hasattr(widget, 'interface')
        assert hasattr(widget, 'viewer')
        assert widget.viewer is viewer

    def test_main_widget_tab_creation(self, make_napari_viewer, project_config):
        """Test that MainWidget creates tabs correctly."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                widget = MainWidget(viewer)
        
        assert widget.count() > 0
        assert len(widget.tab_labels) > 0
        assert 'project' in widget.tab_labels

    def test_main_widget_tab_disabled_initially(self, make_napari_viewer, project_config):
        """Test that non-project tabs are disabled initially."""
        viewer = make_napari_viewer()
        
        with patch('src.muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                widget = MainWidget(viewer)
        
        # Project tab should be enabled
        assert widget.isTabEnabled(0)
        
        # Other tabs should be disabled initially
        if widget.count() > 1:
            assert not widget.isTabEnabled(1)


# Use the canonical package name for unit-level coverage.  The older integration
# tests above intentionally retain their historical ``src.muvis_align`` imports.
interface_module = importlib.import_module("muvis_align.ui.Interface")
CanonicalInterface = interface_module.Interface
CanonicalViewMode = interface_module.ViewMode
CanonicalRegState = interface_module.RegState


@pytest.fixture
def bare_interface():
    interface = CanonicalInterface.__new__(CanonicalInterface)
    interface.reg = MagicMock()
    interface.extra_metadata = {}
    interface.param_widgets = {}
    interface.reg.file_labels = ["image-0"]
    interface.preview_sims = [
        SimpleNamespace(dims=("z", "y", "x"), sizes={"z": 2})
    ]
    return interface


def test_change_param_updates_nested_value_and_writes(bare_interface):
    bare_interface.params = {}
    bare_interface.write_params = MagicMock()

    bare_interface.change_param("registration.method", "phase")

    assert bare_interface.params == {
        "registration": {"method": "phase"}
    }
    bare_interface.write_params.assert_called_once_with()


def test_tab_changed_clears_feature_view_and_stops_timer(bare_interface):
    bare_interface.viewer = MagicMock()
    bare_interface.view_mode = CanonicalViewMode.FEATURES
    bare_interface.pair_metrics_timer = MagicMock()
    bare_interface._clear_napari_view = MagicMock()

    bare_interface.tab_changed("fusion")

    bare_interface._clear_napari_view.assert_called_once_with(
        bare_interface.viewer
    )
    bare_interface.pair_metrics_timer.stop.assert_called_once_with()
    assert bare_interface.view_mode is None


@pytest.mark.parametrize(
    ("method_name", "section", "axis"),
    [
        ("source_position_z", "position", "z"),
        ("source_position_y", "position", "y"),
        ("source_position_x", "position", "x"),
        ("source_scale_z", "scale", "z"),
        ("source_scale_y", "scale", "y"),
        ("source_scale_x", "scale", "x"),
    ],
)
def test_source_metadata_setters(
    bare_interface, method_name, section, axis
):
    bare_interface.source_metadata = {}

    getattr(bare_interface, method_name)(2.5)

    assert bare_interface.source_metadata[section][axis] == 2.5


def test_source_rotation_sets_valid_value(bare_interface):
    bare_interface.source_metadata = {}

    bare_interface.source_rotation(12.5)

    assert bare_interface.source_metadata["rotation"] == 12.5


@pytest.mark.parametrize("exists", [True, False], ids=["existing", "new"])
def test_project_path_handles_existing_and_new_projects(
    bare_interface, monkeypatch, exists
):
    bare_interface.template = {"template": True}
    bare_interface.reset = MagicMock()
    bare_interface.update_widgets = MagicMock()
    bare_interface.write_params = MagicMock()
    bare_interface.update_input_output_path = MagicMock()
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
    if exists:
        bare_interface.update_widgets.assert_called_once_with()
        bare_interface.write_params.assert_not_called()
    else:
        bare_interface.write_params.assert_called_once_with()
        bare_interface.update_input_output_path.assert_called_once_with()


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
    bare_interface.reg.file_labels = ["left", "right"]

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
    image1_widget.set_value.assert_called_once_with(
        "left", choices=["left", "right"]
    )
    image2_widget.set_value.assert_called_once_with(
        "right", choices=["left", "right"]
    )


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
    bare_interface.reg.sims = [object()]
    monkeypatch.setattr(
        interface_module, "get_transforms", lambda _: transforms
    )

    assert bare_interface.get_best_transform_key() == expected


def test_update_views_adds_enabled_preview_layers(
    bare_interface, monkeypatch
):
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sims = bare_interface.preview_sims
    bare_interface.params = {
        "input_output": {
            "preview_images": True,
            "preview_shapes": True,
        }
    }
    bare_interface.reg.fileset_label = "sample"
    bare_interface.get_best_transform_key = MagicMock(
        return_value="registered"
    )
    bare_interface._clear_napari_view = MagicMock()
    shapes = [np.zeros((4, 2))]
    shape_data = (shapes, ["0"], ["image-0"], [(1, 1, 1)])
    image_data = object()
    bare_interface._create_napari_shapes = MagicMock(
        return_value=shape_data
    )
    bare_interface._create_napari_data = MagicMock(
        return_value=image_data
    )
    bare_interface._napari_view_add_data = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda _: {"z": 0},
    )

    bare_interface.update_views(show_preprocessed=True)

    assert bare_interface._clear_napari_view.call_args_list == [
        call(bare_interface.viewer),
        call(bare_interface.overview),
    ]
    assert bare_interface._create_napari_shapes.call_args_list == [
        call("registered", force_2d=False),
        call("registered", force_2d=True),
    ]
    bare_interface._create_napari_data.assert_called_once_with(
        "registered",
        show_preprocessed=True,
    )
    bare_interface._napari_view_add_data.assert_called_once_with(
        bare_interface.viewer, image_data, "sample data"
    )
    expected_shape_call = (
        shapes, ["0"], ["image-0"], [(1, 1, 1)], "sample shapes"
    )
    bare_interface._update_view_add_shapes.assert_any_call(
        bare_interface.viewer,
        *expected_shape_call,
    )
    bare_interface._update_view_add_shapes.assert_any_call(
        bare_interface.overview,
        *expected_shape_call,
    )
    assert bare_interface._update_view_add_shapes.call_count == 2
    assert bare_interface.view_mode is CanonicalViewMode.OVERVIEW


def test_update_napari_shapes_adds_3d_box_with_overlap_metadata(
    bare_interface, monkeypatch
):
    viewer = MagicMock()
    image_shape = np.zeros((8, 3))
    overlap_shape = np.ones((8, 3))
    bbox_layer = MagicMock()
    bare_interface.reg.get_metrics.return_value = 0.75
    create_shapes = MagicMock(return_value=[image_shape])
    create_overlaps = MagicMock(
        return_value=([overlap_shape], [np.array([0, 0])])
    )
    bounding_box_layer = MagicMock(return_value=bbox_layer)
    set_edges = MagicMock()
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda _: {"z": 0},
    )
    monkeypatch.setattr(interface_module, "create_sim_shapes", create_shapes)
    monkeypatch.setattr(
        interface_module, "create_overlap_shapes", create_overlaps
    )
    monkeypatch.setattr(
        interface_module, "metric_to_rgb", lambda _: (0.1, 0.2, 0.3)
    )
    monkeypatch.setattr(
        interface_module, "BoundingBoxLayer", bounding_box_layer
    )
    monkeypatch.setattr(
        interface_module, "set_oriented_bounding_box_edges", set_edges
    )

    shape_data = bare_interface._create_napari_shapes("registered")
    bare_interface._update_view_add_shapes(viewer, *shape_data, "boxes")

    kwargs = bounding_box_layer.call_args.kwargs
    assert kwargs["features"]["refs"] == ["0", "0 0"]
    assert kwargs["features"]["labels"] == ["image-0", ""]
    np.testing.assert_allclose(
        kwargs["face_color"][1], [0.1, 0.2, 0.3]
    )
    viewer.add_layer.assert_called_once_with(bbox_layer)
    set_edges.assert_called_once_with(
        bbox_layer, [image_shape, overlap_shape]
    )


def test_update_napari_shapes_uses_shapes_layer_for_2d(
    bare_interface, monkeypatch
):
    bare_interface.preview_sims = [
        SimpleNamespace(dims=("y", "x"), sizes={"y": 10, "x": 10})
    ]
    viewer = MagicMock()
    shape = np.zeros((4, 2))
    monkeypatch.setattr(
        interface_module.si_utils, "get_origin_from_sim", lambda _: {}
    )
    monkeypatch.setattr(
        interface_module, "create_sim_shapes", lambda *_, **__: [shape]
    )

    shapes = [shape]
    bare_interface._update_view_add_shapes(
        viewer, shapes, ["0"], ["image-0"], [(1, 1, 1)], "boxes"
    )

    args, kwargs = viewer.add_shapes.call_args
    np.testing.assert_allclose(args[0], [shape])
    assert kwargs["features"]["refs"] == ["0"]


def test_update_napari_features_dispatches_all_layer_types(
    bare_interface, monkeypatch
):
    viewer = MagicMock()
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
        interface_module, "TqdmCallback", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        interface_module,
        "TemporarilyDisabledWidgets",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(
        interface_module, "VisibleActivityDock", lambda _: nullcontext()
    )


def test_run_pair_registration_serializes_quality_and_time_bbox(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    import xarray as xr

    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.metrics_methods = ["ncc"]
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.reg.sims = ["sim"]
    bare_interface.reg.register_sims = ["register-sim"]
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
    bbox = xr.DataArray(
        [[[1, 2], [3, 4]]],
        dims=("t", "corner", "axis"),
        coords={"t": [0]},
    )
    monkeypatch.setattr(
        interface_module.nx,
        "get_edge_attributes",
        lambda *_: {(0, 1): bbox},
    )

    actual = bare_interface.run_pair_registration()

    assert actual is results
    bare_interface.reg.register_pairs.assert_called_once_with(
        ["sim"],
        ["register-sim"],
        params={"method": "phase", "metrics": ["ncc"]},
    )
    bare_interface.reg.save_pair_mappings.assert_called_once_with(
        {(0, 1): "mapping"},
        {(0, 1): 0.9},
        {(0, 1): [[1, 2], [3, 4]]},
    )


def test_run_global_registration_persists_all_results(
    bare_interface, mocked_activity_contexts
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.reg.sims = ["sim"]
    bare_interface.reg.msims = ["msim"]
    bare_interface.reg.register_indices = [0]
    results = {
        "mappings": {"image-0": "mapping"},
        "metrics": {"summary": {}},
    }
    bare_interface.reg.register_global.return_value = results

    actual = bare_interface.run_global_registration()

    assert actual is results
    bare_interface.reg.register_global.assert_called_once_with(
        ["sim"],
        ["msim"],
        register_indices=[0],
        params={"method": "phase"},
    )
    bare_interface.reg.save_mappings.assert_called_once_with(
        results["mappings"]
    )
    bare_interface.reg.save_mappings_csv.assert_called_once_with(
        results["mappings"]
    )
    bare_interface.reg.save_metrics.assert_called_once_with(
        results["metrics"]
    )


@pytest.mark.parametrize(
    ("global_registered", "pairs_registered", "reply", "runs"),
    [
        (True, False, None, False),
        (False, False, "No", False),
        (False, False, "Yes", True),
        (False, True, "Yes", True),
    ],
)
def test_pair_registration_confirmation_paths(
    bare_interface,
    monkeypatch,
    global_registered,
    pairs_registered,
    reply,
    runs,
):
    bare_interface.reg.is_global_registered.return_value = global_registered
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.source_transform_key = "source_metadata"
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.update_registered = MagicMock()
    warning = MagicMock()
    monkeypatch.setattr(interface_module, "show_warning", warning)
    if reply is not None:
        monkeypatch.setattr(
            interface_module.QMessageBox,
            "question",
            lambda *_: getattr(interface_module.QMessageBox, reply),
        )

    bare_interface.pair_registration()

    if global_registered:
        warning.assert_called_once()
    assert bare_interface.run_pair_registration.called is runs
    assert bare_interface.update_registered.called is runs


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
    pairs_registered,
    reply,
    run_pair,
    run_global,
):
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.sims = ["sim"]
    bare_interface.reg.reg_transform_key = "registered"
    bare_interface.preview_sims = ["preview"]
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.run_global_registration = MagicMock()
    bare_interface.enable_tabs = MagicMock()
    bare_interface.update_registered = MagicMock()
    copy = MagicMock()
    monkeypatch.setattr(interface_module, "copy_transforms", copy)
    monkeypatch.setattr(
        interface_module.QMessageBox,
        "question",
        lambda *_: getattr(interface_module.QMessageBox, reply),
    )

    bare_interface.registration_process()

    assert bare_interface.run_pair_registration.called is run_pair
    assert bare_interface.run_global_registration.called is run_global
    if run_global:
        copy.assert_called_once_with(["sim"], ["preview"], "registered")
        bare_interface.enable_tabs.assert_called_once_with(True, 4)
        bare_interface.update_registered.assert_called_once_with(
            view_transform_key="registered"
        )


@pytest.mark.parametrize(
    ("tile_size", "expected"),
    [("1024", 1024), ("512, 1024", [512, 1024])],
)
def test_fusion_process_parses_tile_size_and_updates_state(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    tile_size,
    expected,
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
    bare_interface.reg.sims = ["sim"]
    bare_interface.reg.fuse.return_value = ("fused", None)
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_image = MagicMock()
    monkeypatch.setattr(
        interface_module.QMessageBox,
        "question",
        lambda *_: interface_module.QMessageBox.Yes,
    )

    bare_interface.fusion_process()

    assert bare_interface.reg.fuse.call_args.kwargs["tile_size"] == expected
    assert (
        bare_interface.reg.fuse.call_args.kwargs["output_filename"]
        == "registered"
    )
    bare_interface._napari_view_add_image.assert_called_once_with(
        bare_interface.viewer, "fused", "Fused"
    )
    assert bare_interface.reg.state is CanonicalRegState.FUSED
    assert bare_interface.view_mode is CanonicalViewMode.FUSED
