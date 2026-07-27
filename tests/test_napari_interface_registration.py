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
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import yaml
import numpy as np

from src.muvis_align._widget import MainWidget
from src.muvis_align.ui.Interface import Interface, ViewMode
from src.muvis_align.MVSRegistration import RegState


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
                                    with patch.object(interface, '_add_napari_image'):
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
        try:
            from PyQt5.QtWidgets import QMessageBox
        except ModuleNotFoundError:
            pytest.skip("PyQt5 not available in test environment")
        
        viewer = make_napari_viewer()
        mock_question.return_value = QMessageBox.Yes  # Simulate "Yes" click

        with patch('src.muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())

        interface.view_mode = ViewMode.PAIRS
        interface.pair_indices = ('key1', 'key2')
        
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
                
                mock_calc.return_value = mock_transform_with_t
                
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
                        with patch.object(interface, 'update_overview') as mock_overview:
                            with patch.object(interface, 'update_view') as mock_view:
                                # This should not raise KeyError about dimension mismatch
                                interface.update_registered()

                                assert mock_pop_coord.called
                                assert mock_pop_meta.called
                                assert mock_pop_metrics.called
                                assert mock_overview.called
                                assert mock_view.called


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
