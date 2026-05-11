from enum import Enum, auto
from multiview_stitcher import spatial_image_utils as si_utils, param_utils
from napari.utils.notifications import show_warning
import os.path

from muvis_align.file.project_yaml import read_params, get_template_params, write_params
from muvis_align.MVSRegistrationNapari import MVSRegistrationNapari
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final, get_overlap_images, \
    affine_from_intrinsic_affine
from muvis_align.file.resources import get_project_template
from muvis_align.metrics import calc_sims_metrics
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import print_dict_simple, set_dict_value, is_valid_value


class ViewMode(Enum):
    OVERVIEW = auto()
    PAIRS = auto()
    FEATURES = auto()


class Interface:
    def __init__(self, viewer, enable_tabs, verbose=False):
        self.viewer = viewer
        self.enable_tabs = enable_tabs
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.param_widgets = {}
        self.params = {}
        self.source_metadata = {}
        self.transform_key = 'source_metadata'
        self.view_mode = None

        self.reg = MVSRegistrationNapari(self.viewer)

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def project_path(self, path):
        self.params_path = path
        if os.path.exists(path):
            self.params = read_params(path)
            self.update_widgets()
        else:
            self.params = get_template_params(self.template)
            self.write_params()

    def update_widgets(self):
        for param_name, param_widget in self.param_widgets.items():
            keys = param_name.split('.')
            value = self.params.get(keys[0], {}).get(keys[1])
            if value is not None:
                param_widget.set_value(value)

    def write_params(self):
        write_params(self.params_path, self.params)

    def change_param(self, param_name, value):
        keys = param_name.split('.')
        if keys[0] not in self.params:
            self.params[keys[0]] = {}
        self.params[keys[0]][keys[1]] = value
        self.write_params()

    def source_position_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'z'], value)

    def source_position_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'y'], value)

    def source_position_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'x'], value)

    def source_size_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['size', 'z'], value)

    def source_size_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['size', 'y'], value)

    def source_size_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['size', 'x'], value)

    def source_rotation(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['rotation'], value)

    def input_output_process(self):
        params = self.params['input_output']
        ok = self.reg.init(input_path=str(params['input_path']),
                           output_path=str(params['output_path']),
                           overwrite=params['overwrite'])
        if ok:
            self.update_metadata_source()
            self.populate_image_selection()
            self.enable_tabs()
        else:
            show_warning('No input images found')

    def update_metadata_source(self):
        if not self.reg.is_registered:
            self.reg.init_sims(source_metadata=self.source_metadata)
        sims = self.reg.sims

        coord_systems = list({a for group in [si_utils.get_tranform_keys_from_sim(sim) for sim in sims] for a in group})
        self.populate_coordinate_systems(coord_systems)
        if self.reg.initialised:
            self.populate_metadata_table(sims)
            self.update_view()

    def populate_coordinate_systems(self, coord_systems):
        param_widget = self.param_widgets.get('input_output.coordinate_system')
        choices = {coord_system: coord_system.replace('_', ' ').capitalize() for coord_system in coord_systems}
        param_widget.widget.choices = param_widget.create_choices(choices)

    def coordinate_system(self, transform_key):
        self.transform_key = transform_key
        if self.reg.initialised:
            self.populate_metadata_table(self.reg.sims, [transform_key])

    def populate_metadata_table(self, sims, transform_keys=None):
        # https://pyapp-kit.github.io/magicgui/api/widgets/Table/
        # https://pyapp-kit.github.io/magicgui/generated_examples/demo_widgets/table/
        table_widget = self.param_widgets.get('input_output.metadata_table')
        data = {
           'label': ["'" + label + "'" for label in self.reg.file_labels],
           'position': [print_dict_simple(get_sim_position_final(sim, transform_keys=transform_keys)) for sim in sims],
           'size': [print_dict_simple(get_sim_physical_size(sim)) for sim in sims]
        }
        table_widget.set_value(data)
        table_widget.read_only = True   # https://github.com/pyapp-kit/magicgui/issues/348

    def populate_image_selection(self):
        labels = self.reg.file_labels
        widget1 = self.param_widgets.get('features.reg_preview_image1').widget
        widget1.choices = labels
        widget1.value = labels[0]

        widget2 = self.param_widgets.get('features.reg_preview_image2').widget
        widget2.choices = labels
        index = 1 if len(labels) > 1 else 0
        widget2.value = labels[index]

    def update_view(self):
        if self.view_mode != ViewMode.OVERVIEW:
            self.reg.clear_napari_view.emit()
            self.view_mode = ViewMode.OVERVIEW
        if self.params['input_output']['preview_images']:
            self.reg.update_napari_data.emit(f'{self.reg.fileset_label} data', self.transform_key)
        if self.params['input_output']['preview_shapes']:
            self.reg.update_napari_shapes.emit(f'{self.reg.fileset_label} shapes', self.transform_key)

    def preprocess(self, sims):
        params_features = self.params['features']
        quantiles = params_features.get('flatfield_quantiles')
        quantiles_array = None
        if quantiles:
            quantiles_array = [float(quantile.strip()) for quantile in quantiles.split(',')]
        return self.reg.preprocess(sims,
                                   quantiles_array,
                                   params_features.get('global_normalisation'),
                                   params_features.get('global_gaussian_sigma'),
                                   params_features.get('filter_foreground')
        )

    def preview_registration(self):
        preview_key = 'registration'
        label1 = self.param_widgets.get('features.reg_preview_image1').get_value()
        label2 = self.param_widgets.get('features.reg_preview_image2').get_value()
        index1 = self.reg.file_labels.index(label1)
        index2 = self.reg.file_labels.index(label2)
        reg_sims, _ = self.preprocess([self.reg.sims[index1], self.reg.sims[index2]])
        overlap1, overlap2, sims_pixel_space = get_overlap_images(reg_sims[0], reg_sims[1], self.reg.source_transform_key)
        overlap1, overlap2 = overlap1.squeeze().compute(), overlap2.squeeze().compute()
        reg_method, pairwise_reg_func, pairwise_reg_func_kwargs = (
            self.reg.create_registration_method(self.reg.sims[0], params=self.params['features']))
        results = pairwise_reg_func(overlap1, overlap2)
        affine_phys = affine_from_intrinsic_affine(results['affine_matrix'], sims_pixel_space, self.reg.source_transform_key)
        si_utils.set_sim_affine(reg_sims[0], param_utils.identity_transform(2), transform_key=preview_key)
        si_utils.set_sim_affine(reg_sims[1], affine_phys, transform_key=preview_key)
        metrics = calc_sims_metrics(reg_sims, self.reg.source_transform_key, preview_key)
        summary = metrics['summary']
        summary[preview_key]['quality'] = float(results['quality'])

        self.populate_metrics_table(summary)

        fixed_points = results.get('fixed_points', [])
        moving_points = results.get('moving_points', [])
        matches = results.get('matches', [])
        inliers = results.get('inliers', [])
        self.reg._update_napari_features(overlap1, fixed_points,
                                         overlap2, moving_points,
                                         matches, inliers)
        self.view_mode = ViewMode.FEATURES

    def populate_metrics_table(self, metrics):
        table_widget = self.param_widgets.get('features.metrics_table')
        table_widget.set_value(metrics)
        table_widget.read_only = True

    def features_process(self):
        reg_sims, _ = self.preprocess(self.reg.sims)
        if not self.reg.is_registered:
            self.reg.register_pairs(self.reg.sims, reg_sims, self.params['features'])
