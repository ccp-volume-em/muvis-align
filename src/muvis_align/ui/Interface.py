from multiview_stitcher import spatial_image_utils as si_utils
import os.path

from muvis_align.file.project_yaml import read_params, get_template_params, write_params
from muvis_align.MVSRegistrationNapari import MVSRegistrationNapari
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final
from muvis_align.file.resources import get_project_template
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import print_dict_simple, set_dict_value, is_valid_value


class Interface:
    def __init__(self, viewer, verbose=False):
        self.viewer = viewer
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.param_widgets = {}
        self.params = {}
        self.source_metadata = {}
        self.transform_key = 'source_metadata'

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

    def update_view(self):
        if self.params['input_output']['preview_images']:
            self.reg.update_napari_data.emit(f'{self.reg.fileset_label} data', self.transform_key)
        if self.params['input_output']['preview_shapes']:
            self.reg.update_napari_shapes.emit(f'{self.reg.fileset_label} shapes', self.transform_key)
