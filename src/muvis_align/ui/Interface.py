import logging
from multiview_stitcher import spatial_image_utils as si_utils
import os.path

from muvis_align.file.project_yaml import read_params, get_template_params, write_params
from muvis_align.MVSRegistrationNapari import MVSRegistrationNapari
from muvis_align.image.util import get_sim_physical_size
from muvis_align.resources import get_project_template
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import dir_regex, find_all_numbers, print_dict


class Interface:
    def __init__(self, viewer, verbose=False):
        self.viewer = viewer
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.params = {}
        self.param_widgets = {}

        self.params_general = {}
        self.params_operation = {}
        self.reg = MVSRegistrationNapari(self.params_general, self.viewer)

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

    def input_images(self, path):
        path = str(path)
        self.params_operation['operation'] = 'register'
        self.params_operation['output'] = '/output'
        self.params_operation['input'] = path

        filenames = dir_regex(path)
        filenames = sorted(filenames, key=lambda file: list(find_all_numbers(file)))  # sort first key first
        fileset_label = os.path.basename(path)
        if len(filenames) == 0:
            logging.warning(f'No files found for path: {path}')
            return
        elif self.verbose:
            logging.info(f'# total files: {len(filenames)}')
        self.reg.init_operation(fileset_label, filenames, self.params_operation)
        sims = self.reg.init_sims()
        self.populate_metadata_table(sims)

    def populate_metadata_table(self, sims):
        table_widget = self.param_widgets.get('input_data.metadata_table')
        #data = {
        #    'label': self.reg.file_labels,
        #    'position': [print_dict(si_utils.get_origin_from_sim(sim)) for sim in sims],
        #    'size': [print_dict(get_sim_physical_size(sim)) for sim in sims]
        #}
        data = {
            "data": [[1, 2, 3], [4, 5, 6]],
            "index": ("r1", "r2"),
            "columns": ("c1", "c2", "c3"),
        }
        table_widget.set_value(data)
        table_widget.read_only = True
