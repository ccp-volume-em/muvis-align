from enum import Enum, auto
from magicclass.ext.napari import ViewerWidget
from multiview_stitcher import spatial_image_utils as si_utils
from napari.utils.notifications import show_warning
import numpy as np
import os.path
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QMessageBox

from muvis_align.file.project_yaml import read_params, get_template_params, write_params, update_params
from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final, \
    affine_from_intrinsic_affine, get_sim_shape_2d, get_overlap_shapes, get_overlap_images, \
    draw_keypoints_matches_napari, get_transforms
from muvis_align.file.resources import get_project_template
from muvis_align.metrics import calc_sims_metrics
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import print_dict_simple, set_dict_value, is_valid_value, metric_to_rgb


class ViewMode(Enum):
    OVERVIEW = auto()
    PAIRS = auto()
    FEATURES = auto()


class Interface:
    def __init__(self, viewer, overview, enable_tabs, verbose=False):
        self.viewer = viewer
        self.overview = overview
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
        self.selected_shape_index = None

        self.reg = MVSRegistration()

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def tab_changed(self, tab_label):
        if tab_label != 'registration' and self.view_mode == ViewMode.FEATURES:
            self._clear_napari_view(self.viewer)
            self.view_mode = None

    def project_path(self, path):
        self.params_path = path
        self.params = get_template_params(self.template)
        if os.path.exists(path):
            self.params = update_params(self.params, read_params(path))
            self.update_widgets()
        else:
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

    def source_scale_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'z'], value)

    def source_scale_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'y'], value)

    def source_scale_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'x'], value)

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
            self.enable_tabs(True, 2)
        else:
            show_warning('No input images found')

    def update_metadata_source(self):
        if not self.reg.is_pairs_registered():
            target_scale = self.params['input_output']['target_scale']
            self.reg.init_sims(source_metadata=self.source_metadata, target_scale=target_scale)
        sims = self.reg.sims

        coord_systems = get_transforms(sims)
        self.populate_coordinate_systems(coord_systems)
        if self.reg.is_initialised():
            self.populate_metadata_table(sims)
            self.update_overview()
            self.update_view()

    def pre_processing_process(self):
        params_features = self.params['pre_processing']
        _, _, modified = self.reg.preprocess(self.reg.sims, **params_features)
        if modified:
            self.update_view(show_preprocessed=True)
        self.enable_tabs(True, 3)

    def populate_coordinate_systems(self, coord_systems):
        param_widget = self.param_widgets.get('input_output.coordinate_system')
        choices = {coord_system: coord_system.replace('_', ' ').capitalize() for coord_system in coord_systems}
        param_widget.widget.choices = param_widget.create_choices(choices)

    def coordinate_system(self, transform_key):
        self.transform_key = transform_key
        if self.reg.is_initialised():
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
        widget1 = self.param_widgets.get('registration.reg_preview_image1')
        widget1.set_value(labels[0], choices=labels)

        widget2 = self.param_widgets.get('registration.reg_preview_image2')
        index = 1 if len(labels) > 1 else 0
        widget2.set_value(labels[index], choices=labels)

    def update_overview(self, overlaps=True):
        transform_key = self.reg.reg_transform_key if self.reg.reg_transform_key in get_transforms(self.reg.sims)\
            else self.reg.source_transform_key
        self._update_napari_shapes(self.overview, f'{self.reg.fileset_label} shapes', transform_key,
                                   overlaps=overlaps)

    def update_view(self, overlaps=False, show_preprocessed=False):
        transform_key = self.reg.reg_transform_key if self.reg.reg_transform_key in get_transforms(self.reg.sims)\
            else self.reg.source_transform_key
        if self.view_mode != ViewMode.OVERVIEW or show_preprocessed:
            self._clear_napari_view(self.viewer)
            self.view_mode = ViewMode.OVERVIEW
        if self.params['input_output']['preview_images'] or show_preprocessed:
            self._update_napari_data(self.viewer, f'{self.reg.fileset_label} data', transform_key,
                                     show_preprocessed)
        if self.params['input_output']['preview_shapes'] and not show_preprocessed:
            self._update_napari_shapes(self.viewer, f'{self.reg.fileset_label} shapes', transform_key,
                                       overlaps=overlaps)

    def _clear_napari_view(self, viewer):
        viewer.layers.clear()

    def _update_napari_data(self, viewer, layer_name, transform_key, show_preprocessed=False):
        if show_preprocessed:
            sims = self.register_sims
        else:
            sims = self.sims
        fused, _ = self.fuse(sims, transform_key=transform_key, fusion_method='additive')
        fused_scale = si_utils.get_spacing_from_sim(fused, asarray=True)
        fused_position = si_utils.get_origin_from_sim(fused, asarray=True)
        if fused is not None:
            if layer_name in viewer.layers:
                viewer.layers[layer_name].data = fused
            else:
                image_layer = viewer.add_image(fused, name=layer_name, scale=fused_scale, translate=fused_position)
                current_index = viewer.layers.index(image_layer)
                # ensure image layer goes on 'bottom'
                if current_index > 0:
                    viewer.layers.move(current_index, 0)

    def _update_napari_shapes(self, viewer, layer_name, transform_key, overlaps=False):
        if isinstance(viewer, ViewerWidget):
            viewer = viewer._qtwidget._viewer_model
        sims = self.reg.sims
        shapes = [get_sim_shape_2d(sim, transform_key=transform_key) for sim in sims]
        refs = [str(index) for index in range(len(sims))]
        labels = list(self.reg.file_labels)
        face_colors = [(1, 1, 1) for _ in range(len(sims))]
        if overlaps:
            shapes2, pairs = get_overlap_shapes(sims, transform_key=transform_key)
            shapes += shapes2
            refs += [f'{index1} {index2}' for index1, index2 in pairs]
            labels += ['' for _ in pairs]
            face_colors += [np.array(metric_to_rgb(self.reg.get_metrics('quality', pair))) for pair in pairs]
        if len(shapes) > 0:
            text = {'string': '{labels}'}
            features = {'refs': refs, 'labels': labels}
            if layer_name in viewer.layers:
                layer = viewer.layers[layer_name]
                layer.data = shapes
                layer.face_color = face_colors
                layer.text = text
                layer.features = features
            else:
                viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5,
                                  face_color=face_colors)

                # layer = viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5,
                #                           face_color=face_colors)
                # @viewer.mouse_move_callbacks.append
                # def on_mouse_move(viewer, event):
                #     self.selected_shape_index = layer._value[0]
                #
                # @viewer.mouse_drag_callbacks.append
                # def on_mouse_drag(viewer, event):
                #     if event.type == "mouse_press" and event.button == 1:
                #         if viewer.layers.selection.active == layer and self.selected_shape_index is not None:
                #             self.on_selection_change(refs[self.selected_shape_index])
                #     yield

    def _update_napari_features(self, viewer, fixed_data2, fixed_points, moving_data2, moving_points, matches, inliers):

        layers = draw_keypoints_matches_napari(fixed_data2, fixed_points,
                                               moving_data2, moving_points,
                                               matches, inliers, points_color='blue')

        viewer.layers.clear()
        for data, kwargs, layer_type in layers:
            if layer_type == "image":
                viewer.add_image(data, **kwargs)
            elif layer_type == "points":
                viewer.add_points(data, **kwargs)
            elif layer_type == "shapes":
                viewer.add_shapes(data, **kwargs)

    def preview_registration(self):
        self._clear_napari_view(self.viewer)
        label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
        label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
        index1 = self.reg.file_labels.index(label1)
        index2 = self.reg.file_labels.index(label2)
        reg_sims = self.reg.register_sims[index1], self.reg.register_sims[index2]
        overlap1, overlap2, sims_pixel_space = get_overlap_images(reg_sims[0], reg_sims[1], self.reg.source_transform_key)
        overlap1, overlap2 = overlap1.squeeze().compute(), overlap2.squeeze().compute()
        reg_method, pairwise_reg_func, pairwise_reg_func_kwargs = (
            self.reg.create_registration_method(self.reg.sims[0], params=self.params['registration']))
        results = pairwise_reg_func(overlap1, overlap2)

        affine_phys = affine_from_intrinsic_affine(results['affine_matrix'], sims_pixel_space, self.reg.source_transform_key)
        transforms = {
            (0, 1): affine_phys
        }
        qualities = {
            (0, 1): np.array(results['quality'])
        }
        metrics = calc_sims_metrics(reg_sims, transforms, qualities, metric_methods=['ncc', 'ssim', 'onmi'])
        self.populate_metrics_table(metrics)

        fixed_points = results.get('fixed_points', [])
        moving_points = results.get('moving_points', [])
        matches = results.get('matches', [])
        inliers = results.get('inliers', [])
        self._update_napari_features(self.viewer, overlap1, fixed_points, overlap2, moving_points, matches, inliers)
        self.view_mode = ViewMode.FEATURES

    def populate_metrics_table(self, metrics_dict):
        transform_keys = []
        metric_keys = []
        item_keys = []
        metrics = metrics_dict.get('summary')
        if metrics:
            item_keys.append('summary')
            for transform_key, transform_value in metrics.items():
                if transform_key not in transform_keys:
                    transform_keys.append(transform_key)
                for metric_key in transform_value.keys():
                    if metric_key not in metric_keys:
                        metric_keys.append(metric_key)
        metrics = metrics_dict.get('pairs')
        if metrics:
            for pair_key, pair_value in metrics.items():
                if pair_key not in item_keys:
                    item_keys.append(pair_key)
                for transform_key, transform_value in pair_value.items():
                    if transform_key not in transform_keys:
                        transform_keys.append(transform_key)
                    for metric_key in transform_value.keys():
                        if metric_key not in metric_keys:
                            metric_keys.append(metric_key)

        metrics_table = np.zeros((len(item_keys), len(transform_keys)), dtype=np.float32)
        item_offset = 0

        metrics = metrics_dict.get('summary')
        if metrics:
            item_offset = 1
            for transform_index, (transform_key, transform_value) in enumerate(metrics.items()):
                for metric_value in transform_value.values():
                    metrics_table[0, transform_index] = metric_value
        metrics = metrics_dict.get('pairs')
        if metrics:
            for pair_index, (pair_key, pair_value) in enumerate(metrics.items()):
                for transform_index, (transform_key, transform_value) in enumerate(pair_value.items()):
                    for metric_value in transform_value.values():
                        metrics_table[pair_index + item_offset, transform_index] = metric_value

        table_widget = self.param_widgets.get('registration.metrics_table')
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((metrics_table, item_keys, transform_keys))
        for rowi in range(len(item_keys)):
            for coli in range(len(transform_keys)):
                table_widget.widget.native.item(rowi, coli).setBackground(
                    QColor(*metric_to_rgb(metrics_table[rowi, coli], range=255, max_light=0.5)))
        table_widget.read_only = True

    def pair_registration(self):
        if self.reg.is_global_registered():
            show_warning('Global registration was already performed')
        else:
            message = 'Pair registration was already performed' if self.reg.is_pairs_registered() else ''
            message += 'Run pair registration?'
            reply = QMessageBox.question(None, 'muvis-align',message,
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._clear_napari_view(self.viewer)
                self.reg.register_pairs(self.reg.sims, self.reg.register_sims, params=self.params['registration'])
                self.update_registered()

    def modify_pair_registration(self):
        pass

    def registration_process(self):
        if not self.reg.is_pairs_registered():
            show_warning('Perform pair registration first')
        elif not self.reg.is_global_registered():
            reply = QMessageBox.question(None, 'muvis-align','Run global registration?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._clear_napari_view(self.viewer)
                self.reg.register_global(self.reg.sims, self.reg.msims, register_indices=self.reg.register_indices,
                                         params=self.params['registration'])
                self.update_registered()

    def update_registered(self):
        sims = self.reg.sims
        coord_systems = list({a for group in [si_utils.get_tranform_keys_from_sim(sim) for sim in sims] for a in group})
        self.populate_coordinate_systems(coord_systems)
        self.populate_metadata_table(sims)
        self.update_overview()
        self.update_view(overlaps=True)
        self.populate_metrics_table(self.reg.metrics)
