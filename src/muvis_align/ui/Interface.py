from enum import Enum, auto
from magicclass.ext.napari import ViewerWidget
from multiview_stitcher import spatial_image_utils as si_utils, param_utils
from napari.utils import progress
from napari.utils.notifications import show_warning
from napari_bbox.boundingbox import BoundingBoxLayer
import networkx as nx
import numpy as np
import os.path
from qtpy.QtCore import QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QMessageBox
from tqdm.dask import TqdmCallback

from muvis_align.constants import zarr_extension, default_transform_key, default_quality_key
from muvis_align.file.project_yaml import read_params, get_template_params, write_params, update_params
from muvis_align.MVSRegistration import MVSRegistration, RegState
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final, \
    affine_from_intrinsic_affine, get_sim_shape, get_overlap_shapes, get_overlap_images, \
    draw_keypoints_matches_napari, get_transforms, copy_transforms
from muvis_align.file.resources import get_project_template
from muvis_align.metrics import calc_sims_metrics
from muvis_align.ui._utils import TemporarilyDisabledWidgets, VisibleActivityDock
from muvis_align.ui.bilayers_util import get_section_dict, to_magicgui_choices
from muvis_align.util import print_dict_simple, set_dict_value, is_valid_value, metric_to_rgb, \
    calculate_rigid_difference


class ViewMode(Enum):
    OVERVIEW = auto()
    PAIRS = auto()
    FEATURES = auto()
    FUSED = auto()


class Interface:
    def __init__(self, viewer, overview, enable_tabs, select_tab, verbose=False):
        self.viewer = viewer
        self.overview = overview
        self.enable_tabs = enable_tabs
        self.select_tab = select_tab
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.param_widgets = {}
        self.params = {}
        self.pre_processing_performed = False
        self.metrics_methods = ['ncc', 'ssim', 'onmi']
        self.transform_key = 'source_metadata'

        self.pair_metrics_timer = QTimer()
        self.pair_metrics_timer.setSingleShot(True)
        self.pair_metrics_timer.setInterval(1000)
        self.pair_metrics_timer.timeout.connect(self.update_pair_metrics)

        self.reg = MVSRegistration()
        self.reset()

    def reset(self):
        self.source_metadata = {}
        self.view_mode = None
        self.selected_shape_index = None
        self.reg.reset()
        self._clear_napari_view(self.overview)
        self._clear_napari_view(self.viewer)
        self.enable_tabs(False, 2)
        self.select_tab(1)

    def get_all_widgets(self):
        all_widgets = {name: param_widget.widget for name, param_widget in self.param_widgets.items()}
        return all_widgets

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def tab_changed(self, tab_label):
        if tab_label != 'registration' and self.view_mode == ViewMode.FEATURES:
            self._clear_napari_view(self.viewer)
            self.view_mode = None
        self.pair_metrics_timer.stop()

    def project_path(self, path):
        self.reset()
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
        output = str(params['output_path'])
        if not self.reg.is_initialised():
            if not output.endswith(os.sep):
                output += os.sep
            ok = self.reg.init(input_path=str(params['input_path']),
                               output_path=output,
                               overwrite=params['overwrite'])
            if ok:
                self.update_metadata_source()
                self.populate_image_selection()
                self.init_progress()
            else:
                show_warning('No input images found')
        elif self.reg.is_pairs_registered():
            self.update_registered()
        else:
            self.update_metadata_source()

    def init_progress(self):
        output_filename = self.params['registration']['operation'].split()[0] + 'ed'
        self.reg.init_progress(output_filename, zarr_extension)
        if self.reg.is_fused():
            self.enable_tabs(True, 4)
            self.select_tab(4)
            self.preview_fusion()
        elif self.reg.is_global_registered():
            self.enable_tabs(True, 4)
            self.select_tab(4)
            self.update_registered()
        elif self.reg.is_pairs_registered():
            self.enable_tabs(True, 3)
            self.select_tab(3)
            self.update_registered()
        else:
            self.enable_tabs(True, 2)
            self.select_tab(2)

    def update_metadata_source(self):
        if not self.reg.is_pairs_registered():
            preview_scale = self.params['input_output']['preview_scale']
            self.preview_sims = self.reg.init_sims(source_metadata=self.source_metadata, target_scale=preview_scale,
                                                   store=False)
            self.reg.init_sims(source_metadata=self.source_metadata)
        sims = self.reg.sims

        coord_systems = get_transforms(sims)
        self.populate_coordinate_systems(coord_systems)
        if self.reg.is_initialised():
            self.populate_metadata_table(sims)
            self.check_3d_view()
            self.update_overview()
            self.update_view()

    def pre_processing_process(self):
        params_features = self.params['pre_processing']
        if self.reg.check_preprocess(**params_features) or self.pre_processing_performed:
            with TqdmCallback(tqdm_class=progress, desc='Pre-processing', bar_format=" "), \
                 TemporarilyDisabledWidgets(self.get_all_widgets()), \
                 VisibleActivityDock(self.viewer):
                _, _, modified = self.reg.preprocess(self.reg.sims, **params_features)
            self.pre_processing_performed = modified
            self.update_view(show_preprocessed=True)
        self.enable_tabs(True, 3)
        self.select_tab(3)

    def populate_coordinate_systems(self, coord_systems):
        param_widget = self.param_widgets.get('input_output.coordinate_system')
        choices = {coord_system: coord_system.replace('_', ' ').capitalize() for coord_system in coord_systems}
        param_widget.widget.choices = to_magicgui_choices(choices)

    def coordinate_system(self, transform_key):
        self.transform_key = transform_key
        if self.reg.is_initialised():
            self.populate_metadata_table(self.reg.sims, [transform_key])

    def populate_metadata_table(self, sims, transform_keys=None):
        # https://pyapp-kit.github.io/magicgui/api/widgets/Table/
        # https://pyapp-kit.github.io/magicgui/generated_examples/demo_widgets/table/
        table_widget = self.param_widgets.get('input_output.metadata_table')
        properties = ['position', 'size']
        if transform_keys is None:
            positions = self.reg.positions
            scales = self.reg.scales
        else:
            positions = [get_sim_position_final(sim, transform_keys=transform_keys) for sim in sims]
            scales = [get_sim_physical_size(sim) for sim in sims]
        data = [[print_dict_simple(position),
                 print_dict_simple(scale)]
                for position, scale in zip(positions, scales)]
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((data, self.reg.file_labels, properties))
        table_widget.set_table_column_resize_mode()
        table_widget.read_only = True   # https://github.com/pyapp-kit/magicgui/issues/348

    def populate_image_selection(self):
        labels = self.reg.file_labels
        widget1 = self.param_widgets.get('registration.reg_preview_image1')
        widget1.set_value(labels[0], choices=labels)

        widget2 = self.param_widgets.get('registration.reg_preview_image2')
        index = 1 if len(labels) > 1 else 0
        widget2.set_value(labels[index], choices=labels)

    def get_best_transform_key(self):
        transforms = get_transforms(self.reg.sims)
        if self.reg.reg_transform_key in transforms:
            transform_key = self.reg.reg_transform_key
        elif default_transform_key in transforms:
            transform_key = default_transform_key
        elif self.reg.source_transform_key in transforms:
            transform_key = self.reg.source_transform_key
        else:
            transform_key = None
        return transform_key

    def check_3d_view(self):
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        ndisplay = 3 if is_3d else 2
        self.viewer.dims.ndisplay = ndisplay
        #self.overview._qtwidget._viewer_model.dims.ndisplay = ndisplay

    def update_overview(self, overlaps=True):
        transform_key = self.get_best_transform_key()
        self._clear_napari_view(self.overview)
        self._update_napari_shapes(self.overview, f'{self.reg.fileset_label} shapes', transform_key,
                                   overlaps=overlaps)

    def update_view(self, overlaps=False, show_preprocessed=False):
        transform_key = self.get_best_transform_key()
        self._clear_napari_view(self.viewer)
        if self.params['input_output']['preview_images']:
            self._update_napari_data(self.viewer, f'{self.reg.fileset_label} data', transform_key,
                                     show_preprocessed=show_preprocessed)
        if self.params['input_output']['preview_shapes']:
            self._update_napari_shapes(self.viewer, f'{self.reg.fileset_label} shapes', transform_key,
                                       overlaps=overlaps)
        self.view_mode = ViewMode.OVERVIEW

    def _clear_napari_view(self, viewer):
        viewer.layers.clear()

    def _update_napari_data(self, viewer, layer_name, transform_key, fusion_method='additive', show_preprocessed=False):
        if show_preprocessed:
            sims = self.reg.register_sims
        else:
            sims = self.preview_sims
            copy_transforms(self.reg.sims, sims, transform_key)
        fused, _ = self.reg.fuse(sims, transform_key=transform_key, fusion_method=fusion_method)
        fused_scale = si_utils.get_spacing_from_sim(fused, asarray=True)
        fused_position = si_utils.get_origin_from_sim(fused, asarray=True)
        if fused is not None:
            image_layer = viewer.add_image(fused, name=layer_name, scale=fused_scale, translate=fused_position)
            current_index = viewer.layers.index(image_layer)
            # ensure image layer goes on 'bottom'
            if current_index > 0:
                viewer.layers.move(current_index, 0)

    def _update_napari_shapes(self, viewer, layer_name, transform_key, overlaps=False):
        bb_supported = True
        if isinstance(viewer, ViewerWidget):
            viewer = viewer._qtwidget._viewer_model
            bb_supported = False
        sims = self.reg.sims
        shapes = [get_sim_shape(sim, transform_key=transform_key, force_2d=not bb_supported) for sim in sims]
        refs = [str(index) for index in range(len(sims))]
        labels = list(self.reg.file_labels)
        face_colors = [(1, 1, 1) for _ in range(len(sims))]

        if overlaps:
            shapes2, pairs = get_overlap_shapes(sims, transform_key=transform_key, force_2d=not bb_supported)
            shapes.extend(shapes2)
            refs += [f'{index1} {index2}' for index1, index2 in pairs]
            labels += ['' for _ in pairs]
            face_colors += [np.array(metric_to_rgb(self.reg.get_metrics('quality', pair))) for pair in pairs]
        if len(shapes) > 0:
            text = {'string': '{labels}'}
            features = {'refs': refs, 'labels': labels}
            shapes = np.array(shapes)
            is_3d = (shapes.shape[-1] == 3)
            if is_3d and bb_supported:
                bbox_layer = BoundingBoxLayer(shapes, name=layer_name, text=text, features=features,
                                              face_color=face_colors, opacity=0.5, edge_width=100)
                self.viewer.add_layer(bbox_layer)
            else:
                viewer.add_shapes(shapes, name=layer_name, text=text, features=features,
                                  face_color=face_colors, opacity=0.5, edge_width=0.1)

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

    def _add_napari_image(self, viewer, data, label, transform=None, color=None, affine_event=False):
        scale = si_utils.get_spacing_from_sim(data, asarray=True)
        position = si_utils.get_origin_from_sim(data, asarray=True)
        layer = viewer.add_image(data, name=label, scale=scale, translate=position, affine=transform,
                                 blending='additive')
        if color:
            layer.colormap = color

        if affine_event:
            layer.events.affine.connect(self.on_image_data_changed)

        return layer

    def on_image_data_changed(self, event):
        self.pair_metrics_timer.stop()
        self.pair_metrics_timer.start()

    def update_pair_metrics(self):
        # filter only selected pair
        sims = [self.reg.sims[index] for index in self.pair_indices]
        transforms = {(0, 1): self.calc_mod_pair_transform()}
        metrics = calc_sims_metrics(sims, transforms, metric_methods=self.metrics_methods)
        self.populate_metrics_table(metrics)

    def preview_registration(self):
        self._clear_napari_view(self.viewer)
        label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
        label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
        index1 = self.reg.file_labels.index(label1)
        index2 = self.reg.file_labels.index(label2)

        if len(self.reg.register_sims) == 0:
            params_features = self.params['pre_processing']
            self.reg.preprocess(self.reg.sims, **params_features)
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
        metrics = calc_sims_metrics(reg_sims, transforms, qualities, metric_methods=self.metrics_methods)
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
                for metric_key, metric_value in transform_value.items():
                    if metric_value is not None and metric_key not in metric_keys:
                        metric_keys.append(metric_key)
        metrics = metrics_dict.get('pairs')
        if metrics:
            for pair_key_indices, pair_value in metrics.items():
                pair_key = self.reg.file_labels[pair_key_indices[0]] + ' - ' + self.reg.file_labels[pair_key_indices[1]]
                if pair_key not in item_keys:
                    item_keys.append(pair_key)
                for transform_key, transform_value in pair_value.items():
                    if transform_key not in transform_keys:
                        transform_keys.append(transform_key)
                    for metric_key, metric_value in transform_value.items():
                        if metric_value is not None and metric_key not in metric_keys:
                            metric_keys.append(metric_key)

        transform_keys = [transform_key.split('_')[0] for transform_key in transform_keys]
        is_metric_cols = (len(transform_keys) <= 1 and len(metric_keys) >= 1)
        col_headers = metric_keys if is_metric_cols else transform_keys

        metrics_table = []
        for rowi in range(len(item_keys)):
            row = [None] * len(col_headers)
            metrics_table.append(row)
        item_offset = 0

        metrics = metrics_dict.get('summary')
        if metrics:
            item_offset = 1
            for transform_index, transform_value in enumerate(metrics.values()):
                for metric_index, metric_value in enumerate(transform_value.values()):
                    if metric_value is not None:
                        col_index = metric_index if is_metric_cols else transform_index
                        metrics_table[0][col_index] = metric_value
        metrics = metrics_dict.get('pairs')
        if metrics:
            for pair_index, pair_value in enumerate(metrics.values()):
                for transform_index, transform_value in enumerate(pair_value.values()):
                    for metric_index, metric_value in enumerate(transform_value.values()):
                        if metric_value is not None:
                            col_index = metric_index if is_metric_cols else transform_index
                            metrics_table[pair_index + item_offset][col_index] = metric_value

        table_widget = self.param_widgets.get('registration.metrics_table')
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((metrics_table, item_keys, col_headers))
        table_widget.set_table_column_resize_mode()
        for rowi in range(len(item_keys)):
            for coli in range(len(col_headers)):
                table_cell = table_widget.get_native_item(rowi, coli)
                if table_cell is not None:
                    table_cell.setBackground(
                        QColor(*metric_to_rgb(metrics_table[rowi][coli], max_light=0.5, output_range=255)))
        table_widget.read_only = True

    def update_registered(self):
        sims = self.reg.sims
        coord_systems = list({a for group in [si_utils.get_tranform_keys_from_sim(sim) for sim in sims] for a in group})
        self.populate_coordinate_systems(coord_systems)
        self.populate_metadata_table(sims)
        self.populate_metrics_table(self.reg.metrics)
        self.update_overview()
        self.update_view(overlaps=True)

    def pair_registration(self):
        if self.reg.is_global_registered():
            show_warning('Global registration was already performed')
        else:
            message = 'Pair registration was already performed. ' if self.reg.is_pairs_registered() else ''
            message += 'Run pair registration?'
            reply = QMessageBox.question(None, 'muvis-align', message,
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                with TqdmCallback(tqdm_class=progress, desc='Pair registration', bar_format=" "), \
                     TemporarilyDisabledWidgets(self.get_all_widgets()), \
                     VisibleActivityDock(self.viewer):
                    if len(self.reg.register_sims) == 0:
                        params_features = self.params['pre_processing']
                        self.reg.preprocess(self.reg.sims, **params_features)
                    results = self.reg.register_pairs(self.reg.sims, self.reg.register_sims,
                                                      params=self.params['registration'] | {'metrics': self.metrics_methods})
                qualities = {key: metric[default_transform_key][default_quality_key]
                             for key, metric in results['metrics']['pairs'].items()}
                bboxes = {key: np.array(value.sel(t=0)).tolist() for key, value in
                          nx.get_edge_attributes(self.reg.pairs_graph, 'bbox').items()}
                self.reg.save_pair_mappings(results['pair_mappings'], qualities, bboxes)
                self.update_registered()

    def modify_pair_registration(self):
        if self.view_mode == ViewMode.PAIRS:
            reply = QMessageBox.question(None, 'muvis-align','Store modified registration?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                # update transforms back into graph
                transform = self.calc_mod_pair_transform()
                pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
                qualities = nx.get_edge_attributes(self.reg.pairs_graph, default_quality_key)
                if 't' in pair_transforms[self.pair_indices].dims:
                    transform = transform.expand_dims({'t': [0]})
                pair_transforms[self.pair_indices] = transform
                qualities[self.pair_indices] = np.array(1)    # set quality to 1
                nx.set_edge_attributes(self.reg.pairs_graph, pair_transforms, default_transform_key)
                nx.set_edge_attributes(self.reg.pairs_graph, qualities, default_quality_key)
                bboxes = {key: np.array(value.sel(t=0)).tolist() for key, value in
                          nx.get_edge_attributes(self.reg.pairs_graph, 'bbox').items()}
                self.reg.save_pair_mappings(pair_transforms, qualities, bboxes)

            self.view_mode = ViewMode.OVERVIEW
            self.update_registered()
            self.temp_widget_state.restore()
        else:
            self.view_mode = ViewMode.PAIRS
            labels = self.reg.file_labels
            label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
            label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
            index1 = labels.index(label1)
            index2 = labels.index(label2)
            indices = index1, index2
            colors = [(0, 1, 0), (1, 0, 1)]     # green, purple
            pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
            if indices not in pair_transforms and tuple(reversed(indices)) in pair_transforms:
                indices = tuple(reversed(indices))

            if indices not in pair_transforms:
                show_warning('No pair registration found for selected images')
            else:
                widgets = self.get_all_widgets()
                widgets.pop('registration.modify_pair_registration')
                self.temp_widget_state = TemporarilyDisabledWidgets(widgets)
                self.temp_widget_state.init()
                self.pair_indices = indices
                pair_transform = np.array(pair_transforms[indices].sel(t=0))
                eye = np.eye(max(pair_transform.shape))
                pair_transforms = pair_transform, eye
                self._clear_napari_view(self.viewer)
                for index, (sim_index, color) in enumerate(zip(indices, colors)):
                    self._add_napari_image(self.viewer, self.reg.sims[sim_index], labels[sim_index],
                                           pair_transforms[index], color, affine_event=True)
                self.update_pair_metrics()

    def calc_mod_pair_transform(self):
        transforms = [layer.affine.affine_matrix for layer in self.viewer.layers]
        matsize = len(si_utils.get_spatial_dims_from_sim(self.reg.sims[0])) + 1
        transform = calculate_rigid_difference(transforms[1][-matsize:, -matsize:],
                                               transforms[0][-matsize:, -matsize:])
        return param_utils.affine_to_xaffine(transform)

    def registration_process(self):
        if not self.reg.is_pairs_registered():
            show_warning('Perform pair registration first')
        else:
            message = 'Global registration was already performed. ' if self.reg.is_global_registered() else ''
            message += 'Run global registration?'
            reply = QMessageBox.question(None, 'muvis-align', message,
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                with TqdmCallback(tqdm_class=progress, desc='Global registration', bar_format=" "), \
                     TemporarilyDisabledWidgets(self.get_all_widgets()), \
                     VisibleActivityDock(self.viewer):
                    results = self.reg.register_global(self.reg.sims, self.reg.msims,
                                                       register_indices=self.reg.register_indices,
                                                       params=self.params['registration'])
                self.reg.save_mappings(results['mappings'])
                self.reg.save_metrics(results['metrics'])
                self.enable_tabs(True, 4)
                self.update_registered()

    def preview_fusion(self):
        self.reg.params_general = {'output': {}}
        self.reg.fusion_params = self.params['fusion']
        self._clear_napari_view(self.viewer)
        self._update_napari_data(self.viewer, 'Fused', transform_key=self.reg.reg_transform_key, fusion_method=self.params['fusion']['method'])
        self.view_mode = ViewMode.FUSED

    def fusion_process(self):
        message = 'Fusion was already performed. ' if self.reg.is_fused() else ''
        message += 'Export fused data?'
        reply = QMessageBox.question(None, 'muvis-align', message,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            operation = self.params['registration']['operation']
            output_filename = operation.split()[0] + 'ed'
            tile_size = self.params['fusion']['tile_size']
            if ',' in tile_size:
                tile_size = [int(size.strip()) for size in tile_size.split(',')]
            elif isinstance(tile_size, str):
                tile_size = int(tile_size.strip())
            with TqdmCallback(tqdm_class=progress, desc='Fusion', bar_format=" "), \
                 TemporarilyDisabledWidgets(self.get_all_widgets()), \
                 VisibleActivityDock(self.viewer):
                fused_image, _ = self.reg.fuse(self.reg.sims, fusion_method=self.params['fusion']['method'],
                                               output_spacing=self.params['fusion']['spacing'],
                                               output_filename=output_filename,
                                               tile_size=tile_size, ome_version=self.params['fusion']['ome_version'])
            self._clear_napari_view(self.viewer)
            self._add_napari_image(self.viewer, fused_image, 'Fused')
            self.reg.state = RegState.FUSED
            self.view_mode = ViewMode.FUSED
