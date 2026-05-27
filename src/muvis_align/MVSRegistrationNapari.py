from magicclass.ext.napari import ViewerWidget
from multiview_stitcher import spatial_image_utils as si_utils
from qtpy.QtCore import QObject, Signal, Slot

from src.muvis_align.image.util import get_sim_shape_2d, draw_keypoints_matches_napari, get_overlap_shapes
from src.muvis_align.MVSRegistration import MVSRegistration


class MVSRegistrationNapari(QObject, MVSRegistration):
    clear_napari_view = Signal()
    update_napari_view_shapes = Signal(str, str, bool)
    update_napari_view_data = Signal(str, str, bool)

    clear_napari_overview = Signal()
    update_napari_overview_shapes = Signal(str, str, bool)
    update_napari_overview_data = Signal(str, str)

    def __init__(self, viewer, overview, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer
        self.overview = overview
        self.clear_napari_view.connect(self._clear_napari_view)
        self.update_napari_view_shapes.connect(self._update_napari_view_shapes)
        self.update_napari_view_data.connect(self._update_napari_view_data)
        self.clear_napari_overview.connect(self._clear_napari_overview)
        self.update_napari_overview_shapes.connect(self._update_napari_overview_shapes)
        self.update_napari_overview_data.connect(self._update_napari_overview_data)
        self.selected_shape_index = None

    @Slot()
    def _clear_napari_view(self):
        self.viewer.layers.clear()

    @Slot()
    def _clear_napari_overview(self):
        self.overview.layers.clear()

    @Slot(str, str, bool)
    def _update_napari_view_shapes(self, layer_name, transform_key, overlaps):
        self._update_napari_shapes(self.viewer, layer_name, transform_key, overlaps)

    @Slot(str, str, bool)
    def _update_napari_overview_shapes(self, layer_name, transform_key, overlaps):
        self._update_napari_shapes(self.overview, layer_name, transform_key, overlaps)

    @Slot(str, str, bool)
    def _update_napari_view_data(self, layer_name, transform_key, show_preprocessed):
        self._update_napari_data(self.viewer, layer_name, transform_key, show_preprocessed)

    @Slot(str, str)
    def _update_napari_overview_data(self, layer_name, transform_key):
        self._update_napari_data(self.overview, layer_name, transform_key)

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
        shapes = [get_sim_shape_2d(sim, transform_key=transform_key) for sim in self.sims]
        refs = [str(index) for index in range(len(self.sims))]
        labels = list(self.file_labels)
        face_colors = [(1, 1, 1) for _ in range(len(self.sims))]
        if overlaps:
            shapes2, pairs = get_overlap_shapes(self.sims, transform_key=transform_key)
            shapes += shapes2
            refs += [f'{index1} {index2}' for index1, index2 in pairs]
            labels += ['' for _ in pairs]
            face_colors += [(1, 1, 0) for _ in pairs]
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
                layer = viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5,
                                          face_color=face_colors)
                @viewer.mouse_move_callbacks.append
                def on_mouse_move(viewer, event):
                    self.selected_shape_index = layer._value[0]

                @viewer.mouse_drag_callbacks.append
                def on_mouse_drag(viewer, event):
                    if event.type == "mouse_press" and event.button == 1:
                        if viewer.layers.selection.active == layer:
                            self.on_selection_change(refs[self.selected_shape_index])
                    yield


    def on_selection_change(self, ref):
        print(f"Currently selected shape: {ref}")

    def _update_napari_features(self, fixed_data2, fixed_points,
                               moving_data2, moving_points,
                               matches, inliers):

        layers = draw_keypoints_matches_napari(fixed_data2, fixed_points,
                                               moving_data2, moving_points,
                                               matches, inliers, points_color='blue')

        self.viewer.layers.clear()
        for data, kwargs, layer_type in layers:
            if layer_type == "image":
                self.viewer.add_image(data, **kwargs)
            elif layer_type == "points":
                self.viewer.add_points(data, **kwargs)
            elif layer_type == "shapes":
                self.viewer.add_shapes(data, **kwargs)
