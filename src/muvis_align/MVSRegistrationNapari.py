from multiview_stitcher import spatial_image_utils as si_utils
from qtpy.QtCore import QObject, Signal, Slot

from src.muvis_align.image.util import get_sim_shape_2d
from src.muvis_align.MVSRegistration import MVSRegistration


class MVSRegistrationNapari(QObject, MVSRegistration):
    update_napari_shapes = Signal(str, str)
    update_napari_data = Signal(str, str)

    def __init__(self, viewer, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer
        self.update_napari_shapes.connect(self._update_napari_shapes)
        self.update_napari_data.connect(self._update_napari_data)

    @Slot(str, str)
    def _update_napari_shapes(self, layer_name, transform_key):
        shapes = [get_sim_shape_2d(sim, transform_key=transform_key) for sim in self.sims]
        if len(shapes) > 0:
            text = {'string': '{labels}'}
            features = {'labels': self.file_labels}
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = shapes
                self.viewer.layers[layer_name].text = text
                self.viewer.layers[layer_name].features = features
            else:
                self.viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5)
            self.viewer.show()

    @Slot(str, str)
    def _update_napari_data(self, layer_name, transform_key):
        fused, _ = self.fuse(self.sims, transform_key=transform_key, fusion_method='additive')
        fused_scale = si_utils.get_spacing_from_sim(fused, asarray=True)
        fused_position = si_utils.get_origin_from_sim(fused, asarray=True)
        if fused is not None:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = fused
            else:
                self.viewer.add_image(fused, name=layer_name, scale=fused_scale, translate=fused_position)
            self.viewer.show()
