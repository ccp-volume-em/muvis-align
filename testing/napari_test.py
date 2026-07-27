from multiview_stitcher import spatial_image_utils as si_utils
import napari
import numpy as np
from napari_bbox.boundingbox import BoundingBoxLayer
from qtpy.QtCore import QObject, QThread, Signal, Slot
from threading import Thread

from src.muvis_align.image.util import create_sim_shape


class NapariTest2d:
    def __init__(self):
        self.viewer = napari.Viewer()

    def start(self):
        shape_data = [
            # Shape 1: Constrained entirely to Z=0
            np.array([[0, 10, 10], [0, 10, 90], [0, 90, 90], [0, 90, 10]]),
            # Shape 2: Constrained entirely to Z=1
            np.array([[1, 20, 20], [1, 20, 80], [1, 80, 80], [1, 80, 20]])
        ]

        # Add the 3D shapes layer to the viewer
        shapes_layer = self.viewer.add_shapes(
            shape_data,
            shape_type='rectangle',
            name='3D_Rectangle',
            edge_width=3
        )


class NapariTest3d(QThread):
    update_shapes = Signal(str, list, list)
    update_data = Signal(str, list)
    update_bounds = Signal(str, list, list)

    def __init__(self, ndisplay=None):
        super().__init__()
        self.viewer = napari.Viewer(ndisplay=ndisplay)
        self.update_shapes.connect(self._update_shapes)
        self.update_data.connect(self._update_data)
        self.update_bounds.connect(self._update_bounds)

    def run(self):
        # self.update_shapes.emit('test_layer', [[0, 0], [0, 1], [1, 1], [1, 0]], ['tile'])

        # self.update_shapes.emit('shapes',
        #                         [[[0, 20, 20], [10, 50, 20], [10, 50, 80], [0, 20, 80]],
        #                          [[20, 20, 20], [30, 50, 20], [30, 50, 80], [20, 20, 80]]],
        #                         ['tile1', 'tile2'])

        # self.update_data.emit('data', np.asarray(create_sim(shape=(10, 30, 60))).tolist())

        # self.update_shapes.emit('shapes',
        #                         [[0, 0, 0], [10, 30, 0], [10, 30, 60], [0, 0, 60]],
        #                         ['tile1'])

        # sims = [create_sim(shape=(10, 10, 10)) for _ in range(10)]
        # shapes = np.array([get_sim_shape(sim, transform_key='source') for sim in sims]).tolist()
        # print(shapes)
        # self.update_shapes.emit('test_layer', shapes, [index for index, _ in enumerate(sims)])

        # sim = create_sim(shape=(10, 30, 60))
        # self.update_data.emit('data', np.asarray(sim).tolist())
        # self.update_shapes.emit('test_layer', get_sim_shape(sim).tolist(), ['test'])

        # box_vertices = np.array([
        #     [0, 0, 0], [0, 0, 20], [0, 20, 20], [0, 20, 0],
        #     [20, 0, 0], [20, 0, 20], [20, 20, 20], [20, 20, 0]
        # ])
        # box_faces = np.array([
        #     [0, 1, 2, 3],  # Bottom
        #     [4, 5, 6, 7],  # Top
        #     [0, 1, 5, 4],  # Side 1
        #     [2, 3, 7, 6],  # Side 2
        #     [1, 2, 6, 5],  # Side 3
        #     [0, 3, 7, 4]  # Side 4
        # ])
        # all_vertices = [box_vertices[face] for face in box_faces]
        # self.update_shapes.emit('layer', all_vertices, ['label'])

        sims = [create_sim(shape=(10, 30, 60)), create_sim(shape=(5, 20, 40), position={'x':5,'y':0,'z':0})]
        self.update_bounds.emit('layer', [create_sim_shape(sim).tolist() for sim in sims], ['label1', 'label2'])

    @Slot(str, list, list)
    def _update_shapes(self, layer_name, shapes, labels):
        if len(shapes) > 0:
            shapes = np.array(shapes)
            # text = {'string': '{labels}'}
            # features = {'labels': labels}
            self.viewer.add_shapes(shapes, name=layer_name, shape_type='polygon',
                                   #text=text, features=features,
                                   opacity=0.5, blending='translucent_no_depth')

            # layer = self.viewer.add_points(shapes, name=layer_name, opacity=0.5, blending='translucent_no_depth')
            # layer.bounding_box.visible = True
            # layer.bounding_box.line_color = 'cyan'
            self.viewer.show()

    @Slot(str, list, list)
    def _update_bounds(self, layer_name, shapes, labels):
        text = {'string': '{labels}'}
        features = {'labels': labels}

        bbox_layer = BoundingBoxLayer(shapes, edge_color='green', face_color='transparent',
                                      text=text, features=features)
        self.viewer.add_layer(bbox_layer)

    @Slot(str, list)
    def _update_data(self, layer_name, data):
        layer = self.viewer.add_image(np.asarray(data), name=layer_name,
                                      opacity=0.5, blending='translucent_no_depth')
        layer.bounding_box.visible = True
        layer.bounding_box.line_color = 'cyan'
        self.viewer.show()

def create_sim(shape, scale={'z': 1, 'x': 1, 'y': 1}, position={'x': 0, 'y': 0, 'z': 0}, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data = np.random.random(shape)
    sim = si_utils.get_sim_from_array(
        data,
        dims=list('zyx'),
        scale=scale,
        translation=position,
        transform_key='source'
    )
    return sim


if __name__ == '__main__':
    #napari_test = NapariTest3d(ndisplay=3)
    napari_test = NapariTest2d()
    napari_test.start()
    napari.run()
