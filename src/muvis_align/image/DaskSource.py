import numpy as np

from src.muvis_align.util import get_value_units_micrometer, find_all_numbers, split_numeric_dict, eval_context, \
    check_contains_value


class DaskSource:
    default_physical_unit = 'µm'

    def __init__(self, filename, source_metadata=None):
        self.filename = filename
        self.dimension_order = ''
        self.is_rgb = False
        self.shapes = []
        self.shape = []
        self.dtype = None
        self.pixel_sizes = []
        self.pixel_size = {}
        self.scales = []
        self.scale_factors = []
        self.position = {}
        self.rotation = 0
        self.channels = []
        self.data = []
        self.init_metadata()
        self.fix_metadata(source_metadata)

    def init_metadata(self):
        raise NotImplementedError("Dask source should implement init_metadata() to initialize metadata")

    def fix_metadata(self, source_metadata=None):
        if isinstance(source_metadata, dict):
            filename_numeric = find_all_numbers(self.filename)
            filename_dict = {key: int(value) for key, value in split_numeric_dict(self.filename).items()}
            context = {'filename_numeric': filename_numeric, 'fn': filename_numeric} | filename_dict
            if 'position' in source_metadata:
                translation = source_metadata['position']
                if 'x' in translation:
                    if not check_contains_value(translation['x'], 'source'):
                        self.position['x'] = eval_context(translation, 'x', 0, context)
                    if check_contains_value(translation['x'], 'invert'):
                        if isinstance(self.position['x'], (tuple, list)):
                            self.position['x'] = -self.position['x'][0], self.position['x'][1]
                        else:
                            self.position['x'] = -self.position['x']
                if 'y' in translation:
                    if not check_contains_value(translation['y'], 'source'):
                        self.position['y'] = eval_context(translation, 'y', 0, context)
                    if check_contains_value(translation['y'], 'invert'):
                        if isinstance(self.position['y'], (tuple, list)):
                            self.position['y'] = -self.position['y'][0], self.position['y'][1]
                        else:
                            self.position['y'] = -self.position['y']
                if 'z' in translation:
                    if not check_contains_value(translation['z'], 'source'):
                        self.position['z'] = eval_context(translation, 'z', 0, context)
                    if check_contains_value(translation['z'], 'invert'):
                        if isinstance(self.position['z'], (tuple, list)):
                            self.position['z'] = -self.position['z'][0], self.position['z'][1]
                        else:
                            self.position['z'] = -self.position['z']
            if 'scale' in source_metadata:
                scale = source_metadata['scale']
                if 'x' in scale:
                    if not check_contains_value(scale['x'], 'source'):
                        self.pixel_size['x'] = eval_context(scale, 'x', 1, context)
                if 'y' in scale:
                    if not check_contains_value(scale['y'], 'source'):
                        self.pixel_size['y'] = eval_context(scale, 'y', 1, context)
                if 'z' in scale:
                    if not check_contains_value(scale['z'], 'source'):
                        self.pixel_size['z'] = eval_context(scale, 'z', 1, context)
            if 'rotation' in source_metadata:
                if not check_contains_value(source_metadata['rotation'], 'source'):
                    self.rotation = eval_context(source_metadata, 'rotation', 0, context)
                if check_contains_value(source_metadata['rotation'], 'invert'):
                    self.rotation = -self.rotation

        self.scale_factors = [{dim: value0 / value for dim, value, value0
                               in zip(self.dimension_order, shape, self.shape) if dim in 'xyz'}
                               for shape in self.shapes]

    def get_shape(self, level=0):
        # shape in pixels
        return self.shapes[level]

    def get_size(self, level=0, asarray=False, axes='zyx'):
        # size in pixels
        size = {dim: size for dim, size in zip(self.dimension_order, self.get_shape(level))}
        if asarray:
            return np.array([size[dim] for dim in axes if dim in size])
        else:
            return size

    def get_pixel_size(self, level=0, asarray=False, axes='zyx'):
        # pixel size in micrometers
        pixel_size = self.pixel_sizes[level]
        if asarray:
            return np.array([pixel_size[dim] for dim in axes if dim in pixel_size])
        else:
            return pixel_size

    def get_physical_size(self, asarray=False, axes='zyx'):
        pixel_size = self.get_pixel_size()
        size = self.get_size()
        physical_size = {dim: size[dim] * pixel_size[dim] for dim in size if dim in pixel_size}
        if asarray:
            return np.array([physical_size[dim] for dim in axes if dim in physical_size])
        else:
            return physical_size

    def get_position(self, asarray=False, axes='zyx'):
        # position in micrometers
        if asarray:
            return np.array([self.position[dim] for dim in axes if dim in self.position])
        else:
            return self.position

    def get_rotation(self):
        # rotation in degrees
        return self.rotation

    def get_nchannels(self):
        return self.get_size().get('c', 1)

    def get_channels(self):
        if len(self.channels) == 0:
            if self.is_rgb:
                return [{'label': ''}]
            else:
                return [{'label': ''}] * self.get_nchannels()
        return self.channels

    def get_data(self, level=0):
        if level < 0:
            return self.data
        else:
            return self.data[level]
