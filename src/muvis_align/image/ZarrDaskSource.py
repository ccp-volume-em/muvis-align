import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os.path

from src.muvis_align.image.DaskSource import DaskSource
from src.muvis_align.util import convert_to_um


class ZarrDaskSource(DaskSource):
    def init_metadata(self):
        location = parse_url(self.filename)
        if location is None:
            raise FileNotFoundError(f'Error parsing ome-zarr file {self.filename}')
        if 'bioformats2raw.layout' in location.root_attrs:
            location = parse_url(os.path.join(self.filename, '0'))
            if location is None:
                raise FileNotFoundError(f'Error parsing ome-zarr file {self.filename}')
        reader = Reader(location)
        nodes = list(reader())
        image_node = nodes[0]
        self.data = image_node.data
        self.metadata = image_node.metadata

        self.shapes = [level.shape for level in self.data]
        self.shape = self.shapes[0]
        self.dtype = self.data[0].dtype
        axes = self.metadata['axes']
        dims = ''.join([axis['name'] for axis in axes])
        self.dimension_order = dims
        units = {axis['name']: axis['unit'] for axis in axes if 'unit' in axis}
        dims_used = [dim for dim, shape in zip(dims, self.shape) if dim in 'xyz' and shape > 1]

        pixel_sizes = []
        position = {}
        channels = []
        scales = []
        scale_factors = []
        scale0 = {}
        for ct_index, transforms in enumerate(self.metadata.get('coordinateTransformations', [])):
            scale = {}
            position = {}
            for transform in transforms:
                if transform['type'] == 'scale':
                    scale = {dim: value for dim, value in zip(dims, transform['scale']) if dim in dims_used}
                if transform['type'] == 'translation':
                    position = {dim: value for dim, value in zip(dims, transform['translation']) if dim in dims_used}
            if ct_index == 0:
                scale0 = scale
            scales.append(scale)
            scale_factors.append({dim: value / scale0.get(dim, 1) for dim, value in scale.items()})
            pixel_size = {dim: convert_to_um(value, units.get(dim, '')) for dim, value in scale.items()}
            pixel_sizes.append(pixel_size)

        colormaps = self.metadata.get('colormap', [])
        for channeli, channel0 in enumerate(self.metadata.get('channel_names', [])):
            channel = {'label': channel0}
            if channeli < len(colormaps):
                channel['color'] = colormaps[channeli][-1]
            channels.append(channel)
        self.pixel_sizes = pixel_sizes
        self.pixel_size = pixel_sizes[0]
        self.position = position
        self.rotation = 0
        self.channels = channels
        self.scales = scales
        self.scale_factors = scale_factors

    def get_data(self, level=0):
        if level < 0:
            return self.data
        else:
            return self.data[level]
