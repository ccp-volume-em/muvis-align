import numpy as np

from src.muvis_align.image.color_conversion import *
from src.muvis_align.image.util import get_image_quantile


def create_axes_metadata(dimension_order):
    axes = []
    for dimension in dimension_order:
        unit1 = None
        if dimension == 't':
            type1 = 'time'
            unit1 = 'millisecond'
        elif dimension == 'c':
            type1 = 'channel'
        else:
            type1 = 'space'
            unit1 = 'micrometer'
        axis = {'name': dimension, 'type': type1}
        if unit1 is not None and unit1 != '':
            axis['unit'] = unit1
        axes.append(axis)
    return axes


def create_transformation_metadata(dimension_order, pixel_size_um, factor, translation_um={}, rotation=None):
    metadata = []
    metadata_scale = []
    metadata_translation = []
    for dim in dimension_order:
        if dim in 'xy':
            pixel_size_scale1 = pixel_size_um[dim] * factor
        else:
            pixel_size_scale1 = 1
        if pixel_size_scale1 == 0:
            pixel_size_scale1 = 1
        metadata_scale.append(pixel_size_scale1)

        # translation_pyramid = translation + (scale - 1) * pixel_size / 2
        if dim in 'xy':
            correction = (factor - 1) * pixel_size_um[dim] / 2
        else:
            correction = 0
        metadata_translation.append(translation_um.get(dim, 0) + correction)

    metadata.append({'type': 'scale', 'scale': metadata_scale})
    metadata.append({'type': 'translation', 'translation': metadata_translation})
    # Supported in ome-zarr V0.6
    #if rotation is not None:
    #    metadata.append({'type': 'rotation', 'rotation': rotation})
    return metadata


def create_channel_metadata(source, ome_version):
    channels = source.get_channels()
    nchannels = source.get_nchannels()

    if len(channels) < nchannels == 3:
        labels = ['Red', 'Green', 'Blue']
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]

    omezarr_channels = []
    for channeli, channel0 in enumerate(channels):
        channel = channel0.copy()
        color = channel.get('color', (1, 1, 1, 1))
        channel['color'] = rgba_to_hexrgb(color)
        if 'window' not in channel:
            channel['window'] = source.get_channel_window(channeli)
        omezarr_channels.append(channel)

    metadata = {
        'version': ome_version,
        'channels': omezarr_channels,
    }
    return metadata


def create_channel_ome_metadata(data, dimension_order, channels, ome_version):
    if 'c' in dimension_order:
        nchannels = data.shape[dimension_order.index('c')]
    else:
        nchannels = 1
    if channels is None or len(channels) < nchannels:
        if nchannels == 3:
            labels = ['Red', 'Green', 'Blue']
            colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
            channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]
        else:
            channels = [{'label': f'Channel {channeli}'} for channeli in range(nchannels)]

    omezarr_channels = []
    for channeli, channel0 in enumerate(channels):
        channel = channel0.copy()
        color = channel.get('color', (1, 1, 1, 1))
        channel['color'] = rgba_to_hexrgb(color)
        if 'window' not in channel:
            channel['window'] = get_channel_window(data, dimension_order, channeli)
        omezarr_channels.append(channel)

    metadata = {
        'version': ome_version,
        'channels': omezarr_channels,
    }
    return metadata


def get_channel_window(data, dimension_order, channeli):
    min_quantile = 0.001
    max_quantile = 0.999

    if data.dtype.kind == 'f':
        #info = np.finfo(dtype)
        min, max = 0, 1
    else:
        info = np.iinfo(data.dtype)
        min, max = info.min, info.max

    if data.dtype.itemsize == 1:
        start, end = min, max
    else:
        if 'c' in dimension_order:
            data = np.take(data, channeli, axis=dimension_order.index('c'))
        start, end = get_image_quantile(data, min_quantile), get_image_quantile(data, max_quantile)
    window = {'min': min, 'max': max, 'start': start, 'end': end}
    return window


def scale_dimensions_xy(shape0, dimension_order, scale):
    shape = []
    if scale == 1:
        return shape0
    for shape1, dimension in zip(shape0, dimension_order):
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape.append(shape1)
    return shape


def scale_dimensions_dict(shape0, scale):
    shape = {}
    if scale == 1:
        return shape0
    for dimension, shape1 in shape0.items():
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape[dimension] = shape1
    return shape
