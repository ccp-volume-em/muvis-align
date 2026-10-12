import json
import os

import numpy as np
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.color_conversion import *
from muvis_align.image.util import get_image_quantile



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



def ngff_dims_to_sim_dims(file_dims):
    """The dims a sim built from `file_dims` ends up with: SPATIAL_IMAGE_DIMS order, the file's spatial dims
    plus 't' and 'c', which get_sim_from_array always adds. Lets metadata be read without building the msim.
    """
    return [dim for dim in si_utils.SPATIAL_IMAGE_DIMS
            if dim in sim_forced_dims or dim in file_dims]


sim_forced_dims = ('t', 'c')
sim_spatial_dims = tuple(dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim not in sim_forced_dims)


def read_ome_zarr_source_metadata(path):
    """The metadata an ImageSource needs from an OME-Zarr, without constructing a msim.

    Read from the consolidated root zarr.json (which ngff_zarr does not use) when possible, else via ngff_zarr's
    version-aware parse (non-consolidated, v0.4, remote, or other coordinate transformations).

    Returns a dict of dimension_order, shapes, dtype, pixel_sizes, position, nchannels, omero and paths, in the
    sim's dimension order.
    """
    metadata = _read_consolidated_ome_zarr_metadata(path)
    if metadata is None:
        metadata = _read_ngff_ome_zarr_metadata(path)
    return metadata


def _build_source_metadata(file_dims, file_shapes, dtype, scales, translation, omero, paths=None):
    """Map per-level file dims/shapes/scales onto the sim's dimension order. `scales` is one dict per level,
    `translation` one for the finest level; missing dims default to 1.0 / 0.0. `paths` are the levels' array
    paths within the store, None when unknown.
    """
    sim_dims = ngff_dims_to_sim_dims(file_dims)
    spatial_dims = [dim for dim in sim_dims if dim in sim_spatial_dims]
    shapes = []
    for shape in file_shapes:
        sizes = dict(zip(file_dims, shape))
        shapes.append(tuple(sizes.get(dim, 1) for dim in sim_dims))
    pixel_sizes = [{dim: float(scale.get(dim, 1.0)) for dim in spatial_dims} for scale in scales]
    position = {dim: float(translation.get(dim, 0.0)) for dim in spatial_dims}
    nchannels = dict(zip(sim_dims, shapes[0])).get('c', 1)
    return {'dimension_order': ''.join(sim_dims), 'shapes': shapes, 'dtype': np.dtype(dtype),
            'pixel_sizes': pixel_sizes, 'position': position, 'nchannels': int(nchannels),
            'omero': omero, 'paths': paths}


def _read_consolidated_ome_zarr_metadata(path):
    """Parse the consolidated root zarr.json directly. Returns None, rather than guessing, for anything it
    cannot fully account for, so it is exactly equivalent to ngff_zarr's parse wherever it applies.
    """
    root_file = os.path.join(str(path), 'zarr.json')
    if not os.path.isfile(root_file):
        return None
    try:
        with open(root_file) as file:
            root = json.load(file)
    except (OSError, ValueError):
        return None
    if root.get('zarr_format') != 3:
        return None
    nodes = root.get('consolidated_metadata', {}).get('metadata')
    if not nodes:
        return None
    multiscales = root.get('attributes', {}).get('ome', {}).get('multiscales')
    if not multiscales:
        return None
    multiscale = multiscales[0]
    # a multiscales-level transform composes with the per-dataset ones - not handled here
    if multiscale.get('coordinateTransformations'):
        return None
    file_dims = tuple(axis['name'] for axis in multiscale.get('axes', []))
    known_dims = si_utils.SPATIAL_IMAGE_DIMS
    if not file_dims or any(dim not in known_dims for dim in file_dims):
        return None

    file_shapes, scales, dtypes = [], [], set()
    translation = {}
    for index, dataset in enumerate(multiscale.get('datasets', [])):
        node = nodes.get(dataset.get('path'))
        if not node or node.get('node_type') != 'array':
            return None
        transforms = dataset.get('coordinateTransformations', [])
        if any(transform.get('type') not in ('scale', 'translation') for transform in transforms):
            return None
        scale_values = next((transform['scale'] for transform in transforms
                             if transform.get('type') == 'scale'), None)
        shape = node.get('shape')
        if scale_values is None or len(scale_values) != len(file_dims):
            return None
        if shape is None or len(shape) != len(file_dims):
            return None
        file_shapes.append(tuple(shape))
        scales.append(dict(zip(file_dims, scale_values)))
        dtypes.add(node.get('data_type'))
        if index == 0:
            translation_values = next((transform['translation'] for transform in transforms
                                       if transform.get('type') == 'translation'), None)
            if translation_values is not None:
                if len(translation_values) != len(file_dims):
                    return None
                translation = dict(zip(file_dims, translation_values))
    if not file_shapes or len(dtypes) != 1:
        return None
    try:
        dtype = np.dtype(dtypes.pop())
    except TypeError:
        return None

    ome = root.get('attributes', {}).get('ome', {})
    paths = [dataset.get('path') for dataset in multiscale.get('datasets', [])]
    return _build_source_metadata(file_dims, file_shapes, dtype, scales, translation,
                                  ome.get('omero'), paths=paths)


def _read_ngff_ome_zarr_metadata(path):
    """ngff_zarr's version-aware parse (v0.4, non-consolidated, remote stores), still without building a msim.
    """
    # imported here: ngff_utils pulls in a large dependency chain the fast path never needs
    from multiview_stitcher import ngff_utils

    multiscales = ngff_utils.read_ngff_multiscales(path)
    images = multiscales.images
    file_dims = tuple(images[0].dims)
    omero = getattr(multiscales.metadata, 'omero', None)
    if omero is not None and not isinstance(omero, dict):
        dump = getattr(omero, 'model_dump', None)
        omero = dump() if dump is not None else None
    datasets = getattr(multiscales.metadata, 'datasets', None) or []
    paths = [getattr(dataset, 'path', None) for dataset in datasets]
    if len(paths) != len(images) or not all(paths):
        # makes _load_data() fall back to the msim reader
        paths = None
    return _build_source_metadata(
        file_dims,
        [tuple(image.data.shape) for image in images],
        images[0].data.dtype,
        [dict(image.scale) for image in images],
        dict(images[0].translation),
        omero,
        paths=paths)
