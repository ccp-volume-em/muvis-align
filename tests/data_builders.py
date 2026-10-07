from pathlib import Path

import numpy as np
import pytest
import tifffile
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'S000'
TIFF_FILES = ['000_000_0.tiff', '000_001_0.tiff', '001_000_0.tiff', '001_001_0.tiff']
ZARR_FILES = ['S000_000_000.ome.zarr', 'S000_000_001.ome.zarr', 'S000_001_000.ome.zarr', 'S000_001_001.ome.zarr']


def make_dummy_blob_spatial_image(shape, points, dims, seed=1234, noise_max=16, radius=None):
    ndim = len(shape)
    if len(dims) != ndim:
        raise ValueError('shape and dims must have the same length')
    if ndim not in (2, 3):
        raise ValueError('only 2D and 3D dummy blob data are supported')

    if radius is None:
        radius = 2.5 if ndim == 2 else 1.75

    rng = np.random.default_rng(seed)
    image = rng.integers(0, noise_max, size=shape, dtype=np.uint8)
    grids = np.ogrid[tuple(slice(0, size) for size in shape)]

    for point in np.asarray(points, dtype=float):
        if len(point) != ndim:
            raise ValueError('point dimensionality must match shape dimensionality')
        distance2 = np.zeros(shape, dtype=np.float32)
        for axis, grid in enumerate(grids):
            distance2 += (grid - point[axis]) ** 2
        blob = distance2 <= radius ** 2
        image = np.maximum(image, np.where(blob, 255, 0).astype(np.uint8))

    return si_utils.get_sim_from_array(image, dims=list(dims))


def make_dummy_blob_spatial_image_2d(shape, points, dims='yx', **kwargs):
    if len(shape) != 2 or len(dims) != 2:
        raise ValueError('2D helper requires 2D shape and dims')
    return make_dummy_blob_spatial_image(shape, points, dims, **kwargs)


def make_sim(data, dims='yx', scale=None, translation=None, transform_key='source_metadata', **kwargs):
    """A sim over `data`, unit pixels at the origin unless given; kwargs go to get_sim_from_array (affine, c_coords)."""
    spatial_dims = [dim for dim in dims if dim in 'zyx']
    scale = scale or {dim: 1.0 for dim in spatial_dims}
    translation = translation or {dim: 0.0 for dim in spatial_dims}
    return si_utils.get_sim_from_array(data, dims=list(dims), scale=scale, translation=translation,
                                       transform_key=transform_key, **kwargs)


def make_msim(data, dims='yx', scale_factors=(), **kwargs):
    """make_sim() as a msim, one level per scale factor beyond the first."""
    return msi_utils.get_msim_from_sim(make_sim(data, dims, **kwargs), scale_factors=list(scale_factors))


def write_tiff_pyramid(path, shape=(2048, 2048), levels=3, dtype=np.uint16, data=None, tile=(256, 256), **kwargs):
    """A tiled TIFF whose subifds halve per level; kwargs go to the first write (metadata, photometric)."""
    data = np.zeros(shape, dtype=dtype) if data is None else data
    with tifffile.TiffWriter(str(path)) as writer:
        writer.write(data, subifds=levels - 1, tile=tile, **kwargs)
        for level in range(1, levels):
            step = 2 ** level
            writer.write(data[..., ::step, ::step], subfiletype=1, tile=tile)
    return str(path)


def write_ome_zarr(path, dim_order='yx', shape=(256, 192), pixel_size=None, translation=None, levels=1,
                   dtype=np.uint16, **kwargs):
    """An OME-Zarr written the way convert writes one, y/x halving per level; kwargs go to save_ome_multiscale_levels."""
    from muvis_align.image.ome_zarr_helper import save_ome_multiscale_levels

    spatial_dims = [dim for dim in dim_order if dim in 'zyx']
    pixel_size = pixel_size or {dim: 1.0 for dim in spatial_dims}
    translation = translation or {dim: 0.0 for dim in spatial_dims}
    data = np.zeros(shape, dtype=dtype)
    written = []
    for level in range(levels):
        factor = 2 ** level
        slicing = tuple(slice(None, None, factor if dim in 'yx' else 1) for dim in dim_order)
        written.append((data[slicing], {dim: pixel_size[dim] * (factor if dim in 'yx' else 1) for dim in spatial_dims}))
    save_ome_multiscale_levels(str(path), written, dim_order, [], translation, **kwargs)
    return str(path)


def _assert_same_sizes(got, expected):
    assert set(got) == set(expected)
    for dim in expected:
        assert got[dim] == pytest.approx(float(expected[dim]))


def assert_same_metadata(got, expected, keys=('dimension_order', 'shapes', 'dtype', 'pixel_sizes')):
    """Compare two source metadata dicts on `keys`: shapes as tuples, pixel sizes and position approximately."""
    for key in keys:
        if key == 'shapes':
            assert [tuple(shape) for shape in got[key]] == [tuple(shape) for shape in expected[key]]
        elif key == 'pixel_sizes':
            assert len(got[key]) == len(expected[key])
            for got_level, expected_level in zip(got[key], expected[key]):
                _assert_same_sizes(got_level, expected_level)
        elif key == 'position':
            _assert_same_sizes(got[key], expected[key])
        else:
            assert got[key] == expected[key], key
