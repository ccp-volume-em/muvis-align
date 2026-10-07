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


def translation_affine(shift_y=0.0, shift_x=0.0, t_coords=None):
    """A 2D translation as an xarray affine, with a t dim when t_coords are given."""
    from multiview_stitcher import param_utils
    return param_utils.affine_to_xaffine(param_utils.affine_from_translation([shift_y, shift_x]), t_coords=t_coords)


def grid_graph(rows=4, cols=4, size=100.0, step=90.0, outlier=None):
    """Tiles on a grid, overlapping their neighbours by 10: every pair registers to identity, bar `outlier`."""
    import networkx as nx
    import xarray as xr

    graph = nx.Graph()
    for row in range(rows):
        for col in range(cols):
            graph.add_node(row * cols + col, stack_props={
                'shape': {'y': int(size), 'x': int(size)}, 'spacing': {'y': 1.0, 'x': 1.0},
                'origin': {'y': row * step, 'x': col * step}, 'transform': translation_affine()})
    for row in range(rows):
        for col in range(cols):
            for other_row, other_col in ((row, col + 1), (row + 1, col)):
                if other_row < rows and other_col < cols:
                    node, other = row * cols + col, other_row * cols + other_col
                    lower = np.array([other_row * step, other_col * step])
                    upper = np.array([row * step, col * step]) + size
                    graph.add_edge(node, other, transform=translation_affine(), quality=0.9, overlap=0.1,
                                   bbox=xr.DataArray(np.array([lower, upper]), dims=['point_index', 'dim']))
    if outlier is not None:
        graph.edges[outlier]['transform'] = translation_affine(0.0, 30.0)
    return graph


def prepared_registration(input_path, output_path, preprocess=True):
    """An MVSRegistration over `input_path`, its data initialised and (by default) pre-processed."""
    from muvis_align.MVSRegistration import MVSRegistration

    registration = MVSRegistration()
    registration.init(operation='register', input_path=input_path, output_path=Path(output_path).as_posix() + '/')
    registration.init_data()
    if preprocess:
        registration.preprocess(registration.msims)
    return registration


def registration_from_resource(resource_file, output_path=None):
    """An MVSRegistration initialised from a resources/ project file: (registration, its first operation's params)."""
    import yaml
    from muvis_align.MVSRegistration import MVSRegistration

    with open(Path('resources') / resource_file, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    operation_params = params['operations'][0]
    if output_path is not None:
        operation_params['output']['path'] = Path(output_path).as_posix() + '/'
    registration = MVSRegistration()
    registration.init_params(params['general'], operation_params)
    registration.init_data()
    return registration, operation_params


class FakeBar:
    """Stands in for napari's progress bar - no Qt needed."""

    def __init__(self, **kwargs):
        self.total = kwargs.get('total')
        self.n = 0
        self.closed = False

    def update(self, step=1):
        self.n += step

    def close(self):
        self.closed = True


def make_phase_factory(phases=1, desc='Operation', progress_class=FakeBar, **kwargs):
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    factory = NapariPhaseProgress(progress_class=progress_class, desc=desc, phases=phases, **kwargs)
    # the heartbeat only logs; nothing here waits long enough for it to fire
    factory.heartbeat_seconds = 0
    return factory


def percent(factory):
    return factory._position / factory.ticks * 100


def recording_phase_factory():
    """A progress factory whose phases record their desc, total and steps done: (factory, records)."""
    records = []

    class RecordingPhase:
        def __init__(self, total=None, desc=None, **_):
            self.record = {'desc': desc, 'total': total, 'done': 0}
            records.append(self.record)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, count=1):
            self.record['done'] += count

    return RecordingPhase, records
