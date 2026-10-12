from types import SimpleNamespace

import pytest

import dask.array as da
import numpy as np
from dask.callbacks import Callback
import tifffile
from multiview_stitcher import msi_utils

from muvis_align.image.lazy_overview import lazy_section_overview, _contrast_limits, MsimLevels
from muvis_align.image.util import coarsest_level_within
from muvis_align.MVSRegistration import MVSRegistration

# z_y_x in the file name: z the section, y/x the tile in 64-pixel steps (1um pixels)
TILES = {(0, 0, 0): 10, (0, 0, 1): 20, (0, 1, 0): 30, (1, 0, 0): 40}


def registration(tmp_path, rotation=None, channels=1):
    for (z, y, x), value in TILES.items():
        if channels > 1:
            tifffile.imwrite(tmp_path / f'tile_{z}_{y}_{x}.tif', np.full((channels, 64, 64), value, dtype=np.uint8),
                             ome=True, metadata={'axes': 'CYX'})
        else:
            tifffile.imwrite(tmp_path / f'tile_{z}_{y}_{x}.tif', np.full((64, 64), value, dtype=np.uint8))
    reg = MVSRegistration()
    reg.init(operation='register', input_path=sorted(str(path) for path in tmp_path.glob('tile_*.tif')),
             output_path=tmp_path.as_posix() + '/output/')
    metadata = {'position': {'z': 'fn[-3]', 'y': 'fn[-2]*64', 'x': 'fn[-1]*64'},
                'scale': {'z': 1, 'y': 1, 'x': 1}}
    if rotation is not None:
        metadata['rotation'] = rotation
    reg.init_data(source_metadata=metadata)
    return reg


def overview(reg, preview_scale=1, **kwargs):
    return lazy_section_overview(reg.sources, reg.positions, reg._msim_transforms, reg._msim_output_order,
                                 reg.source_transform_key, z_scale=reg._msim_z_scale, preview_scale=preview_scale,
                                 **kwargs)


def test_each_tile_is_pasted_where_it_sits_in_its_own_section(tmp_path):
    msim = overview(registration(tmp_path))
    sim = msi_utils.get_sim_from_msim(msim)
    data = np.asarray(sim.data).squeeze()

    assert data.shape[0] == 2
    assert np.all(data[0, :64, :64] == 10) and np.all(data[0, :64, 64:128] == 20)
    assert np.all(data[0, 64:128, :64] == 30) and np.all(data[0, 64:128, 64:128] == 0)
    assert np.all(data[1, :64, :64] == 40) and np.all(data[1, :, 64:] == 0)
    assert float(sim.coords['x'][0]) == 0 and float(sim.coords['x'][1] - sim.coords['x'][0]) == 1


def test_channel_labels_make_each_source_a_channel_of_its_own_on_the_shared_grid(tmp_path):
    labels = [f'tile {index}' for index in range(len(TILES))]
    sim = msi_utils.get_sim_from_msim(overview(registration(tmp_path), channel_labels=labels))
    data = np.asarray(sim.data).squeeze()

    assert list(sim.coords['c'].values) == labels
    assert data.shape[:2] == (4, 2)
    assert np.all(data[0, 0, :64, :64] == 10) and np.all(data[0, 0, :64, 64:] == 0)
    assert np.all(data[1, 0, :64, 64:128] == 20) and np.all(data[1, 0, :64, :64] == 0)
    assert np.all(data[3, 1, :64, :64] == 40) and np.all(data[3, 0] == 0)


def test_each_channel_takes_its_contrast_from_its_own_sources(tmp_path):
    reg = registration(tmp_path)

    assert overview(reg).attrs['contrast_limits'] == [[10.0, 40.0]]
    assert overview(reg, channel_labels=list('abcd')).attrs['contrast_limits'] == [
        [10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]


def test_contrast_comes_from_a_spread_sample_of_coarsest_levels_without_their_zero_fill():
    # level 1 of reader `value`: a 4x4 block of value + 1 inside a zero border
    readers = [SimpleNamespace(pixel_sizes=[{}, {}],
                               level_data=lambda level, value=value: np.pad(np.full((4, 4), value + level), 2))
               for value in (100, 200, 300, 400, 500)]

    # and where the typical value shows: halfway between two equally common values
    assert _contrast_limits(readers, max_sources=2) == ([[101.0, 501.0]], [0.5])
    assert _contrast_limits(readers[:1]) == ([[101.0, 102.0]], [0.0])
    assert _contrast_limits([None]) == ([], [])
    assert _contrast_limits([SimpleNamespace(pixel_sizes=[{}], level_data=lambda level: np.zeros((4, 4)))]) == (
        [None], [None])


def test_a_section_is_built_only_when_it_is_asked_for_and_once(tmp_path):
    planes = overview(registration(tmp_path)).attrs['section_planes'][0]

    assert list(planes._planes) == []
    first = planes.plane(1, prefetch=False)
    assert list(planes._planes) == [1]
    assert planes.plane(1, prefetch=False) is first


def test_the_sections_around_a_viewed_one_are_built_in_the_background(tmp_path):
    planes = overview(registration(tmp_path)).attrs['section_planes'][0]
    planes.prefetch_radius = 1

    planes.plane(0)
    planes._background.submit(lambda: None).result()

    assert sorted(planes._planes) == [0, 1]


def test_kept_sections_stay_within_their_budget_the_least_recently_viewed_dropped(tmp_path):
    planes = overview(registration(tmp_path)).attrs['section_planes'][0]
    planes.max_planes = 1

    planes.plane(0, prefetch=False)
    planes.plane(1, prefetch=False)

    assert list(planes._planes) == [1]


def test_rotated_sources_get_no_lazy_overview(tmp_path):
    assert overview(registration(tmp_path, rotation=30)) is None


def test_multichannel_sources_keep_their_channels_in_each_plane(tmp_path):
    msim = overview(registration(tmp_path, channels=2))
    sim = msi_utils.get_sim_from_msim(msim)
    data = np.asarray(sim.data).squeeze()

    assert data.shape[:2] == (2, 2) and sim.sizes['c'] == 2
    assert np.all(data[:, 0, :64, :64] == 10) and np.all(data[:, 1, :64, :64] == 40)
    assert msim.attrs['contrast_limits'] == [[10.0, 40.0], [10.0, 40.0]]
    # a source's several channels cannot each be a channel of their own
    assert overview(registration(tmp_path, channels=2), channel_labels=list('abcd')) is None


def test_a_tile_fills_its_own_extent_at_a_pixel_size_not_a_whole_ratio_of_its_own(tmp_path):
    sim = msi_utils.get_sim_from_msim(overview(registration(tmp_path), preview_scale='0.4um'))
    row = np.asarray(sim.data).squeeze()[0, 0]
    centres = np.asarray(sim.coords['x'])

    # 1um pixels centred at 0..63 (the next tile from 64): its pixels' edges at -0.5 and 63.5
    assert float(centres[1] - centres[0]) == 0.4
    assert np.all(row[(centres >= -0.5) & (centres < 63.5)] == 10)
    assert np.all(row[(centres >= 63.5) & (centres < 127.5)] == 20)


def test_the_plane_takes_the_preview_scale_as_a_factor_coarsened_to_fit_its_byte_budget(tmp_path):
    reg = registration(tmp_path)

    assert float(np.diff(msi_utils.get_sim_from_msim(overview(reg, preview_scale=4)).coords['x'][:2])[0]) == 4
    capped = msi_utils.get_sim_from_msim(overview(reg, preview_scale=1, max_plane_bytes=40 * 40))
    assert float(np.diff(capped.coords['x'][:2])[0]) == 4


def test_each_source_is_read_at_its_coarsest_level_no_coarser_than_the_plane():
    source = SimpleNamespace(pixel_sizes=[{'y': 1, 'x': 1}, {'y': 2, 'x': 2}, {'y': 4, 'x': 4}])

    assert coarsest_level_within(source.pixel_sizes, {'y': 3, 'x': 3}) == 1
    assert coarsest_level_within(source.pixel_sizes, {'y': 4, 'x': 4}) == 2
    assert coarsest_level_within(source.pixel_sizes, {'y': 0.5, 'x': 0.5}) == 0


def test_readers_give_the_pixels_at_the_sources_place_and_a_missing_one_is_left_empty(tmp_path):
    reg = registration(tmp_path)
    # half-resolution stand-ins for pre-processed sources, the second one filtered out
    readers = [SimpleNamespace(pixel_sizes=[{'y': 2.0, 'x': 2.0}], dtype=np.uint8,
                               level_data=lambda level, value=value: np.full((32, 32), value + 1, dtype=np.uint8))
               for value in TILES.values()]
    readers[1] = None

    data = np.asarray(msi_utils.get_sim_from_msim(overview(reg, readers=readers)).data).squeeze()

    assert np.all(data[0, :64, :64] == 11) and np.all(data[0, :64, 64:128] == 0)
    assert np.all(data[0, 64:128, :64] == 31) and np.all(data[1, :64, :64] == 41)


def test_a_tile_read_leaves_dasks_global_callbacks_alone(tmp_path):
    """napari's slicing cache is a global dask callback, shared by every thread: a tile read on a reader thread
    must not run it, or it clears the cache's start times under napari's own compute of the plane."""
    reg = registration(tmp_path)
    readers = [SimpleNamespace(pixel_sizes=[{'y': 1.0, 'x': 1.0}], dtype=np.uint8,
                               level_data=lambda level, value=value: da.full((64, 64), value, dtype=np.uint8))
               for value in TILES.values()]
    planes = overview(reg, readers=readers).attrs['section_planes'][0]
    started = []

    with Callback(start=started.append):
        plane = planes.plane(0, prefetch=False)

    assert started == []
    assert np.all(plane[0, :64, :64] == 10) and np.all(plane[0, 64:128, :64] == 30)


def test_a_pre_processed_msim_is_read_at_its_own_levels(tmp_path):
    reg = registration(tmp_path)
    msims, _, _ = reg.preprocess(reg.ensure_msims(), normalisation='individual')
    sim = msi_utils.get_sim_from_msim(msims[0])

    levels = MsimLevels(msims[0])

    assert levels.pixel_sizes == [{'y': 1.0, 'x': 1.0}] and levels.dtype == sim.dtype
    assert np.array_equal(np.asarray(levels.level_data(0)), np.asarray(sim.data))


def placed_sources(tmp_path, arrays, positions, axes, scale):
    """A registration of one tiff a source, each placed by a source_metadata position."""
    for index, array in enumerate(arrays):
        tifffile.imwrite(tmp_path / f'source_{index}.tif', array, ome=True, metadata={'axes': axes})
    reg = MVSRegistration()
    reg.init(operation='register', input_path=sorted(str(path) for path in tmp_path.glob('source_*.tif')),
             output_path=tmp_path.as_posix() + '/output/')
    reg.init_data(source_metadata={'position': positions, 'scale': scale})
    return reg


def test_a_later_source_covers_an_earlier_one_where_they_overlap(tmp_path):
    reg = placed_sources(tmp_path, [np.full((8, 8), 10, np.uint8), np.full((8, 8), 20, np.uint8)],
                         [{'x': 0, 'y': 0}, {'x': 4, 'y': 0}], 'YX', {'x': 1, 'y': 1})

    data = np.asarray(msi_utils.get_sim_from_msim(overview(reg)).data).squeeze()

    assert data.shape == (8, 12)
    assert np.all(data[:, :4] == 10) and np.all(data[:, 4:] == 20)


def test_sections_spaced_other_than_one_keep_their_own_planes(tmp_path):
    # meatballs' sections are 0.05 apart, while a size-1 z reports a spacing of 1.0
    reg = placed_sources(tmp_path, [np.full((8, 8), value, np.uint8) for value in (10, 20, 30)],
                         [{'x': 0, 'y': 0, 'z': z} for z in (66.07, 66.12, 66.17)], 'YX', {'x': 1, 'y': 1})

    sim = msi_utils.get_sim_from_msim(overview(reg))

    assert [int(plane.max()) for plane in np.asarray(sim.data).squeeze()] == [10, 20, 30]
    assert float(np.diff(sim.coords['z'][:2])[0]) == pytest.approx(0.05)


def test_volumes_get_a_plane_per_z_each_from_the_nearest_z_of_every_volume_under_it(tmp_path):
    # values by z: 4 planes 1 apart, beside 2 planes 2 apart centred one plane lower
    first = np.arange(4, dtype=np.uint8)[:, None, None] * np.ones((4, 8, 8), np.uint8) + 10
    second = np.arange(2, dtype=np.uint8)[:, None, None] * np.ones((2, 8, 8), np.uint8) + 20
    reg = placed_sources(tmp_path, [first, second], [{'x': 0, 'y': 0, 'z': 0}, {'x': 8, 'y': 0, 'z': 1}], 'ZYX',
                         {'x': 1, 'y': 1, 'z': 1})
    reg.sources[1].pixel_sizes[0]['z'] = 2.0

    sim = msi_utils.get_sim_from_msim(overview(reg))
    data = np.asarray(sim.data).squeeze()

    # planes at the voxel centres, z 0..3
    assert float(np.diff(sim.coords['z'][:2])[0]) == 1 and data.shape == (4, 8, 16)
    assert [int(plane[0, 0]) for plane in data] == [10, 11, 12, 13]
    # the second's voxels centred at z 1 and 3 span z 0-2 and 2-4
    assert [int(plane[0, 8]) for plane in data] == [20, 20, 21, 21]


def test_2d_and_3d_sources_together_get_no_lazy_overview(tmp_path):
    for name, shape, axes in (('flat', (8, 8), 'YX'), ('volume', (2, 8, 8), 'ZYX')):
        tifffile.imwrite(tmp_path / f'source_{name}.tif', np.ones(shape, np.uint8), ome=True, metadata={'axes': axes})
    reg = MVSRegistration()
    reg.init(operation='register', input_path=sorted(str(path) for path in tmp_path.glob('source_*.tif')),
             output_path=tmp_path.as_posix() + '/output/')
    reg.init_data(source_metadata={'scale': {'x': 1, 'y': 1, 'z': 1}})

    assert overview(reg) is None


def test_one_worker_builds_each_plane_in_the_calling_thread_and_none_ahead(tmp_path):
    planes = overview(registration(tmp_path), workers=1).attrs['section_planes'][0]

    plane = planes.plane(0)

    assert planes._background is None and list(planes._planes) == [0]
    assert np.all(plane[0, :64, :64] == 10)
