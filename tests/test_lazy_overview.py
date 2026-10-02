from types import SimpleNamespace

import numpy as np
import tifffile
from multiview_stitcher import msi_utils

from muvis_align.image.lazy_overview import lazy_section_overview, _coarsest_level_within, MsimLevels
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


def test_a_section_is_built_only_when_it_is_asked_for_and_once(tmp_path):
    planes = overview(registration(tmp_path)).attrs['section_planes']

    assert list(planes._planes) == []
    first = planes.plane(1, prefetch=False)
    assert list(planes._planes) == [1]
    assert planes.plane(1, prefetch=False) is first


def test_the_sections_around_a_viewed_one_are_built_in_the_background(tmp_path):
    planes = overview(registration(tmp_path)).attrs['section_planes']
    planes.prefetch_radius = 1

    planes.plane(0)
    planes._background.submit(lambda: None).result()

    assert sorted(planes._planes) == [0, 1]


def test_kept_sections_stay_within_their_budget_the_least_recently_viewed_dropped(tmp_path):
    planes = overview(registration(tmp_path)).attrs['section_planes']
    planes.max_planes = 1

    planes.plane(0, prefetch=False)
    planes.plane(1, prefetch=False)

    assert list(planes._planes) == [1]


def test_rotated_or_multichannel_sources_get_no_lazy_overview(tmp_path):
    for name in ('rotated', 'channels'):
        (tmp_path / name).mkdir()

    assert overview(registration(tmp_path / 'rotated', rotation=30)) is None
    assert overview(registration(tmp_path / 'channels', channels=2)) is None


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

    assert _coarsest_level_within(source, {'y': 3, 'x': 3}) == 1
    assert _coarsest_level_within(source, {'y': 4, 'x': 4}) == 2
    assert _coarsest_level_within(source, {'y': 0.5, 'x': 0.5}) == 0


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


def test_a_pre_processed_msim_is_read_at_its_own_levels(tmp_path):
    reg = registration(tmp_path)
    msims, _, _ = reg.preprocess(reg.ensure_msims(), normalisation='individual')
    sim = msi_utils.get_sim_from_msim(msims[0])

    levels = MsimLevels(msims[0])

    assert levels.pixel_sizes == [{'y': 1.0, 'x': 1.0}] and levels.dtype == sim.dtype
    assert np.array_equal(np.asarray(levels.level_data(0)), np.asarray(sim.data))
