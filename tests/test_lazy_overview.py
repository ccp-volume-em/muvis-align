import numpy as np
import tifffile
from multiview_stitcher import msi_utils

from muvis_align.image.lazy_overview import lazy_section_overview
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


def overview(reg):
    return lazy_section_overview(reg.sources, reg.positions, reg._msim_transforms, reg._msim_output_order,
                                 reg.source_transform_key, z_scale=reg._msim_z_scale)


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
    msim = overview(registration(tmp_path))
    planes = msim.attrs['section_planes']
    data = msi_utils.get_sim_from_msim(msim).data.squeeze()

    assert planes._planes == {}
    first = np.asarray(data[1])
    assert list(planes._planes) == [1]
    assert np.asarray(data[1]) is not first and np.array_equal(np.asarray(data[1]), first)
    assert list(planes._planes) == [1]


def test_rotated_or_multichannel_sources_get_no_lazy_overview(tmp_path):
    for name in ('rotated', 'channels'):
        (tmp_path / name).mkdir()

    assert overview(registration(tmp_path / 'rotated', rotation=30)) is None
    assert overview(registration(tmp_path / 'channels', channels=2)) is None
