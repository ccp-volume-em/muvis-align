import os.path
from ome_zarr.scale import Scaler

from src.muvis_align.constants import zarr_extension, tiff_extension
from src.muvis_align.image.ome_ngff_helper import save_ome_ngff
from src.muvis_align.image.ome_tiff_helper import save_ome_tiff
from src.muvis_align.image.ome_zarr_helper import save_ome_zarr
from src.muvis_align.image.util import *


def save_image(filename, data, output_format=zarr_extension, params={},
               transform_key=None, channels=None, translations0=None, verbose=False):
    if isinstance(data, list):
        data0 = data[0]
        positions = []
        rotations = []
        for index, data1 in enumerate(data):
            translation0 = translations0[index] if translations0 is not None else None
            position, rotation = get_data_mapping(data1, transform_key=transform_key, translation0=translation0)
            positions.append(position)
            rotations.append(rotation)
    else:
        data0 = data
        # metadata: only use coords of fused image
        translation0 = translations0[0] if translations0 is not None else None
        position, rotation = get_data_mapping(data, transform_key=transform_key, translation0=translation0)
        nplanes = data.sizes.get('z', 1) * data.sizes.get('c', 1) * data.sizes.get('t', 1)
        positions = [position] * nplanes
        rotations = [rotation] * nplanes

    if channels is None:
        channels = data.attrs.get('channels', [])

    tile_size = params.get('tile_size')
    if tile_size:
        if not isinstance(tile_size, (list, tuple)):
            tile_size = [tile_size] * 2
        if 'z' in data0.dims and len(tile_size) < 3:
            tile_size = list(tile_size) + [1]
        tile_size = tuple(reversed(tile_size))
        chunking = retuple(tile_size, data0.shape)
        if isinstance(data, list):
            data = [data1.chunk(chunks=chunking) for data1 in data]
        else:
            data = data.chunk(chunks=chunking)

    compression = params.get('compression')
    pyramid_downsample = params.get('pyramid_downsample', 2)
    npyramid_add = get_max_downsamples(data0.shape, params.get('npyramid_add', 0), pyramid_downsample)
    scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)
    ome_version = str(params.get('ome_version', '0.4'))

    dimension_order = ''.join(data0.dims)
    pixel_size = si_utils.get_spacing_from_sim(data0)

    if output_format:
        if 'zar' in output_format:
            if isinstance(data, list):
                save_ome_zarr(str(filename) + zarr_extension, data, dimension_order, pixel_size,
                              channels, positions, rotations, compression=compression, scaler=scaler,
                              ome_version=ome_version)
            else:
                save_ome_ngff(str(filename) + zarr_extension, data, pyramid_downsample=pyramid_downsample,
                              ome_version=ome_version, verbose=verbose)
        if 'tif' in output_format:
            save_ome_tiff(str(filename) + tiff_extension, data.data, dimension_order, pixel_size,
                          channels, positions, rotation, tile_size=tile_size, compression=compression, scaler=scaler)

        open(str(filename), 'w')
