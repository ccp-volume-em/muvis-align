from tifffile import TiffWriter, tifffile

from muvis_align.constants import default_chunk_size
from src.muvis_align.image.color_conversion import rgba_to_int
from src.muvis_align.util import *


def load_tiff(filename):
    return tifffile.imread(filename)


def save_tiff(filename, data, dimension_order=None, pixel_size=None, tile_size=(default_chunk_size, default_chunk_size),
              compression='LZW'):
    _, resolution, resolution_unit = create_tiff_metadata(pixel_size, dimension_order)
    tifffile.imwrite(filename, data, tile=tile_size, compression=compression,
                     resolution=resolution, resolutionunit=resolution_unit)


def save_ome_tiff(filename, data, dimension_order, pixel_size, channels=[], positions=[], rotation=None,
                  tile_size=None, compression=None, scaler=None):

    ome_metadata, resolution0, resolution_unit0 = create_tiff_metadata(pixel_size, dimension_order,
                                                                       channels, positions, rotation, is_ome=True)
    # maximum size (w/o compression)
    max_size = data.size * data.itemsize
    size = max_size
    if scaler is not None:
        npyramid_add = scaler.max_layer
        for i in range(npyramid_add):
            size //= (scaler.downscale ** 2)
            max_size += size
    else:
        npyramid_add = 0
    bigtiff = (max_size > 2 ** 32)

    tile_size = tile_size[-2:]  # assume order zyx (inversed xyz)
    shape_yx = [data.shape[dimension_order.index(dim)] for dim in 'yx']
    if np.any(np.array(tile_size) > np.array(shape_yx)):
        tile_size = None

    with TiffWriter(filename, bigtiff=bigtiff) as writer:
        for i in range(npyramid_add + 1):
            if i == 0:
                subifds = npyramid_add
                subfiletype = None
                metadata = ome_metadata
                resolution = resolution0[:2]
                resolutionunit = resolution_unit0
            else:
                subifds = None
                subfiletype = 1
                metadata = None
                resolution = None
                resolutionunit = None
                data = scaler.resize_image(data)
                data.rechunk()
            writer.write(data, subifds=subifds, subfiletype=subfiletype,
                         tile=tile_size, compression=compression,
                         resolution=resolution, resolutionunit=resolutionunit, metadata=metadata)


def create_tiff_metadata(pixel_size, dimension_order=None, channels=[], positions=[], rotation=None, is_ome=False):
    ome_metadata = None
    resolution = None
    resolution_unit = None

    if pixel_size is not None:
        resolution_unit = 'CENTIMETER'
        resolution = [1e4 / size for size in dict_to_xyz(pixel_size, 'xy')]

    if is_ome:
        ome_metadata = {'Creator': 'muvis-align'}
        if dimension_order is not None:
            #ome_metadata['DimensionOrder'] = dimension_order[::-1].upper()
            ome_metadata['axes'] = dimension_order.upper()
        ome_channels = []
        if pixel_size is not None:
            ome_metadata['PhysicalSizeX'] = pixel_size['x']
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeY'] = pixel_size['y']
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
            if 'z' in pixel_size:
                ome_metadata['PhysicalSizeZ'] = pixel_size['z']
                ome_metadata['PhysicalSizeZUnit'] = 'µm'
        if positions is not None and len(positions) > 0:
            plane_metadata = {}
            plane_metadata['PositionX'] = [position['x'] for position in positions]
            plane_metadata['PositionXUnit'] = ['µm' for _ in positions]
            plane_metadata['PositionY'] = [position['y'] for position in positions]
            plane_metadata['PositionYUnit'] = ['µm' for _ in positions]
            if 'z' in positions[0]:
                plane_metadata['PositionZ'] = [position['z'] for position in positions]
                plane_metadata['PositionZUnit'] = ['µm' for _ in positions]
            ome_metadata['Plane'] = plane_metadata
        if rotation is not None:
            ome_metadata['StructuredAnnotations'] = {'CommentAnnotation': {'Value': f'Angle: {rotation} degrees'}}
        for channeli, channel in enumerate(channels):
            ome_channel = {'Name': channel.get('label', str(channeli))}
            if 'color' in channel:
                ome_channel['Color'] = rgba_to_int(channel['color'])
            ome_channels.append(ome_channel)
        if ome_channels:
            ome_metadata['Channel'] = ome_channels
    return ome_metadata, resolution, resolution_unit
