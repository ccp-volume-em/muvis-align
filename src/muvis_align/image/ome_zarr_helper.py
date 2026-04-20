from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr, Omero, OmeroChannel, OmeroWindow
import ome_zarr.format
import zarr

from muvis_align.constants import default_ome_zarr_version, default_chunk_size
from src.muvis_align.image.util import create_compression_filter
from src.muvis_align.image.ome_zarr_util import create_transformation_metadata, get_channel_window


def save_ome_zarr(filename, datas, dim_order, pixel_size, channels, translations, rotations,
                  compression=None, scaler=None, ome_version=default_ome_zarr_version):
    # experimental
    is_series = isinstance(datas, list)
    if not is_series:
        datas = [datas]

    zarr_format, ome_zarr_format = get_ome_zarr_format(ome_version)

    root = zarr.create_group(store=filename, zarr_format=zarr_format, overwrite=True)
    multi_metadata = []
    omero_metadata = None
    for index, data in enumerate(datas):
        translation = translations[index] if translations is not None else None
        rotation = rotations[index] if rotations is not None else None
        if is_series:
            path = filename + '/' + str(index)
        else:
            path = filename
        metadata = save_ome_image(data, path=path, dim_order=dim_order, pixel_size=pixel_size, channels=channels,
                                  translation=translation, rotation=rotation,
                                  scaler=scaler, compression=compression, ome_version=ome_version)

        if is_series:
            multi_metadata.append(metadata)
            if metadata:
                omero_metadata = metadata.omero

    if is_series:
        root.attrs['multiscales'] = multi_metadata
        root.attrs['omero'] = omero_metadata


def save_ome_image(data, path, dim_order, pixel_size, channels, translation, rotation,
                   scaler=None, compression=None, ome_version=default_ome_zarr_version):

    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    if scaler is not None:
        npyramid_add = scaler.max_layer
        pyramid_downsample = scaler.downscale
    else:
        npyramid_add = 0
        pyramid_downsample = 1

    coordinate_transformations = []
    factor = 1
    for i in range(npyramid_add + 1):
        transform = create_transformation_metadata(dim_order, pixel_size, factor, translation, rotation)
        coordinate_transformations.append(transform)
        if pyramid_downsample:
            factor *= pyramid_downsample

    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation, axes_units=axes_units)

    scale_factors = [pyramid_downsample ** (scale + 1) for scale in range(npyramid_add)]
    multiscales = to_multiscales(image, scale_factors=scale_factors, chunks=default_chunk_size)

    if channels:
        omero = Omero(channels=[OmeroChannel(label=channel,
                                             color=channel.get('color'),
                                             window=OmeroWindow(**get_channel_window(multiscales.images[-1], dim_order, index)))
                                for index, channel in enumerate(channels)])

        multiscales.metadata.omero = omero

    chunks_per_shard = None

    to_ngff_zarr(path, multiscales, chunks_per_shard=chunks_per_shard, version=ome_version)

    return multiscales.metadata


def get_ome_zarr_format(ome_version):
    if str(ome_version) == '0.4':
        ome_zarr_format = ome_zarr.format.FormatV04()
    elif str(ome_version) == '0.5':
        ome_zarr_format = ome_zarr.format.FormatV05()   # future support anticipated
    else:
        ome_zarr_format = ome_zarr.format.CurrentFormat()
    zarr_format = 3 if float(ome_zarr_format.version) >= 0.5 else 2
    return zarr_format, ome_zarr_format
