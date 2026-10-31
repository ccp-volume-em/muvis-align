from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr, NgffMultiscales, NgffScene, \
    Omero, OmeroChannel, OmeroWindow
from ngff_zarr.v06.zarr_metadata import Metadata, Affine, Translation, CoordinateSystem, CoordinateSystemIdentifier
import json
import numpy as np
import os
import ome_zarr.format
import zarr

from muvis_align.image.color_conversion import rgba_to_hexrgb
from muvis_align.util import create_chunk_dict
from muvis_align.constants import default_ome_zarr_version, default_chunk_size
from muvis_align.image.util import calc_pyramid_level_factors, create_compression_filter
from muvis_align.image.ome_zarr_util import get_channel_window


def save_ome_zarr(filename, datas, dim_order, pixel_size, channels, translations, rotations, pyramid_downsample=None,
                  compression=None, ome_version=default_ome_zarr_version):
    # experimental: if list of images, store as series of images in ome-zarr format
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
                                  translation=translation, rotation=rotation, pyramid_downsample=pyramid_downsample,
                                  compression=compression, ome_version=ome_version)

        if is_series:
            multi_metadata.append(metadata)
            if metadata:
                omero_metadata = metadata.omero

    if is_series:
        root.attrs['multiscales'] = multi_metadata
        root.attrs['omero'] = omero_metadata


def save_ome_image(data, path, dim_order, pixel_size, channels, translation, rotation,
                   pyramid_downsample=None, compression=None, ome_version=default_ome_zarr_version):

    storage_options = get_storage_options(compression)

    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation, axes_units=axes_units)
    multiscales = to_multiscales(image, scale_factors=pyramid_downsample, chunks=create_chunk_dict(default_chunk_size, dim_order))

    if channels:
        multiscales.metadata.omero = make_omero(channels, multiscales.images[-1].data, dim_order)

    chunks_per_shard = None

    to_ngff_zarr(path, multiscales, chunks_per_shard=chunks_per_shard, version=ome_version, **storage_options)

    return multiscales.metadata


def get_padding_scale_factors(shape, dim_order, **kwargs):
    """Cumulative per-level downsample factors extending a pyramid whose coarsest level is `shape` down to
    overview size, by the rule readers use to synthesize levels: a chunked store cannot synthesize them cheaply.

    In ngff_zarr's form: one dict per extra level, its factor relative to the input.
    """
    return calc_pyramid_level_factors(
        {dim: size for dim, size in zip(dim_order, shape) if dim in 'xyz'}, **kwargs)


def get_storage_options(compression):
    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters
    return storage_options


def save_ome_multiscale_levels(path, levels, dim_order, channels, translation,
                               min_length=128, compression=None, ome_version=default_ome_zarr_version):
    """Write pre-built pyramid levels exactly as given, without resampling (see make_ome_multiscale_levels())."""
    multiscales = make_ome_multiscale_levels(levels, dim_order, channels, translation, min_length=min_length)
    to_ngff_zarr(path, multiscales, version=ome_version, **get_storage_options(compression))
    return multiscales.metadata


def make_ome_multiscale_levels(levels, dim_order, channels, translation, min_length=128, name=None):
    """Pre-built pyramid levels exactly as given, without resampling, padded with resampled coarser levels,
    halving until the largest spatial extent is at most min_length.

    The padding factors are computed here: to_multiscales(scale_factors=<int>) never writes a level smaller
    than one chunk, leaving no level coarse enough for a zoomed-out preview.

    levels: list of (data, pixel_size) pairs, finest first - data can be numpy or dask.
    """
    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    chunks = create_chunk_dict(default_chunk_size, dim_order)
    images = []
    datasets = []
    coordinate_systems = None
    for data, pixel_size in levels:
        # scale_factors=[] only wraps the level in OME metadata, without resampling
        ngff_image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation,
                                   axes_units=axes_units)
        level_multiscales = to_multiscales(ngff_image, scale_factors=[], chunks=chunks)
        images.append(level_multiscales.images[0])
        datasets.append(level_multiscales.metadata.datasets[0])
        if coordinate_systems is None:
            coordinate_systems = level_multiscales.metadata.coordinateSystems

    smallest_data, smallest_pixel_size = levels[-1]
    smallest_ngff_image = to_ngff_image(smallest_data, dims=dim_order, scale=smallest_pixel_size,
                                        translation=translation, axes_units=axes_units)
    extra_scale_factors = get_padding_scale_factors(smallest_data.shape, dim_order,
                                                    min_size=min_length)
    if extra_scale_factors:
        extra_multiscales = to_multiscales(smallest_ngff_image, scale_factors=extra_scale_factors,
                                           chunks=chunks)
        images += extra_multiscales.images[1:]
        datasets += extra_multiscales.metadata.datasets[1:]

    # each level was built separately as 'scale0/image', so paths would otherwise collide
    for index, dataset in enumerate(datasets):
        dataset.path = f'scale{index}/image'
        # 0.6 names the dataset's array as its transforms' input
        for transform in dataset.coordinateTransformations or []:
            if getattr(transform, 'input', None) is not None and transform.input.path is not None:
                transform.input.path = dataset.path
                transform.name = f'scale{index}_to_{transform.output.name}'

    metadata = Metadata(coordinateSystems=coordinate_systems, datasets=datasets, name=name)
    if channels:
        metadata.omero = make_omero(channels, images[-1].data, dim_order)

    return NgffMultiscales(images=images, metadata=metadata)


def make_omero(channels, overview_data, dim_order):
    """OMERO channel metadata, each window from the overview level. A channel without a colour is white, as napari
    shows it; `or`, not a get() default, as a blank label or colour is present but empty."""
    return Omero(channels=[OmeroChannel(label=channel.get('label') or f'Channel {index}',
                                        color=rgba_to_hexrgb(channel.get('color') or (1, 1, 1)),
                                        window=OmeroWindow(**get_channel_window(overview_data, dim_order, index)))
                           for index, channel in enumerate(channels)])


def save_ome_zarr_scene(path, sources, coordinate_system_names, channels, min_length=128, compression=None):
    """Separate images in one OME-Zarr 0.6 scene, each at its own native levels (make_ome_multiscale_levels()),
    placed in every named coordinate system by its affine, written as a translation when it is only one. The image
    itself has no translation: its origin is part of each placement, as napari-ome-zarr only reads a level's scale.

    sources: {image path: (levels, dim_order, translation, spatial_dims, {coordinate system name: affine})}, each
    affine a homogeneous matrix over spatial_dims of the image's physical space, as multiview-stitcher keeps it.
    coordinate_system_names: the first is the default view; napari-ome-zarr picks the alphabetically first instead.
    """
    images = {}
    transforms = []
    for image_path, (levels, dim_order, translation, spatial_dims, affines) in sources.items():
        multiscales = make_ome_multiscale_levels(levels, dim_order, channels, {dim: 0.0 for dim in translation},
                                                 min_length=min_length, name=image_path)
        images[image_path] = multiscales
        image_system = CoordinateSystemIdentifier(path=image_path, name=multiscales.metadata.coordinateSystems[0].name)
        origin = np.eye(len(spatial_dims) + 1)
        origin[:-1, -1] = [translation.get(dim, 0) for dim in spatial_dims]
        for name in coordinate_system_names:
            matrix = full_dims_affine(np.asarray(affines[name]) @ origin, dim_order, spatial_dims)
            output_system = CoordinateSystemIdentifier(name=name)
            # a translation is read by more viewers than an affine; only an exact one, so nothing is lost
            if np.array_equal(matrix[:, :-1], np.eye(len(dim_order))):
                transforms.append(Translation(translation=matrix[:, -1].tolist(), input=image_system,
                                              output=output_system))
            else:
                transforms.append(Affine(affine=matrix.tolist(), input=image_system, output=output_system))
    axes = next(iter(images.values())).metadata.coordinateSystems[0].axes
    coordinate_systems = [CoordinateSystem(name=name, axes=axes) for name in coordinate_system_names]
    scene = NgffScene(images=images, coordinateTransformations=transforms, coordinateSystems=coordinate_systems)
    scene.to_ome_zarr(path, version='0.6', **get_storage_options(compression))


def read_ome_zarr_scene(path):
    """The scene metadata of an OME-Zarr 0.6 scene group, or None for any other store."""
    metadata_path = os.path.join(path, 'zarr.json')
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, encoding='utf8') as file:
        return json.load(file).get('attributes', {}).get('ome', {}).get('scene')


def spatial_affine(transform, axes, spatial_dims):
    """A scene's translation or affine transform (as written by save_ome_zarr_scene()) over axes, as the
    homogeneous affine over spatial_dims."""
    if transform['type'] == 'translation':
        matrix = np.hstack([np.eye(len(axes)), np.array(transform['translation'], dtype=float)[:, None]])
    else:
        matrix = np.asarray(transform['affine'])
    indices = [axes.index(dim) for dim in spatial_dims]
    affine = np.eye(len(indices) + 1)
    affine[:-1, :-1] = matrix[np.ix_(indices, indices)]
    affine[:-1, -1] = matrix[indices, -1]
    return affine


def full_dims_affine(affine, dim_order, spatial_dims):
    """A spatial homogeneous affine as OME-Zarr's (N, N+1) matrix over all of dim_order, identity on t/c."""
    affine = np.asarray(affine)
    matrix = np.eye(len(dim_order), len(dim_order) + 1)
    for row, output_dim in enumerate(spatial_dims):
        output_index = dim_order.index(output_dim)
        for column, input_dim in enumerate(spatial_dims):
            matrix[output_index, dim_order.index(input_dim)] = affine[row, column]
        matrix[output_index, -1] = affine[row, -1]
    return matrix


def get_ome_zarr_format(ome_version):
    if str(ome_version) == '0.4':
        ome_zarr_format = ome_zarr.format.FormatV04()
    elif str(ome_version) == '0.5':
        ome_zarr_format = ome_zarr.format.FormatV05()
    elif ome_version is None:
        ome_zarr_format = ome_zarr.format.CurrentFormat()
    else:
        # ome-zarr-py writes 0.4/0.5 only: anything else would be written as its current format, labelled as asked
        raise ValueError(f'OME-Zarr {ome_version} cannot be written here: 0.4 or 0.5')
    zarr_format = 3 if float(ome_zarr_format.version) >= 0.5 else 2
    return zarr_format, ome_zarr_format
