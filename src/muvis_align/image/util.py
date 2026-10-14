import functools
import itertools
import logging
from math import ceil
import time

import cv2 as cv
import dask
import dask.array as da
import numpy as np
import networkx as nx
from multiview_stitcher import msi_utils, param_utils, fusion, mv_graph
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.registration import _get_overlap_bboxes, sims_to_intrinsic_coord_system, \
    get_affine_from_intrinsic_affine
from scipy.spatial import ConvexHull
from skimage.filters import gaussian
from skimage.feature import blob_log
from skimage.transform import resize
import xarray as xr
from xarray import DataTree

try:
    import matplotlib as mpl
    mpl.use('Agg')
    #mpl.rcParams['backend'] = 'svg'
    mpl.rcParams['figure.dpi'] = 300
    import matplotlib.pyplot as plt
except Exception as e:
    print(f'matplotlib import error:\n{e}')

from muvis_align.constants import (default_chunk_size, default_contrast_limits_max_tasks,
                                   default_export_chunk_size, default_export_fusion_chunk_bytes,
                                   default_fusion_chunk_bytes, default_preview_max_bytes, fusion_stack_arrays)
from muvis_align.util import *


def show_image(image, title='', cmap=None):
    if cmap is None:
        nchannels = image.shape[2] if len(image.shape) > 2 else 1
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.show()


def plt_close():
    plt.close()



def _adapt_transform_to_image_dims(sim, transform, transform_key):
    """The sim's `transform_key` affine reduced to the image's spatial dims, e.g. 3x3 out of a 4x4 for a 2D image."""
    sim_spatial_dims = si_utils.get_spatial_dims_from_sim(sim)
    transform = sim.attrs['transforms'][transform_key]
    
    transform_spatial_dims = list(transform.coords['x_in'].values)
    
    # '1' is the homogeneous coordinate, not a dim
    transform_spatial_dim_count = len([d for d in transform_spatial_dims if d != '1'])
    sim_spatial_dim_count = len(sim_spatial_dims)
    
    if transform_spatial_dim_count == sim_spatial_dim_count:
        return transform
    
    relevant_dims = [d for d in sim_spatial_dims if d in transform_spatial_dims]
    
    relevant_dim_names = relevant_dims + ['1']
    adapted = transform.sel(x_in=relevant_dim_names, x_out=relevant_dim_names)
    
    return adapted


def color_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 1:
        return cv.cvtColor(np.array(image), cv.COLOR_GRAY2RGB)
    else:
        return image


def int2float_image(image):
    source_dtype = image.dtype
    if not source_dtype.kind == 'f':
        maxval = 2 ** (8 * source_dtype.itemsize) - 1
        return image / np.float32(maxval)
    else:
        return image


def float2int_image(image, target_dtype=np.dtype(np.uint8)):
    source_dtype = image.dtype
    if source_dtype.kind not in ('i', 'u') and not target_dtype.kind == 'f':
        maxval = 2 ** (8 * target_dtype.itemsize) - 1
        return (image * maxval).astype(target_dtype)
    else:
        return image


def uint8_image(image):
    source_dtype = image.dtype
    if source_dtype.kind == 'f':
        image = image * 255
    elif source_dtype.itemsize != 1:
        factor = 2 ** (8 * (source_dtype.itemsize - 1))
        image = image // factor
    return image.astype(np.uint8)


def ensure_unsigned_type(dtype: np.dtype) -> np.dtype:
    new_dtype = dtype
    if dtype.kind == 'i' or dtype.byteorder == '>' or dtype.byteorder == '<':
        new_dtype = np.dtype(f'u{dtype.itemsize}')
    return new_dtype


def ensure_unsigned_image(image: np.ndarray) -> np.ndarray:
    source_dtype = image.dtype
    dtype = ensure_unsigned_type(source_dtype)
    if dtype != source_dtype:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        new_image = image.astype(dtype) + offset
    else:
        new_image = image
    return new_image


def convert_image_sign_type(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    source_dtype = image.dtype
    if source_dtype.kind == target_dtype.kind:
        new_image = image
    elif source_dtype.kind == 'i':
        new_image = ensure_unsigned_image(image)
    else:
        # conversion without overhead
        offset = 2 ** (8 * target_dtype.itemsize - 1)
        new_image = (image - offset).astype(target_dtype)
    return new_image


def redimension_data(data, old_order, new_order, **indices):
    # indices: the index kept of each removed dim, e.g. t=0, z=0 (default 0)
    if new_order == old_order:
        return data

    new_data = data
    order = old_order
    for o in old_order:
        if o not in new_order:
            index = order.index(o)
            dim_value = indices.get(o, 0)
            new_data = np.take(new_data, indices=dim_value, axis=index)
            order = order[:index] + order[index + 1:]
    for o in new_order:
        if o not in order:
            new_data = np.expand_dims(new_data, 0)
            order = o + order
    old_indices = [order.index(o) for o in new_order]
    new_indices = list(range(len(new_order)))
    new_data = np.moveaxis(new_data, old_indices, new_indices)
    return new_data


def redimension_sim_data(image, old_order, new_order, **indices):
    # lazy xarray equivalent of redimension_data; dims are checked against image.dims, not the order strings,
    # as the image can carry dims neither lists that the final transpose must still include
    if new_order == old_order and set(new_order) == set(image.dims):
        return image

    for o in old_order:
        if o not in new_order and o in image.dims:
            dim_value = indices.get(o, 0)
            image = image.isel({o: dim_value})
    for o in new_order:
        if o not in image.dims:
            image = image.expand_dims(o, axis=0)
    # dims new_order lacks (e.g. a forced 't') are kept, ahead of the requested order
    extra_dims = [dim for dim in image.dims if dim not in new_order]
    return image.transpose(*extra_dims, *list(new_order))


def ensure_spatial_image_dims(image, c_coords=None, t_coords=None):
    # si_utils.get_sim_from_array always adds size-1 'c' and 't' dims, which downstream code relies on:
    # match it, so a sim not built through it is indistinguishable
    if c_coords is None and 'c' in image.coords:
        c_coords = np.atleast_1d(image.coords['c'].values)
    if t_coords is None and 't' in image.coords:
        t_coords = np.atleast_1d(image.coords['t'].values)
    for dim, coords in (('c', c_coords), ('t', t_coords)):
        if dim not in image.dims:
            # a stale scalar (non-dimension) coord of the same name can't survive expand_dims as-is
            image = image.reset_coords(dim, drop=True) if dim in image.coords else image
            image = image.expand_dims(dim, axis=0)
            if coords is not None:
                image = image.assign_coords({dim: list(coords)})
    new_dims = [dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim in image.dims]
    if list(new_dims) != list(image.dims):
        image = image.transpose(*new_dims)
    return image


def rechunk_if_monolithic(image, chunk_size):
    # a single-chunk source (e.g. a TIFF) is split for memory and parallelism downstream
    if chunk_size and len(image.chunksizes.get('x', ())) == 1 and len(image.chunksizes.get('y', ())) == 1:
        image = image.chunk(xyz_to_dict([chunk_size] * 2 if isinstance(chunk_size, int) else chunk_size))
    return image


def calc_pyramid_level_factors(sizes, pyramid_downsample=2, min_size=default_chunk_size):
    """Cumulative per-dim downsample factor for each level needed beyond the one of `sizes` ({dim: size}),
    halving while the largest extent exceeds min_size.

    Shared by build_missing_pyramid_levels() and the export side (ome_zarr_helper.get_padding_scale_factors()), so
    a written pyramid and a synthesized one agree. Sizes round up as they shrink, matching strided slicing
    (data[::2] of 4097 rows is 2049).
    """
    factors = []
    if not sizes:
        return factors
    current = dict(sizes)
    cumulative = {dim: 1 for dim in sizes}
    while max(current.values()) > min_size:
        step = {dim: (pyramid_downsample if size >= pyramid_downsample else 1)
                for dim, size in current.items()}
        if all(value == 1 for value in step.values()):
            break
        current = {dim: ceil(size / step[dim]) for dim, size in current.items()}
        cumulative = {dim: factor * step[dim] for dim, factor in cumulative.items()}
        factors.append(dict(cumulative))
    return factors


def build_missing_pyramid_levels(data, dimension_order, pixel_size, pyramid_downsample=2,
                                 min_size=default_chunk_size):
    """Synthesize coarser levels for a source with only one real resolution, so napari has something to draw
    zoomed out instead of computing the whole finest level.

    Strided (nearest-neighbour) rather than mean-downsampled: the source is usually one already-decoded chunk, and
    a mean would add a reduction pass per level. Levels come from calc_pyramid_level_factors(). Returns
    ([data] + extra levels, [pixel_size] + their pixel sizes).
    """
    spatial_axes = {dim: axis for axis, dim in enumerate(dimension_order) if dim in 'xyz'}
    datas = [data]
    pixel_sizes = [pixel_size]
    if not spatial_axes:
        return datas, pixel_sizes
    sizes = {dim: data.shape[axis] for dim, axis in spatial_axes.items()}
    for factors in calc_pyramid_level_factors(sizes, pyramid_downsample, min_size):
        # factors are cumulative against the finest level, so slice `data` itself, not the previous level
        slicing = tuple(slice(None, None, factors.get(dim, 1))
                        for dim in dimension_order)
        coarse = data[slicing]
        if coarse.shape == datas[-1].shape:
            break
        datas.append(coarse)
        pixel_sizes.append({dim: pixel_size[dim] * data.shape[spatial_axes[dim]] / coarse.shape[spatial_axes[dim]]
                            for dim in pixel_size if dim in spatial_axes})
    return datas, pixel_sizes


def source_levels(source, from_level=0, ends_only=False):
    """The native levels a msim built from `source` carries: every one from `from_level` on, or with
    `ends_only` just that and the coarsest - all that registration and the view's preview cap read."""
    count = len(getattr(source, 'pixel_sizes', None) or getattr(source, 'data', None) or [None])
    levels = list(range(from_level, count))
    return levels[:1] + levels[-1:] if ends_only and len(levels) > 2 else levels


def build_source_redimensioned_msim(source, output_order, chunk_size=default_chunk_size,
                                    from_level=0, ends_only=False):
    """A new msim of `source`'s levels lazily redimensioned into `output_order`, with the 'c'/'t' dims every sim
    needs and any level monolithic in x/y rechunked.

    Independent of per-run geometry, so source.get_msim() caches it. `from_level` drops the finest levels
    (renumbered from scale0) for consumers that only work at a coarser scale. Built from the source's own arrays
    where it has them - see _source_level_images().
    """
    # None, not []: a source reporting no channels at all still gets the forced 'c' dim, and an
    # empty label list cannot be assigned to it
    c_coords = [channel.get('label', '') for channel in source.get_channels()] or None
    datasets = {}
    for level, image in enumerate(_source_level_images(source, output_order, source_levels(source, from_level, ends_only))):
        image = redimension_sim_data(image, source.dimension_order, output_order)
        image = ensure_spatial_image_dims(image, c_coords=c_coords)
        # every 'c', not just one ensure_spatial_image_dims added itself: a channel selected by
        # name (registration's 'channel' param) is found via .sel(c=...), so it needs the label
        if c_coords and 'c' in image.dims and len(c_coords) == image.sizes['c']:
            image = image.assign_coords(c=list(c_coords))
        image = rechunk_if_monolithic(image, chunk_size)
        datasets[f'scale{level}'] = xr.Dataset({'image': image})
    return DataTree.from_dict(datasets)


def coarsest_level_within(pixel_sizes, spacing, dims='yx', tolerance=1e-6):
    """The index of the coarsest of `pixel_sizes` (one a level, finest first) no coarser than `spacing` in `dims`,
    else 0; the tolerance keeps a level computed as exactly the spacing from losing to rounding."""
    best = 0
    for level, pixel_size in enumerate(pixel_sizes):
        if all(pixel_size.get(dim, 0) <= spacing[dim] * (1 + tolerance) for dim in dims if dim in spacing):
            best = level
    return best


def msim_level_spacings(msim):
    """Each level's pixel size, read off its coordinates without building a sim."""
    return [si_utils.get_spacing_from_sim(msim[scale_key].ds['image'])
            for scale_key in msi_utils.get_sorted_scale_keys(msim)]


def complete_translation(translation):
    """A copy of `translation` with x and y at 0 where missing, if any dim is given: si_utils takes all or none."""
    translation = dict(translation or {})
    if translation:
        translation.setdefault('x', 0)
        translation.setdefault('y', 0)
    return translation


def _source_level_images(source, output_order, levels):
    """One image per pyramid level, taken straight off the source's own arrays.

    Avoids building source.msim, whose coordinates build_source_msim() overwrites anyway. A source with no raw
    arrays (an OME-Zarr read natively) still goes through its msim.
    """
    data = getattr(source, 'data', None)
    if not data:
        scale_keys = msi_utils.get_sorted_scale_keys(source.msim)
        return [source.msim[scale_keys[level]].ds['image'] for level in levels]

    translation = complete_translation(source.position)
    images = []
    for level in levels:
        image = xr.DataArray(data[level], dims=list(source.dimension_order))
        # a spatial dim output_order drops keeps the coord get_sim_from_array gave it, as the
        # scalar its isel leaves behind - carry it over rather than silently dropping it
        pixel_size = source.pixel_sizes[level]
        dropped = {dim: translation.get(dim, 0) + np.arange(image.sizes[dim]) * pixel_size.get(dim, 1)
                   for dim in source.dimension_order
                   if dim in 'zyx' and dim not in output_order and dim in image.dims}
        images.append(image.assign_coords(dropped) if dropped else image)
    return images


def build_source_msim(source, output_order, translation, transform, transform_key, z_scale=None,
                      from_level=0, ends_only=False):
    """A new msim of `source`'s real pyramid levels, redimensioned to `output_order` and placed with
    `translation` (intrinsic, per level) and `transform` (extrinsic, every level), built from the cached
    source.get_msim(output_order), which is left unchanged.

    `from_level` and `ends_only` select levels as in source_levels().
    """
    if transform is None:
        spatial_dims = [dim for dim in output_order if dim in 'xyz']
        xaffine = param_utils.identity_transform(len(spatial_dims))
    else:
        xaffine = param_utils.affine_to_xaffine(transform)

    translation_arg = complete_translation(translation)

    redimensioned_msim = source.get_msim(output_order, from_level=from_level, ends_only=ends_only)
    scale_keys = msi_utils.get_sorted_scale_keys(redimensioned_msim)
    datasets = {}
    for level, scale_key in zip(source_levels(source, from_level, ends_only), scale_keys):
        image = redimensioned_msim[scale_key].ds['image']

        # the source's own per-level metadata is still indexed by native level, not by where
        # this (possibly truncated) pyramid starts
        pixel_size = dict(source.pixel_sizes[level])
        if 'z' in output_order and 'z' not in pixel_size:
            pixel_size['z'] = abs(z_scale) if z_scale else 1
        spatial_dims = si_utils.get_spatial_dims_from_sim(image)
        new_coords = {dim: translation_arg.get(dim, 0) + np.arange(image.sizes[dim]) * pixel_size.get(dim, 1)
                      for dim in spatial_dims}
        image = image.assign_coords(new_coords)

        ds = xr.Dataset({'image': image})
        ds[transform_key] = xaffine
        datasets[scale_key] = ds

    return DataTree.from_dict(datasets)


def build_source_stack_props(source, output_order, translation, transform, transform_key,
                             z_scale=None, level=0, promote_z=False):
    """The geometry the shapes preview needs - shape, spacing, origin and transform - from a source's metadata
    alone, allocating nothing and reading no pixels.

    A dim of output_order the source lacks becomes size 1; x/y default to origin 0 once any translation is given;
    promote_z adds a size-1 'z' at the source's z position, widening the transform to 3D.
    """
    spatial_order = [dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim in 'zyx'
                     and (dim in output_order or (promote_z and dim == 'z'))]
    sizes = dict(zip(source.dimension_order, source.get_shape(level)))
    shape = {dim: int(sizes.get(dim, 1)) for dim in spatial_order}

    pixel_size = dict(source.pixel_sizes[level])
    if 'z' in spatial_order and 'z' not in pixel_size:
        pixel_size['z'] = abs(z_scale) if z_scale else 1
    # si_utils derives spacing from coordinate differences, so reports 1.0 for a size-1 dim whatever the pixel
    # size: match that
    spacing = {dim: float(pixel_size.get(dim, 1)) if shape[dim] > 1 else 1.0
               for dim in spatial_order}

    translation_arg = complete_translation(translation)
    origin = {dim: float(translation_arg.get(dim, 0)) for dim in spatial_order}

    stack_props = {'shape': shape, 'spacing': spacing, 'origin': origin}
    if transform_key is not None:
        if transform is None:
            affine = np.eye(len([dim for dim in output_order if dim in 'xyz']) + 1)
        else:
            affine = np.asarray(transform)
        if promote_z and affine.shape[-1] < 4:
            # a 2D affine's y, x, 1 rows land on the last three of the 3D identity
            widened = np.eye(4)
            widened[-affine.shape[-1]:, -affine.shape[-1]:] = affine
            affine = widened
        stack_props['transform'] = _xaffine_template(affine.shape[-1]).copy(deep=False, data=affine)
    return stack_props


@functools.lru_cache(maxsize=None)
def _xaffine_template(size):
    # a shallow copy is far cheaper than building a DataArray's coords for every source
    return param_utils.identity_transform(size - 1)


@functools.lru_cache(maxsize=None)
def _array_template(shape, affine, with_t):
    if not affine:
        return xr.DataArray(np.zeros(shape))
    template = param_utils.identity_transform(shape[0] - 1)
    return template.expand_dims({'t': [0]}) if with_t else template


def xaffine_from_matrix(matrix, with_t=False):
    """param_utils.affine_to_xaffine(matrix), with a 't' of one time point if asked: a shallow copy of a cached one,
    as building each from scratch costs most of loading a saved registration's many."""
    matrix = np.asarray(matrix, dtype=float)
    return _array_template(matrix.shape, True, with_t).copy(deep=False, data=matrix[None] if with_t else matrix)


def data_array_from_values(values):
    """xr.DataArray(values) as a shallow copy of a cached one of its shape (see xaffine_from_matrix)."""
    values = np.asarray(values, dtype=float)
    return _array_template(values.shape, False, False).copy(deep=False, data=values)


def build_source_shape_sim(source, output_order, translation, transform, transform_key, z_scale=None, level=0,
                           promote_z=False):
    """A single-level placeholder sim with build_source_stack_props()' geometry, for multiview_stitcher's exact
    overlap test, which takes sims. Reads no pixels, so never opens the source's file.

    promote_z=True mirrors make_msims_3d() where output_order has no 'z' but sources sit at different z heights:
    a source's z position becomes a size-1 'z' dim instead of being dropped.
    """
    stack_props = build_source_stack_props(source, output_order, translation, transform,
                                           transform_key, z_scale=z_scale, level=level,
                                           promote_z=promote_z)
    c_coords = [channel.get('label', '') for channel in source.get_channels()]
    return sim_from_stack_props(stack_props, source.dtype, transform_key, c_coords=c_coords)


def sim_from_stack_props(stack_props, dtype, transform_key, c_coords=None):
    """A sim with exactly the geometry in `stack_props` and no image data, for multiview_stitcher entry points that
    only read geometry off sims.

    Backed by a lazy one-chunk dask placeholder, as xarray would materialise a zero-strided numpy view.
    name=False skips dask's tokenization, pointless for a placeholder.
    """
    spatial_dims = list(stack_props['shape'])
    shape = tuple(stack_props['shape'][dim] for dim in spatial_dims)
    data = da.zeros(shape, dtype=dtype, chunks=shape, name=False)
    image = si_utils.get_sim_from_array(
        data, dims=spatial_dims,
        scale=stack_props['spacing'], translation=stack_props['origin'],
        transform_key=transform_key, c_coords=c_coords)
    if 'transform' in stack_props:
        si_utils.set_sim_affine(image, stack_props['transform'], transform_key)
    return image


def map_msim_levels(msim, level_func):
    """Apply level_func(sim, scale_key) -> new_sim to every scale level of `msim`, reassembled into a new msim."""
    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    level_sims = [level_func(msi_utils.get_sim_from_msim(msim, scale=scale_key), scale_key)
                  for scale_key in scale_keys]
    return msi_utils.get_msim_from_sims(level_sims)


def get_msim_level_data(msim):
    """Raw dask array per pyramid level, straight off each scale node, as napari's multiscale add_image takes them."""
    return [msim[scale_key].ds['image'].data for scale_key in msi_utils.get_sorted_scale_keys(msim)]


def get_chunk_sizes(dtype, spatial_dims, num_sources=1, num_z_positions=1,
                    xy_chunk_size=default_chunk_size, target_bytes=64 * 1024 ** 2,
                    fusion_target_bytes=None, min_xy_chunk_size=64):
    """Per-spatial-dim chunk sizes for a fused preview, budgeted on the memory fusing one chunk takes, which is set
    by how many sources land in it rather than its own size.

    Fusion stacks every overlapping source as a full-chunk float32 array (fusion_stack_arrays of them). Coarse
    levels are the worst case: fuse() uses one chunk size for every level, so a level about xy_chunk_size across
    puts every source in one chunk. With sources spread over z, a chunk spanning Nz planes costs Nz^2 as much.

    fusion_target_bytes budgets one chunk (default default_fusion_chunk_bytes, from this job's allocation, as one
    chunk is fused per worker). x/y stay as large as the budget allows, in multiples of min_xy_chunk_size, since
    napari shows them in full; z takes the rest, 1 wherever sources are spread over z.
    """
    # sources one output plane can draw from: those at one z position, or all of them at a single height
    sources_per_plane = max(1, round(num_sources / max(1, num_z_positions)))
    # budget expressed in float32 voxels, across the stacked arrays fusion holds at once
    if fusion_target_bytes is None:
        fusion_target_bytes = default_fusion_chunk_bytes
    voxel_budget = max(1.0, fusion_target_bytes / (4 * fusion_stack_arrays))

    def fit_xy(views_per_chunk):
        # largest x/y chunk within budget, in whole min_xy_chunk_size blocks: never less than one, as smaller
        # chunks cost more in tasks and graph build than they save in memory
        size = min(xy_chunk_size, np.sqrt(voxel_budget / views_per_chunk))
        return max(min_xy_chunk_size, int(size // min_xy_chunk_size) * min_xy_chunk_size)

    if 'z' not in spatial_dims:
        # 2D output: one chunk's cost is sources_per_plane * xy_size ** 2
        xy_size = fit_xy(sources_per_plane)
        return {dim: xy_size for dim in spatial_dims if dim in ('x', 'y')}

    xy_size = fit_xy(sources_per_plane)
    sizes = {dim: xy_size for dim in spatial_dims if dim in ('x', 'y')}
    if num_z_positions > 1:
        # a z chunk of Nz sections costs Nz^2 one plane's (Nz times the sources and the voxels): one plane only
        sizes['z'] = 1
    else:
        # a genuine z-stack adds voxels but no views; floor, as napari computes a whole chunk to show one slice
        voxels_per_chunk = min(target_bytes / np.dtype(dtype).itemsize,
                               voxel_budget / sources_per_plane)
        sizes['z'] = max(1, int(voxels_per_chunk // xy_size ** 2))
    return sizes


def get_export_chunk_sizes(dtype, output_stack_properties, msims, num_z_positions=1,
                           xy_chunk_size=default_export_chunk_size,
                           fusion_target_bytes=None, min_xy_chunk_size=64):
    """Per-dim block sizes for a full-resolution export, budgeted on the sources that actually reach one block.

    Unlike the preview (get_chunk_sizes), whose coarse levels put the whole field of view in one chunk, an export
    block meets only the tiles overlapping it: about density * prod(B_d + E_d) sources, for block size B and mean
    source extent E. The largest block within the fusion budget wins.
    """
    if fusion_target_bytes is None:
        fusion_target_bytes = default_export_fusion_chunk_bytes
    voxel_budget = max(1.0, fusion_target_bytes / (4 * fusion_stack_arrays))

    spacing = output_stack_properties['spacing']
    shape = output_stack_properties['shape']
    xy_dims = [dim for dim in ('y', 'x') if dim in shape]
    if not xy_dims or not msims:
        return get_chunk_sizes(dtype, list(shape), num_sources=len(msims) or 1,
                               num_z_positions=num_z_positions, xy_chunk_size=xy_chunk_size,
                               fusion_target_bytes=fusion_target_bytes)

    # each source's own extent, in output pixels
    extents = []
    for msim in msims:
        physical = get_sim_physical_size(msim)
        extents.append([max(1.0, physical.get(dim, 0) / spacing.get(dim, 1)) for dim in xy_dims])
    mean_extent = np.mean(extents, axis=0)
    sources_per_plane = max(1.0, len(msims) / max(1, num_z_positions))
    output_area = float(np.prod([shape[dim] for dim in xy_dims]))
    density = sources_per_plane / max(output_area, 1.0)

    def sources_in_block(size):
        # the catchment is the block grown by one source: never more than the plane holds, never
        # fewer than the one a block always sits on
        reaching = density * float(np.prod([size + extent for extent in mean_extent]))
        return min(sources_per_plane, max(1.0, reaching))

    size = max(min_xy_chunk_size, int(xy_chunk_size // min_xy_chunk_size) * min_xy_chunk_size)
    while size > min_xy_chunk_size and sources_in_block(size) * size ** len(xy_dims) > voxel_budget:
        size -= min_xy_chunk_size
    sizes = {dim: size for dim in xy_dims}

    if 'z' in shape:
        if num_z_positions > 1:
            # as in get_chunk_sizes: a block spanning Nz planes takes Nz times the sources and Nz
            # times the voxels, so one plane per block
            sizes['z'] = 1
        else:
            sizes['z'] = max(1, int((voxel_budget / sources_in_block(size)) // size ** len(xy_dims)))
    return sizes


def get_display_range(values):
    """[low, high] from the 0.1-99.9 percentiles of `values`, so a few hot pixels don't set it, zeros (the fill
    around a stitched or fused image) left out; None if there is nothing else."""
    values = np.asarray(values).ravel()
    values = values[values != 0]
    if not values.size:
        return None
    low, high = (float(value) for value in np.percentile(values, [0.1, 99.9]))
    return [low, high if high > low else low + 1]


def get_display_level(values, display_range):
    """Where the typical value of `values` (their median, zeros left out as in get_display_range) shows within
    `display_range`, 0 (black) to 1 (full colour); None without either."""
    values = np.asarray(values).ravel()
    values = values[values != 0]
    if not values.size or display_range is None:
        return None
    low, high = display_range
    return float(np.clip((np.median(values) - low) / (high - low), 0, 1))


def get_dtype_contrast_limits(dtype):
    if np.issubdtype(dtype, np.integer):
        return [0, np.iinfo(dtype).max]
    return [0.0, 1.0]


def get_contrast_limits(msim, cheap=False, max_tasks=default_contrast_limits_max_tasks, per_channel=False,
                        with_levels=False):
    """Display range (see get_display_range) from the coarsest pyramid level only, for add_image()'s
    contrast_limits: without it napari computes a min/max itself, eagerly running that level's lazy graph.

    cheap=True returns the dtype range instead, computing nothing; the same fallback applies when the coarsest
    level's graph exceeds max_tasks, as a level fused from many sources is costly however few pixels it has.
    per_channel=True gives a list, one range per channel ('c'); with_levels=True also each one's display level
    (get_display_level), None where nothing was computed: (limits, levels).
    """
    coarsest = get_msim_level_data(msim)[-1]
    dims = get_msim_image0(msim).dims
    channel_axis = dims.index('c') if per_channel and 'c' in dims else None
    nchannels = coarsest.shape[channel_axis] if channel_axis is not None else 1
    if not cheap:
        num_tasks = len(coarsest.dask) if hasattr(coarsest, 'dask') else 0
        if num_tasks > max_tasks:
            logging.info(f'Contrast limits: using dtype range instead of computing'
                         f' the coarsest pyramid level ({num_tasks} tasks > {max_tasks})')
            cheap = True
    if cheap:
        limits = [get_dtype_contrast_limits(coarsest.dtype)] * nchannels
        levels = [None] * nchannels
    else:
        values = np.asarray(dask.compute(coarsest)[0])
        channel_values = np.moveaxis(values, channel_axis, 0) if channel_axis is not None else [values]
        ranges = [get_display_range(channel) for channel in channel_values]
        levels = [get_display_level(channel, display_range) for channel, display_range in zip(channel_values, ranges)]
        limits = [display_range or get_dtype_contrast_limits(coarsest.dtype) for display_range in ranges]
    if not per_channel:
        limits, levels = limits[0], levels[0]
    return (limits, levels) if with_levels else limits


def get_msim_image0(msim, level=0):
    """The raw 'image' DataArray at pyramid `level` (default scale0), without building a sim: enough for anything
    reading only .dims/.coords, not .attrs['transforms']."""
    scale_key = msi_utils.get_sorted_scale_keys(msim)[level]
    return msim[scale_key].ds['image']


def get_msim_dims(msim):
    return get_msim_image0(msim).dims


def get_msim_transform_keys(msim):
    # each transform key is a data_var alongside 'image' on every scale node's Dataset
    scale_key = msi_utils.get_sorted_scale_keys(msim)[0]
    return set(msim[scale_key].ds.data_vars.keys()) - {'image'}


def snapshot_msims_transform(msims, transform_key):
    """Each msim's `transform_key` per scale (None where unset), for restore_msims_transform()."""
    return [{scale_key: (msim[scale_key].ds[transform_key] if transform_key in msim[scale_key].ds.data_vars
                         else None)
             for scale_key in msi_utils.get_sorted_scale_keys(msim)}
            for msim in msims]


def restore_msims_transform(msims, transform_key, snapshot):
    for msim, saved in zip(msims, snapshot):
        for scale_key, transform in saved.items():
            dataset = msim[scale_key].to_dataset().drop_vars([transform_key], errors='ignore')
            if transform is not None:
                dataset[transform_key] = transform
            msim[scale_key].dataset = dataset


def extract_sims_from_fused(result):
    """The scale0 sim (or list of sims) of a fuse() result: one fused msim, or in 'compose' mode a list of
    per-source msims. For consumers with no multiscale support, e.g. save_image()."""
    if isinstance(result, list):
        return [msi_utils.get_sim_from_msim(msim, scale=msi_utils.get_sorted_scale_keys(msim)[0]) for msim in result]
    return msi_utils.get_sim_from_msim(result, scale=msi_utils.get_sorted_scale_keys(result)[0])


def wrap_sims_as_msims(sims):
    # a single-level msim per sim with no real pyramid behind it, for msims-only APIs such as fuse();
    # scale_factors=[] skips a downsample pyramid nothing uses
    return [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]


def combine_msims_as_channels(msims, channel_labels):
    """Single-channel msims sharing the same geometry and levels, stacked per level along a new 'c' dim into one
    multichannel msim."""
    scale_keys = msi_utils.get_sorted_scale_keys(msims[0])
    level_sims = []
    for scale_key in scale_keys:
        channel_sims = [msi_utils.get_sim_from_msim(msim, scale=scale_key) for msim in msims]
        channel_sims = [sim.assign_coords({'c': [label]}) for sim, label in zip(channel_sims, channel_labels)]
        combined = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
        level_sims.append(combined)
    return msi_utils.get_msim_from_sims(level_sims)


def get_numpy_slicing(dimension_order, **slicing):
    slices = []
    for axis in dimension_order:
        index = slicing.get(axis)
        index0 = slicing.get(axis + '0')
        index1 = slicing.get(axis + '1')
        if index0 is not None and index1 is not None:
            slice1 = slice(int(index0), int(index1))
        elif index is not None:
            slice1 = int(index)
        else:
            slice1 = slice(None)
        slices.append(slice1)
    return tuple(slices)



def get_level_from_scale(source, target_scale=1):
    # Only downscaling. A factor, possibly as text ('2'), or a pixel size with its unit ('10um')
    target_scale = parse_scale(target_scale)
    if isinstance(target_scale, dict):
        # dict of desired pixel size
        target_pixel_size = target_scale
        target_scale = {dim: target_pixel_size[dim] / source_pixel_size
                        for dim, source_pixel_size in source.get_pixel_size().items()}
    elif isinstance(target_scale, str):
        # target pixel size with unit
        pixel_size = pixel_size_to_um(target_scale)
        target_pixel_size = {dim: pixel_size for dim in source.get_pixel_size()}
        target_scale = {dim: pixel_size / source_pixel_size
                        for dim, source_pixel_size in source.get_pixel_size().items()}
    else:
        # target scale factor
        target_pixel_size = {dim: float(source_pixel_size * target_scale)
                             for dim, source_pixel_size in source.get_pixel_size().items()}
        target_scale = {dim: target_scale for dim in source.get_pixel_size()}
    # only dims the pyramid actually reduces: one constant across levels (e.g. OME-Zarr's size-1 'z') keeps a
    # factor of 1, which would let any level pass however coarse
    reducing_dims = [dim for dim in source.scale_factors[0]
                     if any(factors.get(dim, 1) > 1 for factors in source.scale_factors)] \
        if source.scale_factors else []

    best_level, best_scale = 0, target_scale
    for level, factors in enumerate(source.scale_factors):
        dims = reducing_dims or list(factors)
        if all(np.isclose(factors[dim], target_scale[dim], rtol=1e-4) for dim in dims):
            best_level, best_scale = level, {dim: target_scale[dim] / factors[dim] for dim in factors}
            break
        # the coarsest level that is still at least as fine as asked for - never coarser
        if all(factors[dim] <= target_scale[dim] for dim in dims):
            best_level, best_scale = level, {dim: target_scale[dim] / factors[dim] for dim in factors}
    if best_level == 0:
        for dim in best_scale:
            if best_scale[dim] < 1:
                best_scale[dim] = 1
    return best_level, best_scale, target_pixel_size


def image_reshape(image: np.ndarray, target_size: tuple) -> np.ndarray:
    tw, th = target_size
    sh, sw = image.shape[0:2]
    if sw < tw or sh < th:
        dw = max(tw - sw, 0)
        dh = max(th - sh, 0)
        padding = [(dh // 2, dh - dh //  2), (dw // 2, dw - dw // 2)]
        if len(image.shape) == 3:
            padding += [(0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=(0, 0))
    if tw < sw or th < sh:
        image = image[0:th, 0:tw]
    return image


def resize_image(image, new_size):
    if not isinstance(new_size, (tuple, list, np.ndarray)):
        # use single value for width; apply aspect ratio
        size = np.flip(image.shape[:2])
        new_size = new_size, new_size * size[1] // size[0]
    return cv.resize(image, new_size)



def draw_keypoints(image, points, color=(255, 0, 0)):
    out_image = color_image(float2int_image(image))
    for point in points:
        point = np.round(point).astype(int)
        cv.drawMarker(out_image, tuple(point), color=color, markerType=cv.MARKER_CROSS, markerSize=5, thickness=1)
    return out_image



def draw_keypoints_matches(image1, points1, image2, points2, matches=[], inliers=[],
                           points_color='black', match_color='red', inlier_color='lime',
                           show_plot=True, output_filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    shape = np.max([image.shape for image in [image1, image2]], axis=0)
    shape_y, shape_x = shape[:2]
    if shape_x > 2 * shape_y:
        merge_axis = 0
        offset2 = [shape_y, 0]
    else:
        merge_axis = 1
        offset2 = [0, shape_x]
    image = np.concatenate([
        np.pad(image1, ((0, shape[0] - image1.shape[0]), (0, shape[1] - image1.shape[1]))),
        np.pad(image2, ((0, shape[0] - image2.shape[0]), (0, shape[1] - image2.shape[1])))
    ], axis=merge_axis)
    ax.imshow(image, cmap='gray')

    if len(points1) > 0:
        ax.scatter(
            points1[:, 1],
            points1[:, 0],
            facecolors='none',
            edgecolors=points_color,
        )
    if len(points2) > 0:
        ax.scatter(
            points2[:, 1] + offset2[1],
            points2[:, 0] + offset2[0],
            facecolors='none',
            edgecolors=points_color,
        )

    for i, match in enumerate(matches):
        color = match_color
        if i < len(inliers) and inliers[i]:
            color = inlier_color
        index1, index2 = match
        ax.plot(
            (points1[index1, 1], points2[index2, 1] + offset2[1]),
            (points1[index1, 0], points2[index2, 0] + offset2[0]),
            '-', linewidth=1, alpha=0.5, color=color,
        )

    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()

    return fig, ax


def preview_marker_sizes(extent):
    """(point size, line width) in image pixels for a preview whose largest side is `extent` pixels: sized to the
    view, or a large overlap shown whole shrinks them to nothing."""
    return max(6, extent / 80), max(1, extent / 400)


def side_by_side_extent(shape1, shape2):
    """The largest side of two images shown side by side along their shortest axis, as the preview shows them."""
    shape = np.maximum(np.asarray(shape1), np.asarray(shape2))
    shape[np.argmin(shape)] *= 2
    return int(shape.max())


def grid_point_pairs(fixed_shape, moving_shape, affine_matrix, points_on_longest_side=30, min_points_on_a_side=3,
                     min_rings_apart=2.5):
    """Points on a regular grid over the fixed image and where `affine_matrix` puts them in the moving image, as
    (fixed points, moving points, matches, inliers): what a registration giving only a transform shows in a preview."""
    ndim = len(fixed_shape)
    # spatial dims come last; on registered pairs the matrix takes fixed pixels to where they lie in the moving image
    matrix = np.asarray(affine_matrix, dtype=float)[-(ndim + 1):, -(ndim + 1):]
    # the longest side's spacing in every direction - no closer than the drawn rings allow (small overlaps), but closer
    # where a side would get fewer than the minimum
    ring_size = preview_marker_sizes(side_by_side_extent(fixed_shape, moving_shape))[0]
    step = max(max(fixed_shape) / points_on_longest_side, min_rings_apart * ring_size)
    axis_steps = [min(step, size / min_points_on_a_side) for size in fixed_shape]
    axes = [np.arange(axis_step / 2, size, axis_step) for size, axis_step in zip(fixed_shape, axis_steps)]
    fixed_points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, ndim)
    # all kept, also where the partner falls outside the moving image: its line then shows how far the tiles are offset
    homogeneous = np.column_stack([fixed_points, np.ones(len(fixed_points))])
    moving_points = (matrix @ homogeneous.T).T[:, :ndim]
    matches = np.column_stack([np.arange(len(fixed_points))] * 2)
    return fixed_points, moving_points, matches, np.ones(len(fixed_points), dtype=bool)


def draw_keypoints_matches_napari(image1, points1, image2, points2, matches=[], inliers=[],
                                  points_color='black', match_color='red', inlier_color='lime'):
    def _as_points_array(points):
        points = np.asarray(points)
        if points.size == 0:
            points = np.empty((0, 0), dtype=float)
        elif points.ndim == 1:
            points = points[None, :]
        return points.astype(float, copy=False)

    def _get_image_spatial_dims(image):
        image = np.asarray(image)
        if image.ndim <= 2:
            return image.ndim
        if image.shape[-1] in (3, 4):
            return image.ndim - 1
        return min(image.ndim, 3)

    points1 = _as_points_array(points1)
    points2 = _as_points_array(points2)
    images = [image for image in (image1, image2) if image is not None]

    point_dims = [pts.shape[1] for pts in (points1, points2) if len(pts) > 0]

    if len(point_dims) > 0:
        spatial_dims = min(max(point_dims), 3)
    elif len(images) > 0:
        spatial_dims = max(_get_image_spatial_dims(image) for image in images)
    else:
        spatial_dims = 2

    if len(images) > 0:
        shape = np.max([
            np.array(np.asarray(image).shape[:spatial_dims], dtype=int)
            for image in images
        ], axis=0)
    else:
        shape = np.ones(spatial_dims, dtype=int)
        all_points = [points[:, :spatial_dims] for points in (points1, points2)
                      if len(points) > 0 and points.shape[1] >= spatial_dims]
        if len(all_points) > 0:
            points_shape = np.ceil(np.max(np.vstack(all_points), axis=0)).astype(int) + 1
            shape = np.maximum(shape, points_shape)

    # Concatenate along the smallest spatial axis to keep the merged view compact.
    merge_axis = int(np.argmin(shape))
    offset2 = np.zeros(spatial_dims, dtype=float)
    offset2[merge_axis] = shape[merge_axis]

    # Pad each image to the same shape before concatenation; include non-spatial axes.
    max_ndim = max((image.ndim for image in images), default=spatial_dims)
    target_shape = np.ones(max_ndim, dtype=int)
    target_shape[:spatial_dims] = shape
    for image in images:
        ext_shape = np.array(image.shape + (1,) * (max_ndim - image.ndim), dtype=int)
        target_shape = np.maximum(target_shape, ext_shape)

    def _pad_image(image):
        if image is None:
            return np.zeros(tuple(target_shape), dtype=np.float32)
        image = np.asarray(image)
        if image.ndim < max_ndim:
            image = image.reshape(image.shape + (1,) * (max_ndim - image.ndim))
        padding = tuple((0, max(target_shape[axis] - image.shape[axis], 0)) for axis in range(max_ndim))
        return np.pad(image, padding)

    image = np.concatenate([_pad_image(image1), _pad_image(image2)], axis=merge_axis)

    # Build combined points in napari's expected coordinate order.
    p1 = points1[:, :spatial_dims] if (len(points1) > 0 and points1.shape[1] >= spatial_dims) else np.empty((0, spatial_dims))
    p2 = points2[:, :spatial_dims] + offset2 if (len(points2) > 0 and points2.shape[1] >= spatial_dims) else np.empty((0, spatial_dims))
    points_data = np.vstack([p1, p2]) if (len(p1) or len(p2)) else np.empty((0, spatial_dims))

    # Build match lines as shapes layers (each line is [[p1], [p2]]).
    non_inlier_line_data = []
    inlier_line_data = []
    for i, match in enumerate(matches):
        is_inlier = inliers[i] if i < len(inliers) else False
        i1, i2 = int(match[0]), int(match[1])
        valid_indices = i1 < len(points1) and i2 < len(points2)
        valid_dims = points1.shape[1] >= spatial_dims and points2.shape[1] >= spatial_dims
        if valid_indices and valid_dims:
            start = points1[i1, :spatial_dims]
            end = points2[i2, :spatial_dims] + offset2
            line = np.array([start, end], dtype=float)
            if is_inlier:
                inlier_line_data.append(line)
            else:
                non_inlier_line_data.append(line)

    layers = [
        (
            image,
            {
                "name": "matches_image",
                # For 2D grayscale this is ignored by napari; for RGB it is inferred.
                "rgb": (image.ndim >= 3 and image.shape[-1] in (3, 4)),
                "blending": "translucent_no_depth"
            },
            "image",
        )
    ]

    point_size, line_width = preview_marker_sizes(max(image.shape[:spatial_dims]))

    if len(points_data) > 0:
        layers.append(
            (
                points_data,
                {
                    "name": "keypoints",
                    "size": point_size,
                    "face_color": points_color,
                    "border_color": "transparent",
                    "symbol": "ring",
                    "opacity": 0.5,
                },
                "points",
            )
        )

    if len(non_inlier_line_data) > 0:
        layers.append(
            (
                non_inlier_line_data,
                {
                    "name": "matches",
                    "shape_type": "line",
                    "edge_color": match_color,
                    "edge_width": line_width,
                    "opacity": 0.25,
                },
                "shapes",
            )
        )

    if len(inlier_line_data) > 0:
        layers.append(
            (
                inlier_line_data,
                {
                    "name": "matches_inliers",
                    "shape_type": "line",
                    "edge_color": inlier_color,
                    "edge_width": line_width,
                    "opacity": 0.25,
                },
                "shapes",
            )
        )

    return layers


def metric_to_rgb(value, min_light=0, max_light=1, output_range=1.0):
    # metric range 0...1 to red-yellow-green ranged rgb
    if value is None or np.isnan(value):
        return 0, 0, 0
    colormap = plt.colormaps.get('RdYlGn')
    index = int(value * colormap.N)
    r, g, b, a = [float(value) for value in colormap(index)]
    light = 0.2125 * r + 0.7154 * g + 0.0721 * b
    if light < min_light:
        factor = light / min_light
        r = 1 - (1 - r) * factor
        g = 1 - (1 - g) * factor
        b = 1 - (1 - b) * factor
    elif light > max_light:
        factor = max_light / light
        r *= factor
        g *= factor
        b *= factor
    r *= output_range
    g *= output_range
    b *= output_range
    if isinstance(output_range, int):
        r, g, b = int(r), int(g), int(b)
    return r, g, b


def create_compression_filter(compression: list) -> tuple:
    compressor, compression_filters = None, None
    compression = ensure_list(compression)
    if compression is not None and len(compression) > 0:
        compression_type = compression[0].lower()
        if len(compression) > 1:
            level = int(compression[1])
        else:
            level = None
        if 'lzw' in compression_type:
            from imagecodecs.numcodecs import Lzw
            compression_filters = [Lzw()]
        elif '2k' in compression_type or '2000' in compression_type:
            from imagecodecs.numcodecs import Jpeg2k
            compression_filters = [Jpeg2k(level=level)]
        elif 'jpegls' in compression_type:
            from imagecodecs.numcodecs import Jpegls
            compression_filters = [Jpegls(level=level)]
        elif 'jpegxr' in compression_type:
            from imagecodecs.numcodecs import Jpegxr
            compression_filters = [Jpegxr(level=level)]
        elif 'jpegxl' in compression_type:
            from imagecodecs.numcodecs import Jpegxl
            compression_filters = [Jpegxl(level=level)]
        else:
            compressor = compression
    return compressor, compression_filters


def gaussian_filter_image(image, sigma, is_3d=False):
    ndims = 4 if is_3d else 3
    nchannels = image.shape[-1] if image.ndim > ndims else 1
    if nchannels not in [1, 3]:
        new_image = np.zeros_like(image)
        for channeli in range(nchannels):
            new_image[..., channeli] = gaussian(image[..., channeli], sigma)
    else:
        new_image = gaussian(image, sigma, preserve_range=True)
    return new_image


def calc_images_median(images):
    out_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    median_image = np.median(images, 0, out_image)
    return median_image



def get_image_quantile(image: np.ndarray, quantile: float, axis=None) -> float:
    # materialised first (only ever a small level) so dask's own quantile is never used: some versions pass
    # numpy's removed interpolation= kwarg
    image = np.asarray(image)
    if axis is None:
        image = image.ravel()
        axis = 0
    value = np.quantile(image, quantile, axis=axis).astype(image.dtype)
    return np.array(value).item()


def get_image_window(image, low=0.01, high=0.99):
    window = (
        get_image_quantile(image, low),
        get_image_quantile(image, high)
    )
    return window


def normalise_values(image: np.ndarray, min_value: float=None, max_value: float=None) -> np.ndarray:
    if min_value is None or max_value is None:
        min_value, max_value = get_image_window(image)
    image = (image.astype(np.float32) - min_value) / (max_value - min_value)
    return image.clip(0, 1)



def norm_image_variance2(image0, is_3d=False):
    ncoldims = 4 if is_3d else 3
    if len(image0.shape) == ncoldims and image0.shape[-1] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = ((image - np.mean(image)) / np.std(image) + 1) / 2
    normimage = normimage.clip(0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage



def filter_noise_images(images):
    dtype = images[0].dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    image_vars = [np.asarray(np.std(image)).item() for image in images]
    threshold, mask0 = cv.threshold(np.array(image_vars).astype(dtype), 0, maxval, cv.THRESH_OTSU)
    mask = [flag.item() for flag in mask0.astype(bool)]
    return int(threshold), mask


def invert_data(data):
    if isinstance(data, list):
        return [invert_data(d) for d in data]
    else:
        dtype = data.dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return info.max - data
        elif np.issubdtype(dtype, np.floating):
            return 1.0 - data
        else:
            return data


def detect_area_points(data):
    method = cv.THRESH_OTSU
    threshold = -5
    contours = []
    while len(contours) <= 1 and threshold <= 255:
        _, binimage = cv.threshold(np.array(uint8_image(data)), threshold, 255, method)
        contours0 = cv.findContours(binimage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours0[0] if len(contours0) == 2 else contours0[1]
        method = cv.THRESH_BINARY
        threshold += 5
    area_contours = [(contour, cv.contourArea(contour)) for contour in contours]
    area_contours.sort(key=lambda contour_area: contour_area[1], reverse=True)
    min_area = max(np.mean([area for contour, area in area_contours]), 1)
    area_points = [(get_center(contour), area) for contour, area in area_contours if area > min_area]

    return area_points


def detect_volume_points(data):
    blobs = blob_log(data, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1)
    if blobs.shape[1] > 3:
        blobs = blobs[:, :3]
    return blobs


def get_transforms(items):
    # sims or msims - a msim's transform keys are read directly off its data_vars, no sim built to enumerate them
    groups = [get_msim_transform_keys(item) if isinstance(item, DataTree)
             else si_utils.get_tranform_keys_from_sim(item)
             for item in items]
    return list({a for group in groups for a in group})



def get_sim_position_final(data, position=None, transform_keys=None, get_center=False):
    # a sim or a msim (its scale0 sim is used) - only position/transform metadata is ever read here
    sim = msi_utils.get_sim_from_msim(data, scale='scale0') if isinstance(data, DataTree) else data
    if position is None:
        position = si_utils.get_origin_from_sim(sim)
    if transform_keys is None or len(transform_keys) == 0:
        transform_keys = si_utils.get_tranform_keys_from_sim(sim)

    transforms = []
    transform_dims = []
    for transform_key in transform_keys:
        transform = si_utils.get_affine_from_sim(sim, transform_key)
        if 't' in transform.dims:
            transform = transform.isel(t=0)
        transforms.append(np.array(transform))
        transform_dims = transform['x_in'].data.tolist()
    transform = combine_transforms(transforms)

    new_position = apply_transform_dict([position], transform, transform_dims)[0]
    for dim in position.keys():
        if dim not in new_position:
            new_position[dim] = position[dim]
    if get_center:
        physical_size = get_sim_physical_size(sim)
        new_position = {dim: new_position[dim] + physical_size.get(dim, 0) / 2 for dim in new_position}
    return new_position


def group_sims_by_z(sims, positions=None):
    grouped_sims = []
    if positions is None:
        positions = [si_utils.get_origin_from_sim(sim) for sim in sims]
    z_positions = [position.get('z', 0) for position in positions]
    unique_z_values = len(set(z_positions)) == len(z_positions)
    if not unique_z_values:
        sims_by_z = {}
        for simi, z_pos in enumerate(z_positions):
            if z_pos is not None and z_pos not in sims_by_z:
                sims_by_z[z_pos] = []
            sims_by_z[z_pos].append(simi)
        grouped_sims = list(sims_by_z.values())
    if len(grouped_sims) == 0:
        grouped_sims = [list(range(len(sims)))]
    return grouped_sims


def calc_foreground_map(items):
    # sims or msims (each msim's scale0 sim is used): pixel data, a median-image comparison across all sources
    sims = sims_from_sims_or_msims(items)
    if len(sims) <= 2:
        return [True] * len(sims)
    sims = [sim.squeeze().astype(np.float32) for sim in sims]
    median_image = calc_images_median(sims).astype(np.float32)
    difs = [np.mean(np.abs(sim - median_image), (0, 1)) for sim in sims]
    # or use stddev instead of mean?
    threshold = np.mean(difs, 0)
    #threshold, _ = cv.threshold(np.array(difs).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
    #threshold, foregrounds = filter_noise_images(channel_images)
    map = (difs > threshold)
    if np.all(map == False):
        return [True] * len(sims)
    return map


def sim_with_data(sim, data, transform_key):
    """A new sim of `data` at `sim`'s geometry, coordinates and `transform_key` transform."""
    return si_utils.get_sim_from_array(data, dims=sim.dims, scale=si_utils.get_spacing_from_sim(sim),
                                       translation=si_utils.get_origin_from_sim(sim), transform_key=transform_key,
                                       affine=si_utils.get_affine_from_sim(sim, transform_key), c_coords=sim.c.data,
                                       t_coords=sim.t.data)


def normalise_sim(sim, transform_key, min, range, dtype):
    #image = (sim - min) / range
    image = ((sim - min) / range + 1) / 2   # extended range
    image = float2int_image(image.clip(0, 1), dtype)    # np.clip(image) is not dask-compatible, use image.clip() instead
    return sim_with_data(sim, image, transform_key)


def calc_normalise_stats(items, use_global=True):
    """Per-source (mean, stddev) normalisation stats and the dtype of sims or msims (each msim's scale0 sim), without
    normalising anything."""
    sims = sims_from_sims_or_msims(items)
    dtype = sims[0].dtype
    if use_global:
        mins = [np.mean(sim, dtype=np.float32) for sim in sims]
        ranges = [np.std(sim, dtype=np.float32) for sim in sims]
        #min, max = get_image_window(sim, low=0.01, high=0.99)
        #range = max - min
        min = np.mean(mins)
        range = np.mean(ranges)
        stats = [(min, range)] * len(sims)
    else:
        stats = [(np.mean(sim, dtype=np.float32), np.std(sim, dtype=np.float32)) for sim in sims]
    return stats, dtype


def normalise_sims(sims, transform_key, use_global=True):
    stats, dtype = calc_normalise_stats(sims, use_global=use_global)
    return [normalise_sim(sim, transform_key, min, range, dtype) for sim, (min, range) in zip(sims, stats)]


def gaussian_filter_sim(sim, transform_key, sigma):
    image = np.asarray(sim)
    spatial_dims = set(si_utils.get_spatial_dims_from_sim(sim))
    sigma_by_axis = [sigma if dim in spatial_dims else 0 for dim in sim.dims]
    blurred_image = gaussian(image, sigma=sigma_by_axis, preserve_range=True)
    return sim_with_data(sim, blurred_image.astype(sim.dtype), transform_key)


def get_sim_physical_size(data):
    # a sim, a msim (its scale0 sim is used) or stack properties
    if isinstance(data, dict):
        return {dim: size * data['spacing'].get(dim, 1) for dim, size in data['shape'].items()}
    sim = msi_utils.get_sim_from_msim(data, scale='scale0') if isinstance(data, DataTree) else data
    size = si_utils.get_shape_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    physical_size = {dim: size[dim] * spacing.get(dim, 1) for dim in size}
    return physical_size


def promoted_geometry(item, transform_key, z_position):
    """(stack properties, affine matrix) of a sim's, or a msim's finest level's, geometry as
    make_msims_3d() would promote it - a size-1 z at z_position, the transform widened - read
    from its coordinates alone, building no xarray objects (the promotion's whole cost)."""
    if isinstance(item, DataTree):
        image, affine = item['scale0'].ds['image'], item['scale0'].ds[transform_key]
    else:
        image, affine = item, si_utils.get_affine_from_sim(item, transform_key)
    if 't' in affine.dims:
        affine = affine.sel(t=0)
    shape, spacing, origin = (si_utils.get_shape_from_sim(image), si_utils.get_spacing_from_sim(image),
                              si_utils.get_origin_from_sim(image))
    if 'z' not in shape:
        # si_utils reports a size-1 dim's spacing as 1.0, whatever the nominal pixel size
        shape, spacing, origin = {'z': 1, **shape}, {'z': 1.0, **spacing}, {'z': float(z_position), **origin}
    return {'shape': shape, 'spacing': spacing, 'origin': origin}, widened_affine_matrix(affine)


def sim_views(sims, transform_key):
    """(stack properties, affine matrix) of each sim, the affine at the first time point as fuse() asks it."""
    views = []
    for sim in sims:
        param = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        views.append((si_utils.get_stack_properties_from_sim(sim),
                      np.asarray(param.isel(t=0) if 't' in param.dims else param, dtype=float)))
    return views


def calc_output_properties(items, transform_key, output_spacing_method=None, z_scale=None, z_positions=None):
    # sims or msims, reading only their scale0 geometry. z_positions: sources at several heights, not promoted to
    # 3D - taken as make_msims_3d() would make them (see promoted_geometry), with the same result
    if z_positions is not None:
        views = [promoted_geometry(item, transform_key, z) for item, z in zip(items, z_positions)]
    else:
        views = sim_views(sims_from_sims_or_msims(items), transform_key)
    return calc_output_properties_from_views(views, output_spacing_method=output_spacing_method, z_scale=z_scale)


def calc_output_properties_from_views(views, output_spacing_method=None, z_scale=None):
    """The fused output's stack properties from each source's (stack properties, affine matrix) - geometry only."""
    spacings = [stack['spacing'] for stack, _ in views]
    dims = list(spacings[0])
    is_3d = views[0][0]['shape'].get('z', 0) > 1
    z_origins = [stack['origin'].get('z', 0) for stack, _ in views]
    output_spacing = {}

    if output_spacing_method:
        output_spacing_method = output_spacing_method.lower()
    if not output_spacing_method or 'mean' in output_spacing_method:
        output_spacing = {dim: np.mean([spacing[dim] for spacing in spacings]) for dim in dims}
    elif 'max' in output_spacing_method:
        output_spacing = {dim: max([spacing[dim] for spacing in spacings]) for dim in dims}
    elif 'min' in output_spacing_method:
        output_spacing = {dim: min([spacing[dim] for spacing in spacings]) for dim in dims}

    if z_scale and 'z' in dims and not is_3d:
        output_spacing['z'] = z_scale
    stack = fusion.calc_stack_properties_from_view_properties_and_params(
        [view[0] for view in views], [view[1] for view in views], spacing=output_spacing, mode='union')
    output_properties = {key: {dim: value[index] for index, dim in enumerate(dims)} for key, value in stack.items()}
    if 'z' in output_properties['shape'] and not is_3d:
        z_values = sorted({round(float(z_origin), 9) for z_origin in z_origins})
        z_shape = len(z_values)
        if z_shape <= 1:
            z_shape = len(z_origins)
        elif not z_scale:
            # a size-1 z reports a spacing of 1.0: sections closer than that would all land in the first plane
            output_properties['spacing']['z'] = float(np.min(np.diff(z_values)))
        output_properties['shape']['z'] = z_shape
    return output_properties


def get_properties_from_transform(transform):
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    xtranslation = param_utils.translation_from_affine(transform)
    dims = xtranslation['x_in'].data.tolist()
    translation = {dim: xtranslation.sel(x_in=dim).item() for dim in dims}
    rotation = get_rotation_from_transform(transform)
    scale = get_scale_from_transform(transform)
    return translation, rotation, scale


def get_data_mapping(data, transform_key=None, transform=None, translation0=None, rotation=None):
    if isinstance(data, DataTree):
        sim = msi_utils.get_sim_from_msim(data)
    else:
        sim = data
    translation = si_utils.get_origin_from_sim(sim)
    if 'z' not in translation and translation0 is not None and 'z' in translation0:
        translation['z'] = translation0['z']

    if transform is not None:
        translation1, rotation1, _ = get_properties_from_transform(transform)
        dims = set(list(translation) + list(translation1))
        translation = {dim: translation.get(dim, 0) + translation1.get(dim, 0) for dim in dims}
        if rotation1:
            rotation = (rotation or 0) + rotation1

    if transform_key is not None:
        transform1 = sim.transforms.get(transform_key)
        if transform1 is not None:
            _, rotation1, _ = get_properties_from_transform(transform1)
            if rotation1:
                rotation = (rotation or 0) + rotation1

    return translation, rotation


def extract_z_scale(positions, scales=None):
    z_scale = None

    if scales is not None:
        z_scale0 = np.mean([scale.get('z', 0) for scale in scales])
        if z_scale0 > 0:
            z_scale = z_scale0

    if z_scale is None:
        z_positions = [position.get('z') for position in positions if 'z' in position]
        if len(z_positions) > 1:
            diffs = np.diff(sorted(set(z_positions)))
            if len(diffs) > 0:
                z_scale = min(diffs)
    return z_scale


def _axis_aligned_bb_vertices_2d(lo, hi):
    """The corners _minimal_bb_vertices() returns for an axis-aligned 2D box, counter-clockwise
    from the low one, without its hull search.

    `lo`/`hi` may be single (2,) corners or (N, 2) stacks, giving (4, 2) or (N, 4, 2).
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.stack([
        np.stack([lo[..., 0], lo[..., 1]], axis=-1),
        np.stack([hi[..., 0], lo[..., 1]], axis=-1),
        np.stack([hi[..., 0], hi[..., 1]], axis=-1),
        np.stack([lo[..., 0], hi[..., 1]], axis=-1),
    ], axis=-2)


def _bb_vertices(points):
    """_minimal_bb_vertices(), short-circuited when *points* are already an axis-aligned 2D box
    - which every source is before registration has rotated anything. A rotated or 3D point set
    still takes the full search: there the corner ordering depends on the box's own axes.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 2 and points.shape[1] == 2:
        lo, hi = points.min(axis=0), points.max(axis=0)
        # exact, not isclose: these are the same floats min/max were just taken from
        if np.all((points == lo) | (points == hi)):
            return _axis_aligned_bb_vertices_2d(lo, hi)
    return _minimal_bb_vertices(points)


def _minimal_bb_vertices(points, return_edge_path=False, convex=False):
    """Return the corners of an oriented bounding box around *points*.

    Corners are ordered around one face before the corresponding corners on
    the opposite face.  If ``return_edge_path`` is true, return the continuous
    path used to render all edges of a 3D box without diagonals. ``convex``: 2D
    points that already are a convex polygon in order, so need no hull.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must be an (N, 2) or (N, 3) array")
    if len(points) == 0:
        raise ValueError("at least one point is required")

    centered = points - points.mean(axis=0)
    if points.shape[1] == 2:
        if convex:
            hull_points = points
        else:
            hull = ConvexHull(points) if np.linalg.matrix_rank(centered) == 2 else None
            hull_points = points[hull.vertices] if hull is not None else points
        edges = np.diff(np.vstack((hull_points, hull_points[0])), axis=0)
        directions = edges[np.linalg.norm(edges, axis=1) > 0]
        if len(directions) == 0:
            axes = np.eye(2)
        else:
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            candidates = np.stack(
                (
                    directions,
                    np.column_stack((-directions[:, 1], directions[:, 0])),
                ),
                axis=1,
            )
            projected = points @ candidates.transpose(0, 2, 1)
            areas = np.prod(np.ptp(projected, axis=1), axis=1)
            # of equal boxes the one nearest the image axes, wherever the polygon starts: an axis-aligned one as drawn
            tied = np.flatnonzero(areas <= areas.min() * (1 + 1e-9))
            axes = candidates[tied[np.argmax(candidates[tied, 0, 0])]]
    else:
        rank = np.linalg.matrix_rank(centered)
        candidates = []
        if rank == 3:
            hull = ConvexHull(points)
            for simplex, normal in zip(hull.simplices, hull.equations[:, :3]):
                z_axis = normal / np.linalg.norm(normal)
                face = points[simplex]
                for start, end in ((0, 1), (1, 2), (2, 0)):
                    x_axis = face[end] - face[start]
                    x_axis -= np.dot(x_axis, z_axis) * z_axis
                    length = np.linalg.norm(x_axis)
                    if length != 0:
                        x_axis /= length
                        y_axis = np.cross(z_axis, x_axis)
                        candidates.append(np.stack((x_axis, y_axis, z_axis)))

        if candidates:
            volumes = []
            for candidate in candidates:
                projected = points @ candidate.T
                volumes.append(np.prod(np.ptp(projected, axis=0)))
            axes = candidates[np.argmin(volumes)]
        else:
            # A deterministic orthonormal basis also gives sensible output for
            # lower-rank inputs, for which a 3D minimum-volume box is not unique.
            _, _, vh = np.linalg.svd(centered, full_matrices=True)
            axes = vh

    projected = points @ axes.T
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    face_order = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    corner_bits = face_order
    if points.shape[1] == 3:
        corner_bits = np.vstack(
            (
                np.column_stack((face_order, np.zeros(4, dtype=int))),
                np.column_stack((face_order, np.ones(4, dtype=int))),
            )
        )
    local_corners = mins + corner_bits * (maxs - mins)
    corners = local_corners @ axes

    if not return_edge_path:
        return corners
    if points.shape[1] != 3:
        raise ValueError("an edge path is only defined for 3D boxes")
    edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
    return corners[edge_path]


def set_oriented_bounding_box_edges(layer, shapes):
    """Correct napari-bbox's edge mesh for oriented 3D boxes.

    napari-bbox lexicographically sorts 3D vertices before applying a path
    intended for axis-aligned boxes.  Sorting destroys the topology of a
    rotated box and makes some of those edges diagonals.  Rebuild each private
    edge mesh from our known face-ordered vertices until napari-bbox preserves
    input ordering itself.
    """
    data_view = layer._data_view
    for index, shape in enumerate(shapes):
        bounding_box = data_view.bounding_boxes[index]
        if bounding_box.ndisplay != 3:
            raise RuntimeError(
                "The BoundingBoxLayer must be added to a 3D viewer before "
                "correcting its oriented edge mesh."
            )
        bounding_box._set_meshes(
            _minimal_bb_vertices(
                np.asarray(shape)[:, -3:],
                return_edge_path=True,
            ),
            closed=False,
            face=False,
        )
        data_view._update_mesh_vertices(index, edge=True)


def sims_from_sims_or_msims(items):
    # normalises a list of sims and/or msims to sims (each msim's scale0 sim is used)
    return [msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
           for item in items]


def stack_props_from_any(items, transform_key=None):
    """Stack properties of sims, msims or already-built stack-properties dicts - all the shapes path reads, so
    geometry without image data (build_source_stack_props()) can be passed directly.
    """
    stack_props = []
    for item in items:
        if isinstance(item, dict):
            stack_props.append(item)
            continue
        sim = msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
        if 't' in sim.dims:
            sim = sim.sel(t=0)
        stack_props.append(si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key))
    return stack_props


def create_image_shapes(items, transform_key=None,  force_2d=False):
    # accepts sims, msims or stack properties - only position/size metadata is read, never pixel
    # data, and never anything that would make a source open its file
    all_stack_props = stack_props_from_any(items, transform_key)
    shapes = []
    is_multi_z_shapes = (len(set([props['origin'].get('z', 0) for props in all_stack_props])) > 1)
    for stack_props in all_stack_props:
        points = mv_graph.get_vertices_from_stack_props(stack_props)
        if points.shape[1] == 3 and (len(set(points[:, 0])) == 1 or force_2d):
            # remove constant z coordinate
            points = points[:, 1:]
        shape = _bb_vertices(points)
        if is_multi_z_shapes:
            z_position = stack_props['origin'].get('z', 0)
            shape = [[z_position] + list(element) for element in shape]
        shapes.append(shape)
    return shapes


def _filter_candidate_overlap_pairs(all_stack_props):
    """Pairs whose axis-aligned bounding boxes overlap: a cheap broad phase in front of _get_overlap_bboxes' exact
    linear-programming test. An AABB contains the real (possibly rotated) box, so no overlapping pair is dropped.

    Also returns the per-sim mins/maxs, from which an unrotated pair's overlap shape can be drawn directly.
    """
    mins = np.empty((len(all_stack_props), 3))
    maxs = np.empty((len(all_stack_props), 3))
    for index, stack_props in enumerate(all_stack_props):
        points = mv_graph.get_vertices_from_stack_props(stack_props)
        ndims = points.shape[1]
        mins[index] = 0
        maxs[index] = 0
        mins[index, :ndims] = points.min(axis=0)
        maxs[index, :ndims] = points.max(axis=0)
    return _sweep_candidate_pairs(mins, maxs), mins, maxs


def find_candidate_overlap_pairs(items, transform_key=None):
    """(i, j) index pairs, i < j, of the sims/msims whose bounding boxes overlap: every pair that
    can overlap, and few others - a candidate list for an exact overlap test to confirm.
    """
    pairs, _, _ = _filter_candidate_overlap_pairs(stack_props_from_any(items, transform_key))
    return [(int(first), int(second)) for first, second in pairs]


def _sweep_candidate_pairs(mins, maxs, chunk_candidates=4_000_000):
    """Index pairs whose axis-aligned boxes overlap, by sweeping one axis rather than comparing
    every box with every other - the same pairs, in the same (i < j, row-major) order.

    All-pairs is n^2 in time and memory; sorted by one axis' lower bound, each box is tested only against those
    starting before it ends.
    """
    n = len(mins)
    if n < 2:
        return np.empty((0, 2), dtype=int)

    # sweep whichever axis separates these boxes best (a single-section mosaic gains nothing from z), by count
    best = None
    for axis in range(mins.shape[1]):
        order = np.argsort(mins[:, axis], kind='stable')
        ends = np.searchsorted(mins[order, axis], maxs[order, axis], side='right')
        counts = np.maximum(ends - np.arange(n) - 1, 0)
        total = int(counts.sum())
        if best is None or total < best[0]:
            best = (total, order, counts)
    _, order, counts = best

    mins_sorted, maxs_sorted = mins[order], maxs[order]
    starts = np.arange(n) + 1
    cumulative = np.cumsum(counts)
    blocks = []
    begin = 0
    while begin < n:
        # a large stack can take minutes here: let a cancel stop it
        raise_if_cancelled()
        # never fewer than one box: one whose own candidates exceed the budget still goes whole
        budget = (cumulative[begin - 1] if begin > 0 else 0) + chunk_candidates
        end = max(int(np.searchsorted(cumulative, budget, side='right')), begin + 1)
        block_counts = counts[begin:end]
        n_candidates = int(block_counts.sum())
        if n_candidates > 0:
            index1 = np.repeat(np.arange(begin, end), block_counts)
            # each box's candidates are the contiguous run just after it, so its partner
            # indices are that run's start plus 0..count-1
            block_cumulative = np.cumsum(block_counts)
            offsets = (np.arange(n_candidates)
                       - np.repeat(block_cumulative - block_counts, block_counts))
            index2 = np.repeat(starts[begin:end], block_counts) + offsets
            keep = np.all((mins_sorted[index1] <= maxs_sorted[index2])
                          & (mins_sorted[index2] <= maxs_sorted[index1]), axis=-1)
            if keep.any():
                blocks.append(np.column_stack((order[index1[keep]], order[index2[keep]])))
        begin = end

    if not blocks:
        return np.empty((0, 2), dtype=int)
    pairs = np.concatenate(blocks)
    # sorting by the sweep axis shuffled both which index of a pair comes first, and which
    # pair does - undo both, so the result is indistinguishable from the all-pairs one
    pairs = np.sort(pairs, axis=1)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _aabb_overlap_shapes_2d(all_stack_props, pairs, mins, maxs, force_2d, is_multi_z_shapes):
    """create_overlap_shapes' AABB fast path for every pair at once - same boxes, same order,
    no per-pair Python. See there for why the AABB intersection is exact in this case.
    """
    pairs = np.asarray(pairs).reshape(-1, 2)
    origins_z = np.array([props['origin'].get('z', 0) for props in all_stack_props], dtype=float)
    if force_2d:
        # multi-section 2D data: an overlap only means anything within a section, and the
        # promoted singleton 'z' leaves the real extent in columns 1/2
        singleton_z = np.array([props['shape'].get('z') == 1 for props in all_stack_props])
        both_singleton = singleton_z[pairs[:, 0]] & singleton_z[pairs[:, 1]]
        same_z = origins_z[pairs[:, 0]] == origins_z[pairs[:, 1]]
        pairs = pairs[~both_singleton | same_z]
        axes = slice(1, 3)
    else:
        axes = slice(0, 2)

    # broad_phase already guarantees lo <= hi in every axis (that's its own overlap condition)
    lo = np.maximum(mins[pairs[:, 0]], mins[pairs[:, 1]])[:, axes]
    hi = np.minimum(maxs[pairs[:, 0]], maxs[pairs[:, 1]])[:, axes]
    boxes = _axis_aligned_bb_vertices_2d(lo, hi)
    if is_multi_z_shapes:
        # the shape sits at the first source's z, as the exact path puts it too
        z_positions = np.repeat(origins_z[pairs[:, 0], None, None], boxes.shape[1], axis=1)
        boxes = np.concatenate([z_positions, boxes], axis=2)
    return list(boxes), list(pairs)


def _is_axis_aligned(stack_props):
    """Whether a source's transform scales and shifts only, so its bounding box is the source itself."""
    if 'transform' not in stack_props:
        return True
    matrix = np.asarray(stack_props['transform'])
    linear = matrix.reshape(-1, *matrix.shape[-2:])[0][:-1, :-1]
    return not np.any(linear - np.diag(np.diagonal(linear)))


def _tile_polygon_2d(stack_props):
    """A source's yx footprint as a counter-clockwise polygon - a singleton z dropped."""
    points = np.asarray(mv_graph.get_vertices_from_stack_props(stack_props), dtype=float)[:, -2:]
    points = np.unique(points, axis=0)
    centre = points.mean(axis=0)
    return points[np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))]


def _clip_convex_polygons_2d(subject, clip):
    """Sutherland-Hodgman: the intersection of two convex counter-clockwise polygons, or None where they do not
    overlap in an area. The exact test's answer in 2D, without its linear program."""
    output = list(subject)
    for index in range(len(clip)):
        if not output:
            return None
        start, end = clip[index], clip[(index + 1) % len(clip)]
        edge = end - start

        def inside(point):
            return edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0]) >= 0

        def crossing(point1, point2):
            delta = point2 - point1
            denominator = edge[0] * delta[1] - edge[1] * delta[0]
            ratio = (edge[1] * (point1[0] - start[0]) - edge[0] * (point1[1] - start[1])) / denominator
            return point1 + ratio * delta

        points, output = output, []
        for current, previous in zip(points, points[-1:] + points[:-1]):
            if inside(current):
                if not inside(previous):
                    output.append(crossing(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(crossing(previous, current))
    if len(output) < 3:
        return None
    polygon = np.array(output)
    area = 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
    return polygon if area > 1e-9 * np.ptp(polygon, axis=0).max() ** 2 else None


def create_overlap_shapes(items, transform_key, pairs=None, force_2d=False, dtype=np.uint8):
    # sims, msims or stack properties: only multiview_stitcher's exact test needs sims, built per pair from the
    # stack properties where none were given (get_pair_sim)
    all_stack_props = stack_props_from_any(items, transform_key)
    sims = None if any(isinstance(item, dict) for item in items) else sims_from_sims_or_msims(items)

    def get_pair_sim(index):
        if sims is not None:
            return squeeze_sim_transform_time(sims[index], transform_key)
        return sim_from_stack_props(all_stack_props[index], dtype, transform_key)

    shapes = []
    good_pairs = []
    is_multi_z_shapes = (len(set([props['origin'].get('z', 0) for props in all_stack_props])) > 1)
    # without given pairs, a pair of unrotated sources overlaps exactly in its bounding boxes' intersection:
    # that skips the exact test. A rotated source (e.g. by source_rotation) takes the exact test
    aabbs = None
    if pairs is None:
        broad_phase_start = time.time()
        pairs, mins, maxs = _filter_candidate_overlap_pairs(all_stack_props)
        aabbs = (mins, maxs)
        axis_aligned = np.array([_is_axis_aligned(props) for props in all_stack_props], dtype=bool)
        logging.info(f'create_overlap_shapes: {len(pairs)} candidate pairs from'
                     f' {len(all_stack_props)} sims'
                     f' (broad phase: {time.time() - broad_phase_start:.1f}s)')
    # in 2D the AABB fast path is pure arithmetic per pair, so do every pair at once rather
    # than a Python iteration (and a convex hull) each - 3D and exact tests take the loop
    if aabbs is not None and axis_aligned.all() and (
            force_2d or not any('z' in props['shape'] for props in all_stack_props)):
        shapes, good_pairs = _aabb_overlap_shapes_2d(all_stack_props, pairs, mins, maxs,
                                                     force_2d=force_2d,
                                                     is_multi_z_shapes=is_multi_z_shapes)
        logging.info(f'create_overlap_shapes: {len(shapes)} shapes from broad-phase AABB '
                     f'intersection directly (vectorized, no linprog)')
        return shapes, good_pairs

    n_exact_tests = 0
    exact_test_time = 0.0
    n_fast_shapes = 0
    n_clipped_shapes = 0
    clip_start = time.time()
    for pair in pairs:
        props1 = all_stack_props[pair[0]]
        props2 = all_stack_props[pair[1]]
        shape_z_position = props1['origin'].get('z', 0)
        process_pair = True

        # promoted 2D sections (singleton z) overlap only within a section, tested in 2D: zero-thickness 3D
        # boxes give invalid face normals in _get_overlap_bboxes
        if (
            force_2d
            and props1['shape'].get('z') == 1
            and props2['shape'].get('z') == 1
        ):
            process_pair = (props1['origin'].get('z', 0) == props2['origin'].get('z', 0))

        if not process_pair:
            continue

        if aabbs is not None and axis_aligned[pair[0]] and axis_aligned[pair[1]]:
            mins, maxs = aabbs
            if force_2d:
                # the size-1 'z' is still there: mins/maxs column 0 is z, columns 1/2 the real extent
                axes = slice(1, 3)
            elif 'z' in props1['shape'] and 'z' in props2['shape']:
                axes = slice(0, 3)
            else:
                axes = slice(0, 2)
            lo = np.maximum(mins[pair[0]], mins[pair[1]])[axes]
            hi = np.minimum(maxs[pair[0]], maxs[pair[1]])[axes]
            # the broad phase guarantees lo <= hi in every axis, so the box is never empty
            corners = np.array(list(itertools.product(*zip(lo, hi))))
            shape = _minimal_bb_vertices(corners)
            n_fast_shapes += 1
            if is_multi_z_shapes:
                shape = [[shape_z_position] + list(element) for element in shape]
            shapes.append(shape)
            good_pairs.append(pair)
            continue

        if 'z' not in props1['shape'] and 'z' not in props2['shape'] or (
                force_2d and props1['shape'].get('z') == 1 and props2['shape'].get('z') == 1):
            polygon = _clip_convex_polygons_2d(_tile_polygon_2d(props1), _tile_polygon_2d(props2))
            if polygon is not None:
                shape = _minimal_bb_vertices(polygon, convex=True)
                if is_multi_z_shapes:
                    shape = [[shape_z_position] + list(element) for element in shape]
                shapes.append(shape)
                good_pairs.append(pair)
                n_clipped_shapes += 1
        else:
            # only from here on is an actual sim needed - the exact test below is multiview_stitcher's
            sim1, sim2 = get_pair_sim(pair[0]), get_pair_sim(pair[1])
            # only singleton-z sims (promoted 2D sections) are squeezed; a real volume is intersected
            # in 3D and its points projected to yx below
            if force_2d and all(sim.sizes.get('z', 1) == 1 for sim in (sim1, sim2)):
                projected_sims = []
                for sim in (sim1, sim2):
                    if 'z' in sim.dims:
                        sim_2d = sim.squeeze('z', drop=True)
                        sim_2d.attrs = dict(sim.attrs)
                        sim_2d.attrs['transforms'] = dict(sim.attrs['transforms'])
                        affine_2d = _adapt_transform_to_image_dims(
                            sim_2d,
                            sim_2d.attrs['transforms'][transform_key],
                            transform_key,
                        )
                        si_utils.set_sim_affine(sim_2d, affine_2d, transform_key)
                        sim = sim_2d
                    projected_sims.append(sim)
                sim1, sim2 = projected_sims

            n_exact_tests += 1
            exact_test_start = time.time()
            try:
                # no overlap raises AttributeError; this linprog solve per pair is the dominant cost here
                result = _get_overlap_bboxes(
                    sim1,
                    sim2,
                    input_transform_key=transform_key,
                    output_transform_key=transform_key,
                )
                points = result['intersection'].intersections
                if points.shape[1] == 3 and force_2d:
                    # remove constant z coordinate
                    points = points[:, 1:]
                shape = _minimal_bb_vertices(points)
                if is_multi_z_shapes:
                    shape = [[shape_z_position] + list(element) for element in shape]
                shapes.append(shape)
                good_pairs.append(pair)
            except AttributeError:
                # ignore NoneType error if there is no overlap
                pass
            except ValueError as e:
                logging.exception(f'Error processing pair {pair}: {e}')
            finally:
                exact_test_time += time.time() - exact_test_start
    if n_exact_tests:
        logging.info(f'create_overlap_shapes: {n_exact_tests} exact intersection tests'
                     f' (of {len(pairs)} candidate pairs), {exact_test_time:.1f}s total'
                     f' ({1000 * exact_test_time / n_exact_tests:.1f}ms per test)')
    if n_clipped_shapes:
        logging.info(f'create_overlap_shapes: {n_clipped_shapes} shapes by 2D polygon clipping'
                     f' (of {len(pairs)} pairs), {time.time() - clip_start - exact_test_time:.1f}s')
    if n_fast_shapes:
        logging.info(f'create_overlap_shapes: {n_fast_shapes} shapes from broad-phase AABB '
                     f'intersection directly (no linprog)')
    return shapes, good_pairs


class NoOverlapError(ValueError):
    """Two images that do not overlap where an overlap is needed."""


def get_overlap_images(sim1, sim2, transform_key):
    sims = [sim1.squeeze(), sim2.squeeze()]
    # functionality copied from registration.register_pair_of_msims()
    spatial_dims = si_utils.get_spatial_dims_from_sim(sim1)
    
    # Adapt transforms to match image dimensions (handles 3D transform on 2D images)
    for sim in sims:
        original_transform = sim.attrs['transforms'][transform_key]
        adapted_transform = _adapt_transform_to_image_dims(sim, original_transform, transform_key)
        sim.attrs['transforms'][transform_key] = adapted_transform

    stack_props = [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in sims]
    _, intersection = mv_graph.get_overlap_between_pair_of_stack_props(*stack_props)
    if intersection is None:
        raise NoOverlapError('the images do not overlap')

    result = _get_overlap_bboxes(
        sims[0],
        sims[1],
        input_transform_key=transform_key
    )
    lowers, uppers = result['lowers'], result['uppers']

    reg_sims_spacing = [
        si_utils.get_spacing_from_sim(sim) for sim in sims
    ]

    tol = 1e-6
    overlaps_sims = [
        si_utils.sim_sel_coords(
            sim, sel_dict={
                # add spacing to include bounding pixels
                dim: slice(
                    lowers[isim][idim] - tol - reg_sims_spacing[isim][dim],
                    uppers[isim][idim] + tol + reg_sims_spacing[isim][dim],
                )
                for idim, dim in enumerate(spatial_dims)
            }
        ) for isim, sim in enumerate(sims)
    ]

    sims_pixel_space = sims_to_intrinsic_coord_system(
        overlaps_sims[0],
        overlaps_sims[1],
        transform_key=transform_key,
        overlap_bboxes=(lowers, uppers),
    )

    fixed_data = sims_pixel_space[0].data
    moving_data = sims_pixel_space[1].data

    fixed_data = xr.DataArray(fixed_data, dims=spatial_dims)
    moving_data = xr.DataArray(moving_data, dims=spatial_dims)

    return fixed_data, moving_data, sims_pixel_space


def affine_from_intrinsic_affine(affine, sims_pixel_space, transform_key):
    affine_phys = get_affine_from_intrinsic_affine(
        data_affine=affine,
        sim_fixed=sims_pixel_space[0],
        sim_moving=sims_pixel_space[1],
        transform_key_fixed=transform_key,
        transform_key_moving=transform_key,
    )
    return affine_phys


def combine_transforms(transforms):
    combined_transform = None
    for transform in transforms:
        if combined_transform is None:
            combined_transform = transform
        else:
            combined_transform = np.dot(transform, combined_transform)
    return combined_transform



def squeeze_sim_transform_time(sim, transform_key):
    affine = si_utils.get_affine_from_sim(sim, transform_key)
    if 't' in affine.dims:
        affine = affine.isel(t=0)
        si_utils.set_sim_affine(sim, affine, transform_key)
    return sim



def promote_sim_to_3d(sim, z_position):
    # a level with no native 'z' gets a size-1 one at z_position, and every transform is widened to 3D
    if 'z' not in sim.dims:
        sim = sim.expand_dims({'z': [z_position]}, axis=-3)
    for transform_key in si_utils.get_tranform_keys_from_sim(sim):
        transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        transform_3d = widen_xaffine_to_3d(transform)
        if transform_3d is not transform:
            si_utils.set_sim_affine(sim, transform_3d, transform_key=transform_key)
    return sim


def widen_xaffine_to_3d(transform):
    """A 2D affine embedded in a 3D identity, or unchanged if already 3D.

    Placed by index into a numpy identity, not by label with .loc, whose xarray alignment is costly per pyramid
    level; the indices come from the coordinate labels, so the result is the same.
    """
    if 4 in transform.shape:
        return transform
    return param_utils.affine_to_xaffine(widened_affine_matrix(transform))


def widened_affine_matrix(transform):
    """widen_xaffine_to_3d()'s 4x4 matrix as a plain array, for a caller that only reads it."""
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    if 4 in transform.shape:
        return np.asarray(transform, dtype=float)
    labels_3d = ['z', 'y', 'x', '1']
    axes = [labels_3d.index(str(label)) for label in transform.coords['x_in'].values]
    widened = np.eye(len(labels_3d))
    widened[np.ix_(axes, axes)] = np.asarray(transform)
    return widened


def msim_is_already_3d(msim):
    """True when every level of `msim` already has a 'z' dim and a 3D transform, so promoting it would change
    nothing - a msim can reach make_msims_3d() more than once on the preview path."""
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        dataset = msim[scale_key].ds
        if 'z' not in dataset['image'].dims:
            return False
        # transforms are the data variables besides 'image': 3x3 while 2D, 4x4 once widened
        transforms = [name for name in dataset.data_vars if name != 'image']
        if not transforms:
            return False
        if any(4 not in dataset[name].shape for name in transforms):
            return False
    return True


def make_msims_3d(msims, z_scale=None, positions=None):
    # promote_sim_to_3d() at every level, skipping an msim already in that form (msim_is_already_3d)
    if not z_scale:
        z_scale = 1
    new_msims = []
    for index, msim in enumerate(msims):
        if msim_is_already_3d(msim):
            new_msims.append(msim)
        else:
            z_position = positions[index].get('z', index * z_scale) if positions else index * z_scale
            new_msims.append(_promote_msim_to_3d(msim, z_position))
    return new_msims


def _promote_msim_to_3d(msim, z_position):
    """promote_sim_to_3d() on each level, done on the levels' own datasets: the same tree, without a
    sim <-> msim round trip per level."""
    nodes = {}
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        dataset = msim[scale_key].ds
        image = dataset['image']
        if 'z' not in image.dims:
            image = image.expand_dims({'z': [z_position]}, axis=-3)
        variables = {'image': image}
        # the transforms are the level's other data variables
        variables.update({name: widen_xaffine_to_3d(dataset[name]) for name in dataset.data_vars if name != 'image'})
        nodes[scale_key] = xr.Dataset(variables)
    return DataTree.from_dict(nodes)


def unify_msim_channels(msims):
    """msims whose 'c' labels all match the first's, as fusion selects every source by those. A single-channel
    source named differently (a file with no channel name gets 'channel N') takes the label; several is an error."""
    labels = list(get_msim_image0(msims[0]).coords['c'].values) if msims else []
    new_msims = []
    for msim in msims:
        msim_labels = list(get_msim_image0(msim).coords['c'].values)
        if sorted(msim_labels) == sorted(labels):
            new_msims.append(msim)
        elif len(msim_labels) == len(labels) == 1:
            new_msims.append(_relabel_msim_channels(msim, labels))
        else:
            raise ValueError(f'Sources have different channels: {msim_labels} and {labels}')
    relabelled = sum(new is not old for new, old in zip(new_msims, msims))
    if relabelled:
        logging.warning(f'Fusion: {relabelled} source(s) with another channel name given {str(labels[0])!r}')
    return new_msims


def _relabel_msim_channels(msim, labels):
    nodes = {scale_key: msim[scale_key].ds.assign_coords(c=list(labels))
             for scale_key in msi_utils.get_sorted_scale_keys(msim)}
    return DataTree.from_dict(nodes)


def msim_is_2d(msim):
    """Whether `msim` is already 2D - no z dim and a 3x3 affine on every transform - so make_msims_2d() need not
    rebuild its whole DataTree."""
    sim = msi_utils.get_sim_from_msim(msim, scale='scale0')
    if 'z' in sim.dims:
        return False
    return all(3 in si_utils.get_affine_from_sim(sim, transform_key=transform_key).shape
               for transform_key in si_utils.get_tranform_keys_from_sim(sim))


def make_msims_2d(msims):
    new_msims = []
    for msim in msims:
        if msim_is_2d(msim):
            new_msims.append(msim)
            continue
        def level_func(sim, scale_key):
            if 'z' in sim.dims:
                sim = sim.squeeze('z')
            for transform_key in si_utils.get_tranform_keys_from_sim(sim):
                transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
                if 3 not in transform.shape:
                    has_t = 't' in transform.dims
                    if has_t:
                        transform = transform.sel(t=0)
                    transform = transform[:3, :3]
                    if has_t:
                        transform.loc[{dim: transform.coords[dim] for dim in transform.sel(t=0).dims}] = transform.sel(t=0)
                    si_utils.set_sim_affine(sim, transform, transform_key=transform_key)
            return sim

        new_msims.append(map_msim_levels(msim, level_func))
    return new_msims


def extract_sims_from_msims(msims, sources, transform_key, target_scale):
    """One working-resolution sim per source msim: the native level nearest `target_scale`, resized when none
    matches exactly. For plain scale0, use msi_utils.get_sim_from_msim(msim, scale='scale0').
    """
    sims = []
    for source, msim in zip(sources, msims):
        level, rescale, scale = get_level_from_scale(source, target_scale)
        sim = msi_utils.get_sim_from_msim(msim, scale=f'scale{level}')
        if any(value != 1 for value in rescale.values()):
            data = sim.data
            new_shape = [max(int(size / rescale.get(dim, 1)), 1) for dim, size in zip(sim.dims, data.shape)]
            data = resize(data, new_shape, preserve_range=True).astype(data.dtype)
            translation = si_utils.get_origin_from_sim(sim)
            affine = si_utils.get_affine_from_sim(sim, transform_key)
            channel_labels = list(sim.coords['c'].values) if 'c' in sim.coords else None
            # `scale` covers only the source's own pixel-size dims, but get_sim_from_array needs every spatial
            # dim (e.g. 'z' for a 2D source in a 3D output_order)
            full_scale = si_utils.get_spacing_from_sim(sim)
            if scale:
                full_scale.update(scale)
            sim = si_utils.get_sim_from_array(
                data, dims=list(sim.dims), scale=full_scale, translation=translation,
                affine=affine, transform_key=transform_key, c_coords=channel_labels)
            # native levels are chunked at msim creation; a resized array is new and needs its own check
            sim = rechunk_if_monolithic(sim, default_chunk_size)
        sims.append(sim)
    return sims


def drop_finest_msim_level(msims, levels=1):
    """Each msim minus its `levels` finest levels, as a shallower sub-pyramid. Returns (msims, changed); a
    single-level msim is returned untouched, and `changed` is False when none could be reduced.

    Several levels at once rather than a pass each, as every pass rebuilds all msims.
    """
    result = []
    changed = False
    for msim in msims:
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        dropped = min(levels, len(scale_keys) - 1)
        if dropped > 0:
            changed = True
            result.append(DataTree.from_dict({f'scale{index}': msim[scale_key].ds
                                              for index, scale_key
                                              in enumerate(scale_keys[dropped:])}))
        else:
            result.append(msim)
    return result, changed


def coarsen_msims(msims, factor=2):
    """Each msim as a single coarser level, by striding its coarsest level in x/y - for msims with no coarser level
    left to drop to.

    Strided rather than mean-downsampled, as it only feeds a preview. x/y only: the output's z extent comes from
    the sources' heights, so striding z would only lose sections. Returns (msims, changed); `changed` is False
    when nothing got smaller.
    """
    result = []
    changed = False
    for msim in msims:
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        dataset = msim[scale_keys[-1]].ds
        image = dataset['image']
        slicing = tuple(slice(None, None, factor if dim in 'xy' else 1) for dim in image.dims)
        coarser = image[slicing]
        if coarser.shape == image.shape:
            result.append(msim)
            continue
        changed = True
        # transforms are resolution-independent and carry over; striding the coordinates keeps the origin and
        # scales the spacing
        transforms = {name: dataset[name] for name in dataset.data_vars if name != 'image'}
        result.append(DataTree.from_dict({'scale0': xr.Dataset({'image': coarser, **transforms})}))
    return result, changed

def estimate_fused_size(msims, transform_key, output_spacing_method=None, z_scale=None, z_positions=None):
    """(bytes, output properties) of fusing `msims`: the output shape times the source dtype. Metadata only, so
    cheap enough to ask before committing to a fusion.
    """
    properties = calc_output_properties(msims, transform_key,
                                        output_spacing_method=output_spacing_method,
                                        z_scale=z_scale, z_positions=z_positions)
    itemsize = get_msim_image0(msims[0]).dtype.itemsize
    return int(np.prod([int(size) for size in properties['shape'].values()]) * itemsize), properties


def msims_stack_props(msims, transform_key):
    """Each msim's finest level's stack properties, at its first t/c as mv_graph takes them: one plane, not the stack."""
    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])
    if len(nsdims):
        sims = [si_utils.sim_sel_coords(sim, {nsdim: sim.coords[nsdim][0] for nsdim in nsdims}) for sim in sims]
    return [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in sims]


def build_view_adjacency_graph(msims, transform_key, pairs, overlap_tolerance=None):
    """mv_graph.build_view_adjacency_graph_from_msims() for known pairs: the same nodes and
    'stack_props', and an edge with its 'overlap' for each pair that overlaps.

    That solves a linear program per pair; boxes aligned with the axes (every source before registration) overlap
    exactly in their bounding boxes' intersection, so all pairs are measured at once. Any other box defers the
    whole graph to multiview_stitcher.
    """
    stack_props = msims_stack_props(msims, transform_key)
    if overlap_tolerance is not None:
        stack_props = [si_utils.extend_stack_props(props, overlap_tolerance) for props in stack_props]
    vertices = [mv_graph.get_vertices_from_stack_props(props) for props in stack_props]
    mins = np.array([points.min(axis=0) for points in vertices])
    maxs = np.array([points.max(axis=0) for points in vertices])
    axis_aligned = all(np.all(np.isclose(points, low) | np.isclose(points, high))
                       for points, low, high in zip(vertices, mins, maxs))
    if not axis_aligned:
        return mv_graph.build_view_adjacency_graph_from_msims(
            msims, transform_key=transform_key, pairs=pairs, overlap_tolerance=overlap_tolerance)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(msims)))
    nx.set_node_attributes(graph, dict(enumerate(stack_props)), name='stack_props')
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    extents = np.minimum(maxs[pairs[:, 0]], maxs[pairs[:, 1]]) - np.maximum(mins[pairs[:, 0]], mins[pairs[:, 1]])
    overlaps = np.prod(np.clip(extents, 0, None), axis=1)
    # as there: only boxes sharing more than a face are neighbours
    for index, ((first, second), overlap) in enumerate(zip(pairs, overlaps)):
        if index % 100_000 == 0:
            raise_if_cancelled()
        if overlap > 0:
            graph.add_edge(int(first), int(second), overlap=float(overlap))
    return graph


def build_pairs_graph(stack_props, pairs, overlaps=None):
    """The view adjacency graph for already known pairs, without multiview_stitcher's linear program per pair.

    Nodes carry 'stack_props'; an edge gets an 'overlap' weight only from `overlaps`. Downstream reads it as
    .get('overlap', 1.0), so an edge without one is weighted 1.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(stack_props)))
    nx.set_node_attributes(graph, dict(enumerate(stack_props)), name='stack_props')
    for pair in pairs:
        overlap = (overlaps or {}).get(tuple(pair))
        graph.add_edge(*pair, **({'overlap': overlap} if overlap is not None else {}))
    return graph


def reduce_msims_to_fused_size(msims, transform_key, max_bytes=default_preview_max_bytes,
                               output_spacing_method=None, z_scale=None, max_steps=16,
                               label='preview', z_positions=None):
    """`msims` stepped to coarser levels until fusing them would produce at most max_bytes, keeping an on-screen
    preview from fusing an arbitrarily large stack.

    Bounds the fused size itself, as preview_scale picks levels per source pyramid and so still scales with the
    dataset. Stops when nothing can be reduced further. z_positions: 2D sources at several heights, sized as if
    promoted to 3D (see calc_output_properties) without promoting them.
    """
    size, _ = estimate_fused_size(msims, transform_key, output_spacing_method, z_scale, z_positions)
    if size <= max_bytes:
        return msims
    original_size = size
    reduced = msims
    for _ in range(max_steps):
        # drop as many levels as the excess needs (each quarters the fused size), as every round rebuilds all
        # msims; the estimate below has the last word, since a short pyramid reduces by less
        steps = max(int(np.ceil(np.log(size / max_bytes) / np.log(4))), 1)
        # a real coarser level where one exists (free - it is already in the file), otherwise
        # strided down for display only
        coarser, changed = drop_finest_msim_level(reduced, levels=steps)
        if not changed:
            coarser, changed = coarsen_msims(reduced, factor=2 ** steps)
        if not changed:
            logging.warning(
                f'{label}: fusing {print_hbytes(size)}, over the'
                f' {print_hbytes(max_bytes)} budget - cannot be reduced any further')
            return reduced
        reduced = coarser
        size, _ = estimate_fused_size(reduced, transform_key, output_spacing_method, z_scale, z_positions)
        if size <= max_bytes:
            break
    logging.info(f'{label}: reduced to {print_hbytes(size)} '
                 f'(fusing at the requested resolution would have been {print_hbytes(original_size)},'
                 f' over the {print_hbytes(max_bytes)} budget)')
    return reduced

def select_msim_subpyramid_at_scale(msims, sources, target_scale, shortfall_warn_factor=4):
    """Per source, every native pyramid level from the nearest match to `target_scale` to the coarsest, as a
    sub-pyramid msim - pure slicing, no resize.

    A pyramid that does not reach `target_scale` silently makes fusion cost the square of the shortfall more, so a
    shortfall of at least shortfall_warn_factor is logged, measured only over dims the pyramid reduces. A msim
    built already truncated is sliced from where it starts, not from the source's level 0.
    """
    result = []
    residuals = []
    for source, msim in zip(sources, msims):
        level, residual, _ = get_level_from_scale(source, target_scale)
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        # however many native levels this msim is missing is where it already starts
        from_level = max(len(source.scale_factors) - len(scale_keys), 0) if source.scale_factors else 0
        level = max(level - from_level, 0)
        # only dims this pyramid reduces: one constant across levels always keeps the full target factor
        coarsest = source.scale_factors[-1] if source.scale_factors else {}
        reducible = [value for dim, value in residual.items() if coarsest.get(dim, 1) > 1]
        residuals.append(max(reducible) if reducible else 1)
        result.append(DataTree.from_dict({f'scale{i}': msim[scale_key].ds
                                          for i, scale_key in enumerate(scale_keys[level:])}))
    worst = max(residuals) if residuals else 1
    if worst >= shortfall_warn_factor:
        short = sum(1 for residual in residuals if residual >= shortfall_warn_factor)
        logging.warning(
            f'Preview scale {target_scale} not reachable for {short}/{len(residuals)} sources:'
            f' coarsest available level is up to {worst:.3g}x finer than requested, so the'
            f' preview fuses up to {worst ** 2:.3g}x more pixels per plane than intended.'
            f' Sources lacking coarse pyramid levels (e.g. a single-resolution OME-Zarr) are the'
            f' usual cause - re-converting them with a full pyramid restores the intended cost.')
    return result


def copy_transforms_to_msims(sources, target_msims, transform_key):
    """Copy `transform_key`'s affine from each of `sources` (sims or msims) onto every scale of the matching
    `target_msims`, adapted to the targets' dims.
    """
    dims = list(si_utils.get_origin_from_sim(get_msim_image0(target_msims[0])).keys())
    for source, target_msim in zip(sources, target_msims):
        if isinstance(source, DataTree):
            transform = msi_utils.get_transform_from_msim(source, transform_key)
        else:
            transform = si_utils.get_affine_from_sim(source, transform_key=transform_key)
        transform_dims = np.array(transform.coords['x_in'])
        if len(transform_dims) == 3 and len(dims) == 3:
            transform = widen_xaffine_to_3d(transform)
        elif len(transform_dims) - 1 != len(dims):
            new_transform = param_utils.identity_transform(ndim=len(dims))
            common_dims = [dim for dim in transform.dims if dim in new_transform.dims and dim != 't']
            if len(common_dims) > 0:
                if 't' in transform.dims:
                    transform_slice = transform.sel(t=0)
                else:
                    transform_slice = transform
                new_transform.loc[{dim: transform_slice.coords[dim] for dim in common_dims}] = transform_slice
            transform = new_transform
        set_msim_affine(target_msim, transform, transform_key=transform_key)


def set_msim_affine(msim, xaffine, transform_key, base_transform_key=None):
    """msi_utils.set_affine_transform, assigning one dataset per level rather than a DataTree item, whose alignment
    is costly per source."""
    if base_transform_key is not None:
        xaffine = param_utils.rebase_affine(xaffine,
                                            msi_utils.get_transform_from_msim(msim, transform_key=base_transform_key))
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        node = msim[scale_key]
        node.dataset = node.to_dataset(inherit=False).assign({transform_key: xaffine})


