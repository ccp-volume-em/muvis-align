"""Fusion to OME-Zarr level by level and block by block, only the blocks some source reaches, each from only the
sources reaching it: multiview_stitcher runs Python over every source it gives a block."""
from contextlib import contextmanager
import copy
from dataclasses import asdict
import inspect
import logging
import os
import shutil
import time

import numpy as np
import ngff_zarr
import zarr
import multiview_stitcher.fusion._core as fusion_core
from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.constants import default_export_fusion_chunk_bytes, fusion_stack_arrays
from muvis_align.image.util import coarsest_level_within, msim_level_spacings, sim_views
from muvis_align.util import get_process_memory, print_hbytes, print_memory_usage, print_significants


def source_bounds(sims, transform_key, output_stack_properties, interpolation_order=1):
    """view_bounds() of sims, each at its own geometry and transform."""
    return view_bounds(sim_views(sims, transform_key), output_stack_properties, interpolation_order)


def _grid_aligned_dims(stacks, matrices, output_stack_properties, sdims, tol=1e-6):
    """multiview_stitcher's _get_grid_aligned_translation_dims on plain arrays: the dims every view only translates
    in, at the output's spacing, by whole pixels."""
    ndim = len(sdims)
    linear = matrices[:, :ndim, :ndim]
    aligned = []
    for axis, dim in enumerate(sdims):
        others = [other for other in range(ndim) if other != axis]
        spacing = np.array([stack['spacing'][dim] for stack in stacks])
        offsets = (output_stack_properties['origin'][dim] - matrices[:, axis, ndim]
                   - np.array([stack['origin'][dim] for stack in stacks]))
        pixel_offsets = offsets / np.where(spacing == 0, np.nan, spacing)
        if (np.allclose(linear[:, axis, axis], 1, atol=tol)
                and np.allclose(linear[:, axis, others], 0, atol=tol) and np.allclose(linear[:, others, axis], 0, atol=tol)
                and np.allclose(spacing, output_stack_properties['spacing'][dim], atol=tol)
                and np.all(np.isclose(pixel_offsets, np.round(pixel_offsets), atol=tol))):
            aligned.append(dim)
    return aligned


def view_bounds(views, output_stack_properties, interpolation_order=1):
    """Per view (stack properties, affine matrix) and spatial dim, the (low, high) physical bounds it reaches in the
    output, padded for interpolation as multiview_stitcher does, except in grid-aligned or single-plane dims."""
    stacks = [stack for stack, _ in views]
    sdims = list(stacks[0]['shape'])
    ndim = len(sdims)
    matrices = np.stack([matrix for _, matrix in views]).astype(float)
    shape = np.array([[stack['shape'][dim] for dim in sdims] for stack in stacks], dtype=float)
    spacing = np.array([[stack['spacing'][dim] for dim in sdims] for stack in stacks], dtype=float)
    origin = np.array([[stack['origin'][dim] for dim in sdims] for stack in stacks], dtype=float)
    corners = np.array(list(np.ndindex(*[2] * ndim)), dtype=float)
    local = corners[None] * ((shape - 1) * spacing)[:, None] + origin[:, None]
    vertices = np.einsum('nij,nkj->nki', matrices[:, :ndim, :ndim], local) + matrices[:, None, :ndim, ndim]
    aligned = _grid_aligned_dims(stacks, matrices, output_stack_properties, sdims)
    # a single plane's spacing is a placeholder 1.0: padded by it, a section reaches its neighbours' blocks
    padding = np.where((shape == 1) | np.isin(sdims, aligned)[None], 0.0, interpolation_order * spacing)
    return np.stack([vertices.min(axis=1) - padding, vertices.max(axis=1) + padding], axis=-1)


def _block_overlaps(bounds, origin, spacing, chunk, nblocks):
    """Per block along one dim and source, whether the source's bounds meet the block (widened by half a pixel)."""
    low = origin + (np.arange(nblocks) * chunk - 0.5) * spacing
    high = low + chunk * spacing
    return (low[:, None] <= bounds[None, :, 1]) & (bounds[None, :, 0] <= high[:, None])


def budget_chunksize(bounds, output_stack_properties, output_chunksize, sdims, budget_bytes, bytes_per_voxel,
                     min_xy_chunk=256):
    """The largest yx chunk, shrinking from output_chunksize, whose busiest block fits budget_bytes: a block holds
    every source it meets transformed to its full size, counted from the sources' bounds."""
    chunk = {dim: int(size) for dim, size in output_chunksize.items()}
    xy_dims = [dim for dim in ('y', 'x') if dim in chunk]
    while True:
        overlaps = {}
        for axis, dim in enumerate(sdims):
            nblocks = int(np.ceil(output_stack_properties['shape'][dim] / chunk[dim]))
            overlaps[dim] = _block_overlaps(bounds[:, axis], output_stack_properties['origin'][dim],
                                            output_stack_properties['spacing'][dim], chunk[dim], nblocks)
        busiest = 0
        for z_overlap in (overlaps['z'] if 'z' in overlaps else [np.ones(len(bounds), dtype=bool)]):
            sources = np.flatnonzero(z_overlap)
            if len(sources) and len(xy_dims) == 2:
                counts = overlaps['y'][:, sources].astype(np.int32) @ overlaps['x'][:, sources].T.astype(np.int32)
                busiest = max(busiest, int(counts.max()))
            elif len(sources):
                busiest = max(busiest, len(sources))
        voxels = int(np.prod([chunk[dim] for dim in sdims]))
        needed = busiest * voxels * bytes_per_voxel
        if needed <= budget_bytes or all(chunk[dim] <= min_xy_chunk for dim in xy_dims):
            return chunk
        # shrink by just the factor needed, then recheck: every extra block has a fixed overhead
        factor = min(0.95, np.sqrt(budget_bytes / needed))
        for dim in xy_dims:
            chunk[dim] = max(min_xy_chunk, int(chunk[dim] * factor) // 64 * 64)


def block_sources(bounds, output_stack_properties, output_chunksize, sdims):
    """The blocks some source reaches, grouped by exactly which: {source indices: [spatial block indices]}."""
    overlaps = [_block_overlaps(bounds[:, axis], output_stack_properties['origin'][dim],
                                output_stack_properties['spacing'][dim], int(output_chunksize[dim]),
                                int(np.ceil(output_stack_properties['shape'][dim] / int(output_chunksize[dim]))))
                for axis, dim in enumerate(sdims)]
    groups = {}
    # the first dim's blocks one at a time keeps the remaining product small
    for first_block, first_overlap in enumerate(overlaps[0]):
        candidates = np.flatnonzero(first_overlap)
        if len(candidates):
            rest = [overlap[:, candidates] for overlap in overlaps[1:]]
            for rest_blocks in np.ndindex(*[len(overlap) for overlap in rest]):
                reach = np.ones(len(candidates), dtype=bool)
                for overlap, block in zip(rest, rest_blocks):
                    reach &= overlap[block]
                if reach.any():
                    groups.setdefault(tuple(candidates[reach].tolist()), []).append((first_block,) + rest_blocks)
    return groups


def _view_ranks(params, group_params, group_ranks):
    """Each fused view's rank: multiview_stitcher passes a block only the views it finds there, in order - a
    subsequence of the block's sources, told apart by their affines."""
    ranks, start = [], 0
    for param in params:
        param = np.asarray(param).squeeze()
        for index in range(start, len(group_params)):
            # a block fused in 2D gets the affines without the singleton z: compare their trailing (y, x) part
            size = min(param.shape[-1], group_params[index].shape[-1])
            if np.allclose(param[-size:, -size:], group_params[index][-size:, -size:]):
                ranks.append(group_ranks[index])
                start = index + 1
                break
        else:
            ranks.append(min(group_ranks))
    return np.asarray(ranks, dtype=float)


def prioritise_finer_views(fusion_func, group_params, group_ranks):
    """fusion_func over only the finest views at each pixel: where a tile covers it, the overview under it is left
    out rather than averaged in. Views of equal rank are fused as fusion_func would."""
    wanted = inspect.signature(fusion_func).parameters

    def fusion(**kwargs):
        views = kwargs['transformed_views']
        ranks = _view_ranks(kwargs.pop('params'), group_params, group_ranks)
        ranks = ranks.reshape((-1,) + (1,) * (views.ndim - 1))
        valid = ~np.isnan(views)
        finest = np.min(np.where(valid, ranks, np.inf), axis=0)
        keep = valid & (ranks == finest)
        kwargs['transformed_views'] = np.where(keep, views, np.nan)
        if kwargs.get('blending_weights') is not None:
            weights = np.where(keep, kwargs['blending_weights'], 0)
            total = weights.sum(axis=0)
            kwargs['blending_weights'] = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
        return fusion_func(**{name: value for name, value in kwargs.items() if name in wanted})

    # multiview_stitcher passes what the signature names: fusion_func's own arguments, and the views' affines
    parameters = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in wanted]
    if 'params' not in wanted:
        parameters.append(inspect.Parameter('params', inspect.Parameter.KEYWORD_ONLY))
    fusion.__signature__ = inspect.Signature(parameters)
    return fusion


def fuse_into_zarr_array(sims, store_url, transform_key, output_stack_properties, output_chunksize, fusion_func=None,
                         creation_kwargs=None, batch_options=None, interpolation_order=1, progress=None, share=0,
                         ranks=None, groups=None, log_batches=False):
    """Fuse `sims` into a new zarr array at store_url: only the blocks some source reaches, each from only the sources
    reaching it. With `ranks` (lower first, e.g. pixel sizes), only the lowest-ranked source at each pixel is fused.
    The blocks move `progress` by `share` in all; `log_batches` logs each batch's memory. Returns dims and
    properties."""
    batch_options = batch_options or {}
    dims = list(sims[0].dims)
    sdims = list(si_utils.get_spatial_dims_from_sim(sims[0]))
    block_axes = [dims.index(dim) for dim in sdims]
    if groups is None:
        bounds = source_bounds(sims, transform_key, output_stack_properties, interpolation_order)
        groups = block_sources(bounds, output_stack_properties, output_chunksize, sdims)

    def prepare(sources, create_output):
        fuse_kwargs = {'images': [sims[index] for index in sources], 'transform_key': transform_key,
                       'output_chunksize': output_chunksize,
                       'output_stack_properties': copy.deepcopy(output_stack_properties),
                       'interpolation_order': interpolation_order}
        block_func = fusion_func
        if ranks is not None and len({ranks[index] for index in sources}) > 1:
            group_params = [np.asarray(si_utils.get_affine_from_sim(sims[index], transform_key=transform_key)).squeeze()
                            for index in sources]
            block_func = prioritise_finer_views(fusion_func or fusion_core.weighted_average_fusion, group_params,
                                                [ranks[index] for index in sources])
        # None would replace multiview_stitcher's own default
        if block_func is not None:
            fuse_kwargs['fusion_func'] = block_func
        return fusion_core.prepare_block_fusion(store_url, fuse_kwargs=fuse_kwargs,
                                                zarr_array_creation_kwargs=creation_kwargs,
                                                create_output=create_output, verbose=False)

    # the store is created first, from any source: a block none reaches is never written
    info = prepare((0,), create_output=True)
    nblocks = info['nblocks']
    nonspatial_axes = [axis for axis in range(len(dims)) if axis not in block_axes]
    funcs = {}
    block_stats = {}
    for sources, spatial_blocks in groups.items():
        func = prepare(sources, create_output=False)['func']
        stats = (len(sources), sum(sims[index].data.nbytes for index in sources))
        for nonspatial_block in np.ndindex(*[nblocks[axis] for axis in nonspatial_axes]):
            for spatial_block in spatial_blocks:
                block_id = [0] * len(dims)
                for axis, block in zip(nonspatial_axes + block_axes, nonspatial_block + spatial_block):
                    block_id[axis] = block
                funcs[tuple(block_id)] = func
                block_stats[tuple(block_id)] = stats

    def fuse_block(block_id):
        return funcs[block_id](block_id)

    # what the memory budget counts for a source in a block (budget_chunksize)
    source_block_bytes = int(np.prod([int(output_chunksize[dim]) for dim in sdims])) * 4 * fusion_stack_arrays
    _run_blocks(fuse_block, sorted(funcs), batch_options, progress, share,
                block_stats=block_stats if log_batches else None, source_block_bytes=source_block_bytes)
    return dims, info['output_stack_properties']


def _run_blocks(block_func, block_ids, batch_options, progress=None, share=0, block_stats=None,
                source_block_bytes=0):
    """block_func on each block id, batched through batch_options' batch_func, moving `progress` by `share` over
    them: only these blocks count, as the ones no source reaches cost nothing. With `block_stats` ({block id:
    (sources, their level's bytes)}), each batch is logged with the memory it was budgeted and what it reached."""
    batch_func, n_batch = batch_options.get('batch_func'), batch_options.get('n_batch', 1)
    nbatches = -(-len(block_ids) // n_batch)
    batch_rss = [0]

    def logged_block(block_id):
        block_func(block_id)
        # read as each block ends, while the rest of its batch still runs
        batch_rss[0] = max(batch_rss[0], get_process_memory()[0] or 0)

    run_block = block_func if block_stats is None else logged_block
    for start in range(0, len(block_ids), n_batch):
        batch = block_ids[start:start + n_batch]
        batch_start, batch_rss[0] = time.perf_counter(), 0
        if batch_func is None:
            for block_id in batch:
                run_block(block_id)
        else:
            batch_func(run_block, batch, **(batch_options.get('batch_func_kwargs') or {}))
        if block_stats is not None:
            sources = [block_stats[block_id][0] for block_id in batch]
            source_bytes = [block_stats[block_id][1] for block_id in batch]
            logging.info(f'Fusion batch {start // n_batch + 1}/{nbatches}: {len(batch)} blocks, {sum(sources)} sources'
                         f' (max {max(sources)} a block), budgeted {print_hbytes(sum(sources) * source_block_bytes)},'
                         f' source levels {print_hbytes(sum(source_bytes))}'
                         f' (max {print_hbytes(max(source_bytes))} a block), {time.perf_counter() - batch_start:.1f}s,'
                         f' max rss at a block end {print_hbytes(batch_rss[0])}{print_memory_usage()}')
        if progress is not None:
            progress.update(share * len(batch) / len(block_ids))
    if progress is not None and not block_ids:
        progress.update(share)


def average_into_zarr_array(parent_url, store_url, factors, chunk, spatial_blocks, creation_kwargs=None,
                            batch_options=None, progress=None, share=0):
    """A zarr array averaging the one at parent_url by whole `factors` per trailing spatial dim, writing only the
    `chunk` blocks in spatial_blocks: a pyramid level without z averaged across sections or empty blocks computed."""
    batch_options = batch_options or {}
    parent = zarr.open_array(parent_url, mode='r')
    sdims = [dim for dim in 'zyx' if dim in chunk]
    nonspatial = parent.ndim - len(sdims)
    axis_factors = [1] * nonspatial + [int(factors[dim]) for dim in sdims]
    shape = [size // factor for size, factor in zip(parent.shape, axis_factors)]
    chunks = [1] * nonspatial + [int(chunk[dim]) for dim in sdims]
    output = zarr.create(shape=shape, chunks=chunks, dtype=parent.dtype, store=store_url, overwrite=True,
                         **(creation_kwargs or {}))

    def average_block(block_id):
        target = tuple(slice(block * size, min((block + 1) * size, total))
                       for block, size, total in zip(block_id, chunks, shape))
        data = parent[tuple(slice(piece.start * factor, piece.stop * factor)
                            for piece, factor in zip(target, axis_factors))]
        grouped = data.reshape([size for length, factor in zip(data.shape, axis_factors)
                                for size in (length // factor, factor)])
        output[target] = grouped.mean(axis=tuple(range(1, grouped.ndim, 2))).astype(parent.dtype)

    block_ids = sorted(tuple(nonspatial_block) + tuple(spatial_block)
                       for nonspatial_block in np.ndindex(*shape[:nonspatial]) for spatial_block in spatial_blocks)
    _run_blocks(average_block, block_ids, batch_options, progress, share)


def storage_ngff_version(ngff_version):
    # 0.6 changed the metadata, not the arrays: they are stored as 0.5's, which multiview_stitcher can create
    return '0.5' if str(ngff_version).startswith('0.6') else ngff_version


def _zarr_options(zarr_options):
    zarr_options = zarr_options or {}
    ngff_version = zarr_options.get('ngff_version', '0.4')
    creation_kwargs = ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
        storage_ngff_version(ngff_version), zarr_options.get('zarr_array_creation_kwargs'))
    return ngff_version, creation_kwargs


def write_multiscales_metadata(group, axes, datasets, ngff_version):
    """multiview_stitcher's write_multiscales_metadata, plus OME-Zarr 0.6 (which it lacks) converted by ngff-zarr."""
    if not str(ngff_version).startswith('0.6'):
        ngff_utils.write_multiscales_metadata(group, axes=axes, datasets=datasets, ngff_version=ngff_version)
        return
    metadata = ngff_zarr.Metadata(
        axes=[ngff_zarr.Axis(**dict(axis)) for axis in axes],
        datasets=[ngff_zarr.Dataset(path=dataset['path'],
                                    coordinateTransformations=[_ngff_transform(transform)
                                                               for transform in dataset['coordinateTransformations']])
                  for dataset in datasets],
        coordinateTransformations=None, name=group.name)
    group.attrs['ome'] = {'version': '0.6', 'multiscales': [_without_none(asdict(metadata.to_version('0.6')))]}


def _ngff_transform(transform):
    if transform['type'] == 'scale':
        return ngff_zarr.Scale(scale=list(transform['scale']))
    return ngff_zarr.Translation(translation=list(transform['translation']))


def _without_none(value):
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _remove_existing(output_zarr_url, zarr_options):
    if (zarr_options or {}).get('overwrite', True) and os.path.exists(output_zarr_url):
        shutil.rmtree(output_zarr_url)


def source_level_sizes(source_spacings):
    """{source pixel size: the pixel size of the level it is first fused into}: a size under sqrt(2) above the finest
    of its group joins that group's level, upsampled by less than that, rather than a few odd sources (overview
    sections imaged coarser) adding a level of their own."""
    level_sizes = {}
    start = None
    for size in sorted(set(source_spacings)):
        if start is None or size >= start * np.sqrt(2):
            start = size
        level_sizes[size] = start
    return level_sizes


def native_level_spacings(source_spacings, finest_shape, min_shape=100, tolerance=0.05):
    """Output level pixel sizes: the finest source's, doubling, with a level at each coarser group of source pixel
    sizes (source_level_sizes), until the largest dim would fall to min_shape - every group gets a level, or what
    only it covers would be in none."""
    sizes = sorted(set(source_level_sizes(source_spacings).values()))
    levels, pending = [sizes[0]], sizes[1:]
    largest = max(finest_shape)
    while pending or largest * levels[0] / (levels[-1] * 2) > min_shape:
        doubled = levels[-1] * 2
        if pending and doubled >= pending[0] * (1 - tolerance):
            # a step under sqrt(2) to a source size replaces the doubled level before it, never another source size
            if len(levels) > 1 and levels[-1] not in sizes and pending[0] < levels[-1] * np.sqrt(2):
                levels[-1] = pending.pop(0)
            else:
                levels.append(pending.pop(0))
        else:
            levels.append(doubled)
    return levels


def native_level_stack_properties(level0_properties, level_spacings, scaled_dims):
    """Each level's output stack: level 0's extent at the level's pixel size in `scaled_dims` (other dims kept), its
    origin by OME-Zarr's pixel-centre convention."""
    spacing0 = level0_properties['spacing']
    levels = []
    for level_spacing in level_spacings:
        factors = {dim: level_spacing / level_spacings[0] if dim in scaled_dims else 1.0 for dim in spacing0}
        levels.append({
            'spacing': {dim: spacing0[dim] * factors[dim] for dim in spacing0},
            'origin': {dim: level0_properties['origin'][dim] + (factors[dim] - 1) * spacing0[dim] / 2
                       for dim in spacing0},
            'shape': {dim: max(1, int(level0_properties['shape'][dim] // factors[dim])) for dim in spacing0},
            'factors': factors,
        })
    return levels


def export_levels(source_spacings, level0_properties, scaled_dims, native=True, tolerance=0.05):
    """(pixel size, its native_level_stack_properties, indices of the sources fused into it) per level of an export.
    Native: the levels of native_level_spacings, each from the sources whose group's size (source_level_sizes) is at
    least that fine. Otherwise level 0's own pixel size doubling, every level from every source."""
    finest_shape = [level0_properties['shape'][dim] for dim in scaled_dims]
    if native:
        level_spacings = native_level_spacings(source_spacings, finest_shape, tolerance=tolerance)
    else:
        level_spacings = native_level_spacings([min(level0_properties['spacing'][dim] for dim in 'yx')], finest_shape)
    levels = native_level_stack_properties(level0_properties, level_spacings, scaled_dims)
    level_sizes = source_level_sizes(source_spacings)
    return [(level_spacing, properties,
             [index for index, spacing in enumerate(source_spacings)
              if not native or level_sizes[spacing] <= level_spacing * (1 + tolerance)])
            for level_spacing, properties in zip(level_spacings, levels)]


def level_blocks(views, output_properties, output_chunksize, interpolation_order=1):
    """(chunk size, {source group: blocks}) of one fused level, its views - (stack properties, affine matrix) - at
    the source levels read for it: the chunk clipped to the level, then shrunk to the memory budget."""
    sdims = list(views[0][0]['shape'])
    bounds = view_bounds(views, output_properties, interpolation_order)
    chunk = {dim: min(int(output_chunksize[dim]), output_properties['shape'][dim]) for dim in output_chunksize}
    chunk = budget_chunksize(bounds, output_properties, chunk, sdims, default_export_fusion_chunk_bytes,
                             4 * fusion_stack_arrays)
    return chunk, block_sources(bounds, output_properties, chunk, sdims)


def written_voxels(groups, output_properties, chunk):
    """Spatial voxels in the blocks some source reaches, edge blocks clipped to the output."""
    sdims = list(output_properties['shape'])
    blocks = np.array([block for group_blocks in groups.values() for block in group_blocks]).reshape(-1, len(sdims))
    chunks = np.array([chunk[dim] for dim in sdims])
    shape = np.array([output_properties['shape'][dim] for dim in sdims])
    return int(np.prod(np.minimum(chunks, shape - blocks * chunks), axis=1).sum())


def level_paths(count):
    """Level path names, zero-padded from 11 levels on: readers listing arrays by name (napari) sort them as text."""
    width = len(str(count - 1)) if count > 10 else 1
    return [str(index).zfill(width) for index in range(count)]


def _affine_table(param):
    """(matrix, x_in index, x_out index) of a 2D xaffine: param.sel(x_in=a, x_out=b) is matrix[ins[a], outs[b]]."""
    matrix = np.asarray(param.data, dtype=float)
    if param.dims.index('x_in') != 0:
        matrix = matrix.T
    ins = {label: index for index, label in enumerate(param.coords['x_in'].values.tolist())}
    outs = {label: index for index, label in enumerate(param.coords['x_out'].values.tolist())}
    return matrix, ins, outs


def _affine_value(table, x_in, x_out):
    matrix, ins, outs = table
    return float(matrix[ins[x_in], outs[x_out]])


def _axis_aligned_translation_dims(sparams, sdims, tol=1e-6):
    """fusion_core._get_axis_aligned_translation_dims, each affine read once instead of an xarray sel per entry."""
    tables = [_affine_table(param) for param in sparams]
    aligned = []
    for dim in sdims:
        others = [other for other in sdims if other != dim]
        if all(np.isclose(_affine_value(table, dim, dim), 1, atol=tol)
               and all(np.isclose(_affine_value(table, dim, other), 0, atol=tol)
                       and np.isclose(_affine_value(table, other, dim), 0, atol=tol) for other in others)
               for table in tables):
            aligned.append(dim)
    return aligned


def _grid_aligned_translation_dims(sparams, views_bb, output_stack_properties, sdims, tol=1e-6):
    """fusion_core._get_grid_aligned_translation_dims on affines read once."""
    aligned = set(_axis_aligned_translation_dims(sparams, sdims, tol=tol))
    tables = [_affine_table(param) for param in sparams]
    grid_aligned = []
    for dim in sdims:
        if (dim in aligned
                and all(np.isclose(output_stack_properties['spacing'][dim], view_bb['spacing'][dim], atol=tol)
                        for view_bb in views_bb)
                and all(fusion_core._is_grid_aligned(
                    output_stack_properties['origin'][dim] - _affine_value(table, dim, '1') - view_bb['origin'][dim],
                    view_bb['spacing'][dim], tol=tol) for table, view_bb in zip(tables, views_bb))):
            grid_aligned.append(dim)
    return grid_aligned


def _axis_aligned_translation_overlap(target_bb, query_bb, param, sdims, additional_extent_in_pixels=None, tol=1e-6):
    """fusion_core._get_axis_aligned_translation_overlap on the affine read once."""
    if additional_extent_in_pixels is None:
        additional_extent_in_pixels = {dim: 0 for dim in sdims}
    table = _affine_table(param)
    overlap_origin, overlap_shape = {}, {}
    for dim in sdims:
        query_spacing = query_bb['spacing'][dim]
        translation = _affine_value(table, dim, '1')
        query_min = target_bb['origin'][dim] - translation
        query_max = (target_bb['origin'][dim] + (int(target_bb['shape'][dim]) - 1) * target_bb['spacing'][dim]
                     - translation)
        query_min, query_max = sorted((query_min, query_max))
        additional_extent = additional_extent_in_pixels[dim] * query_spacing
        start = int(np.floor((query_min - additional_extent - query_bb['origin'][dim]) / query_spacing + tol))
        stop = int(np.ceil((query_max + additional_extent - query_bb['origin'][dim]) / query_spacing - tol)) + 1
        overlap_start, overlap_stop = max(start, 0), min(stop, int(query_bb['shape'][dim]))
        if overlap_start >= overlap_stop:
            return None
        overlap_origin[dim] = query_bb['origin'][dim] + overlap_start * query_spacing
        overlap_shape[dim] = overlap_stop - overlap_start
    return {'origin': overlap_origin, 'shape': overlap_shape, 'spacing': query_bb['spacing']}


@contextmanager
def _numpy_fusion_plan():
    # each block's fusion plan checks every source's affine: through xarray, that is most of a small level's time
    patched = {'_get_axis_aligned_translation_dims': _axis_aligned_translation_dims,
               '_get_grid_aligned_translation_dims': _grid_aligned_translation_dims,
               '_get_axis_aligned_translation_overlap': _axis_aligned_translation_overlap}
    originals = {name: getattr(fusion_core, name) for name in patched}
    for name, function in patched.items():
        setattr(fusion_core, name, function)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(fusion_core, name, function)


def fuse_levels_to_ome_zarr(msims, levels, output_zarr_url, transform_key, level0_properties, output_chunksize,
                            fusion_func=None, zarr_options=None, batch_options=None, interpolation_order=1,
                            ranks=None, average_levels=False, level_weights=None):
    """An OME-Zarr multiscale image of `levels` (export_levels), each fused at its own pixel size from its sources at
    their nearest level - or, where its sources are the level above's (every level, with average_levels), averaged
    from that level: a source's own coarse levels end before the output's, and reading a finer one aliases. With `ranks`
    (one a source, lower first), a higher-ranked source under a lower-ranked one is left out. One progress bar runs
    over all levels, each its share of the square root of `level_weights` (the bytes it writes), else of its voxels."""
    ngff_version, creation_kwargs = _zarr_options(zarr_options)
    _remove_existing(output_zarr_url, zarr_options)
    logging.info(f'Fusion levels {[float(round(level[0], 4)) for level in levels]}')
    paths = level_paths(len(levels))
    if level_weights is None or len(level_weights) != len(levels) or not sum(level_weights) > 0:
        level_weights = [float(np.prod(list(properties['shape'].values()))) for _, properties, _ in levels]
    # a level's time grows slower than its size, each block's cost being partly per source
    level_weights = [float(np.sqrt(weight)) for weight in level_weights]
    progress = fusion_core.tqdm(total=float(sum(level_weights)))
    with _numpy_fusion_plan():
        for index, (level_spacing, properties, selected) in enumerate(levels):
            output_properties = {key: properties[key] for key in ('spacing', 'origin', 'shape')}
            # the source level at the output's pixel size: spacings read off coordinates differ in the last digits
            sims = [msi_utils.get_sim_from_msim(msims[source], scale='scale%d' % coarsest_level_within(
                msim_level_spacings(msims[source]), output_properties['spacing'],
                dims=list(output_properties['spacing']))) for source in selected]
            level_chunksize, groups = level_blocks(sim_views(sims, transform_key), output_properties,
                                                   output_chunksize, interpolation_order)
            progress.set_description(f'{index + 1}/{len(levels)} @ {print_significants(level_spacing, 3)} µm')
            if index > 0 and (average_levels or selected == levels[index - 1][2]):
                factors = {dim: round(properties['factors'][dim] / levels[index - 1][1]['factors'][dim])
                           for dim in output_properties['shape']}
                average_into_zarr_array(os.path.join(output_zarr_url, paths[index - 1]),
                                        os.path.join(output_zarr_url, paths[index]), factors, level_chunksize,
                                        [block for blocks in groups.values() for block in blocks],
                                        creation_kwargs=creation_kwargs, batch_options=batch_options,
                                        progress=progress, share=level_weights[index])
            else:
                busiest = max(groups, key=len, default=())
                logging.info(f'Fusion level {index + 1} fused from {len(sims)} sources: chunk {level_chunksize},'
                             f' {sum(len(blocks) for blocks in groups.values())} blocks, budget'
                             f' {print_hbytes(default_export_fusion_chunk_bytes)} a block, busiest block'
                             f' {len(busiest)} sources'
                             f' ({print_hbytes(sum(sims[source].data.nbytes for source in busiest))} of source levels)')
                fuse_into_zarr_array(sims, os.path.join(output_zarr_url, paths[index]), transform_key,
                                     output_properties, level_chunksize, fusion_func=fusion_func,
                                     creation_kwargs=creation_kwargs, batch_options=batch_options,
                                     interpolation_order=interpolation_order, progress=progress,
                                     share=level_weights[index],
                                     ranks=[ranks[source] for source in selected] if ranks is not None else None,
                                     groups=groups, log_batches=index > 0)
        progress.close()
    sim0 = msi_utils.get_sim_from_msim(msims[0], scale='scale0')
    coordtfs, axes = ngff_utils.calc_ngff_coordinate_transformations_and_axes(
        level0_properties, [properties['factors'] for _, properties, _ in levels],
        nsdims=list(si_utils.get_nonspatial_dims_from_sim(sim0)), time_transform=ngff_utils.get_ngff_time_transform(sim0))
    group = zarr.open_group(output_zarr_url, mode='a',
                            **ngff_utils.zarr_group_creation_kwargs_for_ngff_version(storage_ngff_version(ngff_version)))
    write_multiscales_metadata(
        group, axes=axes, ngff_version=ngff_version,
        datasets=[{'path': paths[index], 'coordinateTransformations': coordtfs[index]} for index in range(len(levels))])
    return ngff_utils.read_msim_from_ome_zarr(output_zarr_url, transform_key=transform_key, array_backend='dask')
