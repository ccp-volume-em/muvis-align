"""Fusion written to zarr block by block, each block from only the sources that reach it and only the blocks some
source reaches: multiview_stitcher runs Python over every source it gives a block, so a block's cost grows with the
source count. The steps are its own zarr path's, prepare_block_fusion(create_output=False) attaching to the store."""
import copy
from dataclasses import asdict
import inspect
import logging
import os
import shutil

import dask.array as da
import numpy as np
import ngff_zarr
import zarr
import multiview_stitcher.fusion._core as fusion_core
from multiview_stitcher import msi_utils, mv_graph, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.Timer import Timer
from muvis_align.constants import default_export_fusion_chunk_bytes, fusion_stack_arrays


def source_bounds(sims, transform_key, output_stack_properties, interpolation_order=1):
    """Per source and spatial dim, the (low, high) physical bounds of what it reaches in the output: its box padded
    for interpolation, by multiview_stitcher's own rule - except in a dim the sources are grid-aligned in, or have
    a single plane in."""
    sdims = list(si_utils.get_spatial_dims_from_sim(sims[0]))
    params = [si_utils.get_affine_from_sim(sim, transform_key=transform_key) for sim in sims]
    # per time point, as multiview_stitcher's fuse() asks it
    params = [param.isel(t=0) if 't' in param.dims else param for param in params]
    boxes = [si_utils.get_stack_properties_from_sim(sim) for sim in sims]
    aligned = fusion_core._get_grid_aligned_translation_dims(
        sparams=params, views_bb=boxes, output_stack_properties=output_stack_properties, sdims=sdims)
    bounds = np.empty((len(sims), len(sdims), 2))
    for index, sim in enumerate(sims):
        vertices = mv_graph.get_vertices_from_stack_props(
            si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key))
        for axis, dim in enumerate(sdims):
            # a single plane's spacing is a placeholder 1.0: padded by it, a section reaches its neighbours' blocks
            single_plane = boxes[index]['shape'][dim] == 1
            padding = 0.0 if dim in aligned or single_plane else interpolation_order * boxes[index]['spacing'][dim]
            bounds[index, axis] = vertices[:, axis].min() - padding, vertices[:, axis].max() + padding
    return bounds


def _block_overlaps(bounds, origin, spacing, chunk, nblocks):
    """Per block along one dim and source, whether the source reaches the block: its padded bounds meet the block's pixel centres, widened by half a pixel."""
    low = origin + (np.arange(nblocks) * chunk - 0.5) * spacing
    high = low + chunk * spacing
    return (low[:, None] <= bounds[None, :, 1]) & (bounds[None, :, 0] <= high[:, None])


def budget_chunksize(bounds, output_stack_properties, output_chunksize, sdims, budget_bytes, bytes_per_voxel,
                     min_xy_chunk=256):
    """The largest yx chunk, shrinking from output_chunksize, whose busiest block fits budget_bytes: a block holds
    every source it meets transformed to its full size, so a coarse level's block meeting a whole section's tiles
    at once costs many times a fine one's - counted from the sources' own bounds, not their average density."""
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
        # by just the factor needed (fewer sources may meet the smaller block, so check again): halving went from
        # 1600 to 800 where ~1500 fits, and every extra block costs its own fixed overhead
        factor = min(0.95, np.sqrt(budget_bytes / needed))
        for dim in xy_dims:
            chunk[dim] = max(min_xy_chunk, int(chunk[dim] * factor) // 64 * 64)


def block_sources(bounds, output_stack_properties, output_chunksize, sdims):
    """The blocks some source reaches, grouped by exactly which: {source indices: [spatial block indices]}.
    multiview_stitcher's per-block fusion runs Python over every source it is given, reaching the block or not."""
    overlaps = [_block_overlaps(bounds[:, axis], output_stack_properties['origin'][dim],
                                output_stack_properties['spacing'][dim], int(output_chunksize[dim]),
                                int(np.ceil(output_stack_properties['shape'][dim] / int(output_chunksize[dim]))))
                for axis, dim in enumerate(sdims)]
    groups = {}
    # the first dim's blocks one at a time keeps the rest a small product (a section's sources, at most)
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
                         creation_kwargs=None, batch_options=None, interpolation_order=1, desc=None, ranks=None):
    """Fuse `sims` into a new zarr array at store_url: only the blocks some source reaches, each from only the sources
    reaching it (one prepare_block_fusion per set of them, ~2ms). With `ranks` (lower first, e.g. pixel sizes), a
    block whose sources differ in rank fuses only the lowest-ranked at each pixel. Returns dims and properties."""
    batch_options = batch_options or {}
    dims = list(sims[0].dims)
    sdims = list(si_utils.get_spatial_dims_from_sim(sims[0]))
    block_axes = [dims.index(dim) for dim in sdims]
    bounds = source_bounds(sims, transform_key, output_stack_properties, interpolation_order)
    groups = block_sources(bounds, output_stack_properties, output_chunksize, sdims)
    batch_func, n_batch = batch_options.get('batch_func'), batch_options.get('n_batch', 1)

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
    for sources, spatial_blocks in groups.items():
        func = prepare(sources, create_output=False)['func']
        for nonspatial_block in np.ndindex(*[nblocks[axis] for axis in nonspatial_axes]):
            for spatial_block in spatial_blocks:
                block_id = [0] * len(dims)
                for axis, block in zip(nonspatial_axes + block_axes, nonspatial_block + spatial_block):
                    block_id[axis] = block
                funcs[tuple(block_id)] = func

    def fuse_block(block_id):
        return funcs[block_id](block_id)

    block_ids = sorted(funcs)
    progress = fusion_core.tqdm(total=int(np.prod(nblocks)), desc=desc)
    progress.update(int(np.prod(nblocks)) - len(block_ids))
    for start in range(0, len(block_ids), n_batch):
        batch = block_ids[start:start + n_batch]
        if batch_func is None:
            for block_id in batch:
                fuse_block(block_id)
        else:
            batch_func(fuse_block, batch, **(batch_options.get('batch_func_kwargs') or {}))
        progress.update(len(batch))
    progress.close()
    return dims, info['output_stack_properties']


def storage_ngff_version(ngff_version):
    # 0.6 changed the metadata, not the arrays: they are stored as 0.5's (zarr v3), which multiview_stitcher can create
    return '0.5' if str(ngff_version).startswith('0.6') else ngff_version


def _zarr_options(zarr_options):
    zarr_options = zarr_options or {}
    ome_zarr = zarr_options.get('ome_zarr', False)
    ngff_version = zarr_options.get('ngff_version', '0.4')
    creation_kwargs = zarr_options.get('zarr_array_creation_kwargs')
    if ome_zarr:
        creation_kwargs = ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
            storage_ngff_version(ngff_version), creation_kwargs)
    return ome_zarr, ngff_version, creation_kwargs


def write_multiscales_metadata(group, axes, datasets, ngff_version):
    """multiview_stitcher's write_multiscales_metadata, and OME-Zarr 0.6 too, which it does not know: the same
    metadata converted by ngff-zarr (axes in a coordinate system, each level's transforms a sequence into it)."""
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


def fuse_to_zarr_by_z_slabs(msims, output_zarr_url, transform_key, output_stack_properties, output_chunksize,
                            fusion_func=None, zarr_options=None, batch_options=None, interpolation_order=1):
    """As multiview_stitcher.fusion.fuse(msims, output_zarr_url=...), returning the same, but fusing each z-slab
    of blocks from only the sources that reach it."""
    sims = [msi_utils.get_sim_from_msim(
        msim, scale='scale%s' % msi_utils.get_res_level_from_spacing(msim, output_stack_properties['spacing']))
        for msim in msims]
    ome_zarr, ngff_version, creation_kwargs = _zarr_options(zarr_options)
    store_url = os.path.join(output_zarr_url, '0') if ome_zarr else output_zarr_url
    _remove_existing(output_zarr_url, zarr_options)
    # the caller's size assumes sources spread evenly over a plane; clustered tiles put many more in one block
    budgeted_chunksize = budget_chunksize(
        source_bounds(sims, transform_key, output_stack_properties, interpolation_order), output_stack_properties,
        output_chunksize, list(si_utils.get_spatial_dims_from_sim(sims[0])), default_export_fusion_chunk_bytes,
        4 * fusion_stack_arrays)
    if budgeted_chunksize != output_chunksize:
        logging.info(f'Fusion output_chunksize {budgeted_chunksize}: its busiest block meets more sources than'
                     f' {output_chunksize} allowed for')
    output_chunksize = budgeted_chunksize
    with Timer('fusion by z-slabs: level 0'):
        dims, properties = fuse_into_zarr_array(sims, store_url, transform_key, output_stack_properties,
                                                output_chunksize, fusion_func=fusion_func,
                                                creation_kwargs=creation_kwargs, batch_options=batch_options,
                                                interpolation_order=interpolation_order)
    fused = si_utils.get_sim_from_array(array=da.from_zarr(store_url), dims=dims, transform_key=transform_key,
                                        scale=properties['spacing'], translation=properties['origin'],
                                        c_coords=sims[0].coords['c'].values, t_coords=sims[0].coords['t'].values)
    ngff_utils.copy_ngff_time_transform(sims[0], fused)
    if ome_zarr:
        with Timer('fusion by z-slabs: pyramid levels'):
            ngff_utils.write_sim_to_ome_zarr(fused, output_zarr_url=output_zarr_url, overwrite=False,
                                             batch_options=batch_options, zarr_array_creation_kwargs=creation_kwargs,
                                             ngff_version=ngff_version)
        return ngff_utils.read_msim_from_ome_zarr(output_zarr_url, transform_key=transform_key, array_backend='dask')
    return msi_utils.get_msim_from_sim(fused, scale_factors=[])


def native_level_spacings(source_spacings, finest_shape, min_shape=100, tolerance=0.05):
    """Output level pixel sizes that never upsample a source: the finest source's, doubling, with a level at each
    coarser source pixel size (tiles at 0.01 and overviews at 0.249: 0.01, 0.02, 0.04, 0.08, 0.16, 0.249, 0.498..),
    until the largest dim would fall to min_shape - each source size gets its level, however small, or what only it
    covers would be in none. Sizes within `tolerance` of each other count as one, the finest."""
    sizes = []
    for size in sorted(source_spacings):
        if not sizes or size > sizes[-1] * (1 + tolerance):
            sizes.append(size)
    levels, pending = [sizes[0]], sizes[1:]
    largest = max(finest_shape)
    while pending or largest * levels[0] / (levels[-1] * 2) > min_shape:
        doubled = levels[-1] * 2
        if pending and doubled >= pending[0] * (1 - tolerance):
            # a step of under sqrt(2) to a source's own size replaces a doubled level before it - never another
            # source's size, whose sources would then be in no level at their own size
            if len(levels) > 1 and levels[-1] not in sizes and pending[0] < levels[-1] * np.sqrt(2):
                levels[-1] = pending.pop(0)
            else:
                levels.append(pending.pop(0))
        else:
            levels.append(doubled)
    return levels


def native_level_stack_properties(level0_properties, level_spacings, scaled_dims):
    """Each level's output stack: level 0's extent at the level's pixel size in `scaled_dims` (other dims, such as a
    section stack's z, kept as they are), its origin by OME-Zarr's pixel-centre convention."""
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


def level_paths(count):
    """'0'..'9', zero-padded from 11 levels on: readers that list a group's arrays by name (napari's own) sort them
    as text, putting '10' after '1' - the multiscales metadata, which names them, has them in order either way."""
    width = len(str(count - 1)) if count > 10 else 1
    return [str(index).zfill(width) for index in range(count)]


def fuse_native_levels_to_ome_zarr(msims, source_spacings, output_zarr_url, transform_key, level0_properties,
                                   output_chunksize, scaled_dims, fusion_func=None, zarr_options=None,
                                   batch_options=None, interpolation_order=1, tolerance=0.05):
    """An OME-Zarr multiscale image whose every level is fused at its own pixel size from only the sources at least
    that fine, each read at its own nearest level: no source upsampled, no level downsampled from another."""
    zarr_options = (zarr_options or {}) | {'ome_zarr': True}
    _, ngff_version, creation_kwargs = _zarr_options(zarr_options)
    _remove_existing(output_zarr_url, zarr_options)
    level_spacings = native_level_spacings(source_spacings, [level0_properties['shape'][dim] for dim in scaled_dims],
                                           tolerance=tolerance)
    sizes, counts = np.unique(np.round(source_spacings, 4), return_counts=True)
    logging.info(f'Native fusion levels {[float(round(spacing, 4)) for spacing in level_spacings]}, sources by pixel'
                 f' size {dict(zip(sizes.tolist(), counts.tolist()))}')
    levels = native_level_stack_properties(level0_properties, level_spacings, scaled_dims)
    paths = level_paths(len(levels))
    for index, (level_spacing, properties) in enumerate(zip(level_spacings, levels)):
        output_properties = {key: properties[key] for key in ('spacing', 'origin', 'shape')}
        selected = [msim for msim, spacing in zip(msims, source_spacings) if spacing <= level_spacing * (1 + tolerance)]
        sims = [msi_utils.get_sim_from_msim(
            msim, scale='scale%s' % msi_utils.get_res_level_from_spacing(msim, output_properties['spacing']))
            for msim in selected]
        level_chunksize = {dim: min(int(output_chunksize[dim]), properties['shape'][dim]) for dim in output_chunksize}
        level_chunksize = budget_chunksize(
            source_bounds(sims, transform_key, output_properties, interpolation_order), output_properties,
            level_chunksize, list(si_utils.get_spatial_dims_from_sim(sims[0])), default_export_fusion_chunk_bytes,
            4 * fusion_stack_arrays)
        # tiles over the overview: where a finer source covers a pixel, the coarser one under it is left out
        fuse_into_zarr_array(sims, os.path.join(output_zarr_url, paths[index]), transform_key, output_properties,
                             level_chunksize, fusion_func=fusion_func, creation_kwargs=creation_kwargs,
                             batch_options=batch_options, interpolation_order=interpolation_order,
                             desc=f'Level {index} at {level_spacing:.4g} ({len(sims)} sources)',
                             ranks=[spacing for spacing in source_spacings if spacing <= level_spacing * (1 + tolerance)])
    sim0 = msi_utils.get_sim_from_msim(msims[0], scale='scale0')
    coordtfs, axes = ngff_utils.calc_ngff_coordinate_transformations_and_axes(
        level0_properties, [level['factors'] for level in levels],
        nsdims=list(si_utils.get_nonspatial_dims_from_sim(sim0)), time_transform=ngff_utils.get_ngff_time_transform(sim0))
    group = zarr.open_group(output_zarr_url, mode='a',
                            **ngff_utils.zarr_group_creation_kwargs_for_ngff_version(storage_ngff_version(ngff_version)))
    write_multiscales_metadata(
        group, axes=axes, ngff_version=ngff_version,
        datasets=[{'path': paths[index], 'coordinateTransformations': coordtfs[index]} for index in range(len(levels))])
    return ngff_utils.read_msim_from_ome_zarr(output_zarr_url, transform_key=transform_key, array_backend='dask')
