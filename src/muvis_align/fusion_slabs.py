"""A fusion written to zarr one z-slab of blocks at a time, each slab fused from only the sources that reach it:
multiview_stitcher fuses every block from every source, in Python, so a block's cost grows with the source count.
The steps are its own zarr path's, prepare_block_fusion(create_output=False) attaching each slab to the one store."""
import copy
import os
import shutil

import dask.array as da
import numpy as np
import zarr
import multiview_stitcher.fusion._core as fusion_core
from multiview_stitcher import msi_utils, mv_graph, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils


def slab_sources(sims, transform_key, output_stack_properties, z_chunk, interpolation_order=1):
    """Per z-slab of output blocks (z_chunk voxels thick), the indices of the sources that reach it - by
    multiview_stitcher's own rule: each source's box, padded for interpolation unless z is grid-aligned."""
    sdims = list(si_utils.get_spatial_dims_from_sim(sims[0]))
    params = [si_utils.get_affine_from_sim(sim, transform_key=transform_key) for sim in sims]
    boxes = [si_utils.get_stack_properties_from_sim(sim) for sim in sims]
    is_grid_aligned = 'z' in fusion_core._get_grid_aligned_translation_dims(
        sparams=params, views_bb=boxes, output_stack_properties=output_stack_properties, sdims=sdims)
    origin, spacing = output_stack_properties['origin']['z'], output_stack_properties['spacing']['z']
    nslabs = int(np.ceil(output_stack_properties['shape']['z'] / z_chunk))
    slabs = [[] for _ in range(nslabs)]
    for index, sim in enumerate(sims):
        padding = 0.0 if is_grid_aligned else interpolation_order * boxes[index]['spacing']['z']
        props = si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key)
        z_values = mv_graph.get_vertices_from_stack_props(props)[:, sdims.index('z')]
        first = max(0, int(np.floor((z_values.min() - padding - origin) / (z_chunk * spacing))))
        last = min(nslabs - 1, int(np.floor((z_values.max() + padding - origin) / (z_chunk * spacing))))
        for slab in range(first, last + 1):
            slabs[slab].append(index)
    return slabs


def source_bounds(sims, transform_key, output_stack_properties, interpolation_order=1):
    """Per source and spatial dim, the (low, high) physical bounds of what it reaches in the output: its box padded
    for interpolation, by multiview_stitcher's own rule - except in a dim the sources are grid-aligned in."""
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
            padding = 0.0 if dim in aligned else interpolation_order * boxes[index]['spacing'][dim]
            bounds[index, axis] = vertices[:, axis].min() - padding, vertices[:, axis].max() + padding
    return bounds


def reached_blocks(block_ids, bounds, output_stack_properties, output_chunksize, sdims, block_axes):
    """The blocks some source reaches. One no source reaches keeps the store's fill value unwritten, as fusing it
    from no source would give - at native resolution, most of them."""
    if not len(bounds) or not block_ids:
        return []
    blocks = np.asarray(block_ids, dtype=int).reshape(len(block_ids), -1)
    keep = np.ones(len(blocks), dtype=bool)
    for axis, dim in enumerate(sdims):
        spacing, chunk = output_stack_properties['spacing'][dim], int(output_chunksize[dim])
        # a block's pixel centres, widened by half a pixel each side
        low = output_stack_properties['origin'][dim] + (blocks[:, block_axes[axis]] * chunk - 0.5) * spacing
        high = low + chunk * spacing
        keep &= np.any((low[:, None] <= bounds[None, :, axis, 1]) & (bounds[None, :, axis, 0] <= high[:, None]),
                       axis=1)
    return [block_id for block_id, kept in zip(block_ids, keep) if kept]


def fuse_into_zarr_array(sims, store_url, transform_key, output_stack_properties, output_chunksize, fusion_func=None,
                         creation_kwargs=None, batch_options=None, interpolation_order=1, desc=None):
    """Fuse `sims` into a new zarr array at store_url: a z-slab of blocks at a time from the sources reaching it, and
    only the blocks some source reaches. Returns the array's dims and output stack properties."""
    batch_options = batch_options or {}
    dims = list(sims[0].dims)
    sdims = list(si_utils.get_spatial_dims_from_sim(sims[0]))
    z_axis = dims.index('z') if 'z' in dims else None
    if z_axis is None:
        slabs = [list(range(len(sims)))]
    else:
        slabs = slab_sources(sims, transform_key, output_stack_properties, int(output_chunksize['z']),
                             interpolation_order)
    bounds = source_bounds(sims, transform_key, output_stack_properties, interpolation_order)
    block_axes = [dims.index(dim) for dim in sdims]
    batch_func, n_batch = batch_options.get('batch_func'), batch_options.get('n_batch', 1)
    info, progress = None, None
    for slab, sources in enumerate(slabs):
        if sources or info is None:
            # the first slab creates the store (from any source, if it has none), the others attach to it
            fuse_kwargs = {'images': [sims[index] for index in sources] or sims[:1], 'transform_key': transform_key,
                           'output_chunksize': output_chunksize,
                           'output_stack_properties': copy.deepcopy(output_stack_properties),
                           'interpolation_order': interpolation_order}
            # None would replace multiview_stitcher's own default
            if fusion_func is not None:
                fuse_kwargs['fusion_func'] = fusion_func
            info = fusion_core.prepare_block_fusion(store_url, fuse_kwargs=fuse_kwargs,
                                                    zarr_array_creation_kwargs=creation_kwargs,
                                                    create_output=progress is None, verbose=False)
        if progress is None:
            progress = fusion_core.tqdm(total=int(np.prod(info['nblocks'])), desc=desc)
        ranges = [range(count) for count in info['nblocks']]
        if z_axis is not None:
            ranges[z_axis] = range(slab, slab + 1)
        block_ids = list(np.ndindex(*[len(values) for values in ranges]))
        block_ids = [tuple(values[position] for values, position in zip(ranges, block)) for block in block_ids]
        reached = reached_blocks(block_ids, bounds[sources], output_stack_properties, output_chunksize, sdims,
                                 block_axes)
        progress.update(len(block_ids) - len(reached))
        for start in range(0, len(reached), n_batch):
            batch = reached[start:start + n_batch]
            if batch_func is None:
                for block_id in batch:
                    info['func'](block_id)
            else:
                batch_func(info['func'], batch, **(batch_options.get('batch_func_kwargs') or {}))
            progress.update(len(batch))
    progress.close()
    return dims, info['output_stack_properties']


def _zarr_options(zarr_options):
    zarr_options = zarr_options or {}
    ome_zarr = zarr_options.get('ome_zarr', False)
    ngff_version = zarr_options.get('ngff_version', '0.4')
    creation_kwargs = zarr_options.get('zarr_array_creation_kwargs')
    if ome_zarr:
        creation_kwargs = ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(ngff_version, creation_kwargs)
    return ome_zarr, ngff_version, creation_kwargs


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
    dims, properties = fuse_into_zarr_array(sims, store_url, transform_key, output_stack_properties, output_chunksize,
                                            fusion_func=fusion_func, creation_kwargs=creation_kwargs,
                                            batch_options=batch_options, interpolation_order=interpolation_order)
    fused = si_utils.get_sim_from_array(array=da.from_zarr(store_url), dims=dims, transform_key=transform_key,
                                        scale=properties['spacing'], translation=properties['origin'],
                                        c_coords=sims[0].coords['c'].values, t_coords=sims[0].coords['t'].values)
    ngff_utils.copy_ngff_time_transform(sims[0], fused)
    if ome_zarr:
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
            # a step of under sqrt(2) to a source's own size replaces the level before it instead
            if len(levels) > 1 and pending[0] < levels[-1] * np.sqrt(2):
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
    levels = native_level_stack_properties(level0_properties, level_spacings, scaled_dims)
    for index, (level_spacing, properties) in enumerate(zip(level_spacings, levels)):
        output_properties = {key: properties[key] for key in ('spacing', 'origin', 'shape')}
        selected = [msim for msim, spacing in zip(msims, source_spacings) if spacing <= level_spacing * (1 + tolerance)]
        sims = [msi_utils.get_sim_from_msim(
            msim, scale='scale%s' % msi_utils.get_res_level_from_spacing(msim, output_properties['spacing']))
            for msim in selected]
        level_chunksize = {dim: min(int(output_chunksize[dim]), properties['shape'][dim]) for dim in output_chunksize}
        fuse_into_zarr_array(sims, os.path.join(output_zarr_url, str(index)), transform_key, output_properties,
                             level_chunksize, fusion_func=fusion_func, creation_kwargs=creation_kwargs,
                             batch_options=batch_options, interpolation_order=interpolation_order,
                             desc=f'Level {index} at {level_spacing:.4g} ({len(sims)} sources)')
    sim0 = msi_utils.get_sim_from_msim(msims[0], scale='scale0')
    coordtfs, axes = ngff_utils.calc_ngff_coordinate_transformations_and_axes(
        level0_properties, [level['factors'] for level in levels],
        nsdims=list(si_utils.get_nonspatial_dims_from_sim(sim0)), time_transform=ngff_utils.get_ngff_time_transform(sim0))
    group = zarr.open_group(output_zarr_url, mode='a', **ngff_utils.zarr_group_creation_kwargs_for_ngff_version(ngff_version))
    ngff_utils.write_multiscales_metadata(
        group, axes=axes, ngff_version=ngff_version,
        datasets=[{'path': str(index), 'coordinateTransformations': coordtfs[index]} for index in range(len(levels))])
    return ngff_utils.read_msim_from_ome_zarr(output_zarr_url, transform_key=transform_key, array_backend='dask')
