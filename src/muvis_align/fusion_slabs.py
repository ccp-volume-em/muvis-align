"""A fusion written to zarr one z-slab of blocks at a time, each slab fused from only the sources that reach it:
multiview_stitcher fuses every block from every source, in Python, so a block's cost grows with the source count.
The steps are its own zarr path's, prepare_block_fusion(create_output=False) attaching each slab to the one store."""
import copy
import os
import shutil

import dask.array as da
import numpy as np
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


def fuse_to_zarr_by_z_slabs(msims, output_zarr_url, transform_key, output_stack_properties, output_chunksize,
                            fusion_func=None, zarr_options=None, batch_options=None, interpolation_order=1):
    """As multiview_stitcher.fusion.fuse(msims, output_zarr_url=...), returning the same, but fusing each z-slab
    of blocks from only the sources that reach it."""
    sims = [msi_utils.get_sim_from_msim(
        msim, scale='scale%s' % msi_utils.get_res_level_from_spacing(msim, output_stack_properties['spacing']))
        for msim in msims]
    zarr_options = zarr_options or {}
    batch_options = batch_options or {}
    ome_zarr = zarr_options.get('ome_zarr', False)
    ngff_version = zarr_options.get('ngff_version', '0.4')
    creation_kwargs = zarr_options.get('zarr_array_creation_kwargs')
    if ome_zarr:
        creation_kwargs = ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(ngff_version, creation_kwargs)
    store_url = os.path.join(output_zarr_url, '0') if ome_zarr else output_zarr_url
    if zarr_options.get('overwrite', True) and os.path.exists(output_zarr_url):
        shutil.rmtree(output_zarr_url)

    dims = list(sims[0].dims)
    z_axis = dims.index('z')
    batch_func, n_batch = batch_options.get('batch_func'), batch_options.get('n_batch', 1)
    info, progress = None, None
    for slab, sources in enumerate(slab_sources(sims, transform_key, output_stack_properties,
                                                int(output_chunksize['z']), interpolation_order)):
        if sources or info is None:
            # the first slab creates the store (from any source, if it has none), the others attach to it
            fuse_kwargs = {'images': [sims[index] for index in sources] or sims[:1], 'transform_key': transform_key,
                           'fusion_func': fusion_func, 'output_chunksize': output_chunksize,
                           'output_stack_properties': copy.deepcopy(output_stack_properties),
                           'interpolation_order': interpolation_order}
            info = fusion_core.prepare_block_fusion(store_url, fuse_kwargs=fuse_kwargs,
                                                    zarr_array_creation_kwargs=creation_kwargs,
                                                    create_output=progress is None, verbose=False)
        if progress is None:
            progress = fusion_core.tqdm(total=int(np.prod(info['nblocks'])))
        ranges = [range(count) for count in info['nblocks']]
        ranges[z_axis] = range(slab, slab + 1)
        block_ids = list(np.ndindex(*[len(values) for values in ranges]))
        block_ids = [tuple(values[position] for values, position in zip(ranges, block)) for block in block_ids]
        # a slab no source reaches keeps the store's fill value, as a block fused from none of them would be
        if not sources:
            progress.update(len(block_ids))
        for start in range(0, len(block_ids) if sources else 0, n_batch):
            batch = block_ids[start:start + n_batch]
            if batch_func is None:
                for block_id in batch:
                    info['func'](block_id)
            else:
                batch_func(info['func'], batch, **(batch_options.get('batch_func_kwargs') or {}))
            progress.update(len(batch))
    progress.close()

    properties = info['output_stack_properties']
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
