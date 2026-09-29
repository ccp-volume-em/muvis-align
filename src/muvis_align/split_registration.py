"""Split registration ('split' pairing): each z-plane's (or channel's) tiles registered in x/y first - the normal
pair and global registration, with pairs only within a group - then the stitched groups against each other as
whole images, consecutive groups paired, as registration_dimension says: stacked planes or overlaid channels."""
import logging

import dask
import numpy as np
from multiview_stitcher import fusion, msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.registration import compute_pairwise_registrations

from muvis_align.constants import default_split_group_size
from muvis_align.image.util import (build_view_adjacency_graph, restore_msims_transform,
                                    snapshot_msims_transform)
from muvis_align.util import raise_if_cancelled

SPLIT = 'split'
# transient keys: the stage-1 transforms while the groups are fused, and each fused group's own placement
STAGE1_KEY = 'split_stage1'
GROUP_KEY = 'split_group'


def split_groups(positions, sources, dimension=None):
    """Each source's group index: its channel label for registration_dimension 'c', else its z-plane."""
    if dimension == 'c':
        keys = [(source.get_channels() or [{}])[0].get('label', '') for source in sources]
    else:
        keys = [round(float(position.get('z', 0.0)), 9) for position in positions]
    order = sorted(set(keys))
    return [order.index(key) for key in keys]


def within_group_pairs(pairs, groups):
    return [pair for pair in pairs if groups[pair[0]] == groups[pair[1]]]


def _level_sim(msim, spacing):
    """The msim's coarsest level no coarser than `spacing` - fusing from it resamples least."""
    chosen = None
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        if chosen is None or max(si_utils.get_spacing_from_sim(sim).values()) <= spacing:
            chosen = sim
    return chosen


def fuse_group(msims, transform_key, max_size=default_split_group_size):
    """One group's tiles fused under `transform_key`, its longest side at most `max_size` pixels, as an msim
    placed by GROUP_KEY."""
    finest = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
    stack_props = [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in finest]
    lower = np.min([[props['origin'][dim] for dim in 'yx'] for props in stack_props], axis=0)
    upper = np.max([[props['origin'][dim] + props['shape'][dim] * props['spacing'][dim] for dim in 'yx']
                    for props in stack_props], axis=0)
    finest_spacing = min(min(props['spacing'][dim] for dim in 'yx') for props in stack_props)
    spacing = max(float(np.max(upper - lower)) / max_size, finest_spacing)
    sims = [_level_sim(msim, spacing) for msim in msims]
    with dask.config.set(scheduler='threads'):
        fused = fusion.fuse(sims, transform_key=transform_key, output_spacing={'y': spacing, 'x': spacing})
        data = np.asarray(fused.data).squeeze()
    origin = si_utils.get_origin_from_sim(fused)
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], scale={'y': spacing, 'x': spacing},
                                      translation={dim: origin[dim] for dim in 'yx'}, transform_key=GROUP_KEY)
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    # as register_pairs' own msims: the (only) channel selected, so registration sees spatial dims alone
    return msi_utils.multiscale_sel_coords(msim, {'c': sim.coords['c'].values[0]})


def register_groups(msims, transforms, groups, base_transform_key, pairwise_reg_func, pairwise_reg_func_kwargs=None,
                    resolution_method='robust_linear', resolution_kwargs=None, max_size=default_split_group_size):
    """Stage 2: `transforms` (stage 1's, relative to `base_transform_key`) with each group's own correction
    composed on - from its fused image registered against the next group's and resolved over all groups."""
    ngroups = max(groups) + 1
    if ngroups < 2:
        return transforms
    snapshot = snapshot_msims_transform(msims, STAGE1_KEY)
    try:
        for msim, transform in zip(msims, transforms):
            msi_utils.set_affine_transform(msim, transform, transform_key=STAGE1_KEY,
                                           base_transform_key=base_transform_key)
        group_msims = []
        for index in range(ngroups):
            raise_if_cancelled()
            group_msims.append(fuse_group([msim for msim, group in zip(msims, groups) if group == index],
                                          STAGE1_KEY, max_size=max_size))
    finally:
        restore_msims_transform(msims, STAGE1_KEY, snapshot)

    pairs = [(index, index + 1) for index in range(ngroups - 1)]
    with dask.config.set({'scheduler': 'threads', 'optimization.fuse.active': False}):
        graph = build_view_adjacency_graph(group_msims, GROUP_KEY, pairs, overlap_tolerance=0)
        raise_if_cancelled()
        graph = compute_pairwise_registrations(group_msims, graph, transform_key=GROUP_KEY,
                                               pairwise_reg_func=pairwise_reg_func,
                                               pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                                               n_parallel_pairwise_regs=1)
    logging.info(f'Split registration: {ngroups} groups, {graph.number_of_edges()} group pairs registered')
    group_params, _ = groupwise_resolution(graph, method=resolution_method, **(resolution_kwargs or {}))
    # each group's correction acts in world space, after the stage-1 transform
    return [param_utils.matmul_xparams(group_params[group], transform) for transform, group in zip(transforms, groups)]
