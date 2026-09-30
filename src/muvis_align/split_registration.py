"""Split registration ('split' pairing): each z-plane's (or channel's) tiles registered in x/y first - the normal
pair and global registration, with pairs only within a group - then the stitched groups against each other as
whole images, consecutive groups paired, as registration_dimension says: stacked planes or overlaid channels."""
import logging

import dask
import networkx as nx
import numpy as np
from multiview_stitcher import fusion, msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.registration import compute_pairwise_registrations
from scipy.ndimage import gaussian_filter

from muvis_align.constants import default_split_binning, split_smoothing
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


def smooth_group(data, spacing, tile_size):
    """The fused group smoothed past what every tile repeats, its background (outside its tiles) filled with its
    mean: an empty background, the same in every group, would match best at zero shift."""
    data = np.asarray(data, dtype=np.float32)
    mask = data > 0
    if not np.any(mask):
        return data
    filled = np.where(mask, data, data[mask].mean())
    return gaussian_filter(filled, tile_size * split_smoothing / spacing)


def group_grid(msims, transform_key):
    """The one y/x grid every group is fused onto - the union of all tiles at their finest (pre-processed) pixel
    size - and the median tile size. On one grid two groups overlap in the whole frame, not a crop cut at an outline."""
    finest = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
    stack_props = [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in finest]
    lower = np.min([[props['origin'][dim] for dim in 'yx'] for props in stack_props], axis=0)
    upper = np.max([[props['origin'][dim] + props['shape'][dim] * props['spacing'][dim] for dim in 'yx']
                    for props in stack_props], axis=0)
    spacing = min(min(props['spacing'][dim] for dim in 'yx') for props in stack_props)
    shape = np.ceil((upper - lower) / spacing).astype(int)
    grid = {'origin': {dim: float(value) for dim, value in zip('yx', lower)},
            'spacing': {dim: spacing for dim in 'yx'},
            'shape': {dim: int(value) for dim, value in zip('yx', shape)}}
    tile_size = float(np.median([max(props['shape'][dim] * props['spacing'][dim] for dim in 'yx')
                                 for props in stack_props]))
    return grid, tile_size


def fuse_group(msims, transform_key, grid, tile_size):
    """One group's tiles fused under `transform_key` onto `grid` and smoothed, as an msim placed by GROUP_KEY."""
    spacing = grid['spacing']['y']
    sims = [_level_sim(msim, spacing) for msim in msims]
    with dask.config.set(scheduler='threads'):
        fused = fusion.fuse(sims, transform_key=transform_key, output_stack_properties=grid)
        data = smooth_group(np.asarray(fused.data).squeeze(), spacing, tile_size)
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], scale=grid['spacing'], translation=grid['origin'],
                                      transform_key=GROUP_KEY)
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    # as register_pairs' own msims: the (only) channel selected, so registration sees spatial dims alone
    return msi_utils.multiscale_sel_coords(msim, {'c': sim.coords['c'].values[0]})


def register_group_pair(msim1, msim2, index1, pairwise_reg_func, pairwise_reg_func_kwargs=None,
                        binning=default_split_binning):
    """Groups `index1` and `index1 + 1` as a two-node pair graph, their edge registered by `pairwise_reg_func` with
    their images binned by `binning`."""
    with dask.config.set({'scheduler': 'threads', 'optimization.fuse.active': False}):
        graph = build_view_adjacency_graph([msim1, msim2], GROUP_KEY, [(0, 1)], overlap_tolerance=0)
        if graph.number_of_edges():
            graph = compute_pairwise_registrations([msim1, msim2], graph, transform_key=GROUP_KEY,
                                                   pairwise_reg_func=pairwise_reg_func,
                                                   pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                                                   registration_binning={dim: binning for dim in 'yx'},
                                                   n_parallel_pairwise_regs=1)
    return nx.relabel_nodes(graph, {0: index1, 1: index1 + 1})


def register_groups(msims, transforms, groups, base_transform_key, pairwise_reg_func, pairwise_reg_func_kwargs=None,
                    resolution_method='robust_linear', resolution_kwargs=None, binning=default_split_binning):
    """Stage 2: `transforms` (stage 1's, relative to `base_transform_key`) with each group's own correction
    composed on - from its fused image registered against the next group's and resolved over all groups."""
    ngroups = max(groups) + 1
    if ngroups < 2:
        return transforms
    snapshot = snapshot_msims_transform(msims, STAGE1_KEY)
    graph = nx.Graph()
    try:
        for msim, transform in zip(msims, transforms):
            msi_utils.set_affine_transform(msim, transform, transform_key=STAGE1_KEY,
                                           base_transform_key=base_transform_key)
        grid, tile_size = group_grid(msims, STAGE1_KEY)
        # two groups held at a time: at a thousand planes all of them would not fit in memory
        previous = None
        for index in range(ngroups):
            raise_if_cancelled()
            current = fuse_group([msim for msim, group in zip(msims, groups) if group == index], STAGE1_KEY,
                                 grid, tile_size)
            if previous is not None:
                graph = nx.compose(graph, register_group_pair(previous, current, index - 1, pairwise_reg_func,
                                                              pairwise_reg_func_kwargs, binning))
            previous = current
    finally:
        restore_msims_transform(msims, STAGE1_KEY, snapshot)

    logging.info(f'Split registration: {ngroups} groups, {graph.number_of_edges()} group pairs registered')
    group_params, _ = groupwise_resolution(graph, method=resolution_method, **(resolution_kwargs or {}))
    # each group's correction acts in world space, after the stage-1 transform
    return [param_utils.matmul_xparams(group_params[group], transform) for transform, group in zip(transforms, groups)]
