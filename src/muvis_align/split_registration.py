"""Split registration ('split' pairing): each z-plane's (or channel's) tiles registered in x/y first - the normal
pair and global registration, with pairs only within a group - then the stitched groups against each other as
whole images, consecutive groups paired, as registration_dimension says: stacked planes or overlaid channels."""
import logging
import os

import dask
import networkx as nx
import numpy as np
from multiview_stitcher import fusion, msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.metrics import tile_pair_image_metrics
from multiview_stitcher.registration import compute_pairwise_registrations
from scipy.ndimage import gaussian_filter

from muvis_align.constants import default_split_binning, default_transform_key, split_grid_margin, split_smoothing
from muvis_align.image.util import (build_view_adjacency_graph, coarsest_level_within, msim_level_spacings,
                                    restore_msims_transform, set_msim_affine, snapshot_msims_transform)
from muvis_align.util import raise_if_cancelled

SPLIT = 'split'
# transient keys: the stage-1 transforms while the groups are fused, and each fused group's own placement
STAGE1_KEY = 'split_stage1'
GROUP_KEY = 'split_group'


def split_groups(positions, sources, dimension=None, input_groups=None):
    """Each source's group index: its channel (see channel_groups) for registration_dimension 'c', else its z-plane."""
    if dimension == 'c':
        return channel_groups(sources, input_groups)
    keys = [round(float(position.get('z', 0.0)), 9) for position in positions]
    order = sorted(set(keys))
    return [order.index(key) for key in keys]


def channel_groups(sources, input_groups=None):
    """Each source's channel index: the input path item it came from when there are several, else its metadata
    channel label in order of appearance, else (all labels alike) its own index."""
    if input_groups is not None and len(set(input_groups)) > 1:
        # in the items' order, not their files'
        keys = list(input_groups)
        order = sorted(set(keys))
    else:
        keys = [(source.get_channels() or [{}])[0].get('label') or '' for source in sources]
        if len(set(keys)) <= 1:
            keys = list(range(len(sources)))
        order = list(dict.fromkeys(keys))
    return [order.index(key) for key in keys]


def group_label(labels, separators='_- '):
    """The labels' common prefix up to a separator: 'S000' for 'S000_000_000' ... 'S000_007_007' (not 'S000_00')."""
    prefix = os.path.commonprefix(list(labels))
    if any(len(label) > len(prefix) and label[len(prefix)] not in separators for label in labels):
        cut = max(prefix.rfind(separator) for separator in separators)
        prefix = prefix[:cut + 1] if cut >= 0 else ''
    return prefix.rstrip(separators)


def within_group_pairs(pairs, groups):
    return [pair for pair in pairs if groups[pair[0]] == groups[pair[1]]]


def _level_sim(msim, spacing):
    """The msim's coarsest level no coarser than `spacing` - fusing from it resamples least."""
    level = coarsest_level_within(msim_level_spacings(msim), {'y': spacing, 'x': spacing})
    return msi_utils.get_sim_from_msim(msim, scale=f'scale{level}')


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
    size, with a margin - and the median tile size. On one grid two groups overlap in the whole frame, not a crop
    cut at an outline."""
    finest = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
    stack_props = [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in finest]
    lower = np.min([[props['origin'][dim] for dim in 'yx'] for props in stack_props], axis=0)
    upper = np.max([[props['origin'][dim] + props['shape'][dim] * props['spacing'][dim] for dim in 'yx']
                    for props in stack_props], axis=0)
    spacing = min(min(props['spacing'][dim] for dim in 'yx') for props in stack_props)
    shape = np.ceil((upper - lower) / spacing).astype(int)
    margin = np.ceil(shape * split_grid_margin).astype(int)
    lower, shape = lower - margin * spacing, shape + 2 * margin
    grid = {'origin': {dim: float(value) for dim, value in zip('yx', lower)},
            'spacing': {dim: spacing for dim in 'yx'},
            'shape': {dim: int(value) for dim, value in zip('yx', shape)}}
    tile_size = float(np.median([max(props['shape'][dim] * props['spacing'][dim] for dim in 'yx')
                                 for props in stack_props]))
    return grid, tile_size


def group_msim(data, spacing, origin):
    """A fused group's y/x image as a single-level msim placed by GROUP_KEY."""
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], scale={dim: spacing for dim in 'yx'},
                                      translation=origin, transform_key=GROUP_KEY)
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    # the only channel selected, so registration sees spatial dims alone
    return msi_utils.multiscale_sel_coords(msim, {'c': sim.coords['c'].values[0]})


def metrics_group_msim(data, grid, binning):
    """The unsmoothed fused group binned and cropped to its tiles, NaN between them as outside a tile's image, for
    the pair metrics to measure only where both groups have tiles; None if it has none."""
    shape = np.array(data.shape) // binning
    binned = (np.asarray(data[:shape[0] * binning, :shape[1] * binning], dtype=np.float32)
              .reshape(shape[0], binning, shape[1], binning).mean(axis=(1, 3)))
    mask = binned > 0
    if not np.any(mask):
        return None
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    binned = np.where(mask, binned, np.nan)[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    spacing = grid['spacing']['y']
    # a binned pixel's centre is the mean of its pixels' centres
    origin = {dim: grid['origin'][dim] + (start * binning + (binning - 1) / 2) * spacing
              for dim, start in zip('yx', (rows[0], cols[0]))}
    return group_msim(binned, spacing * binning, origin)


def fuse_group(msims, transform_key, grid, tile_size, metrics_binning=None):
    """One group's tiles fused under `transform_key` onto `grid` and smoothed, as an msim placed by GROUP_KEY, and
    given `metrics_binning` its metrics_group_msim (else None)."""
    spacing = grid['spacing']['y']
    sims = [_level_sim(msim, spacing) for msim in msims]
    with dask.config.set(scheduler='threads'):
        fused = np.asarray(fusion.fuse(sims, transform_key=transform_key, output_stack_properties=grid).data).squeeze()
    metrics_msim = metrics_group_msim(fused, grid, metrics_binning) if metrics_binning else None
    return group_msim(smooth_group(fused, spacing, tile_size), spacing, grid['origin']), metrics_msim


def group_pair_image_metrics(metrics_msims, pair_graph, index1, metric_funcs):
    """The image metrics of groups `index1` and `index1 + 1` under their registered pair transform."""
    if not metric_funcs or any(msim is None for msim in metrics_msims) or not pair_graph.number_of_edges():
        return {}
    graph = nx.relabel_nodes(pair_graph, {index1: 0, index1 + 1: 1})
    with dask.config.set({'scheduler': 'threads', 'optimization.fuse.active': False}):
        result = tile_pair_image_metrics(metrics_msims, base_transform_key=GROUP_KEY, pairs_graph=graph,
                                         metric_funcs=metric_funcs)
    values = result.get('pairs', {}).get((0, 1), {}).get(default_transform_key, {})
    return {metric: float(value) for metric, value in values.items()}


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
                    resolution_method='robust_linear', resolution_kwargs=None, binning=default_split_binning,
                    metric_funcs=None):
    """Stage 2: `transforms` (stage 1's, relative to `base_transform_key`) with each group's own correction
    composed on - from its fused image registered against the next group's and resolved over all groups - and the
    graph of those group pairs, each edge with its `metric_funcs` values under 'metrics'."""
    ngroups = max(groups) + 1
    if ngroups < 2:
        return transforms, nx.Graph()
    snapshot = snapshot_msims_transform(msims, STAGE1_KEY)
    graph = nx.Graph()
    try:
        for msim, transform in zip(msims, transforms):
            set_msim_affine(msim, transform, transform_key=STAGE1_KEY, base_transform_key=base_transform_key)
        grid, tile_size = group_grid(msims, STAGE1_KEY)
        # two groups held at a time: all of them may not fit in memory
        previous, previous_metrics = None, None
        for index in range(ngroups):
            raise_if_cancelled()
            current, current_metrics = fuse_group([msim for msim, group in zip(msims, groups) if group == index],
                                                  STAGE1_KEY, grid, tile_size,
                                                  metrics_binning=binning if metric_funcs else None)
            if previous is not None:
                pair_graph = register_group_pair(previous, current, index - 1, pairwise_reg_func,
                                                 pairwise_reg_func_kwargs, binning)
                if pair_graph.number_of_edges():
                    pair_graph.edges[index - 1, index]['metrics'] = group_pair_image_metrics(
                        [previous_metrics, current_metrics], pair_graph, index - 1, metric_funcs)
                graph = nx.compose(graph, pair_graph)
            previous, previous_metrics = current, current_metrics
    finally:
        restore_msims_transform(msims, STAGE1_KEY, snapshot)

    logging.info(f'Split registration: {ngroups} groups, {graph.number_of_edges()} group pairs registered')
    group_params, _ = groupwise_resolution(graph, method=resolution_method, **(resolution_kwargs or {}))
    # each group's correction acts in world space, after the stage-1 transform
    return ([param_utils.matmul_xparams(group_params[group], transform) for transform, group in zip(transforms, groups)],
            graph)
