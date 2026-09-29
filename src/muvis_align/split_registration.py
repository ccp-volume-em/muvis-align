"""Split registration ('split' pairing): each z-plane's (or channel's) tiles registered in x/y first - the normal
pair and global registration, with pairs only within a group - then the stitched groups against each other as
whole images, consecutive groups paired, as registration_dimension says: stacked planes or overlaid channels."""
import logging

import dask
import networkx as nx
import numpy as np
import xarray as xr
from multiview_stitcher import fusion, msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.param_resolution import groupwise_resolution
from scipy.ndimage import gaussian_filter, shift as shift_image
from skimage.registration import phase_cross_correlation

from muvis_align.constants import default_split_group_size, split_band_high, split_band_low
from muvis_align.image.util import restore_msims_transform, snapshot_msims_transform
from muvis_align.util import raise_if_cancelled

SPLIT = 'split'
# transient key: the stage-1 transforms while the groups are fused
STAGE1_KEY = 'split_stage1'


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


def band_pass_group(data, spacing, tile_size):
    """The fused group without what every tile repeats (fixed pattern, shading), and its mask (inside its tiles):
    left in, a tile's own pattern matches best at zero shift, whatever the content does."""
    data = np.asarray(data, dtype=np.float32)
    mask = data > 0
    if not np.any(mask):
        return data, mask
    filled = np.where(mask, data, data[mask].mean())
    band = (gaussian_filter(filled, tile_size * split_band_low / spacing)
            - gaussian_filter(filled, tile_size * split_band_high / spacing))
    return band, mask


def group_grid(msims, transform_key, max_size=default_split_group_size):
    """The one y/x grid every group is fused onto, the union of all tiles with its longest side at most `max_size`
    pixels, and the median tile size. On one grid a shift between two groups is a shift of their pixels."""
    finest = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
    stack_props = [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in finest]
    lower = np.min([[props['origin'][dim] for dim in 'yx'] for props in stack_props], axis=0)
    upper = np.max([[props['origin'][dim] + props['shape'][dim] * props['spacing'][dim] for dim in 'yx']
                    for props in stack_props], axis=0)
    finest_spacing = min(min(props['spacing'][dim] for dim in 'yx') for props in stack_props)
    spacing = max(float(np.max(upper - lower)) / max_size, finest_spacing)
    shape = np.ceil((upper - lower) / spacing).astype(int)
    grid = {'origin': {dim: float(value) for dim, value in zip('yx', lower)},
            'spacing': {dim: spacing for dim in 'yx'},
            'shape': {dim: int(value) for dim, value in zip('yx', shape)}}
    tile_size = float(np.median([max(props['shape'][dim] * props['spacing'][dim] for dim in 'yx')
                                 for props in stack_props]))
    return grid, tile_size


def fuse_group(msims, transform_key, grid, tile_size):
    """One group's tiles fused under `transform_key` onto `grid`, band-passed: (image, mask)."""
    spacing = grid['spacing']['y']
    sims = [_level_sim(msim, spacing) for msim in msims]
    with dask.config.set(scheduler='threads'):
        fused = fusion.fuse(sims, transform_key=transform_key, output_stack_properties=grid)
        return band_pass_group(np.asarray(fused.data).squeeze(), spacing, tile_size)


def masked_ncc(image1, mask1, image2, mask2, shift):
    """Normalised cross-correlation of the two images where both have data, the second moved by `shift` pixels."""
    moved = shift_image(image2, shift, order=1)
    moved_mask = shift_image(mask2.astype(np.float32), shift, order=0) > 0.5
    both = mask1 & moved_mask
    if np.count_nonzero(both) < 2:
        return 0.0
    values1, values2 = image1[both] - image1[both].mean(), moved[both] - moved[both].mean()
    denominator = np.sqrt(np.sum(values1 ** 2) * np.sum(values2 ** 2))
    return float(np.sum(values1 * values2) / denominator) if denominator > 0 else 0.0


def register_group_pair(group1, group2, grid, timepoints):
    """Two fused groups' edge for the group graph: masked phase correlation, which weighs only where both have
    data - an outline or background would pull it to zero shift. Its quality is the NCC at the shift found."""
    (image1, mask1), (image2, mask2) = group1, group2
    shift = phase_cross_correlation(image1, image2, reference_mask=mask1, moving_mask=mask2)[0]
    quality = max(masked_ncc(image1, mask1, image2, mask2, shift), 0.0)
    spacing = np.array([grid['spacing'][dim] for dim in 'yx'])
    matrix = np.eye(3)
    # the second group's content sits -shift from the first's, so the first's frame maps onto it by that
    matrix[:2, 2] = -shift * spacing
    lower = np.array([grid['origin'][dim] for dim in 'yx'])
    upper = lower + np.array([grid['shape'][dim] for dim in 'yx']) * spacing
    return {'transform': param_utils.affine_to_xaffine(matrix, t_coords=timepoints),
            'quality': xr.DataArray([quality] * len(timepoints), dims=['t'], coords={'t': timepoints}),
            'bbox': xr.DataArray([[lower, upper]] * len(timepoints), dims=['t', 'point_index', 'dim'],
                                 coords={'t': timepoints}),
            'overlap': float(np.count_nonzero(mask1 & mask2) * np.prod(spacing))}


def register_groups(msims, transforms, groups, base_transform_key, resolution_method='robust_linear',
                    resolution_kwargs=None, max_size=default_split_group_size):
    """Stage 2: `transforms` (stage 1's, relative to `base_transform_key`) with each group's own correction
    composed on - from its fused image registered against the next group's and resolved over all groups. The
    groups only translate: their images are registered by phase correlation."""
    ngroups = max(groups) + 1
    if ngroups < 2:
        return transforms
    timepoints = transforms[0].coords['t'].values
    snapshot = snapshot_msims_transform(msims, STAGE1_KEY)
    try:
        for msim, transform in zip(msims, transforms):
            msi_utils.set_affine_transform(msim, transform, transform_key=STAGE1_KEY,
                                           base_transform_key=base_transform_key)
        grid, tile_size = group_grid(msims, STAGE1_KEY, max_size=max_size)
        graph = nx.Graph()
        graph.add_nodes_from(range(ngroups), stack_props=grid)
        # two groups held at a time: at a thousand planes all of them would not fit in memory
        previous = None
        for index in range(ngroups):
            raise_if_cancelled()
            current = fuse_group([msim for msim, group in zip(msims, groups) if group == index], STAGE1_KEY,
                                 grid, tile_size)
            if previous is not None:
                graph.add_edge(index - 1, index, **register_group_pair(previous, current, grid, timepoints))
            previous = current
    finally:
        restore_msims_transform(msims, STAGE1_KEY, snapshot)

    qualities = [float(graph.edges[edge]['quality'].mean()) for edge in graph.edges]
    logging.info(f'Split registration: {ngroups} groups, {graph.number_of_edges()} group pairs registered,'
                 f' quality (NCC) median {np.median(qualities):.3f}, min {np.min(qualities):.3f}')
    group_params, _ = groupwise_resolution(graph, method=resolution_method, **(resolution_kwargs or {}))
    # each group's correction acts in world space, after the stage-1 transform
    return [param_utils.matmul_xparams(group_params[group], transform) for transform, group in zip(transforms, groups)]
