"""Split registration ('split' pairing): each z-plane's (or channel's) tiles registered in x/y first - the normal
pair and global registration, on the pairs within a group only - then each group moved as one, from the tile pairs
across groups: stacked planes or overlaid channels, as registration_dimension says."""
import logging

import networkx as nx
import numpy as np
from multiview_stitcher import param_utils
from multiview_stitcher.param_resolution import groupwise_resolution

from muvis_align.robust_resolution import default_robust_rounds, scalar, voxel_diagonal
from muvis_align.util import raise_if_cancelled

SPLIT = 'split'


def split_groups(positions, sources, dimension=None):
    """Each source's group index: its channel label for registration_dimension 'c', else its z-plane."""
    if dimension == 'c':
        keys = [(source.get_channels() or [{}])[0].get('label', '') for source in sources]
    else:
        keys = [round(float(position.get('z', 0.0)), 9) for position in positions]
    order = sorted(set(keys))
    return [order.index(key) for key in keys]


def cross_group_edges(graph, groups):
    return [edge for edge in graph.edges if groups[edge[0]] != groups[edge[1]]]


def within_group_graph(graph, groups):
    """`graph` without its edges across groups; every node kept, a lone tile's group included."""
    within = graph.copy()
    within.remove_edges_from(cross_group_edges(graph, groups))
    return within


def fit_transform(source_points, target_points, weights, transform_type='rigid'):
    """The homogeneous matrix M of `transform_type` best mapping source to target points, weighted least squares."""
    weights = weights / np.sum(weights)
    ndim = source_points.shape[1]
    source_mean, target_mean = weights @ source_points, weights @ target_points
    source_centred, target_centred = source_points - source_mean, target_points - target_mean
    if transform_type == 'translation':
        linear = np.eye(ndim)
    elif transform_type == 'affine':
        root_weights = np.sqrt(weights)[:, None]
        linear = np.linalg.lstsq(source_centred * root_weights, target_centred * root_weights, rcond=None)[0].T
    else:
        # rigid / similarity: weighted Kabsch (Umeyama), a reflection flipped back into a rotation
        u, singular, vt = np.linalg.svd((target_centred * weights[:, None]).T @ source_centred)
        correction = np.ones(ndim)
        correction[-1] = np.sign(np.linalg.det(u @ vt))
        linear = u @ np.diag(correction) @ vt
        if transform_type == 'similarity':
            linear *= np.sum(singular * correction) / (weights @ np.sum(source_centred ** 2, axis=1))
    matrix = np.eye(ndim + 1)
    matrix[:ndim, :ndim] = linear
    matrix[:ndim, ndim] = target_mean - linear @ source_mean
    return matrix


def weighted_median(values, weights):
    """Per column: the value at half the total weight."""
    medians = []
    for column in np.asarray(values).T:
        order = np.argsort(column)
        cumulative = np.cumsum(weights[order])
        medians.append(column[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    return np.array(medians)


def transform_points(points, matrix):
    return points @ matrix[:-1, :-1].T + matrix[:-1, -1]


def fit_group_pair(sources, targets, qualities, transform_type, scale, rounds=default_robust_rounds):
    """One transform for all of a group pair's tile pairs (each a set of bead points), each tile pair weighted by
    its quality x the Cauchy weight of its residual, 1 / (1 + (residual / scale)^2); also those Cauchy weights."""
    ndim = sources.shape[2]
    matrix = np.eye(ndim + 1)
    # a start the bad pairs cannot drag: a least-squares first fit would be pulled by all of them
    matrix[:ndim, ndim] = weighted_median((targets - sources).mean(axis=1), qualities)
    cauchy = np.ones(len(qualities))
    for _ in range(rounds):
        residuals = np.sqrt(np.mean(np.sum((transform_points(sources, matrix) - targets) ** 2, axis=2), axis=1))
        cauchy = 1 / (1 + (residuals / scale) ** 2)
        weights = np.repeat(qualities * cauchy, sources.shape[1])
        if np.sum(weights) > 0:
            matrix = fit_transform(sources.reshape(-1, ndim), targets.reshape(-1, ndim), weights, transform_type)
    return matrix, cauchy


def tile_pair_beads(graph, edge, transforms, groups, t):
    """The edge's overlap corners placed by stage 1 in both tiles: (points in the lower group's tile, the same points
    in the higher group's). The edge's own transform P maps its first node's frame to its second's (p_a x ~ p_b P x),
    so between stage-1 placements it is T_b P T_a^-1."""
    first, second = sorted(edge)
    data = graph.edges[edge]
    lower, upper = np.asarray(data['bbox'].sel(t=t).data)
    corners = np.array(list(np.ndindex(*([2] * len(lower)))))
    vertices = corners * (upper - lower) + lower
    pair_matrix = np.asarray(data['transform'].sel(t=t).data)
    first_points = transform_points(vertices, np.asarray(transforms[first].sel(t=t).data))
    second_points = transform_points(vertices, np.asarray(transforms[second].sel(t=t).data) @ pair_matrix)
    if groups[first] < groups[second]:
        return first_points, second_points
    return second_points, first_points


def build_group_graph(graph, transforms, groups, transform_type='rigid', rounds=default_robust_rounds):
    """A graph of the groups, one edge per group pair with tile pairs between them: its transform fitted to theirs
    (in the stage-1 placement), its quality their mean weighted as the fit weighted them."""
    group_graph = nx.Graph()
    for node in graph.nodes:
        if groups[node] not in group_graph:
            group_graph.add_node(groups[node], stack_props=graph.nodes[node]['stack_props'])
    edges_by_group_pair = {}
    for edge in cross_group_edges(graph, groups):
        group_pair = tuple(sorted((groups[edge[0]], groups[edge[1]])))
        edges_by_group_pair.setdefault(group_pair, []).append(edge)
    scale = 0.5 * voxel_diagonal(graph)
    for group_pair, edges in edges_by_group_pair.items():
        raise_if_cancelled()
        template = graph.edges[edges[0]]
        timepoints = template['transform'].coords['t'].values
        qualities = np.array([scalar(graph.edges[edge].get('quality')) for edge in edges])
        usable = np.isfinite(qualities) & (qualities > 0)
        qualities = np.where(usable, qualities, 0.0)
        if not np.any(usable):
            logging.warning(f'Split registration: groups {group_pair} have no usable tile pair')
        else:
            matrices, group_qualities, boxes = [], [], []
            for t in timepoints:
                beads = [tile_pair_beads(graph, edge, transforms, groups, t) for edge in edges]
                sources = np.array([source for source, _ in beads])
                targets = np.array([target for _, target in beads])
                matrix, cauchy = fit_group_pair(sources, targets, qualities, transform_type, scale, rounds)
                matrices.append(matrix)
                group_qualities.append(np.sum(qualities[usable] * cauchy[usable]) / np.sum(cauchy[usable]))
                points = sources.reshape(-1, sources.shape[2])
                boxes.append([points.min(axis=0), points.max(axis=0)])
            group_graph.add_edge(*group_pair,
                                 transform=template['transform'].copy(data=np.array(matrices)),
                                 quality=template['quality'].copy(data=np.array(group_qualities)),
                                 bbox=template['bbox'].copy(data=np.array(boxes)),
                                 overlap=float(sum(scalar(graph.edges[edge].get('overlap'), 0.0) for edge in edges)))
    if not nx.is_connected(group_graph):
        logging.warning(f'Split registration: the groups fall apart into {nx.number_connected_components(group_graph)}'
                        f' parts without tile pairs between them')
    return group_graph, sum(len(edges) for edges in edges_by_group_pair.values())


def register_groups(graph, transforms, groups, resolution_method='robust_linear', resolution_kwargs=None,
                    rounds=default_robust_rounds):
    """Stage 2: `transforms` (stage 1's, one a node of the pair `graph`) with each group's own correction composed
    on - from the tile pairs across groups, resolved over all groups."""
    ngroups = max(groups) + 1
    if ngroups < 2:
        return transforms
    transform_type = (resolution_kwargs or {}).get('transform', 'rigid')
    group_graph, ntile_pairs = build_group_graph(graph, transforms, groups, transform_type=transform_type,
                                                 rounds=rounds)
    logging.info(f'Split registration: {ngroups} groups, {group_graph.number_of_edges()} group pairs'
                 f' from {ntile_pairs} tile pairs')
    if group_graph.number_of_edges() == 0:
        return transforms
    group_params, _ = groupwise_resolution(group_graph, method=resolution_method, **(resolution_kwargs or {}))
    # each group's correction acts in world space, after the stage-1 transform
    return [param_utils.matmul_xparams(group_params[group], transform) for transform, group in zip(transforms, groups)]
