"""A robust groupwise resolution method: multiview_stitcher's linear_two_pass as the solver of an iteratively
reweighted least squares. See notes/multiview_stitcher.md."""
import logging

import numpy as np
import xarray as xr
from multiview_stitcher.param_resolution import register_groupwise_resolution_method
from multiview_stitcher.param_resolution.linear_two_pass import groupwise_resolution_linear_two_pass

from muvis_align.util import raise_if_cancelled

ROBUST_LINEAR = 'robust_linear'
default_robust_rounds = 10
default_robust_tolerance = 0.01


def scalar(value, default=1.0):
    if value is None:
        return default
    if isinstance(value, xr.DataArray):
        value = value.data
    return float(np.mean(value))


def find_reference_view(graph, weight_key='quality'):
    """The node with the largest summed edge weight, as mv_graph.get_node_with_maximal_edge_weight_sum_from_graph
    picks it, in one pass over the edges rather than one per node (hours at 34k nodes)."""
    totals = dict.fromkeys(graph.nodes, 0.0)
    for node1, node2, data in graph.edges(data=True):
        weight = data[weight_key]
        weight = float(np.sum(weight.data if isinstance(weight, xr.DataArray) else weight))
        totals[node1] += weight
        totals[node2] += weight
    return max(totals, key=totals.get)


def voxel_diagonal(graph):
    """global_optimization's default abs_tol: the largest voxel diagonal among the views."""
    return max(float(np.sqrt(sum(value ** 2 for value in graph.nodes[node]['stack_props']['spacing'].values())))
               for node in graph.nodes)


def groupwise_resolution_robust_linear(g_reg_component_tp, reference_view=None, transform='rigid', scale=None,
                                       rounds=default_robust_rounds, tolerance=default_robust_tolerance, **kwargs):
    """linear_two_pass re-solved up to `rounds` times, each edge weighted by its pair quality times a Cauchy weight
    of its last residual, 1 / (1 + (residual / scale)^2): bad pairs stop pulling the fit without being cut.
    Stops early once no residual moved more than `tolerance * scale` in a round, as the weights then hardly change.
    `scale` defaults to half the voxel diagonal. Per connected component and timepoint, as any resolver."""
    graph = g_reg_component_tp.copy()
    if reference_view is None or reference_view not in graph:
        reference_view = find_reference_view(graph)
    if scale is None:
        scale = 0.5 * voxel_diagonal(graph)
    edges = [tuple(sorted(edge)) for edge in graph.edges]
    qualities = {edge: scalar(graph.edges[edge].get('quality')) for edge in edges}
    qualities = {edge: quality if np.isfinite(quality) else 0.0 for edge, quality in qualities.items()}
    weights = dict.fromkeys(edges, 1.0)
    params, info, last_residuals = None, None, None
    for round_index in range(rounds):
        raise_if_cancelled()
        for edge in edges:
            graph.edges[edge]['quality'] = qualities[edge] * weights[edge]
        # pruning off: every edge stays in, down-weighted instead
        params, info = groupwise_resolution_linear_two_pass(
            graph, reference_view=reference_view, transform=transform, weight_mode='quality',
            residual_threshold=np.inf, keep_mst=False, **kwargs)
        metrics = info['metrics']
        if metrics is None:
            break
        residuals = {tuple(sorted((node1, node2))): residual
                     for node1, node2, residual in zip(metrics['u'], metrics['v'], metrics['residual'])}
        weights = {edge: 1.0 / (1.0 + (residuals[edge] / scale) ** 2) if np.isfinite(residuals[edge]) else 0.0
                   for edge in edges}
        change = largest_residual_change(last_residuals, residuals)
        last_residuals = residuals
        logging.info(f'Robust linear resolution: round {round_index + 1}/{rounds}, {len(edges)} edges,'
                     f' median residual {np.median(list(residuals.values())):.3g}, largest change {change:.3g}')
        if change <= tolerance * scale:
            logging.info(f'Robust linear resolution: converged after {round_index + 1} rounds')
            break
    return params, info


def largest_residual_change(last_residuals, residuals):
    if last_residuals is None:
        return np.inf
    changes = [abs(residuals[edge] - last_residuals[edge]) for edge in residuals
               if np.isfinite(residuals[edge]) and np.isfinite(last_residuals[edge])]
    return max(changes, default=0.0)


register_groupwise_resolution_method(ROBUST_LINEAR, groupwise_resolution_robust_linear)
