"""A robust groupwise resolution method: multiview_stitcher's linear_two_pass as the solver of an iteratively
reweighted least squares. See notes/multiview_stitcher.md."""

import logging
import time
from contextlib import contextmanager

import numpy as np
import xarray as xr
from multiview_stitcher import transformation
from multiview_stitcher.param_resolution import linear_two_pass, register_groupwise_resolution_method
from multiview_stitcher.param_resolution.linear_two_pass import groupwise_resolution_linear_two_pass
from threadpoolctl import threadpool_limits

from muvis_align.image.util import xaffine_from_matrix
from muvis_align.util import raise_if_cancelled

ROBUST_LINEAR = 'robust_linear'
default_robust_rounds = 10
default_robust_tolerance = 0.01
# a few edges keep flipping between rounds on a large graph, so the largest change never settles
default_robust_change_quantile = 0.99


def scalar(value, default=1.0):
    if value is None:
        return default
    if isinstance(value, xr.DataArray):
        value = value.data
    return float(np.mean(value))


def find_reference_view(graph, weight_key='quality'):
    """The node with the largest summed edge weight, as mv_graph.get_node_with_maximal_edge_weight_sum_from_graph
    picks it, in one pass over the edges rather than one per node."""
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


def _build_params_from_templates(nodes, translations, rotations, scales, transform, ndim, x_in=None, x_out=None):
    """linear_two_pass._build_params_from_components with each transform a copy of a cached xarray: building them
    from scratch, twice a solve, is most of a round's time."""
    params = {}
    reorder = None
    for node in nodes:
        linear = np.eye(ndim)
        if transform == 'rigid':
            linear = linear_two_pass._vector_to_rotation(rotations[node], ndim)
        elif transform != 'translation':
            linear = np.exp(scales[node]) * linear_two_pass._vector_to_rotation(rotations[node], ndim)
        matrix = np.eye(ndim + 1)
        matrix[:ndim, :ndim] = linear
        matrix[:ndim, ndim] = translations[node]
        xparams = xaffine_from_matrix(matrix)
        if reorder is None:
            reorder = (x_in is not None and x_out is not None
                       and not (np.array_equal(xparams.coords['x_in'], x_in)
                                and np.array_equal(xparams.coords['x_out'], x_out)))
        params[node] = xparams.sel(x_in=x_in, x_out=x_out) if reorder else xparams
    return params


def _edge_residuals(g_reg, params, ndim=None):
    """param_resolution.utils.compute_edge_residuals on plain matrices: it builds an unused identity xarray per node
    and converts an xarray for every point it transforms."""
    matrices = {node: np.asarray(param.data if isinstance(param, xr.DataArray) else param, dtype=float)
                for node, param in params.items()}
    residuals = {}
    for edge in g_reg.edges:
        node1, node2 = sorted(edge)
        lower, upper = g_reg.edges[edge]['bbox'].data
        vertices = np.array(list(np.ndindex(tuple([2] * len(lower))))) * (upper - lower) + lower
        affine = g_reg.edges[edge]['transform']
        affine = np.asarray(affine.data if isinstance(affine, xr.DataArray) else affine)
        points1 = transformation.transform_pts(vertices, matrices[node1])
        points2 = transformation.transform_pts(transformation.transform_pts(vertices, affine), matrices[node2])
        residuals[(node1, node2)] = float(np.sqrt(np.mean(np.sum((points1 - points2) ** 2, axis=1))))
    return residuals


@contextmanager
def _numpy_internals():
    # only while our own solves run: the originals stay for any other caller
    originals = linear_two_pass._build_params_from_components, linear_two_pass.compute_edge_residuals
    linear_two_pass._build_params_from_components = _build_params_from_templates
    linear_two_pass.compute_edge_residuals = _edge_residuals
    try:
        yield
    finally:
        linear_two_pass._build_params_from_components, linear_two_pass.compute_edge_residuals = originals


def groupwise_resolution_robust_linear(g_reg_component_tp, reference_view=None, transform='rigid', scale=None,
                                       rounds=default_robust_rounds, tolerance=default_robust_tolerance,
                                       change_quantile=default_robust_change_quantile, progress=None, **kwargs):
    """linear_two_pass re-solved up to `rounds` times, each edge weighted by its pair quality times a Cauchy weight
    of its last residual, 1 / (1 + (residual / scale)^2): bad pairs stop pulling the fit without being cut.
    Stops early once the `change_quantile` of the residuals' changes in a round is within `tolerance * scale`, as
    the weights then hardly change. `scale` defaults to half the voxel diagonal. Returns the round with the lowest
    median residual. `progress` is called after each round. Per connected component and timepoint, as any resolver."""
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
    # the reweighting can swing between a good and a bad fit, so the last round is not necessarily the best
    best_params, best_info, best_residual = None, None, np.inf
    for round_index in range(rounds):
        raise_if_cancelled()
        for edge in edges:
            graph.edges[edge]['quality'] = qualities[edge] * weights[edge]
        round_start, round_cpu = time.perf_counter(), time.process_time()
        # pruning off: every edge stays in, down-weighted instead. One BLAS thread: a round is many tiny numpy calls,
        # which a pool spread over every core only slows, its threads spinning
        with _numpy_internals(), threadpool_limits(limits=1):
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
        changes = residual_changes(last_residuals, residuals)
        last_residuals = residuals
        change = np.quantile(changes, change_quantile) if len(changes) else np.inf
        largest_change = max(changes, default=np.inf)
        median_residual = float(np.median(list(residuals.values())))
        if median_residual < best_residual:
            best_params, best_info, best_residual = params, info, median_residual
        logging.info(f'Robust linear resolution: round {round_index + 1}/{rounds}, {len(edges)} edges,'
                     f' median residual {median_residual:.3g},'
                     f' change {change_quantile:.0%} {change:.3g} (largest {largest_change:.3g},'
                     f' tolerance {tolerance * scale:.3g}) in {time.perf_counter() - round_start:.0f}s,'
                     f' {time.process_time() - round_cpu:.0f}s cpu')
        if progress is not None:
            progress()
        if change <= tolerance * scale:
            logging.info(f'Robust linear resolution: converged after {round_index + 1} rounds')
            break
    if best_params is None:
        return params, info
    return best_params, best_info


def residual_changes(last_residuals, residuals):
    if last_residuals is None:
        return []
    return [abs(residuals[edge] - last_residuals[edge]) for edge in residuals
            if np.isfinite(residuals[edge]) and np.isfinite(last_residuals[edge])]


register_groupwise_resolution_method(ROBUST_LINEAR, groupwise_resolution_robust_linear)
