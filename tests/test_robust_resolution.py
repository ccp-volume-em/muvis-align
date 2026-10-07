import networkx as nx
import numpy as np
import xarray as xr
from multiview_stitcher import mv_graph
from multiview_stitcher.param_resolution import groupwise_resolution

from muvis_align.robust_resolution import ROBUST_LINEAR, find_reference_view


def affine(shift_y=0.0, shift_x=0.0):
    matrix = np.eye(3)
    matrix[:2, 2] = shift_y, shift_x
    return xr.DataArray(matrix, dims=['x_in', 'x_out'], coords={'x_in': ['y', 'x', '1'], 'x_out': ['y', 'x', '1']})


def grid_graph(rows=4, cols=4, size=100.0, step=90.0, outlier=None):
    """Tiles on a grid, overlapping their neighbours by 10: every pair registers to identity, bar `outlier`."""
    graph = nx.Graph()
    for row in range(rows):
        for col in range(cols):
            graph.add_node(row * cols + col, stack_props={
                'shape': {'y': int(size), 'x': int(size)}, 'spacing': {'y': 1.0, 'x': 1.0},
                'origin': {'y': row * step, 'x': col * step}, 'transform': affine()})
    for row in range(rows):
        for col in range(cols):
            for other_row, other_col in ((row, col + 1), (row + 1, col)):
                if other_row < rows and other_col < cols:
                    node, other = row * cols + col, other_row * cols + other_col
                    lower = np.array([other_row * step, other_col * step])
                    upper = np.array([row * step, col * step]) + size
                    graph.add_edge(node, other, transform=affine(), quality=0.9, overlap=0.1,
                                   bbox=xr.DataArray(np.array([lower, upper]), dims=['point_index', 'dim']))
    if outlier is not None:
        graph.edges[outlier]['transform'] = affine(0.0, 30.0)
    return graph


def largest_shift(params):
    return max(float(np.abs(np.asarray(value)[:2, 2]).max()) for value in params.values())


def test_reference_view_is_the_one_multiview_stitcher_picks():
    graph = grid_graph()
    rng = np.random.default_rng(0)
    for edge in graph.edges:
        graph.edges[edge]['quality'] = xr.DataArray(rng.random())

    assert find_reference_view(graph) == mv_graph.get_node_with_maximal_edge_weight_sum_from_graph(graph, 'quality')


def test_consistent_pairs_resolve_to_identity():
    params, _ = groupwise_resolution(grid_graph(), method=ROBUST_LINEAR, transform='translation')

    assert largest_shift(params) < 1e-6


def test_an_outlier_pair_pulls_the_fit_far_less_than_in_plain_least_squares():
    """Down-weighting by residual is the point: one bad pair must not drag its tiles along."""
    graph = grid_graph(outlier=(5, 6))
    plain, _ = groupwise_resolution(graph, method='linear_two_pass', transform='translation',
                                    residual_threshold=np.inf, keep_mst=False)
    robust, _ = groupwise_resolution(graph, method=ROBUST_LINEAR, transform='translation')

    assert largest_shift(plain) > 3
    assert largest_shift(robust) < 0.1 * largest_shift(plain)


def test_a_cancel_stops_the_robust_rounds():
    import pytest
    from muvis_align.util import OperationCancelled, cancellable, request_cancel

    with cancellable():
        request_cancel()
        with pytest.raises(OperationCancelled):
            groupwise_resolution(grid_graph(), method=ROBUST_LINEAR, transform='translation')



def scripted_solver(monkeypatch, residual_rounds, flipping_edge=None):
    """linear_two_pass replaced by one giving every edge the next round's residual, bar `flipping_edge`, which
    alternates between 0 and 1: the list of solves it saw."""
    from muvis_align import robust_resolution

    calls = []

    def solve(graph, **_):
        residual = residual_rounds[min(len(calls), len(residual_rounds) - 1)]
        calls.append(residual)
        edges = list(graph.edges)
        residuals = [float(len(calls) % 2) if tuple(sorted(edge)) == flipping_edge else residual for edge in edges]
        metrics = {'u': [edge[0] for edge in edges], 'v': [edge[1] for edge in edges], 'residual': residuals}
        return {node: affine() for node in graph.nodes}, {'metrics': metrics}

    monkeypatch.setattr(robust_resolution, 'groupwise_resolution_linear_two_pass', solve)
    return calls


def test_the_rounds_stop_once_no_residual_moves(monkeypatch):
    from muvis_align.robust_resolution import groupwise_resolution_robust_linear

    # with scale 1 the third round moves 0.005, within the default 0.01
    calls = scripted_solver(monkeypatch, [0.5, 0.2, 0.195, 0.195])
    groupwise_resolution_robust_linear(grid_graph(), transform='translation', scale=1.0, rounds=10)

    assert calls == [0.5, 0.2, 0.195]


def test_the_rounds_run_out_while_residuals_keep_moving(monkeypatch):
    from muvis_align.robust_resolution import groupwise_resolution_robust_linear

    calls = scripted_solver(monkeypatch, [0.1 * (index % 2) for index in range(10)])
    groupwise_resolution_robust_linear(grid_graph(), transform='translation', scale=1.0, rounds=4)

    assert len(calls) == 4


def test_one_flipping_edge_of_many_does_not_keep_the_rounds_going(monkeypatch):
    """On a large graph a few edges flip every round, so the largest change never settles: the 99th percentile does."""
    from muvis_align.robust_resolution import groupwise_resolution_robust_linear

    calls = scripted_solver(monkeypatch, [0.5, 0.2, 0.195, 0.195], flipping_edge=(0, 1))
    rounds = []
    groupwise_resolution_robust_linear(grid_graph(rows=8, cols=8), transform='translation', scale=1.0, rounds=10,
                                       progress=lambda: rounds.append(len(calls)))

    assert calls == [0.5, 0.2, 0.195]
    assert rounds == [1, 2, 3]
