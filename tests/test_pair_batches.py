"""Pairs are registered and measured in batches, so a large project reports progress and releases
memory between them - rather than multiview_stitcher planning every pair in one graph (115549
pairs added 110GB and reported nothing in 9 hours). Batched results must match a single call."""

import networkx as nx
import numpy as np
import pytest
from multiview_stitcher import param_utils

from muvis_align.image.util import batch_graph_edges
from muvis_align.metrics import calc_pair_metrics


def test_batches_cover_every_edge_once_keeping_attributes():
    graph = nx.Graph()
    for index in range(7):
        graph.add_edge(index, index + 1, overlap=index)
    graph.add_node(20, stack_props='props')
    graph.add_edge(20, 0)

    batches = batch_graph_edges(graph, 3)

    assert [batch.number_of_edges() for batch in batches] == [3, 3, 2]
    edges = [frozenset(edge) for batch in batches for edge in batch.edges]
    assert sorted(edges, key=sorted) == sorted((frozenset(edge) for edge in graph.edges), key=sorted)
    assert all(batch.edges[edge].get('overlap') == graph.edges[edge].get('overlap')
               for batch in batches for edge in batch.edges)
    assert any(batch.nodes[20].get('stack_props') == 'props' for batch in batches if 20 in batch)


def test_graph_without_edges_is_one_batch():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))

    batches = batch_graph_edges(graph, 2)

    assert len(batches) == 1 and batches[0].number_of_nodes() == 3


@pytest.fixture(scope='module')
def register_msims():
    from muvis_align.MVSRegistration import MVSRegistration

    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[f'data/S000/S000_00{y}_00{x}.ome.zarr' for y in range(2) for x in range(2)],
        output_path='../../output/test_pair_batches/',
    )
    reg.init_data()
    reg.preprocess(reg.msims)
    return reg.register_msims, reg.source_transform_key


def test_batched_pair_metrics_match_one_call(register_msims):
    msims, transform_key = register_msims
    graph = nx.Graph()
    for pair in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        graph.add_edge(*pair, transform=param_utils.identity_transform(ndim=2), quality=0.5)

    whole = calc_pair_metrics(msims, graph, ['ncc'], transform_key)
    batched = calc_pair_metrics(msims, graph, ['ncc'], transform_key, n_parallel_pairs=1)

    assert set(batched['pairs']) == set(whole['pairs'])
    for pair, value in whole['pairs'].items():
        assert np.isclose(batched['pairs'][pair]['transform']['ncc'], value['transform']['ncc'],
                          equal_nan=True)
    assert batched['summary']['transform']['quality'] == whole['summary']['transform']['quality']
