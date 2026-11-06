"""What-if global registration from a project's saved pair registration only: no image data is read.

    python testing/global_solve_whatif.py <project output folder> [--max-shift 10 20] [--min-quality 0.05]

Re-solves the saved pairs (pair_mappings.json) with the robust linear solver as the plugin does, once as saved (the
project's own minimum pair quality) and once per --max-shift, leaving out pairs whose shift from the stage
position is larger. For each it reports the pairs left out, the residuals, and how far tiles move from the
as-saved solve. Where the same file name is in two folders (channels), it also compares their solved layouts:
two channels of the same stage tiles should agree.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import yaml
from multiview_stitcher.param_resolution import groupwise_resolution

from muvis_align.image.util import data_array_from_values, xaffine_from_matrix
from muvis_align.robust_resolution import ROBUST_LINEAR, _edge_residuals, find_reference_view


def project_settings(output):
    path = os.path.join(output, 'parameters.yaml')
    registration = {}
    if os.path.exists(path):
        with open(path, encoding='utf8') as file:
            registration = (yaml.safe_load(file) or {}).get('registration', {})
    quality = registration.get('post_registration_quality_threshold')
    return float(quality) if quality not in (None, '') else 0.0, registration.get('transform_type') or 'rigid'


def pixel_spacing(output):
    """The tiles' pixel size, from mappings.csv's physical and pixel positions."""
    ratios = []
    with open(os.path.join(output, 'mappings.csv'), encoding='utf8') as file:
        for row in csv.reader(file):
            values = row[2:6]
            if row[0] != 'id' and all(values) and abs(float(values[0])) > 1:
                ratios.append(float(row[5]) / float(row[2]))
    return float(np.median(np.abs(ratios))) if ratios else 1.0


def load_pairs(output):
    """(filenames, the pairs graph as the plugin loads it, each edge's shift as a plain vector)."""
    pairs = json.load(open(os.path.join(output, 'pair_mappings.json'), encoding='utf8'))
    filenames, graph = [], nx.Graph()
    index = {}
    for key, value in pairs.items():
        if value.get('kind') is None and 'bbox' in value:
            names = json.loads(key)
            for name in names:
                if name not in index:
                    index[name] = len(filenames)
                    filenames.append(name)
                    graph.add_node(index[name])
            matrix = np.array(value['mapping'], dtype=float)
            graph.add_edge(index[names[0]], index[names[1]], transform=xaffine_from_matrix(matrix, with_t=True),
                           matrix=matrix, quality=float(value.get('quality', 0)),
                           bbox=data_array_from_values(value['bbox']),
                           shift=float(np.linalg.norm(matrix[:-1, -1])))
    return filenames, graph


def solve(graph, spacing, transform, min_quality, max_shift=None):
    """(each tile's translation, residual per pair in the solve, pairs left out, tiles left without pairs)."""
    left_out = [edge for edge, data in graph.edges.items()
                if not data['quality'] > min_quality or (max_shift is not None and data['shift'] > max_shift)]
    resolve = graph.copy()
    resolve.remove_edges_from(left_out)
    for node in resolve.nodes:
        resolve.nodes[node]['stack_props'] = {'spacing': {'y': spacing, 'x': spacing}}
    kwargs = {'transform': transform}
    if resolve.number_of_nodes() > 2 and nx.is_connected(resolve):
        kwargs['reference_view'] = find_reference_view(resolve)
    params, _ = groupwise_resolution(resolve, method=ROBUST_LINEAR, **kwargs)
    matrices = {node: np.asarray(param).squeeze() for node, param in params.items()}
    plain = nx.Graph()
    plain.add_edges_from((edge[0], edge[1], {'transform': resolve.edges[edge]['matrix'],
                                              'bbox': resolve.edges[edge]['bbox']}) for edge in resolve.edges)
    residuals = _edge_residuals(plain, matrices)
    without = [node for node in resolve.nodes if resolve.degree(node) == 0 and graph.degree(node) > 0]
    return {node: matrix[:-1, -1] for node, matrix in matrices.items()}, residuals, left_out, without


def group_offsets(filenames, translations, reference):
    """Each tile's translation change from `reference`, less its folder's median change (a channel's common shift)."""
    folders = defaultdict(list)
    for node, name in enumerate(filenames):
        folders[os.path.dirname(name)].append(node)
    changes = {}
    for nodes in folders.values():
        deltas = np.array([translations[node] - reference[node] for node in nodes])
        median = np.median(deltas, axis=0)
        changes.update({node: float(np.linalg.norm(delta - median)) for node, delta in zip(nodes, deltas)})
    return changes


def channel_agreement(filenames, graph, translations):
    """For each pair present in two folders under the same file names, how far apart the two folders' solved
    relative positions are."""
    by_name = defaultdict(dict)
    for node, name in enumerate(filenames):
        by_name[os.path.basename(name)][os.path.dirname(name)] = node
    folders = sorted({os.path.dirname(name) for name in filenames})
    differences = []
    if len(folders) == 2:
        first, second = folders
        for node1, node2 in graph.edges:
            names = os.path.basename(filenames[node1]), os.path.basename(filenames[node2])
            if os.path.dirname(filenames[node1]) == first and all(second in by_name[name] for name in names):
                other1, other2 = by_name[names[0]][second], by_name[names[1]][second]
                relative1 = translations[node1] - translations[node2]
                relative2 = translations[other1] - translations[other2]
                differences.append(float(np.linalg.norm(relative1 - relative2)))
    return differences


def describe(values, unit=''):
    if not values:
        return 'none'
    values = np.asarray(values)
    return (f'median {np.median(values):.3g}{unit}, 90% {np.percentile(values, 90):.3g}{unit},'
            f' max {values.max():.3g}{unit}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', help="the project's output folder")
    parser.add_argument('--max-shift', type=float, nargs='*', default=[10, 20],
                        help='shift limits to try, in physical units (default 10 20)')
    parser.add_argument('--min-quality', type=float, help="minimum pair quality (default: the project's)")
    args = parser.parse_args()

    min_quality, transform = project_settings(args.output)
    if args.min_quality is not None:
        min_quality = args.min_quality
    spacing = pixel_spacing(args.output)
    filenames, graph = load_pairs(args.output)
    folders = sorted({os.path.dirname(name) for name in filenames})
    print(f'{len(filenames)} tiles, {graph.number_of_edges()} pairs in {len(folders)} folder(s); pixel size'
          f' {spacing:.4g}; transform {transform}; minimum quality {min_quality:g}')
    used = [data['shift'] for data in graph.edges.values() if data['quality'] > min_quality]
    print(f'pair shifts (pairs above the minimum quality): {describe(used)}')

    saved = json.load(open(os.path.join(args.output, 'mappings.json'), encoding='utf8'))
    saved_translations = {node: np.array(saved[name]['affine'], dtype=float)[:, -1]
                          for node, name in enumerate(filenames) if name in saved}
    baseline = None
    for max_shift in [None] + list(args.max_shift):
        translations, residuals, left_out, without = solve(graph, spacing, transform, min_quality, max_shift)
        name = 'as saved' if max_shift is None else f'max shift {max_shift:g}'
        print(f'\n== {name}: {len(left_out)} pairs left out, {len(without)} tiles left without pairs')
        print(f'   residuals: {describe(list(residuals.values()))}, over 10: '
              f'{sum(value > 10 for value in residuals.values())}')
        if baseline is None:
            baseline = translations
            if len(saved_translations) == len(filenames):
                print(f'   vs saved mappings.json (less each folder\'s common shift): '
                      f'{describe(list(group_offsets(filenames, translations, saved_translations).values()))}')
        else:
            moved = list(group_offsets(filenames, translations, baseline).values())
            print(f'   tiles moved from "as saved": {describe(moved)}; over 5: {sum(value > 5 for value in moved)}')
        agreement = channel_agreement(filenames, graph, translations)
        if agreement:
            print(f'   channels\' solved layouts differ per pair: {describe(agreement)};'
                  f' within 5: {sum(value <= 5 for value in agreement)} of {len(agreement)}')


if __name__ == '__main__':
    main()
