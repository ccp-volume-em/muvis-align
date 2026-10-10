from contextlib import contextmanager, nullcontext
from functools import partial
import threading
import dask
import dask.local
import dask.threaded
#import frc
import multiview_stitcher.metrics
import networkx as nx
from multiview_stitcher import mv_graph
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from skimage.metrics import structural_similarity, normalized_mutual_information, mean_squared_error
from sklearn.metrics import euclidean_distances
from threadpoolctl import threadpool_limits
from xarray import DataArray

from muvis_align.constants import (default_pair_worker_tasks, default_pair_workers, default_quality_key,
                                   default_transform_key)
from muvis_align.image.util import image_reshape, get_msim_transform_keys
from muvis_align.util import (apply_transform, picklable, release_memory, result_unless_cancelled, rolling_map,
                              worker_process_pool)


def create_metric_methods(metric_methods, msim):
    data_range = np.iinfo(msim["scale0/image"].dtype).max
    all_metric_funcs = {
        'ncc': multiview_stitcher.metrics.normalized_cross_correlation,
        # the crops are single-channel: no channel axis
        'ssim': lambda im1, im2: structural_similarity(np.nan_to_num(im1), np.nan_to_num(im2),
                                                       data_range=data_range),
        'onmi': lambda im1, im2: normalized_mutual_information(np.nan_to_num(im1), np.nan_to_num(im2)) - 1,
        'mse': lambda im1, im2: 1 / mean_squared_error(im1, im2),
    }
    if metric_methods == 'all':
        metric_funcs = all_metric_funcs
    else:
        metric_funcs = {metric_method: all_metric_funcs[metric_method] for metric_method in metric_methods}
    return metric_funcs


def quality_to_scalar(value):
    # edge/mapping quality values are often an xr.DataArray with a 't' dim (e.g. from
    # register_pair_of_msims_over_time) - reduce to the plain scalar these callers display
    if isinstance(value, DataArray):
        if 't' in value.dims:
            value = value.sel(t=0)
        value = value.item()
    return value


def _scheduler(name):
    # callbacks given, so dask never swaps its global set: on a worker that raced napari's slicing cache
    get = partial(dask.threaded.get if name == 'threads' else dask.local.get_sync, callbacks=())
    if threading.current_thread() is threading.main_thread():
        return get

    def schedule(dsk, keys, **kwargs):
        # the config is global: set on a worker, the main thread's own computes (napari's slicing) keep dask's default
        if threading.current_thread() is threading.main_thread():
            return dask.threaded.get(dsk, keys, **kwargs)
        return get(dsk, keys, **kwargs)
    return schedule


@contextmanager
def _pair_metrics_compute(scheduler='threads'):
    """OpenBLAS is held to one thread per call: each pair's overlap mask is a matmul, and OpenBLAS
    starting its own pool from many threads at once crashed the process."""
    # fusion off for the fused key collision described in MVSRegistration.register_pairs
    with (dask.config.set({'scheduler': _scheduler(scheduler), 'optimization.fuse.active': False}),
          threadpool_limits(1, user_api='blas')):
        yield


def pair_image_metrics(msims, nodes, metric_methods, base_transform_key, query_transform_keys=None,
                       pairs_graph=None, scheduler=None):
    """tile_pair_image_metrics for one pair, given only its two msims (every call builds a sim for each msim
    it is given), with its keys renumbered back to `nodes`. Module-level, so a worker process can run it.
    `scheduler` None: the caller holds _pair_metrics_compute - never entered per call from many threads."""
    metric_funcs = create_metric_methods(metric_methods, msims[0])
    with (_pair_metrics_compute(scheduler) if scheduler is not None else nullcontext()):
        if pairs_graph is not None:
            result = multiview_stitcher.metrics.tile_pair_image_metrics(
                msims, base_transform_key=base_transform_key, pairs_graph=pairs_graph, metric_funcs=metric_funcs)
        else:
            result = multiview_stitcher.metrics.tile_pair_image_metrics(
                msims, base_transform_key=base_transform_key, query_transform_keys=query_transform_keys,
                metric_funcs=metric_funcs)
    for key in ('pairs', 'bboxes'):
        result[key] = {(nodes[fixed], nodes[moving]): value for (fixed, moving), value in result.get(key, {}).items()}
    return result


def map_pair_metrics(msims, edges, pair_arguments, workers, progress_factory=None, desc='Pair metrics',
                     threaded=True):
    """pair_image_metrics for each edge (a sorted node pair), one pair per worker process when its arguments
    pickle - on threads they are GIL-bound - else one per thread; results in completion order.
    `threaded` False: without worker processes, one pair at a time in the calling thread - the overlap mode's
    linprog (HiGHS) hung and crashed when called from changing threads. Worker processes are unaffected."""
    pool = None
    if workers > 1 and len(edges) > workers and picklable(pair_arguments(edges[0])):
        pool = worker_process_pool(workers, max_tasks_per_child=default_pair_worker_tasks)
    elif not threaded:
        workers = 1

    def pair_metrics(edge):
        arguments = pair_arguments(edge)
        if pool is None:
            return pair_image_metrics(*arguments)
        return result_unless_cancelled(pool.submit(pair_image_metrics, *arguments, scheduler='synchronous'))

    results = []
    progress = (progress_factory(total=len(edges), desc=desc) if progress_factory is not None else nullcontext(None))
    # on threads, set once here: BLAS thread limits changed from many threads at once crash the process
    compute = (_pair_metrics_compute('synchronous' if workers > 1 else 'threads') if pool is None
               else nullcontext())
    completed = (rolling_map(pair_metrics, edges, workers) if pool is not None or threaded
                 else ((edge, pair_metrics(edge)) for edge in edges))
    with (pool if pool is not None else nullcontext()), compute, progress as pbar:
        for count, (edge, result) in enumerate(completed, start=1):
            results.append(result)
            if count % (2 * workers) == 0:
                release_memory(generation=1)
            if pbar is not None:
                pbar.update(1)
    return results


def calc_pair_metrics(msims, pairs_graph, metric_methods, base_transform_key, reg_channel=None,
                      n_parallel_pairs=None, progress_factory=None):
    workers = n_parallel_pairs or default_pair_workers

    def pair_arguments(edge):
        # renumbered in order, so the lower index stays the fixed one
        nodes = tuple(sorted(edge))
        graph = nx.relabel_nodes(pairs_graph.edge_subgraph([edge]).copy(),
                                 {node: index for index, node in enumerate(nodes)})
        return [msims[node] for node in nodes], nodes, metric_methods, base_transform_key, None, graph

    metric_results = merge_metric_results(
        map_pair_metrics(msims, list(pairs_graph.edges), pair_arguments, workers, progress_factory))

    qualities = nx.get_edge_attributes(pairs_graph, default_quality_key)

    quality_values = []
    for pair_key, value in qualities.items():
        value = quality_to_scalar(value)
        if value:
            metric_results['pairs'][pair_key][default_transform_key][default_quality_key] = value
            quality_values.append(value)

    value = float(np.nanmean(quality_values)) if quality_values else None
    metric_results['summary'][default_transform_key][default_quality_key] = value

    return metric_results


def _bbox_area(bbox):
    return float(np.prod(np.asarray(bbox['upper']) - np.asarray(bbox['lower']))) if bbox is not None else 0.0


def merge_metric_results(results):
    """One tile_pair_image_metrics() result from several over disjoint sets of pairs. Its summary
    weights each pair by its overlap polygon's area, which is not returned: the pair's comparison
    bbox area stands in, equal for axis-aligned tiles and close for slightly rotated ones."""
    merged = {'pairs': {}, 'bboxes': {}, 'summary': {}}
    for result in results:
        merged['pairs'].update(result.get('pairs', {}))
        merged['bboxes'].update(result.get('bboxes', {}))
        for candidate_key, metric_values in result.get('summary', {}).items():
            # kept even with no metrics asked for: the caller adds the quality under it
            summary = merged['summary'].setdefault(candidate_key, {})
            for metric_key in metric_values:
                summary[metric_key] = np.nan
    for candidate_key, metric_values in merged['summary'].items():
        for metric_key in metric_values:
            values_and_weights = [(float(pair[candidate_key][metric_key]), _bbox_area(merged['bboxes'].get(pair_key)))
                                  for pair_key, pair in merged['pairs'].items()
                                  if metric_key in pair.get(candidate_key, {})]
            valid = [(value, weight) for value, weight in values_and_weights if not np.isnan(value) and weight > 0]
            if valid:
                metric_values[metric_key] = float(sum(value * weight for value, weight in valid)
                                                  / sum(weight for _, weight in valid))
    return merged


def calc_global_metrics(msims, base_transform_key, reg_transform_key, metric_methods, reg_channel=None,
                        reg_results=None, n_parallel_pairs=None, progress_factory=None):
    query_transform_keys = [base_transform_key, reg_transform_key]
    if reg_results is not None:
        # only the registered pairs, each on its own: over all msims at once the overlap mode measures every
        # overlapping pair (1764 for 831 registered, locally) and the rest would be dropped below
        workers = n_parallel_pairs or default_pair_workers
        edges = sorted({tuple(sorted(edge)) for edge in reg_results['pairwise_registration']['graph'].edges()})

        def pair_arguments(edge):
            return [msims[node] for node in edge], edge, metric_methods, base_transform_key, query_transform_keys

        metric_results = merge_metric_results(
            map_pair_metrics(msims, edges, pair_arguments, workers, progress_factory, desc='Global metrics',
                             threaded=False))
    else:
        metric_funcs = create_metric_methods(metric_methods, msims[0])
        with _pair_metrics_compute():
            metric_results = multiview_stitcher.metrics.tile_pair_image_metrics(
                msims,
                base_transform_key=base_transform_key,  # defines overlap region
                query_transform_keys=query_transform_keys,
                metric_funcs=metric_funcs,
                n_parallel_pairs=n_parallel_pairs
            )

    if reg_results is not None:
        # the summary as the plain mean over the registered pairs
        for candidate_key, metric_values in metric_results['summary'].items():
            for metric_key in list(metric_values):
                values = [value[candidate_key][metric_key] for value in metric_results['pairs'].values()
                         if candidate_key in value and value[candidate_key].get(metric_key) is not None
                         and not np.isnan(value[candidate_key][metric_key])]
                metric_values[metric_key] = float(np.mean(values)) if values else None

        qualities = reg_results['pairwise_registration']['metrics']['qualities']
        # a pair_key here may use the opposite (fixed, moving) direction from the one Mode 1
        # happened to pick for the same logical pair, or (rarely, if Mode 1's own overlap
        # detection missed it) not appear in metric_results['pairs'] at all - look it up (or
        # create it) by its unordered identity rather than assuming an exact key match
        pairs_by_unordered_key = {frozenset(key): key for key in metric_results['pairs']}

        quality_values = []
        for pair_key, value in qualities.items():
            value = quality_to_scalar(value)
            if value:
                actual_key = pairs_by_unordered_key.get(frozenset(pair_key), pair_key)
                metric_results['pairs'].setdefault(actual_key, {})
                metric_results['pairs'][actual_key].setdefault(reg_transform_key, {})[default_quality_key] = value
                quality_values.append(value)

        metric_results['summary'][reg_transform_key][default_quality_key] = float(np.nanmean(quality_values))

    return metric_results


def calc_msims_metrics(msims, pair_transforms, qualities=None, base_transform_key=None, metric_methods='all',
                       reg_channel=None, n_parallel_pairs=None):
    if base_transform_key is None:
        base_transform_key = next(iter(get_msim_transform_keys(msims[0])))
    with dask.config.set(scheduler=_scheduler('single-threaded')):
        pairs_graph = mv_graph.build_view_adjacency_graph_from_msims(
            msims,
            transform_key=base_transform_key,
            pairs=list(pair_transforms.keys())
        )
    nx.set_edge_attributes(pairs_graph, pair_transforms, default_transform_key)
    if qualities:
        nx.set_edge_attributes(pairs_graph, qualities, default_quality_key)
    return calc_pair_metrics(msims=msims, pairs_graph=pairs_graph, base_transform_key=base_transform_key,
                             metric_methods=metric_methods, reg_channel=reg_channel, n_parallel_pairs=n_parallel_pairs)


def calc_match_metrics(points1, points2, transform, threshold, lowe_ratio=None):
    metrics = {}
    final_matches = []
    inliers = []
    transformed_points1 = apply_transform(points1, transform)
    npoints1, npoints2 = len(points1), len(points2)
    npoints = min(npoints1, npoints2)
    if npoints1 == 0 or npoints2 == 0:
        return metrics

    swapped = (npoints1 > npoints2)
    if swapped:
        points1, points2 = points2, points1

    distance_matrix = euclidean_distances(transformed_points1, points2)
    matching_distances = np.diag(distance_matrix)
    if npoints1 == npoints2 and np.mean(matching_distances < threshold) > 0.5:
        # already matching points lists
        nmatches = np.sum(matching_distances < threshold)
    else:
        matches = []
        distances0 = []
        for rowi, row in enumerate(distance_matrix):
            sorted_indices = np.argsort(row)
            index0 = sorted_indices[0]
            distance0 = row[index0]
            matches.append((rowi, sorted_indices))
            distances0.append(distance0)
        sorted_matches = np.argsort(distances0)

        done = []
        nmatches = 0
        matching_distances = []
        for sorted_match in sorted_matches:
            i, match = matches[sorted_match]
            for j_index, j in enumerate(match):
                if j not in done:
                    # found best, available match
                    distance0 = distance_matrix[i, j]
                    # second best match distance
                    distance1 = distance_matrix[i, match[j_index + 1]] if j_index + 1 < len(match) else np.inf
                    matching_distances.append(distance0)    # use all distances to also weigh in the non-matches
                    final_matches.append((int(i), int(j)))
                    if distance0 < threshold and (lowe_ratio is None or distance0 < lowe_ratio * distance1):
                        done.append(j)
                        nmatches += 1
                        inliers.append(True)
                    else:
                        inliers.append(False)
                    break

    metrics['matches'] = final_matches
    metrics['inliers'] = inliers
    metrics['nmatches'] = nmatches
    metrics['match_rate'] = nmatches / npoints if npoints > 0 else 0
    distance = np.mean(matching_distances) if nmatches > 0 else np.inf
    metrics['distance'] = float(distance)
    metrics['norm_distance'] = float(distance / threshold)
    return metrics


def calc_ncc(image1, image2):
    max_size = np.flip(np.max([image1.shape, image2.shape], 0))
    image1 = image_reshape(image1, max_size)
    image2 = image_reshape(image2, max_size)

    normimage1 = np.array(image1 - np.mean(image1))
    normimage2 = np.array(image2 - np.mean(image2))
    ncc = np.sum(normimage1 * normimage2) / (np.linalg.norm(normimage1) * np.linalg.norm(normimage2))
    return float(ncc)


def calc_ncc2(image1, image2):
    max_size = np.flip(np.max([image1.shape, image2.shape], 0))
    image1 = image_reshape(image1, max_size)
    image2 = image_reshape(image2, max_size)

    normimage1 = (image1 - np.mean(image1)) / np.std(image1)
    normimage2 = (image2 - np.mean(image2)) / np.std(image2)
    array1 = np.array(normimage1).reshape(-1)
    array2 = np.array(normimage2).reshape(-1)
    ncc = (np.correlate(array1, array2) / max(len(array1), len(array2)))[0]
    return float(ncc)


def calc_ssim(image1, image2):
    dtype = image1.dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    max_size = np.flip(np.max([image1.shape, image2.shape], 0))
    image1 = image_reshape(image1, max_size)
    image2 = image_reshape(image2, max_size)
    try:
        ssim = structural_similarity(np.array(image1), np.array(image2), data_range=maxval)
    except ValueError:
        ssim = np.nan
    return float(ssim)


def calc_frc(image1, image2):
    pixel_size1 = si_utils.get_spacing_from_sim(image1)
    pixel_size2 = si_utils.get_spacing_from_sim(image2)
    pixel_size = np.mean([pixel_size1['x'], pixel_size1['y'], pixel_size2['x'], pixel_size2['y']])
    max_size = np.flip(np.max([image1.shape, image2.shape], 0))
    image1 = frc.util.square_image(image_reshape(image1, max_size), add_padding=True)
    image2 = frc.util.square_image(image_reshape(image2, max_size), add_padding=True)

    frc_curve = frc.two_frc(image1, image2)
    xs_pix = np.arange(len(frc_curve)) / max(max_size)
    # scale has units [pixels <length unit>^-1] corresponding to original image
    xs_nm_freq = xs_pix / pixel_size
    frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, max_size)
    #plt.plot(xs_nm_freq, thres(xs_nm_freq))
    #plt.plot(xs_nm_freq, frc_curve)
    #plt.show()
    return frc_res
