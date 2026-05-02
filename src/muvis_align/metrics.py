#import frc
import multiview_stitcher.metrics
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from skimage.metrics import structural_similarity, normalized_mutual_information, mean_squared_error
from sklearn.metrics import euclidean_distances

from src.muvis_align.image.util import image_reshape
from src.muvis_align.util import apply_transform



def create_metric_methods(metric_methods, msim, reg_channel=None):
    data_range = np.iinfo(msim["scale0/image"].dtype).max
    all_metric_funcs = {
        'ncc': multiview_stitcher.metrics.normalized_cross_correlation,
        'ssim': lambda im1, im2: structural_similarity(np.nan_to_num(im1), np.nan_to_num(im2),
                                                       data_range=data_range, channel_axis=reg_channel),
        'onmi': lambda im1, im2: normalized_mutual_information(np.nan_to_num(im1), np.nan_to_num(im2)) - 1,
        "mse": lambda im1, im2: 1 / mean_squared_error(im1, im2),
    }
    metric_funcs = {metric_method: all_metric_funcs[metric_method] for metric_method in metric_methods}
    return metric_funcs


def calc_pair_metrics(msims, pairs_graph, metric_methods, base_transform_key, reg_channel=None):
    metric_funcs = create_metric_methods(metric_methods, msims[0], reg_channel=reg_channel)
    metric_results = multiview_stitcher.metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,  # defines overlap region
        pairs_graph=pairs_graph,
        metric_funcs=metric_funcs,
    )
    return metric_results


def calc_global_metrics(msims, base_transform_key, reg_transform_key, metric_methods, reg_channel=None):
    metric_funcs = create_metric_methods(metric_methods, msims[0], reg_channel=reg_channel)
    metric_results = multiview_stitcher.metrics.tile_pair_image_metrics(
        msims,
        base_transform_key=base_transform_key,  # defines overlap region
        query_transform_keys=[
            base_transform_key,
            reg_transform_key
        ],
        metric_funcs=metric_funcs
    )
    return metric_results


def calc_match_metrics(points1, points2, transform, threshold, lowe_ratio=None):
    metrics = {}
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
            for ji, j in enumerate(match):
                if j not in done:
                    # found best, available match
                    distance0 = distance_matrix[i, j]
                    distance1 = distance_matrix[i, match[ji + 1]] if ji + 1 < len(match) else np.inf
                    matching_distances.append(distance0)    # use all distances to also weigh in the non-matches
                    if distance0 < threshold and (lowe_ratio is None or distance0 < lowe_ratio * distance1):
                        done.append(j)
                        nmatches += 1
                    break

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
