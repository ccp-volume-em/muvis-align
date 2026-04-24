import cv2 as cv
import numpy as np
from multiview_stitcher import msi_utils, param_utils, fusion, mv_graph
from multiview_stitcher import spatial_image_utils as si_utils
from skimage.filters import gaussian
from skimage.feature import plot_matched_features
from skimage.transform import downscale_local_mean
from xarray import DataTree

try:
    import matplotlib as mpl
    #mpl.use('TkAgg')
    #mpl.rcParams['backend'] = 'svg'
    mpl.rcParams['figure.dpi'] = 300
    import matplotlib.pyplot as plt
except Exception as e:
    print(f'matplotlib import error:\n{e}')

from src.muvis_align.util import *


def show_image(image, title='', cmap=None):
    if cmap is None:
        nchannels = image.shape[2] if len(image.shape) > 2 else 1
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.show()


def plt_close():
    plt.close()


def grayscale_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 4:
        return cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    elif nchannels > 1:
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        return image


def color_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 1:
        return cv.cvtColor(np.array(image), cv.COLOR_GRAY2RGB)
    else:
        return image


def int2float_image(image):
    source_dtype = image.dtype
    if not source_dtype.kind == 'f':
        maxval = 2 ** (8 * source_dtype.itemsize) - 1
        return image / np.float32(maxval)
    else:
        return image


def float2int_image(image, target_dtype=np.dtype(np.uint8)):
    source_dtype = image.dtype
    if source_dtype.kind not in ('i', 'u') and not target_dtype.kind == 'f':
        maxval = 2 ** (8 * target_dtype.itemsize) - 1
        return (image * maxval).astype(target_dtype)
    else:
        return image


def uint8_image(image):
    source_dtype = image.dtype
    if source_dtype.kind == 'f':
        image = image * 255
    elif source_dtype.itemsize != 1:
        factor = 2 ** (8 * (source_dtype.itemsize - 1))
        image = image // factor
    return image.astype(np.uint8)


def ensure_unsigned_type(dtype: np.dtype) -> np.dtype:
    new_dtype = dtype
    if dtype.kind == 'i' or dtype.byteorder == '>' or dtype.byteorder == '<':
        new_dtype = np.dtype(f'u{dtype.itemsize}')
    return new_dtype


def ensure_unsigned_image(image: np.ndarray) -> np.ndarray:
    source_dtype = image.dtype
    dtype = ensure_unsigned_type(source_dtype)
    if dtype != source_dtype:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        new_image = image.astype(dtype) + offset
    else:
        new_image = image
    return new_image


def convert_image_sign_type(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    source_dtype = image.dtype
    if source_dtype.kind == target_dtype.kind:
        new_image = image
    elif source_dtype.kind == 'i':
        new_image = ensure_unsigned_image(image)
    else:
        # conversion without overhead
        offset = 2 ** (8 * target_dtype.itemsize - 1)
        new_image = (image - offset).astype(target_dtype)
    return new_image


def redimension_data(data, old_order, new_order, **indices):
    # able to provide optional dimension values e.g. t=0, z=0
    if new_order == old_order:
        return data

    new_data = data
    order = old_order
    # remove
    for o in old_order:
        if o not in new_order:
            index = order.index(o)
            dim_value = indices.get(o, 0)
            new_data = np.take(new_data, indices=dim_value, axis=index)
            order = order[:index] + order[index + 1:]
    # add
    for o in new_order:
        if o not in order:
            new_data = np.expand_dims(new_data, 0)
            order = o + order
    # move
    old_indices = [order.index(o) for o in new_order]
    new_indices = list(range(len(new_order)))
    new_data = np.moveaxis(new_data, old_indices, new_indices)
    return new_data


def get_numpy_slicing(dimension_order, **slicing):
    slices = []
    for axis in dimension_order:
        index = slicing.get(axis)
        index0 = slicing.get(axis + '0')
        index1 = slicing.get(axis + '1')
        if index0 is not None and index1 is not None:
            slice1 = slice(int(index0), int(index1))
        elif index is not None:
            slice1 = int(index)
        else:
            slice1 = slice(None)
        slices.append(slice1)
    return tuple(slices)


def get_image_size_info(sizes_xyzct: list, pixel_nbytes: int, pixel_type: np.dtype, channels: list) -> str:
    image_size_info = 'XYZCT:'
    size = 0
    for i, size_xyzct in enumerate(sizes_xyzct):
        w, h, zs, cs, ts = size_xyzct
        size += np.int64(pixel_nbytes) * w * h * zs * cs * ts
        if i > 0:
            image_size_info += ','
        image_size_info += f' {w} {h} {zs} {cs} {ts}'
    image_size_info += f' Pixel type: {pixel_type} Uncompressed: {print_hbytes(size)}'
    if sizes_xyzct[0][3] == 3:
        channel_info = 'rgb'
    else:
        channel_info = ','.join([channel.get('Name', '') for channel in channels])
    if channel_info != '':
        image_size_info += f' Channels: {channel_info}'
    return image_size_info


def pilmode_to_pixelinfo(mode: str) -> tuple:
    pixelinfo = (np.uint8, 8, 1)
    mode_types = {
        'I': (np.uint32, 32, 1),
        'F': (np.float32, 32, 1),
        'RGB': (np.uint8, 24, 3),
        'RGBA': (np.uint8, 32, 4),
        'CMYK': (np.uint8, 32, 4),
        'YCbCr': (np.uint8, 24, 3),
        'LAB': (np.uint8, 24, 3),
        'HSV': (np.uint8, 24, 3),
    }
    if '16' in mode:
        pixelinfo = (np.uint16, 16, 1)
    elif '32' in mode:
        pixelinfo = (np.uint32, 32, 1)
    elif mode in mode_types:
        pixelinfo = mode_types[mode]
    pixelinfo = (np.dtype(pixelinfo[0]), pixelinfo[1])
    return pixelinfo


def calc_pyramid(xyzct: tuple, npyramid_add: int = 0, pyramid_downsample: float = 2,
                 volumetric_resize: bool = False) -> list:
    x, y, z, c, t = xyzct
    if volumetric_resize and z > 1:
        size = (x, y, z)
    else:
        size = (x, y)
    sizes_add = []
    scale = 1
    for _ in range(npyramid_add):
        scale /= pyramid_downsample
        scaled_size = np.maximum(np.round(np.multiply(size, scale)).astype(int), 1)
        sizes_add.append(scaled_size)
    return sizes_add


def get_level_from_scale(source_scales, source_pixel_size, target_scale=1):
    # Only downscaling
    mean_source_pixel_size = float(np.mean([source_pixel_size[dim] for dim in 'xy']))
    if isinstance(target_scale, dict):
        target_pixel_size = target_scale
        target_scale = float(np.mean([target_scale[dim] for dim in 'xy'])) / mean_source_pixel_size
    elif isinstance(target_scale, str):
        index = target_scale.find(next(filter(str.isalpha, target_scale)))
        pixel_size = convert_to_um(float(target_scale[:index]), target_scale[index:])
        target_pixel_size = {dim: pixel_size for dim in 'xy'}
        target_scale = pixel_size / mean_source_pixel_size
    else:
        target_pixel_size = source_pixel_size
    best_level, best_scale = 0, target_scale
    for level, scale in enumerate(source_scales):
        if np.isclose(scale, target_scale, rtol=1e-4):
            target_pixel_size = {dim: float(source_pixel_size[dim] * scale) for dim in source_pixel_size}
            return level, 1, target_pixel_size
        if scale <= target_scale:
            best_level, best_scale = level, target_scale / scale
    if best_level == 0 and best_scale < 1:
        best_scale = 1
    return best_level, best_scale, target_pixel_size


def image_reshape(image: np.ndarray, target_size: tuple) -> np.ndarray:
    tw, th = target_size
    sh, sw = image.shape[0:2]
    if sw < tw or sh < th:
        dw = max(tw - sw, 0)
        dh = max(th - sh, 0)
        padding = [(dh // 2, dh - dh //  2), (dw // 2, dw - dw // 2)]
        if len(image.shape) == 3:
            padding += [(0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=(0, 0))
    if tw < sw or th < sh:
        image = image[0:th, 0:tw]
    return image


def resize_image(image, new_size):
    if not isinstance(new_size, (tuple, list, np.ndarray)):
        # use single value for width; apply aspect ratio
        size = np.flip(image.shape[:2])
        new_size = new_size, new_size * size[1] // size[0]
    return cv.resize(image, new_size)


def image_resize(image: np.ndarray, target_size0: tuple, dimension_order: str = 'yxc') -> np.ndarray:
    shape = image.shape
    x_index = dimension_order.index('x')
    y_index = dimension_order.index('y')
    c_is_at_end = ('c' in dimension_order and dimension_order.endswith('c'))
    size = shape[x_index], shape[y_index]
    if np.mean(np.divide(size, target_size0)) < 1:
        interpolation = cv.INTER_CUBIC
    else:
        interpolation = cv.INTER_AREA
    dtype0 = image.dtype
    image = ensure_unsigned_image(image)
    target_size = tuple(np.maximum(np.round(target_size0).astype(int), 1))
    if dimension_order in ['yxc', 'yx']:
        new_image = cv.resize(np.asarray(image), target_size, interpolation=interpolation)
    elif dimension_order == 'cyx':
        new_image = np.moveaxis(image, 0, -1)
        new_image = cv.resize(np.asarray(new_image), target_size, interpolation=interpolation)
        new_image = np.moveaxis(new_image, -1, 0)
    else:
        ts = image.shape[dimension_order.index('t')] if 't' in dimension_order else 1
        zs = image.shape[dimension_order.index('z')] if 'z' in dimension_order else 1
        target_shape = list(image.shape).copy()
        target_shape[x_index] = target_size[0]
        target_shape[y_index] = target_size[1]
        new_image = np.zeros(target_shape, dtype=image.dtype)
        for t in range(ts):
            for z in range(zs):
                slices = get_numpy_slicing(dimension_order, z=z, t=t)
                image1 = image[slices]
                if not c_is_at_end:
                    image1 = np.moveaxis(image1, 0, -1)
                new_image1 = np.atleast_3d(cv.resize(np.asarray(image1), target_size, interpolation=interpolation))
                if not c_is_at_end:
                    new_image1 = np.moveaxis(new_image1, -1, 0)
                new_image[slices] = new_image1
    new_image = convert_image_sign_type(new_image, dtype0)
    return new_image


def precise_resize(image: np.ndarray, factors) -> np.ndarray:
    if image.ndim > len(factors):
        factors = list(factors) + [1]
    new_image = downscale_local_mean(np.asarray(image), tuple(factors)).astype(image.dtype)
    return new_image


def draw_keypoints(image, points, color=(255, 0, 0)):
    out_image = color_image(float2int_image(image))
    for point in points:
        point = np.round(point).astype(int)
        cv.drawMarker(out_image, tuple(point), color=color, markerType=cv.MARKER_CROSS, markerSize=5, thickness=1)
    return out_image


def draw_keypoints_matches_cv(image1, points1, image2, points2, matches=None, inliers=None,
                              color=(255, 0, 0), inlier_color=(0, 255, 0), radius = 15, thickness = 2):
    # based on https://gist.github.com/woolpeeker/d7e1821e1b5c556b32aafe10b7a1b7e8
    image1 = uint8_image(image1)
    image2 = uint8_image(image2)
    # We're drawing them side by side.  Get dimensions accordingly.
    new_shape = (max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3)
    out_image = np.zeros(new_shape, image1.dtype)
    # Place images onto the new image.
    out_image[0:image1.shape[0], 0:image1.shape[1]] = color_image(image1)
    out_image[0:image2.shape[0], image1.shape[1]:image1.shape[1] + image2.shape[1]] = color_image(image2)

    if matches is not None:
        # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
        for index, match in enumerate(matches):
            if inliers is not None and inliers[index]:
                line_color = inlier_color
            else:
                line_color = color
            # So the keypoint locs are stored as a tuple of floats.  cv2.line() wants locs as a tuple of ints.
            end1 = tuple(np.round(points1[match[0]]).astype(int))
            end2 = tuple(np.round(points2[match[1]]).astype(int) + np.array([image1.shape[1], 0]))
            cv.line(out_image, end1, end2, line_color, thickness)
            cv.circle(out_image, end1, radius, line_color, thickness)
            cv.circle(out_image, end2, radius, line_color, thickness)
    else:
        # Draw all points if no matches are provided.
        for point in points1:
            point = tuple(np.round(point).astype(int))
            cv.circle(out_image, point, radius, color, thickness)
        for point in points2:
            point = tuple(np.round(point).astype(int) + np.array([image1.shape[1], 0]))
            cv.circle(out_image, point, radius, color, thickness)
    return out_image


def draw_keypoints_matches_sk(image1, points1, image2, points2, matches=np.array([]),
                              show_plot=True, output_filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    shape_y, shape_x = image1.shape[:2]
    if shape_x > 2 * shape_y:
        alignment = 'vertical'
    else:
        alignment = 'horizontal'
    plot_matched_features(
        image1,
        image2,
        keypoints0=points1,
        keypoints1=points2,
        matches=matches,
        ax=ax,
        alignment=alignment,
        only_matches=True,
    )
    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()


def draw_keypoints_matches(image1, points1, image2, points2, matches=[], inliers=[],
                           points_color='black', match_color='red', inlier_color='lime',
                           show_plot=True, output_filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    shape = np.max([image.shape for image in [image1, image2]], axis=0)
    shape_y, shape_x = shape[:2]
    if shape_x > 2 * shape_y:
        merge_axis = 0
        offset2 = [shape_y, 0]
    else:
        merge_axis = 1
        offset2 = [0, shape_x]
    image = np.concatenate([
        np.pad(image1, ((0, shape[0] - image1.shape[0]), (0, shape[1] - image1.shape[1]))),
        np.pad(image2, ((0, shape[0] - image2.shape[0]), (0, shape[1] - image2.shape[1])))
    ], axis=merge_axis)
    ax.imshow(image, cmap='gray')

    if len(points1) > 0:
        ax.scatter(
            points1[:, 1],
            points1[:, 0],
            facecolors='none',
            edgecolors=points_color,
        )
    if len(points2) > 0:
        ax.scatter(
            points2[:, 1] + offset2[1],
            points2[:, 0] + offset2[0],
            facecolors='none',
            edgecolors=points_color,
        )

    for i, match in enumerate(matches):
        color = match_color
        if i < len(inliers) and inliers[i]:
            color = inlier_color
        index1, index2 = match
        ax.plot(
            (points1[index1, 1], points2[index2, 1] + offset2[1]),
            (points1[index1, 0], points2[index2, 0] + offset2[0]),
            '-', linewidth=1, alpha=0.5, color=color,
        )

    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()

    return fig, ax


def create_compression_filter(compression: list) -> tuple:
    compressor, compression_filters = None, None
    compression = ensure_list(compression)
    if compression is not None and len(compression) > 0:
        compression_type = compression[0].lower()
        if len(compression) > 1:
            level = int(compression[1])
        else:
            level = None
        if 'lzw' in compression_type:
            from imagecodecs.numcodecs import Lzw
            compression_filters = [Lzw()]
        elif '2k' in compression_type or '2000' in compression_type:
            from imagecodecs.numcodecs import Jpeg2k
            compression_filters = [Jpeg2k(level=level)]
        elif 'jpegls' in compression_type:
            from imagecodecs.numcodecs import Jpegls
            compression_filters = [Jpegls(level=level)]
        elif 'jpegxr' in compression_type:
            from imagecodecs.numcodecs import Jpegxr
            compression_filters = [Jpegxr(level=level)]
        elif 'jpegxl' in compression_type:
            from imagecodecs.numcodecs import Jpegxl
            compression_filters = [Jpegxl(level=level)]
        else:
            compressor = compression
    return compressor, compression_filters


def gaussian_filter_image(image, sigma):
    nchannels = image.shape[2] if image.ndim == 3 else 1
    if nchannels not in [1, 3]:
        new_image = np.zeros_like(image)
        for channeli in range(nchannels):
            new_image[..., channeli] = gaussian(image[..., channeli], sigma)
    else:
        new_image = gaussian(image, sigma, preserve_range=True)
    return new_image


def calc_images_median(images):
    out_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    median_image = np.median(images, 0, out_image)
    return median_image


def calc_images_quantiles(images, quantiles):
    quantile_images = [image.astype(np.float32) for image in np.quantile(images, quantiles, 0)]
    return quantile_images


def get_image_quantile(image: np.ndarray, quantile: float, axis=None) -> float:
    value = np.quantile(image, quantile, axis=axis).astype(image.dtype)
    return value.item()


def get_image_window(image, low=0.01, high=0.99):
    window = (
        get_image_quantile(image, low),
        get_image_quantile(image, high)
    )
    return window


def normalise_values(image: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    image = (image.astype(np.float32) - min_value) / (max_value - min_value)
    return image.clip(0, 1)


def norm_image_variance(image0):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = (image - np.mean(image)) / np.std(image)
    normimage = normimage.clip(0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def norm_image_variance2(image0):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = ((image - np.mean(image)) / np.std(image) + 1) / 2
    normimage = normimage.clip(0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def norm_image_quantiles(image0, quantile=0.99):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    min_value = np.quantile(image, 1 - quantile)
    max_value = np.quantile(image, quantile)
    normimage = (image - np.mean(image)) / (max_value - min_value)
    normimage = normimage.clip(0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def get_max_downsamples(shape, npyramid_add, pyramid_downsample):
    shape = list(shape)
    for i in range(npyramid_add):
        shape[-1] //= pyramid_downsample
        shape[-2] //= pyramid_downsample
        if shape[-1] < 1 or shape[-2] < 1:
            return i
    return npyramid_add


def filter_noise_images(images):
    dtype = images[0].dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    image_vars = [np.asarray(np.std(image)).item() for image in images]
    threshold, mask0 = cv.threshold(np.array(image_vars).astype(dtype), 0, maxval, cv.THRESH_OTSU)
    mask = [flag.item() for flag in mask0.astype(bool)]
    return int(threshold), mask


def invert_data(data):
    if isinstance(data, list):
        return [invert_data(d) for d in data]
    else:
        dtype = data.dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return info.max - data
        elif np.issubdtype(dtype, np.floating):
            return 1.0 - data
        else:
            return data


def detect_area_points(image):
    method = cv.THRESH_OTSU
    threshold = -5
    contours = []
    while len(contours) <= 1 and threshold <= 255:
        _, binimage = cv.threshold(np.array(uint8_image(image)), threshold, 255, method)
        contours0 = cv.findContours(binimage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours0[0] if len(contours0) == 2 else contours0[1]
        method = cv.THRESH_BINARY
        threshold += 5
    area_contours = [(contour, cv.contourArea(contour)) for contour in contours]
    area_contours.sort(key=lambda contour_area: contour_area[1], reverse=True)
    min_area = max(np.mean([area for contour, area in area_contours]), 1)
    area_points = [(get_center(contour), area) for contour, area in area_contours if area > min_area]

    #image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    #for point in area_points:
    #    radius = int(np.round(np.sqrt(point[1]/np.pi)))
    #    cv.circle(image, tuple(np.round(point[0]).astype(int)), radius, (255, 0, 0), -1)
    #show_image(image)
    return area_points


def get_sim_position_final(sim, position=None, transform_keys=None, get_center=False):
    if position is None:
        position = si_utils.get_origin_from_sim(sim)
    if transform_keys is None:
        transform_keys = si_utils.get_tranform_keys_from_sim(sim)
    transform = combine_transforms([np.array(si_utils.get_affine_from_sim(sim, transform_key))
                                    for transform_key in transform_keys])
    transform_dims = si_utils.get_affine_from_sim(sim, transform_keys[0])['x_in'].data.tolist()
    new_position = apply_transform_dict([position], transform, transform_dims)[0]
    for dim in position.keys():
        if dim not in new_position:
            new_position[dim] = position[dim]
    if get_center:
        physical_size = get_sim_physical_size(sim)
        new_position = {dim: new_position[dim] + physical_size.get(dim, 0) / 2 for dim in new_position}
    return new_position


def group_sims_by_z(sims, positions=None):
    grouped_sims = []
    if positions is None:
        positions = [si_utils.get_origin_from_sim(sim) for sim in sims]
    z_positions = [position.get('z', 0) for position in positions]
    unique_z_values = len(set(z_positions)) == len(z_positions)
    if not unique_z_values:
        sims_by_z = {}
        for simi, z_pos in enumerate(z_positions):
            if z_pos is not None and z_pos not in sims_by_z:
                sims_by_z[z_pos] = []
            sims_by_z[z_pos].append(simi)
        grouped_sims = list(sims_by_z.values())
    if len(grouped_sims) == 0:
        grouped_sims = [list(range(len(sims)))]
    return grouped_sims


def calc_foreground_map(sims):
    if len(sims) <= 2:
        return [True] * len(sims)
    sims = [sim.squeeze().astype(np.float32) for sim in sims]
    median_image = calc_images_median(sims).astype(np.float32)
    difs = [np.mean(np.abs(sim - median_image), (0, 1)) for sim in sims]
    # or use stddev instead of mean?
    threshold = np.mean(difs, 0)
    #threshold, _ = cv.threshold(np.array(difs).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
    #threshold, foregrounds = filter_noise_images(channel_images)
    map = (difs > threshold)
    if np.all(map == False):
        return [True] * len(sims)
    return map


def normalise_sims(sims, transform_key, use_global=True):
    new_sims = []
    dtype = sims[0].dtype
    # global mean and stddev
    if use_global:
        mins = []
        ranges = []
        for sim in sims:
            min = np.mean(sim, dtype=np.float32)
            range = np.std(sim, dtype=np.float32)
            #min, max = get_image_window(sim, low=0.01, high=0.99)
            #range = max - min
            mins.append(min)
            ranges.append(range)
        min = np.mean(mins)
        range = np.mean(ranges)
    else:
        min = 0
        range = 1
    # normalise all images
    for sim in sims:
        if not use_global:
            min = np.mean(sim, dtype=np.float32)
            range = np.std(sim, dtype=np.float32)
        #image = (sim - min) / range
        image = ((sim - min) / range + 1) / 2   # extended range
        image = float2int_image(image.clip(0, 1), dtype)    # np.clip(image) is not dask-compatible, use image.clip() instead
        new_sim = si_utils.get_sim_from_array(
            image,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
            transform_key=transform_key,
            affine=si_utils.get_affine_from_sim(sim, transform_key),
            c_coords=sim.c.data,
            t_coords=sim.t.data
        )
        new_sims.append(new_sim)
    return new_sims


def gaussian_filter_sim(sim, transform_key, sigma):
    image = np.asarray(sim)
    blurred_image = gaussian_filter_image(image, sigma)
    new_sim = si_utils.get_sim_from_array(
        blurred_image.astype(sim.dtype),
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=si_utils.get_origin_from_sim(sim),
        transform_key=transform_key,
        affine=si_utils.get_affine_from_sim(sim, transform_key),
        c_coords=sim.c.data,
        t_coords=sim.t.data
    )
    return new_sim


def get_sim_physical_size(sim):
    size = si_utils.get_shape_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    physical_size = {dim: size[dim] * spacing.get(dim, 1) for dim in size}
    return physical_size


def calc_output_properties(sims, transform_key, output_spacing=None, z_scale=None):
    spacings = [si_utils.get_spacing_from_sim(sim) for sim in sims]
    dims = list(spacings[0])
    if not output_spacing or 'mean' in output_spacing.lower():
        output_spacing = {dim: np.mean([spacing[dim] for spacing in spacings]) for dim in dims}
    elif 'max' in output_spacing.lower():
        output_spacing = {dim: max([spacing[dim] for spacing in spacings]) for dim in dims}
    elif 'min' in output_spacing.lower():
        output_spacing = {dim: min([spacing[dim] for spacing in spacings]) for dim in dims}
    if z_scale is not None:
        output_spacing['z'] = z_scale
    output_properties = fusion.calc_fusion_stack_properties(
        sims,
        [si_utils.get_affine_from_sim(sim, transform_key) for sim in sims],
        output_spacing,
        mode='union',
    )
    if 'z' in output_properties['shape']:
        z_positions = sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims]))
        z_shape = len(z_positions)
        if z_shape <= 1:
            z_shape = len(sims)
        output_properties['shape']['z'] = z_shape
    return output_properties


def get_sim_shape_2d(sim, transform_key=None):
    if 't' in sim.coords.xindexes:
        # work-around for points error in get_overlap_bboxes()
        sim1 = si_utils.sim_sel_coords(sim, {'t': 0})
    else:
        sim1 = sim
    stack_props = si_utils.get_stack_properties_from_sim(sim1, transform_key=transform_key)
    vertices = mv_graph.get_vertices_from_stack_props(stack_props)
    if vertices.shape[1] == 3:
        # remove z coordinate
        vertices = vertices[:, 1:]
    if len(vertices) >= 8:
        # remove redundant x/y vertices
        vertices = vertices[:4]
    if len(vertices) >= 4:
        # last 2 vertices appear to be swapped
        vertices[2:] = np.array(list(reversed(vertices[2:])))
    return vertices


def get_properties_from_transform(transform):
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    xtranslation = param_utils.translation_from_affine(transform)
    dims = xtranslation['x_in'].data.tolist()
    translation = {dim: xtranslation.sel(x_in=dim).item() for dim in dims}
    rotation = get_rotation_from_transform(transform)
    scale = get_scale_from_transform(transform)
    return translation, rotation, scale


def get_data_mapping(data, transform_key=None, transform=None, translation0=None, rotation=None):
    if rotation is None:
        rotation = 0

    if isinstance(data, DataTree):
        sim = msi_utils.get_sim_from_msim(data)
    else:
        sim = data
    translation = si_utils.get_origin_from_sim(sim)
    if 'z' not in translation and translation0 is not None and 'z' in translation0:
        translation['z'] = translation0['z']

    if transform is not None:
        translation1, rotation1, _ = get_properties_from_transform(transform)
        dims = set(list(translation) + list(translation1))
        translation = {dim: translation.get(dim, 0) + translation1.get(dim, 0) for dim in dims}
        rotation += rotation1

    if transform_key is not None:
        transform1 = sim.transforms.get(transform_key)
        if transform1 is not None:
            _, rotation1, _ = get_properties_from_transform(transform1)
            rotation += rotation1

    return translation, rotation


def combine_transforms(transforms):
    combined_transform = None
    for transform in transforms:
        if combined_transform is None:
            combined_transform = transform
        else:
            combined_transform = np.dot(transform, combined_transform)
    return combined_transform


def make_sims_3d(sims, z_scale=None, positions=None):
    new_sims = []
    if not z_scale:
        z_scale = 1
    for index, sim in enumerate(sims):
        # check if already 3D
        if 'z' not in sim.dims:
            z_position = positions[index].get('z', index * z_scale)
            sim = sim.expand_dims({'z': [z_position]}, axis=-3)
        # set 3D affine transforms from 2D registration params
        for transform_key in si_utils.get_tranform_keys_from_sim(sim):
            transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
            if 4 not in transform.shape:
                transform_3d = param_utils.identity_transform(ndim=3)
                if 't' in transform.dims:
                    transform_3d.loc[{dim: transform.coords[dim] for dim in transform.sel(t=0).dims}] = transform.sel(t=0)
                else:
                    transform_3d.loc[{dim: transform.coords[dim] for dim in transform.dims}] = transform
                si_utils.set_sim_affine(sim, transform_3d, transform_key=transform_key)
        new_sims.append(sim)
    return new_sims
