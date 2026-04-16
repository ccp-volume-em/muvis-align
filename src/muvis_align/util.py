import ast
from configparser import ConfigParser
import csv
import cv2 as cv
from datetime import datetime
import glob
import json
import math
import numpy as np
import os.path
import re
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
from xarray import DataArray


def get_default(x, default):
    return default if x is None else x


def ensure_list(x) -> list:
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]


def reorder(items: list, old_order: str, new_order: str, default_value: int = 0) -> list:
    new_items = []
    for label in new_order:
        if label in old_order:
            item = items[old_order.index(label)]
        else:
            item = default_value
        new_items.append(item)
    return new_items


def numpy_to_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [numpy_to_native(v) for v in value]
    elif isinstance(value, tuple):
        return (numpy_to_native(v) for v in value)
    elif isinstance(value, dict):
        return {k: numpy_to_native(v) for k, v in value.items()}
    elif hasattr(value, 'dtype'):
        return value.item()
    else:
        return value


def is_valid_value(value):
    return value is not None and value != ''


def set_dict_value(dct, keys, value):
    try:
        value = float(value)
    except:
        pass
    for index, key in enumerate(keys):
        if index == len(keys) - 1:
            dct[key] = value
        else:
            if key not in dct:
                dct[key] = {}
            dct = dct[key]


def filter_dict(dict0: dict) -> dict:
    new_dict = {}
    for key, value0 in dict0.items():
        if value0 is not None:
            values = []
            for value in ensure_list(value0):
                if isinstance(value, dict):
                    value = filter_dict(value)
                values.append(value)
            if len(values) == 1:
                values = values[0]
            new_dict[key] = values
    return new_dict


def desc_to_dict(desc: str) -> dict:
    desc_dict = {}
    if desc.startswith('{'):
        try:
            metadata = ast.literal_eval(desc)
            return metadata
        except:
            pass
    for item in re.split(r'[\r\n\t|]', desc):
        item_sep = '='
        if ':' in item:
            item_sep = ':'
        if item_sep in item:
            items = item.split(item_sep)
            key = items[0].strip()
            value = items[1].strip()
            for dtype in (int, float, bool):
                try:
                    value = dtype(value)
                    break
                except:
                    pass
            desc_dict[key] = value
    return desc_dict


def print_dict(dct: dict, indent: int = 0) -> str:
    s = ''
    if isinstance(dct, dict):
        for key, value in dct.items():
            s += '\n'
            if not isinstance(value, list):
                s += '\t' * indent + str(key) + ': '
            if isinstance(value, dict):
                s += print_dict(value, indent=indent + 1)
            elif isinstance(value, list):
                for v in value:
                    s += print_dict(v)
            else:
                s += str(value)
    else:
        s += str(dct)
    return s


def print_dict_simple(dct: dict) -> str:
    items = []
    for key, value in dct.items():
        if isinstance(value, float):
            value = f'{value:.3f}'
        items.append(f'{key}: {value}')
    return ' '.join(items)


def print_dict_xyz(dct: dict, dims='xyz', decimals=3, as_tuple=False) -> str:
    s = ''
    for dim in dims:
        if dim in dct:
            if s:
                s += ' '
            if as_tuple:
                s += f'{dct[dim]:.{decimals}f}'
            else:
                s += f'{dim}:{dct[dim]:.{decimals}f}'
    return s


def print_hbytes(nbytes: int) -> str:
    exps = ['', 'K', 'M', 'G', 'T', 'P', 'E']
    div = 1024
    exp = 0

    while nbytes > div:
        nbytes /= div
        exp += 1
    if exp < len(exps):
        e = exps[exp]
    else:
        e = f'e{exp * 3}'
    return f'{nbytes:.1f}{e}B'


def check_round_significants(a: float, significant_digits: int) -> float:
    rounded = round_significants(a, significant_digits)
    if a != 0:
        dif = 1 - rounded / a
    else:
        dif = rounded - a
    if abs(dif) < 10 ** -significant_digits:
        return rounded
    return a


def round_significants(a: float, significant_digits: int) -> float:
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a


def split_path(path: str) -> list:
    return os.path.normpath(path).split(os.path.sep)


def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    title = os.path.splitext(filebase)[0].rstrip('.ome')
    return title


def dir_regex(pattern):
    files = []
    for pattern_item in ensure_list(pattern):
        files.extend(glob.glob(pattern_item, recursive=True))
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def find_all_numbers(text: str) -> list:
    return list(map(int, re.findall(r'\d+', text)))


def split_path_parts(text: str) -> list:
    return text.replace('/', '_').replace('\\', '_').replace('.', '_').split('_')


def split_numeric(text: str) -> list:
    num_parts = []
    parts = split_path_parts(text)
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            num_parts.append(part)
    return num_parts


def find_target_numeric(text: str, target: str) -> int|None:
    parts = split_path_parts(text)
    for part in parts:
        if part.startswith(target):
            part = part.lstrip(target)
            if part.isdecimal():
                return int(part)
    return None


def split_numeric_dict(text: str) -> dict:
    num_parts = {}
    parts = split_path_parts(text)
    parti = 0
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            index = num_span.start()
            label = part[:index]
            if label == '':
                label = parti
            num_parts[label] = num_span.group()
            parti += 1
    return num_parts


def get_unique_nums(all_parts: list) -> list:
    keys = []
    for parts in all_parts:
        for key in parts:
            if key not in keys:
                keys.append(key)

    changing_keys = []
    for key in keys:
        values = [parts.get(key) for parts in all_parts]
        if len(set(values)) > 1:
            changing_keys.append(key)

    final_parts = [{key: parts[key] for key in changing_keys if key in parts} for parts in all_parts]
    return final_parts


def get_unique_file_labels(filenames: list) -> list:
    file_labels = []

    ntot = len(filenames)
    parts_dic = get_unique_nums([split_numeric_dict(get_filetitle(filename)) for filename in filenames])
    parts_dic_full = get_unique_nums([split_numeric_dict(filename) for filename in filenames])
    parts_num_full = get_unique_nums([{index: value for index, value in enumerate(split_numeric(filename))} for filename in filenames])

    dic_ok = (len(set(['_'.join(parts.values()) for parts in parts_dic])) == ntot)
    full_dic_ok = (len(set(['_'.join(parts.values()) for parts in parts_dic_full])) == ntot)
    full_num_ok = (len(set(['_'.join(parts.values()) for parts in parts_num_full])) == ntot)

    if dic_ok:
        all_parts = parts_dic
    elif full_dic_ok:
        all_parts = parts_dic_full
    elif full_num_ok:
        all_parts = parts_num_full
    else:
        all_parts = [{0: filename} for filename in filenames]

    for parts in all_parts:
        file_label = '_'.join([key + part if isinstance(key, str) else part for key, part in parts.items()])
        file_labels.append(file_label)

    if len(set(file_labels)) < len(file_labels):
        # fallback for duplicate labels
        file_labels = [get_filetitle(filename) for filename in filenames]

    return file_labels


def split_num_text(text: str) -> list:
    num_texts = []
    block = ''
    is_num0 = None
    if text is None:
        return []

    for c in text:
        is_num = (c.isnumeric() or c == '.')
        if is_num0 is not None and is_num != is_num0:
            num_texts.append(block)
            block = ''
        block += c
        is_num0 = is_num
    if block != '':
        num_texts.append(block)

    num_texts2 = []
    for block in num_texts:
        block = block.strip()
        try:
            block = float(block)
        except:
            pass
        if block not in [' ', ',', '|']:
            num_texts2.append(block)
    return num_texts2


def split_value_unit_list(text: str) -> list:
    value_units = []
    if text is None:
        return None

    items = split_num_text(text)
    if isinstance(items[-1], str):
        def_unit = items[-1]
    else:
        def_unit = ''

    i = 0
    while i < len(items):
        value = items[i]
        if i + 1 < len(items):
            unit = items[i + 1]
        else:
            unit = ''
        if not isinstance(value, str):
            if isinstance(unit, str):
                i += 1
            else:
                unit = def_unit
            value_units.append((value, unit))
        i += 1
    return value_units


def eval_context(data, key, default_value, context):
    value = data.get(key, default_value)
    if isinstance(value, str):
        try:
            value = value.format_map(context)
        except:
            pass
        try:
            value = eval(value, context)
        except:
            pass
    if not isinstance(value, (float, int)):
        value = default_value
    return value


def get_value_units_micrometer(value_units0: list|dict) -> list|dict|None:
    conversions = {
        'nm': 1e-3,
        'µm': 1, 'um': 1, 'micrometer': 1, 'micron': 1,
        'mm': 1e3, 'millimeter': 1e3,
        'cm': 1e4, 'centimeter': 1e4,
        'm': 1e6, 'meter': 1e6
    }
    if value_units0 is None:
        return None

    if isinstance(value_units0, dict):
        values_um = {}
        for dim, value_unit in value_units0.items():
            if isinstance(value_unit, (list, tuple)):
                value_um = value_unit[0] * conversions.get(value_unit[1], 1)
            else:
                value_um = value_unit
            values_um[dim] = value_um
    else:
        values_um = []
        for value_unit in value_units0:
            if isinstance(value_unit, (list, tuple)):
                value_um = value_unit[0] * conversions.get(value_unit[1], 1)
            else:
                value_um = value_unit
            values_um.append(value_um)
    return values_um


def convert_to_um(value, unit):
    conversions = {
        'nm': 1e-3,
        'µm': 1, 'um': 1, 'micrometer': 1, 'micron': 1,
        'mm': 1e3, 'millimeter': 1e3,
        'cm': 1e4, 'centimeter': 1e4,
        'm': 1e6, 'meter': 1e6
    }
    return value * conversions.get(unit, 1)


def convert_rational_value(value) -> float:
    if value is not None and isinstance(value, tuple):
        if value[0] == value[1]:
            value = value[0]
        else:
            value = value[0] / value[1]
    return value


def get_moments(data, offset=(0, 0)):
    moments = cv.moments((np.array(data) + offset).astype(np.float32))    # doesn't work for float64!
    return moments


def get_moments_center(moments, offset=(0, 0)):
    return np.array([moments['m10'], moments['m01']]) / moments['m00'] + np.array(offset)


def get_center(data, offset=(0, 0)):
    moments = get_moments(data, offset=offset)
    if moments['m00'] != 0:
        center = get_moments_center(moments)
    else:
        center = np.mean(data, 0).flatten()  # close approximation
    return center.astype(np.float32)


def create_transform0(center=(0, 0), angle=0, scale=1, translate=(0, 0)):
    transform = cv.getRotationMatrix2D(center[:2], angle, scale)
    transform[:, 2] += translate
    if len(transform) == 2:
        transform = np.vstack([transform, [0, 0, 1]])   # create 3x3 matrix
    return transform


def create_transform(center, angle, matrix_size=3):
    if isinstance(center, dict):
        center = dict_to_xyz(center)
    if len(center) == 2:
        center = np.array(list(center) + [0])
    if angle is None:
        angle = 0
    r = Rotation.from_euler('z', angle, degrees=True)
    t = center - r.apply(center, inverse=True)
    transform = np.eye(matrix_size)
    transform[:3, :3] = np.transpose(r.as_matrix())
    transform[:3, -1] += t
    return transform


def apply_transform(points, transform):
    new_points = []
    for point in points:
        point_len = len(point)
        while len(point) < len(transform):
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(np.array(transform)))
        new_points.append(new_point[:point_len])
    return new_points


def apply_transform_dict(points, transform, transform_dims='xyz'):
    new_points = []
    for point in points:
        point = dict_to_xyz(point, dims=transform_dims)
        while len(point) < len(transform):
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(np.array(transform)))
        new_point = xyz_to_dict(new_point, dims=transform_dims)
        new_point.pop('1', None)
        new_points.append(new_point)
    return new_points


def validate_transform(transform, max_rotation=None):
    if transform is None:
        return False
    transform = np.array(transform)
    if np.any(np.isnan(transform)):
        return False
    if np.any(np.isinf(transform)):
        return False
    if np.linalg.det(transform) == 0:
        return False
    if  max_rotation is not None and abs(normalise_rotation(get_rotation_from_transform(transform))) > max_rotation:
        return False
    return True


def get_scale_from_transform(transform):
    scale = np.mean(np.linalg.norm(transform, axis=0)[:-1])
    return float(scale)


def get_translation_from_transform(transform):
    ndim = len(transform) - 1
    #translation = transform[:ndim, ndim]
    translation = apply_transform([[0] * ndim], transform)[0]
    return translation


def get_center_from_transform(transform):
    # from opencv:
    # t0 = (1-alpha) * cx - beta * cy
    # t1 = beta * cx + (1-alpha) * cy
    # where
    # alpha = cos(angle) * scale
    # beta = sin(angle) * scale
    # isolate cx and cy:
    t0, t1 = transform[:2, 2]
    scale = 1
    angle = np.arctan2(transform[0][1], transform[0][0])
    alpha = np.cos(angle) * scale
    beta = np.sin(angle) * scale
    cx = (t1 + t0 * (1 - alpha) / beta) / (beta + (1 - alpha) ** 2 / beta)
    cy = ((1 - alpha) * cx - t0) / beta
    return cx, cy


def get_rotation_from_transform(transform, dims='xyz'):
    # TODO: assume 2D rotation, expand to 3D
    # Rotation.from_matrix(transform).as_euler() only works for simple rotation matrices
    if isinstance(transform, DataArray):
        dims = transform['x_in'].data.tolist()
    x_index, y_index = dims.index('x'), dims.index('y')
    transform = np.array(transform)
    if y_index > x_index:
        rotation = np.arctan2(transform[0][1], transform[0][0])
    else:
        rotation = np.arctan2(transform[1][0], transform[1][1])
    return float(np.rad2deg(rotation))


def normalise_rotation(rotation):
    """
    Normalise rotation to be in the range [-180, 180].
    """
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360
    return rotation


def points_to_3d(points):
    return [list(point) + [0] for point in points]


def xyz_to_dict(xyz, dims='xyz'):
    dct = {dim: float(value) for dim, value in zip(dims, xyz)}
    return dct


def dict_to_xyz(dct, dims='xyz', add_zeros=False):
    array = [dct[dim] for dim in dims if dim in dct]
    if len(array) < len(dims) and add_zeros:
        array = array + [0] * (len(dims) - len(array))
    return array


def normalise_rotated_positions(centers0, rotations0, sizes, center, ndims):
    # in [xy(z)]
    centers = []
    rotations = []
    _, angles = get_pairs(centers0, sizes)
    for center0, rotation in zip(centers0, rotations0):
        if rotation is None and len(angles) > 0:
            rotation = -float(np.mean(angles))
        angle = -rotation if rotation is not None else None
        transform = create_transform(center=center, angle=angle, matrix_size=ndims + 1)
        center = apply_transform_dict([center0], transform)[0]
        centers.append(center)
        rotations.append(rotation)
    return centers, rotations


def get_nn_distance(points0):
    points = list(set(map(tuple, points0)))     # get unique points
    if len(points) >= 2:
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points, k=2)
        nn_distance = np.median(dist[:, 1])
    else:
        nn_distance = 1
    return nn_distance


def get_mean_nn_distance(points1, points2):
    return np.mean([get_nn_distance(points1), get_nn_distance(points2)])


def filter_edge_points(points, bounds, filter_factor=0.1, threshold=0.5):
    center = np.array(bounds) / 2
    dist_center = np.abs(points / center - 1)
    position_weights = np.clip((1 - np.max(dist_center, axis=-1)) / filter_factor, 0, 1)
    order_weights = 1 - np.array(range(len(points))) / len(points) / 2
    weights = position_weights * order_weights
    return weights > threshold


def draw_edge_filter(bounds):
    out_image = np.zeros(np.flip(bounds))
    y, x = np.where(out_image == 0)
    points = np.transpose([x, y])

    center = np.array(bounds) / 2
    dist_center = np.abs(points / center - 1)
    position_weights = np.clip((1 - np.max(dist_center, axis=-1)) * 10, 0, 1)
    return position_weights.reshape(np.flip(bounds))


def get_pairs(positions, sizes, pairing=None):
    """
    Get pairs of orthogonal neighbors from a list of tiles.
    Tiles don't have to be placed on a regular grid.
    """
    pairs = []
    angles = []
    z_positions = [position['z'] for position in positions if 'z' in position]
    ordered_z = sorted(set(z_positions))
    is_mixed_3dstack = len(ordered_z) < len(z_positions)
    for i, j in np.transpose(np.triu_indices(len(positions), 1)):
        posi, posj = positions[i], positions[j]
        sizei, sizej = sizes[i], sizes[j]
        if is_mixed_3dstack:
            # ignore z value for distance
            distance = math.dist([posi[dim] for dim in 'xy'], [posj[dim] for dim in 'xy'])
            min_distance = max([size[dim] for size in [sizei, sizej] for dim in 'xy'])
            is_same_z = (posi['z'] == posj['z'])
            is_close_z = abs(ordered_z.index(posi['z']) - ordered_z.index(posj['z'])) <= 1
            if not is_close_z:
                # if not close, discard as pair
                min_distance = 0
            elif not is_same_z:
                # for tiles in different z stack, require greater overlap
                min_distance *= 0.8
        else:
            distance = math.dist(posi.values(), posj.values())
            min_distance = max(list(sizei.values()) + list(sizej.values()))

        if pairing and 'overla' in pairing:
            ok = (distance / min_distance < 0.5)
        else:
            ok = (distance < min_distance)
        if ok:
            pairs.append((int(i), int(j)))
            vector = np.array(list(posi.values())) - np.array(list(posj.values()))
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            if distance < min(list(sizei.values()) + list(sizej.values())):
                angle += 90
            while angle < -90:
                angle += 180
            while angle > 90:
                angle -= 180
            angles.append(angle)
    return pairs, angles


def retuple(chunks, shape):
    # from ome-zarr-py
    """
    Expand chunks to match shape.

    E.g. if chunks is (64, 64) and shape is (3, 4, 5, 1028, 1028)
    return (3, 4, 5, 64, 64)

    If chunks is an integer, it is applied to all dimensions, to match
    the behaviour of zarr-python.
    """

    if isinstance(chunks, int):
        return tuple([chunks] * len(shape))

    dims_to_add = len(shape) - len(chunks)
    return *shape[:dims_to_add], *chunks


def import_metadata(content, fields=None, input_path=None):
    # return dict[id] = {values}
    if isinstance(content, str):
        ext = os.path.splitext(content)[1].lower()
        if input_path:
            if isinstance(input_path, list):
                input_path = input_path[0]
            content = os.path.normpath(os.path.join(os.path.dirname(input_path), content))
        if ext == '.csv':
            content = import_csv(content)
        elif ext in ['.json', '.ome.json']:
            content = import_json(content)
    if fields is not None:
        content = [[data[field] for field in fields] for data in content]
    return content


def import_json(filename):
    with open(filename, encoding='utf8') as file:
        data = json.load(file)
    return data


def export_json(filename, data):
    with open(filename, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4)


def import_csv(filename):
    with open(filename, encoding='utf8') as file:
        data = csv.reader(file)
    return data


def export_csv(filename, data, header=None):
    with open(filename, 'w', encoding='utf8', newline='') as file:
        csvwriter = csv.writer(file)
        if header is not None:
            csvwriter.writerow(header)
        for row in data:
            csvwriter.writerow(row)


def load_sbemimage_best_config(metapath, filename):
    target_datetime = datetime.fromtimestamp(os.path.getmtime(filename))

    for config_filename in sorted(glob.glob(os.path.join(metapath, 'logs/config_*.txt')), reverse=True):
        match = re.split(r'config_(\d+-\d+-\d+).txt', config_filename)
        if len(match) >= 2:
            file_date = datetime.strptime(match[1], '%Y-%m-%d%H%M%S%f')
            if file_date <= target_datetime:
                with open(config_filename, 'r') as file:
                    sbemimage_config = file.read()
                return sbemimage_config
    return None


def adjust_sbemimage_properties(translation, scale, size, filename, sbemimage_config):
    cfg = ConfigParser()
    cfg.read_string(sbemimage_config)
    if bool(cfg['sys'].get('use_microtome')):
        props = cfg['microtome']
    else:
        props = cfg['sem']
    scale_x = float(props.get('stage_scale_factor_x'))
    scale_y = float(props.get('stage_scale_factor_y'))
    rotation_x = float(props.get('stage_rotation_angle_x'))
    rotation_y = float(props.get('stage_rotation_angle_y'))
    rotation_diff = rotation_x - rotation_y
    rot_mat_a = math.cos(rotation_y) / math.cos(rotation_diff)
    rot_mat_b = -math.sin(rotation_y) / math.cos(rotation_diff)
    rot_mat_c = math.sin(rotation_x) / math.cos(rotation_diff)
    rot_mat_d = math.cos(rotation_x) / math.cos(rotation_diff)
    rot_mat_determinant = rot_mat_a * rot_mat_d - rot_mat_b * rot_mat_c

    pixel_size = None
    parts = split_numeric_dict(filename)
    if 't' in parts:
        grids = cfg['grids']
        if 'r' in parts:
            grid_index = json.loads(grids.get('roi_index')).index(int(parts['r']))
        else:
            grid_index = int(parts['g'])
        pixel_size = json.loads(grids.get('pixel_size'))[grid_index] * 1e-3
    else:
        ov = cfg['overviews']
        if 'ov' in parts:
            ov_index = int(parts['ov'])
            pixel_size = json.loads(ov.get('ov_pixel_size'))[ov_index] * 1e-3
        elif '_stubov_' in filename.lower():
            pixel_size = float(ov.get('stub_ov_pixel_size')) * 1e-3

    if pixel_size:
        scale = {'x': pixel_size, 'y': pixel_size}
    else:
        scale = None

    stage_x, stage_y = translation['x'], translation['y']
    stage_x /= scale_x
    stage_y /= scale_y
    dx = ((rot_mat_d * stage_x - rot_mat_b * stage_y) / rot_mat_determinant)
    dy = ((-rot_mat_c * stage_x + rot_mat_a * stage_y) / rot_mat_determinant)
    # convert center to top/left
    physical_size = {dim: size[dim] * scale[dim] for dim in size}
    translation['x'] = dx - physical_size['x'] / 2
    translation['y'] = dy - physical_size['y'] / 2

    return translation, scale


def get_label_element(elements, label):
    for element in elements:
        if element.get('label') == label:
            return element
    return None
