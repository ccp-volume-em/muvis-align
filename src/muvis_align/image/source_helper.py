import numpy as np
import os

from muvis_align.image.TiffDaskSource import TiffDaskSource
from muvis_align.image.ZarrDaskSource import ZarrDaskSource
from muvis_align.util import get_pairs, get_unique_file_labels, print_dict_xyz


def create_dask_source(filename, source_metadata=None):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        dask_source = TiffDaskSource(filename, source_metadata)
    elif '.zar' in filename.lower():
        dask_source = ZarrDaskSource(filename, source_metadata)
    else:
        if ext:
            raise ValueError(f'Unsupported file type: {ext}')
        else:
            raise ValueError(f'Unsupported: {filename}')
    return dask_source


def get_images_metadata(filenames, source_metadata=None):
    summary = 'Filename\tPixel size\tSize\tPosition\tRotation\n'
    sizes = []
    centers = []
    rotations = []
    positions = []
    max_positions = []
    pixel_sizes = []
    file_labels = get_unique_file_labels(filenames)
    for filename, label in zip(filenames, file_labels):
        source = create_dask_source(filename, source_metadata)
        pixel_size = source.get_pixel_size()
        size = source.get_physical_size()
        sizes.append(size)
        position = source.get_position()
        rotation = source.get_rotation()
        rotations.append(rotation)

        summary += (f'{label}'
                    f'\t{print_dict_xyz(pixel_size)}'
                    f'\t{print_dict_xyz(size)}'
                    f'\t{print_dict_xyz(position)}')
        if rotation is not None:
            summary += f'\t{rotation}'
        summary += '\n'

        center = {dim: position[dim] + size.get(dim, 0)/2 for dim in position}
        pixel_sizes.append(pixel_size)
        centers.append(center)
        positions.append(position)
        max_positions.append({dim: position[dim] + size.get(dim, 0) for dim in position})
    pixel_size = {dim: float(np.mean([pixel_size[dim] for pixel_size in pixel_sizes])) for dim in pixel_sizes[0]}
    center = {dim: float(np.mean([center[dim] for center in centers])) for dim in centers[0]}
    min_position = {dim: min([position[dim] for position in positions]) for dim in positions[0]}
    max_position = {dim: max([position[dim] for position in max_positions]) for dim in max_positions[0]}
    area = {dim: max_position[dim] - min_position[dim] for dim in max_position}
    summary += f'Area: {print_dict_xyz(area)} Center: {print_dict_xyz(center)}\n'

    rotations2 = []
    for rotation in rotations:
        if rotation is None:
            _, angles = get_pairs(centers, sizes)
            if len(angles) > 0:
                rotation = -np.mean(angles)
                rotations2.append(rotation)
    if len(rotations2) > 0:
        rotation = np.mean(rotations2)
    else:
        rotation = None
    return {'pixel_size': pixel_size,
            'center': center,
            'area': area,
            'rotation': rotation,
            'summary': summary}
