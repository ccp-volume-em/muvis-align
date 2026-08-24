import numpy as np

from muvis_align.image.util import combine_transforms
from muvis_align.util import dir_regex, find_all_numbers, split_numeric_dict, import_json, export_json


def get_composite_transforms(transforms, global_transforms):
    transforms2 = {}
    for key, tile_transforms in transforms.items():
        global_transform = global_transforms[key]
        tile_transforms2 = {}
        for tile_key, transform in tile_transforms.items():
            tile_transforms2[tile_key] = combine_transforms([transform, global_transform]).tolist()
        transforms2[key] = tile_transforms2
    return transforms2


def make_z_transforms(transforms, to3d=False):
    transforms2 = {}
    for key, transforms1 in transforms.items():
        for key2, transform in transforms1.items():
            if len(transform) == 3 and to3d:
                transform2 = np.eye(4)
                transform2[1:, 1:] = transform
                transform = transform2.tolist()
            transforms2[f'{key}_{key2}'] = transform
    return transforms2


if __name__ == "__main__":
    # /nemo/project/proj-ccp-vem/datasets/12193
    stitched_path = 'D:/slides/12193/stitched_hpc/S???/mappings.json'
    aligned_path = 'D:/slides/12193/aligned_hpc/mappings.json'
    output_path = 'aligned_stitched_mappings1.json'

    stitched_filenames = dir_regex(stitched_path)
    stitched_filenames = sorted(stitched_filenames, key=lambda file: list(find_all_numbers(file)))  # sort first key first
    stitched_transforms = {'S' + split_numeric_dict(filename)['S']: import_json(filename) for filename in stitched_filenames}
    aligned_transforms = import_json(aligned_path)
    transforms2 = make_z_transforms(get_composite_transforms(stitched_transforms, aligned_transforms), to3d=True)
    export_json(output_path, transforms2)
    print(transforms2)
