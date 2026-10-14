"""Write a synthetic project the size of a large HPC set - many tiny tiles over many sections - with a saved
registration, to time opening it and refreshing the view at that scale in minutes:

    python testing/make_synthetic_project.py <folder> [--sections 1133] [--rows 5] [--cols 6]

Each tile is 64x64 pixels of 0.4um, overlapping its neighbours by 10%. Pairs join each tile to its neighbours in its
section and to the tiles at and beside it in the next section. The registration is a small offset per tile.
"""
import argparse
import json
import os

import numpy as np
import tifffile
import yaml

from muvis_align.constants import default_quality_key
from muvis_align.file.transforms import write_transforms
from muvis_align.util import create_transform, export_json

TILE, PIXEL_SIZE, OVERLAP = 64, 0.4, 0.1
SECTION_OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]
NEXT_SECTION_OFFSETS = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--sections', type=int, default=1133)
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--cols', type=int, default=6)
    args = parser.parse_args()
    folder = os.path.abspath(args.folder).replace(os.sep, '/')
    output = f'{folder}/output'
    os.makedirs(output, exist_ok=True)
    rng = np.random.default_rng(0)
    step = TILE * PIXEL_SIZE * (1 - OVERLAP)

    filenames = {}
    for section in range(args.sections):
        os.makedirs(f'{folder}/S{section:04d}', exist_ok=True)
        for row in range(args.rows):
            for col in range(args.cols):
                path = f'{folder}/S{section:04d}/tile_{section:04d}_{row:02d}_{col:02d}.tif'
                tifffile.imwrite(path, rng.integers(20, 235, (TILE, TILE), dtype=np.uint8))
                filenames[section, row, col] = path

    pairs = {}
    for (section, row, col), path in filenames.items():
        neighbours = [(section, row + dy, col + dx) for dy, dx in SECTION_OFFSETS]
        neighbours += [(section + 1, row + dy, col + dx) for dy, dx in NEXT_SECTION_OFFSETS]
        for key in neighbours:
            if key in filenames:
                pairs[json.dumps([path, filenames[key]])] = {
                    'mapping': np.eye(3).tolist(), default_quality_key: float(rng.uniform(0.3, 1)),
                    'bbox': [[0.0, 0.0], [TILE * PIXEL_SIZE * OVERLAP, TILE * PIXEL_SIZE]]}
    export_json(f'{output}/pair_mappings.json', pairs)
    export_json(f'{output}/metrics.json', {key: {default_quality_key: value[default_quality_key],
                                                 'ncc': float(rng.uniform(0.3, 1))} for key, value in pairs.items()})
    write_transforms(f'{output}/mappings.json',
                     {path: create_transform(None, 0, translation=list(rng.uniform(-1, 1, 2)), matrix_size=3)
                      for path in filenames.values()})

    project = {'input_output': {
        'source_position_z': 'fn[-3]', 'source_position_y': f'fn[-2]*{step}', 'source_position_x': f'fn[-1]*{step}',
        'source_scale_z': '1', 'source_scale_y': str(PIXEL_SIZE), 'source_scale_x': str(PIXEL_SIZE),
        'source_rotation': '0', 'normalise_rotated_positions': False, 'preview_scale': '1',
        'coordinate_system': 'source_metadata', 'metadata_table': [], 'registration_dimension': 'space',
        'output_path': 'output', 'overwrite': True, 'preview_shapes': True, 'preview_images': True,
        'input_path': 'S*/*.tif'},
        'pre_processing': {'scale': '1', 'flatfield_quantiles': '', 'normalisation': 'none',
                           'filter_foreground': False, 'gaussian_sigma': 0.0},
        'registration': {'operation': 'register', 'pairing': 'orthogonal', 'transform_type': 'rigid',
                         'method': 'phase_correlation', 'groupwise_resolution_method': 'robust_linear',
                         'channel': '0', 'metrics_table': []},
        'fusion': {'method': 'average', 'spacing': 'native', 'ome_version': '0.5'}}
    with open(f'{folder}/muvis_align_project.yml', 'w') as file:
        yaml.safe_dump(project, file, sort_keys=False)
    print(f'{len(filenames)} tiles in {args.sections} sections, {len(pairs)} pairs: {folder}/muvis_align_project.yml')


if __name__ == '__main__':
    main()
