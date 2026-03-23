import logging
from multiview_stitcher import fusion
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os.path
import shutil

from muvis_align.image.source_helper import create_dask_source
from muvis_align.image.util import calc_output_properties, make_sims_3d
from muvis_align.Timer import Timer
from muvis_align.util import print_hbytes


def create_sim(shape, scale={'x': 1, 'y': 1}, position={'x': 0, 'y': 0}, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data = np.random.random(shape)
    sim = si_utils.get_sim_from_array(
        data,
        dims=list('yx'),
        scale=scale,
        translation=position,
        transform_key='reg'
    )
    return sim

def fuse(sims, output_filename, z_scale=None):
    output_stack_properties = calc_output_properties(sims, 'reg', output_spacing='mean', z_scale=z_scale)
    fuse_func = fusion.simple_average_fusion
    zarr_options = {'ome_zarr': True, 'ngff_version': '0.5'}
    output_chunksize = {'x': 1024, 'y': 1024, 'z': 1}
    data_size = np.prod(list(output_stack_properties['shape'].values())) * sims[0].dtype.itemsize
    print(f'Fusing {print_hbytes(data_size)}')
    fused = fusion.fuse(
        sims,
        fusion_func=fuse_func,
        transform_key='reg',
        output_stack_properties=output_stack_properties,
        output_zarr_url=output_filename,
        zarr_options=zarr_options,
        output_chunksize=output_chunksize
    )
    return fused

if __name__ == '__main__':
    shape = (5000, 5000)
    nz = 10
    scale = {'x': 0.05, 'y': 0.05}
    z_scales = [1, 0.05, 5]
    filename = '../output/fused.ome.zarr'

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('ome_zarr').setLevel(logging.WARNING)

    for z_scale in z_scales:
        if os.path.exists(filename):
            shutil.rmtree(filename)

        print('Init sims...')
        sims = [create_sim(shape, scale=scale, seed=index) for index in range(nz)]
        positions = [{'x': 0, 'y': 0, 'z': index * z_scale} for index in range(nz)]
        sims = make_sims_3d(sims, z_scale=z_scale, positions=positions)
        with Timer('test'):
            fused = fuse(sims, filename, z_scale=z_scale)

        fused2 = create_dask_source(filename).get_data()
        for index in range(nz):
            assert np.all(fused2[0, 0, index] == fused[0, 0, index].data)
