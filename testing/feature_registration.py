import logging
import numpy as np
import yaml

from src.muvis_align.MVSRegistration import MVSRegistration
from src.muvis_align.Timer import Timer
from src.muvis_align.image.source_helper import create_dask_source
from src.muvis_align.image.util import *
from src.muvis_align.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures as RegMethod


def test_feature_registration():
    params = 'resources/params_EMPIAR12193.yml'
    with open(params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation = params['operations'][1]

    target_scale = 4
    reg = MVSRegistration(params['general'])
    sims = reg.init_sims(target_scale=target_scale)
    sims, norm_sims, _ = reg.preprocess(sims, operation)
    sim0 = norm_sims[0]
    reg_method = RegMethod(sim0.dtype, operation['method'])

    sizes = [get_sim_physical_size(sim) for sim in sims]
    origins = np.array([get_sim_position_final(sim, position) for sim, position in zip(sims, reg.positions)])
    pairs, _ = get_pairs(origins, sizes)
    for pair in pairs:
        overlap_sim1, overlap_sim2, _ = get_overlap_images(norm_sims[pair[0]], norm_sims[pair[1]], reg.source_transform_key)
        result = reg_method.registration(overlap_sim1, overlap_sim2)
        print(result)


def test_feature_registration_simple():
    folder = 'D:/slides/12193/data_overlaps/'
    filenames = [
        folder + 'tile_37.tiff',
        folder + 'tile_46.tiff'
    ]
    reg_params = {
        'name': 'sift',
        'gaussian_sigma': 4,
        'downscale_factor': 1.414,
        'inlier_threshold_factor': 0.05,
        'max_trials': 10000,
        'ransac_iterations': 10,
    }

    images = [create_dask_source(filename).get_data() for filename in filenames]
    image0 = images[0]

    reg_method = RegMethod(image0, reg_params, debug=True)
    result = reg_method.registration(*images)
    print(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with Timer('Registration', auto_unit=False):
        #test_feature_registration()
        test_feature_registration_simple()
