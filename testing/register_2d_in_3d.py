# example of embedding 2d sim into 3d with correct transforms
# for registering slices in a 3d stack

from copy import deepcopy
import dask
from matplotlib import pyplot as plt
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import registration, param_utils, msi_utils, fusion
import numpy as np

from muvis_align.image.util import get_sim_position_final, show_image
from muvis_align.util import get_pairs

dask.config.set(scheduler='threads')


def generate_brownian_noise(size):
    """
    Generates Brownian noise by integrating white noise.

    Brownian noise has a power spectrum that decreases with frequency (1/f²),
    creating a "rumbling" sound that emphasizes lower frequencies.
    Also known as red noise or random walk noise.

    Args:
        size (int / list): Number of samples to generate

    Returns:
        numpy.ndarray: Array of Brownian noise samples normalized to prevent clipping
    """
    # Start with white noise as the base signal
    white_noise = np.random.normal(0, 1, size)
    # Perform a cumulative sum (integration) to create the brownian effect
    # This accumulates previous values, creating the characteristic low-frequency emphasis
    brownian_noise = white_noise.cumsum(-1)
    for i in range(2, white_noise.ndim + 1):
        np.cumsum(brownian_noise, axis=-i, out=brownian_noise)
    brownian_noise = np.abs(brownian_noise - np.mean(brownian_noise)) / (len(brownian_noise) * 2)
    return brownian_noise


def register_3d_sims_in_2d(
    fixed_data,
    moving_data,
    *,
    fixed_origin,
    moving_origin,
    fixed_spacing,
    moving_spacing,
    initial_affine,
    transform_types=None,
    **ants_registration_kwargs,
    ):
    """
    Register two 3d sims by projecting them to 2d and using 2d registration.
    The z component of the resulting affine matrix is set to identity.
    """
    fixed_data = fixed_data.max('z')
    moving_data = moving_data.max('z')

    dims2d = ['y', 'x']
    fixed_origin = {dim: fixed_origin[dim] for dim in dims2d}
    moving_origin = {dim: moving_origin[dim] for dim in dims2d}
    fixed_spacing = {dim: fixed_spacing[dim] for dim in dims2d}
    moving_spacing = {dim: moving_spacing[dim] for dim in dims2d}

    initial_affine = initial_affine[1: , 1:]

    # call 2d registration on the projected sims
    # reg_res_2d = registration.phase_correlation_registration(
    #     sim1, sim2, **kwargs)
    reg_res_2d = registration.registration_ANTsPy(
        fixed_data,
        moving_data,
        fixed_origin=fixed_origin,
        moving_origin=moving_origin,
        fixed_spacing=fixed_spacing,
        moving_spacing=moving_spacing,
        initial_affine=initial_affine,
        transform_types=transform_types,
        **ants_registration_kwargs,
        )

    # embed resulting 2d affine matrix into 3d affine matrix
    reg_res_3d = deepcopy(reg_res_2d)
    reg_res_3d['affine_matrix'] = param_utils.identity_transform(3)
    reg_res_3d['affine_matrix'][1:, 1:] = reg_res_2d['affine_matrix']

    return reg_res_3d


def register_3d_sims_in_2d_simple(
    fixed_data,
    moving_data,
    **ants_registration_kwargs,
    ):
    """
    Register two 3d sims by projecting them to 2d and using 2d registration.
    The z component of the resulting affine matrix is set to identity.
    """
    fixed_data = fixed_data.max('z')
    moving_data = moving_data.max('z')

    # call 2d registration on the projected sims
    reg_res_2d = registration.phase_correlation_registration(
        fixed_data,
        moving_data,
        **ants_registration_kwargs,
        )

    # embed resulting 2d affine matrix into 3d affine matrix
    reg_res_3d = deepcopy(reg_res_2d)
    reg_res_3d['affine_matrix'] = param_utils.identity_transform(3)
    reg_res_3d['affine_matrix'][1:, 1:] = reg_res_2d['affine_matrix']

    return reg_res_3d


# create simulated 2d sims
dz = 1 # currently only works for dz = 1
dxy = 0.1
size = (1, 1000, 1000)
z_tiles, y_tiles, x_tiles = [6, 5, 5]
sims = []

full_size = generate_brownian_noise(size)

for z in range(z_tiles):
    for y in range(y_tiles):
        for x in range(x_tiles):
            x0 = max(x * 200 - 20, 0)
            x1 = min(x0 + 240, size[2])
            y0 = max(y * 200 - 20, 0)
            y1 = min(y0 + 240, size[1])

            sims.append(si_utils.get_sim_from_array(
                full_size[:, y0:y1, x0:x1],
                translation={'z': z * dz, 'y': y * 200 * dxy, 'x': x * 200 * dxy},
                scale={'y': dxy, 'x': dxy, 'z': dz},
                ))

origins = np.array([get_sim_position_final(sim) for sim in sims])
pairs, _ = get_pairs(origins, [size] * len(sims))
msims = [msi_utils.get_msim_from_sim(im) for im in sims]

# register indicating a z overlap tolerance for pairing slices
params = registration.register(
    msims,
    transform_key='affine_metadata',
    new_transform_key='affine_registered_in_2d',
    reg_channel_index=0,
    overlap_tolerance={'z': 1}, # allow pairing of slices that don't initially overlap
    pairwise_reg_func=register_3d_sims_in_2d_simple,
    pairs=pairs,
    n_parallel_pairwise_regs=1
)

# calculate output stack properties for fusion
output_stack_properties = fusion.calc_fusion_stack_properties(
    sims, params, spacing={'z': dz, 'y': dxy, 'x': dxy}, mode='union')

# fuse registered sims
fused = fusion.fuse(
    [msi_utils.get_sim_from_msim(msim) for msim in msims],
    transform_key='affine_registered_in_2d',
    output_stack_properties=output_stack_properties
).compute()

# visualize fused result
plt.figure()
fig, axs = plt.subplots(1, z_tiles + 1)
axs[0].imshow(full_size[0, :, :])
for i in range(z_tiles):
    axs[i + 1].imshow(fused.data[0, 0, i, :, :])
plt.show()
