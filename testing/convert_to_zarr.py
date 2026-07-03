import dask.array as da
import glob
from multiview_stitcher import spatial_image_utils as si_utils
import os.path

from muvis_align.logging import init_logging
from src.muvis_align.constants import zarr_extension
from src.muvis_align.image.ome_helper import save_image
from src.muvis_align.image.util import redimension_data
from src.muvis_align.image.source_helper import create_dask_source
from src.muvis_align.Timer import Timer
from src.muvis_align.util import get_unique_file_labels


def convert_to_zarr(filename, output_filename, source_metadata, output_order='tczyx'):
    print('Converting ' + filename + ' to ' + output_filename + zarr_extension)
    source = create_dask_source(filename, source_metadata)
    data = redimension_data(source.get_data(), source.dimension_order, output_order)
    print(source.get_position())
    sim = si_utils.get_sim_from_array(
        data,
        dims=list(output_order),
        scale=source.get_pixel_size(),
        translation=source.get_position()
    )
    save_image(output_filename, sim, channels=source.get_channels(), npyramid_add=4)


def convert_stack_to_zarr(filenames, output_filename, source_metadata, output_order='tczyx'):
    sources = [create_dask_source(filename, source_metadata) for filename in filenames]
    source0 = sources[0]
    z_axis = output_order.index('z')
    data = da.concatenate([redimension_data(source.get_data(), source.dimension_order, output_order) for source in sources], axis=z_axis)
    position = source0.get_position()
    scale = source0.get_pixel_size()
    print(position, scale)
    sim = si_utils.get_sim_from_array(
        data,
        dims=list(output_order),
        scale=scale,
        translation=position
    )
    save_image(output_filename, sim, npyramid_add=4)


if __name__ == "__main__":
    # input_path = 'data/*/*.tiff'
    # output_path = 'output/'
    # source_metadata = {
    #     'scale': {'x': 0.032, 'y': 0.032, 'z': 0.1},
    #     'position': {'y': 'fn[-3]*24', 'x': 'fn[-2]*24', 'z': 'S*0.1'}}

    # filenames = glob.glob(input_path)
    # file_labels = get_unique_file_labels(filenames)
    #
    # for filename, file_label in zip(filenames, file_labels):
    #     output_filename = os.path.join(output_path, file_label)
    #     convert_to_zarr(filename, output_filename, source_metadata)



    folders = glob.glob('D:/slides/Emb3_2x2_Mosaic_2CAMs/*')
    file_pattern = 'CAM1--*.tif'
    output_path = 'D:/slides/Emb3_2x2_Mosaic_2CAMs/'
    source_metadata = {
        'scale': {'x': 6.5, 'y': 6.5, 'z': 6.5},
        'position': {'y': 'Y*12000', 'x': 'X*12000', 'z': 0}}

    init_logging()

    for folder in folders:
        if os.path.isdir(folder):
            filenames = sorted(glob.glob(os.path.join(folder, file_pattern)))
            if filenames:
                output_filename = os.path.join(output_path, folder)
                print(f'Converting stack {folder} ({len(filenames)} files) to {output_filename}')
                with Timer(''):
                    convert_stack_to_zarr(filenames, output_filename, source_metadata)
