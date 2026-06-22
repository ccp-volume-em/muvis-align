import glob
from multiview_stitcher import spatial_image_utils as si_utils
import os.path

from src.muvis_align.constants import zarr_extension
from src.muvis_align.image.ome_helper import save_image
from src.muvis_align.image.util import redimension_data
from src.muvis_align.util import get_filetitle, get_unique_file_labels
from src.muvis_align.image.source_helper import create_dask_source


def convert_to_zarr(filename, output_filename, source_metadata, output_order='tczyx'):
    print('Converting ' + filename + ' to ' + output_filename + zarr_extension)
    source = create_dask_source(filename, source_metadata)
    data = redimension_data(source.get_data(), source.dimension_order, output_order)
    sim = si_utils.get_sim_from_array(
        data,
        dims=list(output_order),
        scale=source.get_pixel_size(),
        translation=source.get_position()
    )
    save_image(output_filename, sim, channels=source.get_channels(), npyramid_add=4)


if __name__ == "__main__":
    input_path = 'data/*/*.tiff'
    output_path = 'output/'
    source_metadata = {
        'scale': {'x': 0.032, 'y': 0.032, 'z': 0.1},
        'position': {'y': 'fn[-3]*24', 'x': 'fn[-2]*24', 'z': 'S*0.1'}}

    filenames = glob.glob(input_path)
    file_labels = get_unique_file_labels(filenames)

    for filename, file_label in zip(filenames, file_labels):
        output_filename = os.path.join(output_path, file_label)
        convert_to_zarr(filename, output_filename, source_metadata)
