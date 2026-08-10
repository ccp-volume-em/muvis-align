import tifffile
from ngff_zarr import tiff_file_to_ngff_images

from muvis_align.image.ome_tiff_helper import extract_ome_translation
from muvis_align.util import convert_to_um
from src.muvis_align.image.DaskSource import DaskSource
from src.muvis_align.image.color_conversion import hexrgb_to_rgba


class TiffDaskSource(DaskSource):
    def init_metadata(self):
        # work-around to get translation from OME-TIFF metadata
        # since ngff-zarr does not support it yet
        self.position = extract_ome_translation(self.filename)

        ngff_image_datas = tiff_file_to_ngff_images(self.filename, reuse_existing_pyramids=True)
        for ngff_image_data in ngff_image_datas:
            multiscales = ngff_image_data[1]
            metadata = multiscales.metadata
            dims = metadata.dimension_names
            self.dimension_order = ''.join(dims)
            for index, ngff_image in enumerate(multiscales.images):
                if index == 0:
                    #self.position = {dim: convert_to_um(value, ngff_image.axes_units.get(dim, 'um'))
                    #                 for dim, value in ngff_image.translation.items()}
                    if ngff_image.channel_names:
                        for index, channel_name in enumerate(ngff_image.channel_names):
                            channel = {'label': channel_name}
                            if ngff_image.channel_colors:
                                channel['color'] = hexrgb_to_rgba(ngff_image.channel_colors[index])
                            self.channels.append(channel)
                self.pixel_sizes.append({dim: convert_to_um(value, ngff_image.axes_units.get(dim, 'um'))
                                         for dim, value in ngff_image.scale.items() if dim in 'xyz'})
                self.data.append(ngff_image.data)

        self.dtype = self.data[0].dtype
        self.shapes = [data.shape for data in self.data]
        self.shape = self.shapes[0]
        self.is_rgb = (self.get_nchannels() in (3, 4))  # TODO: check with RGB image if better approach is possible

        self.pixel_size = self.pixel_sizes[0]
        self.rotation = 0


def tags_to_dict(tags: tifffile.TiffTags) -> dict:
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict
