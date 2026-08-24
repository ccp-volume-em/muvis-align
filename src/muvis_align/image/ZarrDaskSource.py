import ngff_zarr as nz

from muvis_align.image.DaskSource import DaskSource
from muvis_align.util import convert_to_um


class ZarrDaskSource(DaskSource):
    def init_metadata(self):
        multiscales = nz.from_ngff_zarr(self.filename)
        for index, ngff_image in enumerate(multiscales.images):
            self.pixel_sizes.append({dim: convert_to_um(value, ngff_image.axes_units.get(dim, 'um'))
                                     for dim, value in ngff_image.scale.items() if dim in 'xyz'})
            if index == 0:
                self.position = {dim: convert_to_um(value, ngff_image.axes_units.get(dim, 'um'))
                                 for dim, value in ngff_image.translation.items()}
                if ngff_image.channel_names:
                    for index, channel_name in enumerate(ngff_image.channel_names):
                        channel = {'label': channel_name}
                        if ngff_image.channel_colors:
                            channel['color'] = ngff_image.channel_colors[index]
                        self.channels.append(channel)
            self.data.append(ngff_image.data)

        metadata = multiscales.metadata
        dims = metadata.dimension_names
        self.dimension_order = ''.join(dims)
        self.shapes = [level.shape for level in self.data]
        self.shape = self.shapes[0]
        self.dtype = self.data[0].dtype

        self.pixel_size = self.pixel_sizes[0]
        self.rotation = 0
