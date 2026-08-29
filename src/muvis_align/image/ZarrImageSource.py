from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.color_conversion import hexrgb_to_rgba


class ZarrImageSource(ImageSource):
    def init_metadata(self):
        msim = ngff_utils.read_msim_from_ome_zarr(self.filename, array_backend='dask',
                                                  transform_key=self.transform_key)
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        sims = [msi_utils.get_sim_from_msim(msim, scale=scale_key) for scale_key in scale_keys]
        sim0 = sims[0]

        self.dimension_order = ''.join(sim0.dims)
        # per-level raw arrays, so the base class can rebuild self.msim with final (override-aware) geometry
        self.data = [sim.data for sim in sims]
        self.shapes = [sim.shape for sim in sims]
        self.shape = self.shapes[0]
        self.dtype = sim0.dtype

        self.pixel_sizes = [si_utils.get_spacing_from_sim(sim) for sim in sims]
        self.pixel_size = self.pixel_sizes[0]
        self.position = si_utils.get_origin_from_sim(sim0)

        if 'c' in sim0.dims:
            self.channels = [{'label': str(label)} for label in sim0.coords['c'].values]
            omero = msim.attrs.get('omero')
            if omero:
                for channel, ch_meta in zip(self.channels, omero.get('channels', [])):
                    if 'color' in ch_meta:
                        channel['color'] = hexrgb_to_rgba(ch_meta['color'])

        self.rotation = 0
