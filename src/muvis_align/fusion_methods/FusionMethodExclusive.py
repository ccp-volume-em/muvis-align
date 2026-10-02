import numpy as np

from muvis_align.fusion_methods.FusionMethod import FusionMethod


class FusionMethodExclusive(FusionMethod):
    def fusion(self, transformed_views):
        """
        Exclusive fusion: each pixel from a single view, no blending - the earliest imaged view covering it.

        Parameters
        ----------
        transformed_views : ndarray
            transformed input views (views, *spatial), NaN where a view does not reach

        Returns
        -------
        ndarray
            Fusion of input views, 0 where no view reaches
        """
        valid = ~np.isnan(transformed_views)
        # views arrive in source order, taken as imaging order: the first covering one per pixel wins
        first = np.argmax(valid, axis=0)
        fused = np.take_along_axis(transformed_views, first[None], axis=0)[0]
        return np.where(np.isnan(fused), 0, fused).astype(transformed_views.dtype, copy=False)
