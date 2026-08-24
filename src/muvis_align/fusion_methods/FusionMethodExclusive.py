import numpy as np

from muvis_align.fusion_methods.FusionMethod import FusionMethod


class FusionMethodExclusive(FusionMethod):
    def fusion(self, transformed_views):
        """
        Simple exclusive fusion:
        1. in case of mixed pixel sizes: do not blend, in lieu of knowing actual pixel sizes prioritise 'smallest' (non-NaN) tiles first
        2. avoid blending in general - exclusive use of single source information

        Parameters
        ----------
        transformed_views : list of ndarrays
            transformed input views

        Returns
        -------
        ndarray
            Fusion of input views
        """
        needs_cleanup = False
        if len(transformed_views) > 1:
            needs_cleanup = True
            # create exclusive weights map for all views
            weights = np.zeros(transformed_views.shape, dtype=bool)
            # mask with all non-NaN pixels
            mask = ~np.isnan(transformed_views)
            # remaining mask (single mask for all views)
            rem_mask = np.ones(transformed_views.shape[1:], dtype=bool)
            n = np.count_nonzero(mask, axis=(1,2))
            indices = np.argsort(n)
            # prioritise smallest shapes (in lieu of actual pixel size info)
            for index in indices:
                mask1 = mask[index]
                weights[index][mask1 * rem_mask] = True
                # remove already assigned pixels from remaining mask
                rem_mask *= ~mask1
            # apply weights
            transformed_views *= weights

        fused = np.nansum(transformed_views, axis=0, dtype=transformed_views[0].dtype)
        if needs_cleanup:
            del weights
            del mask
            del rem_mask
        return fused
