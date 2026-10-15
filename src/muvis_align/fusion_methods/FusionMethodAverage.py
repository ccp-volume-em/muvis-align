import numpy as np

from muvis_align.fusion_methods.FusionMethod import FusionMethod


class FusionMethodAverage(FusionMethod):
    def fusion(self, transformed_views):
        """
        Average fusion: the mean of the views reaching each pixel, as multiview_stitcher's simple_average_fusion,
        in one pass instead of a stack and a NaN-free copy per view - the same sums in the same order.

        Parameters
        ----------
        transformed_views : ndarray
            transformed input views (views, *spatial), NaN where a view does not reach

        Returns
        -------
        ndarray
            Fusion of input views, NaN where no view reaches
        """
        valid = ~np.isnan(transformed_views)
        counts = valid.sum(axis=0, dtype=np.float32)
        counts[counts == 0] = np.nan
        return (np.where(valid, transformed_views, 0).sum(axis=0) / counts).astype(transformed_views[0].dtype)
