import numpy as np

from src.muvis_align.fusion_methods.FusionMethod import FusionMethod


class FusionMethodAdditive(FusionMethod):
    def fusion(self, transformed_views):
        """
        Simple additive fusion

        Parameters
        ----------
        transformed_views : list of ndarrays
            transformed input views

        Returns
        -------
        ndarray
            Fusion of input views
        """
        maxval = 2 ** (8 * self.source_type.itemsize) - 1
        return np.nanmean(transformed_views, axis=0, dtype=transformed_views[0].dtype).clip(0, maxval)
