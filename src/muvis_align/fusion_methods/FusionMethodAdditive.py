from muvis_align.fusion_methods.FusionMethodAverage import FusionMethodAverage


class FusionMethodAdditive(FusionMethodAverage):
    def fusion(self, transformed_views):
        """
        Simple additive fusion: the average, clipped to the source data type's range

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
        return super().fusion(transformed_views).clip(0, maxval)
