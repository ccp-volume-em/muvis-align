# Quality metrics

### Metrics returned by multiview-sticher register()
- Quality: Pairwise spearman correlation on the overlap between tiles
- Residual: the distance by which the groupwise resolution shifts pairs of views with respect to their pairwise registered positions
  - Virtual point correspondences between pairs of views, defined by the pairwise registration parameters. the residuals are the distances between these points after groupwise resolution. How are the points defined? It's basically the vertices of the overlap area/volume for each of the views. The method is similar to the one used in bigstitcher (section 12): https://www.janelia.org/sites/default/files/H%C3%B6rl%202019.SOM_.pdf

### Core metrics
- Normalized Cross Correlation Coefficient (NCC) (-1 ... 1, larger (magnitude) is better)
- Mean Structural Similarity Index (SSIM) (-1 ... 1, larger (magnitude) is better)
- Overlapping Normalized Mutual Information (ONMI) (0 ... 1, larger is better - note: skimage returns values 1 ... 2)
- Single image Fourier ring correlation - related to determining optical resolution (not possible to compute alignment accuracy for 2 unaligned images)
  - https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-12-21767
