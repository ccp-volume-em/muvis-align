# MuVis-ALIGN <br>next generation image registration

## Joost de Folter (Francis Crick Institute, London), Marvin Albert (ETH, Zurich)

Large-volume microscopy datasets present significant challenges for existing image registration and stitching workflows. Commonly used tools often assume that data fit into memory, rely on monolithic image formats, or lack support for emerging standards such as OME-Zarr, an implementation of the OME Next-Generation File Format (NGFF) designed for scalable access to large imaging data. These limitations become increasingly restrictive as microscopy datasets grow in size and complexity.

Here, we present muvis-align, a scalable workflow for alignment and stitching of large imaging datasets across microscopy modalities. Building on multiview-stitcher, muvis-align provides a streamlined pipeline designed for straightforward deployment on high-performance computing (HPC) infrastructure. Operating directly on chunked, multiscale OME-Zarr data enables processing of larger-than-memory datasets while supporting interoperability with the wider bioimaging software ecosystem. Emphasis is placed on usability, reproducible deployment, and preservation of image metadata throughout processing.

By combining scalable computation with standardized data representation and metadata handling, muvis-align promotes FAIR data principles and reproducible processing. It provides a practical route from large microscopy acquisitions to aligned datasets suitable for downstream visualization and quantitative analysis.

[github.com/ccp-volume-em/muvis-align](https://github.com/ccp-volume-em/muvis-align)

[github.com/multiview-stitcher/multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher)



**CCP-volumeEM**: Martin Jones, Lucy Collinson, Michele Darrow, Matthew Hartley, Leandro Lemgruber, Helen Spiers, Amy Strange, Paul Verkade, Martyn Winn

https://zenodo.org/records/...
