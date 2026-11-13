# MuViS-ALIGN <br>next generation image registration

## Joost de Folter (Francis Crick Institute, London), Marvin Albert (ETH, Zurich)

Large-volume microscopy datasets present significant challenges for existing image registration and stitching workflows. Commonly used tools often assume that data fit into memory, rely on monolithic image formats, or lack support for emerging standards such as OME-Zarr, an implementation of the OME Next-Generation File Format (NGFF) designed for scalable access to large imaging data. These limitations become increasingly restrictive as microscopy datasets grow in size and complexity.

Here, we present muvis-align, a scalable workflow for alignment and stitching of large imaging datasets across microscopy modalities. Building on multiview-stitcher, muvis-align provides a streamlined pipeline designed for straightforward deployment on high-performance computing (HPC) infrastructure. Operating directly on chunked, multiscale OME-Zarr data enables processing of larger-than-memory datasets while supporting interoperability with the wider bioimaging software ecosystem. Emphasis is placed on usability, reproducible deployment, and preservation of image metadata throughout processing.

By combining scalable computation with standardized data representation and metadata handling, muvis-align promotes FAIR data principles and reproducible processing. It provides a practical route from large microscopy acquisitions to aligned datasets suitable for downstream visualization and quantitative analysis.

INPUT:
- 2D / 3D tiles
- Multi-channel
- Multi-modal

OUTPUT:
- 2D / 3D OME-Zarr
- Transforms
- Quality metrics
- RO-Crate packaged

1. Scalable, interoperable data handling
- Modality-independent: EM and LM
- Input formats: TIFF, OME-TIFF, Zarr, OME-Zarr
- NGFF support: Multiscale OME-Zarr input/output
- Large-volume handling: Lazy loading and chunking
- Sparse output: Native-resolution tiles and pyramids (Compose)
- Flexible export: Fused and sparse OME-Zarr, OME-TIFF

2. Reproducible, FAIR workflows
- Configuration: YAML and napari projects
- Provenance: RO-Crate metadata
- Transform preservation: Pairwise and global mappings
- Interoperability: NGFF coordinate transformations
- Metadata: Acquisition and processing information
- Output packaging: Images, transforms and metrics
- Resumability: Recovery of previous results

3. Performance and resource efficiency
- Larger-than-memory: Lazy loading and blockwise fusion
- Memory efficiency: Adaptive block sizing
- Multicore processing: Parallel registration and fusion
- Multiscale registration: Downsampled alignment
- Quality metrics: NCC, SSIM, ONMI, residuals
- Performance monitoring: Runtime and memory usage
- HPC support: Slurm and containers

4. Interactive alignment and curation
- Napari interface: Interactive workflow execution
- Dataset inspection: Tile positions and overlaps
- Registration preview: Pairwise alignment and features
- Visual feedback: Quality metrics and colour-coded overlaps
- Manual curation: Interactive transformation correction
- Global optimization: Alignment inspection and filtering
- Fusion preview: Review before export
- Remote access: Browser-based HPC interaction

[github.com/ccp-volume-em/muvis-align](https://github.com/ccp-volume-em/muvis-align)

[github.com/multiview-stitcher/multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher)

**CCP-volume-EM**: Martin Jones, Lucy Collinson, Michele Darrow, Matthew Hartley, Leandro Lemgruber, Helen Spiers, Amy Strange, Paul Verkade, Martyn Winn

https://zenodo.org/records/...
