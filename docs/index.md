# muvis-align

Image registration pipeline for large volume EM and LM datasets, built on
[multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher), with
a napari plugin.

muvis-align handles x-y stitching and z reconstruction across image modalities, is
designed for datasets in the terabyte range, and reads and writes the
[OME-Zarr](https://ngff.openmicroscopy.org/) next-generation file format throughout.

![EMPIAR-12193 overlay](images/EMPAIR12193overlay.png)

## Installation

```bash
pip install muvis-align
```

Or, with napari and Qt included:

```bash
pip install "muvis-align[all]"
```

## Two ways to run it

| | |
|---|---|
| **[Command-line pipeline](pipeline.md)** | A YAML parameter file describing one or more operations, run with `python run.py params.yml`. This is what to use for batch and cluster runs. |
| **[napari plugin](napari.md)** | Load a dataset, inspect the source layout, preview a single pair or the fused result, and run each step interactively. This is what to use to work out the parameters. |

The two keep separate configuration files, so a project set up in the plugin is not
directly a pipeline parameter file.

## Getting started

1. Install as above.
2. Work out your parameters interactively in the [napari plugin](napari.md), or copy one
   of the example parameter files under `resources/` for the
   [command-line pipeline](pipeline.md).
3. For a full worked example, see the [Jupyter notebooks](notebooks.md).

## Reference

- [Pipeline parameters](pipeline.md) - every option the YAML file accepts
- [napari plugin](napari.md) - the interactive interface
- [Notebooks](notebooks.md) - worked examples and debugging workflows
- [API reference](references.md) - generated from the source
- [Poster](poster.md) - project background and results

## Links

- Source: [github.com/ccp-volume-em/muvis-align](https://github.com/ccp-volume-em/muvis-align)
- Generated documentation: [deepwiki.com/ccp-volume-em/muvis-align](https://deepwiki.com/ccp-volume-em/muvis-align)
- multiview-stitcher: [github.com/multiview-stitcher/multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher)

Distributed under the GNU GPL v3.0 licence.
