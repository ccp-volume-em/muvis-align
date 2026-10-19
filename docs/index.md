# muvis-align

Image registration pipeline for large volume EM and LM datasets, built on
[multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher), with
a napari plugin.

muvis-align handles x-y stitching and z reconstruction across image modalities, is
designed for datasets in the terabyte range, and reads and writes the
[OME-Zarr](https://ngff.openmicroscopy.org/) next-generation file format throughout.

## Installation

muvis-align needs Python 3.12 or later.

```bash
pip install muvis-align
```

Or, with napari and Qt included:

```bash
pip install "muvis-align[all]"
```

The command-line script (`run.py`), the example parameter files (`resources/`) and the
notebooks (`notebooks/`) are in the [source repository](https://github.com/ccp-volume-em/muvis-align),
not the pip package, so clone it to use those.

## Three ways to run it

| | |
|---|---|
| **[napari plugin](napari.md)** | Load a dataset, inspect the source layout, preview a single pair or the fused result, and run each step interactively. This is what to use to work out the parameters. |
| **[Command-line pipeline](pipeline.md)** | A YAML parameter file describing one or more operations, run with `python run.py params.yml`. This is what to use for batch and cluster runs. |
| **[Jupyter notebooks](notebooks.md)** | The registration classes driven step by step from Python, from a complete pipeline to feature-matching and overlap visualisation. This is what to use to script a custom workflow or look inside a registration. |

The plugin and the pipeline keep separate configuration files, so a project set up in
the plugin is not directly a pipeline parameter file.

The plugin can also run in a container, in the browser through Xpra: see the
[Dockerfile](technical.md#interactive-interface-with-xpra-container).

## Getting started

1. Install as above.
2. Open the [napari plugin](napari.md) and follow the [step-by-step guide](napari_guide.md)
   to set up a project and work out your parameters.
3. For batch runs, copy one of the example parameter files under `resources/` and run it
   with the [command-line pipeline](pipeline.md).
4. To work from Python, start with `pipeline.ipynb` in the [notebooks](notebooks.md).

## Reference

- [napari plugin](napari.md) - the interactive interface
- [Pipeline parameters](pipeline.md) - every option the YAML file accepts
- [Notebooks](notebooks.md) - worked examples and debugging workflows
- [Technical overview](technical.md) - how the source implements NGFF, lazy data, FAIR metadata and resumable runs
- [API reference](references.md) - generated from the source
- [Poster](poster.md) - project background and results

## Links

- Source: [github.com/ccp-volume-em/muvis-align](https://github.com/ccp-volume-em/muvis-align)
- Generated documentation: [deepwiki.com/ccp-volume-em/muvis-align](https://deepwiki.com/ccp-volume-em/muvis-align)
- multiview-stitcher: [github.com/multiview-stitcher/multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher)

Distributed under the MIT licence.
