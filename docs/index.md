# muvis-align

Image registration pipeline for large volume EM and LM datasets, built on
[multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher), with
a napari plugin.

muvis-align handles x-y stitching and z reconstruction across image modalities, is
designed for datasets in the terabyte range, and reads and writes the
[OME-Zarr](https://ngff.openmicroscopy.org/) next-generation file format throughout.

## Quick start in your browser

With an account on a Slurm cluster (HPC) that has Apptainer, you can use the napari plugin in your web browser
without installing Python or muvis-align yourself: it runs on the cluster, in a container.

You need:

- ssh access to the cluster's login node, ideally with an
  [SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
  so you are not asked for your password
- a terminal: on Windows Git Bash (part of [Git for Windows](https://gitforwindows.org/)), on macOS and Linux the
  built-in one
- the scripts `xpra-connect.sh`, `xpra-run.sh` and `xpra-pull.sh` together in one folder: download the
  [repository as a ZIP](https://github.com/ccp-volume-em/muvis-align/archive/refs/heads/main.zip) and unzip it
  (or clone it)

Then, in that folder, run (on Windows in Git Bash, or as `.\xpra-connect.sh ...` from PowerShell):

```bash
bash xpra-connect.sh <user>@<login node>
```

The first time, this builds the container on the cluster, which takes longer. It then starts a job and opens napari
with the plugin in your browser, where you open your data from the cluster's storage. Closing napari ends the job;
Ctrl-C in the terminal disconnects and offers to cancel it. Add `--pull` to update to the latest version.
Your cluster's support team can tell you the login node's name and whether Apptainer is available; see
[On a Slurm cluster](technical.md#on-a-slurm-cluster) for the options.

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
| **[napari plugin](napari.md)** | Load a dataset, inspect the source layout, preview a single pair or the fused result, and run each step interactively. This is what to use to work out the parameters; on a cluster it needs no install (see the [quick start](#quick-start-in-your-browser)). |
| **[Command-line pipeline](pipeline.md)** | A YAML parameter file describing one or more operations, run with `python run.py params.yml`. This is what to use for batch and cluster runs. |
| **[Jupyter notebooks](notebooks.md)** | The registration classes driven step by step from Python, from a complete pipeline to feature-matching and overlap visualisation. This is what to use to script a custom workflow or look inside a registration. |

The plugin and the pipeline keep separate configuration files, so a project set up in
the plugin is not directly a pipeline parameter file.

The plugin can also run in a container, in the browser through Xpra: on a cluster with the
[quick start](#quick-start-in-your-browser), elsewhere with the
[Dockerfile](technical.md#interactive-interface-with-xpra-container).

## Getting started

1. Install as above, or without Python use the [quick start](#quick-start-in-your-browser) on a cluster.
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
