# muvis-align

[![License MIT](https://img.shields.io/github/license/ccp-volume-em/muvis-align?color=green)](https://github.com/ccp-volume-em/muvis-align/raw/main/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/muvis-align.svg?color=green)](https://pypi.org/project/muvis-align)
[![Python Version](https://img.shields.io/pypi/pyversions/muvis-align.svg?color=green)](https://python.org)
[![tests](https://github.com/ccp-volume-em/muvis-align/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/ccp-volume-em/muvis-align/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/ccp-volume-em/muvis-align/branch/main/graph/badge.svg)](https://codecov.io/gh/ccp-volume-em/muvis-align)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

Image registration pipeline for large volume EM and LM datasets, built on
[multiview-stitcher], with a [napari] plugin, a command-line pipeline and Jupyter notebooks.

## Documentation

Main documentation: [ccp-volume-em.github.io/muvis-align](https://ccp-volume-em.github.io/muvis-align/)

Generated documentation: [deepwiki.com/ccp-volume-em/muvis-align](https://deepwiki.com/ccp-volume-em/muvis-align)

## Installation

### From napari

The simplest way: in [napari], open *Plugins > Install/Uninstall Plugins...*, search for `muvis-align` and
install it. It is listed on the [napari hub](https://napari-hub.org/plugins/muvis-align).

### With pip

muvis-align needs Python 3.12 or later. Install it via [pip]:

```
pip install muvis-align
```

If napari is not already installed, you can install `muvis-align` with napari and Qt via:

```
pip install "muvis-align[all]"
```

The command-line script, example parameter files and notebooks are in this repository,
not the pip package: see the [documentation](https://ccp-volume-em.github.io/muvis-align/).

## Running on a Slurm cluster (HPC)

The napari plugin can run in a container on a Slurm cluster, shown in your browser through [Xpra].
This needs Apptainer on the cluster, ssh access to its login node (preferably with an ssh key), and a clone of this
repository on your own machine.

From your own machine, one command builds the container on the cluster if needed, starts the job, and opens napari
in your browser (on Windows in Git Bash, or as `.\xpra-connect.sh ...` from PowerShell):

```
bash xpra-connect.sh <user>@<login node> [folder on the cluster, default: muvis-align] [--pull]
```

Or step by step, on the cluster's login node in a folder with [xpra-pull.sh] and [xpra-run.sh]:

```
bash xpra-pull.sh     # once, and to update the container
bash xpra-run.sh      # starts the job and prints how to connect from your own machine
```

See the [documentation](https://ccp-volume-em.github.io/muvis-align/technical/#on-a-slurm-cluster) for details.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"muvis-align" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

[multiview-stitcher]: https://github.com/multiview-stitcher/multiview-stitcher
[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[MIT]: http://opensource.org/licenses/MIT
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[Xpra]: https://xpra.org/
[xpra-pull.sh]: https://github.com/ccp-volume-em/muvis-align/blob/main/xpra-pull.sh
[xpra-run.sh]: https://github.com/ccp-volume-em/muvis-align/blob/main/xpra-run.sh
[file an issue]: https://github.com/ccp-volume-em/muvis-align/issues
