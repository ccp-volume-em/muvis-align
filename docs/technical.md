# Technical overview

[muvis-align](https://github.com/ccp-volume-em/muvis-align) is an image registration package for aligning 2D and 3D images.
The package is implemented as a [napari](https://napari.org/stable/) plugin, notebooks using the package classes, and a command line interface (CLI) for batch processing.
The [napari](https://napari.org/stable/) plugin can also be run as a container on a local machine or on a cloud platform via an interactive web interface.

# muvis-align practices demonstrated in the source code

Repository: [https://github.com/ccp-volume-em/muvis-align](https://github.com/ccp-volume-em/muvis-align)

`muvis-align` demonstrates practical patterns for scalable, reproducible image registration workflows across a [napari](https://napari.org/stable/) plugin, notebooks, CLI execution, batch/HPC use, and containerized interactive access.

---

## Next Generation File Formats (NGFF)

`muvis-align` uses OME-Zarr / NGFF as a primary output format.

OME-Zarr / NGFF writing is delegated to `to_multiscales` and `to_ngff_zarr` from [ngff-zarr](https://pypi.org/project/ngff-zarr/), including multiscale pyramid generation.

Source: [`src/muvis_align/image/ome_zarr_helper.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/ome_zarr_helper.py)

```python
from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr


def save_ome_image(data, path, dim_order, pixel_size, channels, translation, rotation,
                   pyramid_downsample=None, compression=None, ome_version=default_ome_zarr_version):

    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation, axes_units=axes_units)
    multiscales = to_multiscales(image, scale_factors=pyramid_downsample, chunks=create_chunk_dict(default_chunk_size, dim_order))
    ...
    to_ngff_zarr(path, multiscales, chunks_per_shard=chunks_per_shard, version=ome_version, **storage_options)

    return multiscales.metadata
```

---

## Handling larger-than-memory data

The code uses [Dask](https://www.dask.org/)-backed image sources for lazy loading and chunked processing.

Source: [`src/muvis_align/image/source_helper.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/source_helper.py)

```python
def create_image_source(filename, source_metadata=None, file_label=None, index=None, transform_key=None):
    kwargs = {'file_label': file_label, 'index': index}
    if transform_key is not None:
        kwargs['transform_key'] = transform_key
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        source = TiffImageSource(filename, source_metadata, **kwargs)
    elif '.zar' in filename.lower():
        source = ZarrImageSource(filename, source_metadata, **kwargs)
    else:
        if ext:
            raise ValueError(f'Unsupported file type: {ext}')
        else:
            raise ValueError(f'Unsupported: {filename}')
    return source
```

TIFF files are read lazily through tifffile's Zarr store (`aszarr`), each pyramid level wrapped as a [Dask](https://www.dask.org/) array, avoiding immediate full-volume loading.

Source: [`src/muvis_align/image/ome_tiff_helper.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/ome_tiff_helper.py)

```python
with tifffile.TiffFile(filename) as tif:
    series = tif.series[0]
    axes = series.axes
    # multiscales=True: a group even for one level, levels ordered finest first
    group = zarr.open_group(store=tif.aszarr(series=0, multiscales=True), mode='r')
    attributes = group.attrs.get('ome', group.attrs)
    datasets = attributes['multiscales'][0]['datasets']
    arrays = []
    for dataset in datasets:
        array = group[dataset['path']]
        ...
for level, array in enumerate(arrays):
    # meta given, so dask does not probe the dtype with a read
    data = da.from_array(array, chunks=array.chunks, meta=np.empty((0,) * array.ndim, dtype=array.dtype),
                         name='tiff-' + tokenize(filename, stat.st_size, stat.st_mtime_ns, level))
    datas.append(data)
```

OME-Zarr metadata is read from the store's consolidated metadata (falling back to [ngff-zarr](https://pypi.org/project/ngff-zarr/)'s parser), and each multiscale level is opened directly as a Dask array.

Source: [`src/muvis_align/image/ZarrImageSource.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/ZarrImageSource.py)

```python
def init_metadata(self):
    # metadata only, off consolidated metadata: the arrays wait for first access to self.data
    metadata = read_ome_zarr_source_metadata(self.filename)

    self.dimension_order = metadata['dimension_order']
    self.shapes = metadata['shapes']
    self.shape = self.shapes[0]
    self.dtype = metadata['dtype']
    ...

def _load_data(self):
    ...
    datas = []
    for path, sim_shape in zip(self._level_paths, self.shapes):
        data = da.from_zarr(self.filename, component=path)
        if tuple(data.shape) != tuple(sim_shape):
            data = data.reshape(sim_shape)
        datas.append(data)
    self._data = datas
```

Fusion writes directly to OME-Zarr, level by level and block by block, rather than creating a full fused array in memory first.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py), [`src/muvis_align/fusion_blocks.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/fusion_blocks.py)

```python
with dask.config.set(scheduler='threads'), warnings.catch_warnings():
    if saving_zarr:
        levels = export_levels(source_spacings, output_stack_properties, scaled_dims, native=is_native)
        fused_image = fuse_levels_to_ome_zarr(
            msims, levels, output_filename, transform_key, output_stack_properties, output_chunksize,
            fusion_func=fuse_func, zarr_options=zarr_options,
            batch_options=self._fusion_batch_options(saving_zarr, max_workers=workers),
            ranks=source_spacings if is_native else None, average_levels=not is_native,
            level_weights=level_bytes)
    else:
        fused_image = fusion.fuse(msims, fusion_func=fuse_func, transform_key=transform_key,
                                  output_stack_properties=output_stack_properties,
                                  output_chunksize=output_chunksize)
```

---

## FAIR implementation

Rather than only documenting FAIR principles, `muvis-align` implements concrete FAIR-supporting mechanisms:

* project template defined as workflow schema in Bilayers format
* structured parameters in YAML
* OME-Zarr / OME-TIFF image outputs
* RO-Crate generation
* persistent mappings and metrics files
* napari and CLI interfaces
* container metadata for reproducible execution


## Declarative UI definition based on workflow schema

Source: [`src/muvis_align/ui/project_template.yaml`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/ui/project_template.yaml)

The project_template.yaml file is written in the Bilayers format and serves as the declarative description of the muvis-align workflow.
The file is used to dynamically generate the [magicgui](https://pyapp-kit.github.io/magicgui/)-based [napari](https://napari.org/stable/) user interface, allowing workflow parameters, input definitions, and processing options to be defined in a structured configuration file rather than being hard-coded in the application.
This approach simplifies interface maintenance, improves reproducibility, and enables the same workflow definition to be reused across the [napari](https://napari.org/stable/) plugin, command-line execution, and containerized deployments.
Bilayers schemas are represented as a [Pydantic](https://docs.pydantic.dev/) model, which adds typed validation and structured serialization for this workflow definition layer.

Source: [`src/muvis_align/ui/bilayers_util.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/ui/bilayers_util.py)

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BilayersField(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str
    type: str
    label: str | None = None
    default: Any = None
    description: str | None = None
    section_id: str | None = None
    section_key: str | None = None
    cli_order: int = 0
    options: list[BilayersOption] | None = None
    file_count: str | None = None
    output_dir_set: bool = False

    @field_validator('type', mode='before')
    @classmethod
    def _normalize_type(cls, value):
        return str(value).lower()

    @field_validator('cli_order', mode='before')
    @classmethod
    def _normalize_cli_order(cls, value):
        if value is None or value == '':
            return 0
        return int(value)


def get_section_dict(template, keys=None):
    sections = {}
    if keys is None:
        keys = [None]

    for key in keys:
        items = template.get(key, []) if key is not None else template
        for raw_item in items:
            item = BilayersField.model_validate({**raw_item, 'section_key': key})
            section_id = item.section_id or key
            sections.setdefault(section_id, []).append(item)

    return {
        section_id: [item.model_dump() for item in sorted(section_items, key=lambda item: item.cli_order)]
        for section_id, section_items in sections.items()
    }
```

The package creates a Workflow Run RO-Crate for each run, recording the project parameters, source images and results.

Source: [`src/muvis_align/file/rocrate_utils.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/file/rocrate_utils.py)

```python
def create_workflow_run_crate(dest_path, params_path, params, source_paths, result_paths,
                              start_time=None, end_time=None):
    crate = ROCrate()
    root = crate.root_dataset
    root['name'] = f'muvis-align run: {get_filetitle(os.path.normpath(dest_path))}'
    for profile_id, profile_name in WORKFLOW_RUN_PROFILES.items():
        profile = crate.add(ContextEntity(crate, profile_id, {'@type': ['CreativeWork', 'Profile'],
                                                              'name': profile_name, 'version': '0.5'}))
        root.append_to('conformsTo', profile)

    workflow_entity = add_workflow(crate)
    parameters = add_formal_parameters(crate, workflow_entity)
    ...
    action_entity = crate.add_action(workflow_entity, identifier='#run-001', object=objects, result=results,
                                     properties=action_properties)
    root.append_to('mentions', action_entity)

    crate.write(dest_path)
    return crate
```

OME-Zarr outputs receive a Zarr-specific RO-Crate, recording the sources, acquisition, registration metrics and the fusion step.

Source: [`src/muvis_align/file/rocrate_utils.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/file/rocrate_utils.py)

```python
def create_zarr_ro_crate(dest_path, sources=None, source_paths=(), metrics=None):
    crate = ZarrCrate()
    # the crate is written as the fusion ends
    fusion_time = to_iso_datetime(datetime.now())
    properties = {'name': get_filetitle(dest_path),
                  'description': f'OME-Zarr image fused by muvis-align from {len(source_paths)} sources',
                  'datePublished': fusion_time,
                  'encodingFormat': ZARR_ENCODING_FORMAT}
    dataset_entity = crate.add_dataset(dest_path='.', properties=properties)
    source_entities = [crate.add(ContextEntity(crate, to_crate_path(source_path, dest_path), {
        '@type': 'Dataset' if os.path.isdir(source_path) else 'File'})) for source_path in source_paths]
    if source_entities:
        dataset_entity['isBasedOn'] = source_entities
    if sources:
        add_data_capture(crate, dataset_entity, sources, source_entities)
    ...
    fusion_properties = {'@type': 'CreateAction', 'name': 'Fusion to OME-Zarr', 'endTime': fusion_time,
                         'instrument': add_software(crate), 'result': dataset_entity}
    if source_entities:
        fusion_properties['object'] = source_entities
    fusion_entity = crate.add(ContextEntity(crate, '#fusion-001', fusion_properties))
    dataset_entity.append_to('mentions', fusion_entity)
    crate.write(dest_path)
    return crate
```

---

## Persistent metadata

Input metadata is read from OME-TIFF or OME-Zarr and converted into physical units, positions, rotations, channels, scales, and transforms.

### Interoperable NGFF RFC 6 transforms

Registration transforms are written in the [OME-NGFF RFC 6 coordinate-transform format](https://github.com/ome/ngff-spec/blob/main/index.md#coordinatetransformations-metadata) rather than as application-specific matrix arrays. Each affine identifies its image path and names its input (`source_metadata`) and output (`registered`) coordinate systems. Using the `ome-zarr-models` v0.6 model for both serialization and validation makes the saved transforms portable to other tools that implement the NGFF coordinate-system and transformation model.

Source: [`src/muvis_align/file/transforms.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/file/transforms.py)

```python
def metadata_models(path, transform, source_key='source_metadata', transform_key='registered'):
    affine = Affine(
        path=path,
        affine=transform,
        input=CoordinateSystemIdentifier(name=source_key),
        output=CoordinateSystemIdentifier(name=transform_key),
    )
    return affine.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True)


def read_transforms(filename):
    transforms = import_json(filename)
    return {label: Affine.model_validate(transform).affine for label, transform in transforms.items()}
```

Source: [`src/muvis_align/image/ome_zarr_util.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/ome_zarr_util.py)

```python
multiscale = multiscales[0]
file_dims = tuple(axis['name'] for axis in multiscale.get('axes', []))

file_shapes, scales, dtypes = [], [], set()
translation = {}
for index, dataset in enumerate(multiscale.get('datasets', [])):
    node = nodes.get(dataset.get('path'))
    transforms = dataset.get('coordinateTransformations', [])
    scale_values = next((transform['scale'] for transform in transforms
                         if transform.get('type') == 'scale'), None)
    shape = node.get('shape')
    file_shapes.append(tuple(shape))
    scales.append(dict(zip(file_dims, scale_values)))
    if index == 0:
        translation_values = next((transform['translation'] for transform in transforms
                                   if transform.get('type') == 'translation'), None)
        if translation_values is not None:
            translation = dict(zip(file_dims, translation_values))
```

The workflow builds an `MSIM` object for each source and performs pairwise/global registration on these multiscale spatial images.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py)

```python
def build_msim(index):
    return build_source_msim(self.sources[index], self._msim_output_order, self.positions[index],
                             self._msim_transforms[index], self.source_transform_key,
                             z_scale=self._msim_z_scale, from_level=from_levels[index] if from_levels else 0,
                             ends_only=ends_only)

msims = pooled_map(build_msim, len(self.sources), default_msim_build_workers, progress_factory,
                   desc='Building sources', weight=weight,
                   timing_label='Build msims' if self.logging_time else None)
...
results = self.register_global(self.pair_msims, register_indices=register_indices, params=params)
```

The parsed OME metadata (axes, units, voxel scale, translations/positions, channels, and coordinate transforms) is preserved in this [`xarray`](https://docs.xarray.dev/)-based `MSIM` (multiscale-spatial-image) representation, where image data remains in labeled dimensions and spatial context is stored as coordinates/attributes. This metadata is propagated through image operations, updated during pairwise/global registration, and carried into fusion/export (for example, OME-Zarr) to preserve physical-space interpretation and multiscale provenance.

The CLI runs the pipeline from a YAML parameter file.

Source: [`run.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/run.py)

```python
parser = argparse.ArgumentParser(description=f'muvis-align')
parser.add_argument('params',
                    help='The parameters file',
                    default='resources/params.yml')

args = parser.parse_args()
print(f'Parameters file: {args.params}')
with open(args.params, 'r', encoding='utf8') as file:
    params = yaml.safe_load(file)

pipeline = Pipeline(params, params_path=args.params)
pipeline.run()
```

---

## Performance metrics


The project calculates registration metrics such as NCC, SSIM, normalized mutual information, MSE-derived score, match rate, and distance-based feature metrics.
### Registration quality metrics

`muvis-align` computes several complementary image similarity metrics to evaluate registration quality. Each metric captures a different aspect of alignment quality, making it useful to evaluate multiple metrics together rather than relying on a single score.

| Metric | Full name                                 | Measures                                             | Typical range |
| ------ | ----------------------------------------- | ---------------------------------------------------- | ------------- |
| NCC    | Normalized Cross-Correlation              | Intensity correlation between images                 | -1 to 1       |
| SSIM   | Structural Similarity Index Measure       | Similarity of local image structures                 | 0 to 1        |
| ONMI   | Overlapping Normalized Mutual Information | Shared information between overlapping image regions | 0 to 1+       |

The metrics are implemented in `metrics.py`:

```python
all_metric_funcs = {
    'ncc': multiview_stitcher.metrics.normalized_cross_correlation,
    'ssim': lambda im1, im2: structural_similarity(
        np.nan_to_num(im1),
        np.nan_to_num(im2),
        data_range=data_range),
    'onmi': lambda im1, im2:
        normalized_mutual_information(
            np.nan_to_num(im1),
            np.nan_to_num(im2)) - 1,
    'mse': lambda im1, im2:
        1 / mean_squared_error(im1, im2),
}
```

Source: [`src/muvis_align/metrics.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/metrics.py)

#### NCC — Normalized Cross-Correlation

NCC measures the correlation of image intensities between two images. It is particularly useful when corresponding structures have similar brightness values.

* `NCC = 1` indicates perfect correlation.
* `NCC = 0` indicates no correlation.
* Negative values indicate inverse correlation.

NCC is computationally efficient and widely used for rigid image registration, although it can be sensitive to intensity variations between acquisitions.

#### SSIM — Structural Similarity Index Measure

SSIM evaluates the similarity of local image structure by comparing:

* intensity
* contrast
* local texture

Unlike pixel-wise metrics, SSIM is sensitive to structural preservation and is therefore useful for assessing whether biological features remain aligned after registration.

* `SSIM = 1` indicates identical structure.
* Values close to zero indicate poor structural agreement.

SSIM is often more representative of perceived image quality than correlation-based metrics.

#### ONMI — Overlapping Normalized Mutual Information

ONMI measures the amount of shared information between overlapping image regions.

Mutual information is particularly valuable when images originate from different modalities, channels, or acquisition conditions because it does not assume similar intensity values.

Higher ONMI values indicate that:

* corresponding image regions contain similar information,
* overlapping structures are well aligned,
* registration improves the statistical dependence between views.

Because ONMI is based on image information content rather than intensity similarity, it is often more robust for multimodal registration problems.

Using NCC, SSIM, and ONMI together provides complementary measures of registration quality:

* **NCC** evaluates intensity agreement.
* **SSIM** evaluates structural similarity.
* **ONMI** evaluates shared image information.

This combination gives a more robust assessment of registration performance than any single metric alone.

---

## Storing data for resumable workflows

`muvis-align` stores intermediate registration state in mapping and metrics files, allowing workflows to continue from pairwise registration, global registration, or fused output.

Source: [`src/muvis_align/constants.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/constants.py)

```python
default_pair_mappings_name = 'pair_mappings.json'
default_mappings_name = 'mappings.json'
metrics_name = 'metrics.json'
default_mappings_tabular_name = 'mappings.csv'
```

The registration state machine tracks progress through the workflow.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py)

```python
class RegState(Enum):
    UNINIT = auto()
    INIT = auto()
    SIMS_INIT = auto()
    PAIRS_REG = auto()
    GLOBAL_REG = auto()
    FUSED = auto()
```

Existing output files are detected before re-running expensive steps.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py)

```python
def check_progress(self, output_filename, output_format):
    pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
    mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
    if self.output_exists(output_filename, output_format):
        self.state = RegState.FUSED
    elif os.path.exists(mappings_filename):
        self.state = RegState.GLOBAL_REG
    elif os.path.exists(pair_mappings_filename):
        self.state = RegState.PAIRS_REG
```

Pair mappings, final mappings, and metrics are saved separately.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py)

```python
def register(self, register_msims=None, register_indices=None, params=None):
    pair_results = self.register_pairs(register_msims=register_msims, register_indices=register_indices, params=params)
    self.save_pair_results(pair_results['pair_mappings'], metrics=pair_results['metrics'])
    results = self.register_global(self.pair_msims, register_indices=register_indices, params=params)
    self.save_group_pair_mappings()
    self.save_mappings(results['mappings'])
    self.save_metrics(results['metrics'])
    return results
```

### Restoring registered transforms from intermediate output

Registered alignment transforms are persisted as JSON intermediate outputs (`pair_mappings.json` and `mappings.json`) and reloaded on later runs when these files already exist.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py)

```python
def check_progress(self, output_filename, output_format):
    pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
    mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
    if self.output_exists(output_filename, output_format):
        self.state = RegState.FUSED
    elif os.path.exists(mappings_filename):
        self.state = RegState.GLOBAL_REG
    elif os.path.exists(pair_mappings_filename):
        self.state = RegState.PAIRS_REG
```

When progress indicates pairwise/global registration already exists, `init_progress` loads saved mappings and reapplies transforms to the MSIMs instead of recomputing registration.

Source: [`src/muvis_align/MVSRegistration.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py)

```python
if self.is_global_registered():
    mappings = read_transforms(mappings_filename)
    ...
    self.mappings = {}
    for index, filename in enumerate(self.filenames):
        self.mappings[index] = xaffine_from_matrix(find_file_dict_item(mappings, filename))
    if self._msims is not None:
        self._apply_mappings(self._msims)
```

JSON persistence/reload is handled by shared utilities:

Source: [`src/muvis_align/util.py`](https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/util.py)

```python
def import_json(filename):
    with open(filename, encoding='utf8') as file:
        data = json.load(file)
    return data


def export_json(filename, data):
    with open(filename, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4)
```

---

## Interactive interface with Xpra container

The Dockerfile defines a standard napari container and a second Xpra-enabled image for browser-based interactive use.

Source: [`Dockerfile`](https://github.com/ccp-volume-em/muvis-align/blob/main/Dockerfile). `docker-build-push.sh [version]`
builds the Xpra image and pushes it to `quay.io/ccp-volume-em/muvis-align-xpra`, tagged with the version (by default
the latest GitHub release) and `latest`; it needs `docker login quay.io`.

```dockerfile
FROM python:3.12-slim-bookworm AS muvis-align
...
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install napari[all]
RUN pip install napari-ome-zarr

COPY run.py .
COPY pyproject.toml .
COPY src/ src/

RUN --mount=type=bind,source=.git,target=/app/.git \
    pip install .

ENTRYPOINT ["python3", "-m", "napari", "--plugin", "muvis-align"]
```

The Xpra stage installs Xpra and launches napari through a browser-accessible session.

Source: [`Dockerfile`](https://github.com/ccp-volume-em/muvis-align/blob/main/Dockerfile)

```dockerfile
FROM muvis-align AS muvis-align-xpra
...
RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
        menu-xdg \
        xdg-utils \
        xterm \
        sshfs \
        x11-xkb-utils \
        xkb-data && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_START="python3 -c \"import napari; viewer = napari.Viewer(); viewer.window.add_plugin_dock_widget('muvis-align'); viewer.window._qt_window.showMaximized(); napari.run()\""
ENV XPRA_EXIT_WITH_CHILDREN="yes"
ENV XPRA_EXIT_WITH_CLIENT="no"
ENV XPRA_XVFB_SCREEN="8192x4096x24+32"
EXPOSE 9876
```

```dockerfile
CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT or $(hostname -i):$XPRA_PORT"; \
    PATH=/usr/bin:/usr/local/bin:$PATH xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    ...
    --resize-display=yes \
    --start-child="$XPRA_START" \
    --exit-with-children="$XPRA_EXIT_WITH_CHILDREN" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite +extension RANDR -screen 0 $XPRA_XVFB_SCREEN -dpi 96 -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY
```

### On a Slurm cluster

On the cluster, Apptainer runs the Xpra image from a sandbox that `xpra-pull.sh` builds, and `xpra-run.sh`
starts it in a Slurm job. Both run from a folder holding the scripts, which also holds the container.

#### From your own machine

`xpra-connect.sh` does all of this from your own machine with one command:

```bash
bash xpra-connect.sh <user>@<login node> [folder on the cluster, default: muvis-align] [--pull]
```

On Windows, run it in Git Bash, or from PowerShell as `.\xpra-connect.sh <user>@<login node>`, which opens it in a
Git Bash window that waits for Enter before closing. `bash` in PowerShell is WSL's; there it uses Windows' own ssh
when WSL's cannot sign in.

It copies `xpra-run.sh` and `xpra-pull.sh` into the folder on the cluster (replacing what is there), builds the
container first if there is none yet (or with `--pull`), submits the job, tunnels to the compute node the job
starts on, and opens napari in your browser, already logged in. Closing napari ends the job; Ctrl-C closes the
connection and offers to cancel the job. `MUVIS_PARTITION`, `MUVIS_DATA_DIR` and `MUVIS_CONTAINER_DIR` set locally
are passed on to the cluster. With an
[SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
on the login node you are not asked for a password.

#### Step by step, on the cluster

With the two scripts in a folder on the cluster (e.g. a clone of this repository), on the login node:

```bash
bash xpra-pull.sh     # once, and to update: builds the container sandbox beside the scripts
bash xpra-run.sh      # submits the job, waits for it to start and prints how to connect
```

Then follow the printed steps on your own machine: open the `ssh -N -L ...` tunnel it shows, open
`http://localhost:9876/` in your browser and log in with the user name and password it shows. `sbatch xpra-run.sh`
works too; the steps are then in `muvis-align-<job id>.log` and the password in `~/.xpra/job-<job id>/passwd`.
The partition and data directory can be set with the same `MUVIS_*` variables; the job's resources are the
`#SBATCH` lines at the top of `xpra-run.sh`. Its memory (`--mem`) is passed into the container, where fusion sizes
its blocks against it.

`sbatch apptainer-check.sh` checks whether the compute nodes can run the image's SIF file directly, or unpack it on
every start, in which case the sandbox `xpra-pull.sh` builds is the faster choice; its verdict is at the end of
`muvis-align-apptainer-check-<job id>.log`.

---

## Summary table

| Practice                | Source-code implementation                                                  |
| ----------------------- | --------------------------------------------------------------------------- |
| NGFF                    | OME-Zarr default extension, NGFF v0.5, `to_ngff_zarr`                       |
| Larger-than-memory data | Dask-backed TIFF and Zarr sources, chunked fusion, direct Zarr output       |
| FAIR implementation     | package metadata, YAML configuration, RO-Crate generation, OME-Zarr outputs |
| Persistent metadata     | OME metadata parsing, `xarray`/`MSIM` spatial transforms, positions, scales, channels; interoperable NGFF RFC 6 affine transforms |
| Performance metrics     | NCC, SSIM, ONMI, MSE-derived score, match metrics, timing logs              |
| Resumable workflow      | state machine, pair mappings, global mappings, metrics JSON                 |
| Xpra interface          | Docker stage exposing browser-based napari over Xpra                        |

