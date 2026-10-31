import numpy as np
from ome_zarr_models.v06.coordinate_transforms import CoordinateSystemIdentifier, Affine

from muvis_align.util import export_json, import_json


def metadata_models(path, transform, source_key='source_metadata', transform_key='registered'):
    # OME-Zarr 0.6 affines are (N, N+1): the homogeneous last row is left out
    affine = Affine(
        path=path,
        affine=np.asarray(transform)[:-1].tolist(),
        input=CoordinateSystemIdentifier(name=source_key),
        output=CoordinateSystemIdentifier(name=transform_key),
    )
    return affine.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True)


def write_transforms(filename, transforms_dict, source_key='source_metadata', transform_key='registered'):
    transforms = {label: metadata_models(label, transform, source_key=source_key, transform_key=transform_key)
                  for label, transform in transforms_dict.items()}
    export_json(filename, transforms)


def read_transforms(filename):
    """Each label's affine as a square homogeneous matrix; files written before as square are read as they are."""
    transforms = import_json(filename)
    return {label: homogeneous_affine(Affine.model_validate(transform).affine) for label, transform in transforms.items()}


def homogeneous_affine(affine):
    affine = np.asarray(affine, dtype=float)
    if affine.shape[0] == affine.shape[1]:
        return affine
    return np.vstack([affine, np.eye(1, affine.shape[1], affine.shape[1] - 1)])
