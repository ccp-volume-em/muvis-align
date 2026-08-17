from ome_zarr_models.v06.coordinate_transforms import CoordinateSystemIdentifier, Affine

from muvis_align.util import export_json, import_json


def metadata_models(path, transform, source_key='source_metadata', transform_key='registered'):
    affine = Affine(
        path=path,
        affine=transform,
        input=CoordinateSystemIdentifier(name=source_key),
        output=CoordinateSystemIdentifier(name=transform_key),
    )
    return affine.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True)


def write_transforms(filename, transforms_dict, source_key='source_metadata', transform_key='registered'):
    transforms = {label: metadata_models(label, transform, source_key=source_key, transform_key=transform_key)
                  for label, transform in transforms_dict.items()}
    export_json(filename, transforms)


def read_transforms(filename):
    transforms = import_json(filename)
    return {label: Affine.model_validate(transform).affine for label, transform in transforms.items()}
