from ome_zarr_models.v06.coordinate_transforms import CoordinateSystemIdentifier, Affine

from muvis_align.util import export_json, import_json


def metadata_models(label, transform):
    affine = Affine(
        path=label,
        affine=transform,
        input=CoordinateSystemIdentifier(name='source_metadata'),
        output=CoordinateSystemIdentifier(name='registered'),
    )
    return affine.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True)


def write_transforms(filename, transforms_dict):
    transforms = {label: metadata_models(label, transform)
                  for label, transform in transforms_dict.items()}
    export_json(filename, transforms)


def read_transforms(filename):
    transforms = import_json(filename)
    return {label: Affine.model_validate(transform).affine for label, transform in transforms.items()}
