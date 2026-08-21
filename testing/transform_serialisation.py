# https://github.com/bogovicj/ngff-rfc5-coordinate-transformation-examples/
# https://ome-zarr-models-py.readthedocs.io/en/stable/how-to/
# https://ome-zarr-models-py.readthedocs.io/en/stable/api/dev/image/#coordinate-transformation-metadata
# https://github.com/jo-mueller/ome-zarr-models-py/blob/main/tests/v06/test_transforms.py
# https://github.com/ome-zarr-models/ome-zarr-models-py/blob/main/src/ome_zarr_models/v06/scene.py
# https://ngff-zarr.readthedocs.io/en/latest/rfc5.html


def metadata_models():
    from ome_zarr_models.v06.coordinate_transforms import (
        Axis,
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Affine,
        Scale,
        Transform,
    )
    from ome_zarr_models.v06.multiscales import Dataset

    ct = Affine(
        affine=(
            (1.0, 0.0, 0.0, 5.0),
            (0.0, 1.0, 0.0, 10.0),
            (0.0, 0.0, 1.0, 15.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        path='a-b',
        input=CoordinateSystemIdentifier(name="source_metadata"),
        output=CoordinateSystemIdentifier(name="registered"),
    )
    #return ct.model_dump_json(exclude_none=True, exclude_defaults=True, exclude_unset=True)
    return ct.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True)


def metadata_models_list():
    from ome_zarr_models.v06.coordinate_transforms import (
        Axis,
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Affine,
        Scale,
        Transform,
    )
    from ngff_zarr.v06.zarr_metadata import TransformSequence

    input_cs = CoordinateSystemIdentifier(name="source_metadata")
    output_cs = CoordinateSystemIdentifier(name="registered")

    ct = Affine(
        affine=(
            (1.0, 0.0, 0.0, 5.0),
            (0.0, 1.0, 0.0, 10.0),
            (0.0, 0.0, 1.0, 15.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        path='a-b',
        input=input_cs,
        output=output_cs,
    )
    transforms = TransformSequence(transformations=[ct], input=input_cs, output=output_cs)
    dct = transforms.to_dict()
    transforms1 = TransformSequence.from_dict(dct)
    assert transforms1 == transforms
    return dct


def metadata_models_dict():
    dct = {'a-b': metadata_models()}
    return dct


def metadata_nz():
    from ngff_zarr.v06.zarr_metadata import Metadata, Dataset, CoordinateSystem, CoordinateSystemIdentifier, Axis, Affine, Scale

    metadata = Metadata(
        datasets=[Dataset(path="example",
                          coordinateTransformations=[Scale(scale=[1.0, 1.0, 1.0],
                                                           input=CoordinateSystemIdentifier(name='physical'),
                                                           output=CoordinateSystemIdentifier(name='source_metadata'),
                                                           )])
                  ],
        coordinateSystems=[
            CoordinateSystem(
                name="source_metadata",
                axes=[
                    Axis(name="z", type="space"),
                    Axis(name="y", type="space"),
                    Axis(name="x", type="space"),
                ]
            ),
            CoordinateSystem(
                name="registered",
                axes=[
                    Axis(name="z", type="space"),
                    Axis(name="y", type="space"),
                    Axis(name="x", type="space"),
                ]
            )
        ],
        coordinateTransformations=[
            Affine(
                affine=[
                    [1.0, 0.0, 0.0, 5.0],
                    [0.0, 1.0, 0.0, 10.0],
                    [0.0, 0.0, 1.0, 15.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                input=CoordinateSystemIdentifier(name='source_metadata'),
                output=CoordinateSystemIdentifier(name='registered'),
            )
        ],
    )
    return metadata


def metadata_serialisation():
    print(metadata_models_dict())


if __name__ == "__main__":
    metadata_serialisation()
