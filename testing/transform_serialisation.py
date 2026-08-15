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
        Identity,
        Scale,
        Transform,
    )
    from ome_zarr_models.v06.multiscales import Dataset

    ct = Identity(
        input=CoordinateSystemIdentifier(name="a"),
        output=CoordinateSystemIdentifier(name="b"),
    )
    return ct


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
    return metadata_nz()


if __name__ == "__main__":
    metadata_serialisation()
