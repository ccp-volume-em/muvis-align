# https://ngff-zarr.readthedocs.io/en/latest/rfc5.html
# https://github.com/bogovicj/ngff-rfc5-coordinate-transformation-examples/


from ngff_zarr.v06.zarr_metadata import Metadata, Dataset, CoordinateSystem,CoordinateSystemIdentifier, Axis, Affine, Scale


def metadata_serialisation():
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
    pass


if __name__ == "__main__":
    metadata_serialisation()
