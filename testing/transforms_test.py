import numpy as np

from muvis_align.image.util import get_properties_from_transform
from muvis_align.util import create_transform, get_translation_from_transform, get_rotation_from_transform, \
    get_scale_from_transform


def transforms_test1():
    a1 = create_transform((1, 1), 20)
    a2 = create_transform((1, 1), -10)
    print_transform(a1)
    print_transform(a2)

    dif = np.dot(np.transpose(a2), a1)
    print_transform(dif)


def transforms_test2():
    a1 = np.array([
[0.99999882, -0.00153727, -1.02925182],
[0.00153727,  0.99999882, -0.15658382],
[0,  0,  1]
    ])
    a2 = np.array([
[1.,          0.,         -1.21226118],
[0.,          1.,          1.5950805 ],
[0, 0, 1]
    ])
    print_transform(a1)
    print_transform(a2)

    dif = np.dot(a1, a2)
    print_transform(dif)


def print_transform(matrix):
    print(matrix)
    translation = get_translation_from_transform(matrix)
    rotation = get_rotation_from_transform(matrix)
    scale = get_scale_from_transform(matrix)
    print('t', translation, 'r', rotation, 's', scale)
    print()


if __name__ == "__main__":
    transforms_test2()
