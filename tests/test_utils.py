import os

import numpy as np
import pytest

from muvis_align.util import calculate_rigid_difference, create_transform, \
    resolve_to_project_dir, relativize_to_project_dir


@pytest.mark.parametrize(
    'transform1, transform2, expected',
    [
        (
            np.eye(3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
        ),
        (
            create_transform((0, 0), 0, translation=[-1, -2], matrix_size=3),
            np.eye(3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
        ),
        (
            np.eye(3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
        ),
        (
            create_transform((0, 0), -10, translation=[0, 0], matrix_size=3),
            np.eye(3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
        ),
        (
            create_transform((0, 0), 10, translation=[1, 2], matrix_size=3),
            create_transform((0, 0), 35, translation=[4, 6], matrix_size=3),
            create_transform((0, 0), 25, translation=[2.25983055, 4.46017555], matrix_size=3),
        ),
        (
            create_transform((0, 0), 15, translation=[1, 2, 3], matrix_size=4),
            create_transform((0, 0), 40, translation=[5, 7, 11], matrix_size=4),
            create_transform((0, 0), 25, translation=[2.56960808, 5.86490531, 8], matrix_size=4),
        ),
    ],
)
def test_calculate_rigid_difference(transform1, transform2, expected):
    """Calculate rigid differences for 2D and 3D affine transforms."""
    np.testing.assert_allclose(calculate_rigid_difference(transform1, transform2), expected)


def test_resolve_to_project_dir_joins_relative_path():
    base_dir = os.path.abspath('project')
    resolved = resolve_to_project_dir('data/input', base_dir)
    assert resolved == os.path.normpath(os.path.join(base_dir, 'data/input')).replace('\\', '/')


def test_resolve_to_project_dir_leaves_absolute_path_unchanged():
    absolute = os.path.abspath('somewhere/else')
    assert resolve_to_project_dir(absolute, os.path.abspath('project')) == absolute.replace('\\', '/')


def test_resolve_to_project_dir_handles_multiple_comma_separated_paths():
    base_dir = os.path.abspath('project')
    absolute = os.path.abspath('somewhere/else')
    resolved = resolve_to_project_dir(f'data/a, {absolute}', base_dir)
    expected_joined = os.path.normpath(os.path.join(base_dir, "data/a")).replace('\\', '/')
    assert resolved == f'{expected_joined}, {absolute.replace(chr(92), "/")}'


def test_resolve_to_project_dir_always_returns_forward_slashes():
    """Even on Windows, os.path.join()/normpath() naturally produce backslashes - the result
    must be normalised to forward slashes so the same value is safe to show in the UI and to
    store in the (OS-portable) project file."""
    base_dir = os.path.abspath('project')
    resolved = resolve_to_project_dir('data/input', base_dir)
    assert '\\' not in resolved


@pytest.mark.parametrize('path, base_dir', [('', os.path.abspath('project')), ('data/input', None)])
def test_resolve_to_project_dir_no_op_without_path_or_base_dir(path, base_dir):
    assert resolve_to_project_dir(path, base_dir) == path


def test_relativize_to_project_dir_converts_absolute_path_under_base_dir():
    base_dir = os.path.abspath('project')
    absolute = os.path.join(base_dir, 'data', 'input')
    assert relativize_to_project_dir(absolute, base_dir) == 'data/input'


def test_relativize_to_project_dir_leaves_relative_path_unchanged():
    assert relativize_to_project_dir('data/input', os.path.abspath('project')) == 'data/input'


def test_relativize_to_project_dir_round_trips_with_resolve_to_project_dir():
    """The pair together must be idempotent: display-resolving a stored relative path and then
    relativizing the (now absolute) value the widget reports back must reproduce the original -
    otherwise every project load would silently rewrite the project file's paths to absolute."""
    base_dir = os.path.abspath('project')
    original = 'data/input'

    resolved = resolve_to_project_dir(original, base_dir)
    round_tripped = relativize_to_project_dir(resolved, base_dir)

    assert round_tripped == original


def test_relativize_to_project_dir_falls_back_to_absolute_on_different_drive(monkeypatch):
    def raise_value_error(path, start):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, 'relpath', raise_value_error)

    absolute = os.path.abspath('somewhere/else')
    assert relativize_to_project_dir(absolute, os.path.abspath('project')) == absolute.replace('\\', '/')
