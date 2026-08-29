import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from muvis_align.MVSRegistration import MVSRegistration, RegState


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (RegState.UNINIT, (False, False, False, False)),
        (RegState.INIT, (True, False, False, False)),
        (RegState.PAIRS_REG, (True, True, False, False)),
        (RegState.GLOBAL_REG, (True, True, True, False)),
        (RegState.FUSED, (True, True, True, True)),
    ],
    ids=lambda value: value.name if isinstance(value, RegState) else None,
)
def test_registration_state_predicates(state, expected):
    registration = MVSRegistration()
    registration.state = state

    actual = (
        registration.is_initialised(),
        registration.is_pairs_registered(),
        registration.is_global_registered(),
        registration.is_fused(),
    )

    assert actual == expected


def test_reset_clears_registration_state():
    registration = MVSRegistration()
    registration.state = RegState.FUSED
    registration.sims = [object()]
    registration.metrics = {"quality": 1}

    registration.reset()

    assert registration.state is RegState.UNINIT
    assert registration.sims == []
    assert registration.register_sims == []
    assert registration.sources == []
    assert registration.metrics == {}
    assert registration.register_indices is None


def test_init_with_explicit_files_sets_labels_and_output(tmp_path):
    inputs = [str(tmp_path / "tile_01.tif"), str(tmp_path / "tile_02.tif")]

    registration = MVSRegistration()
    result = registration.init(
        operation="register",
        label="sample",
        input_path=inputs,
        input_labels=["left", "right"],
        output_path="results/",
    )

    assert result is True
    assert registration.state is RegState.INIT
    assert registration.filenames == [Path(path).as_posix() for path in inputs]
    assert registration.file_labels == ["left", "right"]
    assert registration.output == str(tmp_path / "results") + "/"


def test_init_returns_false_when_pattern_has_no_files(tmp_path):
    registration = MVSRegistration()

    with patch("muvis_align.MVSRegistration.dir_regex", return_value=[]):
        result = registration.init(
            input_path=str(tmp_path / "*.tif"),
            output_path="results/",
        )

    assert result is False


def test_init_params_normalises_sections_and_forwards_options():
    registration = MVSRegistration()
    params = {
        "operation": "register",
        "input": "images/*.tif",
        "output": "results/",
        "preprocessing": {"scale": 2},
        "registration": {"method": "phase"},
        "fusion": {"method": "max"},
    }
    general = {
        "overwrite": True,
        "clear": True,
        "ui": "napari",
        "verbose": True,
        "debug": True,
    }

    with patch.object(registration, "init", return_value=True) as init:
        result = registration.init_params(general, params, label="sample")

    assert result is True
    assert registration.input_params == {"path": "images/*.tif"}
    assert registration.output_params == {"path": "results/"}
    assert registration.preprocess_params == {"scale": 2}
    assert init.call_args.kwargs["overwrite"] is True
    assert init.call_args.kwargs["label"] == "sample"


@pytest.mark.parametrize(
    ("kwargs", "expected_modified"),
    [
        ({}, False),
        ({"scale": 1}, False),
        ({"normalisation": "none"}, False),
        ({"normalisation": False}, False),
        ({"gaussian_sigma": 1}, True),
    ],
)
def test_preprocess_sets_modified_flag_for_enabled_steps(kwargs, expected_modified):
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    sims = [np.ones((4, 4), dtype=np.uint16)]

    with patch(
        "muvis_align.MVSRegistration.gaussian_filter_sim",
        side_effect=lambda sim, *_: sim,
    ):
        _, _, modified = registration.preprocess(sims, **kwargs)

    assert modified is expected_modified


def test_preprocess_applies_scale_via_init_data():
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    sims = [np.ones((4, 4), dtype=np.uint16)]

    with patch.object(registration, "init_data", return_value=sims) as init_data:
        _, _, modified = registration.preprocess(sims, scale=2)

    assert modified is True
    init_data.assert_called_once_with(target_scale=2, store=False)


def test_validate_overlap_reports_near_images():
    registration = MVSRegistration()
    sims = [object(), object()]
    positions = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.0}]

    with (
        patch(
            "muvis_align.MVSRegistration.get_sim_position_final",
            side_effect=positions,
        ),
        patch(
            "muvis_align.MVSRegistration.get_sim_physical_size",
            return_value={"x": 1.0, "y": 1.0},
        ),
    ):
        distances, overlaps = registration.validate_overlap(
            sims, ["left", "right"]
        )

    assert len(distances) == 2
    assert overlaps == [True, True]


def test_get_metrics_supports_summary_tuple_and_numpy_pair():
    registration = MVSRegistration()
    registration.metrics = {
        "summary": {"source": {"quality": 0.5}},
        "pairs": {(0, 1): {"registered": {"quality": 0.8}}},
    }

    assert registration.get_metrics("quality") == 0.5
    assert registration.get_metrics(
        "quality", np.array([0, 1])
    ) == pytest.approx(0.8)
    assert registration.get_metrics(pair=(3, 4)) == {}


def test_output_exists_supports_zarr_and_regular_files(tmp_path):
    registration = MVSRegistration()
    registration.output = str(tmp_path) + "/"
    zarr_output = tmp_path / "fused.zarr"
    zarr_output.mkdir()
    (zarr_output / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preview.tif").write_bytes(b"image")

    assert registration.output_exists("fused", ".zarr")
    assert registration.output_exists("preview", "tif")
    assert not registration.output_exists("missing", ".zarr")


@pytest.mark.parametrize(
    ("output_exists", "existing_path", "expected_state"),
    [
        (True, None, RegState.FUSED),
        (False, "mappings.json", RegState.GLOBAL_REG),
        (False, "pairs.json", RegState.PAIRS_REG),
    ],
)
def test_check_progress_uses_most_advanced_available_state(
    output_exists, existing_path, expected_state
):
    registration = MVSRegistration()
    registration.output = "output/"
    registration.output_params = {
        "pair_mappings": "pairs.json",
        "mappings": "mappings.json",
    }

    def path_exists(path):
        return existing_path is not None and path.endswith(existing_path)

    with (
        patch.object(
            registration, "output_exists", return_value=output_exists
        ),
        patch(
            "muvis_align.MVSRegistration.os.path.exists",
            side_effect=path_exists,
        ),
    ):
        registration.check_progress("fused", ".zarr")

    assert registration.state is expected_state
