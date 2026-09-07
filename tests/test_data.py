from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from src.data import (
    RAW_TO_INTERNAL_LABELS,
    build_25d_input,
    extract_central_label_slice,
    load_analyze_volume,
    normalize_nonzero_intensity,
    remap_labels,
    remove_trailing_singleton_dimension,
    validate_matching_shapes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_synthetic_analyze_volume(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    header_path = tmp_path / "sample.hdr"
    nib.AnalyzeImage(values, affine=np.eye(4)).to_filename(str(header_path))

    loaded = load_analyze_volume(header_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)


def test_removes_only_trailing_singleton_dimension() -> None:
    values = np.arange(6, dtype=np.int16).reshape(1, 2, 3, 1)

    loaded = remove_trailing_singleton_dimension(values)

    assert loaded.shape == (1, 2, 3)
    np.testing.assert_array_equal(loaded, values[..., 0])


def test_leaves_volume_without_trailing_singleton_unchanged() -> None:
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    loaded = remove_trailing_singleton_dimension(values)

    assert loaded is values
    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)


def test_accepts_matching_training_shapes() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 3, 4))
    labels = np.zeros((2, 3, 4), dtype=np.int16)

    validate_matching_shapes(t1, t2, labels)


def test_accepts_matching_test_modalities_without_labels() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 3, 4))

    validate_matching_shapes(t1, t2)


def test_rejects_mismatched_modalities() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 5, 4))

    with pytest.raises(ValueError, match=r"T1=\(2, 3, 4\).*T2=\(2, 5, 4\)"):
        validate_matching_shapes(t1, t2)


def test_rejects_mismatched_labels() -> None:
    t1 = np.zeros((2, 3, 4))
    t2 = np.ones((2, 3, 4))
    labels = np.zeros((2, 3, 5), dtype=np.int16)

    with pytest.raises(ValueError, match=r"labels=\(2, 3, 5\)"):
        validate_matching_shapes(t1, t2, labels)


def test_remaps_raw_labels_to_internal_class_indices() -> None:
    labels = np.array([[0, 10], [150, 250]], dtype=np.uint8)
    original = labels.copy()

    remapped = remap_labels(labels)

    np.testing.assert_array_equal(remapped, np.array([[0, 1], [2, 3]], dtype=np.uint8))
    assert remapped.shape == labels.shape
    assert remapped.dtype == np.uint8
    np.testing.assert_array_equal(labels, original)


def test_rejects_unknown_label_values() -> None:
    labels = np.array([[0, 42]], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"Unknown label values: \[42\]"):
        remap_labels(labels)


def test_normalizes_nonzero_voxels_and_preserves_zeros() -> None:
    volume = np.array([[0, 1, 2, 3, 0]], dtype=np.int16)
    original = volume.copy()

    normalized = normalize_nonzero_intensity(volume)

    expected = np.array([[0, -1.2247449, 0, 1.2247449, 0]], dtype=np.float32)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
    assert normalized.shape == volume.shape
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(volume, original)


def test_rejects_all_zero_volume() -> None:
    volume = np.zeros((2, 3), dtype=np.int16)

    with pytest.raises(ValueError, match="without non-zero voxels"):
        normalize_nonzero_intensity(volume)


def test_rejects_constant_nonzero_volume() -> None:
    volume = np.array([[0, 5, 5, 0]], dtype=np.int16)

    with pytest.raises(ValueError, match="zero non-zero variance"):
        normalize_nonzero_intensity(volume)


def test_builds_25d_input_with_expected_channel_order() -> None:
    t1 = np.zeros((2, 3, 4), dtype=np.float32)
    t2 = np.zeros((2, 3, 4), dtype=np.float32)
    for z in range(4):
        t1[..., z] = 10 + z
        t2[..., z] = 20 + z
    t1_original = t1.copy()
    t2_original = t2.copy()

    result = build_25d_input(t1, t2, z=1)

    expected_values = [10, 11, 12, 20, 21, 22]
    assert result.shape == (6, 2, 3)
    assert result.dtype == np.float32
    for channel, value in enumerate(expected_values):
        np.testing.assert_array_equal(result[channel], value)
    np.testing.assert_array_equal(t1, t1_original)
    np.testing.assert_array_equal(t2, t2_original)


def test_repeats_closest_slice_at_volume_boundaries() -> None:
    t1 = np.stack([np.full((2, 3), value, dtype=np.float32) for value in range(4)], axis=-1)
    t2 = t1 + 10

    first = build_25d_input(t1, t2, z=0)
    last = build_25d_input(t1, t2, z=3)

    np.testing.assert_array_equal(first[:, 0, 0], [0, 0, 1, 10, 10, 11])
    np.testing.assert_array_equal(last[:, 0, 0], [2, 3, 3, 12, 13, 13])


def test_extracts_central_label_slice() -> None:
    labels = np.stack([np.full((2, 3), value, dtype=np.uint8) for value in range(4)], axis=-1)

    target = extract_central_label_slice(labels, z=2)

    assert target.shape == (2, 3)
    np.testing.assert_array_equal(target, 2)


def test_rejects_invalid_25d_inputs() -> None:
    t1 = np.zeros((2, 3, 4), dtype=np.float32)
    t2 = np.zeros((2, 3, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Volume shapes do not match"):
        build_25d_input(t1, t2, z=1)
    with pytest.raises(IndexError, match="out of range"):
        build_25d_input(t1, t1, z=4)
    with pytest.raises(TypeError, match="must be an integer"):
        build_25d_input(t1, t1, z=1.5)
    with pytest.raises(ValueError, match="3-dimensional"):
        extract_central_label_slice(np.zeros((2, 3), dtype=np.uint8), z=0)


def test_rejects_missing_header(tmp_path: Path) -> None:
    header_path = tmp_path / "missing.hdr"

    with pytest.raises(FileNotFoundError, match="header file not found"):
        load_analyze_volume(header_path)


def test_rejects_missing_image(tmp_path: Path) -> None:
    header_path = tmp_path / "sample.hdr"
    header_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="image file not found"):
        load_analyze_volume(header_path)


def test_rejects_non_header_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.img"

    with pytest.raises(ValueError, match="Expected an Analyze .hdr path"):
        load_analyze_volume(image_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("data/training/subject-1-T1.hdr"),
        Path("data/testing/subject-23-T1.hdr"),
        Path("data/testing/subject-23-T2.hdr"),
    ],
)
def test_loads_real_analyze_volumes(relative_path: Path) -> None:
    header_path = PROJECT_ROOT / relative_path
    if not header_path.is_file():
        pytest.skip(f"Local dataset not available: {header_path}")

    loaded = load_analyze_volume(header_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim >= 2
    assert loaded.size > 0
    assert np.issubdtype(loaded.dtype, np.number)

    reference = np.asanyarray(nib.load(str(header_path)).dataobj)
    expected = remove_trailing_singleton_dimension(reference)
    assert loaded.shape == expected.shape
    np.testing.assert_array_equal(loaded, expected)


def test_validates_training_subject_1() -> None:
    relative_paths = (
        Path("data/training/subject-1-T1.hdr"),
        Path("data/training/subject-1-T2.hdr"),
        Path("data/training/subject-1-label.hdr"),
    )
    header_paths = [PROJECT_ROOT / relative_path for relative_path in relative_paths]
    if not all(path.is_file() for path in header_paths):
        pytest.skip("Local training dataset not available")

    t1, t2, labels = [load_analyze_volume(path) for path in header_paths]

    validate_matching_shapes(t1, t2, labels)


def test_validates_testing_subject_23_without_labels() -> None:
    relative_paths = (
        Path("data/testing/subject-23-T1.hdr"),
        Path("data/testing/subject-23-T2.hdr"),
    )
    header_paths = [PROJECT_ROOT / relative_path for relative_path in relative_paths]
    if not all(path.is_file() for path in header_paths):
        pytest.skip("Local testing dataset not available")

    t1, t2 = [load_analyze_volume(path) for path in header_paths]

    validate_matching_shapes(t1, t2)


def test_remaps_real_training_labels_subject_1() -> None:
    header_path = PROJECT_ROOT / "data/training/subject-1-label.hdr"
    if not header_path.is_file():
        pytest.skip("Local training dataset not available")

    labels = load_analyze_volume(header_path)
    remapped = remap_labels(labels)

    assert sorted(np.unique(labels).tolist()) == sorted(RAW_TO_INTERNAL_LABELS)
    assert sorted(np.unique(remapped).tolist()) == [0, 1, 2, 3]
    assert remapped.shape == labels.shape


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("data/training/subject-1-T1.hdr"),
        Path("data/training/subject-1-T2.hdr"),
    ],
)
def test_normalizes_real_training_modalities(relative_path: Path) -> None:
    header_path = PROJECT_ROOT / relative_path
    if not header_path.is_file():
        pytest.skip("Local training dataset not available")

    volume = load_analyze_volume(header_path)
    normalized = normalize_nonzero_intensity(volume)
    nonzero_values = normalized[volume != 0]

    assert normalized.shape == volume.shape
    assert normalized.dtype == np.float32
    assert np.all(normalized[volume == 0] == 0)
    assert np.mean(nonzero_values) == pytest.approx(0, abs=1e-5)
    assert np.std(nonzero_values) == pytest.approx(1, abs=1e-5)


def test_builds_real_25d_subject_1_example() -> None:
    relative_paths = (
        Path("data/training/subject-1-T1.hdr"),
        Path("data/training/subject-1-T2.hdr"),
        Path("data/training/subject-1-label.hdr"),
    )
    header_paths = [PROJECT_ROOT / relative_path for relative_path in relative_paths]
    if not all(path.is_file() for path in header_paths):
        pytest.skip("Local training dataset not available")

    t1, t2, labels = [load_analyze_volume(path) for path in header_paths]
    z = t1.shape[-1] // 2

    model_input = build_25d_input(t1, t2, z)
    target = extract_central_label_slice(labels, z)

    assert model_input.shape == (6, t1.shape[0], t1.shape[1])
    assert target.shape == (labels.shape[0], labels.shape[1])


def test_builds_real_25d_subject_23_example_without_labels() -> None:
    relative_paths = (
        Path("data/testing/subject-23-T1.hdr"),
        Path("data/testing/subject-23-T2.hdr"),
    )
    header_paths = [PROJECT_ROOT / relative_path for relative_path in relative_paths]
    if not all(path.is_file() for path in header_paths):
        pytest.skip("Local testing dataset not available")

    t1, t2 = [load_analyze_volume(path) for path in header_paths]
    model_input = build_25d_input(t1, t2, z=0)

    assert model_input.shape == (6, t1.shape[0], t1.shape[1])


def test_inspection_script_saves_training_figure(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts/inspect_data.py"
    output_path = tmp_path / "subject-1.png"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--split",
            "training",
            "--subject",
            "1",
            "--slice",
            "0",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg"},
    )

    assert output_path.is_file()
    assert "input_2.5d=(6, 144, 192)" in result.stdout


def test_inspection_script_saves_testing_figure_without_labels(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts/inspect_data.py"
    output_path = tmp_path / "subject-23.png"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--split",
            "testing",
            "--subject",
            "23",
            "--slice",
            "0",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg"},
    )

    assert output_path.is_file()
    assert "input_2.5d=(6, 160, 192)" in result.stdout
