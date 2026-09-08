from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.data.dataset import ISeg25DSliceDataset
from src.data.splits import TRAIN_SUBJECT_IDS, VALIDATION_SUBJECT_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_subject(
    data_directory: Path,
    subject_id: int,
    shape: tuple,
) -> None:
    voxel_count = int(np.prod(shape))
    t1 = np.arange(voxel_count, dtype=np.int16).reshape(shape)
    t2 = np.arange(voxel_count, 0, -1, dtype=np.int16).reshape(shape)
    raw_classes = np.array([0, 10, 150, 250], dtype=np.int16)
    labels = np.resize(raw_classes, shape)

    for suffix, volume in (("T1", t1), ("T2", t2), ("label", labels)):
        header_path = data_directory / f"subject-{subject_id}-{suffix}.hdr"
        nib.AnalyzeImage(volume, affine=np.eye(4)).to_filename(str(header_path))


def test_patient_splits_are_fixed_and_disjoint() -> None:
    assert TRAIN_SUBJECT_IDS == (1, 2, 3, 4, 5, 6, 7, 8)
    assert VALIDATION_SUBJECT_IDS == (9, 10)
    assert set(TRAIN_SUBJECT_IDS).isdisjoint(VALIDATION_SUBJECT_IDS)


def test_returns_normalized_25d_tensor_and_central_label(tmp_path: Path) -> None:
    shape = (4, 5, 3)
    _write_subject(tmp_path, subject_id=1, shape=shape)

    dataset = ISeg25DSliceDataset(tmp_path, subject_ids=(1,))
    image, label = dataset[1]

    assert len(dataset) == shape[-1]
    assert image.shape == (6, shape[0], shape[1])
    assert label.shape == shape[:2]
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    assert set(torch.unique(label).tolist()).issubset({0, 1, 2, 3})
    assert torch.isfinite(image).all()


def test_orders_subjects_and_slices_deterministically(tmp_path: Path) -> None:
    first_shape = (2, 3, 2)
    second_shape = (2, 3, 3)
    _write_subject(tmp_path, subject_id=1, shape=first_shape)
    _write_subject(tmp_path, subject_id=2, shape=second_shape)

    dataset = ISeg25DSliceDataset(tmp_path, subject_ids=(2, 1))

    assert dataset.subject_ids == (1, 2)
    assert dataset.sample_index == (
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    )
    assert len(dataset) == first_shape[-1] + second_shape[-1]


def test_repeats_edge_slices_at_subject_boundaries(tmp_path: Path) -> None:
    _write_subject(tmp_path, subject_id=1, shape=(2, 3, 3))
    dataset = ISeg25DSliceDataset(tmp_path, subject_ids=(1,))

    first_image, _ = dataset[0]
    last_image, _ = dataset[-1]

    torch.testing.assert_close(first_image[0], first_image[1])
    torch.testing.assert_close(first_image[3], first_image[4])
    torch.testing.assert_close(last_image[1], last_image[2])
    torch.testing.assert_close(last_image[4], last_image[5])


def test_rejects_invalid_subject_identifiers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        ISeg25DSliceDataset(tmp_path, subject_ids=())
    with pytest.raises(ValueError, match="positive integers"):
        ISeg25DSliceDataset(tmp_path, subject_ids=(0,))
    with pytest.raises(ValueError, match="unique"):
        ISeg25DSliceDataset(tmp_path, subject_ids=(1, 1))


def test_returns_real_subject_1_slice_with_expected_shape_and_types() -> None:
    data_directory = PROJECT_ROOT / "data/training"
    required_paths = tuple(
        data_directory / f"subject-1-{suffix}.hdr" for suffix in ("T1", "T2", "label")
    )
    if not all(path.is_file() for path in required_paths):
        pytest.skip("Local training dataset not available")

    dataset = ISeg25DSliceDataset(data_directory, subject_ids=(1,))
    image, label = dataset[0]

    assert image.shape == (6, 144, 192)
    assert label.shape == (144, 192)
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    assert set(torch.unique(label).tolist()).issubset({0, 1, 2, 3})
