"""PyTorch dataset providing preprocessed iSeg 2.5D slices."""

from pathlib import Path
from typing import Dict, NamedTuple, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .analyze import load_analyze_volume
from .labels import remap_labels
from .normalization import normalize_nonzero_intensity
from .slices import build_25d_input, extract_central_label_slice
from .validation import validate_matching_shapes


class _SubjectVolumes(NamedTuple):
    t1: np.ndarray
    t2: np.ndarray
    labels: np.ndarray


class ISeg25DSliceDataset(Dataset):
    """Return one normalized 2.5D image and its central label per slice.

    Every requested subject is loaded and preprocessed once when the dataset is
    created. Samples are then ordered by subject identifier and slice index.

    Parameters
    ----------
    data_directory:
        Directory containing the training Analyze pairs.
    subject_ids:
        Non-empty sequence of unique, positive subject identifiers.
    """

    def __init__(
        self,
        data_directory: Union[str, Path],
        subject_ids: Sequence[int],
    ) -> None:
        self.data_directory = Path(data_directory)
        if not self.data_directory.is_dir():
            raise FileNotFoundError(
                f"Training data directory not found: {self.data_directory}"
            )

        self.subject_ids = self._validate_subject_ids(subject_ids)
        self._volumes: Dict[int, _SubjectVolumes] = {}
        sample_index = []

        for subject_id in self.subject_ids:
            volumes = self._load_subject(subject_id)
            self._volumes[subject_id] = volumes
            sample_index.extend(
                (subject_id, slice_index) for slice_index in range(volumes.t1.shape[-1])
            )

        self.sample_index: Tuple[Tuple[int, int], ...] = tuple(sample_index)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subject_id, slice_index = self.sample_index[index]
        volumes = self._volumes[subject_id]

        image_array = np.ascontiguousarray(
            build_25d_input(volumes.t1, volumes.t2, slice_index)
        )
        label_array = np.ascontiguousarray(
            extract_central_label_slice(volumes.labels, slice_index)
        )

        image = torch.from_numpy(image_array).to(dtype=torch.float32)
        label = torch.from_numpy(label_array).to(dtype=torch.long)
        return image, label

    def _load_subject(self, subject_id: int) -> _SubjectVolumes:
        prefix = self.data_directory / f"subject-{subject_id}"
        t1 = load_analyze_volume(Path(f"{prefix}-T1.hdr"))
        t2 = load_analyze_volume(Path(f"{prefix}-T2.hdr"))
        labels = load_analyze_volume(Path(f"{prefix}-label.hdr"))

        validate_matching_shapes(t1, t2, labels)
        return _SubjectVolumes(
            t1=normalize_nonzero_intensity(t1),
            t2=normalize_nonzero_intensity(t2),
            labels=remap_labels(labels),
        )

    @staticmethod
    def _validate_subject_ids(subject_ids: Sequence[int]) -> Tuple[int, ...]:
        identifiers = tuple(subject_ids)
        if not identifiers:
            raise ValueError("At least one subject identifier is required")
        if any(
            isinstance(subject_id, bool)
            or not isinstance(subject_id, (int, np.integer))
            or subject_id <= 0
            for subject_id in identifiers
        ):
            raise ValueError("Subject identifiers must be positive integers")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Subject identifiers must be unique")
        return tuple(sorted(int(subject_id) for subject_id in identifiers))
