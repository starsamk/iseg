#!/usr/bin/env python3
"""Inspect one preprocessed iSeg-2017 subject and display a slice."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data import (  # noqa: E402
    build_25d_input,
    extract_central_label_slice,
    load_analyze_volume,
    normalize_nonzero_intensity,
    remap_labels,
    validate_matching_shapes,
)


def load_subject_volumes(
    repository_root: Path, split: str, subject_id: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load and prepare T1, T2 and optional labels for one subject."""

    data_directory = repository_root / "data" / split
    subject_prefix = f"subject-{subject_id}"
    t1 = load_analyze_volume(data_directory / f"{subject_prefix}-T1.hdr")
    t2 = load_analyze_volume(data_directory / f"{subject_prefix}-T2.hdr")
    labels_path = data_directory / f"{subject_prefix}-label.hdr"

    labels = None
    if labels_path.is_file():
        labels = remap_labels(load_analyze_volume(labels_path))

    validate_matching_shapes(t1, t2, labels)
    return (
        normalize_nonzero_intensity(t1),
        normalize_nonzero_intensity(t2),
        labels,
    )


def create_subject_figure(
    t1: np.ndarray,
    t2: np.ndarray,
    labels: Optional[np.ndarray],
    slice_index: int,
):
    """Create a figure containing T1, T2 and an optional annotation slice."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1[..., slice_index], cmap="gray")
    axes[0].set_title("T1 normalisé")
    axes[1].imshow(t2[..., slice_index], cmap="gray")
    axes[1].set_title("T2 normalisé")

    if labels is None:
        axes[2].text(0.5, 0.5, "Annotation indisponible", ha="center", va="center")
        axes[2].set_title("Annotation")
    else:
        target = extract_central_label_slice(labels, slice_index)
        axes[2].imshow(target, cmap="viridis", vmin=0, vmax=3)
        axes[2].set_title("Annotation interne")

    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("training", "testing"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--slice", dest="slice_index", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    t1, t2, labels = load_subject_volumes(
        REPOSITORY_ROOT, arguments.split, arguments.subject
    )
    depth = t1.shape[-1]
    slice_index = depth // 2 if arguments.slice_index is None else arguments.slice_index
    if slice_index < 0 or slice_index >= depth:
        raise IndexError(f"Slice index out of range: {slice_index} for depth {depth}")

    model_input = build_25d_input(t1, t2, slice_index)
    figure = create_subject_figure(t1, t2, labels, slice_index)
    print(
        f"subject-{arguments.subject} | split={arguments.split} | "
        f"slice={slice_index}/{depth - 1} | input_2.5d={model_input.shape}"
    )

    if arguments.output is None:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.output, dpi=150)
        print(f"figure={arguments.output}")


if __name__ == "__main__":
    main()
