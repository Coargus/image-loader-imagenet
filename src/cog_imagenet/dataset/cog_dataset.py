"""Coargus's Detected Object Schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cog_cv_abstraction.schema.dataset import CoargusImageDataset

if TYPE_CHECKING:
    import numpy as np


class CoargusImageNetDataset(CoargusImageDataset):
    """ImageNet Dataset."""

    unique_labels: list
    labels: list[list[str]]
    images: list[np.ndarray]  # Dataset

    def get_all_images_by_label(self, target_label: str) -> list[np.ndarray]:
        """Returns all images with the specified label from a dataset.

        Args:
            target_label (str): Label to filter images by.

        Returns:
        list[np.ndarray]: Images matching the target label.
        """
        return self.images.get_all_images_by_label(target_label)
