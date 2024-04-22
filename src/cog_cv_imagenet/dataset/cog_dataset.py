"""Coargus's Detected Object Schema."""

from __future__ import annotations

import secrets
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

    def get_random_sample(
        self, sample_size: int
    ) -> dict[list[str], list[np.ndarray]]:
        """Returns a random sample of images from the dataset.

        Args:
            sample_size (int): Number of images to sample.

        Returns:
            list[np.ndarray]: Random sample of images.
        """
        images = []
        labels = []
        for _ in range(sample_size):
            random_index = secrets.randbelow(len(self.images))
            image, class_name = self.images[random_index]
            images.append(image)
            labels.append(class_name)

        return {"images": images, "labels": labels}
