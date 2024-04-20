"""A data loader for the ImageNet dataset."""

from __future__ import annotations

import os

from .dataset.cog_dataset import CoargusImageNetDataset

# from .meta_to_imagenet import META_TO_IMAGENET, filter  # noqa: ERA001
from .dataset.torch_dataset import ImageNetTorchDataset
from .label_mapper.mapper_utils import get_mapper_metadata

# Define the global variable for the metadata
MAPPER_METADATA = None


class CogImageNetDataloader:
    """A data loader for the ImageNet dataset.

    Attributes:
        imagenet_dir_path (str): Directory path where the ImageNet data is stored.
        mapping_to (str, optional): Indicates the dataset to map to (default: None).
        version (str, optional): Version tag for the dataset (default: '').
        batch_id (int | str, optional): Identifier for the data batch (default: 1).
    """  # noqa: E501

    def __init__(
        self,
        imagenet_dir_path: str,
        mapping_to: str | None = None,  # coco
        version: str = "",
    ) -> None:
        """Initializes the data loader with the provided parameters.

        Sets up the dataset, potentially mapping it to another format if specified.
        """  # noqa: E501
        # Create an imagenet dataset
        self.name = "ImageNet" + version
        if mapping_to is not None:
            global MAPPER_METADATA  # noqa: PLW0603
            MAPPER_METADATA = get_mapper_metadata(
                loader_name=os.path.basename(__file__),  # noqa: PTH119
                mapping_to=mapping_to,
            )
            is_mapping = True
        else:
            is_mapping = False

        self.imagenet = ImageNetTorchDataset(
            imagenet_dir_path, is_mapping=is_mapping, mapping_to=mapping_to
        )
        self.dataset = self.process_data(raw_data=self.load_data())

    def load_data(self) -> dict:
        """Loads and organizes the labels for the ImageNet data.

        Returns:
            dict: A dictionary containing the dataset and its corresponding labels.
        """  # noqa: E501
        labels = [0 for _ in range(len(self.imagenet))]
        mapped_labels = [0 for _ in range(len(self.imagenet))]
        cum_count = 0
        for idx, (class_, count) in enumerate(
            self.imagenet.class_counts.items()
        ):
            cum_count += count
            for j in range(cum_count - count, cum_count):
                labels[j] = idx
                mapped_labels[j] = [class_]

        return {"dataset": self.imagenet, "labels": mapped_labels}

    def process_data(self, raw_data: dict) -> any:
        """Processes the raw data into a structured format.

        Returns:
            any: The processed dataset ready for benchmarking.
        """
        return CoargusImageNetDataset(
            unique_labels=list(self.imagenet.class_names),
            labels=raw_data["labels"],
            images=raw_data["dataset"],
        )
