from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from cog_cv_imagenet.label_mapper.mapper_utils import get_mapper_metadata

# Define the global variable for the metadata
MAPPER_METADATA = None


class ImageNetTorchDataset(Dataset):
    """ImageNet Dataset."""

    mapping_number_to_class: dict
    class_mapping_dict: dict
    class_mapping_dict_number: dict
    mapping_class_to_number: dict
    class_name_to_class_id: dict

    meta_class_to_imagenet_class: dict
    imagenet_class_to_metaclass: dict
    valid_classes: list

    # TODO: Use the filtered meta to imagenet to rd
    def __init__(
        self,
        imagenet_dir_path: str,
        type: str = "train",
        target_size: tuple = (224, 224, 3),
        is_mapping: bool = False,
        mapping_to: str | None = None,  # "coco"
    ) -> None:
        super().__init__()
        """Load ImageNet dataset from file."""
        self.imagenet_path = imagenet_dir_path
        mapping_path = imagenet_dir_path + "/LOC_synset_mapping.txt"

        self.class_mapping_dict = {}
        self.class_mapping_dict_number = {}
        self.mapping_class_to_number = {}
        self.mapping_number_to_class = {}
        self.class_name_to_class_id = {}
        i = 0

        for line in open(mapping_path):
            class_name = line[9:].strip().split(", ")[0]
            class_name = class_name.replace(" ", "_")
            self.class_mapping_dict[line[:9].strip()] = class_name
            self.class_mapping_dict_number[i] = class_name

            if class_name in self.class_name_to_class_id:
                self.class_name_to_class_id[class_name].append(line[:9].strip())
            else:
                self.class_name_to_class_id[class_name] = [line[:9].strip()]

            self.mapping_class_to_number[line[:9].strip()] = i
            self.mapping_number_to_class[i] = line[:9].strip()
            i += 1

        self.length_dataset = 0
        self.image_path = imagenet_dir_path + "/Data/CLS-LOC/" + type + "/"
        self._num_images_per_class = {}

        for root in self.class_mapping_dict:
            files = os.listdir(self.image_path + root)
            self.length_dataset += len(files)
            self._num_images_per_class[root] = len(files)

        self.target_size = target_size

        logging.info(
            f"loaded imagenet dataset ({type}) with {self.length_dataset} images and {len(self.class_mapping_dict.keys())} classes: "
        )

        if mapping_to is not None:
            global MAPPER_METADATA  # noqa: PLW0603
            MAPPER_METADATA = get_mapper_metadata(
                loader_name="imagenet",
                mapping_to=mapping_to,
            )
            self.is_mapping = True
        else:
            self.is_mapping = False

        if self.is_mapping:
            # Mapped dataset with respect to COCO metaclasses
            self.map_data()
            logging.info(
                f"After mapping imagenet dataset({type}), we have {self.length_dataset} images and {len(self.meta_class_to_imagenet_class.keys())} classes: "
            )

    def get_all_images_by_label(self, target_label: str) -> list[np.ndarray]:
        """Returns all images with the specified label from a dataset.

        Args:
            target_label (str): Label to filter images by.

        Returns:
        list[np.ndarray]: Images matching the target label.
        """
        images = []
        if self.is_mapping:
            imagenet_class_for_metaclass = self.meta_class_to_imagenet_class[
                target_label
            ]
            class_ids = []
            for map_key in imagenet_class_for_metaclass:
                class_ids.append(self.class_name_to_class_id[map_key])
            for class_id in class_ids:
                class_id = class_id[0]
                images = self.get_images_by_class_id(class_id)
            return images

        class_id = self.class_name_to_class_id.get(target_label)
        # Collect images where target_label is in their corresponding label list
        if class_id:
            class_id = class_id[0]
            return self.get_images_by_class_id(class_id)

        msg = f"Label {target_label} not found in dataset"
        raise ValueError(msg)

    def get_images_by_class_id(self, class_id: str) -> list[np.ndarray]:
        """Get images by class ID.

        Args:
            class_id (str): Class ID to filter images by.

        Returns:
        list[np.ndarray]: Images matching the class ID.
        """
        images = []
        img_ids = os.listdir(self.image_path + class_id)
        for img_id in img_ids:
            # Load the image
            image = plt.imread(self.image_path + class_id + "/" + img_id)
            if len(image.shape) == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            # Resize
            image = cv2.resize(image, self.target_size[:2])
            images.append(image)
        return images

    def __getitem__(self, index):
        """Get item from dataset."""
        index_copy = index
        if not self.is_mapping:
            # Find the class ID where the index is located
            class_id = 0
            while (
                index
                >= self._num_images_per_class[
                    self.mapping_number_to_class[class_id]
                ]
            ):
                index -= self._num_images_per_class[
                    self.mapping_number_to_class[class_id]
                ]
                class_id += 1
            # Find the image ID within the class
            class_name = self.mapping_number_to_class[class_id]
            image_id = os.listdir(self.image_path + class_name)[index]
            # Load the image
            image = plt.imread(self.image_path + class_name + "/" + image_id)
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            # Resize
            image = cv2.resize(image, self.target_size[:2])

            return image, class_name
        else:
            # Obtain the metaclass where the index is located
            metaclass_id = 0
            meta_class_name = list(self.meta_class_to_imagenet_class.keys())[
                metaclass_id
            ]
            cum_count = 0
            while index >= self._num_images_per_metaclass[meta_class_name]:
                index -= self._num_images_per_metaclass[meta_class_name]
                cum_count += self._num_images_per_metaclass[meta_class_name]
                metaclass_id += 1
                meta_class_name = list(
                    self.meta_class_to_imagenet_class.keys()
                )[metaclass_id]

            imagenet_class_for_metaclass = self.meta_class_to_imagenet_class[
                meta_class_name
            ]
            index = index_copy - cum_count

            class_id_val = 0
            class_id = self.class_name_to_class_id[
                imagenet_class_for_metaclass[class_id_val]
            ][0]
            while index >= self._num_images_per_class[class_id]:
                index -= self._num_images_per_class[class_id]
                class_id_val += 1
                class_id = self.class_name_to_class_id[
                    imagenet_class_for_metaclass[class_id_val]
                ][0]

            image_id = os.listdir(self.image_path + class_id)[index]

            image = plt.imread(self.image_path + class_id + "/" + image_id)
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            # Resize
            image = cv2.resize(image, self.target_size[:2])

            return image, meta_class_name

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.length_dataset

    def class_to_class_number(self, id):
        """Get the class number from the class ID."""
        return self.mapping_class_to_number[id]

    def class_number_to_class(self, id):
        """Get the class ID from the class number."""
        return self.mapping_number_to_class[id]

    def class_number_to_class_name(self, id):
        """Get the class name from the class ID."""
        return self.class_mapping_dict_number[id]

    def class_to_class_name(self, id):
        """Get the class name from the class ID."""
        return self.class_mapping_dict[id]

    def filter(self, meta_to_imagenet: dict, imagenet_path: str) -> dict:
        """Returns the filtered META_TO_IMAGENET dictionary."""
        classes = set()
        for _j, line in enumerate(open(imagenet_path)):
            cs = line[9:].strip().split(", ")[0]
            cs = cs.replace(" ", "_")
            classes.add(cs)
        filtered_meta_to_imagenet = {}
        for k in meta_to_imagenet:
            filtered_meta_to_imagenet[k] = []
            for v in meta_to_imagenet[k]:
                if v in classes:
                    filtered_meta_to_imagenet[k].append(v)

        return filtered_meta_to_imagenet

    def map_data(self) -> None:
        """Map data meta data on to the imagenet classes."""
        filtered_meta_data = self.filter(
            MAPPER_METADATA, self.imagenet_path + "/LOC_synset_mapping.txt"
        )

        self.meta_class_to_imagenet_class = filtered_meta_data
        self.imagenet_class_to_metaclass = {}
        self.length_dataset = 0
        self._num_images_per_metaclass = {}
        for key, val in self.meta_class_to_imagenet_class.items():
            num_images = 0
            for v in val:
                v_id = self.class_name_to_class_id[v][0]
                self.imagenet_class_to_metaclass[v] = key
                self.length_dataset += self._num_images_per_class[v_id]
                num_images += self._num_images_per_class[v_id]

            self._num_images_per_metaclass[key] = num_images
        # From the mapped data evalaute the length of the dataset

    @property
    def class_names(self) -> list[str]:
        if self.is_mapping:
            keys = []
            for k, v in self._num_images_per_metaclass.items():
                if v != 0:
                    keys.append(k)
            return keys
        else:
            return list(self.class_mapping_dict.keys())

    @property
    def class_counts(self) -> dict:
        if self.is_mapping:
            return self._num_images_per_metaclass
        elif self._num_images_per_class:
            # remap keys to class_names
            return {
                self.class_mapping_dict[k]: v
                for k, v in self._num_images_per_class.items()
            }
        return None
