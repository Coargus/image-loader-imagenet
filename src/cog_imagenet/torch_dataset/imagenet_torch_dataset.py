from __future__ import annotations

import os

import cv2
import numpy as np
from torch.utils.data import Dataset

# from .meta_to_imagenet import META_TO_IMAGENET, filter  # noqa: ERA001

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
        batch_id: int | str = 1,
        target_size: tuple = (224, 224, 3),
        is_mapping: bool = False,
    ):
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

        for root in self.class_mapping_dict.keys():
            files = os.listdir(self.image_path + root)
            self.length_dataset += len(files)
            self._num_images_per_class[root] = len(files)

        self.target_size = target_size

        print(
            f"loaded imagenet dataset ({type}) with {self.length_dataset} images and {len(self.class_mapping_dict.keys())} classes: "
        )

        self.is_mapping = is_mapping

        if self.is_mapping:
            # Mapped dataset with respect to COCO metaclasses
            self.map_data()
            print(
                f"After mapping imagenet dataset({type}), we have {self.length_dataset} images and {len(self.meta_class_to_imagenet_class.keys())} classes: "
            )

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

            return image, class_id
        else:
            # Obtain the metaclass where the index is located
            metaclass_id = 0
            metaclassname = list(self.meta_class_to_imagenet_class.keys())[
                metaclass_id
            ]
            cum_count = 0
            while index >= self._num_images_per_metaclass[metaclassname]:
                index -= self._num_images_per_metaclass[metaclassname]
                cum_count += self._num_images_per_metaclass[metaclassname]
                metaclass_id += 1
                metaclassname = list(self.meta_class_to_imagenet_class.keys())[
                    metaclass_id
                ]

            imagenet_class_for_metaclass = self.meta_class_to_imagenet_class[
                metaclassname
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

            return image, metaclass_id

    def __len__(self):
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
        """Returns the filtered META_TO_IMAGENET dictionary"""
        classes = set()
        for j, line in enumerate(open(imagenet_path)):
            cs = line[9:].strip().split(", ")[0]
            cs = cs.replace(" ", "_")
            classes.add(cs)
        filtered_meta_to_imagenet = dict()
        for k in meta_to_imagenet:
            filtered_meta_to_imagenet[k] = []
            for v in meta_to_imagenet[k]:
                if v in classes:
                    filtered_meta_to_imagenet[k].append(v)

        return filtered_meta_to_imagenet

    def map_data(self):
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
