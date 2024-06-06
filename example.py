"""Example run script."""

from cog_cv_imagenet import CogImageNetDataloader
from cog_cv_imagenet.dataset.cog_dataset import CoargusImageNetDataset

if __name__ == "__main__":
    image_dir = "/store/datasets/ILSVRC"
    loader = CogImageNetDataloader(
        imagenet_dir_path=image_dir, mapping_to="coco"
    )
    dataset: CoargusImageNetDataset = loader.dataset

    data = dataset.get_all_images_by_label("person")
