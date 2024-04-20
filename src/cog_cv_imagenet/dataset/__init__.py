"""Package containing Coargus's ImageNet."""

from .cog_dataset import CoargusImageNetDataset
from .torch_dataset import ImageNetTorchDataset

__all__ = ["ImageNetTorchDataset", "CoargusImageNetDataset"]
