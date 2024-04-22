"""Getting mapper metadata."""

from __future__ import annotations


def get_mapper_metadata(loader_name: str, mapping_to: str) -> dict:
    """Get the metadata for the specified mapper."""
    loader_name = loader_name.split(".py")[0]
    if loader_name == "imagenet":
        valid_mapper = ["coco"]
        assert (  # noqa: S101
            mapping_to in valid_mapper
        ), "please use valid mapper for ImageNet: coco"
    if mapping_to == "coco":
        from cog_cv_imagenet.label_mapper.metadata.imagenet_to_coco import (
            MAPPER_METADATA,
        )
    return MAPPER_METADATA
