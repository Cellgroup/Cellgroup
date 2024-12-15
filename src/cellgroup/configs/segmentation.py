from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cellgroup.configs.models import StardistConfig

SegModelConfig = Union[StardistConfig]
"""Configuration for segmentation models."""


class SegmentationConfig(BaseModel):
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    modality: Literal["training", "inference"]
    """The current modality, either training or inference."""
    
    model: SegModelConfig