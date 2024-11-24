"""Here we will put config for datasets using Pydantic."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cellgroup.data.utils import SampleID, ChannelID


class DatasetConfig(BaseModel):
    """Config for datasets."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    samples: list[SampleID]
    """List of samples to load from the dataset."""
    
    channels: list[ChannelID]
    """List of channels to load from the dataset."""
    
    time_steps: tuple[int, int, int]
    """Tuple of (start, end, step) for the time steps to load."""
    
    img_dim: Literal["2D", "3D"]
    """Dimensionality of the images."""
    
    patch_size: tuple[int, int, int]
    """Size of the patches to extract."""
    
    @field_validator("time_steps")
    @classmethod
    def validate_time_steps(cls, v):
        """Validate the time steps."""
        assert len(v) >= 2, "You need to provide at least (start, end)."
        assert len(v) <= 3, "You can provide at most (start, end, step)."
        return v
    
    