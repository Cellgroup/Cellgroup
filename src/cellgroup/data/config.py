"""Here we will put config for datasets using Pydantic."""
from typing import Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cellgroup.data.utils import SampleID, ChannelID


class DatasetConfig(BaseModel):
    """Config for datasets."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    samples: list[SampleID]
    """List of samples to load from the dataset."""
    
    channels: list[ChannelID]
    """List of channels to load from the dataset."""
    
    t_steps_slice: Optional[tuple[int, int, int]] = None
    """Tuple of (start, end, step) for the time steps to load."""
    
    img_dim: Literal["2D", "3D"] = "2D"
    """Dimensionality of the images."""
    
    patch_size: tuple[int, int, int]
    """Size of the patches to extract."""
    
    @field_validator("time_steps")
    @classmethod
    def validate_time_steps(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate the time steps."""
        assert len(v) >= 2, "You need to provide at least (start, end)."
        assert len(v) <= 3, "You can provide at most (start, end, step)."
        return v
    
    @model_validator(mode="after")
    def validate_patch_size(self: Self) -> Self:
        """Validate the patch size."""
        if self.img_dim == "2D":
            assert len(self.patch_size) == 2, "You need to provide (Y, X) as patch size."
        elif self.img_dim == "3D":
            assert len(self.patch_size) == 3, "You need to provide (Z, Y, X) as patch size"
        return self
    
    