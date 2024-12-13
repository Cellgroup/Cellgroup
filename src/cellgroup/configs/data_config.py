"""Here we will put config for datasets using Pydantic."""
from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cellgroup.utils import Sample, Channel


class DataConfig(BaseModel):
    """Config for datasets."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    samples: list[Sample]
    """List of samples to load from the dataset."""
    
    channels: list[Channel]
    """List of channels to load from the dataset."""
    
    time_steps: Optional[tuple[int, int, int]] = None
    """Tuple of (start, end, step) for the time steps to load."""
    
    img_dim: Literal["2D", "3D"] = "2D"
    """Dimensionality of the images."""
    
    patch_size: tuple[int, ...]
    """Size of the patches to extract."""
    
    patch_overlap: Optional[tuple[int, ...]] = None
    """Overlap of the patches. If None, patching is done sequentially on a grid."""
    
    batch_size: int = 1
    """Batch size for the dataloader."""
    
    dloader_kwargs: Optional[dict[str, Any]] = None
    """Additional kwargs for the dataloader."""
    
    @field_validator("time_steps")
    @classmethod
    def validate_time_steps(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Validate the time steps."""
        assert len(v) >= 2, "You need to provide at least (start, end)."
        assert len(v) <= 3, "You can provide at most (start, end, step)."
        return v
    
    @field_validator("dloader_kwargs")
    @classmethod
    def validate_dloader_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate the dataloader kwargs."""
        if v is None:
            return {}
        return v
    
    @model_validator(mode="after")
    def validate_patch_size(self: Self) -> Self:
        """Validate the patch size."""
        if self.img_dim == "2D":
            assert len(self.patch_size) == 2, "You need to provide (Y, X) as patch size."
        elif self.img_dim == "3D":
            assert len(self.patch_size) == 3, "You need to provide (Z, Y, X) as patch size"
        return self
    
    # TODO: add validation for patch_overlap and patch_size
    
    