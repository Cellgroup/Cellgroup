from typing import Any
from typing_extensions import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class SimulationConfig(BaseModel):
    """Configuration for simulation parameters."""

    # Time-related parameters
    duration: int = Field(gt=0, description="Total simulation duration in timesteps")
    time_step: float = Field(gt=0.0, le=1.0, description="Time step size")
    save_frequency: int = Field(gt=0, description="Save state every N steps")

    # Space configuration
    space_size: tuple[int, ...] = Field(description="Size of simulation space")
    space_scale: tuple[float, ...] = Field(description="Scale of simulation space")

    # Population parameters
    initial_clusters: int = Field(ge=0, description="Number of initial clusters")
    nuclei_per_cluster: int = Field(ge=0, description="Initial nuclei per cluster")
    min_cluster_separation: float = Field(gt=0.0, description="Minimum separation between clusters")

    # Physical parameters
    noise_strength: float = Field(ge=0.0, description="Strength of random motion")
    repulsion_strength: float = Field(ge=0.0, description="Repulsion strength between nuclei")
    adhesion_strength: float = Field(ge=0.0, description="Adhesion strength between nuclei")

    # Biological parameters
    growth_rate: float = Field(gt=0.0, description="Base growth rate")
    division_threshold: float = Field(gt=0.0, description="Size threshold for division")
    death_probability: float = Field(ge=0.0, le=1.0, description="Base death probability")

    # Performance parameters
    max_snapshots: int = Field(default=1000, gt=0, description="Maximum number of snapshots to store")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")

    @field_validator("space_size", "space_scale")
    def validate_dimensions(cls, v: Any) -> Any:
        if len(v) not in (2, 3):
            raise ValueError("Space dimensions must be 2D or 3D")
        return v

    @model_validator(mode="after")
    def validate_scale_dimensions(self: Self) -> Self:
        if len(self.space_scale) != len(self.space_size):
            raise ValueError("space_size and space_scale must have same dimensions")
        return self

    @model_validator(mode="after")
    def validate_config_consistency(self: Self) -> Self:
        """Validate consistency between different config parameters."""
        if self.time_step * self.duration > 1e6:
            raise ValueError("Total simulation steps exceed maximum limit")

        if self.initial_clusters * self.nuclei_per_cluster > 10000:
            raise ValueError("Total initial nuclei exceed maximum limit")

        return self