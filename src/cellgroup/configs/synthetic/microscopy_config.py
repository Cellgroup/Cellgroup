from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MicroscopyConfig(BaseModel):
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    save_dir: Union[str, Path]
    """Path to the directory where to save the simulated images."""
    
    random_seed: int = 123
    """The random seed to use for the simulation."""
    
    space_shape: tuple[int, int, int] = (256, 256, 256)
    """The shape of the simulation space."""
    
    space_scale: tuple[float, float, float] = (0.1, 0.1, 0.1)
    """The scale (i.e., voxel size) of the simulation space (in μm)."""
    
    space_downscaling: Union[int, tuple[int, int, int]] = 1
    """The downscaling factor to apply to the simulation space."""
    
    fluorophores: Sequence[str]
    """The fluorophores associated with the structures to simulate."""
    
    laser_wavelengths: Optional[Sequence[int]] = None
    """List of lasers used for excitation. If not provided, the excitation peaks of the
    fluorophores will be used to place the laser sources."""
    
    laser_powers: Sequence[float]
    """List of powers associate to each light source (work as scaling factors)."""
    
    laser_filters_bandwidth: int = 5
    """The bandwidth of the bandpass filter (in nm) used for the excitation lasers.
    Used to stop the excitation light from being acquired."""
    
    emission_filters_bandwidth: Union[int, Sequence[int]] = 50
    """The bandwidth of filters used at the emission stage (i.e., wavelength ranges for
    the acquisition of each multiplexed image). If a single value is provided, it is
    used for all fluorophores, otherwise, a list of values for each fluorophore must
    be provided."""
    
    pinhole_au: float = 1.0
    """The pinhole size in Airy units."""
    
    exposure_ms: float = 50
    """The exposure time for the detector cameras (in ms)."""
    
    detector_quantum_eff: float = Field(0.8, ge=0, le=1)
    """The quantum efficiency of the detector cameras."""
    
    read_noise: float = 6
    """The read noise of the detector cameras in electrons."""
    
    bit_depth: Literal[8, 16, 32] = 16
    """The bit depth of the acquired images."""
    
    # TODO: set excitation lights to excitation peaks of the fluorophores if not provided

    
    @field_validator("emission_filters_bandwidth")
    def _validate_emission_filters(
        cls, v: Union[int, Sequence[int]], values: dict[str, Any]
    ) -> Sequence[int]:
        if isinstance(v, int):
            return [v] * len(values["fluorophores"])
        else:
            assert len(v) == len(values["fluorophores"]), (
                "The number of emission filters must be the same as the number of fluorophores."
            )
            return v
    
    @model_validator(mode="after")
    def _validate_lasers(self):
        if self.laser_wavelengths is not None:
            if len(self.laser_wavelengths) != len(self.laser_powers):
                raise ValueError("The number of light sources and light powers must be the same.")
        return self
    
    @model_validator(mode="after")
    def _validate_fluorophores_and_lasers(self):
        if len(self.fluorophores) != len(self.laser_powers):
            raise ValueError("The number of labels and fluorophores must be the same.")
        return self