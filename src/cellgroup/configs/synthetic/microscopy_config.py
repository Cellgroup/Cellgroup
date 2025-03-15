from pathlib import Path
from typing import Callable, Literal, Sequence, Union
from warnings import warn

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MicroscopyConfig(BaseModel):
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    save_dir: Union[str, Path]
    """Path to the directory where to save the simulated images."""
    
    space_shape: tuple[int, int, int] = (256, 256, 256)
    """The shape of the simulation space."""
    
    space_scale: tuple[float, float, float] = (0.1, 0.1, 0.1)
    """The scale (i.e., voxel size) of the simulation space (in μm)."""
    
    space_downscaling: Union[int, tuple[int, int, int]] = 1
    """The downscaling factor to apply to the simulation space."""
    
    fluorophores: Sequence[str]
    """The fluorophores associated with the structures to simulate."""
    
    laser_wavelengths: Sequence[int]
    """List of lasers to use for excitation."""
    
    laser_powers: Sequence[float]
    """List of powers associate to each light source (work as scaling factors)."""
    
    laser_filters_bandwidth: int = 5
    """The bandwidth of the bandpass filter (in nm) used for the excitation lasers."""
    
    excitation_filters_bandwidth: int = 50
    """The bandwidth of filters used at the excitation stage (i.e., wavelength ranges for
    the excitation of each fluorophore)."""
    # TODO: is it needed given that the excitation source is a laser?
    
    emission_filters_bandwidth: int = 50
    """The bandwidth of filters used at the emission stage (i.e., wavelength ranges for
    the acquisition of each multiplexed image)."""
    
    exposure_ms: float = 50
    """The exposure time for the detector cameras (in ms)."""
    
    detector_quantum_eff: float = Field(0.8, ge=0, le=1)
    """The quantum efficiency of the detector cameras."""
    
    read_noise: float = 6
    """The read noise of the detector cameras in electrons."""
    
    bit_depth: Literal[8, 16, 32] = 16
    """The bit depth of the acquired images."""
    
    # TODO: set excitation lights to excitation peaks of the fluorophores if not provided

    
    @model_validator(mode="after")
    def _validate_lasers(self):
        if len(self.laser_wavelengths) != len(self.laser_powers):
            raise ValueError("The number of light sources and light powers must be the same.")
        return self
    
    @model_validator(mode="after")
    def _validate_fluorophores_and_lasers(self):
        if len(self.fluorophores) != len(self.laser_wavelengths):
            raise ValueError("The number of labels and fluorophores must be the same.")
        return self
    
    @model_validator(mode="after")
    def _check_num_sim_multiple_batch_size(self):
        if self.n_simulations % self.batch_size != 0:
            nearest_multiple = self.n_simulations // self.batch_size * self.batch_size
            msg = (
                "The number of simulations is not a multiple of the batch size."
                " Setting `num_simulations` to the closest multiple of `batch_size`."
                f" i.e., {nearest_multiple}."
            )
            warn(msg, UserWarning, stacklevel=2)
            self.n_simulations = nearest_multiple
        return self