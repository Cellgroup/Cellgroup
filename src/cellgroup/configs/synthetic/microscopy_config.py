from pathlib import Path
from typing import Callable, Literal, Sequence, Union
from warnings import warn

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MicroscopyConfig(BaseModel):
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    save_dir: Union[str, Path]
    """Path to the directory where to save the simulated images."""
    
    n_simulations: int = 100
    """The number of images to simulate."""
    
    batch_size: int = 1
    """The number of images to simulate at once."""
    
    fluorophores: Sequence[str]
    """The fluorophores associated with the structures to simulate."""
    
    space_shape: tuple[int, int, int] = (256, 256, 256)
    """The shape of the simulation space."""
    
    space_scale: tuple[float, float, float] = (0.1, 0.1, 0.1)
    """The scale (i.e., voxel size) of the simulation space (in μm)."""
    
    space_downscaling: Union[int, tuple[int, int, int]] = 1
    """The downscaling factor to apply to the simulation space."""
    
    light_wavelengths: Sequence[int]
    """List of lasers to use for excitation."""
    
    light_powers: Sequence[float]
    """List of powers associate to each light source (work as scaling factors)."""
    
    wavelength_range: tuple[int, int] = (400, 700)
    """The range of wavelengths of the acquired spectrum in nm."""
    
    exposure_ms: float = 50
    """The exposure time for the detector cameras (in ms)."""
    
    detector_quantum_eff: float = Field(0.8, ge=0, le=1)
    """The quantum efficiency of the detector cameras."""
    
    detector_bandpass_bandwidth: float = 5
    """The bandwidth of the bandpass filter in nm used in the spectral detector."""
    
    read_noise: float = 6
    """The read noise of the detector cameras in electrons."""
    
    bit_depth: Literal[8, 16, 32] = 16
    """The bit depth of the acquired images."""
    
    # TODO: set excitation lights to excitation peaks of the fluorophores (complicated ...)
    
    @model_validator(mode="after")
    def _validate_labels_fluorophores(self):
        if len(self.fluorophores) != len(self.labels):
            raise ValueError("The number of labels and fluorophores must be the same.")
        return self
    
    @model_validator(mode="after")
    def _validate_light_sources(self):
        if len(self.light_wavelengths) != len(self.light_powers):
            raise ValueError("The number of light sources and light powers must be the same.")
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