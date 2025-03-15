import os
from functools import cached_property
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr
from microsim import schema as ms
from numpy.typing import NDArray
from tqdm import tqdm

from cellgroup.configs.synthetic import MicroscopyConfig
from cellgroup.synthetic.imaging.utils import create_FP_distribution_from_array


class MicroscopeSimulator:
    """Simulator for spectral data using `microsim`."""
    
    def __init__(self, simulation_config: MicroscopyConfig):
        """Initialize the simulator with the given configuration."""
        
        self.sim_config: MicroscopyConfig = simulation_config
        self.optical_config: Sequence[ms.OpticalConfig] = self.get_optical_config()

    @property
    def optical_config(self) -> Sequence[ms.OpticalConfig]:
        """Create a list of optical configurations."""
        pass
        
    @property
    def PSNR(self) -> NDArray:
        """Compute the average channel-wise PSNR over the simulated unmixed images.
        
        Returns
        -------
        float
            The average PSNR over the simulated unmixed simages.
        """
        raise NotImplementedError("PSNR computation not yet implemented.")
        assert hasattr(self, "sim_imgs"), (
            "Simulate the dataset first before computing the PSNR."
        )
        
        psnr_lst = []
        for sim_img in self.sim_imgs:
            gt = sim_img["GT"]
            unmixed = sim_img["unmixed"]
            # if downscaling, we need to downsample also GT image
            if gt.shape != unmixed.shape:
                gt = coarsen_img(gt, self.data_sim_config.space_downscaling)
            # compute PSNR (for each channel)
            psnr_lst.append(
                [
                    scale_invariant_psnr(gt=gt[i], pred=unmixed[i])
                    for i in range(len(gt))
                ]
            )
        return np.asarray(psnr_lst).mean(axis=0)

    
    def _get_em_rates(self, sim: ms.Simulation) -> xr.DataArray:
        """Get the emission rates for the current simulation.
        
        Returns
        -------
        xr.DataArray
            The emission rates for the current simulation.
        """
        if not hasattr(self, "_em_rates"):
            self._em_rates = sim.filtered_emission_rates()
        
        return self._em_rates
    

    def _img_to_uint(self, img: NDArray) -> NDArray:
        """Cast image to uint by restricting intensities to the given bit range.
        
        Parameters
        ----------
        img : NDArray
            Image to be casted to uint.
        
        Returns
        -------
        NDArray
            Casted mage with intensities restricted to the given bit range.
        """
        bit_depth = self.sim_config.bit_depth
        if bit_depth == 8:
            return np.clip(img, 0, 2**8 - 1).astype(np.uint8)
        elif bit_depth == 16:
            return np.clip(img, 0, 2**16 - 1).astype(np.uint16)
        elif bit_depth == 32:
            return np.clip(img, 0, 2**32 - 1).astype(np.uint32)


    def create_sample(self, inputs: Sequence[NDArray]) -> ms.Sample:
        """Create a sample from a list of input arrays and fluorophores.
        
        Parameters
        ----------
        inputs : Sequence[Union[NDArray, Union[str, Path]]]
            The inputs to create the sample, i.e., sequences of fluorophore
            distributions for the different labels. They come as a sequence
            of arrays, one for each channel. Shape: (C, [Z], Y, X), where is
            the number of channels.
        
        Returns
        -------
        ms.Sample
            The sample object for the given distributions and fluorophores.
        """
        assert len(self.sim_config.fluorophores) == len(inputs), (
            "The number of inputs and fluorophores must be the same."
        )
        
        # load each fp distribution from a different array channel
        return ms.Sample(
            labels=[
                create_FP_distribution_from_array(fluorophore, array)
                for array, fluorophore in zip(
                    inputs, self.sim_config.fluorophores
                )
            ]
        )


    def init(self, samples: list[ms.Sample], seed: int) -> ms.Simulation:
        """Initialize a simulation object given a sample and a channel configuration.
        
        Parameters
        ----------
        samples : list[ms.Sample]
            The sample objects used to simulate data.
        seed: int
            The random seed for the simulation reproducibility.
            
        Returns
        -------
        ms.Simulation
            The initialized simulation object.
        """
        # --- disallow caching ---
        # we want to simulate new images each time given the same `Simulation` instance
        custom_cache_settings = ms.settings.CacheSettings(
            read=False,
            write=False,
        )
        
        # --- initialize `Simulation` instance ---
        return ms.Simulation(
            truth_space=ms.ShapeScaleSpace(
                shape=self.sim_config.space_shape,
                scale=self.sim_config.space_scale,
            ),
            output_space={"downscale": self.sim_config.space_downscaling},
            samples=samples,
            channels=self.optical_config,
            modality=ms.Identity(),
            settings=ms.Settings(
                cache=custom_cache_settings,
                random_seed=seed,
                spectral_bins_per_emission_channel=1,
            ),
            detector=ms.CameraCCD(
                qe=self.sim_config.detector_quantum_eff, 
                read_noise=self.sim_config.read_noise, 
                bit_depth=self.sim_config.bit_depth
            ),
        )

        
    def run(self, sim: ms.Simulation) -> tuple[NDArray, NDArray, NDArray]:
        """Run the `microsim` simulation returning the optical and digital images.
        
        Parameters
        ----------
        sim : ms.Simulation
            The simulation object.
        
        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            The unmixed optical, unmixed digital and images for the current samples as
            numpy arrays. In particular:
            - unmixed optical: (S, F, [Z], Y, X) -> high-SNR unmixed image.
            - unmixed digital: (S, F, [Z], Y, X) -> "real" noisy unmixed image.
            - digital: (S, C, [Z], Y, X) -> mixed spectral image.
        """
        # --- simulate images ---
        em_rates = self._get_em_rates(sim)
        opt_img_per_fluor = sim.optical_image_per_fluor(em_rates) # (S, C, F, Z, Y, X)
        opt_img = opt_img_per_fluor.sum("f") # (S, C, Z, Y, X)
        dig_img_per_fluor = sim.digital_image(
            opt_img_per_fluor, exposure_ms=self.sim_config.exposure_ms
        ) # (S, C, F, Z, Y, X)
        digital_img = sim.digital_image(
            opt_img, exposure_ms=self.sim_config.exposure_ms
        ) # (S, C, Z, Y, X) 
        
        # --- postprocess images ---
        # sum over channels
        opt_img_per_fluor = opt_img_per_fluor.sum("c") # (S, F, [Z], Y, X)
        dig_img_per_fluor = dig_img_per_fluor.sum("c") # (S, F, [Z], Y, X)
        # remove Z dimension, if singleton, convert to numpy
        opt_img_per_fluor = opt_img_per_fluor.squeeze("z", drop=True).values
        dig_img_per_fluor = dig_img_per_fluor.squeeze("z", drop=True).values
        digital_img = digital_img.squeeze("z", drop=True).values
        # cast to uint
        opt_img_per_fluor = self._img_to_uint(opt_img_per_fluor)
        dig_img_per_fluor = self._img_to_uint(dig_img_per_fluor)
        digital_img = self._img_to_uint(digital_img)
        return opt_img_per_fluor, dig_img_per_fluor, digital_img

    
    def simulate_img(
        self, input_: NDArray, seed: int = 123
    ) -> list[dict[str, NDArray]]:
        """Simulate one images (unmixed high-SNR & real + spectral mixed) given one
        input array of fluorophore distributions.
        
        Parameters
        ----------
        input : Sequence[NDArray]
            The input array for the simulation, containing the different fluorophore
            distributions in the different channels. Shape is (C, [Z], Y, X).
        seed : int
            The random seed for the simulation reproducibility. Default is 123.
        
        Returns
        -------
        NDArray
            The simulated microscope image resulting from the given input.
            Shape is (C, [Z], Y, X).
        """
        # --- create sample ---
        sample = self.create_sample(input_)
        
        # --- initialize simulation ---
        simulation = self.init(sample, seed)
        
        # --- run simulation ---
        return self.run(simulation)


    def simulate_dataset(self, input_data: Sequence[NDArray]) -> NDArray:
        """Simulate a dataset of spectral images for the current configuration.
        
        Parameters
        ----------
        input_data : Sequence[NDArray]
            The input data to simulate images from. Each input is an array of
            fluorophore distributions for the different channels.
            Shape is (S, C, [Z], Y, X), where S is the number of samples.
        
        Returns
        -------
        NDArray
            The simulated images for the given input data. Shape is (S, C, [Z], Y, X).
        """
        self.sim_imgs: list[NDArray] = []
        curr_seed = self.sim_config.random_seed
        for input_ in tqdm(input_data, desc="Simulating images"):
            self.sim_imgs.extend(
                self.simulate_img(input_, seed=curr_seed)
            )
            curr_seed += 1

        return self.sim_imgs


    def save(self) -> str:
        """Save the simulated images.
        
        Returns
        -------
        str
            The path to the saved images.
        """
        # set sim_info dict for save_dir naming
        sim_info = {
            "labels": self.sim_config.labels,
            "n_simulations": self.sim_config.n_simulations,
            "exposure": self.sim_config.exposure_ms,
            "read_noise": self.sim_config.read_noise,
        }
        
        # create unique save directory
        save_dirpath = get_save_dirpath(self.data_sim_config.save_dir, sim_info)
        print(f"Saving images into {save_dirpath}...")
        os.makedirs(save_dirpath, exist_ok=True)
        
        # save images
        save_simulation_results(
            images=self.sim_imgs, save_dir=save_dirpath
        )
        
        # save metadata
        save_simulation_metadata(metadata=self.data_sim_config, save_dir=save_dirpath)
        
        return save_dirpath


def simulate_spectral_data(
    simulation_config: MicroscopyConfig, input_data: Sequence[NDArray]
) -> None:
    """Simulate spectral data and save them on disk.
    
    Parameters
    ----------
    data_simulation_config : AnyDataSimulationConfig
        The configuration for the spectral data simulation.
    input_data : Sequence[NDArray]
        The input data to simulate images from. Each input is an array of
        fluorophore distributions for the different channels.
        Shape is (S, C, [Z], Y, X), where S is the number of samples.    
    """
    simulator = MicroscopeSimulator(simulation_config=simulation_config)
    simulator.simulate_dataset()
    print(f"Spectral data simulation done! PSNR: {simulator.PSNR.mean():.2f}")
    
    # save data & metadata
    save_path = simulator.save()
    metadata_path = os.path.join(
        save_path, "data_simulation_config.json"
    )