import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr
from microsim import schema as ms
from numpy.typing import NDArray
from tqdm import tqdm

from cellgroup.configs.synthetic import ImagingConfig
from cellgroup.synthetic.imaging.utils import create_FP_distribution_from_array

# NOTE: there are 2 different cases:
# 1. Simulation data come from uncorrelated channels (i.e., each channel comes from a
# different file/image). This is the case for the BioSR dataset.
# 2. Simulation data come from correlated channels (i.e., all channels come from the same
# file/image). This is the case for the Lung-Tonsile dataset.


@dataclass
class MicroscopeSimulator:
    """Simulator for spectral data using `microsim`."""
    
    sim_config: ImagingConfig 

    @cached_property
    def optical_config(self) -> Sequence[ms.OpticalConfig]:
        """Create a list of optical configurations."""
        pass
        
        
    @cached_property
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
        bit_depth = self.data_sim_config.bit_depth
        if bit_depth == 8:
            return np.clip(img, 0, 2**8 - 1).astype(np.uint8)
        elif bit_depth == 16:
            return np.clip(img, 0, 2**16 - 1).astype(np.uint16)
        elif bit_depth == 32:
            return np.clip(img, 0, 2**32 - 1).astype(np.uint32)

    
    def _get_input_data(
        self, filepaths: Union[list[Union[str, Path]], list[list[Union[str, Path]]]], seed: Optional[int] = None
    ) -> list[Union[NDArray, list[Union[str, Path]]]]:
        """Get the input data for creating a `Sample` instance.
        
        Parameters
        ----------
        filepaths : Union[list[Union[str, Path]], list[list[Union[str, Path]]]]
            The list of file paths for the current simulation. If the channels are
            uncorrelated, then `filepaths` is a list of lists where each inner list
            contains the file paths for a specific label. Otherwise, `filepaths` is a
            list of file paths, each one referring to a multi-channel image.
        seed : Optional[int]
            The random seed for the simulation reproducibility (i.e., chosing the same
            filepaths over different simulations). If `None`, filepaths are chosen
            randomly without guaranteeing reproducibility for future experiments.
        
        Returns
        -------
        list[Union[NDArray, list[Union[str, Path]]]]
            The input data for creating a `Sample` instance. If the channels are
            uncorrelated, then the input is a list of file paths. Otherwise, the input
            is a list of arrays, one for each channel.
        """
        # set seed for reproducibility (if provided)
        if seed is not None:
            np.random.seed(seed)
        
        # get input data
        if isinstance(filepaths[0], list): # uncorrelated channels
            # --- randomly sample files ---
            inputs: list[Union[str, Path]] = []
            for _ in range(self.data_sim_config.batch_size):
                # sample one file per label
                inputs.append([
                    fpaths_per_label[np.random.randint(len(fpaths_per_label))]
                    for fpaths_per_label in filepaths
                ])
            # TODO: also implement deterministic sampling
        else: # correlated channels
            # --- read the file, return array ---
            raise NotImplementedError("Correlated channels not yet implemented.")
            inputs = self.data_sim_config.imreader(filepaths)
        
        return inputs


    def create_sample(self, inputs: Sequence[Union[NDArray, Union[str, Path]]]) -> ms.Sample:
        """Create a sample from a list of labels and fluorophores.
        
        Parameters
        ----------
        inputs : Sequence[Union[NDArray, Union[str, Path]]]
            The inputs to create the sample, i.e., sequences of fluorophore
            distributions for the different labels. They can either come as a list of
            file paths to load (in case of uncorrelated channels) or as a sequence of
            arrays, one for each channel (in case of correlated channels).
        
        Returns
        -------
        ms.Sample
            The sample object for the given distributions and fluorophores.
        """
        assert len(self.data_sim_config.fluorophores) == len(inputs), (
            "The number of inputs and fluorophores must be the same."
        )
        
        if self.data_sim_config.uncorrelated_channels:
            assert self.data_sim_config.imreader is not None, (
                "When channels are uncorrelated, an image reader function must be provided."
            )
            assert all(isinstance(inp, (str, Path)) for inp in inputs), (
                "When channels are uncorrelated, the inputs must be file paths."
            )
            # load each fp distribution from a different file
            return ms.Sample(
                labels=[
                    create_FP_distribution_from_file(
                        fluorophore, fpath, self.data_sim_config.imreader
                    )
                    for fpath, fluorophore in zip(
                        inputs, self.data_sim_config.fluorophores
                    )
                ]
            )
        else:
            assert all(isinstance(inp, NDArray) for inp in inputs), (
                "When channels are correlated, the inputs must be arrays."
            )
            # load each fp distribution from a different array channel
            return ms.Sample(
                labels=[
                    create_FP_distribution_from_array(fluorophore, array)
                    for array, fluorophore in zip(
                        inputs, self.data_sim_config.fluorophores
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
                shape=self.data_sim_config.space_shape,
                scale=self.data_sim_config.space_scale,
            ),
            output_space={"downscale": self.data_sim_config.space_downscaling},
            samples=samples,
            channels=self.optical_config,
            modality=ms.Identity(),
            settings=ms.Settings(
                cache=custom_cache_settings,
                random_seed=seed,
                spectral_bins_per_emission_channel=1,
            ),
            detector=ms.CameraCCD(
                qe=self.data_sim_config.detector_quantum_eff, 
                read_noise=self.data_sim_config.read_noise, 
                bit_depth=self.data_sim_config.bit_depth
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
            opt_img_per_fluor, exposure_ms=self.data_sim_config.exposure_ms
        ) # (S, C, F, Z, Y, X)
        digital_img = sim.digital_image(
            opt_img, exposure_ms=self.data_sim_config.exposure_ms
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

    
    def simulate_imgs(
        self, 
        filepaths: Union[list[Union[str, Path]], list[list[Union[str, Path]]]],
        seed: int
    ) -> list[dict[str, NDArray]]:
        """Simulate one set of images (unmixed high-SNR & real + spectral mixed) given
        a batch of file paths.
        
        Parameters
        ----------
        filepaths : Union[list[Union[str, Path]], list[list[Union[str, Path]]]]
            The list of file paths for the current simulation. If the channels are
            uncorrelated, then `filepaths` is a list of lists where each inner list
            contains the file paths for a specific label. Otherwise, `filepaths` is a
            list of file paths, each one referring to a multi-channel image.
        seed : int
            The random seed for the simulation reproducibility.
        
        Returns
        -------
        list[dict[str, NDArray]]
            A list of dictionaries containing the following keys:
            - "GT": the ground truth image (high-SNR unmixed).
            - "unmixed": the unmixed digital image (real noisy unmixed).
            - "spectral": the spectral mixed image.
        """
        # --- get inputs for creating the sample ---
        input_data = self._get_input_data(filepaths, seed)
        
        # --- create sample ---
        samples = [self.create_sample(input_) for input_ in input_data]
        
        # --- initialize simulation ---
        simulation = self.init(samples, seed)
        
        # --- run simulation ---
        opt_img_per_fp, dig_img_per_fp, dig_img = self.run(simulation)
        
        return [
            {
                "GT": oipf, 
                "unmixed": dipf, 
                "spectral": di
            }
            for oipf, dipf, di in zip(opt_img_per_fp, dig_img_per_fp, dig_img)
        ]


    def simulate_dataset(self) -> list[dict[str, NDArray]]:
        """Simulate a dataset of spectral images for the current configuration.
        
        Returns
        -------
        list[dict[str, NDArray]]
            A list of dictionaries containing the following keys:
            - "GT": the ground truth image (high-SNR unmixed).
            - "unmixed": the unmixed digital image (real noisy unmixed).
            - "spectral": the spectral mixed image.
        """
        # --- get file paths for the current dataset ---
        filepaths = get_filepaths(
            dataset_name=self.data_sim_config.dataset_name,
            data_dir=self.data_sim_config.data_dir,
            labels=self.data_sim_config.labels,
            uncorrelated_ch=self.data_sim_config.uncorrelated_channels,
        )
        
        self.sim_imgs = []
        curr_seed = self.data_sim_config.random_seed
        n_iters = self.data_sim_config.n_simulations // self.data_sim_config.batch_size
        for _ in tqdm(range(n_iters), desc="Simulating images"):
            self.sim_imgs.extend(
                self.simulate_imgs(filepaths=filepaths, seed=curr_seed)
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
            "labels": self.data_sim_config.labels,
            "n_simulations": self.data_sim_config.n_simulations,
            "exposure": self.data_sim_config.exposure_ms,
            "read_noise": self.data_sim_config.read_noise,
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
    data_simulation_config: "AnyDataSimulationConfig"
) -> tuple[str, str]:
    """Simulate spectral data and save them on disk.
    
    Parameters
    ----------
    data_simulation_config : AnyDataSimulationConfig
        The configuration for the spectral data simulation.
        
    Returns
    -------
    tuple[str, str]
        Paths to the simulated data and metadata.
    """
    simulator = SpectralDataSimulator(
        data_sim_config=data_simulation_config
    )
    simulator.simulate_dataset()
    print(f"Spectral data simulation done! PSNR: {simulator.PSNR.mean():.2f}")
    # TODO: implement option to avoid saving the data + loading them later
    save_path = simulator.save()

    # return data & metadata paths
    metadata_path = os.path.join(
        save_path, "data_simulation_config.json"
    )
    return save_path, metadata_path