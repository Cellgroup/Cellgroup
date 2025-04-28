import datetime
import json
import os
from pathlib import Path
from typing import Any, Union, Sequence

import tifffile as tiff
from numpy.typing import NDArray
from tqdm import tqdm

from cellgroup.configs.synthetic import MicroscopyConfig


def _append_info_to_save_dirpath(
    save_dirpath: Union[str, Path], exposure: float, read_noise: float,
) -> str:
    """Append information to the save directory path to create indetifiable
    simulated data directory.
    
    Specifically, we append:
    - The initials of the labels used for simulation (e.g., CCPs -> C, ER -> E, ...).
    - The number of simulations (e.g., 'n200').
    - The computed PSNR value (unmixed w.r.t. ground truth image).
    Hence the final save directory path will look like the following:
    `save_dirpath_CEM_n200_exp100_rn5`.
    """
    exposure_str = f"exp{int(exposure)}"
    read_noise_str = f"rn{int(read_noise)}"
    identifier = "_".join(
        ("sim_data", exposure_str, read_noise_str)
    )
    return os.path.join(save_dirpath, identifier)


def get_save_dirpath(base_dirpath: Union[str, Path], sim_info: dict[str, Any]) -> str:
    """Get the save directory path appending date and version to the base path.
    
    Parameters
    ----------
    base_dirpath : Union[str, Path]
        The base save directory path.
    sim_info : dict[str, Any]
        Information about the simulation to append to the save directory name
        to uniquely identify the simulated data. It should contain the following keys:
        - labels: Sequence[str]
        - n_simulations: int
        - exposure: float
        - read_noise: float
    
    Returns
    -------
    str
        The save directory path with a specific name for the simulated data.
    """
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%y%m")
    current_dir = os.path.join(base_dirpath, formatted_date)
    return _append_info_to_save_dirpath(current_dir, **sim_info)


def save_simulation_results(images: list[dict[str, NDArray]], save_dir: str) -> None:
    """Save the simulation results into the given directory.
    
    Parameters
    ----------
    images : list[dict[str, NDArray]]
        The simulated images to save.
    save_dir : str
        The directory where to save the images.
    """
    # save images
    for i, img_dict in tqdm(enumerate(images), desc="Saving images"):
        for k, img in img_dict.items():
            curr_save_dir = os.path.join(save_dir, k)
            os.makedirs(curr_save_dir, exist_ok=True)
            img = img_dict[k]
            fname = f"{k}_img_{i+1}.tif"
            tiff.imwrite(os.path.join(curr_save_dir, fname), img.squeeze())


def save_simulation_metadata(metadata: MicroscopyConfig, save_dir: str) -> None:
    """Save the simulation metadata into the given directory.
    
    Parameters
    ----------
    metadata : MicroscopyConfig
        The metadata to save, i.e., the configuration used for data simulation.
    save_dir : str
        The directory where to save the metadata.
    """
    with open(os.path.join(save_dir, "data_simulation_config.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=4)