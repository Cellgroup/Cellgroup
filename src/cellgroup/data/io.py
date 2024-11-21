import os
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from warnings import warn

import tifffile as tiff
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from cellgroup.data.utils import ChannelID, WellID, _sort_files_by_time


def get_fnames(
    data_dir: Union[str, Path],
    well_ids: Sequence[WellID],
    channel_ids: Sequence[ChannelID],
    img_dim: Literal["2D", "3D"] = "2D",
    t_steps: Optional[int] = None,
    every_n_steps: Optional[int] = None,
) -> dict[WellID, dict[ChannelID, list[Path]]]:
    """Get the filenames for the given wells and channels.
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        The directory containing the data.
    well_ids : Sequence[WellID]
        The IDs of the wells to include.
    channel_ids : Sequence[ChannelID]
        The IDs of the channels to include.
    img_dim : Literal["2D", "3D"]
        The dimensionality of the images. By default "2D".
    t_steps : Optional[int]
        The number of time steps to include.
        If None, all time steps are taken. By default None.
    every_n_steps : Optional[int]
        The number of time steps to include.
        If None, all time steps are taken. By default None.
    
    Returns
    -------
    dict[WellID, dict[ChannelID, list[Path]]]
        The list of filenames, organized by well and channel.
    """
    assert img_dim == "2D", "Only 2D images are supported for now."
    assert WellID.A05 not in well_ids, "Well A05 not available in the dataset for now."
    
    subdir = "slices" if img_dim == "2D" else "stacks"
    fnames_dict = {}
    for well_id in well_ids:
        fnames_dict[well_id.name] = {}
        well_subdir = os.path.join(data_dir, subdir, well_id.value)
        for channel_id in channel_ids:
            curr_fnames = [
                fname for fname in os.listdir(well_subdir) if channel_id.value in fname
            ]
            # --- select timesteps
            curr_fnames = _sort_files_by_time(curr_fnames)
            if t_steps:
                if t_steps > len(curr_fnames):
                    warn(
                        message=(
                            f"You picked {t_steps} timesteps with only"
                            f" {len(curr_fnames)} available. Taking all of them."
                        ),
                        stacklevel=2,
                    )
                curr_fnames = curr_fnames[:t_steps]
            # --- subsample timesteps
            if every_n_steps:
                curr_fnames = curr_fnames[::every_n_steps]
            # --- append to dict
            fnames_dict[well_id.name][channel_id.name] = [
                Path(os.path.join(well_subdir, fname)) for fname in curr_fnames
            ]
    return fnames_dict


def load_images(
    data_dir: Union[str, Path],
    well_ids: Sequence[WellID],
    channel_ids: Sequence[ChannelID],
    img_dim: Literal["2D", "3D"] = "2D",
    t_steps: Optional[int] = None,
    every_n_steps: Optional[int] = None,
) -> dict[WellID, dict[ChannelID, NDArray]]:
    """Load an image from a file.
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        The directory containing the data.
    well_ids : Sequence[WellID]
        The IDs of the wells to include.
    channel_ids : Sequence[ChannelID]
        The IDs of the channels to include.
    img_dim : Literal["2D", "3D"]
        The dimensionality of the images. By default "2D".
    t_steps : Optional[int]
        The number of time steps to include.
        If None, all time steps are taken. By default None.
    every_n_steps : Optional[int]
        The number of time steps to include.
        If None, all time steps are taken. By default None.
    
    Returns
    -------
    dict[WellID, dict[ChannelID, NDArray]]
        The loaded images, organized by well and channel.
    """
    fnames_dict = get_fnames(
        data_dir=data_dir, 
        well_ids=well_ids, 
        channel_ids=channel_ids, 
        img_dim=img_dim, 
        t_steps=t_steps, 
        every_n_steps=every_n_steps
    )
    images_dict = {}
    for well_id, channel_dict in fnames_dict.items():
        images_dict[well_id] = {}
        for channel_id, fnames in channel_dict.items():
            msg = f"Loading images for Well{well_id}-{channel_id}"
            imgs = [tiff.imread(fname) for fname in tqdm(fnames, desc=msg)]
            images_dict[well_id][channel_id] = np.stack(imgs)
    return images_dict