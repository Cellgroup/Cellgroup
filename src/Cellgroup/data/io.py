import os
import re
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import tifffile as tiff
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class WellID(Enum):
    """IDs for the different wells in the dataset."""
    A05 = "_A05_"
    A06 = "_A06_"
    A07 = "_A07_"
    A08 = "_A08_"
    A09 = "_A09_"
    
class ChannelID(Enum):
    """IDs for the different channels in the dataset."""
    Ch1 = "C01"
    Ch2 = "C02"
    Ch5 = "C05"
    Ch6 = "C06"
    Ch13 = "C13"
    

def _sort_files_by_time(files: list[str]) -> list[str]:
    """Sort a list of filenames based on the day and time step encoded in their names.
    
    Day is denoted by 'D#xx' and time step by 'Txxxx'.
    
    Parameters
    ----------
    files list[str]: 
        List of filenames to be sorted.
    
    Returns
    -------
    list[str]: 
        Sorted list of filenames.
    """
    def extract_day_and_time(filename: str):
        """Extract day and time using regex."""
        match = re.search(r'D#(\d+)_T(\d+)', filename)
        if match:
            day = int(match.group(1))
            time_step = int(match.group(2))
            return (day, time_step)
        # Sort invalid entries to the end
        return (float('inf'), float('inf'))

    return sorted(files, key=extract_day_and_time)


def get_fnames(
    data_dir: Union[str, Path],
    well_ids: Sequence[WellID],
    channel_ids: Sequence[ChannelID],
    img_dim: Literal["2D", "3D"] = "2D",
    t_steps: Optional[int] = None,
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
            curr_fnames = _sort_files_by_time(curr_fnames)
            curr_fnames = curr_fnames[:t_steps] if t_steps else curr_fnames
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
    
    Returns
    -------
    dict[WellID, dict[ChannelID, NDArray]]
        The loaded images, organized by well and channel.
    """
    fnames_dict = get_fnames(data_dir, well_ids, channel_ids, img_dim, t_steps)
    images_dict = {}
    for well_id, channel_dict in fnames_dict.items():
        images_dict[well_id] = {}
        for channel_id, fnames in channel_dict.items():
            msg = f"Loading images for Well{well_id}-{channel_id}"
            imgs = [tiff.imread(fname) for fname in tqdm(fnames, desc=msg)]
            images_dict[well_id][channel_id] = np.stack(imgs)
    return images_dict


# if __name__ == "__main__":
#     # Test the function
#     DATA_DIR = "/group/jug/federico/data/Cellgroup/"
#     res = load_images(DATA_DIR, [WellID.A06], [ChannelID.Ch1], t_steps=5)