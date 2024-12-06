import os
import re
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from warnings import warn

import tifffile as tiff
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from cellgroup.utils import SampleID, ChannelID

class SampleIDHarvard(SampleID):
    """IDs for the different samples in the dataset.
    
    NOTE: each sample is a well in the experiment.
    """
    A05 = "_A05_"
    A06 = "_A06_"
    A07 = "_A07_"
    A08 = "_A08_"
    A09 = "_A09_"
    
class ChannelIDHarvard(ChannelID):
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


def _subsample_timesteps(
    fnames: list[str], 
    t_steps_slice: tuple[int, int, int]
) -> list[str]:
    """Subsample filenames given an interval of timesteps.
    
    Parameters
    ----------
    fnames : list[str]
        List of filenames to subsample.
    t_steps_slice : tuple[int, int, int]
        An interval of time steps to include, in the form (start, end, step).
    
    Returns
    -------
    list[str]
        Subsampled list of filenames.
    """
    fnames = _sort_files_by_time(fnames)
    
    if len(t_steps_slice) == 2:
        start, end = t_steps_slice
        step = 1
    elif len(t_steps_slice) == 3:
        start, end, step = t_steps_slice
    
    if end > len(fnames):
        warn(
            message=(
                f"You picked the {end} timestep with only "
                f"{len(fnames)} available. Taking last one available."
            ),
            stacklevel=2,
        )
    if start >= len(fnames):
        raise ValueError(
            f"Invalid start timestep {start} with only {len(fnames)} available."
        )
     
    return fnames[slice(start, end, step)]


def get_fnames(
    data_dir: Union[str, Path],
    samples: Sequence[SampleIDHarvard],
    channels: Sequence[ChannelIDHarvard],
    img_dim: Literal["2D", "3D"] = "2D",
    time_steps: Optional[tuple[int, int, int]] = None,
) -> dict[SampleIDHarvard, dict[ChannelIDHarvard, list[Path]]]:
    """Get the filenames for the given samples and channels.
    
    The resulting filenames are ordered by timesteps and organized in a dict by
    `SampleID` and `ChannelID`. Specifically the structure is:
    ```
    {
        SampleID: {
            ChannelID: [Path, Path, ...]
        }
    }
    ```
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        The directory containing the data.
    samples : Sequence[SampleIDHarvard]
        The IDs of the samples to include.
    channels : Sequence[ChannelIDHarvard]
        The IDs of the channels to include.
    img_dim : Literal["2D", "3D"]
        The dimensionality of the images. By default "2D".
    time_steps : Optional[tuple[int, int, int]]
        A slice of time steps to include, in the form (start, end, step).
        If None, all time steps are taken. By default None.
    
    Returns
    -------
    dict[SampleID, dict[ChannelID, list[Path]]]
        The list of filenames, organized by sample and channel.
    """
    assert img_dim == "2D", "Only 2D images are supported for now."
    assert SampleIDHarvard.A05 not in samples, "Well A05 not available in the dataset for now."
    
    subdir = "slices" if img_dim == "2D" else "stacks"
    fnames_dict = {}
    for sample_id in samples:
        fnames_dict[sample_id] = {}
        sample_subdir = os.path.join(data_dir, subdir, sample_id.value)
        for channel_id in channels:
            curr_fnames = [
                fname for fname in os.listdir(sample_subdir) if channel_id.value in fname
            ]
            curr_fnames = _sort_files_by_time(curr_fnames)
            if time_steps is not None:
                curr_fnames = _subsample_timesteps(curr_fnames, time_steps)
            fnames_dict[sample_id][channel_id] = [
                Path(os.path.join(sample_subdir, fname)) for fname in curr_fnames
            ]
    return fnames_dict


# TODO: remove?
def load_images(
    data_dir: Union[str, Path],
    well_ids: Sequence[SampleID],
    channel_ids: Sequence[ChannelID],
    img_dim: Literal["2D", "3D"] = "2D",
    t_steps_subsample: Optional[tuple[int, int, int]] = None,
) -> dict[SampleID, dict[ChannelID, NDArray]]:
    """Load an image from a file.
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        The directory containing the data.
    well_ids : Sequence[SampleID]
        The IDs of the wells to include.
    channel_ids : Sequence[ChannelID]
        The IDs of the channels to include.
    img_dim : Literal["2D", "3D"]
        The dimensionality of the images. By default "2D".
    t_steps_subsample : Optional[tuple[int, int, int]]
        An interval of time steps to include, in the form (start, end, step).
        If None, all time steps are taken. By default None.
    
    Returns
    -------
    dict[SampleID, dict[ChannelID, NDArray]]
        The loaded images, organized by well and channel.
    """
    raise NotImplementedError("This function is not implemented yet.")
    fnames_dict = get_fnames(
        data_dir=data_dir, 
        well_ids=well_ids, 
        channel_ids=channel_ids, 
        img_dim=img_dim,
        t_steps_subsample=t_steps_subsample
    )
    images_dict = {}
    for well_id, channel_dict in fnames_dict.items():
        images_dict[well_id] = {}
        for channel_id, fnames in channel_dict.items():
            msg = f"Loading images for Well{well_id}-{channel_id}"
            imgs = [tiff.imread(fname) for fname in tqdm(fnames, desc=msg)]
            images_dict[well_id][channel_id] = np.stack(imgs)
    return images_dict