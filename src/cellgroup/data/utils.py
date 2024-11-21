import re
from enum import Enum
from warnings import warn


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