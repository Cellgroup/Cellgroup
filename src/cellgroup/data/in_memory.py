import os
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import tifffile as tiff
import xarray as xr
from numpy.typing import NDArray
from torch.utils.data import Dataset

from cellgroup.data.datasets.harvard import get_fnames


class InMemoryDataset(Dataset):
    """A PyTorch Dataset that loads all the data in memory.
    
    Data are assumed to have the following characteristics:
        - They can be 2D or 3D.
        - They can have one or multiple channels.
        - They are time-lapses.
    Therefore, data can be represented as (N, T, C, Z, Y, X) arrays.
    """
    
    def __init__(
        self,
        data_dir: Path,
        get_fnames_fn: Callable,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        data_dir : Path
            The directory containing the data.
        """
        self.data_dir = data_dir
        self.fnames = get_fnames_fn(data_dir)
        self.data = self._load_data()

    def _load_img(self, fname: Path) -> NDArray:
        if self.ext == ".tif":
            return tiff.imread(self.fname)
        else:
            raise ValueError(f"Unsupported file extension {self.ext}")
    
    def _load_data(self) -> xr.DataArray:
        """Load the data and store them in a `xarray.DataArray`.
        
        Returns
        -------
        xr.DataArray
            The loaded data. Shape is (N, T, C, Z, Y, X).
        """
        data = []
        self.ext = self.fnames[0].suffix
        for fname in self.fnames:
            img = self._load_img(fname)
            data.append(img)
        data = xr.DataArray(np.stack(data))
        return data

    def preprocess(self, data: xr.DataArray) -> xr.DataArray:
        """Preprocess the data.
        
        Parameters
        ----------
        data : xr.DataArray
            The data to preprocess.
        
        Returns
        -------
        xr.DataArray
            The preprocessed data.
        """
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
