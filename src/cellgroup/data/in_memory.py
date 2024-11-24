import os
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import tifffile as tiff
import xarray as xr
from numpy.typing import NDArray
from torch.utils.data import Dataset

from cellgroup.data.config import DatasetConfig
from cellgroup.data.utils import Axis


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
        data_config: DatasetConfig,
        get_fnames_fn: Callable,
        # TODO: in case of synthetic data we need to pass a different function
    ):
        """Initialize the dataset.

        Parameters
        ----------
        data_dir : Path
            The directory containing the data.
        """
        self.data_dir = data_dir
        self.data_config = data_config
        self.dims = self._get_dims()
        self.fnames = get_fnames_fn(data_dir, **self.data_config.model_dump())
        self.data = self._load_data()
        self.patches = self._prepare_patches()
        
    def _get_dims(self) -> Sequence[Axis]:
        """Get the dimensions of the data."""
        if self.data_config.img_dim == "2D":
            return [Axis.N, Axis.C, Axis.T, Axis.Y, Axis.X]
        elif self.data_config.img_dim == "3D":
            return [Axis.N, Axis.C, Axis.T, Axis.Z, Axis.Y, Axis.X]
        else:
            raise ValueError(f"Unsupported image dimension {self.data_config.img_dim}")
        
    
    def _load_img(self, fname: Path) -> NDArray:
        if self.ext == ".tif":
            return tiff.imread(fname)
        else:
            raise ValueError(f"Unsupported file extension {self.ext}")
    
    def _load_data(self) -> xr.DataArray:
        """Load the data and store them in a `xarray.DataArray`.
        
        Returns
        -------
        xr.DataArray
            The loaded data. Shape is (N, C, T, [Z], Y, X).
        """
        data = []
        coords = {Axis.N: [], Axis.C: [], Axis.T: []}
        self.ext = self.fnames[0].suffix
        for sample in self.fnames.keys():
            for channel in self.fnames[sample].keys():
                for fname in self.fnames[sample][channel]:
                    img = self._load_img(fname)
                    coords[Axis.N].append(sample)
                    coords[Axis.C].append(channel)
                    # TODO: add time information
                    data.append(img)
        return xr.DataArray(
            np.stack(data), 
            coords=coords,
            dims=self.dims
        )
        
    def _prepare_patches(self) -> xr.DataArray:
        """Prepare the data for patch-based training."""

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
