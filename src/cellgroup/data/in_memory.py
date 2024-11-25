import inspect
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tifffile as tiff
import xarray as xr
from numpy.typing import NDArray
from torch.utils.data import Dataset

from cellgroup.data.config import DatasetConfig
from cellgroup.data.utils import Axis
from cellgroup.data.utils.patching import extract_sequential_patches


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
        data_config : DatasetConfig
            Configuration for the dataset.
        get_fnames_fn : Callable
            Function to get the filenames to load from the data directory.
        """
        self.data_dir = data_dir
        self.data_config = data_config
        self.dims = self._get_dims()
        self.data = self._load_data(get_fnames_fn)
        self.data_stats = self._get_data_stats()
        self.patches = self._prepare_patches()
        
    def _get_dims(self) -> Sequence[Axis]:
        """Get the dimensions of the data."""
        if self.data_config.img_dim == "2D":
            return [
                Axis.N.value, Axis.C.value, Axis.T.value, Axis.Y.value, Axis.X.value
            ]
        elif self.data_config.img_dim == "3D":
            return [
                Axis.N.value, Axis.C.value, Axis.T.value, Axis.Z.value, Axis.Y.value, Axis.X.value
            ]
        else:
            raise ValueError(f"Unsupported image dimension {self.data_config.img_dim}")
        
    def _load_img(self, fname: Path) -> NDArray:
        if self.ext == ".tif":
            return tiff.imread(fname)
        else:
            raise ValueError(f"Unsupported file extension {self.ext}")
    
    def _get_fnames_internal(self, get_fnames_fn: Callable) -> list[Path]:
        """Get the filenames to load from the data directory."""
        fn_signature = inspect.signature(get_fnames_fn)
        fn_params = set(fn_signature.parameters.keys())
        filtered_args ={
            k: v for k, v in self.data_config.model_dump().items() if k in fn_params
        }
        return get_fnames_fn(self.data_dir, **filtered_args)
        
    
    def _load_data(self, get_fnames_fn: Callable) -> xr.DataArray:
        """Load the data and store them in a `xarray.DataArray`.
        
        Parameters
        ----------
        get_fnames_fn : Callable
            Function to get the filenames to load from the data directory.
        
        Returns
        -------
        xr.DataArray
            The loaded data. Shape is (N, C, T, [Z], Y, X).
        """
        self.fnames = self._get_fnames_internal(get_fnames_fn)
        data = []
        coords = {Axis.N.value: [], Axis.C.value: [],} # TODO: add time coordinates
        for sample in self.fnames.keys():
            per_sample_data = []
            coords[Axis.N.value].append(sample)
            for channel in self.fnames[sample].keys():
                per_channel_data = []
                coords[Axis.C.value].append(channel)
                for i, fname in enumerate(self.fnames[sample][channel]):
                    self.ext = fname.suffix
                    img = self._load_img(fname)
                    # TODO: add time coordinates
                    per_channel_data.append(img)
                per_sample_data.append(np.stack(per_channel_data))
            data.append(np.stack(per_sample_data))
        return xr.DataArray(
            np.stack(data), 
            coords=coords,
            dims=self.dims
        )
    
    def _get_data_stats(self) -> dict:
        """Get statistics about the data.
        
        NOTE: statistics are computed per channel on the entire dataset.
        
        Returns
        -------
        dict
            Data statistics stored in a dict, whose keys are "mean" and "std".
        """
        return {
            "mean": self.data.mean(dim=self.dims),
            "std": self.data.std(dim=self.dims),
        }
        
        
    def _prepare_patches(self) -> xr.DataArray:
        """Prepare the data for patch-based training.
        
        Returns
        -------
        xr.DataArray
            The data in patch form. Shape is (n_patches, C, T, [Z'], Y', X').
        """
        patches = extract_sequential_patches(
            data=self.data.values,
            patch_size=self.data_config.patch_size,
        )
        new_dims = self.dims.copy()
        new_dims[0] = "p"
        return xr.DataArray(patches, dims=new_dims)
        

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
        return self.data.sizes[Axis.N]

    def __getitem__(self, idx):
        return self.patches[idx]
