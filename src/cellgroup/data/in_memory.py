import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import tifffile as tiff
import xarray as xr
from numpy.typing import NDArray
from torch.utils.data import Dataset

from cellgroup.configs import DataConfig
from cellgroup.data.preprocessing import normalize
from cellgroup.utils import Axis
from cellgroup.data.patching import (
    extract_sequential_patches, extract_overlapped_patches, PatchInfo
)


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
        data_config: DataConfig,
        get_fnames_fn: Callable,
        # TODO: in case of synthetic data we need to pass a different function
    ):
        """Initialize the dataset.

        Parameters
        ----------
        data_dir : Path
            The directory containing the data.
        data_config : DataConfig
            Configuration for the dataset.
        get_fnames_fn : Callable
            Function to get the filenames to load from the data directory.
        """
        self.data_dir: str = data_dir
        self.data_config: DataConfig = data_config
        
        self.dims: Sequence[Axis] = self._get_dims()
        
        self.data: xr.DataArray = self._load_data(get_fnames_fn) 
        # (N, C, T, [Z], Y, X)
        
        self.data_stats: dict[str, float] = self._get_data_stats()
        
        self.patches: xr.DataArray = self._prepare_patches() 
        # (N, C, T, P, [Z'], Y', X')
        
    def _get_dims(self) -> Sequence[Axis]:
        """Get the dimensions of the data."""
        if self.data_config.img_dim == "2D":
            return [
                Axis.N, Axis.C, Axis.T, Axis.Y, Axis.X
            ]
        elif self.data_config.img_dim == "3D":
            return [
                Axis.N, Axis.C, Axis.T, Axis.Z, Axis.Y, Axis.X
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
        coords = {
            Axis.N: list(self.fnames.keys()),
            Axis.C: list(self.fnames.values()[0].keys()),
            # TODO: this will break if different samples have different channels ...
            Axis.T: list(range(len(self.fnames.values()[0].values()[0]))),
        }
        for sample in self.fnames.keys():
            per_sample_data = []
            coords[Axis.N.value].append(sample)
            for channel in self.fnames[sample].keys():
                per_channel_data = []
                for fname in self.fnames[sample][channel]:
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
    
    def _get_data_stats(self) -> dict[str, float]:
        """Get statistics about the data.
        
        Returns
        -------
        dict[str, float]
            Data statistics stored in a dict, whose keys are "mean" and "std".
        """
        # TODO: compute by channel (?)
        return {
            "mean": self.data.mean(dim=self.dims),
            "std": self.data.std(dim=self.dims),
        }
        
        
    def _prepare_patches(self) -> tuple[xr.DataArray, Optional[PatchInfo]]:
        """Prepare the data for patch-based training.
        
        Returns
        -------
        xr.DataArray
            The data in patch form. Shape is (n_patches, C, T, [Z'], Y', X').
        """
        # --- preprocess data
        # TODO: move somewhere else (?)
        self.data = self.preprocess(self.data)
        
        # --- extract patches
        if self.data_config.patch_overlap is None:
            patches = extract_sequential_patches(
                data=self.data,
                patch_size=self.data_config.patch_size,
            ) 
        else:
            patches = extract_overlapped_patches(
                data=self.data,
                patch_size=self.data_config.patch_size,
                patch_overlap=self.data_config.patch_overlap,
            )
        return patches

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
        data = normalize(data, self.data_stats)
        return data

    def __len__(self):
        return np.prod(self.patches.shape[:4])

    def __getitem__(self, idx: int) -> tuple[NDArray, dict[str, dict[str, Any]]]:
        """Get a patch and its coordinates.
        
        Parameters
        ----------
        idx : int
            The index of the patch to get.
        
        Returns
        -------
        tuple[np.ndarray, dict[str, dict[str, Any]]]
            The patch as a `numpy` array with its coords and dims in a dict.
            The dict has the following structure:
            ```
            {
                "coords": {Axis : int | PatchInfo},
                "dims": list[Axis],
            }
            ```
        """
        assert idx < len(self), f"Index {idx} is out of bounds."
        
        # TODO: return torch tensor (?)
        N, C, T, P, *spatial = self.patches.shape
        n = idx // (C * T * P)
        remainder = idx % (C * T * P)
        c = remainder // (T * P)
        t = (remainder % (T * P)) // P
        p = remainder % P
        
        # TODO: use torch (?)
        patch = self.patches[n, c, t, p, ...].values
        coords = {
            Axis.N: self.patches.coords[Axis.N][n].values.item(),
            Axis.C: self.patches.coords[Axis.C][c].values.item(),
            Axis.T: t, # TODO: add time coordinates
            Axis.P: self.patches.coords[Axis.P][p].values.item(),
        }
        dims = list(self.patches.dims)
        info = {
            "coords": coords,
            "dims": dims,
        }
        return patch, info
