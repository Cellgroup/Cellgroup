import xarray as xr


def normalize(
    data: xr.DataArray, data_stats: dict[str, float]
) -> xr.DataArray:
        """Normalize the data.
        
        Parameters
        ----------
        data : xr.DataArray
            The data to normalize.
        data_stats : dict[str, float]
            Data statistics dictionary. Keys are "mean" and "std".
        
        Returns
        -------
        xr.DataArray
            The normalized data.
        """
        return (data - data_stats["mean"]) / data_stats["std"]