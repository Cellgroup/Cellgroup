import xarray as xr


def standardize(
    data: xr.DataArray, data_stats: dict[str, float]
) -> xr.DataArray:
        """Standardize the data.
        
        Parameters
        ----------
        data : xr.DataArray
            The data to normalize.
        data_stats : dict[str, float]
            Data statistics dictionary. Keys are "mean" and "std".
        
        Returns
        -------
        xr.DataArray
            The standardized data.
        """
        return (data - data_stats["mean"]) / data_stats["std"]
    

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
    return (data - data_stats["min"]) / (data_stats["max"] - data_stats["min"])