import numpy as np
import xarray as xr
from microsim.schema.sample import BaseDistribution, FluorophoreDistribution
from numpy.typing import NDArray


class FromArray(BaseDistribution):
    """Fluorophore distribution obtained from aan.
    
    For instance, this is the case when the entire image comes from a single file.
    
    Attributes
    ----------
    data : NDArray
        Data to crop, of shape (C, [Z], Y, X).
    """
    data: NDArray
    """Ground truth array."""
    
    def _crop_to_shape(self, shape: tuple[int, ...]) -> NDArray:
        """Center-crop data to a specific shape.
        
        Parameters
        ----------
        shape : tuple[int, ...]
            Target shape in ([Z], Y, X) format.
        
        Returns
        -------
        NDArray
            Cropped data.
        """
        _, *spatial_shapes = self.data.shape
        center = [s // 2 for s in spatial_shapes]
        slices = [slice(c - s // 2, c + s // 2) for c, s in zip(center, shape)]
        return self.data[(slice(None),) + tuple(slices)]
        
    def render(self, space: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        # --- crop data to shape (if necessary)
        if space.shape != data.shape:
            data = self._crop_to_shape(space.shape[-2:])
        
        return space + np.asarray(data).astype(space.dtype)
    
    
def create_FP_distribution_from_array(
    fluorophore: str,
    array: NDArray,
) -> FluorophoreDistribution:
    """Create a fluorophore distribution given a fluorophore and an array."""
    return FluorophoreDistribution(
        distribution=FromArray(array=array),
        fluorophore=fluorophore,
    )