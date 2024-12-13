import itertools
import builtins
from typing import Sequence, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from cellgroup.data.patching._base import PatchInfo
from cellgroup.utils import Axis


def stitch_patches_single(
    patches: Sequence[NDArray],
    patch_infos: Sequence[PatchInfo],
) -> NDArray:
    """Stitch patches of a single frame.

    Parameters
    ----------
    patches : Sequence[NDArray]
        Sequence of arrays of shape ((Z), Y, X).
    patch_infos : Sequence[PatchInfo]
        Sequence of patch information objects.

    Returns
    -------
    NDArray
        Array of shape (N, C, T, (Z), Y, X).
    """
    input_shape = patch_infos[0].array_shape
    predicted_image = np.zeros(input_shape, dtype=np.float32)

    for tile, tile_info in zip(patches, patch_infos):

        # Compute coordinates for cropping predicted tile
        crop_slices: tuple[Union[builtins.ellipsis, slice], ...] = (
            ...,
            *[slice(c[0], c[1]) for c in tile_info.overlap_crop_coords],
        )

        # Crop predited tile according to overlap coordinates
        cropped_tile = tile[crop_slices]

        # Insert cropped tile into predicted image using stitch coordinates
        image_slices = (..., *[slice(c[0], c[1]) for c in tile_info.stitch_coords])
        predicted_image[image_slices] = cropped_tile.astype(np.float32)

    return predicted_image