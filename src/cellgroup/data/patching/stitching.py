import builtins
from typing import Any, Sequence, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from tqdm import tqdm

from cellgroup.data.patching._base import PatchInfo
from cellgroup.data.utils import reorder_images
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


def stitch_patches(
    patches: Sequence[NDArray],
    infos: Sequence[dict[str, dict[str, Any]]],
) -> NDArray:
    """Stitch patches of a single frame.

    Parameters
    ----------
    patches : Sequence[NDArray]
        Sequence of arrays of shape ((Z), Y, X).
    infos : Sequence[dict[str, dict[str, Any]]]
        Sequence of additional information for each patch, comprising patch coords,
        dims, as well as `PatchInfo` objects for stitching.
        The dict must have the following structure:
        ```
        {
            "coords": {Axis : int | PatchInfo},
            "dims": list[Axis],
        }
        ```
    
    Returns
    -------
    NDArray
        Array of shape (N, C, T, (Z), Y, X).
    """
    patch_infos = [
        info["coords"][Axis.P] for info in infos
    ]
    start, end = 0, 0
    imgs = []
    imgs_info = []
    for i in tqdm(range(len(patch_infos)), desc="Stitching patches"):
        if patch_infos[i].last_patch:
            end = i + 1
            img = stitch_patches_single(patches[start:end], patch_infos[start:end])
            imgs.append(img)
            curr_info = infos[start]  # take only first since all equal
            curr_info["coords"].pop(Axis.P) # remove patch info
            curr_info["dims"].remove(Axis.P)
            imgs_info.append(curr_info)
            start = end
    # # --- reorder images into [N, C, T, (Z), Y, X] array
    # imgs = reorder_images(imgs, imgs_info)
    return imgs, imgs_info
    