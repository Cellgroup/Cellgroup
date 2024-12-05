import itertools
from typing import Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from cellgroup.data.patching._base import PatchInfo
from cellgroup.utils import Axis


def _compute_crop_and_stitch_coords_1d(
    axis_size: int, patch_size: int, overlap: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Compute the coordinates of each tile along an axis, given the overlap.

    Parameters
    ----------
    axis_size : int
        Length of the axis.
    patch_size : int
        Size of the tile for the given axis.
    overlap : int
        Size of the overlap for the given axis.

    Returns
    -------
    Tuple[Tuple[int, ...], ...]
        Tuple of all coordinates for given axis.
    """
    # Compute the step between tiles
    step = patch_size - overlap
    crop_coords = []
    stitch_coords = []
    overlap_crop_coords = []

    # Iterate over the axis with step
    for i in range(0, max(1, axis_size - overlap), step):
        # Check if the tile fits within the axis
        if i + patch_size <= axis_size:
            # Add the coordinates to crop one tile
            crop_coords.append((i, i + patch_size))

            # Add the pixel coordinates of the cropped tile in the original image space
            stitch_coords.append(
                (
                    i + overlap // 2 if i > 0 else 0,
                    (
                        i + patch_size - overlap // 2
                        if crop_coords[-1][1] < axis_size
                        else axis_size
                    ),
                )
            )

            # Add the coordinates to crop the overlap from the prediction.
            overlap_crop_coords.append(
                (
                    overlap // 2 if i > 0 else 0,
                    (
                        patch_size - overlap // 2
                        if crop_coords[-1][1] < axis_size
                        else patch_size
                    ),
                )
            )

        # If the tile does not fit within the axis, perform the abovementioned
        # operations starting from the end of the axis
        else:
            # if (axis_size - patch_size, axis_size) not in crop_coords:
            crop_coords.append((max(0, axis_size - patch_size), axis_size))
            last_tile_end_coord = stitch_coords[-1][1] if stitch_coords else 1
            stitch_coords.append((last_tile_end_coord, axis_size))
            overlap_crop_coords.append(
                (patch_size - (axis_size - last_tile_end_coord), patch_size)
            )
            break
    return crop_coords, stitch_coords, overlap_crop_coords


# TODO: find a way to decouple the patch extraction from the coords computation
# Indeed, the latter can be done once and for all before the loop
def _extract_overlapped_patches(
    arr: NDArray,
    patch_size: Sequence[int],
    patch_overlap: Sequence[int],
) -> tuple[list[NDArray], list[PatchInfo]]:
    # Create a list of coordinates for cropping and stitching all axes.
    # [crop coordinates, stitching coordinates, overlap crop coordinates]
    # For axis of size 35 and patch size of 32 compute_crop_and_stitch_coords_1d
    # will output ([(0, 32), (3, 35)], [(0, 20), (20, 35)], [(0, 20), (17, 32)])
    crop_and_stitch_coords_list = [
        _compute_crop_and_stitch_coords_1d(
            arr.shape[i], patch_size[i], patch_overlap[i]
        )
        for i in range(len(patch_size))
    ]

    # Rearrange crop coordinates from a list of coordinate pairs per axis to a list
    # grouped by type.
    all_crop_coords, all_stitch_coords, all_overlap_crop_coords = zip(
        *crop_and_stitch_coords_list
    )

    # Maximum tile index
    max_tile_idx = np.prod([len(axis) for axis in all_crop_coords]) - 1

    # Iterate over generated coordinate pairs:
    patches = []
    patch_info_lst = []
    for tile_idx, (crop_coords, stitch_coords, overlap_crop_coords) in enumerate(
        zip(
            itertools.product(*all_crop_coords),
            itertools.product(*all_stitch_coords),
            itertools.product(*all_overlap_crop_coords),
        )
    ):
        # Extract tile from the sample
        patches.append(arr[(..., *[slice(c[0], c[1]) for c in list(crop_coords)])])

        # Check if we are at the end of the sample by computing the length of the
        # array that contains all the tiles
        if tile_idx == max_tile_idx:
            last_tile = True
        else:
            last_tile = False

        # create tile information
        patch_info_lst.append(
            PatchInfo(
                array_shape=arr.shape,
                last_tile=last_tile,
                overlap_crop_coords=overlap_crop_coords,
                stitch_coords=stitch_coords
            )
        )

    return patches, patch_info_lst


def extract_overlapped_patches(
    data: xr.DataArray,
    patch_size: Sequence[int],
    patch_overlap: Sequence[int],
) -> xr.DataArray:
    """Generate tiles from the input array with specified overlap.

    The tiles cover the whole array. The method returns a generator that yields
    tuples of array and tile information, the latter includes whether
    the tile is the last one, the coordinates of the overlap crop, and the coordinates
    of the stitched tile.

    Input array has shape (N, C, T, (Z), Y, X), while the returned patch array has
    shape (N, C, T, P, (Z'), Y', X'), where P is the number of patches.

    Parameters
    ----------
    data : xr.DataArray
        Array of shape (N, C, T, (Z), Y, X).
    patch_size : Sequence[int]
        Patch size in each dimension, in the order ([Z'], Y', X').
    patch_overlap : Sequence[int]
        Overlap values in each dimension, of length 2 or 3.

    Yields
    ------
    xr.DataArray
        Array of patched data, shape is (N, C, T, P, (Z'), Y', X').
    """
    all_patches = []
    for sample_idx in range(data.shape[0]):
        curr_ch = []
        for channel_idx in range(data.shape[1]):
            curr_t = []
            for time_idx in range(data.shape[2]):
                patches, patches_info = _extract_overlapped_patches(
                    data[sample_idx, channel_idx, time_idx].values,
                    patch_size,
                    patch_overlap,
                )
                curr_t.append(patches)
            curr_ch.append(curr_t)
        all_patches.append(curr_ch)
                
    new_coords = data.coords.copy()
    new_coords["p"] = patches_info
    new_dims = data.dims[:3] + (Axis.P,) + data.dims[3:]
    patches = xr.DataArray(
        np.asarray(all_patches),
        coords=new_coords,
        dims=new_dims,
    )
    return patches
