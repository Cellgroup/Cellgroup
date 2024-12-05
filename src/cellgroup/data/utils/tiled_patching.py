import itertools
from typing import Generator, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class TileInformation(BaseModel):
    """Tile information model.

    This model is used to represent the information required to stitch back a tile into
    a larger image. It is used throughout the prediction pipeline of CAREamics.

    Array shape should be C(Z)YX, where Z is an optional dimensions.
    """

    model_config = ConfigDict(validate_default=True)

    array_shape: Sequence[int]
    """Shape of the original (untiled) array."""

    last_tile: bool = False
    """Whether this tile is the last one of the array."""

    overlap_crop_coords: tuple[tuple[int, ...], ...]
    """Inner coordinates of the tile where to crop the prediction in order to stitch
    it back into the original image."""

    stitch_coords: tuple[tuple[int, ...], ...]
    """Coordinates in the original image where to stitch the cropped tile back."""

    sample_id: int
    """Sample ID of the tile."""
    

    def __eq__(self, other_tile: object):
        """Check if two tile information objects are equal.

        Parameters
        ----------
        other_tile : object
            Tile information object to compare with.

        Returns
        -------
        bool
            Whether the two tile information objects are equal.
        """
        if not isinstance(other_tile, TileInformation):
            return NotImplemented

        return (
            self.array_shape == other_tile.array_shape
            and self.last_tile == other_tile.last_tile
            and self.overlap_crop_coords == other_tile.overlap_crop_coords
            and self.stitch_coords == other_tile.stitch_coords
            and self.sample_id == other_tile.sample_id
        )


def _compute_crop_and_stitch_coords_1d(
    axis_size: int, tile_size: int, overlap: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Compute the coordinates of each tile along an axis, given the overlap.

    Parameters
    ----------
    axis_size : int
        Length of the axis.
    tile_size : int
        Size of the tile for the given axis.
    overlap : int
        Size of the overlap for the given axis.

    Returns
    -------
    Tuple[Tuple[int, ...], ...]
        Tuple of all coordinates for given axis.
    """
    # Compute the step between tiles
    step = tile_size - overlap
    crop_coords = []
    stitch_coords = []
    overlap_crop_coords = []

    # Iterate over the axis with step
    for i in range(0, max(1, axis_size - overlap), step):
        # Check if the tile fits within the axis
        if i + tile_size <= axis_size:
            # Add the coordinates to crop one tile
            crop_coords.append((i, i + tile_size))

            # Add the pixel coordinates of the cropped tile in the original image space
            stitch_coords.append(
                (
                    i + overlap // 2 if i > 0 else 0,
                    (
                        i + tile_size - overlap // 2
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
                        tile_size - overlap // 2
                        if crop_coords[-1][1] < axis_size
                        else tile_size
                    ),
                )
            )

        # If the tile does not fit within the axis, perform the abovementioned
        # operations starting from the end of the axis
        else:
            # if (axis_size - tile_size, axis_size) not in crop_coords:
            crop_coords.append((max(0, axis_size - tile_size), axis_size))
            last_tile_end_coord = stitch_coords[-1][1] if stitch_coords else 1
            stitch_coords.append((last_tile_end_coord, axis_size))
            overlap_crop_coords.append(
                (tile_size - (axis_size - last_tile_end_coord), tile_size)
            )
            break
    return crop_coords, stitch_coords, overlap_crop_coords


def extract_tiles(
    arr: NDArray,
    tile_size: Sequence[int],
    overlaps: Sequence[int],
) -> Generator[tuple[NDArray, TileInformation], None, None]:
    """Generate tiles from the input array with specified overlap.

    The tiles cover the whole array. The method returns a generator that yields
    tuples of array and tile information, the latter includes whether
    the tile is the last one, the coordinates of the overlap crop, and the coordinates
    of the stitched tile.

    Input array should have shape SC(Z)YX, while the returned tiles have shape C(Z)YX,
    where C can be a singleton.

    Parameters
    ----------
    arr : NDArray
        Array of shape (S, C, (Z), Y, X).
    tile_size : Sequence[int]
        Tile sizes in each dimension, of length 2 or 3.
    overlaps : Sequence[int]
        Overlap values in each dimension, of length 2 or 3.

    Yields
    ------
    Generator[Tuple[NDArray, TileInformation], None, None]
        Tile generator, yields the tile and additional information.
    """
    for sample_idx in range(arr.shape[0]):
        sample: NDArray = arr[sample_idx, ...]

        # Create a list of coordinates for cropping and stitching all axes.
        # [crop coordinates, stitching coordinates, overlap crop coordinates]
        # For axis of size 35 and patch size of 32 compute_crop_and_stitch_coords_1d
        # will output ([(0, 32), (3, 35)], [(0, 20), (20, 35)], [(0, 20), (17, 32)])
        crop_and_stitch_coords_list = [
            _compute_crop_and_stitch_coords_1d(
                sample.shape[i + 1], tile_size[i], overlaps[i]
            )
            for i in range(len(tile_size))
        ]

        # Rearrange crop coordinates from a list of coordinate pairs per axis to a list
        # grouped by type.
        all_crop_coords, all_stitch_coords, all_overlap_crop_coords = zip(
            *crop_and_stitch_coords_list
        )

        # Maximum tile index
        max_tile_idx = np.prod([len(axis) for axis in all_crop_coords]) - 1

        # Iterate over generated coordinate pairs:
        for tile_idx, (crop_coords, stitch_coords, overlap_crop_coords) in enumerate(
            zip(
                itertools.product(*all_crop_coords),
                itertools.product(*all_stitch_coords),
                itertools.product(*all_overlap_crop_coords),
            )
        ):
            # Extract tile from the sample
            tile: NDArray = sample[
                (..., *[slice(c[0], c[1]) for c in list(crop_coords)])  # type: ignore
            ]

            # Check if we are at the end of the sample by computing the length of the
            # array that contains all the tiles
            if tile_idx == max_tile_idx:
                last_tile = True
            else:
                last_tile = False

            # create tile information
            tile_info = TileInformation(
                array_shape=sample.shape,
                last_tile=last_tile,
                overlap_crop_coords=overlap_crop_coords,
                stitch_coords=stitch_coords,
                sample_id=sample_idx,
            )

            yield tile, tile_info
