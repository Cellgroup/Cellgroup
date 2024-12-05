from typing import Sequence

from pydantic import BaseModel, ConfigDict


class PatchInfo(BaseModel):
    """Patch information model.

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
        if not isinstance(other_tile, PatchInfo):
            return NotImplemented

        return (
            self.array_shape == other_tile.array_shape
            and self.last_tile == other_tile.last_tile
            and self.overlap_crop_coords == other_tile.overlap_crop_coords
            and self.stitch_coords == other_tile.stitch_coords
            and self.sample_id == other_tile.sample_id
        )
