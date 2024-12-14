from typing import Any, Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def in_memory_collate_fn(
    batch: list[tuple[NDArray, dict[str, Any]]]
) -> tuple[NDArray, list[dict[str, Any]]]:
    """Custom collate function for `InMemoryDataset` class.

    Parameters
    ----------
    batch : list[tuple[NDArray, dict[str, Any]]]
        List of tuples of patches and associated coordinates. Shape of patches is
        [N, C, T, P, (Z'), Y', X'].

    Returns
    -------
    tuple[NDArray, list[dict[str, Any]]]
        Processed patches and coordinates dictionaries.
    """
    inputs, coords = zip(*batch)
    inputs = np.stack(inputs) # TODO: convert `inputs` to `torch` (?)
    return inputs, coords


def reorder_images(
    images: Sequence[xr.DataArray],
    infos: Sequence[dict[str, dict[str, Any]]],
) -> xr.DataArray:
    """Reorder images according to their coords.

    Parameters
    ----------
    imgs : NDArray
        Sequence of `DataArray`'s of shape [(Z), Y, X]
    infos : Sequence[dict[str, dict[str, Any]]]
        Sequence of dictionaries containing the coordinates of the patches.
    
    Returns
    -------
    xr.DataArray
        Reordered array of shape [N, C, T, (Z), Y, X].
    """
    assert len(images) == len(infos), "Images and infos must have the same length."
    # --- sort infos and images by coords: T -> C -> N
    sorted_items = sorted(
        zip(images, infos),
        key=lambda x: (x["coords"]["T"], x["coords"]["C"], x["coords"]["N"])
    )
    images, infos = zip(*sorted_items)
    images, same_N, same_C, same_T = [], [], [images[0]]
    curr_T = infos[0]["coords"]["T"]
    curr_C = infos[0]["coords"]["C"]
    curr_N = infos[0]["coords"]["N"]
    for img in images[1:]:
        if infos["coords"]["T"] == curr_T:
            same_T.append(img)
        else:
            same_C.append(same_T)
            curr_T = infos["coords"]["T"]
        
        if infos["coords"]["C"] == curr_C:
            same_C.append(img)
        else:
            images.append(same_C)
            curr_C = infos["coords"]["C"]
    return images