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


def _get_array_coords_from_images(
    img_coords: Sequence[dict[str, Any]],  
) -> dict[str, Any]:
    """Get coordinates of the whole array of images from coords of the single images.
    
    Parameters
    ----------
    img_coords : Sequence[dict[str, Any]]
        List of dictionaries containing the coordinates of the patches.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the coordinates of the whole array of images.
    """
    coords = {}
    for axis in img_coords[0]:
        # --- get (sorted) unique values of coords for the axis
        coords[axis] = sorted(list(set([c[axis] for c in img_coords])))
    return coords


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
    
    # TODO: implement checks for the array shapes 
    
    # TODO: hacky, refactor!
    # --- get coords for the entire array
    arr_dims = infos[0]["dims"]
    images_coords = [info["coords"] for info in infos]
    arr_coords = _get_array_coords_from_images(images_coords)
    arr_shape = [len(arr_coords[dim]) for dim in arr_dims[:3]] + list(images[0].shape)
    arr = xr.DataArray(
        np.zeros(arr_shape),
        coords=arr_coords,
        dims=arr_dims,
    )
    # --- fill the array with images
    for i, image in enumerate(images):
        arr.loc[images_coords[i]] = image
    return arr
    
    
    
    
    # # --- sort infos and images by coords: T -> C -> N
    # sorted_items = sorted(
    #     zip(images, infos),
    #     key=lambda x: (x["coords"]["T"], x["coords"]["C"], x["coords"]["N"])
    # )
    # images, infos = zip(*sorted_items)
    # images, same_N, same_C, same_T = [], [], [images[0]]
    # curr_T = infos[0]["coords"]["T"]
    # curr_C = infos[0]["coords"]["C"]
    # curr_N = infos[0]["coords"]["N"]
    # for img in images[1:]:
    #     if infos["coords"]["T"] == curr_T:
    #         same_T.append(img)
    #     else:
    #         same_C.append(same_T)
    #         curr_T = infos["coords"]["T"]
        
    #     if infos["coords"]["C"] == curr_C:
    #         same_C.append(img)
    #     else:
    #         images.append(same_C)
    #         curr_C = infos["coords"]["C"]
    # return images