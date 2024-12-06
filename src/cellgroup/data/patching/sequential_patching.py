from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from skimage.util import view_as_windows

from cellgroup.utils import Axis


def _compute_number_of_patches(
    arr_shape: tuple[int, ...], patch_sizes: Sequence[int]
) -> tuple[int, ...]:
    """
    Compute the number of patches that fit in each dimension.
    
    NOTE: when the array dimensions are NOT divisible by the patch sizes,
    we still keep the last patch.
    
    Parameters
    ----------
    arr_shape : tuple[int, ...]
        Shape of the input array.
    patch_sizes : Sequence[int]
        Shape of the patches.

    Returns
    -------
    tuple[int, ...]
        Number of patches in each dimension.
    """
    assert len(arr_shape) == len(patch_sizes), (
        "Array and patch sizes must have the same number of dimensions."
    )
    
    n_patches = [
        np.ceil(arr_shape[i] / patch_sizes[i]).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(n_patches)


def _compute_overlap(
    arr_shape: tuple[int, ...], patch_sizes: Sequence[int]
) -> tuple[int, ...]:
    """
    Compute the overlap between patches in each dimension.

    If the array dimensions are divisible by the patch sizes, then the overlap is
    0. Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr_shape : tuple[int, ...]
        Input array shape.
    patch_sizes : Union[list[int], tuple[int, ...]]
        Size of the patches.

    Returns
    -------
    tuple[int, ...]
        Overlap between patches in each dimension.
    """
    n_patches = _compute_number_of_patches(arr_shape, patch_sizes)

    overlap = [
        np.ceil(
            np.clip(n_patches[i] * patch_sizes[i] - arr_shape[i], 0, None)
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def _compute_patch_steps(
    patch_sizes: Union[list[int], tuple[int, ...]], overlaps: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Compute steps between patches.

    Parameters
    ----------
    patch_sizes : tuple[int]
        Size of the patches.
    overlaps : tuple[int]
        Overlap between patches.

    Returns
    -------
    tuple[int]
        Steps between patches.
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


def _compute_patch_views(
    arr: NDArray,
    window_shape: list[int],
    step: tuple[int, ...],
    output_shape: tuple[int],
) -> NDArray:
    """
    Compute views of an array corresponding to patches.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted.
    window_shape : tuple[int]
        Shape of the views.
    step : tuple[int]
        Steps between views.
    output_shape : tuple[int]
        Shape of the output array.

    Returns
    -------
    np.ndarray
        Array with views dimension.
    """
    rng = np.random.default_rng()
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches


def extract_sequential_patches(
    data: xr.DataArray,
    patch_size: Sequence[int],
    # TODO: introduce overlap between patches
) -> NDArray:
    """Extract patches from the input array.
    
    Parameters
    ----------
    data : xr.DataArray
        Input array. Shape is (N, C, T, [Z], Y, X).
    patch_size : Sequence[int]
        Size of the patches to extract. It covers the spatial dimensions only.
    
    Returns
    -------
    xr.DataArray
        Extracted patches. Shape is (N, C, T, P, [Z'], Y', X'), where
        ([Z'], Y', X') are the patch size.
    """    
    # Update patch size to encompass N, C, and T dimensions
    patch_size = [*data.shape[:3], *patch_size]

    # Compute overlap
    overlaps = _compute_overlap(arr_shape=data.shape, patch_sizes=patch_size)

    # Create view window and overlaps
    window_steps = _compute_patch_steps(patch_sizes=patch_size, overlaps=overlaps)
    
    # Generate a view of the input array containing pre-calculated number of patches
    # in each dimension with overlap.
    out_shape = patch_size.copy()
    out_shape.insert(3, -1)
    patches = _compute_patch_views(
        data.values,
        window_shape=patch_size,
        step=window_steps,
        output_shape=out_shape,
    ) # shape: (N, C, T, P, [Z'], Y', X')
    new_dims = list(data.dims)
    new_dims.insert(3, Axis.P)
    # TODO: add coords for P (?)
    patches = xr.DataArray(patches, coords=data.coords, dims=new_dims)
    return patches



