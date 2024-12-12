from typing import Any

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