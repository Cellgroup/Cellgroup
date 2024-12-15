from ._base import PatchInfo
from .sequential_patching import extract_sequential_patches
from .stitching import stitch_patches_single, stitch_patches
from .tiled_patching import extract_overlapped_patches


# TODO: refactor patching
# In the future we will need:
# - tiled patching -> overlap = 0 gives sequential patching
# - random patching (for training)