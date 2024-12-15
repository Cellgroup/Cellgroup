"""Supported options for various stuff."""
from enum import Enum

from cellgroup.data.preprocessing import standardize, normalize


class SupportedPreprocessing(Enum):
    """Supported preprocessing functions."""
    STANDARDIZE = standardize
    NORMALIZE = normalize