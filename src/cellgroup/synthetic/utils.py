"""Utility functions for synthetic data generation."""
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

class Status(Enum):
    """Enumeration of nucleus status."""
    ALIVE = "alive"
    DEAD = "dead"
    DIVIDED = "divided"
    
    def __str__(self):
        return self.value