from enum import Enum


class Axis(Enum):
    X = "x"
    Y = "y"
    Z = "z"
    C = "c"
    T = "t"
    N = "n"
    
    
class SampleID(Enum):
    """IDs for the different samples in the dataset."""
    pass
    
class ChannelID(Enum):
    """IDs for the different channels in the dataset."""
    pass