from enum import Enum


class Axis(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"
    C = "c" # Channel dimension
    T = "t" # Time dimension
    N = "n" # Sample dimension
    P = "p" # Patch dimension
    
    def __repr__(self) -> str:
        return f"<Axis.{self.name}>"

    def __str__(self) -> str:
        return self.value
    
    
# TODO: make abstract classes
class SampleID(Enum):
    """IDs for the different samples in the dataset."""
    pass
    
class ChannelID(Enum):
    """IDs for the different channels in the dataset."""
    pass