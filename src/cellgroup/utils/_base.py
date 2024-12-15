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
    

class Sample(Enum):
    """IDs for the different samples in the dataset."""
    pass

    def __repr__(self) -> str:
        return f"<Sample.{self.name}>"

    def __str__(self) -> str:
        return self.value
    
    def __lt__(self, other):
        """Comparison operator for sorting."""
        if isinstance(other, Sample):
            return self.value < other.value
        return NotImplemented

    
class Channel(Enum):
    """IDs for the different channels in the dataset."""
    pass

    def __repr__(self) -> str:
        return f"<Channel.{self.name}>"

    def __str__(self) -> str:
        return self.value
    
    def __lt__(self, other):
        """Comparison operator for sorting."""
        if isinstance(other, Channel):
            return self.value < other.value
        return NotImplemented