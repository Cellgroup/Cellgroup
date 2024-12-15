from pydantic import BaseModel


class Space(BaseModel):
    """Defines a space where synthetic data live."""
    
    space: tuple[int, int, int]
    
    scale: tuple[int, int, int]
    """Voxel size in each dimension in μm."""
        
    