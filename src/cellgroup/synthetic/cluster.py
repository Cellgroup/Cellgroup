from pydantic import BaseModel, Field, model_validator

from cellgroup.synthetic import NucleusFluorophoreDistribution, Space


class Cluster(BaseModel):
    """Defines a cluster of nuclei."""
        
    nuclei: list[NucleusFluorophoreDistribution]
    
    space: Space
    
    # --- geometric properties ---
    max_radius: tuple[int, int, int]
    
    concentration: float
    """Density of nuclei in the cluster. This determines the position of new nuclei."""
    
    @property
    def count(self) -> int:
        return len(self.nuclei)
    
    @property
    def volume(self) -> float:
        # take max coordinates of the nuclei in the cluster
        raise NotImplementedError
    
    def update():
        """Update the status of nuclei in the cluster cluster."""
        raise NotImplementedError
        for nucleus in self.nuclei:
            nucleus.update()
            
    def render():
        """Render the cluster."""
        raise NotImplementedError
        for nucleus in self.nuclei:
            nucleus.render()
        