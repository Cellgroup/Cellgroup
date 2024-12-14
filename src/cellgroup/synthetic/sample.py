from pydantic import BaseModel, Field, model_validator

from cellgroup.synthetic import Cluster


class Sample(BaseModel):
    """Defines a sample with multiple clusters that evolve over time."""
    
    clusters: list[Cluster]
    
    @property
    def count(self) -> int:
        return len(self.clusters)
    
    # --- methods ---
    def update():
        """Update the status of clusters in the sample."""
        raise NotImplementedError
        for cluster in self.clusters:
            cluster.update()
            
    def render():
        """Render the sample."""
        raise NotImplementedError
        for cluster in self.clusters:
            cluster.render()