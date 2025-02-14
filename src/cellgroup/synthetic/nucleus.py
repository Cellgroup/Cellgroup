from __future__ import annotations

from typing import Annotated, Optional

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from numpy.typing import NDArray

from cellgroup.synthetic.space import Space
from cellgroup.synthetic.utils import Status

# NOTE: we do all the geometric simulation with [X, Y, Z] order for simplicity.
# Then we switch to [Z, Y, X] order for rendering purposes.

# TODO: make separate classes for 2D and 3D nuclei (?)
# TODO: do we really need pydantic here? -> maybe make pydantic model for attributes and separately pass to Nucleus instance
class Nucleus(BaseModel):
    """Defines a nucleus instance with minimal core properties and growth dynamics.
    
    TODO: add overall description of the class and its purpose.
    
    Attributes
    ----------
    idx : int
        Unique nucleus index.
    cluster_idx : Optional[int]
        Cluster index. Can be `None` if nucleus does not belong to any cluster.
    time : int
        Global timestep of simulation.
    eta : int, default=0
        Age of nucleus in timesteps.
    status : Status, default=Status.ALIVE
        Current status of the nucleus, one of `Status.ALIVE`, `Status.DEAD`, or
        `Status.DIVIDED`.
    centroid : tuple[float, ...]
        Coordinate of nucleus centroid as [X, Y, [Z]].
    semi_axes : tuple[float, ...]
        Semi-axes of the nucleus, in [X, Y, [Z]] ordering.
    angle_x : float, default=0.0
        Orientation angle relative to X-axis (in degrees). Also referred to as `theta`.
    angle_y : Optional[float], default=None
        Orientation angle relative to Y-axis (in degrees). Also referred to as `phi`.
    angle_z : Optional[float], default=None
        Orientation angle relative to Z-axis (in degrees). Also referred to as `psi`.
    raw_int_density : Optional[float], default=None
        Raw integrated density, i.e., fluorescence intensity of the nucleus.
    growth_rate : Optional[float], default=0.1
        Base growth rate.
    max_size : float, default=1000.0
        Maximum area.
    min_division_size : Optional[float], default=500.0
        Minimum size for division.
    min_viable_size : Optional[float], default=50.0
        Minimum viable size.
    max_age : Optional[int], default=200
        Maximum age in timesteps.
    lineage : Optional[list[int]], default_factory=list
        List of parent nuclei IDs.
    death_prob : Optional[float], default=0.0
        Probability of death.
    division_prob : Optional[float], default=0.0
        Probability of division.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True, # TODO: why is this needed?
        validate_assignment=True,
        validate_default=True,
    )
    
    # Essential identification and tracking
    idx: int # TODO: check how to handle unique IDs generation
    "Unique nucleus index." 
    cluster_idx: Optional[int] = None # TODO: can a nucleus not belong to any cluster?
    "Cluster index."
    time: int
    "Global timestep of simulation."
    eta: int = 0
    "Age of nucleus in timesteps."
    status: Status = Status.ALIVE
    "Viability status."
    space: Space
    "Geometrical space where the nucleus exists."

    # Core positional and geometric properties
    # TODO: introduce unit of measurement to have more realistic reference values!
    centroid: Annotated[tuple[float, ...], AfterValidator(lambda x: np.asarray(x))]
    "Coordinate of nucleus centroid as [X, Y, [Z]]."
    semi_axes: Annotated[tuple[float, ...], AfterValidator(lambda x: np.asarray(x))]
    "Semi-axes of the nucleus, in [X, Y, [Z]] ordering."
    angle_x: float = 0.0
    """Orientation angle relative to X-axis (in degrees). Also referred to as `theta`.
    This is the only angle needed for the 2D case."""
    angle_y: Optional[float] = None
    """Orientation angle relative to Y-axis (in degrees). Also referred to as `phi`.
    Needed for the 3D case."""
    angle_z: Optional[float] = None
    """Orientation angle relative to Z-axis (in degrees). Also referred to as `psi`.
    Needed for the 3D case."""
    
    # Core intensity properties
    raw_int_density: Optional[float] = None # TODO: not sure is needed
    """Raw integrated density, i.e., fluorescence intensity of the nucleus.
    Disabled if `None`."""
    
    # Growth and death properties
    # TODO: put all of these in a config file!
    # TODO: it would be nice to set ranges for these values to avoid unrealistic values
    growth_rate: Optional[float] = 0.1
    "Base growth rate. Disabled if `None`."
    max_size: float = 1000.0
    "Maximum area. Disabled if `None`."
    min_division_size: Optional[float] = 500.0
    "Minimum size for division. Disabled if `None`."
    min_viable_size: Optional[float] = 50.0
    "Minimum viable size. Disabled if `None`."
    max_age: Optional[int] = 200
    "Maximum age in timesteps. Disabled if `None`."
    lineage: Optional[list[int]] = Field(default_factory=list)
    "List of parent nuclei IDs. Disabled if `None`."
    death_prob: Optional[float] = 0.0
    "Probability of death. Disabled if `None`."
    division_prob: Optional[float] = 0.0
    "Probability of division. Disabled if `None`."
    
    @model_validator(mode="after")
    def _validate_dims(self):
        if len(self.centroid) not in (2, 3):
            raise ValueError("Nucleus centroid must have 2 or 3 dimensions.")
        if len(self.semi_axes) != len(self.centroid):
            raise ValueError(
                f"Found {len(self.centroid)}-dimensional centroid with "
                f"{len(self.semi_axes)} semi-axes."
            )
        return self
    
    @model_validator(mode="after")
    def _validate_angles(self):
        if self.is_3D:
            if self.angle_y is None:
                self.angle_y = 0.0
            if self.angle_z is None:
                self.angle_z = 0.0
        else:
            if self.angle_y is not None or self.angle_z is not None:
                raise ValueError("`angle_y` and `angle_z` are only used for 3D case.")
        return self
    
    @model_validator(mode="after")
    def _validate_wrt_space(self):
        if len(self.centroid) != self.space.ndim:
            raise ValueError(
                f"Centroid and space dimensions must match: "
                f"{len(self.centroid)} != {self.space.ndim}."
            )
        return self
    
    # TODO: add more validators (if needed)
    
    #TODO: implement nice __repr__ method to get a summary of the sample

    # --- Useful properties ---
    @property
    def is_3D(self) -> bool:
        """Check if nucleus is 3D."""
        return len(self.centroid) == 3
    
    @property
    def ndims(self) -> int:
        """Return number of dimensions."""
        return len(self.centroid)
    
    @property
    def angles(self) -> tuple[float, ...]:
        """Return orientation angles as a tuple in [X, Y, (Z)] order."""
        if self.is_3D:
            return self.angle_x, self.angle_y, self.angle_z
        else:
            return self.angle_x,
    
    @property
    def bounding_box(self) -> tuple[tuple[float, float], ...]:
        """Calculate bounding box of nucleus."""
        return tuple(
            (c - a, c + a) for c, a in zip(self.centroid, self.semi_axes)
        )
    
    
    # --- Derived geometric properties ---    
    @property
    def area(self) -> float:
        """Calculate area using ellipse formula."""
        if self.is_3D:
            raise ValueError("Area calculation not supported for 3D nuclei.")
        else:
            return np.pi * np.prod(self.semi_axes)
    
    @property
    def volume(self) -> float:
        """Calculate volume using ellipsoid formula."""
        if not self.is_3D:
            raise ValueError("Volume calculation not supported for 2D nuclei.")
        else:
            return (4/3) * np.pi * np.prod(self.semi_axes)
        
    @property
    def _size(self) -> float:
        """Private property to refer to area or volume depending on 2D/3D."""
        return np.prod(self.semi_axes)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return np.min(self.semi_axes) / np.max(self.semi_axes)

    @property
    def roundness(self) -> float:
        """Calculate roundness as the ratio between the ellipse (resp. ellipsoid) area
        (resp. volume) and the one of a circle (resp. sphere) with the longer semiaxis
        as its radius."""
        if self.is_3D:
            return self.volume / (4/3 * np.pi * np.max(self.semi_axes) ** 3)
        else:
            return self.area / (4 * np.pi * np.max(self.semi_axes) ** 2)

    @property
    def perimeter(self) -> float:
        """Calculate perimeter using Ramanujan approximation."""
        if self.is_3D:
            raise ValueError("Perimeter calculation not supported for 3D nuclei.")
        else:
            a = self.semi_axes[0]
            b = self.semi_axes[1]
            h = ((a - b) / (a + b)) ** 2
            return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
    @property
    def surface_area(self) -> float:
        """Calculate surface area of 3D nucleus using approximate formula.
        
        See: https://en.wikipedia.org/wiki/Ellipsoid#Surface_area
        """
        if not self.is_3D:
            raise ValueError("Surface area calculation not supported for 2D nuclei.")
        else:
            return 4 * np.pi * (
                np.sum((
                    (self.semi_axes[0] * self.semi_axes[1]) ** 1.6,
                    (self.semi_axes[0] * self.semi_axes[2]) ** 1.6,
                    (self.semi_axes[1] * self.semi_axes[2]) ** 1.6
                )) / 3
            ) ** (1 / 1.6)

    @property # TODO: remove ?
    def solidity(self) -> float:
        """Approximate solidity (area/convex hull area)."""
        raise NotImplementedError

    @property
    def mean_int_density(self) -> Optional[float]:
        """Calculate mean intensity."""
        if self.raw_int_density is None:
            return None
        else:
            return self.raw_int_density / self._size            

        
    def _calculate_growth_factor(self) -> float:
        """Calculate isotropic growth factor based on current size and conditions."""
        # Logistic growth factor
        size_factor = 1 - (self._size / self.max_size)
        
        # Age-dependent modulation
        age_factor = np.exp(-self.eta / 100)  # Decreases with age

        # Calculate base growth increment
        growth_increment = self.growth_rate * size_factor * age_factor

        # Add some random variation
        noise = np.random.normal(0, 0.02)  # 2% random variation 
        #TODO: check if this is a good value to hardcode, or if it should be a parameter

        return 1 + growth_increment + noise

    def check_death(self) -> bool:
        """Check if the nucleus should die based on various conditions."""
        if self.status == Status.DEAD:
            return False

        # Death conditions
        if (
            self._size < self.min_viable_size or  # Too small
            self.eta > self.max_age or  # Too old
            np.random.random() < self.death_prob # Random death
        ):
            return True

        return False

    def die(self) -> None:
        """Simulate death of nucleus by progressively reducing its size."""
        self.status = Status.DEAD
        
        # simulate death with a rapid size decrease
        shrink_factor = 0.5
        self.semi_axes = self.semi_axes * shrink_factor
        if self.raw_int_density is not None:
            self.raw_int_density *= shrink_factor
        
    def check_division(self) -> bool:
        """Check if the nucleus should divide based on various conditions."""
        if (
            self._size > self.min_division_size and  # Big enough
            np.random.random() < self.division_prob  # Random division
        ):
            return True

        return False    

    def divide(self) -> tuple["Nucleus", "Nucleus"]:
        """Divide nucleus if conditions are met and return the two daughter nuclei.
        
        **Assumptions** 
        1. Time resolution of desired timelapses is way larger than the typical
        division time of a nucleus. Hence, we don't need to model the position of cells
        after division with high precision.
        2. In our pipeline, division is the first thing happening in the update call.
        Specifically, we assume that division happens immediately at the beginning of
        the timestep, hence daughter cells have time to grow and move during the same
        timestep.

        **How it works**
        - Division happens at the mid point of the longest axis.
        - Volume is divided equally plus some noise.
        - Intensity density is divided roughly equally plus some noise.
        - Centroids of daughter cells are places at the vertices of the longest axis.
        - Daughter cells have the same orientation of the mother.
        Last two points are not realistic, but it is fine as these nuclei will move
        and rotate during the same timestep.
        """
        # --- Create daughter nucleus with same properties
        d1, d2 = self.model_copy(), self.model_copy()
        d1.idx, d2.idx = self.idx + 1000, self.idx + 1001
        d1.eta = d2.eta = 0
        d1.lineage, d2.lineage = self.lineage + [self.idx], self.lineage + [self.idx]

        # --- Rescale sizes (maintaining ~ total size)
        scale_factor = 1 / 2**(1/3)
        scale_factor = np.random.normal(scale_factor, 0.01)
        d1.semi_axes = self.semi_axes * scale_factor
        d2.semi_axes = self.semi_axes * scale_factor

        # --- Divide intensity roughly equally (with some noise)
        if self.raw_int_density is not None:
            intensity_ratio = np.random.normal(0.5, 0.05)
            d1.raw_int_density *= intensity_ratio
            d2.raw_int_density *= (1 - intensity_ratio)

        # --- Calculate new centroids position
        longest_axis_idx = np.argmax(self.semi_axes)
        longest_axis = self.semi_axes[longest_axis_idx]
        # get vertices of the longest axis
        d1.centroid = np.zeros_like(self.centroid)
        d1.centroid[longest_axis_idx] = longest_axis
        d2.centroid = -d1.centroid
        # rotate vertices and translate to mother centroid
        d1.centroid = np.dot(self._get_rotation_matrix(), d1.centroid) + self.centroid
        d2.centroid = np.dot(self._get_rotation_matrix(), d2.centroid) + self.centroid
        
        # --- Remove mother cell from simulation
        self.status = Status.DIVIDED
        
        return d1, d2
    
    def move(self) -> None:
        """Simulate random movement as Brownian motion."""
        # TODO: checks to implement:
        # - make sure nucleus does not exit the cluster (or maybe it could?)
        
        # simulate displacement
        diffusion_coefficient = 10.0  #TODO: move into parameters
        displacements = np.random.normal(
            0, np.sqrt(2 * diffusion_coefficient), len(self.centroid)
        )
        # constraint displacements to stay in the image space
        margin = 10  #TODO: move into parameters
        self.centroid = np.minimum(
            displacements + self.centroid, np.asarray(self.space.size) - margin
        )

    def _update_angle(self, angle: float) -> None:
        """Update orientation of a single angle."""
        rotation_rate = 20  # Degrees per timestep #TODO: move into parameters
        dangle = np.random.normal(0, rotation_rate)
        return (angle + dangle) % 360
    
    def rotate(self) -> None:
        """Simulate random rotation."""
        self.angle_x = self._update_angle(self.angle_x)
        if self.is_3D:
            self.angle_y = self._update_angle(self.angle_y)
            self.angle_z = self._update_angle(self.angle_z)
        
    def update_properties(self) -> None:
        """Update nucleus properties based on size, age, and other factors."""
        # --- Update division probability based on size and age ---
        size_factor = max(
            0, (self._size - self.min_division_size) / self.min_division_size
        )
        age_factor = np.exp(-self.eta / 50)  # Decreases with age
        self.division_prob = 0.1 * size_factor * age_factor  # Base rate * factors

        # --- Update death probability based on age and size ---
        stress_factor = max(0, (self._size - self.max_size) / self.max_size)
        age_factor = self.eta / self.max_age
        self.death_prob = min(0.8, age_factor + stress_factor)

    def update(self) -> Status:
        """Update nucleus properties returning its status after the update.
        
        Returns
        -------
        Status
            Status of the nucleus after the update, i.e., one of `Status.ALIVE`,
            `Status.DEAD`, or `Status.DIVIDED`.
        """
        if self.status != Status.ALIVE:
            return self.status

        # Increment age
        self.eta += 1

        # --- Simulate death ---
        if self.check_death():
            self.die()
            return self.status
        
        # --- Simulate division ---
        if self.check_division():
            self.status = Status.DIVIDED
            # NOTE: division actually performed at cluster/sample level
            return self.status

        # --- Simulate growth ---
        growth_factor = self._calculate_growth_factor()
        self.semi_axes = self.semi_axes * np.sqrt(growth_factor)

        # Update intensity proportionally to area
        if self.raw_int_density is not None:
            self.raw_int_density *= growth_factor

        # --- Simulate random movement (Brownian motion) ---
        self.move()

        # --- Simulate random rotation ---
        self.rotate()

        # --- Update nucleus properties ---
        self.update_properties()

        return self.status
    
    # --- Methods for rendering purpose ---
    # TODO: make it a property?
    def _get_rotation_matrix(self) -> NDArray:
        """Calculate rotation matrix for nucleus orientation."""
        theta = np.radians(self.angle_x)
        if not self.is_3D:
            return np.array([
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]
            ])
        else:
            phi = np.radians(self.angle_y)
            psi = np.radians(self.angle_z)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
            Ry = np.array([
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]
            ])
            Rz = np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]
            ])
            return np.dot(Rz, np.dot(Ry, Rx))
            
    def _compute_ellipsoidal_distances(self) -> NDArray:
        """Calculate distances from nucleus centroid in the ellipsoidal space.
        
        Returns
        -------
        distances : NDArray
            Distances from nucleus centroid in ellipsoidal space for each pixel in the
            coordinate grid.
        """
        # Generate grid of coordinates
        coords = np.mgrid[tuple(slice(0, s) for s in self.space.size)] # shape: (ndims, (Z), Y, X)
        coords = coords.reshape(len(self.space.size), -1)
        
        # Center coordinates
        coords = coords - self.centroid[:, None]
        
        # Rotate coordinates
        coords = np.dot(self._get_rotation_matrix(), coords)
        
        # Normalize coordinates
        coords = coords / self.semi_axes[:, None]
        
        # Calculate distances
        distances = np.sqrt(np.sum(coords ** 2, axis=0))
        
        return distances.reshape(self.space.size)
    
    def render(self) -> NDArray:
        """Render the nucleus as a binary mask in the given space.
        
        Returns
        -------
        mask : NDArray
            Binary mask of the nucleus in the space.
        """
        # Calculate distances from centroid
        distances = self._compute_ellipsoidal_distances()
        
        # Create binary mask
        mask = distances <= 1.0      
        return mask.astype(np.uint8)






    # TODO: we can come with a better way to init this
    # @classmethod
    # def create_from_measurements(
    #     cls,
    #     measurements: dict
    # ) -> "Nucleus":
    #     """Create a nucleus instance from minimal required measurements."""
    #     return cls(
    #         idx=measurements.get('id', np.random.randint(1, 100000)),
    #         label=measurements['Labels'],
    #         timestep=measurements['Time'],
    #         XM=measurements['X'],
    #         YM=measurements['Y'],
    #         Major=measurements['Major'],
    #         Minor=measurements['Minor'],
    #         Angle=measurements['Angle'],
    #         RawIntDen=measurements['RawIntDen']
    #     )    

    
    