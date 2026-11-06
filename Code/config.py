
# import statements
from dataclasses import dataclass, field
import math

# config class
@dataclass
class Config:
    # environment dimensions and coverage grid
    width: float = 20.0
    height: float = 20.0
    grid_size: float= 0.25

    # robot count and their size and sensing range
    robot_c: int = 5
    robot_r: float = 0.2
    sensing_r: float = 2.5

    # repulsion settings
    robot_interact_r: float = 4.0
    robot_gain: float = 2.0
    obstacle_interact_r: float = 2.0
    obstacle_gain: float = 0.50

    # movement of the robots
    mass: float = 1.0
    damping: float = 1.5
    max_speed: float = 1.5
    max_acc: float = 3.0
    epsilon: float = 1e-6

    # simulation settings and obstacle boundary sampling
    duration: float = 60.0
    dt: float = 0.05
    seed: int = 600
    obstacle_boundary_sampling: float = 0.5
    
    # discovery controller
    controller: str = 'baseline'
    discovery_gain: float = 1.2
    discovery_r: float = 5.0

    # circular obstacles
    obstacles: list = field(default_factory = lambda: [
        (9.0, 7.0, 1.5),
        (11.0, 13.0, 1.5),
        (15.0, 5.0, 1.5)
    ])

    # validation function
    def validate(self):
        if not math.isfinite(self.width) or not math.isfinite(self.height) or self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive and finite.")
        if not math.isfinite(self.grid_size) or self.grid_size <= 0:
            raise ValueError("Grid size must be positive and finite.")


        if not isinstance(self.robot_c, int):
            raise TypeError("Robot count must be an integer.")
        if self.robot_c <= 0:
            raise ValueError("Robot count must be positive.")
        if not math.isfinite(self.robot_r) or self.robot_r <= 0:
            raise ValueError("Robot radius must be positive and finite.")
        if not math.isfinite(self.sensing_r) or self.sensing_r <= 0:
            raise ValueError("Sensing radius must be positive and finite.")


        if not math.isfinite(self.robot_interact_r) or self.robot_interact_r <= 0:
            raise ValueError("Robot interaction radius must be positive and finite.")
        if not math.isfinite(self.robot_gain) or self.robot_gain < 0:
            raise ValueError("Robot gain must be nonnegative and finite.")
        if not math.isfinite(self.obstacle_interact_r) or self.obstacle_interact_r <= 0:
            raise ValueError("Obstacle interaction radius must be positive and finite.")
        if not math.isfinite(self.obstacle_gain) or self.obstacle_gain < 0:
            raise ValueError("Obstacle gain must be nonnegative and finite.")


        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Mass must be positive and finite.")
        if not math.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("Damping must be positive and finite.")
        if not math.isfinite(self.max_speed) or self.max_speed <= 0:
            raise ValueError("Maximum speed must be positive and finite.")
        if not math.isfinite(self.max_acc) or self.max_acc <= 0:
            raise ValueError("Maximum acceleration must be positive and finite.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Epsilon must be positive and finite.")


        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Duration must be positive and finite.")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Time step must be positive and finite.")
        if not math.isfinite(self.obstacle_boundary_sampling) or self.obstacle_boundary_sampling <= 0:
            raise ValueError("Obstacle boundary sampling must be positive and finite.")


        if self.controller not in ('baseline', 'discovery'):
            raise ValueError("Invalid controller.")
        if not math.isfinite(self.discovery_gain) or self.discovery_gain < 0:
            raise ValueError("Discovery gain must be nonnegative and finite.")
        if not math.isfinite(self.discovery_r) or self.discovery_r <= 0:
            raise ValueError("Discovery radius must be positive and finite.")


        if 2 * self.robot_r >= min(self.width, self.height):
            raise ValueError("Robot radius is too large for the given width and height.")
        if self.dt * self.damping / self.mass > 1:
            raise ValueError("Invalid time step or system parameters.")

        
        for x, y, r in self.obstacles:
            if not all(math.isfinite(val) for val in (x, y, r)):
                raise ValueError("Obstacle coordinates and radius must be finite.")
            if r <= 0:
                raise ValueError("Obstacle radius must be positive.")
            if not (r <= x <= self.width - r and r <= y <= self.height - r):
                raise ValueError("Obstacle is outside the environment bounds.")
