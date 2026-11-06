
# import statements
import numpy as np

# environment class
class Environment:
    def __init__(self, config):
        self.config = config

        # environment grid
        gx = int(np.ceil(self.config.width / self.config.grid_size))
        gy = int(np.ceil(self.config.height / self.config.grid_size))
        self.x_edges = np.linspace(0, self.config.width, gx + 1)
        self.y_edges = np.linspace(0, self.config.height, gy + 1)
        self.x, self.y = np.meshgrid((self.x_edges[1:] + self.x_edges[:-1]) /2, (self.y_edges[1:] + self.y_edges[:-1]) /2)
        self.free = np.ones(self.x.shape, dtype=bool)

        # obstacle boundaries
        points = []
        for x, y, r in config.obstacles:
            self.free &= (self.x -x) ** 2 + (self.y - y) ** 2 > r ** 2
            count = max(12, int(np.ceil(2 * np.pi * r / config.obstacle_boundary_sampling)))
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            points.extend(np.column_stack((x + r * np.cos(angles), y + r * np.sin(angles))))

        # add the points along the environment walls
        spacing = config.obstacle_boundary_sampling
        x_wall_points = np.linspace(0, self.config.width, int(np.ceil(self.config.width / spacing)) + 1)
        y_wall_points = np.linspace(0, self.config.height, int(np.ceil(self.config.height / spacing)) + 1)
        points.extend((x,y) for x in x_wall_points for y in (0, self.config.height))
        points.extend((x,y) for y in y_wall_points[1:-1] for x in (0, self.config.width))
        self.obstacle_points = np.asarray(points)

        self.explored = np.zeros_like(self.free)
        if not self.free.any():
            raise ValueError("No free space available in the environment.")

    # validate function to check if the robots position is valid
    def valid_positions(self, positions):
        if not np.isfinite(positions).all():
            return False
        if np.any(positions < self.config.robot_r) or np.any(positions > [self.config.width - self.config.robot_r, self.config.height - self.config.robot_r]):
            return False
        for x, y, r in self.config.obstacles:
            if np.any(np.linalg.norm(positions - np.array([x, y]), axis=1) < r + self.config.robot_r):
                return False
        for i in range(len(positions)):
            if np.any(np.linalg.norm(positions[i + 1:] - positions[i], axis=1) < 2 * self.config.robot_r):
                return False
        return True


    # function to update the explored cells based on the robot sensing range
    def sense(self, positions):
        for x, y in positions:
            self.explored |= self.free & ((self.x -x) ** 2 + (self.y - y) ** 2 <= self.config.sensing_r ** 2)
        return 100.0 * self.explored.sum() / self.free.sum()


    # function to calculate how far the robots move without collisions
    def safe_fraction(self, positions, moves):
        fraction = 1.0
        for p, move in zip(positions, moves):
            for axis, limit in enumerate((self.config.width, self.config.height)):
                if move[axis] > 0:
                    fraction = min(fraction, (limit - self.config.robot_r - p[axis]) / move[axis])
                elif move[axis] < 0:
                    fraction = min(fraction, (self.config.robot_r - p[axis]) / move[axis])
            for x, y, r in self.config.obstacles:
                fraction = min(fraction, contact_fraction(p - [x, y], move, r + self.config.robot_r))


        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                # Check for collisions between robots
                fraction = min(fraction, contact_fraction(positions[i] - positions[j], moves[i] - moves[j], 2 * self.config.robot_r))
        if fraction < 1:
            return max(0.0, fraction * (1 - 1e-10))
        return 1.0


# function to calculate the fraction of the movement before the contact
def contact_fraction(offset, move, radius):
    # Implementation for calculating contact fraction
    a = np.dot(move, move)
    b = np.dot(offset, move)
    if a == 0 or b >= 0:
        return 1.0
    c = np.dot(offset, offset) - radius ** 2
    discriminant = b ** 2 - a * c
    if discriminant <= 0:
        return 1.0
    return min(1.0, max(0.0, c / (-b + np.sqrt(discriminant))))
