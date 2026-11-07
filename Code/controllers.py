# import statements
import numpy as np


# function to calculate repulsion from nearby points
def repulsion(position, points, gain, radius, epsilon):
    offsets = position - points
    distances = np.linalg.norm(offsets, axis=1)
    nearby = distances <= radius
    return np.sum(gain * offsets[nearby] /
                  (distances[nearby, None] ** 3 + epsilon), axis=0)


# function to calculate the baseline forces
def baseline_forces(positions, environment, config):
    forces = np.zeros_like(positions)
    for i, position in enumerate(positions):
        neighbors = np.delete(positions, i, axis=0)
        forces[i] = repulsion(position, neighbors, config.robot_gain,
                              config.robot_interact_r, config.epsilon)
        forces[i] += repulsion(position, environment.obstacle_points,
                               config.obstacle_gain,
                               config.obstacle_interact_r, config.epsilon)
    if not np.isfinite(forces).all():
        raise FloatingPointError('Nonfinite controller force')
    return forces


# function to select the nearest unexplored free cell
def discovery_target(position, x, y, known_free, explored, config):
    distances = np.hypot(x - position[0], y - position[1])
    eligible = known_free & ~explored & (distances <= config.discovery_r)
    if not eligible.any():
        return None
    index = np.argmin(np.where(eligible, distances, np.inf))
    return np.array([x.ravel()[index], y.ravel()[index]])


# function to calculate the force towards the selected cell
def discovery_force(position, target, config):
    if target is None:
        return np.zeros(2)
    offset = target - position
    distance = np.linalg.norm(offset)
    force = config.discovery_gain * offset / max(distance, config.epsilon)
    if not np.isfinite(force).all():
        raise FloatingPointError('Nonfinite discovery force')
    return force
