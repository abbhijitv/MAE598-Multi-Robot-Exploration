# import statements
import numpy as np


# function to limit the size of the vectors
def limit_vectors(vectors, limit):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1.0, limit / np.maximum(lengths, 1e-15))


# function to place the robots without overlaps
def initial_positions(environment, config):
    rng = np.random.default_rng(config.seed)
    positions = []
    for attempt in range(10000):
        if len(positions) == config.robot_c:
            return np.asarray(positions)
        if attempt < 5000:
            candidate = rng.uniform([0.12 * config.width, 0.38 * config.height],
                                    [0.28 * config.width, 0.62 * config.height])
        else:
            candidate = rng.uniform(config.robot_r,
                                    [config.width - config.robot_r,
                                     config.height - config.robot_r])
        proposed = np.asarray(positions + [candidate])
        if environment.valid_positions(proposed):
            positions.append(candidate)
    raise ValueError('Could not place robots without overlaps; reduce robot count or radius')


# function to update the robot positions and velocities
def advance(positions, velocities, forces, environment, config, dt):
    acceleration = limit_vectors((forces - config.damping * velocities) / config.mass,
                                 config.max_acc)
    new_velocities = limit_vectors(velocities + acceleration * dt, config.max_speed)
    moves = new_velocities * dt
    # limit the movement before a collision
    fraction = environment.safe_fraction(positions, moves)
    new_positions = positions + fraction * moves
    if fraction < 1:
        new_velocities = np.zeros_like(velocities)
    if not environment.valid_positions(new_positions):
        raise RuntimeError('Motion update produced an invalid configuration')
    return new_positions, new_velocities, fraction < 1
