# import statements
import numpy as np
from controllers import baseline_forces, discovery_target, discovery_force
from environment import Environment
from robots import initial_positions, advance


# function to run the robot simulation
def run_simulation(config, positions=None, velocities=None):
    config.validate()
    environment = Environment(config)
    if positions is None:
        positions = initial_positions(environment, config)
    positions = np.array(positions, dtype=float, copy=True)
    if positions.shape != (config.robot_c, 2) or not environment.valid_positions(positions):
        raise ValueError('Initial positions must be finite, inside free space, and separated')
    if velocities is None:
        velocities = np.zeros_like(positions)
    velocities = np.array(velocities, dtype=float, copy=True)
    if velocities.shape != positions.shape or not np.isfinite(velocities).all():
        raise ValueError('Invalid initial velocities')
    if np.any(np.linalg.norm(velocities, axis=1) > config.max_speed):
        raise ValueError('Initial velocities exceed max_speed')
    steps = int(np.ceil(config.duration / config.dt))
    times = np.minimum(np.arange(steps + 1) * config.dt, config.duration)
    history = [positions.copy()]
    velocity_history = [velocities.copy()]
    coverage = [environment.sense(positions)]
    explored = [environment.explored.copy()]
    interventions = 0
    safety_events = []
    # private maps for each robot
    known_free = np.zeros((config.robot_c,) + environment.free.shape, dtype=bool)
    local_explored = np.zeros_like(known_free)
    targets = []
    if config.controller == 'discovery':
        update_local_maps(positions, environment, known_free, local_explored, config)
    # calculate the forces and update the robot movement
    for dt in np.diff(times):
        forces = baseline_forces(positions, environment, config)
        selected = np.full_like(positions, np.nan)
        if config.controller == 'discovery':
            for i, position in enumerate(positions):
                target = discovery_target(position, environment.x, environment.y,
                                          known_free[i], local_explored[i], config)
                forces[i] += discovery_force(position, target, config)
                if target is not None:
                    selected[i] = target
        targets.append(selected)
        positions, velocities, blocked = advance(positions, velocities, forces,
                                                 environment, config, dt)
        interventions += int(blocked)
        safety_events.append(bool(blocked))
        if config.controller == 'discovery':
            update_local_maps(positions, environment, known_free, local_explored, config)
        history.append(positions.copy())
        velocity_history.append(velocities.copy())
        coverage.append(environment.sense(positions))
        explored.append(environment.explored.copy())
    return {'time': times, 'positions': np.asarray(history),
            'velocities': np.asarray(velocity_history), 'coverage': np.asarray(coverage),
            'explored': np.asarray(explored), 'environment': environment,
            'safety_interventions': interventions,
            'safety_events': np.asarray(safety_events), 'targets': np.asarray(targets),
            'known_free': known_free, 'local_explored': local_explored}


# function to update the local free and explored cells
def update_local_maps(positions, environment, known_free, explored, config):
    for i, position in enumerate(positions):
        distance_squared = ((environment.x - position[0]) ** 2
                            + (environment.y - position[1]) ** 2)
        known_free[i] |= environment.free & (distance_squared <= config.discovery_r ** 2)
        explored[i] |= environment.free & (distance_squared <= config.sensing_r ** 2)

