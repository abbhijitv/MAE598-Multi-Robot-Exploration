# import statements
import numpy as np


# function to calculate the closest distances during movement
def segment_distances(offsets, moves):
    squared_lengths = np.sum(moves * moves, axis=-1)
    fraction = np.clip(-np.sum(offsets * moves, axis=-1)
                       / np.maximum(squared_lengths, 1e-30), 0, 1)
    return np.linalg.norm(offsets + fraction[..., None] * moves, axis=-1)


# function to check the simulation results
def validate_result(result, config):
    positions = result['positions']
    velocities = result['velocities']
    environment = result['environment']
    moves = np.diff(positions, axis=0)
    boundary_clearance = min(float(positions.min() - config.robot_r),
                             float((config.width - config.robot_r - positions[:, :, 0]).min()),
                             float((config.height - config.robot_r - positions[:, :, 1]).min()))
    obstacle_clearance = None
    obstacle_collisions = 0
    for x, y, radius in config.obstacles:
        distances = segment_distances(positions[:-1] - [x, y], moves)
        clearance = distances - radius - config.robot_r
        obstacle_clearance = float(clearance.min()) if obstacle_clearance is None else min(
            obstacle_clearance, float(clearance.min()))
        obstacle_collisions += int((clearance < -1e-9).sum())
    robot_distance = None
    robot_collisions = 0
    for i in range(config.robot_c):
        for j in range(i + 1, config.robot_c):
            distances = segment_distances(positions[:-1, i] - positions[:-1, j],
                                          moves[:, i] - moves[:, j])
            robot_distance = float(distances.min()) if robot_distance is None else min(
                robot_distance, float(distances.min()))
            robot_collisions += int((distances < 2 * config.robot_r - 1e-9).sum())
    speed = np.linalg.norm(velocities, axis=2)
    acceleration_change = np.linalg.norm(np.diff(velocities, axis=0), axis=2)
    allowed_change = config.max_acc * np.diff(result['time'])[:, None] + 1e-10
    normal_steps = ~result['safety_events']
    counts = result['explored'].sum(axis=(1, 2))
    checks = {
        'finite_states': bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),
        'inside_boundaries': boundary_clearance >= -1e-9,
        'swept_obstacle_clearance': obstacle_collisions == 0,
        'swept_robot_separation': robot_collisions == 0,
        'speed_limits': bool((speed <= config.max_speed + 1e-10).all()),
        'acceleration_limits_outside_safety_stops': bool(
            (acceleration_change[normal_steps] <= allowed_change[normal_steps]).all()),
        'coverage_bounds': bool(((result['coverage'] >= 0) & (result['coverage'] <= 100)).all()),
        'exploration_monotonic': bool((np.diff(counts) >= 0).all()),
        'coverage_counts_match': bool(np.allclose(result['coverage'], 100 * counts / environment.free.sum())),
        'obstacle_cells_unexplored': not bool((result['explored'] & ~environment.free).any()),
        'safety_events_match_count': int(result['safety_events'].sum()) == result['safety_interventions']}
    # check the targets using each robots local observations
    if config.controller == 'discovery':
        eligible_targets = True
        private_free = np.zeros_like(result['known_free'])
        private_explored = np.zeros_like(result['local_explored'])
        for step, state in enumerate(positions):
            for i, position in enumerate(state):
                distances = np.hypot(environment.x - position[0], environment.y - position[1])
                private_free[i] |= environment.free & (distances <= config.discovery_r)
                private_explored[i] |= environment.free & (distances <= config.sensing_r)
                if step == len(positions) - 1:
                    continue
                target = result['targets'][step, i]
                eligible = private_free[i] & ~private_explored[i] & (distances <= config.discovery_r)
                if np.isnan(target).all():
                    eligible_targets &= not eligible.any()
                elif not np.isfinite(target).all():
                    eligible_targets = False
                else:
                    index = np.argmin((environment.x - target[0]) ** 2 + (environment.y - target[1]) ** 2)
                    eligible_targets &= bool(eligible.ravel()[index])
                    eligible_targets &= bool(np.allclose([environment.x.ravel()[index],
                                                         environment.y.ravel()[index]], target))
        checks['targets_locally_known_free_and_privately_unexplored'] = bool(eligible_targets)
        checks['private_map_history_matches'] = bool(
            np.array_equal(private_free, result['known_free'])
            and np.array_equal(private_explored, result['local_explored']))
        checks['union_private_exploration_matches_evaluation'] = bool(
            np.array_equal(private_explored.any(axis=0), environment.explored))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError('Validation failed: ' + ', '.join(failed))
    return {'checks': checks, 'time_samples': len(positions),
            'minimum_boundary_clearance': boundary_clearance,
            'minimum_swept_obstacle_clearance': obstacle_clearance,
            'minimum_swept_robot_center_distance': robot_distance,
            'robot_collision_intervals': robot_collisions,
            'obstacle_collision_intervals': obstacle_collisions,
            'safety_interventions': result['safety_interventions'],
            'safety_intervention_end_times_s': result['time'][1:][result['safety_events']].tolist()}
