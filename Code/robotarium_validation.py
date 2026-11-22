# import statements
import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path
import platform
import subprocess
import warnings
import numpy as np
import pandas as pd
from config import Config
from controllers import baseline_forces, discovery_target, discovery_force
from environment import Environment
from robots import initial_positions, limit_vectors
from simulation import update_local_maps
from validation import segment_distances


# function to load the optional official simulator
# source: https://github.com/robotarium/robotarium_python_simulator
def robotarium_api():
    try:
        from rps.robotarium import Robotarium
        from rps.utilities.transformations import create_si_to_uni_mapping
        from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
    except ImportError as error:
        raise ImportError('Install requirements-robotarium.txt to run Robotarium validation') from error

    return Robotarium, create_si_to_uni_mapping, create_si_barrier_certificate_with_boundary


# function to convert project coordinates to Robotarium meters
def to_robotarium(positions, config, scale):
    return scale * (np.asarray(positions) - [config.width / 2, config.height / 2])


# function to convert Robotarium meters to project coordinates
def to_project(positions, config, scale):
    return np.asarray(positions) / scale + [config.width / 2, config.height / 2]


# function to build a separate platform configuration
def platform_config(count=5, seed=598, duration=99.0, layout='open'):
    if layout not in ('open', 'obstacles'):
        raise ValueError('Layout must be open or obstacles')

    config = Config(robot_c=count, seed=seed, duration=duration, dt=0.033,
                    robot_r=0.095 / 0.09, obstacle_gain=0.35)
    if layout == 'open':
        config.obstacles = []
    else:
        config.obstacles = [(9.0, 7.0, 1.5), (11.0, 13.0, 2.0), (15.0, 6.0, 1.3)]

    return config


# function to initialize matched physical poses without random headings from Robotarium
def initial_poses(config, scale=0.09, projection=0.03):
    spacing = 0.25 + 0.02
    start = replace(config, robot_r=max(config.robot_r, spacing / (2 * scale)))
    positions = initial_positions(Environment(start), start)
    headings = np.full(config.robot_c, np.random.default_rng(config.seed + 1).uniform(-np.pi, np.pi))
    return np.vstack((to_robotarium(positions, config, scale).T, headings))


# function to record the installed software and source version
def software_record():
    import rps.robotarium
    package = importlib.metadata.distribution('robotarium-python-simulator')
    direct = json.loads(package.read_text('direct_url.json') or '{}')
    commit = direct.get('vcs_info', {}).get('commit_id')
    url = direct.get('url', '')

    if not commit and url.startswith('file://'):
        source = Path(url[7:])
        if (source / '.git').is_dir():
            commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()

    files = sorted(Path(rps.robotarium.__file__).parent.rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(Path(rps.robotarium.__file__).parent)).encode())
        digest.update(path.read_bytes())

    return {'python': platform.python_version(), 'robotarium': package.version,
            'upstream_commit': commit, 'installed_source_sha256': digest.hexdigest(),
            'cvxopt': importlib.metadata.version('cvxopt'),
            'numpy': np.__version__, 'platform': platform.platform()}


# function to check swept body clearances independently of the command guard
def safety_metrics(poses, environment, config, scale):
    positions = to_project(poses[:, :2, :].transpose(0, 2, 1), config, scale)
    physical = poses[:, :2, :].transpose(0, 2, 1)
    moves = np.diff(physical, axis=0)
    radius = config.robot_r * scale
    minimum = None
    violations = 0

    for i in range(config.robot_c):
        for j in range(i + 1, config.robot_c):
            distances = segment_distances(physical[:-1, i] - physical[:-1, j],
                                          moves[:, i] - moves[:, j])
            value = float(distances.min())
            minimum = value if minimum is None else min(minimum, value)
            violations += int((distances < 2 * radius - 1e-9).sum())

    boundary = float(min(positions.min() - config.robot_r,
                         (config.width - config.robot_r - positions[:, :, 0]).min(),
                         (config.height - config.robot_r - positions[:, :, 1]).min()) * scale)
    obstacle = None
    obstacle_violations = 0

    for x, y, r in config.obstacles:
        center = to_robotarium(np.array([x, y]), config, scale)
        clearances = segment_distances(physical[:-1] - center, moves) - scale * r - radius
        value = float(clearances.min())
        obstacle = value if obstacle is None else min(obstacle, value)
        obstacle_violations += int((clearances < -1e-9).sum())

    return {'minimum_center_separation_m': minimum,
            'minimum_body_boundary_clearance_m': boundary,
            'minimum_body_obstacle_clearance_m': obstacle,
            'robot_collision_intervals': violations,
            'obstacle_collision_intervals': obstacle_violations,
            'boundary_violation': boundary < -1e-9}


# function to run the existing controller through the official unicycle engine
def run_robotarium(config, scale=0.09, projection=0.03, poses=None, control_at_projection=False):
    Robotarium, mapping, barrier_factory = robotarium_api()
    config.validate()
    steps = int(round(config.duration / Robotarium.TIME_STEP))

    if config.dt != Robotarium.TIME_STEP or not np.isclose(steps * config.dt, config.duration):
        raise ValueError('Duration must be a multiple of Robotarium TIME_STEP=0.033 s')
    if config.robot_c < 1 or config.robot_c > 50 or scale <= 0 or projection <= 0:
        raise ValueError('Invalid Robotarium count or coordinate scale')
    if config.robot_r * scale < 0.095 - 1e-12:
        raise ValueError('The platform body radius must cover the native offset collider')

    limits = scale * np.array([-config.width / 2, config.width / 2,
                              -config.height / 2, config.height / 2])
    if np.any(limits[[0, 2]] < Robotarium.BOUNDARIES[[0, 2]]) or np.any(
            limits[[1, 3]] > Robotarium.BOUNDARIES[[1, 3]]):
        raise ValueError('Scaled workspace does not fit the official arena')

    poses = initial_poses(config, scale, projection) if poses is None else np.array(poses, copy=True)
    if poses.shape != (3, config.robot_c) or not np.isfinite(poses).all():
        raise ValueError('Initial poses must be a finite 3 by N array')

    environment = Environment(config)
    guard = Environment(replace(config, robot_r=config.robot_r + 1e-6 / scale))
    positions = to_project(poses[:2].T, config, scale)
    if not environment.valid_positions(positions):
        raise ValueError('Initial poses violate the project body geometry')

    # unused native sensor noise is seeded locally and the caller RNG state is restored
    random_state = np.random.get_state()
    np.random.seed(config.seed)
    import cvxopt.solvers
    options = dict(cvxopt.solvers.options)
    cvxopt.solvers.options.update(feastol=1e-8, reltol=1e-7, abstol=1e-8, maxiters=100)

    try:
        robotarium = Robotarium(number_of_robots=config.robot_c, initial_conditions=poses,
                                show_figure=False, sim_in_real_time=False, skip_initialization=True)
        si_to_uni, uni_to_si = mapping(projection_distance=projection)
        barrier = barrier_factory(safety_radius=0.25, barrier_gain=100.0,
                                  magnitude_limit=0.08, boundary_points=limits)
        known = np.zeros((config.robot_c,) + environment.free.shape, dtype=bool)
        explored = np.zeros_like(known)
        velocity = np.zeros((config.robot_c, 2))
        history = [poses.copy()]
        coverage = [environment.sense(positions)]
        grids = [environment.explored.copy()]
        references, commands, events = [], [], []
        velocity_history = [velocity.copy()]
        targets = []
        state = robotarium.get_poses()

        for step in range(steps):
            positions = to_project(state[:2].T, config, scale)
            control_positions = to_project(uni_to_si(state).T, config, scale) if control_at_projection else positions
            update_local_maps(positions, environment, known, explored, config)
            forces = baseline_forces(control_positions, environment, config)
            selected = np.full((config.robot_c, 2), np.nan)

            if config.controller == 'discovery':
                for i, position in enumerate(control_positions):
                    target = discovery_target(position, environment.x, environment.y, known[i], explored[i], config)
                    forces[i] += discovery_force(position, target, config)
                    if target is not None:
                        selected[i] = target

            acceleration = limit_vectors((forces - config.damping * velocity) / config.mass, config.max_acc)
            velocity = limit_vectors(velocity + config.dt * acceleration, config.max_speed)
            nominal = limit_vectors(scale * velocity, 0.08).T

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                safe = barrier(nominal, uni_to_si(state))
            if not np.isfinite(safe).all():
                raise FloatingPointError('Nonfinite barrier output')

            command = si_to_uni(safe, state)
            wheel_speeds = np.vstack((command[0] - Robotarium.BASE_LENGTH / 2 * command[1],
                                      command[0] + Robotarium.BASE_LENGTH / 2 * command[1]))
            wheel_scale = np.minimum(1.0, Robotarium.MAX_LINEAR_VELOCITY /
                                     np.maximum(np.abs(wheel_speeds).max(axis=0), 1e-15))
            command *= wheel_scale
            moves = config.dt * command[0, :, None] * np.column_stack((np.cos(state[2]), np.sin(state[2])))
            # a one-micrometer body margin avoids rounding through contact
            fraction = guard.safe_fraction(positions, moves / scale)
            command *= fraction
            velocity = safe.T / scale * wheel_scale[:, None] * fraction
            robotarium.set_velocities(np.arange(config.robot_c), command)
            robotarium.step()
            next_state = robotarium.get_poses()
            positions = to_project(next_state[:2].T, config, scale)
            if not environment.valid_positions(positions):
                raise RuntimeError('Robotarium motion left the safe body geometry')

            state = next_state
            history.append(next_state.copy())
            velocity_history.append(velocity.copy())
            coverage.append(environment.sense(positions))
            grids.append(environment.explored.copy())
            references.append(np.concatenate((nominal.T, safe.T), axis=1))
            commands.append(command.T.copy())
            targets.append(selected)
            events.append([bool(np.max(np.abs(safe - nominal)) > 1e-7),
                           bool(np.any(wheel_scale < 1 - 1e-12)), fraction < 1, len(caught)])

        native = dict(robotarium._errors)
    finally:
        np.random.set_state(random_state)
        cvxopt.solvers.options.clear()
        cvxopt.solvers.options.update(options)

    history = np.asarray(history)
    result = {'poses': history, 'coverage': np.asarray(coverage), 'explored': np.asarray(grids),
              'references': np.asarray(references), 'commands': np.asarray(commands),
              'events': np.asarray(events), 'targets': np.asarray(targets),
              'reference_velocities': np.asarray(velocity_history),
              'control_at_projection': control_at_projection,
              'time': np.arange(steps + 1) * config.dt, 'native_errors': native,
              'known_free': known, 'local_explored': explored}
    safety = safety_metrics(history, environment, config, scale)
    if safety['robot_collision_intervals'] or safety['obstacle_collision_intervals'] or safety['boundary_violation']:
        raise RuntimeError('Independent swept safety validation failed: ' + str(safety))

    physical = history[:, :2, :].transpose(0, 2, 1)
    result['summary'] = {'controller': config.controller, 'robot_count': config.robot_c, 'seed': config.seed,
                         'initial_coverage_percent': float(result['coverage'][0]),
                         'final_coverage_percent': float(result['coverage'][-1]),
                         'travel_distance_m': float(np.linalg.norm(np.diff(physical, axis=0), axis=2).sum()),
                         'barrier_intervention_steps': int(result['events'][:, 0].sum()),
                         'wheel_limit_steps': int(result['events'][:, 1].sum()),
                         'swept_guard_steps': int(result['events'][:, 2].sum()),
                         'qp_warning_count': int(result['events'][:, 3].sum()),
                         'native_errors': native, **safety}
    return result


# function to save new platform artifacts without replacing existing outputs
def save_run(result, config, folder, scale, projection):
    folder.mkdir(parents=True, exist_ok=False)
    count = config.robot_c
    poses = result['poses']
    pd.DataFrame({'time_s': np.repeat(result['time'], count), 'robot_id': np.tile(np.arange(count), len(poses)),
                  'x_m': poses[:, 0].ravel(), 'y_m': poses[:, 1].ravel(),
                  'theta_rad': poses[:, 2].ravel()}).to_csv(folder / 'states.csv', index=False)
    pd.DataFrame({'time_s': result['time'], 'coverage_percent': result['coverage']}).to_csv(folder / 'coverage.csv', index=False)
    columns = ['v_m_s', 'omega_rad_s', 'nominal_x_m_s', 'nominal_y_m_s', 'safe_x_m_s', 'safe_y_m_s']
    data = np.concatenate((result['commands'], result['references']), axis=2).reshape(-1, 6)
    table = pd.DataFrame(data, columns=columns)
    table.insert(0, 'robot_id', np.tile(np.arange(count), len(result['commands'])))
    table.insert(0, 'time_s', np.repeat(result['time'][:-1], count))
    table.to_csv(folder / 'inputs.csv', index=False)
    pd.DataFrame(result['events'], columns=['barrier', 'wheel_limit', 'swept_guard', 'qp_warnings']).to_csv(
        folder / 'safety.csv', index=False)
    np.savez_compressed(folder / 'maps.npz', explored=result['explored'], known_free=result['known_free'],
                        local_explored=result['local_explored'], targets=result['targets'])
    parameters = {'configuration': asdict(config), 'scale_m_per_unit': scale,
                  'projection_distance_m': projection, 'initial_poses': poses[0].tolist(),
                  'effective_body_radius_m': config.robot_r * scale,
                  'safety_radius_projected_m': 0.25, 'si_velocity_limit_m_s': 0.08,
                  'swept_guard_margin_m': 1e-6,
                  'control_at_projection': result.get('control_at_projection', False),
                  'physical_workspace_m': (scale * np.array([-config.width / 2, config.width / 2,
                                                           -config.height / 2, config.height / 2])).tolist()}
    for name, data in [('parameters', parameters), ('summary', result['summary'])]:
        (folder / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')


# function to execute matched counts, seeds and obstacle trials
def run_suite(output, duration=99.0, short=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; choose a new Robotarium results folder')

    output.mkdir(parents=True, exist_ok=True)
    record = software_record()
    cases = [(5, 598, 'open')] if short else [(n, 598, 'open') for n in (3, 5, 8)]
    if not short:
        cases += [(5, seed, 'open') for seed in (42, 123, 2026, 7)] + [(5, 598, 'obstacles')]
    rows = []
    for count, seed, layout in cases:
        config = platform_config(count, seed, duration, layout)
        poses = initial_poses(config)
        paired = {}
        for mode in ('baseline', 'discovery'):
            current = replace(config, controller=mode)
            result = run_robotarium(current, poses=poses)
            folder = output / f'{layout}_n{count}_seed{seed}' / mode
            save_run(result, current, folder, 0.09, 0.03)
            row = dict(result['summary'], layout=layout, result_folder=folder.relative_to(output).as_posix())
            rows.append(row)
            paired[mode] = result['poses'][0]
            print(layout, count, seed, mode, '{:.3f}%'.format(row['final_coverage_percent']), flush=True)
        np.testing.assert_array_equal(paired['baseline'], paired['discovery'])
    pd.DataFrame(rows).to_csv(output / 'all_results.csv', index=False)
    seeds = pd.DataFrame(rows)
    seeds = seeds[(seeds.robot_count == 5) & (seeds.layout == 'open')]
    seeds.groupby('controller').agg(mean_coverage=('final_coverage_percent', 'mean'),
                                   sample_sd_coverage=('final_coverage_percent', 'std'),
                                   mean_travel_m=('travel_distance_m', 'mean')).to_csv(
                                       output / 'seed_statistics.csv')
    record.update(duration_s=duration, cases=cases, paired_initial_states=True, local_simulation_only=True)
    (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return pd.DataFrame(rows)


# function to verify saved controller runs without replacing their outputs
def verify_runs(source, destination):
    if destination.exists():
        raise FileExistsError('Verification destination already exists')
    rows = []
    for path in sorted(source.glob('*/parameters.json')):
        parameters = json.loads(path.read_text())
        config = Config(**parameters['configuration'])
        result = run_robotarium(config, scale=parameters['scale_m_per_unit'],
                                projection=parameters['projection_distance_m'],
                                poses=np.asarray(parameters['initial_poses']),
                                control_at_projection=parameters.get('control_at_projection', False))
        states = pd.read_csv(path.parent / 'states.csv', float_precision='round_trip')
        saved = states[['x_m', 'y_m', 'theta_rad']].to_numpy().reshape(-1, config.robot_c, 3).transpose(0, 2, 1)
        np.testing.assert_allclose(result['poses'], saved, rtol=0, atol=1e-10)
        coverage = pd.read_csv(path.parent / 'coverage.csv', float_precision='round_trip').coverage_percent.to_numpy()
        np.testing.assert_array_equal(result['coverage'], coverage)
        inputs = pd.read_csv(path.parent / 'inputs.csv', float_precision='round_trip')
        commands = inputs[['v_m_s', 'omega_rad_s']].to_numpy().reshape(-1, config.robot_c, 2)
        np.testing.assert_allclose(result['commands'], commands, rtol=0, atol=1e-10)
        with np.load(path.parent / 'maps.npz') as maps:
            for key in ('explored', 'known_free', 'local_explored', 'targets'):
                np.testing.assert_allclose(result[key], maps[key], rtol=0, atol=0, equal_nan=True)
        row = {'run': path.parent.name, 'passed': True,
               'maximum_pose_error': float(np.abs(result['poses'] - saved).max()),
               'coverage_percent': float(result['coverage'][-1])}
        rows.append(row)
        print(row, flush=True)
    if not rows:
        raise ValueError('No controller records found in the verification source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'runs': rows, 'software': software_record()}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Official Robotarium validation of coverage and discovery')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'Results/robotarium/experiments')
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--duration', type=float, default=99.0)
    parser.add_argument('--verify', type=Path, help='Paired case folder containing saved baseline and discovery records')
    parser.add_argument('--verification-output', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.verification_output is None:
            parser.error('--verify requires a separate --verification-output')
        verify_runs(args.verify, args.verification_output)
    else:
        run_suite(args.output, duration=3.3 if args.short else args.duration, short=args.short)
