# import statements
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from config import Config
from controllers import baseline_forces
from environment import Environment


# function to calculate the potential for the repulsive force
def radial_potential(distance, gain, radius, epsilon):
    if distance >= radius:
        return 0.0
    return quad(lambda s: gain * s / (s ** 3 + epsilon), distance, radius,
                epsabs=1e-12, epsrel=1e-12)[0]


# function to calculate the total potential energy
def potential_energy(positions, environment, config):
    energy = 0.0
    for i, position in enumerate(positions):
        for other in positions[i + 1:]:
            energy += radial_potential(np.linalg.norm(position - other), config.robot_gain,
                                       config.robot_interact_r, config.epsilon)
        for point in environment.obstacle_points:
            energy += radial_potential(np.linalg.norm(position - point), config.obstacle_gain,
                                       config.obstacle_interact_r, config.epsilon)
    return energy


# function to calculate the potential hessian
def potential_hessian(position, points, gain, epsilon):
    hessian = np.zeros((2, 2))
    for point in points:
        offset = position - point
        distance = np.linalg.norm(offset)
        denominator = distance ** 3 + epsilon
        hessian += gain * (3 * distance * np.outer(offset, offset) / denominator ** 2
                           - np.eye(2) / denominator)
    return hessian


# square environment and equilibrium used in the analysis
def equilibrium_case():
    config = Config(width=3.0, height=3.0, robot_c=1, obstacles=[],
                    duration=20.0, dt=0.01, obstacle_gain=0.35)
    environment = Environment(config)
    equilibrium = np.array([1.5, 1.5])
    distances = np.linalg.norm(equilibrium - environment.obstacle_points, axis=1)
    active = distances < config.obstacle_interact_r
    points = environment.obstacle_points[active]
    hessian = potential_hessian(equilibrium, points, config.obstacle_gain, config.epsilon)
    radius = 0.01
    bound = 30 * config.obstacle_gain * len(points) / (distances[active].min() - radius) ** 4
    certificate = {'equilibrium': equilibrium.tolist(), 'velocity': [0.0, 0.0],
                   'active_wall_points': points.tolist(), 'active_point_count': len(points),
                   'hessian': hessian.tolist(), 'hessian_eigenvalues': np.linalg.eigvalsh(hessian).tolist(),
                   'equilibrium_force_norm': float(np.linalg.norm(
                       baseline_forces(equilibrium[None, :], environment, config))),
                   'position_ball_radius': radius,
                   'cutoff_distance_margin_at_center': float(np.abs(distances - config.obstacle_interact_r).min()),
                   'hessian_lipschitz_upper_bound_computed': float(bound),
                   'proved_hessian_lower_bound': 0.36, 'proved_hessian_upper_bound': 1.47,
                   'invariant_sublevel_energy': 1.2e-5,
                   'position_boundary_energy_lower_bound': 1.8e-5,
                   'proof_source': 'Research_Notes/Step_15_Theoretical_Analysis.md'}
    return config, environment, equilibrium, certificate


# function to integrate the fixed field robot dynamics
def fixed_field_trajectory(config, environment, position, velocity):
    times = np.linspace(0, config.duration, int(round(config.duration / config.dt)) + 1)

    # function to calculate velocity and acceleration for the fixed field
    def dynamics(time, state):
        force = baseline_forces(state[:2][None, :], environment, config)[0]
        return np.concatenate((state[2:], (force - config.damping * state[2:]) / config.mass))

    solution = solve_ivp(dynamics, (0, config.duration), np.concatenate((position, velocity)),
                         method='DOP853', t_eval=times, rtol=1e-11, atol=1e-13, max_step=0.02)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[:2].T, solution.y[2:].T


# function to run the theoretical validation
def run_validation(output):
    output = Path(output)
    results = Path(__file__).resolve().parents[1] / 'Results'
    if results not in output.resolve().parents:
        raise ValueError('Theoretical outputs must be a subfolder of Results')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; choose a new folder')
    config, environment, equilibrium, certificate = equilibrium_case()
    offset = np.array([0.003, -0.002])
    initial_velocity = np.array([0.001, -0.0005])
    initial_upper = (0.5 * certificate['proved_hessian_upper_bound'] * np.dot(offset, offset)
                     + 0.5 * config.mass * np.dot(initial_velocity, initial_velocity))
    if initial_upper >= certificate['invariant_sublevel_energy']:
        raise ValueError('Initial state is outside the proved sublevel bound')
    times, positions, velocities = fixed_field_trajectory(
        config, environment, equilibrium + offset, initial_velocity)
    # potential energy, kinetic energy and damping loss
    potential = np.array([potential_energy(p[None, :], environment, config) for p in positions])
    potential_at_equilibrium = potential_energy(equilibrium[None, :], environment, config)
    kinetic = 0.5 * config.mass * np.sum(velocities ** 2, axis=1)
    total = potential + kinetic
    relative = total - potential_at_equilibrium
    speed = np.linalg.norm(velocities, axis=1)
    distance = np.linalg.norm(positions - equilibrium, axis=1)
    rate = -config.damping * speed ** 2
    dissipated = cumulative_trapezoid(-rate, times, initial=0.0)
    active = np.linalg.norm(equilibrium - environment.obstacle_points, axis=1) <= config.obstacle_interact_r
    unchanged = all(np.array_equal(
        np.linalg.norm(p - environment.obstacle_points, axis=1) <= config.obstacle_interact_r, active)
                    for p in positions)
    acceleration = np.array([(baseline_forces(p[None, :], environment, config)[0]
                              - config.damping * v) / config.mass
                             for p, v in zip(positions, velocities)])
    checks = {'finite_states_and_energy': bool(np.isfinite(total).all() and np.isfinite(positions).all()
                                               and np.isfinite(velocities).all()),
              'inside_proved_position_ball': bool(distance.max() < certificate['position_ball_radius']),
              'fixed_active_set': bool(unchanged),
              'collision_free': all(environment.valid_positions(p[None, :]) for p in positions),
              'energy_nonincrease_with_roundoff_tolerance': bool(np.diff(total).max() < 1e-11),
              'numerical_energy_balance_within_1e_9': bool(np.abs(total - total[0] + dissipated).max() < 1e-9),
              'initial_energy_inside_proved_sublevel': bool(relative[0] < certificate['invariant_sublevel_energy']),
              'observed_recovery': bool(distance[-1] < distance[0] * 1e-4 and speed[-1] < 1e-7)}
    if not all(checks.values()):
        raise RuntimeError('Theoretical validation failed: ' + str(checks))
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({'time_s': times, 'x': positions[:, 0], 'y': positions[:, 1],
                          'vx': velocities[:, 0], 'vy': velocities[:, 1],
                          'potential_energy': potential, 'kinetic_energy': kinetic,
                          'total_energy': total, 'relative_mechanical_energy': relative,
                          'energy_derivative': rate, 'integrated_dissipation': dissipated,
                          'mean_speed': speed, 'maximum_speed': speed,
                          'distance_to_equilibrium': distance})
    table.to_csv(output / 'fixed_field_history.csv', index=False)
    parameters = {'configuration': asdict(config), 'method': 'DOP853',
                  'rtol': 1e-11, 'atol': 1e-13, 'maximum_internal_step_s': 0.02,
                  'output_interval_s': config.dt, 'initial_position': positions[0].tolist(),
                  'initial_velocity': initial_velocity.tolist(),
                  'model': 'Continuous fixed-field baseline ODE; no caps or safety resets applied',
                  'energy': 'Regularized radial-potential quadrature plus kinetic energy',
                  'quadrature_absolute_and_relative_tolerances': 1e-12}
    summary = {'checks': checks, 'initial_energy_upper_bound': float(initial_upper),
               'initial_relative_energy': float(relative[0]), 'final_relative_energy': float(relative[-1]),
               'maximum_energy_increment': float(np.diff(total).max()),
               'maximum_energy_balance_error': float(np.abs(total - total[0] + dissipated).max()),
               'initial_distance_to_equilibrium': float(distance[0]),
               'final_distance_to_equilibrium': float(distance[-1]),
               'maximum_distance_to_equilibrium': float(distance.max()),
               'final_speed': float(speed[-1]), 'maximum_speed': float(speed.max()),
               'maximum_acceleration': float(np.linalg.norm(acceleration, axis=1).max()),
               'minimum_robot_body_wall_clearance': float(min(positions.min(), (3 - positions).min())
                                                          - config.robot_r)}
    for name, data in [('parameters', parameters), ('equilibrium_certificate', certificate), ('summary', summary)]:
        (output / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    plots = [('energy_vs_time.png', relative, 'Relative mechanical energy U - U* + T',
              'Fixed conservative ODE: energy dissipation'),
             ('speed_vs_time.png', speed, 'Speed (units/s); mean = maximum for one robot',
              'Fixed conservative ODE: speed decay'),
             ('distance_to_equilibrium.png', distance, 'Distance to (1.5, 1.5) (units)',
              'Recovery near the verified isolated equilibrium')]
    for name, values, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, values, label='Perturbed fixed-field trajectory')
        ax.set(xlabel='Time (s)', ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(positions[:, 0] - equilibrium[0], positions[:, 1] - equilibrium[1], label='Recovery path')
    ax.scatter(*offset, marker='x', color='orange', label='Perturbed initial position')
    ax.scatter(0, 0, marker='*', color='black', s=90, label='Verified equilibrium')
    ax.set(xlabel='x - 1.5 (units)', ylabel='y - 1.5 (units)',
           title='Local perturbation and recovery', aspect='equal')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'perturbation_recovery.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed conservative model validation for Step 15')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'Results' / 'theoretical_validation')
    run_validation(parser.parse_args().output)
