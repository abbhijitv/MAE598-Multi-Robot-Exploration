# import statements
import argparse
import json
import math
import subprocess
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, cumulative_trapezoid
from config import Config
from controllers import baseline_forces
from environment import Environment
from theoretical_analysis import potential_energy, potential_hessian


# rational intervals for the equilibrium certificate
class Interval:
    # function to store exact rational endpoints
    def __init__(self, lower, upper=None):
        self.lower = Fraction(str(lower))
        self.upper = self.lower if upper is None else Fraction(str(upper))

        if self.lower > self.upper:
            raise ValueError('Interval endpoints are reversed')

    # function to add intervals
    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    # function to negate an interval
    def __neg__(self):
        return Interval(-self.upper, -self.lower)

    # function to subtract intervals
    def __sub__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return self + (-other)

    # function to multiply intervals
    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        values = [a * b for a in (self.lower, self.upper)
                  for b in (other.lower, other.upper)]
        return Interval(min(values), max(values))

    __rmul__ = __mul__

    # function to divide intervals
    def __truediv__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)

        if other.lower <= 0 <= other.upper:
            raise ValueError('Interval division includes zero')

        return self * Interval(1 / other.upper, 1 / other.lower)

    # function to divide a number by an interval
    def __rtruediv__(self, other):
        return Interval(other) / self

    # function to square an interval
    def square(self):
        values = [self.lower ** 2, self.upper ** 2]
        lower = 0 if self.lower <= 0 <= self.upper else min(values)
        return Interval(lower, max(values))

    # function to bound a square root with rational endpoints
    def sqrt(self):
        if self.lower < 0:
            raise ValueError('Square root includes negative values')

        scale = 10 ** 24
        lower = math.isqrt(self.lower.numerator * scale ** 2 // self.lower.denominator)
        upper = math.isqrt(self.upper.numerator * scale ** 2 // self.upper.denominator) + 1
        return Interval(Fraction(lower, scale), Fraction(upper, scale))

    # function to export outward rounded bounds
    def bounds(self):
        return [float(np.nextafter(float(self.lower), -np.inf)),
                float(np.nextafter(float(self.upper), np.inf))]


# function to bound the force and wall hessian at the symmetric positions
def symmetric_bounds(a, environment, config):
    force = Interval(0)
    hxx, hyy = Interval(0), Interval(0)
    distances = []
    active = []

    for point in environment.obstacle_points:
        dx, dy = a - point[0], Interval(1) - point[1]
        distance = (dx.square() + dy.square()).sqrt()
        distances.append(distance)

        if distance.upper < config.obstacle_interact_r:
            active.append(True)
            denominator = distance * distance.square() + config.epsilon
            force += config.obstacle_gain * dx / denominator
            hxx += config.obstacle_gain * (
                3 * distance * dx.square() / denominator.square() - Interval(1) / denominator)
            hyy += config.obstacle_gain * (
                3 * distance * dy.square() / denominator.square() - Interval(1) / denominator)
        elif distance.lower > config.obstacle_interact_r:
            active.append(False)
        else:
            raise ValueError('Equilibrium interval crosses a wall cutoff')

    gap = Interval(3) - 2 * a
    denominator = gap * gap.square() + config.epsilon
    force -= config.robot_gain * gap / denominator
    radial = config.robot_gain * (2 * gap * gap.square() - config.epsilon) / denominator.square()
    tangent = -config.robot_gain * Interval(1) / denominator
    eigenvalues = [hxx, hyy, hxx + 2 * radial, hyy + 2 * tangent]

    return force, eigenvalues, distances, active, gap


# function to calculate the full interacting robot hessian
def collective_hessian(positions, environment, config):
    hessian = np.zeros((4, 4))

    for i, position in enumerate(positions):
        active = np.linalg.norm(environment.obstacle_points - position, axis=1) < config.obstacle_interact_r
        hessian[2 * i:2 * i + 2, 2 * i:2 * i + 2] = potential_hessian(
            position, environment.obstacle_points[active], config.obstacle_gain, config.epsilon)

    if np.linalg.norm(positions[0] - positions[1]) < config.robot_interact_r:
        pair = potential_hessian(positions[0], positions[1:], config.robot_gain, config.epsilon)
        hessian[:2, :2] += pair
        hessian[2:, 2:] += pair
        hessian[:2, 2:] -= pair
        hessian[2:, :2] -= pair

    return hessian


# function to certify the two robot equilibrium and its neighborhood
def equilibrium_case():
    config = Config(width=3.0, height=2.0, robot_c=2, obstacles=[],
                    duration=20.0, dt=0.01, obstacle_gain=0.35)
    environment = Environment(config)
    bracket = Interval('0.770124', '0.770126')

    # interval bisection keeps a strict force sign at both ends
    for step in range(20):
        middle = (bracket.lower + bracket.upper) / 2
        force = symmetric_bounds(Interval(middle), environment, config)[0]

        if force.lower > 0:
            bracket = Interval(middle, bracket.upper)
        elif force.upper < 0:
            bracket = Interval(bracket.lower, middle)
        else:
            break

    left_force = symmetric_bounds(Interval(bracket.lower), environment, config)[0]
    right_force = symmetric_bounds(Interval(bracket.upper), environment, config)[0]
    force, eigenvalues, distances, active, gap = symmetric_bounds(bracket, environment, config)
    radius = Interval('0.001')
    sqrt_two = Interval(2).sqrt()
    bound = Interval(0)

    for distance, included in zip(distances, active):
        if included:
            minimum = Interval(distance.lower) - radius
            bound += 2 * 30 * config.obstacle_gain / minimum.square().square()

    minimum_pair = Interval(gap.lower) - sqrt_two * radius
    bound += 2 * sqrt_two * 30 * config.robot_gain / minimum_pair.square().square()
    lipschitz = math.ceil(bound.upper)
    alpha = Fraction(math.floor((min(v.lower for v in eigenvalues)
                                - lipschitz * radius.upper) * 1000), 1000)
    beta = Fraction(math.ceil((max(v.upper for v in eigenvalues)
                              + lipschitz * radius.upper) * 1000), 1000)
    cutoff_margin = min((Interval(config.obstacle_interact_r) - d).lower
                        if included else (d - config.obstacle_interact_r).lower
                        for d, included in zip(distances, active))
    body_wall_clearance = min(bracket.lower, Fraction(1)) - radius.upper - Fraction('0.2')
    body_pair_clearance = minimum_pair.lower - Fraction('0.4')
    pair_cutoff_margin = (Interval(config.robot_interact_r) - gap - sqrt_two * radius).lower
    sublevel = alpha * radius.lower ** 2 / 4
    boundary = alpha * radius.lower ** 2 / 2

    checks = {'strict_root_bracket': left_force.lower > 0 and right_force.upper < 0,
              'positive_collective_hessian': all(v.lower > 0 for v in eigenvalues),
              'positive_hessian_on_position_ball': alpha > 0,
              'wall_active_sets_fixed_on_ball': cutoff_margin > radius.upper,
              'robot_pair_active_on_ball': pair_cutoff_margin > 0,
              'collision_free_position_ball': body_wall_clearance > 0 and body_pair_clearance > 0,
              'strict_invariant_sublevel': 0 < sublevel < boundary}

    if not all(checks.values()):
        raise ValueError('Equilibrium certificate failed: ' + str(checks))

    a = float((bracket.lower + bracket.upper) / 2)
    equilibrium = np.array([[a, 1.0], [3 - a, 1.0]])
    left_error = Interval(Fraction.from_float(a)) - bracket
    right_error = Interval(Fraction.from_float(float(equilibrium[1, 0]))) - (Interval(3) - bracket)
    root_error = (left_error.square() + right_error.square()).sqrt()
    certificate = {'checks': checks, 'robot_count': 2,
                   'equilibrium': equilibrium.tolist(),
                   'root_interval': bracket.bounds(),
                   'root_interval_exact': [str(bracket.lower), str(bracket.upper)],
                   'left_endpoint_force_interval': left_force.bounds(),
                   'right_endpoint_force_interval': right_force.bounds(),
                   'root_force_interval': force.bounds(),
                   'equilibrium_representation_error_upper_bound': root_error.bounds()[1],
                   'collective_eigenvalue_intervals': [v.bounds() for v in eigenvalues],
                   'collective_hessian': collective_hessian(equilibrium, environment, config).tolist(),
                   'equilibrium_force_norm': float(np.linalg.norm(
                       baseline_forces(equilibrium, environment, config))),
                   'active_wall_points_robot_1': environment.obstacle_points[active].tolist(),
                   'active_wall_point_count_per_robot': sum(active),
                   'active_robot_pairs': [[0, 1]],
                   'position_ball_radius': float(radius.lower),
                   'hessian_lipschitz_upper_bound': lipschitz,
                   'proved_hessian_lower_bound': float(alpha),
                   'proved_hessian_upper_bound': float(beta),
                   'invariant_sublevel_energy': float(sublevel),
                   'position_boundary_energy_lower_bound': float(boundary),
                   'wall_cutoff_margin_at_equilibrium_lower_bound': float(cutoff_margin),
                   'pair_cutoff_margin_on_ball_lower_bound': float(pair_cutoff_margin),
                   'body_wall_clearance_on_ball_lower_bound': float(body_wall_clearance),
                   'body_pair_clearance_on_ball_lower_bound': float(body_pair_clearance),
                   'arithmetic': 'Exact Fraction operations; square roots enclosed to 24 decimal places; '
                                 'exported interval endpoints rounded outward',
                   'proof_source': 'Results/theoretical_validation_multi_robot/mathematical_derivation.md'}

    return config, environment, equilibrium, certificate


# function to integrate the continuous interacting robot dynamics
def fixed_field_trajectory(config, environment, positions, velocities):
    times = np.linspace(0, config.duration, int(round(config.duration / config.dt)) + 1)

    # function to calculate velocity and acceleration without caps or resets
    def dynamics(time, state):
        positions = state[:4].reshape(2, 2)
        velocities = state[4:].reshape(2, 2)
        acceleration = (baseline_forces(positions, environment, config)
                        - config.damping * velocities) / config.mass

        return np.concatenate((velocities.ravel(), acceleration.ravel()))

    solution = solve_ivp(dynamics, (0, config.duration),
                         np.concatenate((positions.ravel(), velocities.ravel())),
                         method='DOP853', t_eval=times, rtol=1e-11, atol=1e-13, max_step=0.02)

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.t, solution.y[:4].T.reshape(-1, 2, 2), solution.y[4:].T.reshape(-1, 2, 2)


# function to check robot and wall interaction sets
def active_sets(positions, environment, config):
    walls = np.linalg.norm(positions[:, None, :] - environment.obstacle_points, axis=2)
    pair = np.linalg.norm(positions[0] - positions[1]) <= config.robot_interact_r

    return walls <= config.obstacle_interact_r, pair


# function to draw the workspace and local recovery paths
def recovery_axes(equilibrium, config):
    fig = plt.figure(figsize=(8, 8), dpi=120)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    workspace = fig.add_subplot(grid[0, :])
    local = [fig.add_subplot(grid[1, i]) for i in range(2)]
    workspace.set(xlim=(0, config.width), ylim=(0, config.height), aspect='equal',
                  xlabel='x (units)', ylabel='y (units)', title='Two interacting baseline robots')
    workspace.plot(equilibrium[:, 0], equilibrium[:, 1], '--', color='gray',
                   linewidth=1, label='Active robot interaction')

    for i, (position, ax) in enumerate(zip(equilibrium, local)):
        workspace.scatter(*position, marker='*', color='black', s=55, zorder=5)
        ax.scatter(0, 0, marker='*', color='black', s=65, label='Equilibrium')
        ax.set(xlim=(-1.6e-4, 1.6e-4), ylim=(-1.6e-4, 1.6e-4), aspect='equal',
               xlabel='x - x* (units)', ylabel='y - y* (units)',
               title='Robot ' + str(i + 1) + ': local displacement')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax.grid(alpha=0.25)

    workspace.grid(alpha=0.25)
    workspace.legend(loc='upper right', fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))

    return fig, workspace, local


# function to render and validate the continuous model video
def render_video(path, times, positions, equilibrium, relative, config):
    fig, workspace, local = recovery_axes(equilibrium, config)
    bodies, trails, markers = [], [], []

    for i, ax in enumerate(local):
        color = 'C' + str(i)
        body = Circle(positions[0, i], config.robot_r, color=color, zorder=4)
        workspace.add_patch(body)
        bodies.append(body)
        trails.append(ax.plot([], [], color=color, linewidth=1.1, alpha=0.55)[0])
        markers.append(ax.plot([], [], 'o', color=color, markersize=5, zorder=4)[0])
        ax.scatter(*(positions[0, i] - equilibrium[i]), marker='x', color=color,
                   s=35, label='Initial position')
        ax.legend(loc='lower left', fontsize=7)

    status = fig.text(0.5, 0.015, '', ha='center', fontsize=9)
    writer = FFMpegWriter(fps=20, codec='libx264',
                         extra_args=['-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart'])
    frames = np.arange(0, len(times), 5)

    if frames[-1] != len(times) - 1:
        frames = np.append(frames, len(times) - 1)

    with writer.saving(fig, str(path), dpi=120):
        for step in frames:
            for i in range(2):
                bodies[i].center = positions[step, i]
                displacement = positions[:step + 1, i] - equilibrium[i]
                trails[i].set_data(displacement[:, 0], displacement[:, 1])
                markers[i].set_data([displacement[-1, 0]], [displacement[-1, 1]])

            status.set_text('t = %.2f s     relative energy = %.3e     local panels show actual displacement'
                            % (times[step], relative[step]))
            writer.grab_frame()

    plt.close(fig)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,duration',
                            '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                   check=True, capture_output=True)

    if (stream['codec_name'] != 'h264' or stream['width'] != 960 or stream['height'] != 960
            or stream['r_frame_rate'] != '20/1' or int(stream['nb_frames']) != len(frames)):
        raise RuntimeError('Unexpected video specifications: ' + str(stream))

    return dict(stream, size_bytes=path.stat().st_size, full_decode_passed=True)


# function to run the separate interacting robot validation
def run_validation(output, video):
    output, video = Path(output), Path(video)

    for folder in (output, video):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Output is not empty; choose a new folder')

    if output.resolve() == video.resolve():
        raise ValueError('Choose separate result and video folders')

    config, environment, equilibrium, certificate = equilibrium_case()
    offset = np.array([[0.0001, -0.00008], [-0.00007, 0.00009]])
    initial_velocity = np.array([[0.00002, -0.00001], [-0.00001, 0.00002]])
    error = certificate['equilibrium_representation_error_upper_bound'] + 1e-14
    root = Interval(*certificate['root_interval_exact'])
    exact_equilibrium = [[root, Interval(1)], [Interval(3) - root, Interval(1)]]
    position_bound, velocity_bound = Interval(0), Interval(0)

    for i in range(2):
        for axis in range(2):
            initial = Interval(Fraction.from_float(float((equilibrium + offset)[i, axis])))
            position_bound += (initial - exact_equilibrium[i][axis]).square()
            velocity_bound += Interval(Fraction.from_float(float(initial_velocity[i, axis]))).square()

    energy_bound = (certificate['proved_hessian_upper_bound'] * position_bound
                    + config.mass * velocity_bound) / 2
    initial_upper = energy_bound.bounds()[1]

    if initial_upper >= certificate['invariant_sublevel_energy']:
        raise ValueError('Initial state is outside the proved sublevel bound')

    times, positions, velocities = fixed_field_trajectory(
        config, environment, equilibrium + offset, initial_velocity)
    potential = np.array([potential_energy(p, environment, config) for p in positions])
    kinetic = 0.5 * config.mass * np.sum(velocities ** 2, axis=(1, 2))
    total = potential + kinetic
    relative = total - potential_energy(equilibrium, environment, config)
    speed = np.linalg.norm(velocities, axis=2)
    distance = np.linalg.norm(positions - equilibrium, axis=(1, 2))
    rate = -config.damping * np.sum(velocities ** 2, axis=(1, 2))
    dissipated = cumulative_trapezoid(-rate, times, initial=0.0)
    reference_walls, reference_pair = active_sets(equilibrium, environment, config)
    unchanged = True

    for p in positions:
        walls, pair = active_sets(p, environment, config)
        unchanged &= np.array_equal(walls, reference_walls) and pair == reference_pair

    acceleration = np.array([(baseline_forces(p, environment, config) - config.damping * v) / config.mass
                             for p, v in zip(positions, velocities)])
    balance = total - total[0] + dissipated
    checks = {'finite_states_and_energy': bool(np.isfinite(total).all() and np.isfinite(positions).all()
                                               and np.isfinite(velocities).all()),
              'initial_energy_inside_proved_sublevel': bool(initial_upper < certificate['invariant_sublevel_energy']),
              'inside_proved_position_ball': bool(distance.max() + error < certificate['position_ball_radius']),
              'fixed_active_sets_at_all_output_samples': bool(unchanged),
              'collision_free_at_all_output_samples': all(environment.valid_positions(p) for p in positions),
              'energy_nonincrease_with_roundoff_tolerance': bool(np.diff(total).max() < 1e-11),
              'numerical_energy_balance_within_1e_10': bool(np.abs(balance).max() < 1e-10),
              'observed_recovery': bool(distance[-1] < distance[0] * 1e-4 and speed[-1].max() < 1e-7)}

    if not all(checks.values()):
        raise RuntimeError('Multi-robot validation failed: ' + str(checks))

    output.mkdir(parents=True, exist_ok=True)
    video.mkdir(parents=True, exist_ok=True)
    table = {'time_s': times, 'potential_energy': potential, 'kinetic_energy': kinetic,
             'total_energy': total, 'relative_mechanical_energy': relative,
             'energy_derivative': rate, 'integrated_dissipation': dissipated,
             'energy_balance_error': balance, 'distance_to_equilibrium': distance,
             'mean_speed': speed.mean(axis=1), 'maximum_speed': speed.max(axis=1)}

    for i in range(2):
        for axis, name in enumerate(('x', 'y')):
            prefix = 'robot_' + str(i + 1) + '_'
            table[prefix + name] = positions[:, i, axis]
            table[prefix + 'v' + name] = velocities[:, i, axis]
            table[prefix + 'a' + name] = acceleration[:, i, axis]

    pd.DataFrame(table).to_csv(output / 'fixed_field_history.csv', index=False)
    parameters = {'configuration': asdict(config), 'method': 'DOP853', 'rtol': 1e-11, 'atol': 1e-13,
                  'maximum_internal_step_s': 0.02, 'output_interval_s': config.dt,
                  'initial_position': positions[0].tolist(), 'initial_velocity': initial_velocity.tolist(),
                  'position_perturbation': offset.tolist(),
                  'model': 'Continuous interacting baseline ODE; no caps, projections or safety resets',
                  'quadrature_absolute_and_relative_tolerances': 1e-12,
                  'video': {'resolution': [960, 960], 'fps': 20, 'trail_linewidth': 1.1, 'trail_alpha': 0.55}}
    summary = {'checks': checks, 'initial_energy_upper_bound': float(initial_upper),
               'initial_relative_energy': float(relative[0]), 'final_relative_energy': float(relative[-1]),
               'maximum_energy_increment': float(np.diff(total).max()),
               'maximum_energy_balance_error': float(np.abs(balance).max()),
               'initial_distance_to_equilibrium': float(distance[0]),
               'final_distance_to_equilibrium': float(distance[-1]),
               'maximum_distance_to_equilibrium': float(distance.max()),
               'final_maximum_speed': float(speed[-1].max()), 'maximum_speed': float(speed.max()),
               'maximum_acceleration': float(np.linalg.norm(acceleration, axis=2).max()),
               'minimum_robot_pair_distance': float(np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1).min()),
               'minimum_robot_body_wall_clearance': float(min(positions.min(),
                   (config.width - positions[:, :, 0]).min(),
                   (config.height - positions[:, :, 1]).min()) - config.robot_r)}

    for name, data in [('parameters', parameters), ('equilibrium_certificate', certificate), ('summary', summary)]:
        (output / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')

    plots = [('energy_vs_time.png', relative, 'Relative mechanical energy U - U* + T',
              'Interacting conservative ODE: energy dissipation'),
             ('distance_to_equilibrium.png', distance, 'Collective equilibrium error (units)',
              'Recovery near the certified two-robot equilibrium')]

    for name, values, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, values, label='Perturbed continuous trajectory')
        ax.set(xlabel='Time (s)', ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / name, dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    for i in range(2):
        ax.plot(times, speed[:, i], label='Robot ' + str(i + 1))

    ax.set(xlabel='Time (s)', ylabel='Speed (units/s)', title='Interacting conservative ODE: speed decay')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'speed_vs_time.png', dpi=300)
    plt.close(fig)
    fig, workspace, local = recovery_axes(equilibrium, config)

    for i, ax in enumerate(local):
        displacement = positions[:, i] - equilibrium[i]
        color = 'C' + str(i)
        workspace.add_patch(Circle(positions[0, i], config.robot_r, color=color, zorder=4))
        ax.plot(displacement[:, 0], displacement[:, 1], color=color, linewidth=1.1, alpha=0.55)
        ax.scatter(*displacement[0], marker='x', color=color, label='Initial position')
        ax.legend(loc='lower left', fontsize=7)

    fig.savefig(output / 'perturbation_recovery.png', dpi=300)
    plt.close(fig)
    specifications = render_video(video / 'fixed_field.mp4', times, positions, equilibrium, relative, config)
    (output / 'video_validation.json').write_text(json.dumps(specifications, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

    return summary


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Certified interacting baseline equilibrium validation')
    parser.add_argument('--output', type=Path, default=root / 'Results' / 'theoretical_validation_multi_robot')
    parser.add_argument('--video-output', type=Path, default=root / 'Videos' / 'theoretical_validation_multi_robot')
    args = parser.parse_args()
    run_validation(args.output, args.video_output)
