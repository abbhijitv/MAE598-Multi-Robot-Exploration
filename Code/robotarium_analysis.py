# import statements
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from robotarium_validation import run_robotarium, platform_config, to_robotarium, to_project
from robotarium_validation import save_run, robotarium_api
from theoretical_analysis import equilibrium_case, potential_energy


# function to run near-boundary and near-neighbor safety trials
def additional_trials(root):
    config = platform_config(count=2, duration=9.9)
    poses = np.array([[.804, .600], [.600, .600], [0., 0.]])
    for mode in ('baseline', 'discovery'):
        current = replace(config, controller=mode)
        result = run_robotarium(current, poses=poses)
        save_run(result, current, root / 'safety_stress' / mode, .09, .03)

    # the fixed-field trial evaluates forces and energy at the projected point
    config, environment, equilibrium, certificate = equilibrium_case()
    config = replace(config, dt=.033, duration=19.8)
    point = equilibrium + [.003, -.002]
    physical = to_robotarium(point, config, .5)
    poses = np.array([[physical[0] - .03], [physical[1]], [0.]])
    result = run_robotarium(config, scale=.5, poses=poses, control_at_projection=True)
    save_run(result, config, root / 'fixed_field', .5, .03)
    states = result['poses']
    projected = states[:, :2, 0] + .03 * np.column_stack((np.cos(states[:, 2, 0]), np.sin(states[:, 2, 0])))
    points = to_project(projected, config, .5)
    velocity = result['reference_velocities'][:, 0]
    potential = np.array([potential_energy(p[None, :], environment, config) for p in points])
    energy = potential - potential_energy(equilibrium[None, :], environment, config)
    energy += .5 * config.mass * np.sum(velocity ** 2, axis=1)
    distance = np.linalg.norm(points - equilibrium, axis=1)
    table = pd.DataFrame({'time_s': result['time'], 'relative_reference_energy': energy,
                          'projected_distance_to_equilibrium': distance,
                          'reference_speed': np.linalg.norm(velocity, axis=1)})
    table.to_csv(root / 'fixed_field/energy.csv', index=False)
    report = {'initial_reference_energy': float(energy[0]), 'final_reference_energy': float(energy[-1]),
              'maximum_energy_increment': float(np.diff(energy).max()),
              'initial_projected_error': float(distance[0]), 'final_projected_error': float(distance[-1]),
              'maximum_projected_error': float(distance.max()), 'position_ball_radius': .01,
              'all_states_inside_ball': bool(distance.max() < .01),
              'control_at_projection': True, 'initial_reference_velocity': [0., 0.],
              'interpretation': 'Discrete unicycle projected-point evidence; not a proof or exact second-order equivalence'}
    (root / 'fixed_field/energy_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Fixed-field trial:', report, flush=True)


# function to save a figure with readable labels
def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=180)
    plt.close(fig)


# function to regenerate figures only from saved Robotarium data
def figures(root):
    output = root / 'figures'
    output.mkdir(exist_ok=False)
    table = pd.read_csv(root / 'experiments/all_results.csv')
    selected = table[(table.robot_count == 5) & (table.seed == 598) & (table.layout == 'open')]
    colors = {'baseline': '#3366aa', 'discovery': '#cc6622'}
    for row in selected.itertuples():
        folder = root / 'experiments' / row.result_folder
        states = pd.read_csv(folder / 'states.csv')
        fig, ax = plt.subplots(figsize=(9, 5) if row.controller == 'discovery' else (7, 5))
        for robot, data in states.groupby('robot_id'):
            ax.plot(data.x_m, data.y_m, linewidth=1, label='Robot ' + str(robot + 1))
            ax.scatter(data.x_m.iloc[0], data.y_m.iloc[0], marker='x')
        ax.set(xlabel='x (m)', ylabel='y (m)', xlim=(-.9, .9), ylim=(-.9, .9),
               title=row.controller.capitalize() + ': official Robotarium unicycle trajectories', aspect='equal')
        if row.controller == 'discovery':
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        else:
            ax.legend(fontsize=8)
        save_figure(fig, output, row.controller + '_trajectories.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5))
    for ax, mode in zip(axes, ('baseline', 'discovery')):
        folder = root / 'experiments/obstacles_n5_seed598' / mode
        params = json.loads((folder / 'parameters.json').read_text())
        scale = params['scale_m_per_unit']
        for x, y, radius in params['configuration']['obstacles']:
            ax.add_patch(Circle((scale * (x - 10), scale * (y - 10)), scale * radius, color='gray', alpha=.6))
        states = pd.read_csv(folder / 'states.csv')
        for robot, data in states.groupby('robot_id'):
            line, = ax.plot(data.x_m, data.y_m, linewidth=1, label='Robot ' + str(robot + 1))
            ax.scatter(data.x_m.iloc[0], data.y_m.iloc[0], marker='x', color=line.get_color(), s=20)
            ax.scatter(data.x_m.iloc[-1], data.y_m.iloc[-1], marker='o', color=line.get_color(), s=12)
        ax.scatter([], [], marker='x', color='black', label='Start')
        ax.scatter([], [], marker='o', color='black', label='End')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=4, fontsize=7)
        ax.set(xlabel='x (m)', ylabel='y (m)', xlim=(-.9, .9), ylim=(-.9, .9),
               title=mode.capitalize() + ': circular obstacles', aspect='equal')
    save_figure(fig, output, 'obstacle_trajectories.png')

    fig, ax = plt.subplots(figsize=(7, 4))
    for row in selected.itertuples():
        coverage = pd.read_csv(root / 'experiments' / row.result_folder / 'coverage.csv')
        ax.plot(coverage.time_s, coverage.coverage_percent, color=colors[row.controller], label=row.controller)
    ax.set(xlabel='Simulation time (s)', ylabel='Cumulative coverage (%)', ylim=(0, 100), title='Paired five-robot coverage')
    ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig, output, 'coverage_vs_time.png')

    selections = [('baseline_vs_discovery.png', selected, 'controller', 'Final coverage: paired open case'),
                  ('robot_counts.png', table[(table.seed == 598) & (table.layout == 'open')], 'robot_count', 'Robot counts: open layout, seed 598'),
                  ('random_seeds.png', table[(table.robot_count == 5) & (table.layout == 'open')], 'seed', 'Five initialization seeds'),
                  ('obstacle_comparison.png', table[(table.robot_count == 5) & (table.seed == 598)], 'layout', 'Open and obstacle layouts')]
    for name, values, key, title in selections:
        fig, ax = plt.subplots(figsize=(8, 4))
        if key == 'controller':
            ax.bar(values.controller, values.final_coverage_percent, color=[colors[x] for x in values.controller])
        else:
            groups = list(values[key].drop_duplicates())
            x = np.arange(len(groups))
            for i, mode in enumerate(('baseline', 'discovery')):
                rows = values[values.controller == mode].set_index(key).loc[groups]
                ax.bar(x + (i - .5) * .35, rows.final_coverage_percent, width=.35, color=colors[mode], label=mode)
            ax.set_xticks(x, [str(x) for x in groups])
            if key == 'seed':
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            else:
                ax.legend()
        if key == 'robot_count':
            ax.set_xlabel('Robot count')
        ax.set(ylabel='Final coverage (%)', ylim=(0, 100), title=title)
        save_figure(fig, output, name)

    stats = pd.read_csv(root / 'experiments/seed_statistics.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.controller, stats.mean_coverage, yerr=stats.sample_sd_coverage, capsize=5,
           color=[colors[x] for x in stats.controller])
    ax.set(ylabel='Final coverage (%)', ylim=(0, 105), title='Five seeds: mean and sample standard deviation')
    save_figure(fig, output, 'seed_statistics.png')

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for row in selected.itertuples():
        states = pd.read_csv(root / 'experiments' / row.result_folder / 'states.csv')
        p = states[['x_m', 'y_m']].to_numpy().reshape(-1, 5, 2)
        distances = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i in range(5) for j in range(i)])
        time = states.time_s.drop_duplicates()
        axes[0].plot(time, distances.min(axis=0), label=row.controller, color=colors[row.controller])
        axes[1].plot(time, .9 - np.abs(p).max(axis=(1, 2)) - .095, color=colors[row.controller], label=row.controller)
    axes[0].axhline(.19, color='black', linestyle='--', label='Conservative body separation')
    axes[0].set(ylabel='Center separation (m)', title='Actual Robotarium states: separation and boundary clearance')
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Body-wall clearance (m)')
    save_figure(fig, output, 'safety_vs_time.png')

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for mode in ('baseline', 'discovery'):
        states = pd.read_csv(root / 'safety_stress' / mode / 'states.csv')
        positions = states[['x_m', 'y_m']].to_numpy().reshape(-1, 2, 2)
        time = states.time_s.drop_duplicates()
        axes[0].plot(time, np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1), label=mode)
        axes[1].plot(time, .9 - np.abs(positions).max(axis=(1, 2)) - .095)
    axes[0].axhline(.19, color='black', linestyle='--')
    axes[0].set(ylabel='Center separation (m)', title='Near-neighbor and near-boundary stress trial')
    axes[0].legend()
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Body-wall clearance (m)')
    save_figure(fig, output, 'safety_stress.png')

    energy = pd.read_csv(root / 'fixed_field/energy.csv')
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, key, label in zip(axes, ['relative_reference_energy', 'reference_speed', 'projected_distance_to_equilibrium'],
                             ['Reference energy', 'Reference speed (units/s)', 'Projected error (units)']):
        ax.plot(energy.time_s, energy[key])
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_title('Separate fixed-field projected-point trial; simulation evidence')
    axes[-1].set_xlabel('Simulation time (s)')
    save_figure(fig, output, 'fixed_field_validation.png')


# function to replay saved inputs through the native Robotarium figure
# source: official rps.robotarium.Robotarium and Matplotlib FFMpegWriter
def videos(root, output):
    Robotarium, mapping, barrier = robotarium_api()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Video destination is not empty')
    output.mkdir(parents=True, exist_ok=True)
    for mode in ('baseline', 'discovery'):
        folder = root / 'experiments/open_n5_seed598' / mode
        states = pd.read_csv(folder / 'states.csv')
        inputs = pd.read_csv(folder / 'inputs.csv')
        history = states[['x_m', 'y_m', 'theta_rad']].to_numpy().reshape(-1, 5, 3).transpose(0, 2, 1)
        commands = inputs[['v_m_s', 'omega_rad_s']].to_numpy().reshape(-1, 5, 2).transpose(0, 2, 1)
        params = json.loads((folder / 'parameters.json').read_text())
        config = params['configuration']
        coverage = pd.read_csv(folder / 'coverage.csv').coverage_percent.to_numpy()
        grids = np.load(folder / 'maps.npz')['explored']
        random = np.random.get_state()
        np.random.seed(config['seed'])
        r = Robotarium(number_of_robots=5, initial_conditions=history[0], show_figure=True,
                       sim_in_real_time=False, skip_initialization=True)
        r.show_figure = False
        fig, ax = r._fig, r._axes_handle
        fig.set_size_inches(9.6, 6.4)
        fig.subplots_adjust(left=.08, right=.98, bottom=.11, top=.90)
        image = ax.imshow(grids[0], origin='lower', extent=(-.9, .9, -.9, .9),
                          cmap='Greens', vmin=0, vmax=1, alpha=.35, zorder=-5)
        ax.add_patch(Rectangle((-.9, -.9), 1.8, 1.8, fill=False, linestyle='--', edgecolor='gray'))
        ax.set_axis_on()
        ax.set(xlabel='x (m)', ylabel='y (m)')
        trails = [ax.plot([], [], linewidth=1, alpha=.55, zorder=1, label='Robot ' + str(i + 1))[0] for i in range(5)]
        ax.legend(loc='lower right', fontsize=8)
        title = ax.set_title('', fontsize=11)
        writer = FFMpegWriter(fps=10, codec='libx264', extra_args=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        frames = 0
        with writer.saving(fig, str(output / (mode + '.mp4')), dpi=100):
            for step in range(len(history)):
                actual = r.get_poses()
                np.testing.assert_allclose(actual, history[step], rtol=0, atol=1e-12)
                if step % 10 == 0 or step == len(history) - 1:
                    r._draw_robots()
                    image.set_data(grids[step])
                    for i, trail in enumerate(trails):
                        trail.set_data(history[:step + 1, 0, i], history[:step + 1, 1, i])
                    title.set_text('Robotarium local simulator | {} | t={:.2f} s | coverage={:.2f}%'.format(
                        mode, step * .033, coverage[step]))
                    writer.grab_frame()
                    frames += 1
                if step < len(commands):
                    r.set_velocities(np.arange(5), commands[step])
                    r.step()
        np.random.set_state(random)
        plt.close(fig)
        print(mode, 'native-engine replay verified;', frames, 'video frames', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robotarium stress/theory trials, figures and native replay videos')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'Results/robotarium')
    parser.add_argument('--trials', action='store_true')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--videos', type=Path)
    args = parser.parse_args()
    if args.trials:
        additional_trials(args.root)
    if args.figures:
        figures(args.root)
    if args.videos:
        videos(args.root, args.videos)
