# import statements
import json
from dataclasses import asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd


# function to draw the environment and obstacles
def draw_environment(ax, config):
    ax.set(xlim=(0, config.width), ylim=(0, config.height),
           xlabel='x (units)', ylabel='y (units)', aspect='equal')
    for x, y, radius in config.obstacles:
        ax.add_patch(Circle((x, y), radius, color='0.25'))


# function to display the explored and unexplored cells
def coverage_image(ax, explored, environment, config):
    values = np.where(environment.free, explored.astype(int), 2)
    return ax.imshow(values, origin='lower', extent=(0, config.width, 0, config.height),
                     cmap=ListedColormap(['#eeeeee', '#a6dba0', '#404040']),
                     norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation='nearest')


# function to save the simulation results
def save_results(result, config, output, animation=True):
    output.mkdir(parents=True, exist_ok=True)
    label = config.controller.capitalize()
    times = result['time']
    positions = result['positions']
    velocities = result['velocities']
    environment = result['environment']
    # coverage and robot position data
    counts = result['explored'].sum(axis=(1, 2))
    pd.DataFrame({'time_s': times, 'explored_cells': counts,
                  'free_cells': environment.free.sum(),
                  'coverage_percent': result['coverage']}).to_csv(output / 'coverage.csv', index=False)
    pd.DataFrame({'time_s': np.repeat(times, config.robot_c),
                  'robot_id': np.tile(np.arange(1, config.robot_c + 1), len(times)),
                  'x': positions[:, :, 0].ravel(), 'y': positions[:, :, 1].ravel(),
                  'vx': velocities[:, :, 0].ravel(), 'vy': velocities[:, :, 1].ravel()
                  }).to_csv(output / 'robot_positions.csv', index=False)
    parameters = asdict(config)
    parameters['controller'] = config.controller
    parameters['integration'] = 'Velocity-first semi-implicit Euler with norm caps'
    parameters['safety'] = 'Synchronous swept disks; shorten all moves and stop on first contact'
    parameters['sensing'] = 'Cell centers within radius; no obstacle occlusion'
    if config.controller == 'discovery':
        parameters['discovery_selection'] = 'Nearest known-free, privately unexplored cell within discovery_r'
        parameters['discovery_weight'] = 'Singleton weight 1 / max(distance, epsilon)'
        parameters['discovery_scaling'] = 's(c) = 1; no global coverage input'
        parameters['information'] = 'Private maps, no communication, known bounds and own pose'
        parameters['occupancy_lookahead'] = 'Local occupancy sensing at discovery_r; available to both modes, unused by baseline'
        parameters['assumption'] = 'Occupancy lookahead is distinct from the unchanged exploration sensing_r; no occlusion'
    parameters['initial_positions'] = positions[0].tolist()
    parameters['initial_velocities'] = velocities[0].tolist()
    (output / 'parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    minimum_distance = None
    if config.robot_c > 1:
        minimum_distance = min(float(np.linalg.norm(positions[:, i] - positions[:, j], axis=1).min())
                               for i in range(config.robot_c)
                               for j in range(i + 1, config.robot_c))
    summary = {'initial_coverage_percent': float(result['coverage'][0]),
               'final_coverage_percent': float(result['coverage'][-1]),
               'minimum_robot_center_distance': minimum_distance,
               'safety_interventions': result['safety_interventions'],
               'maximum_speed': float(np.linalg.norm(velocities, axis=2).max()),
               'final_maximum_speed': float(np.linalg.norm(velocities[-1], axis=1).max()),
               'path_lengths': np.linalg.norm(np.diff(positions, axis=0), axis=2).sum(axis=0).tolist()}
    if config.controller == 'discovery':
        summary['coverage_gain_percentage_points'] = float(result['coverage'][-1] - result['coverage'][0])
        summary['total_travel_distance'] = float(sum(summary['path_lengths']))
        recent = times >= max(0, times[-1] - 10)
        summary['coverage_gain_last_10_seconds'] = float(result['coverage'][-1] - result['coverage'][recent][0])
        gained = np.flatnonzero(np.diff(result['coverage']) > 0)
        summary['last_new_coverage_time_s'] = float(times[gained[-1] + 1]) if len(gained) else 0.0
        summary['mean_final_speed'] = float(np.linalg.norm(velocities[-1], axis=1).mean())
        pd.DataFrame({'step_start_s': times[:-1], 'step_end_s': times[1:],
                      'safety_intervention': result['safety_events']}).to_csv(output / 'safety_events.csv', index=False)
        selected = result['targets']
        pd.DataFrame({'time_s': np.repeat(times[:-1], config.robot_c),
                      'robot_id': np.tile(np.arange(1, config.robot_c + 1), len(times) - 1),
                      'target_x': selected[:, :, 0].ravel(),
                      'target_y': selected[:, :, 1].ravel()}).to_csv(output / 'discovery_targets.csv', index=False)
        np.savez_compressed(output / 'local_maps.npz', known_free=result['known_free'],
                            explored=result['local_explored'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # robot positions and coverage plots
    colors = plt.get_cmap('tab10')(np.arange(config.robot_c) % 10)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_environment(ax, config)
    for i, color in enumerate(colors):
        ax.scatter(*positions[0, i], marker='x', color=color)
        ax.add_patch(Circle(positions[-1, i], config.robot_r, color=color,
                            label='Robot {}'.format(i + 1)))
        ax.add_patch(Circle(positions[-1, i], config.sensing_r, fill=False,
                            color=color, alpha=0.3))
    ax.plot([], [], 'kx', label='Initial positions')
    ax.legend(fontsize=8)
    ax.set_title(label + ' final positions and sensing footprints')
    fig.tight_layout()
    fig.savefig(output / 'environment.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_environment(ax, config)
    for i, color in enumerate(colors):
        ax.plot(positions[:, i, 0], positions[:, i, 1], color=color,
                label='Robot {}'.format(i + 1))
        ax.scatter(*positions[0, i], marker='x', color=color)
        ax.scatter(*positions[-1, i], color=color, s=25)
    ax.plot([], [], 'kx', label='Start')
    ax.plot([], [], 'ko', label='End')
    ax.legend(fontsize=8)
    ax.set_title(label + ' robot trajectories')
    fig.tight_layout()
    fig.savefig(output / 'trajectories.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    coverage_image(ax, environment.explored, environment, config)
    draw_environment(ax, config)
    ax.legend(handles=[Patch(color='#eeeeee', label='Unexplored free cells'),
                       Patch(color='#a6dba0', label='Explored free cells'),
                       Patch(color='#404040', label='Obstacles')], fontsize=8)
    ax.set_title('Cumulative coverage: {:.2f}%'.format(result['coverage'][-1]))
    fig.tight_layout()
    fig.savefig(output / 'coverage_map.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, result['coverage'], label=label)
    ax.set(xlabel='Simulation time (s)', ylabel='Explored free cells (%)', ylim=(0, 100),
           title='Cumulative coverage versus time')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'coverage_vs_time.png', dpi=160)
    plt.close(fig)
    np.savez_compressed(output / 'coverage_grid.npz', free=environment.free,
                        explored=environment.explored, x_edges=environment.x_edges,
                        y_edges=environment.y_edges)
    # animation of the robot movement and explored cells
    if animation:
        fig, ax = plt.subplots(figsize=(6, 6))
        image = coverage_image(ax, result['explored'][0], environment, config)
        draw_environment(ax, config)
        disks = []
        for i, color in enumerate(colors):
            disk = Circle(positions[0, i], config.robot_r, color=color)
            ax.add_patch(disk)
            disks.append(disk)
        ax.legend(handles=[Patch(color='#eeeeee', label='Unexplored'),
                           Patch(color='#a6dba0', label='Explored')], fontsize=8)
        title = ax.set_title('')
        frames = np.unique(np.linspace(0, len(times) - 1,
                                       min(151, len(times)), dtype=int))

        # function to update the robot positions and coverage for each frame
        def update(frame):
            image.set_data(np.where(environment.free, result['explored'][frame].astype(int), 2))
            for i, disk in enumerate(disks):
                disk.center = positions[frame, i]
            title.set_text('t = {:.1f} s, coverage = {:.2f}%'.format(
                times[frame], result['coverage'][frame]))
            return [image, title] + disks

        movie = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        fig.tight_layout()
        movie.save(output / (config.controller + '.gif'), writer=PillowWriter(fps=10), dpi=85)
        plt.close(fig)
