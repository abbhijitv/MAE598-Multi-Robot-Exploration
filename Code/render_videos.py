# import statements
import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
from config import Config
from simulation import run_simulation
from validation import validate_result
from visualization import draw_environment, coverage_image
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Patch


# function to replay and check the historical simulations
def historical_runs(root):
    aliases = {'robot_count': 'robot_c', 'robot_radius': 'robot_r',
               'sensing_radius': 'sensing_r', 'robot_interaction_radius': 'robot_interact_r',
               'obstacle_interaction_radius': 'obstacle_interact_r', 'max_acceleration': 'max_acc',
               'obstacle_point_spacing': 'obstacle_boundary_sampling', 'discovery_radius': 'discovery_r'}
    discovery = json.loads((root / 'Results/discovery/parameters.json').read_text())
    runs = {}
    for mode in ('baseline', 'discovery'):
        folder = root / 'Results' / mode
        saved = json.loads((folder / 'parameters.json').read_text())
        # the baseline record predates the inactive discovery settings
        merged = dict(discovery)
        merged.update(saved)
        values = {aliases.get(k, k): v for k, v in merged.items()
                  if aliases.get(k, k) in Config.__dataclass_fields__ and k != 'controller'}
        if set(values) != set(Config.__dataclass_fields__) - {'controller'}:
            raise ValueError('Historical configuration is incomplete')
        config = Config(controller=mode, **values)
        result = run_simulation(config, positions=saved['initial_positions'],
                                velocities=saved['initial_velocities'])
        states = np.genfromtxt(folder / 'robot_positions.csv', delimiter=',', names=True)
        coverage = np.genfromtxt(folder / 'coverage.csv', delimiter=',', names=True)
        for key, columns in [('positions', ('x', 'y')), ('velocities', ('vx', 'vy'))]:
            expected = np.column_stack([states[c] for c in columns]).reshape(result[key].shape)
            np.testing.assert_allclose(result[key], expected, rtol=0, atol=1e-13)
        np.testing.assert_allclose(result['coverage'], coverage['coverage_percent'], rtol=0, atol=1e-13)
        np.testing.assert_allclose(result['time'], coverage['time_s'], rtol=0, atol=1e-13)
        validate_result(result, config)
        if len(result['time']) != 1201 or result['time'][0] != 0 or result['time'][-1] != 60:
            raise ValueError('Expected the complete historical 60-second simulation')
        runs[mode] = (config, result)
        print('{}: historical replay verified, final coverage {:.10f}%'.format(
            mode, result['coverage'][-1]), flush=True)
    baseline = asdict(runs['baseline'][0])
    discovery = asdict(runs['discovery'][0])
    if any(baseline[k] != discovery[k] for k in baseline if k != 'controller'):
        raise ValueError('Historical physical settings differ between controllers')
    for key in ('positions', 'velocities'):
        np.testing.assert_array_equal(runs['baseline'][1][key][0], runs['discovery'][1][key][0])
    return runs


# function to render the historical states with robot trails
def render_video(config, result, destination):
    times = result['time']
    positions = result['positions']
    environment = result['environment']
    fig, ax = plt.subplots(figsize=(8, 8))
    image = coverage_image(ax, result['explored'][0], environment, config)
    draw_environment(ax, config)
    colors = plt.get_cmap('tab10')(np.arange(config.robot_c))
    trails, disks = [], []
    for i, color in enumerate(colors):
        trail, = ax.plot([], [], color=color, linewidth=1.1, alpha=0.55, zorder=3)
        disk = Circle(positions[0, i], config.robot_r, color=color, edgecolor=None, zorder=5)
        ax.add_patch(disk)
        trails.append(trail)
        disks.append(disk)
    ax.legend(handles=[Patch(color='#eeeeee', label='Unexplored'),
                       Patch(color='#a6dba0', label='Explored'),
                       Patch(color='0.25', label='Obstacle')], loc='upper left', fontsize=9)
    title = ax.set_title('', fontsize=12)
    fig.tight_layout()
    writer = FFMpegWriter(fps=20, codec='libx264',
                         metadata={'title': config.controller.capitalize() + ' exploration — historical configuration'},
                         extra_args=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    try:
        with writer.saving(fig, str(destination), dpi=120):
            for frame, time in enumerate(times):
                image.set_data(np.where(environment.free, result['explored'][frame].astype(int), 2))
                for i, disk in enumerate(disks):
                    disk.center = positions[frame, i]
                    trails[i].set_data(positions[:frame + 1, i, 0], positions[:frame + 1, i, 1])
                title.set_text('{} | t = {:.2f} / 60.00 s | coverage = {:.2f}%'.format(
                    config.controller.capitalize(), time, result['coverage'][frame]))
                writer.grab_frame()
                if frame % 300 == 0:
                    print(config.controller, 'encoded frame', frame, 'of', len(times), flush=True)
    finally:
        plt.close(fig)


# function to check the encoded video format and playback
def check_video(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size:stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames',
        '-of', 'json', str(path)]))
    stream = info['streams'][0]
    expected = {'codec_name': 'h264', 'pix_fmt': 'yuv420p', 'width': 960, 'height': 960,
                'r_frame_rate': '20/1', 'nb_frames': '1201'}
    if any(stream.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected video specifications: ' + str(path))
    if abs(float(info['format']['duration']) - 60.05) > 0.001:
        raise ValueError('Unexpected video duration: ' + str(path))
    print(path.name, json.dumps(info), flush=True)


# function to regenerate the historical comparison videos
def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Render verified historical baseline and discovery videos')
    parser.add_argument('--output', type=Path, default=root / 'Videos')
    parser.add_argument('--overwrite', action='store_true', help='replace existing destination MP4s')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == root / 'Results' or root / 'Results' in output.parents:
        parser.error('Video output must be outside Results')
    if any((output / (mode + '.mp4')).exists() for mode in ('baseline', 'discovery')) and not args.overwrite:
        parser.error('Destination videos exist; use a new --output folder or explicit --overwrite')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('Install FFmpeg with ffprobe and libx264 support and add it to PATH')
    runs = historical_runs(root)
    output.mkdir(parents=True, exist_ok=True)
    # validate both candidates before replacing any destination video
    with tempfile.TemporaryDirectory(prefix='mae598_render_') as directory:
        temporary = Path(directory)
        for mode, (config, result) in runs.items():
            render_video(config, result, temporary / (mode + '.mp4'))
            check_video(temporary / (mode + '.mp4'))
        for mode in runs:
            shutil.copyfile(temporary / (mode + '.mp4'), output / (mode + '.mp4'))
    print('Saved both verified videos to', output, flush=True)


if __name__ == '__main__':
    main()
