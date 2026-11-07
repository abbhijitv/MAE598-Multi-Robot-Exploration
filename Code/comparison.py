# import statements
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {'baseline': '#1f77b4', 'discovery': '#ff7f0e'}


# function to find the saved results after moving the project
def resolve_result_folder(root, recorded):
    root = Path(root)
    folder = Path(recorded)
    for index, part in enumerate(folder.parts):
        if part in ('robot_counts', 'environments', 'random_seeds', 'smoke'):
            local = root.joinpath(*folder.parts[index:])
            if local.is_dir():
                return local
    if not folder.is_absolute() and (root / folder).is_dir():
        return root / folder
    if folder.is_dir():
        return folder
    raise FileNotFoundError('Cannot locate saved experiment folder: ' + str(recorded))


# function to save the comparison plot
def save_plot(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / name, dpi=160)
    plt.close(fig)


# function to compare the controllers using bar plots
def grouped_bars(table, key, order, column, title, ylabel, output, name, labels=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    for i, controller in enumerate(('baseline', 'discovery')):
        values = table[table.controller == controller].set_index(key).loc[order, column]
        ax.bar(x + (i - 0.5) * 0.36, values, 0.36,
               label=controller.capitalize(), color=COLORS[controller])
    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None else order)
    ax.set(title=title, ylabel=ylabel, xlabel=key.replace('_', ' ').capitalize())
    if column == 'final_coverage_percent':
        ax.set_ylim(0, 105)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    save_plot(fig, output, name)


# function to create the experiment comparison plots
def make_comparisons(root):
    root = Path(root)
    output = root / 'comparisons'
    output.mkdir(exist_ok=True)
    table = pd.read_csv(root / 'all_results.csv')
    original = table[(table.experiment_category == 'robot_counts') & (table.robot_count == 5)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in original.itertuples():
        history = pd.read_csv(resolve_result_folder(root, row.result_folder) / 'coverage.csv')
        ax.plot(history.time_s, history.coverage_percent, color=COLORS[row.controller],
                label=row.controller.capitalize())
    ax.set(xlabel='Simulation time (s)', ylabel='Cumulative coverage (%)', ylim=(0, 100),
           title='Original environment: five robots, seed 598')
    ax.grid(alpha=0.2)
    ax.legend()
    save_plot(fig, output, '01_coverage_vs_time.png')
    grouped_bars(table[table.experiment_category == 'robot_counts'], 'robot_count', [3, 5, 8],
                 'final_coverage_percent', 'Robot count: original environment, seed 598',
                 'Final coverage (%)', output, '02_robot_counts.png')
    grouped_bars(table[table.experiment_category == 'environments'], 'obstacle_layout',
                 ['original', 'open', 'cluttered'], 'final_coverage_percent',
                 'Obstacle environments: five robots, seed 598', 'Final coverage (%)',
                 output, '03_environments.png')
    grouped_bars(table[table.experiment_category == 'random_seeds'], 'seed', [598, 42, 123, 2026, 7],
                 'final_coverage_percent', 'Initialization seeds: five robots, original environment',
                 'Final coverage (%)', output, '04_random_seeds.png')
    stats = pd.read_csv(root / 'summary_statistics.csv').set_index('controller')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, controller in enumerate(('baseline', 'discovery')):
        ax.bar(i, stats.loc[controller, 'mean_final_coverage_percent'],
               yerr=stats.loc[controller, 'sample_std_final_coverage_percent'],
               capsize=8, color=COLORS[controller], label=controller.capitalize())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Discovery'])
    ax.set(ylabel='Mean final coverage (%)', xlabel='Controller', ylim=(0, 108),
           title='Five seeds: mean and sample standard deviation')
    ax.legend()
    save_plot(fig, output, '05_mean_coverage.png')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, controller in enumerate(('baseline', 'discovery')):
        row = original[original.controller == controller].iloc[0]
        x = np.arange(3) + (i - 0.5) * 0.36
        values = np.array([row['time_' + str(t) + '_s'] for t in (50, 75, 90)])
        # mark the thresholds that were not reached
        valid = np.isfinite(values)
        ax.bar(x[valid], values[valid], 0.36, color=COLORS[controller], label=controller.capitalize())
        ax.scatter(x[~valid], np.full((~valid).sum(), 63), marker='x', color=COLORS[controller])
        for position in x[~valid]:
            ax.text(position, 64, 'NR', ha='center', fontsize=9)
    ax.axhline(60, color='0.5', linestyle=':', linewidth=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['50%', '75%', '90%'])
    ax.set(xlabel='Coverage threshold', ylabel='First recorded threshold time (s)', ylim=(0, 70),
           title='Original five-robot case: NR = not reached by 60 s')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker='x', linestyle='none', color='0.4'))
    ax.legend(handles, labels + ['NR marker (not a measured time)'])
    save_plot(fig, output, '06_threshold_times.png')
    unique = table[~table.reused_run].copy()
    unique['case'] = unique.apply(lambda row: '{} / n={} / {}'.format(
        row.obstacle_layout, row.robot_count, row.seed), axis=1)
    order = list(unique['case'].drop_duplicates())
    grouped_bars(unique, 'case', order, 'total_travel_distance',
                 'Total team travel: all nine unique paired cases', 'Travel distance (units)',
                 output, '07_travel_distance.png', labels=[name.replace(' / ', '\n') for name in order])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot completed experiment data')
    parser.add_argument('output', type=Path)
    make_comparisons(parser.parse_args().output)
