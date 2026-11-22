# import statements
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from config import Config
from environment import Environment
from robots import initial_positions


# function to draw the workspace using the existing initialization
def workspace_figure():
    config = Config(seed=598)
    config.validate()
    positions = initial_positions(Environment(config), config)

    with plt.rc_context({'font.family': 'serif', 'font.size': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'axes.linewidth': 1.0}):
        fig = plt.figure(figsize=(3.5, 4.3))
        ax = fig.add_axes([.13, .28, .83, .675])
        ax.set(xlim=(0, config.width), ylim=(0, config.height),
               xlabel='x (units)', ylabel='y (units)', aspect='equal')
        ax.set_xticks([i * config.width / 4 for i in range(5)])
        ax.set_yticks([i * config.height / 4 for i in range(5)])
        ax.set_title('Initial workspace (seed 598)', fontsize=8, pad=5)
        ax.grid(color='0.85', linewidth=.35, zorder=0)

        # show all three ranges about the first robot
        ranges = [(config.sensing_r, '#0072B2', '-', 'Sensing'),
                  (config.robot_interact_r, '#009E73', '--', 'Robot interaction'),
                  (config.discovery_r, '#D55E00', '-.', 'Discovery search')]
        for radius, color, style, label in ranges:
            ax.add_patch(Circle(positions[0], radius, fill=False, color=color,
                                linestyle=style, linewidth=1.0, zorder=2))

        for i, (x, y, radius) in enumerate(config.obstacles):
            ax.add_patch(Circle((x, y), radius, facecolor='0.83',
                                edgecolor='0.25', linewidth=.7, zorder=3))
            ax.text(x, y, '$O_{}$\n({:g}, {:g})'.format(i + 1, x, y),
                    ha='center', va='center', fontsize=6, zorder=4)

        offsets = [(8, 9), (-10, -9), (9, -7), (-10, 5), (9, -8)]
        for i, position in enumerate(positions):
            ax.add_patch(Circle(position, config.robot_r, facecolor='#173F5F',
                                edgecolor='black', linewidth=.45, zorder=5))
            offset = offsets[i] if i < len(offsets) else (8, 8)
            ax.annotate('$R_{}$'.format(i + 1), position, xytext=offset,
                        textcoords='offset points', fontsize=6.5,
                        ha='left' if offset[0] > 0 else 'right', va='center',
                        arrowprops={'arrowstyle': '-', 'color': '0.3', 'lw': .4},
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .3},
                        zorder=6)

        handles = [Line2D([], [], color='black', linewidth=1, label='Workspace boundary'),
                   Patch(facecolor='0.83', edgecolor='0.25',
                         label='Obstacle (r = {:g})'.format(config.obstacles[0][2])),
                   Line2D([], [], marker='o', color='none', markerfacecolor='#173F5F',
                          markeredgecolor='black', markersize=4,
                          label='Robot body (r = {:g})'.format(config.robot_r))]
        handles += [Line2D([], [], color=color, linestyle=style, linewidth=1,
                           label='{} (r = {:g})'.format(label, radius))
                    for radius, color, style, label in ranges]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .06),
                   ncol=2, fontsize=6.5, frameon=False, columnspacing=1,
                   handlelength=2.2, labelspacing=.65)
        fig.text(.5, .015, 'Ranges shown about $R_1$; all radii in workspace units.',
                 ha='center', fontsize=6)

    return fig, config, positions


# function to save the report figure without running a simulation
def main():
    fig, config, positions = workspace_figure()
    output = Path(__file__).resolve().parents[1] / 'Results/workspace_model.png'
    parameters = {'workspace': [config.width, config.height], 'seed': config.seed,
                  'initial_positions': positions.tolist(), 'obstacles': config.obstacles,
                  'robot_body_radius': config.robot_r, 'sensing_radius': config.sensing_r,
                  'robot_interaction_radius': config.robot_interact_r,
                  'discovery_search_radius': config.discovery_r}
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor='white',
                metadata={'Title': 'MAE 598 workspace model',
                          'Description': json.dumps(parameters)})
    plt.close(fig)
    print(output)
    print(json.dumps(parameters, indent=2))


if __name__ == '__main__':
    main()
