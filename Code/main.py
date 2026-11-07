# import statements
import argparse
import json
from pathlib import Path
from config import Config
from simulation import run_simulation
from visualization import save_results
from validation import validate_result


# function to run the selected controller
def main():
    # simulation settings from the command line
    parser = argparse.ArgumentParser(description='Baseline or discovery multi-robot simulation')
    parser.add_argument('--controller', choices=['baseline', 'discovery'], default='baseline')
    parser.add_argument('--discovery-gain', type=float, default=1.2)
    parser.add_argument('--discovery-radius', type=float, default=5.0)
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=598)
    parser.add_argument('--robots', type=int, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-animation', action='store_true')
    args = parser.parse_args()
    results_root = Path(__file__).resolve().parents[1] / 'Results'
    output = (args.output or results_root / args.controller).resolve()
    # check the output folder before saving results
    if results_root not in output.parents:
        parser.error('--output must be a subfolder of Results')
    if output.exists() and any(output.iterdir()):
        parser.error('Output folder is not empty; choose a new Results subfolder')
    config = Config(duration=args.duration, dt=args.dt, seed=args.seed,
                    robot_c=args.robots, controller=args.controller,
                    discovery_gain=args.discovery_gain, discovery_r=args.discovery_radius)
    result = run_simulation(config)
    validation = validate_result(result, config) if config.controller == 'discovery' else None
    save_results(result, config, output, animation=not args.no_animation)
    if validation is not None:
        (output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Final coverage: {:.2f}%'.format(result['coverage'][-1]))
    print('Safety interventions:', result['safety_interventions'])
    print('Saved results to', output)


if __name__ == '__main__':
    main()
