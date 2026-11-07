# import statements
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
from config import Config
from environment import Environment
from robots import initial_positions
from simulation import run_simulation
from validation import validate_result
from visualization import save_results


RESULT_COLUMNS = ['experiment_category', 'scenario', 'controller', 'robot_count', 'seed',
                  'obstacle_layout', 'initial_coverage_percent', 'final_coverage_percent',
                  'coverage_gain_pp', 'total_travel_distance', 'collision_count',
                  'safety_interventions', 'status', 'time_50_s', 'time_75_s', 'time_90_s',
                  'threshold_50_status', 'threshold_75_status', 'threshold_90_status',
                  'result_folder', 'reused_run']


# obstacle layouts for the experiments
def obstacle_layouts():
    original = list(Config().obstacles)
    return {'original': original, 'open': [],
            'cluttered': original + [(7.0, 3.0, 1.0), (6.5, 16.5, 1.0),
                                     (14.5, 16.5, 1.0), (16.5, 11.0, 1.0),
                                     (11.0, 2.5, 1.0)]}


# robot counts, environments and random seeds
def experiment_cases():
    cases = []
    for count in (3, 5, 8):
        cases.append(('robot_counts', 'robots_' + str(count), count, 598, 'original'))
    for layout in ('original', 'open', 'cluttered'):
        cases.append(('environments', layout, 5, 598, layout))
    for seed in (598, 42, 123, 2026, 7):
        cases.append(('random_seeds', 'seed_' + str(seed), 5, seed, 'original'))
    return cases


# function to find when the coverage threshold is reached
def threshold_time(times, coverage, threshold):
    reached = np.flatnonzero(np.asarray(coverage) >= threshold)
    return float(times[reached[0]]) if len(reached) else None


# function to calculate the statistics across random seeds
def summarize_seeds(results):
    seeds = results[results.experiment_category == 'random_seeds']
    return seeds.groupby('controller', sort=False).agg(
        seed_count=('seed', 'count'),
        mean_final_coverage_percent=('final_coverage_percent', 'mean'),
        sample_std_final_coverage_percent=('final_coverage_percent', 'std'),
        mean_total_travel_distance=('total_travel_distance', 'mean')).reset_index()


# function to calculate coverage, travel and safety measurements
def run_metrics(result, config, validation):
    coverage = result['coverage']
    metrics = {'controller': config.controller, 'robot_count': config.robot_c,
               'seed': config.seed, 'initial_coverage_percent': float(coverage[0]),
               'final_coverage_percent': float(coverage[-1]),
               'coverage_gain_pp': float(coverage[-1] - coverage[0]),
               'total_travel_distance': float(np.linalg.norm(
                   np.diff(result['positions'], axis=0), axis=2).sum()),
               'collision_count': validation['robot_collision_intervals']
                                  + validation['obstacle_collision_intervals'],
               'safety_interventions': result['safety_interventions'], 'status': 'completed'}
    for threshold in (50, 75, 90):
        value = threshold_time(result['time'], coverage, threshold)
        metrics['time_' + str(threshold) + '_s'] = value
        metrics['threshold_' + str(threshold) + '_status'] = 'reached' if value is not None else 'not reached'
    return metrics


# function to run both controllers from the same positions
def run_pair(config, folder, cache=None):
    if cache is None:
        cache = {}
    config.validate()
    positions = initial_positions(Environment(config), config)
    velocities = np.zeros_like(positions)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for controller in ('baseline', 'discovery'):
        current = replace(config, controller=controller)
        values = asdict(current)
        key = json.dumps(values, sort_keys=True)
        output = folder / controller
        # reuse the results for an identical configuration
        if key in cache:
            cached = cache[key]
            np.testing.assert_array_equal(positions, cached['positions'])
            np.testing.assert_array_equal(velocities, cached['velocities'])
            output.mkdir()
            (output / 'reference.json').write_text(json.dumps(
                {'canonical_result_folder': cached['folder'], 'configuration': values,
                 'reason': 'Identical configuration already executed in this suite'}, indent=2) + '\n')
            metrics = dict(cached['metrics'])
            metrics['reused_run'] = True
        else:
            result = run_simulation(current, positions=positions, velocities=velocities)
            if result['time'][-1] != current.duration:
                raise RuntimeError('Simulation did not reach its requested duration')
            validation = validate_result(result, current)
            save_results(result, current, output, animation=False)
            (output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
            if controller == 'baseline':
                pd.DataFrame({'step_start_s': result['time'][:-1], 'step_end_s': result['time'][1:],
                              'safety_intervention': result['safety_events']}).to_csv(
                                  output / 'safety_events.csv', index=False)
            metrics = run_metrics(result, current, validation)
            metrics['result_folder'] = str(output.resolve())
            metrics['reused_run'] = False
            cache[key] = {'metrics': dict(metrics), 'positions': result['positions'][0].copy(),
                          'velocities': result['velocities'][0].copy(),
                          'folder': str(output.resolve())}
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        rows.append(metrics)
    configs = [asdict(replace(config, controller=name)) for name in ('baseline', 'discovery')]
    configs[0].pop('controller')
    configs[1].pop('controller')
    if configs[0] != configs[1] or rows[0]['initial_coverage_percent'] != rows[1]['initial_coverage_percent']:
        raise RuntimeError('Paired settings or initial coverage differ')
    fairness = {'shared_configuration_identical': True, 'shared_configuration': configs[0],
                'initial_positions_identical': True, 'initial_positions': positions.tolist(),
                'initial_velocities_identical': True, 'initial_velocities': velocities.tolist(),
                'coverage_method': 'Cumulative free-cell centers within sensing_r',
                'control_information': 'Discovery uses private maps and 5-unit local occupancy; baseline ignores occupancy lookahead',
                'no_replacement_of_invalid_initial_states': True}
    (folder / 'fairness.json').write_text(json.dumps(fairness, indent=2) + '\n')
    return rows


# function to run the experiment cases
def run_experiments(output, duration=60.0, short=False):
    output = Path(output)
    root = Path(__file__).resolve().parents[1] / 'Results'
    if root not in output.resolve().parents:
        raise ValueError('Experiment output must be a subfolder of Results')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Experiment output is not empty; choose a new folder')
    output.mkdir(parents=True, exist_ok=True)
    cases = [('smoke', 'original', 5, 598, 'original')] if short else experiment_cases()
    manifest = {'duration_s': duration, 'short_smoke_test': short,
                'default_configuration': asdict(Config(duration=duration)),
                'obstacle_layouts': obstacle_layouts(), 'cases': cases,
                'threshold_method': 'First recorded sample at or above threshold; no interpolation',
                'unreached_thresholds': 'Empty numeric CSV field plus explicit not reached status',
                'statistics': 'Five seeds in original environment, sample standard deviation with ddof=1',
                'duplicate_policy': 'Execute once per exact configuration, reference canonical result',
                'completed': False}
    manifest_path = output / 'experiment_config.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    cache = {}
    rows = []
    for category, scenario, count, seed, layout in cases:
        config = Config(robot_c=count, seed=seed, obstacles=obstacle_layouts()[layout], duration=duration)
        try:
            pair = run_pair(config, output / category / scenario, cache)
        except Exception as error:
            (output / 'failure.json').write_text(json.dumps(
                {'category': category, 'scenario': scenario, 'status': 'failed',
                 'error': str(error), 'configuration': asdict(config)}, indent=2) + '\n')
            raise
        for row in pair:
            row.update(experiment_category=category, scenario=scenario, obstacle_layout=layout)
            rows.append(row)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output / 'all_results.csv', index=False)
        print(category, scenario, 'baseline {:.2f}%, discovery {:.2f}%'.format(
            pair[0]['final_coverage_percent'], pair[1]['final_coverage_percent']), flush=True)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not short:
        summarize_seeds(results).to_csv(output / 'summary_statistics.csv', index=False)
        from comparison import make_comparisons
        make_comparisons(output)
    manifest.update(completed=True, unique_simulation_runs=len(cache), comparison_entries=len(rows))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return results


# function to compare the simulations with the saved results
def verify_experiments(output, validation_output=None):
    from comparison import resolve_result_folder
    # configuration names used in the older saved results
    config_names = {'robot_count': 'robot_c', 'robot_radius': 'robot_r',
                    'sensing_radius': 'sensing_r', 'robot_interaction_radius': 'robot_interact_r',
                    'obstacle_interaction_radius': 'obstacle_interact_r', 'max_acceleration': 'max_acc',
                    'obstacle_point_spacing': 'obstacle_boundary_sampling', 'discovery_radius': 'discovery_r'}
    output = Path(output)
    table = pd.read_csv(output / 'all_results.csv')
    if list(table.columns) != RESULT_COLUMNS or len(table) != 22:
        raise RuntimeError('Unexpected experiment table schema or row count')
    reports = []
    for folder_name in table.result_folder.drop_duplicates():
        folder = resolve_result_folder(output, folder_name)
        parameters = json.loads((folder / 'parameters.json').read_text())
        config = Config(**{config_names.get(name, name): value for name, value in parameters.items()
                           if config_names.get(name, name) in Config.__dataclass_fields__})
        result = run_simulation(config)
        validation = validate_result(result, config)
        positions = pd.read_csv(folder / 'robot_positions.csv')
        coverage = pd.read_csv(folder / 'coverage.csv')
        np.testing.assert_allclose(result['positions'].reshape(-1, 2), positions[['x', 'y']], rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['velocities'].reshape(-1, 2), positions[['vx', 'vy']], rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['time'], coverage.time_s, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['coverage'], coverage.coverage_percent, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result['explored'].sum(axis=(1, 2)), coverage.explored_cells)
        grid = np.load(folder / 'coverage_grid.npz')
        np.testing.assert_array_equal(result['explored'][-1], grid['explored'])
        np.testing.assert_array_equal(result['environment'].free, grid['free'])
        expected = run_metrics(result, config, validation)
        for row in table[table.result_folder == folder_name].to_dict('records'):
            for key, value in expected.items():
                if value is None:
                    if not pd.isna(row[key]):
                        raise RuntimeError('Unreached threshold stored as a measured time')
                elif isinstance(value, str):
                    if row[key] != value:
                        raise RuntimeError('Incorrect experimental label')
                else:
                    np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-10)
        if config.controller == 'discovery':
            targets = pd.read_csv(folder / 'discovery_targets.csv')[['target_x', 'target_y']]
            np.testing.assert_allclose(result['targets'].reshape(-1, 2), targets,
                                       rtol=0, atol=1e-12, equal_nan=True)
            private = np.load(folder / 'local_maps.npz')
            np.testing.assert_array_equal(result['known_free'], private['known_free'])
            np.testing.assert_array_equal(result['local_explored'], private['explored'])
        saved_events = pd.read_csv(folder / 'safety_events.csv')
        np.testing.assert_array_equal(result['safety_events'], saved_events.safety_intervention)
        reports.append({'folder': str(folder), 'full_replay_matches_saved_data': True,
                        'validation': validation})
    for category, scenario, count, seed, layout in experiment_cases():
        pair = table[(table.experiment_category == category) & (table.scenario == scenario)]
        settings = []
        for row in pair.itertuples():
            parameters = json.loads((resolve_result_folder(output, row.result_folder) / 'parameters.json').read_text())
            parameters = {config_names.get(name, name): value for name, value in parameters.items()}
            settings.append({k: parameters[k] for k in Config.__dataclass_fields__ if k != 'controller'})
        if len(settings) != 2 or settings[0] != settings[1]:
            raise RuntimeError('Paired configurations do not match')
        starts = [pd.read_csv(resolve_result_folder(output, row.result_folder) / 'robot_positions.csv').iloc[:count]
                  for row in pair.itertuples()]
        np.testing.assert_array_equal(starts[0][['x', 'y', 'vx', 'vy']], starts[1][['x', 'y', 'vx', 'vy']])
    stats = pd.read_csv(output / 'summary_statistics.csv').set_index('controller')
    for controller in ('baseline', 'discovery'):
        values = table[(table.experiment_category == 'random_seeds')
                       & (table.controller == controller)].final_coverage_percent.to_numpy()
        if len(values) != 5:
            raise RuntimeError('Five seed trials are required')
        np.testing.assert_allclose(stats.loc[controller, 'mean_final_coverage_percent'], values.mean())
        np.testing.assert_allclose(stats.loc[controller, 'sample_std_final_coverage_percent'], values.std(ddof=1))
    report = {'completed': True, 'unique_full_run_replays': len(reports),
              'comparison_entries_checked': len(table), 'paired_cases_checked': 11,
              'paired_initial_states_and_configurations_match': True,
              'summary_mean_and_sample_std_checked_independently': True,
              'runs': reports}
    destination = Path(validation_output) if validation_output is not None else output / 'validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matched baseline and discovery experiments')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'Results' / 'experiments')
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--verification-output', type=Path)
    arguments = parser.parse_args()
    if arguments.verify:
        report = verify_experiments(arguments.output, arguments.verification_output)
        print('Verified', report['unique_full_run_replays'], 'full simulation replays')
    else:
        run_experiments(arguments.output, duration=5.0 if arguments.short else 60.0,
                        short=arguments.short)
