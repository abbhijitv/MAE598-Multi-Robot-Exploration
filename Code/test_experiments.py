# import statements
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from config import Config
from environment import Environment
from experiments import RESULT_COLUMNS, experiment_cases, obstacle_layouts, threshold_time
from experiments import summarize_seeds, run_pair, run_metrics
from robots import initial_positions
from simulation import run_simulation
from validation import validate_result


# experiment tests class
class ExperimentTests(unittest.TestCase):
    # function to test the experiment robot counts and random seeds
    def test_robot_counts_and_seeds(self):
        cases = experiment_cases()
        self.assertEqual([c[2] for c in cases if c[0] == 'robot_counts'], [3, 5, 8])
        self.assertEqual([c[3] for c in cases if c[0] == 'random_seeds'], [598, 42, 123, 2026, 7])
        self.assertEqual(len(cases), 11)
        self.assertEqual(len(set((c[2], c[3], c[4]) for c in cases)), 9)

    # function to test the obstacle layouts and clearances
    def test_obstacle_layouts(self):
        layouts = obstacle_layouts()
        self.assertEqual(layouts['original'], Config().obstacles)
        self.assertEqual(layouts['open'], [])
        self.assertEqual(len(layouts['cluttered']), 8)
        self.assertEqual(layouts['cluttered'][:3], layouts['original'])
        for layout in layouts.values():
            Config(obstacles=layout).validate()
        for i, (x, y, radius) in enumerate(layouts['cluttered']):
            for x2, y2, radius2 in layouts['cluttered'][i + 1:]:
                self.assertGreater(np.hypot(x - x2, y - y2), radius + radius2 + 0.4)

    # function to test the first time a coverage threshold is reached
    def test_threshold_first_sample_and_equality(self):
        self.assertEqual(threshold_time([0, 1, 2, 3], [20, 49, 50, 80], 50), 2)
        self.assertEqual(threshold_time([0, 1], [50, 60], 50), 0)

    # function to test coverage thresholds that are not reached
    def test_unreached_threshold(self):
        self.assertIsNone(threshold_time([0, 60], [10, 49], 50))
        self.assertIsNone(threshold_time([], [], 90))

    # function to test the mean and sample standard deviation
    def test_sample_summary_statistics(self):
        rows = []
        for controller, values in [('baseline', [10, 20, 30, 40, 50]),
                                   ('discovery', [50, 60, 70, 80, 90])]:
            for seed, value in enumerate(values):
                rows.append({'experiment_category': 'random_seeds', 'controller': controller,
                             'seed': seed, 'final_coverage_percent': value, 'total_travel_distance': 12})
        rows.append({'experiment_category': 'robot_counts', 'controller': 'baseline',
                     'seed': 99, 'final_coverage_percent': 100, 'total_travel_distance': 12})
        summary = summarize_seeds(pd.DataFrame(rows)).set_index('controller')
        self.assertEqual(summary.loc['baseline', 'mean_final_coverage_percent'], 30)
        self.assertEqual(summary.loc['discovery', 'mean_final_coverage_percent'], 70)
        self.assertAlmostEqual(summary.loc['baseline', 'sample_std_final_coverage_percent'], np.sqrt(250))
        self.assertTrue((summary.seed_count == 5).all())

    # function to test matched starting conditions and reuse of results
    def test_paired_outputs_and_duplicate_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(duration=0.1)
            cache = {}
            rows = run_pair(config, root / 'first', cache)
            baseline = json.loads((root / 'first/baseline/parameters.json').read_text())
            discovery = json.loads((root / 'first/discovery/parameters.json').read_text())
            self.assertEqual(baseline['initial_positions'], discovery['initial_positions'])
            self.assertEqual(baseline['initial_velocities'], discovery['initial_velocities'])
            for name in Config.__dataclass_fields__:
                if name != 'controller':
                    self.assertEqual(baseline[name], discovery[name])
            for row in rows:
                row.update(experiment_category='smoke', scenario='test', obstacle_layout='original')
                self.assertEqual(set(row), set(RESULT_COLUMNS))
                self.assertEqual(row['threshold_90_status'], 'not reached')
            reused = run_pair(config, root / 'second', cache)
            self.assertTrue(all(row['reused_run'] for row in reused))
            self.assertEqual(len(cache), 2)
            self.assertTrue((root / 'second/baseline/reference.json').exists())

    # function to test valid and repeatable experiment starting positions
    def test_each_case_initialization_valid_and_repeatable(self):
        for category, name, count, seed, layout in experiment_cases():
            c = Config(robot_c=count, seed=seed, obstacles=obstacle_layouts()[layout])
            env = Environment(c)
            first = initial_positions(env, c)
            self.assertEqual(first.shape, (count, 2))
            self.assertTrue(env.valid_positions(first))
            np.testing.assert_array_equal(first, initial_positions(env, c))

    # function to test coverage limits and reproducible measurements
    def test_coverage_and_reproducible_measurements(self):
        for controller in ('baseline', 'discovery'):
            c = Config(controller=controller, duration=0.2, seed=42)
            first = run_simulation(c)
            second = run_simulation(c)
            np.testing.assert_array_equal(first['positions'], second['positions'])
            self.assertTrue(((first['coverage'] >= 0) & (first['coverage'] <= 100)).all())
            self.assertTrue((np.diff(first['coverage']) >= 0).all())
            metrics = run_metrics(first, c, validate_result(first, c))
            self.assertEqual(metrics['status'], 'completed')
            self.assertEqual(metrics['collision_count'], 0)

    # function to test saved result hashes and unchanged controller behavior
    def test_existing_controller_files_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        # saved file hashes for the comparison
        manifest = root / 'Results/experiments/preservation.json'
        if not manifest.exists():
            manifest = Path('/tmp/mae_step14_preserved.json')
        recorded = json.loads(manifest.read_text())
        config_names = {'robot_count': 'robot_c', 'robot_radius': 'robot_r',
                        'sensing_radius': 'sensing_r', 'robot_interaction_radius': 'robot_interact_r',
                        'obstacle_interaction_radius': 'obstacle_interact_r', 'max_acceleration': 'max_acc',
                        'obstacle_point_spacing': 'obstacle_boundary_sampling', 'discovery_radius': 'discovery_r'}
        for filename in ('baseline', 'discovery'):
            folder = root / 'Results' / filename
            for name in ('parameters.json', 'robot_positions.csv', 'coverage.csv'):
                name = 'Results/' + filename + '/' + name
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), recorded[name])
            parameters = json.loads((folder / 'parameters.json').read_text())
            config = Config(controller=filename, **{config_names.get(name, name): value
                            for name, value in parameters.items() if name != 'controller'
                            and config_names.get(name, name) in Config.__dataclass_fields__})
            saved = np.genfromtxt(folder / 'robot_positions.csv', delimiter=',', names=True)
            coverage = np.genfromtxt(folder / 'coverage.csv', delimiter=',', names=True)
            result = run_simulation(config)
            p = np.column_stack((saved['x'], saved['y'])).reshape(result['positions'].shape)
            v = np.column_stack((saved['vx'], saved['vy'])).reshape(result['velocities'].shape)
            np.testing.assert_allclose(result['positions'], p, rtol=0, atol=1e-13)
            np.testing.assert_allclose(result['velocities'], v, rtol=0, atol=1e-13)
            np.testing.assert_allclose(result['coverage'], coverage['coverage_percent'], rtol=0, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
