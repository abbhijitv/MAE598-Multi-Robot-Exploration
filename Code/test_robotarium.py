# import statements
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
import numpy as np
from environment import Environment
from robotarium_validation import platform_config, initial_poses, to_project, to_robotarium
from robotarium_validation import run_robotarium, safety_metrics, save_run, robotarium_api, verify_runs


# optional simulator tests
@unittest.skipUnless(importlib.util.find_spec('rps') and importlib.util.find_spec('cvxopt'),
                     'Optional Robotarium dependencies are not installed')
class RobotariumTests(unittest.TestCase):
    # function to test the installed official API
    def test_official_package_import(self):
        Robotarium, mapping, barrier = robotarium_api()
        self.assertEqual(Robotarium.__module__, 'rps.robotarium')
        self.assertEqual(Robotarium.TIME_STEP, 0.033)
        self.assertTrue(callable(mapping) and callable(barrier))

    # function to test the coordinate transformation
    def test_coordinate_round_trip(self):
        c = platform_config()
        points = np.array([[0., 0.], [10., 10.], [20., 20.]])
        np.testing.assert_allclose(to_project(to_robotarium(points, c, .09), c, .09), points)
        np.testing.assert_allclose(to_robotarium(points, c, .09), [[-.9, -.9], [0., 0.], [.9, .9]])
        self.assertAlmostEqual(c.robot_r * .09, .095)
        self.assertAlmostEqual(c.sensing_r * .09, .225)

    # function to test the projected-point velocity mapping
    def test_mapping_equations(self):
        Robotarium, factory, barrier = robotarium_api()
        forward, project = factory(projection_distance=.03)
        states = np.array([[0., .4], [0., .4], [.2, -.6]])
        desired = np.array([[.02, -.03], [.01, .025]])
        command = forward(desired, states)
        theta = states[2]
        actual = np.vstack((command[0] * np.cos(theta) - .03 * command[1] * np.sin(theta),
                            command[0] * np.sin(theta) + .03 * command[1] * np.cos(theta)))
        np.testing.assert_allclose(actual, desired, atol=1e-14)
        self.assertEqual(project(states).shape, (2, 2))

    # function to test deterministic safe starting poses
    def test_initialization(self):
        for count in (3, 5, 8):
            c = platform_config(count=count)
            first = initial_poses(c)
            np.testing.assert_array_equal(first, initial_poses(c))
            self.assertEqual(first.shape, (3, count))
            self.assertTrue(Environment(c).valid_positions(to_project(first[:2].T, c, .09)))
            distances = np.linalg.norm(first[:2, :, None] - first[:2, None, :], axis=0)
            self.assertGreater(distances[np.triu_indices(count, 1)].min(), .25)

    # function to test finite states and physical command limits
    def test_commands_and_state_dimensions(self):
        c = platform_config(count=3, duration=.33)
        r = run_robotarium(replace(c, controller='discovery'))
        self.assertEqual(r['poses'].shape, (11, 3, 3))
        self.assertEqual(r['commands'].shape, (10, 3, 2))
        self.assertTrue(np.isfinite(r['commands']).all())
        wheels = np.stack((r['commands'][:, :, 0] - .055 * r['commands'][:, :, 1],
                           r['commands'][:, :, 0] + .055 * r['commands'][:, :, 1]))
        self.assertLessEqual(np.abs(wheels).max(), .2 + 1e-10)
        self.assertEqual(r['summary']['robot_collision_intervals'], 0)
        self.assertFalse(r['summary']['boundary_violation'])
        self.assertEqual(r['native_errors'], {})

    # function to test coverage from actual robot positions
    def test_coverage_definition(self):
        c = platform_config(count=3, duration=.132)
        r = run_robotarium(c)
        env = Environment(c)
        for i, state in enumerate(r['poses']):
            expected = env.sense(to_project(state[:2].T, c, .09))
            self.assertEqual(r['coverage'][i], expected)
        self.assertTrue((np.diff(r['coverage']) >= 0).all())

    # function to test paired initial conditions
    def test_paired_fairness(self):
        c = platform_config(count=3, duration=.132)
        poses = initial_poses(c)
        baseline = run_robotarium(c, poses=poses)
        discovery = run_robotarium(replace(c, controller='discovery'), poses=poses)
        np.testing.assert_array_equal(baseline['poses'][0], discovery['poses'][0])
        self.assertEqual(baseline['coverage'][0], discovery['coverage'][0])

    # function to test zero discovery gain against baseline
    def test_zero_discovery_gain(self):
        c = platform_config(count=3, duration=.33)
        baseline = run_robotarium(c)
        discovery = run_robotarium(replace(c, controller='discovery', discovery_gain=0))
        np.testing.assert_array_equal(baseline['poses'], discovery['poses'])
        np.testing.assert_array_equal(baseline['coverage'], discovery['coverage'])

    # function to test independent collision detection
    def test_safety_detection(self):
        c = platform_config(count=2, duration=.132)
        states = np.zeros((2, 3, 2))
        report = safety_metrics(states, Environment(c), c, .09)
        self.assertGreater(report['robot_collision_intervals'], 0)

    # function to test the swept obstacle guard with an inward command
    def test_obstacle_command_guard(self):
        c = platform_config(count=1, duration=1.65, layout='obstacles')
        poses = np.array([[-.321], [-.27], [0.]])
        with patch('robotarium_validation.baseline_forces', return_value=np.array([[100., 0.]])):
            result = run_robotarium(c, poses=poses)
        self.assertGreater(result['summary']['swept_guard_steps'], 0)
        self.assertGreaterEqual(result['summary']['minimum_body_obstacle_clearance_m'], -1e-9)
        self.assertEqual(result['summary']['obstacle_collision_intervals'], 0)

    # function to test near-boundary and near-neighbor motion
    def test_safety_stress(self):
        c = platform_config(count=2, duration=.99)
        poses = np.array([[.804, .600], [.600, .600], [0., 0.]])
        result = run_robotarium(replace(c, controller='discovery'), poses=poses)
        self.assertGreater(result['summary']['barrier_intervention_steps'], 0)
        self.assertGreaterEqual(result['summary']['minimum_body_boundary_clearance_m'], 0)
        self.assertGreaterEqual(result['summary']['minimum_center_separation_m'], .19)
        self.assertEqual(result['native_errors'], {})

    # function to test repeatable simulation and saved outputs
    def test_repeatability_and_outputs(self):
        c = platform_config(count=3, duration=.132, layout='obstacles')
        first = run_robotarium(c)
        second = run_robotarium(c)
        np.testing.assert_array_equal(first['poses'], second['poses'])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'run'
            save_run(first, c, folder, .09, .03)
            self.assertTrue((folder / 'states.csv').is_file())
            self.assertTrue((folder / 'inputs.csv').is_file())
            self.assertTrue((folder / 'maps.npz').is_file())
            params = json.loads((folder / 'parameters.json').read_text())
            np.testing.assert_array_equal(params['initial_poses'], first['poses'][0])
            verification = Path(directory) / 'verification.json'
            verify_runs(Path(directory), verification)
            self.assertTrue(json.loads(verification.read_text())['runs'][0]['passed'])
            with self.assertRaises(FileExistsError):
                save_run(first, c, folder, .09, .03)

    # function to test timestep and pose validation
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            run_robotarium(platform_config(duration=.1))
        with self.assertRaises(ValueError):
            run_robotarium(platform_config(duration=.132), poses=np.zeros((2, 5)))

    # function to test that native sensor noise does not change the caller RNG
    def test_random_state_preserved(self):
        before = np.random.get_state()
        run_robotarium(platform_config(count=3, duration=.132))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == '__main__':
    unittest.main()
