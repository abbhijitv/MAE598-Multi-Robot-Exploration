# import statements
import unittest
import json
from pathlib import Path
import numpy as np
from config import Config
from controllers import discovery_target, discovery_force, baseline_forces
from environment import Environment
from simulation import run_simulation, update_local_maps
from validation import validate_result


# discovery tests class
class DiscoveryTests(unittest.TestCase):
    # function to set up the discovery test environment
    def setUp(self):
        self.config = Config(controller='discovery', robot_c=1, obstacles=[])
        self.environment = Environment(self.config)
        self.position = np.array([[5.0, 10.0]])
        shape = (1,) + self.environment.free.shape
        self.known_free = np.zeros(shape, dtype=bool)
        self.explored = np.zeros(shape, dtype=bool)

    # function to test discovery force direction and value
    def test_force_direction_and_weight(self):
        force = discovery_force(np.array([1.0, 1.0]), np.array([4.0, 5.0]), self.config)
        np.testing.assert_allclose(force, self.config.discovery_gain * np.array([0.6, 0.8]))

    # function to test selection of the nearest eligible cell
    def test_nearest_eligible_cell_selected(self):
        x = np.array([[1.0, 3.0, 2.0]])
        y = np.zeros_like(x)
        known = np.array([[False, True, True]])
        explored = np.array([[False, False, False]])
        target = discovery_target(np.zeros(2), x, y, known, explored, self.config)
        np.testing.assert_array_equal(target, [2, 0])
        explored[0, 2] = True
        target = discovery_target(np.zeros(2), x, y, known, explored, self.config)
        np.testing.assert_array_equal(target, [3, 0])

    # function to test that obstacles and explored cells are excluded
    def test_obstacles_and_explored_cells_excluded(self):
        c = Config(controller='discovery', robot_c=1, obstacles=[(8, 10, 1)])
        env = Environment(c)
        known = np.zeros((1,) + env.free.shape, dtype=bool)
        explored = np.zeros_like(known)
        update_local_maps(self.position, env, known, explored, c)
        self.assertFalse((known[0] & ~env.free).any())
        for step in range(40):
            target = discovery_target(self.position[0], env.x, env.y, known[0], explored[0], c)
            self.assertIsNotNone(target)
            index = np.argmin((env.x - target[0]) ** 2 + (env.y - target[1]) ** 2)
            self.assertTrue(env.free.ravel()[index])
            self.assertFalse(explored[0].ravel()[index])
            explored[0].ravel()[index] = True

    # function to test discovery with no eligible cells
    def test_empty_or_fully_explored_map(self):
        env = self.environment
        target = discovery_target(self.position[0], env.x, env.y,
                                  self.known_free[0], self.explored[0], self.config)
        self.assertIsNone(target)
        self.known_free[:] = True
        self.explored[:] = True
        self.assertIsNone(discovery_target(self.position[0], env.x, env.y,
                                          self.known_free[0], self.explored[0], self.config))
        np.testing.assert_array_equal(discovery_force(self.position[0], None, self.config), [0, 0])

    # function to test finite discovery forces near zero
    def test_finite_near_zero_force(self):
        for distance in (0, 1e-15, 1e-9, 1.0):
            force = discovery_force(np.zeros(2), np.array([distance, 0]), self.config)
            self.assertTrue(np.isfinite(force).all())
            self.assertLessEqual(np.linalg.norm(force), self.config.discovery_gain + 1e-12)

    # function to test local occupancy and exploration sensing ranges
    def test_private_sensor_ranges(self):
        env = self.environment
        update_local_maps(self.position, env, self.known_free, self.explored, self.config)
        distances = np.hypot(env.x - 5, env.y - 10)
        np.testing.assert_array_equal(self.known_free[0], distances <= self.config.discovery_r)
        np.testing.assert_array_equal(self.explored[0], distances <= self.config.sensing_r)
        self.assertFalse(self.known_free[0][distances > self.config.discovery_r].any())

    # function to test that target selection uses only the robots private map
    def test_no_other_robot_map_or_global_coverage_input(self):
        env = self.environment
        p = np.array([[5.0, 10.0], [17.0, 10.0]])
        known = np.zeros((2,) + env.free.shape, dtype=bool)
        explored = np.zeros_like(known)
        update_local_maps(p, env, known, explored, self.config)
        right = env.x > 15
        self.assertFalse(explored[0][right].any())
        self.assertTrue(explored[1][right].any())
        before = discovery_target(p[0], env.x, env.y, known[0], explored[0], self.config)
        env.explored[:] = True
        explored[1][:] = True
        after = discovery_target(p[0], env.x, env.y, known[0], explored[0], self.config)
        np.testing.assert_array_equal(before, after)

    # function to test that distant obstacles do not affect local decisions
    def test_hidden_obstacles_do_not_affect_local_decisions(self):
        c1 = Config(controller='discovery', robot_c=1, obstacles=[])
        c2 = Config(controller='discovery', robot_c=1, obstacles=[(17, 10, 1)])
        targets = []
        forces = []
        for c in (c1, c2):
            env = Environment(c)
            known = np.zeros((1,) + env.free.shape, dtype=bool)
            explored = np.zeros_like(known)
            update_local_maps(self.position, env, known, explored, c)
            target = discovery_target(self.position[0], env.x, env.y, known[0], explored[0], c)
            targets.append(target)
            forces.append(baseline_forces(self.position, env, c)[0]
                          + discovery_force(self.position[0], target, c))
        np.testing.assert_array_equal(targets[0], targets[1])
        np.testing.assert_array_equal(forces[0], forces[1])

    # function to test that targets outside the search range are excluded
    def test_targets_outside_search_range_excluded(self):
        target = discovery_target(np.zeros(2), np.array([[6.0]]), np.array([[0.0]]),
                                  np.array([[True]]), np.array([[False]]), self.config)
        self.assertIsNone(target)

    # function to test discovery safety, coverage and reproducible results
    def test_discovery_safety_coverage_and_reproducibility(self):
        c = Config(controller='discovery', duration=12)
        first = run_simulation(c)
        second = run_simulation(c)
        for key in ['positions', 'velocities', 'coverage', 'explored', 'targets', 'local_explored']:
            np.testing.assert_array_equal(first[key], second[key])
        env = first['environment']
        for p, next_p in zip(first['positions'][:-1], first['positions'][1:]):
            self.assertTrue(env.valid_positions(p))
            self.assertEqual(env.safe_fraction(p, next_p - p), 1.0)
        self.assertTrue(env.valid_positions(first['positions'][-1]))
        self.assertTrue(np.isfinite(first['velocities']).all())
        self.assertTrue(np.all(np.linalg.norm(first['velocities'], axis=2) <= c.max_speed + 1e-12))
        self.assertTrue(np.all((first['coverage'] >= 0) & (first['coverage'] <= 100)))
        self.assertTrue(np.all(np.diff(first['explored'].sum(axis=(1, 2))) >= 0))
        np.testing.assert_array_equal(first['local_explored'].any(axis=0), env.explored)
        self.assertEqual(first['safety_events'].sum(), first['safety_interventions'])

    # function to test that zero discovery gain reproduces the baseline
    def test_zero_gain_reproduces_baseline(self):
        baseline = run_simulation(Config(duration=5))
        discovery = run_simulation(Config(controller='discovery', discovery_gain=0, duration=5))
        for key in ['positions', 'velocities', 'coverage', 'explored']:
            np.testing.assert_array_equal(baseline[key], discovery[key])

    # function to test the baseline against saved results
    def test_baseline_matches_preserved_sixty_second_results(self):
        folder = Path(__file__).resolve().parents[1] / 'Results' / 'baseline'
        # saved baseline results for the comparison
        saved = np.genfromtxt(folder / 'robot_positions.csv', delimiter=',', names=True)
        coverage = np.genfromtxt(folder / 'coverage.csv', delimiter=',', names=True)
        parameters = json.loads((folder / 'parameters.json').read_text())
        config_names = {'robot_count': 'robot_c', 'robot_radius': 'robot_r',
                        'sensing_radius': 'sensing_r', 'robot_interaction_radius': 'robot_interact_r',
                        'obstacle_interaction_radius': 'obstacle_interact_r', 'max_acceleration': 'max_acc',
                        'obstacle_point_spacing': 'obstacle_boundary_sampling', 'discovery_radius': 'discovery_r'}
        config = Config(controller='baseline', **{config_names.get(name, name): value
                        for name, value in parameters.items() if name != 'controller'
                        and config_names.get(name, name) in Config.__dataclass_fields__})
        result = run_simulation(config)
        p = np.column_stack((saved['x'], saved['y'])).reshape(result['positions'].shape)
        v = np.column_stack((saved['vx'], saved['vy'])).reshape(result['velocities'].shape)
        np.testing.assert_allclose(result['positions'], p, rtol=0, atol=1e-13)
        np.testing.assert_allclose(result['velocities'], v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(result['coverage'], coverage['coverage_percent'], rtol=0, atol=1e-13)

    # function to test that validation detects movement through obstacles
    def test_validation_rejects_obstacle_tunneling(self):
        c = Config(controller='discovery', robot_c=1, duration=0.05,
                   obstacles=[(10, 10, 0.5)])
        result = run_simulation(c, positions=[[5, 10]])
        result['positions'][-1, 0] = [15, 10]
        self.assertTrue(result['environment'].valid_positions(result['positions'][-1]))
        with self.assertRaisesRegex(RuntimeError, 'swept_obstacle_clearance'):
            validate_result(result, c)

    # function to test that invalid discovery settings are rejected
    def test_invalid_discovery_configuration(self):
        for options in [{'controller': 'other'}, {'discovery_gain': -1},
                        {'discovery_r': 0}, {'discovery_gain': float('nan')}]:
            with self.assertRaises(ValueError):
                Config(**options).validate()


if __name__ == '__main__':
    unittest.main()
