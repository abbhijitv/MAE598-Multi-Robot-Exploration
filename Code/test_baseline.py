# import statements
import unittest
import numpy as np
from config import Config
from controllers import repulsion, baseline_forces
from environment import Environment
from robots import advance
from simulation import run_simulation


# baseline tests class
class BaselineTests(unittest.TestCase):
    # function to test robot repulsion direction and value
    def test_robot_repulsion_direction_and_value(self):
        c = Config(obstacles=[], robot_c=2)
        env = Environment(c)
        p = np.array([[8.0, 10.0], [10.0, 10.0]])
        force = baseline_forces(p, env, c)
        np.testing.assert_allclose(force[0], [-2 * c.robot_gain / (8 + c.epsilon), 0])
        np.testing.assert_allclose(force[1], -force[0])

    # function to test obstacle repulsion direction
    def test_obstacle_repulsion_direction(self):
        c = Config(obstacles=[(10, 10, 1)])
        force = baseline_forces(np.array([[11.5, 10.0]]), Environment(c), c)
        self.assertGreater(force[0, 0], 0)
        self.assertAlmostEqual(force[0, 1], 0, places=10)

    # function to test repulsion limits and finite values near zero
    def test_cutoff_and_finite_near_zero(self):
        for distance in (0.0, 1e-15, 1e-8, 0.5):
            f = repulsion(np.array([distance, 0.0]), np.array([[0.0, 0.0]]), 2, 4, 1e-6)
            self.assertTrue(np.isfinite(f).all())
        np.testing.assert_array_equal(repulsion(np.zeros(2), np.array([[4.1, 0]]),
                                               2, 4, 1e-6), [0, 0])
        self.assertLess(repulsion(np.zeros(2), np.array([[4.0, 0]]), 2, 4, 1e-6)[0], 0)

    # function to test that robots do not pass through obstacles
    def test_swept_obstacle_prevents_tunneling(self):
        c = Config(obstacles=[(10, 10, 0.5)], robot_c=1)
        env = Environment(c)
        p = np.array([[5.0, 10.0]])
        move = np.array([[10.0, 0.0]])
        fraction = env.safe_fraction(p, move)
        self.assertLess(fraction, 0.5)
        self.assertTrue(env.valid_positions(p + fraction * move))

    # function to test that robots do not pass through each other
    def test_swept_robot_swap_prevented(self):
        c = Config(obstacles=[], robot_c=2)
        env = Environment(c)
        p = np.array([[5.0, 10.0], [15.0, 10.0]])
        move = np.array([[10.0, 0.0], [-10.0, 0.0]])
        fraction = env.safe_fraction(p, move)
        self.assertLess(fraction, 0.5)
        self.assertTrue(env.valid_positions(p + fraction * move))

    # function to test that robots stop before leaving the environment
    def test_boundary_motion_and_stop(self):
        c = Config(obstacles=[], robot_c=1)
        env = Environment(c)
        p, v, blocked = advance(np.array([[19.79, 10.0]]), np.array([[1.5, 0.0]]),
                                np.zeros((1, 2)), env, c, c.dt)
        self.assertTrue(blocked)
        self.assertTrue(env.valid_positions(p))
        np.testing.assert_array_equal(v, np.zeros((1, 2)))

    # function to test damped movement and speed and acceleration limits
    def test_damped_update_and_caps(self):
        c = Config(obstacles=[], robot_c=1)
        env = Environment(c)
        p = np.array([[10.0, 10.0]])
        v = np.array([[1.0, 0.0]])
        new_p, new_v, _ = advance(p, v, np.zeros((1, 2)), env, c, c.dt)
        np.testing.assert_allclose(new_v, v * (1 - c.damping * c.dt / c.mass))
        np.testing.assert_allclose(new_p, p + c.dt * new_v)
        _, new_v, _ = advance(p, v, np.array([[1e10, 1e10]]), env, c, c.dt)
        self.assertLessEqual(np.linalg.norm(new_v - v), c.max_acc * c.dt + 1e-12)
        self.assertLessEqual(np.linalg.norm(new_v), c.max_speed)

    # function to test coverage calculation from explored cells
    def test_coverage_definition(self):
        c = Config(width=4, height=4, grid_size=1, obstacles=[], sensing_r=0.1)
        env = Environment(c)
        self.assertEqual(env.sense(np.array([[0.5, 0.5]])), 100 / 16)
        self.assertEqual(env.sense(np.array([[0.5, 0.5]])), 100 / 16)

    # function to test safe movement, coverage and reproducible results
    def test_simulation_invariants_and_reproducibility(self):
        c = Config(duration=8)
        first = run_simulation(c)
        second = run_simulation(c)
        for key in ['positions', 'velocities', 'coverage', 'explored']:
            np.testing.assert_array_equal(first[key], second[key])
        for p in first['positions']:
            self.assertTrue(first['environment'].valid_positions(p))
        self.assertTrue(np.all((first['coverage'] >= 0) & (first['coverage'] <= 100)))
        self.assertTrue(np.all(np.diff(first['explored'].sum(axis=(1, 2))) >= 0))
        self.assertTrue(np.all(np.linalg.norm(first['velocities'], axis=2) <= c.max_speed + 1e-12))

    # function to test that overlapping initial positions are rejected
    def test_invalid_initial_overlap_rejected(self):
        with self.assertRaises(ValueError):
            run_simulation(Config(robot_c=2, duration=0.1), positions=[[4, 4], [4, 4]])

    # function to test the final step for a fractional duration
    def test_fractional_last_step(self):
        result = run_simulation(Config(duration=0.12, dt=0.05))
        np.testing.assert_allclose(result['time'], [0, 0.05, 0.10, 0.12])

    # function to test that coverage does not affect baseline forces
    def test_baseline_does_not_read_coverage(self):
        c = Config()
        env = Environment(c)
        p = np.array([[3.0, 8.0], [4.0, 9.0]])
        before = baseline_forces(p, env, c)
        env.explored[:] = True
        np.testing.assert_array_equal(before, baseline_forces(p, env, c))


if __name__ == '__main__':
    unittest.main()
