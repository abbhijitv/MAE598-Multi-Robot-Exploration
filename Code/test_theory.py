# import statements
import unittest
import tempfile
from pathlib import Path
from comparison import resolve_result_folder
import numpy as np
from config import Config
from controllers import baseline_forces
from environment import Environment
from theoretical_analysis import radial_potential, potential_energy, potential_hessian
from theoretical_analysis import equilibrium_case, fixed_field_trajectory


# theoretical analysis tests class
class TheoryTests(unittest.TestCase):
    # function to test the derivative of the regularized potential
    def test_regularized_potential_derivative(self):
        gain, radius, epsilon = 0.35, 2.0, 1e-6
        for distance in (0.05, 0.5, 1.5):
            step = 1e-6
            derivative = (radial_potential(distance + step, gain, radius, epsilon)
                          - radial_potential(distance - step, gain, radius, epsilon)) / (2 * step)
            self.assertAlmostEqual(derivative, -gain * distance / (distance ** 3 + epsilon), delta=1e-6)

    # function to test finite potential and zero values outside the range
    def test_potential_finite_and_zero_beyond_cutoff(self):
        self.assertTrue(np.isfinite(radial_potential(0, 2, 4, 1e-6)))
        self.assertEqual(radial_potential(4, 2, 4, 1e-6), 0)
        self.assertEqual(radial_potential(5, 2, 4, 1e-6), 0)
        self.assertGreater(radial_potential(1.99, 2, 2, 1e-6), 0)

    # function to test that robot forces match the negative energy gradient
    def test_multirobot_force_is_negative_energy_gradient(self):
        config = Config(width=3, height=3, robot_c=2, obstacles=[])
        env = Environment(config)
        positions = np.array([[1.2, 1.3], [1.9, 1.65]])
        force = baseline_forces(positions, env, config)
        for i in range(2):
            for axis in range(2):
                plus, minus = positions.copy(), positions.copy()
                plus[i, axis] += 1e-5
                minus[i, axis] -= 1e-5
                gradient = (potential_energy(plus, env, config)
                            - potential_energy(minus, env, config)) / 2e-5
                self.assertAlmostEqual(gradient, -force[i, axis], delta=1e-7)

    # function to test the potential hessian against force derivatives
    def test_analytic_hessian_matches_force_jacobian(self):
        config, env, equilibrium, certificate = equilibrium_case()
        active = np.array(certificate['active_wall_points'])
        hessian = potential_hessian(equilibrium, active, config.obstacle_gain, config.epsilon)
        for axis in range(2):
            offset = np.eye(2)[axis] * 1e-5
            derivative = (baseline_forces((equilibrium + offset)[None, :], env, config)[0]
                          - baseline_forces((equilibrium - offset)[None, :], env, config)[0]) / 2e-5
            np.testing.assert_allclose(derivative, -hessian[:, axis], atol=1e-8)

    # function to test the equilibrium and active wall point distances
    def test_equilibrium_and_exact_shell_structure(self):
        config, env, equilibrium, certificate = equilibrium_case()
        self.assertEqual(certificate['active_point_count'], 20)
        self.assertLess(certificate['equilibrium_force_norm'], 1e-14)
        distances = np.linalg.norm(equilibrium - np.array(certificate['active_wall_points']), axis=1)
        values, counts = np.unique(distances, return_counts=True)
        np.testing.assert_allclose(values, [1.5, np.sqrt(2.5), np.sqrt(3.25)])
        np.testing.assert_array_equal(counts, [4, 8, 8])
        self.assertTrue((np.linalg.eigvalsh(certificate['hessian']) > 0.79).all())

    # function to test the neighborhood and energy bounds from the analysis
    def test_proved_neighborhood_and_sublevel_bounds(self):
        config, env, equilibrium, certificate = equilibrium_case()
        # check the bounds from the theoretical analysis
        self.assertLess(certificate['hessian_lipschitz_upper_bound_computed'], 43)
        self.assertGreater(certificate['cutoff_distance_margin_at_center'], 0.01)
        self.assertLess(certificate['invariant_sublevel_energy'], certificate['position_boundary_energy_lower_bound'])
        for angle in np.linspace(0, 2 * np.pi, 17):
            point = equilibrium + 0.01 * np.array([np.cos(angle), np.sin(angle)])
            self.assertTrue(env.valid_positions(point[None, :]))
            hessian = potential_hessian(point, np.array(certificate['active_wall_points']),
                                          config.obstacle_gain, config.epsilon)
            self.assertGreater(np.linalg.eigvalsh(hessian).min(), 0.36)

    # function to test the energy derivative with damping
    def test_energy_derivative_algebra(self):
        config, env, equilibrium, _ = equilibrium_case()
        position = equilibrium + [0.003, -0.002]
        velocity = np.array([0.001, -0.0005])
        force = baseline_forces(position[None, :], env, config)[0]
        acceleration = (force - config.damping * velocity) / config.mass
        derivative = np.dot(-force, velocity) + config.mass * np.dot(velocity, acceleration)
        self.assertAlmostEqual(derivative, -config.damping * np.dot(velocity, velocity), places=15)

    # function to test recovery to equilibrium and decreasing energy
    def test_continuous_recovery_and_energy(self):
        config, env, equilibrium, certificate = equilibrium_case()
        times, positions, velocities = fixed_field_trajectory(
            config, env, equilibrium + [0.003, -0.002], np.array([0.001, -0.0005]))
        energies = np.array([potential_energy(p[None, :], env, config)
                             + 0.5 * config.mass * np.dot(v, v)
                             for p, v in zip(positions[::20], velocities[::20])])
        self.assertTrue((np.diff(energies) < 1e-11).all())
        self.assertLess(np.linalg.norm(positions[-1] - equilibrium), 1e-7)
        self.assertLess(np.linalg.norm(velocities[-1]), 1e-7)
        self.assertLess(np.linalg.norm(positions - equilibrium, axis=1).max(), 0.01)

    # function to test saved result paths after moving the project
    def test_relocated_historical_experiment_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / 'robot_counts/robots_5/baseline'
            local.mkdir(parents=True)
            old = '/previous_machine/Results/experiments/robot_counts/robots_5/baseline'
            self.assertEqual(resolve_result_folder(root, old), local)
            self.assertEqual(resolve_result_folder(root, 'robot_counts/robots_5/baseline'), local)

    # function to test that missing result folders are rejected
    def test_missing_experiment_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_result_folder(Path(directory), '/missing/Results/experiments/robot_counts/robots_5/baseline')

    # function to test that nearby flat field positions are equilibria
    def test_flat_equilibrium_is_not_isolated(self):
        config = Config(robot_c=1, obstacles=[])
        env = Environment(config)
        for position in (np.array([[10.0, 10.0]]), np.array([[10.1, 10.1]])):
            np.testing.assert_array_equal(baseline_forces(position, env, config), [[0, 0]])
            self.assertEqual(potential_energy(position, env, config), 0)


if __name__ == '__main__':
    unittest.main()
