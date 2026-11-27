# import statements
import unittest
from dataclasses import replace
from fractions import Fraction
import numpy as np
from controllers import baseline_forces
from theoretical_analysis import potential_energy
from multi_robot_theoretical_analysis import Interval, collective_hessian
from multi_robot_theoretical_analysis import equilibrium_case, fixed_field_trajectory, active_sets


# interacting robot theoretical tests class
class MultiRobotTheoryTests(unittest.TestCase):
    # function to prepare the equilibrium certificate
    @classmethod
    def setUpClass(cls):
        cls.config, cls.env, cls.equilibrium, cls.certificate = equilibrium_case()

    # function to test exact interval arithmetic and square root bounds
    def test_rational_interval_bounds(self):
        interval = Interval('-0.2', '0.3')
        self.assertEqual(interval.square().lower, 0)
        self.assertEqual(interval.square().upper, Fraction('0.09'))
        root = Interval(2).sqrt()
        self.assertLessEqual(root.lower ** 2, 2)
        self.assertGreaterEqual(root.upper ** 2, 2)
        quotient = Interval(2, 3) / Interval(4, 5)
        self.assertEqual(quotient.lower, Fraction(2, 5))
        self.assertEqual(quotient.upper, Fraction(3, 4))

        with self.assertRaises(ValueError):
            Interval(1) / Interval(-1, 1)

    # function to test equilibrium existence and the collective eigenvalue bounds
    def test_equilibrium_root_and_collective_certificate(self):
        c, env, equilibrium, certificate = self.config, self.env, self.equilibrium, self.certificate
        self.assertTrue(all(certificate['checks'].values()))
        self.assertGreater(certificate['left_endpoint_force_interval'][0], 0)
        self.assertLess(certificate['right_endpoint_force_interval'][1], 0)
        self.assertLess(np.linalg.norm(baseline_forces(equilibrium, env, c)), 1e-10)
        self.assertTrue(env.valid_positions(equilibrium))
        self.assertEqual(certificate['active_robot_pairs'], [[0, 1]])
        self.assertEqual(certificate['active_wall_point_count_per_robot'], 15)
        hessian = collective_hessian(equilibrium, env, c)
        self.assertGreater(np.linalg.norm(hessian[:2, 2:]), 1)
        intervals = sorted(certificate['collective_eigenvalue_intervals'])

        for value, (lower, upper) in zip(np.linalg.eigvalsh(hessian), intervals):
            self.assertGreater(value, lower)
            self.assertLess(value, upper)
            self.assertGreater(lower, 0.85)

    # function to test all collective force derivatives including pair coupling
    def test_collective_hessian_matches_force_jacobian(self):
        hessian = collective_hessian(self.equilibrium, self.env, self.config)

        for axis in range(4):
            offset = np.eye(4)[axis].reshape(2, 2) * 1e-6
            derivative = (baseline_forces(self.equilibrium + offset, self.env, self.config)
                          - baseline_forces(self.equilibrium - offset, self.env, self.config)) / 2e-6
            np.testing.assert_allclose(derivative.ravel(), -hessian[:, axis], atol=2e-9)

    # function to test the certified ball and unchanged interaction sets
    def test_neighborhood_hessian_and_active_sets(self):
        c, env, equilibrium, certificate = self.config, self.env, self.equilibrium, self.certificate
        radius = certificate['position_ball_radius']
        self.assertGreater(certificate['proved_hessian_lower_bound'], 0)
        self.assertLess(certificate['invariant_sublevel_energy'],
                        certificate['position_boundary_energy_lower_bound'])
        self.assertGreater(certificate['wall_cutoff_margin_at_equilibrium_lower_bound'], radius)
        self.assertGreater(certificate['body_pair_clearance_on_ball_lower_bound'], 1)
        reference, pair = active_sets(equilibrium, env, c)

        for direction in np.vstack((np.eye(4), -np.eye(4), [[0.5, -0.5, -0.5, 0.5]])):
            positions = equilibrium + radius * direction.reshape(2, 2)
            walls, active_pair = active_sets(positions, env, c)
            np.testing.assert_array_equal(walls, reference)
            self.assertEqual(active_pair, pair)
            self.assertTrue(env.valid_positions(positions))
            values = np.linalg.eigvalsh(collective_hessian(positions, env, c))
            self.assertGreater(values.min(), certificate['proved_hessian_lower_bound'])
            self.assertLess(values.max(), certificate['proved_hessian_upper_bound'])

    # function to test energy dissipation and recovery for interacting robots
    def test_continuous_energy_and_recovery(self):
        c, env, equilibrium, certificate = self.config, self.env, self.equilibrium, self.certificate
        offset = np.array([[0.0001, -0.00008], [-0.00007, 0.00009]])
        velocity = np.array([[0.00002, -0.00001], [-0.00001, 0.00002]])
        upper = (0.5 * certificate['proved_hessian_upper_bound']
                 * (np.linalg.norm(offset) + 2e-12) ** 2 + 0.5 * c.mass * np.sum(velocity ** 2))
        self.assertLess(upper, certificate['invariant_sublevel_energy'])
        times, positions, velocities = fixed_field_trajectory(c, env, equilibrium + offset, velocity)
        energy = np.array([potential_energy(p, env, c) + 0.5 * c.mass * np.sum(v ** 2)
                           for p, v in zip(positions[::10], velocities[::10])])
        self.assertLess(np.diff(energy).max(), 1e-11)
        self.assertLess(np.linalg.norm(positions[-1] - equilibrium), 1e-8)
        self.assertLess(np.linalg.norm(velocities[-1]), 1e-8)
        self.assertLess(np.linalg.norm(positions - equilibrium, axis=(1, 2)).max(),
                        certificate['position_ball_radius'])

        for p, v in zip(positions[::100], velocities[::100]):
            force = baseline_forces(p, env, c)
            acceleration = (force - c.damping * v) / c.mass
            derivative = np.sum(-force * v) + c.mass * np.sum(v * acceleration)
            self.assertAlmostEqual(derivative, -c.damping * np.sum(v ** 2), delta=1e-20)

    # function to test that the continuous integration does not use motion caps
    def test_continuous_dynamics_ignore_discrete_caps(self):
        c = replace(self.config, duration=0.1)
        capped = replace(c, max_speed=1e-12, max_acc=1e-12)
        position = self.equilibrium + [[0.0001, 0], [0, -0.0001]]
        velocity = np.array([[0.00002, 0], [0, 0.00002]])
        reference = fixed_field_trajectory(c, self.env, position, velocity)
        result = fixed_field_trajectory(capped, self.env, position, velocity)
        np.testing.assert_array_equal(reference[1], result[1])
        np.testing.assert_array_equal(reference[2], result[2])
        self.assertGreater(np.abs(result[2]).max(), capped.max_speed)


if __name__ == '__main__':
    unittest.main()
