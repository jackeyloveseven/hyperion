import unittest
import numpy as np
from hyperion.metrics.spatial_coherence import compute_spatial_neighbor_correlation


class TestMetricsSpatial(unittest.TestCase):

    def test_spatial_neighbor_correlation_constant_field(self):
        field = np.ones((16, 16, 8), dtype=np.float32)
        rho = compute_spatial_neighbor_correlation(field, distance=1)
        self.assertTrue(np.isclose(rho, 1.0, atol=1e-4))

    def test_spatial_neighbor_correlation_random_field(self):
        rng = np.random.default_rng(42)
        field = rng.normal(0, 1, size=(32, 32, 16)).astype(np.float32)
        rho = compute_spatial_neighbor_correlation(field, distance=1)
        self.assertLess(abs(rho), 0.15)

    def test_spatial_neighbor_correlation_distance_parameter(self):
        rng = np.random.default_rng(42)
        field = rng.normal(0, 1, size=(32, 32, 8)).astype(np.float32)
        rho1 = compute_spatial_neighbor_correlation(field, distance=1)
        rho2 = compute_spatial_neighbor_correlation(field, distance=2)
        self.assertGreaterEqual(rho1, -1.0)
        self.assertLessEqual(rho1, 1.0)
        self.assertGreaterEqual(rho2, -1.0)
        self.assertLessEqual(rho2, 1.0)

    def test_spatial_neighbor_correlation_2d_field(self):
        field_2d = np.ones((16, 16), dtype=np.float32)
        rho = compute_spatial_neighbor_correlation(field_2d, distance=1)
        self.assertTrue(np.isclose(rho, 1.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
