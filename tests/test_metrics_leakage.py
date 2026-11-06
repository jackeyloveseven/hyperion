import unittest
import torch
from hyperion.metrics.layout_leakage import (
    compute_patch_self_similarity_matrix,
    compute_layout_leakage_from_patches,
)


class TestMetricsLeakage(unittest.TestCase):

    def test_self_similarity_matrix(self):
        n, d = 32, 64
        patches = torch.randn(n, d)
        sim = compute_patch_self_similarity_matrix(patches)
        self.assertEqual(sim.shape, (n, n))
        diag = torch.diag(sim)
        self.assertTrue(torch.allclose(diag, torch.ones_like(diag), atol=1e-5))

    def test_layout_leakage_correlation(self):
        torch.manual_seed(42)
        n, d = 32, 64
        patches1 = torch.randn(n, d)
        leak_same = compute_layout_leakage_from_patches(patches1, patches1)
        self.assertAlmostEqual(leak_same, 1.0, places=4)

        patches2 = torch.randn(n, d)
        leak_diff = compute_layout_leakage_from_patches(patches1, patches2)
        self.assertGreaterEqual(leak_diff, -1.0)
        self.assertLessEqual(leak_diff, 1.0)

    def test_layout_leakage_unequal_patches(self):
        torch.manual_seed(42)
        patches1 = torch.randn(25, 64)
        patches2 = torch.randn(36, 64)
        leak = compute_layout_leakage_from_patches(patches1, patches2)
        self.assertGreaterEqual(leak, -1.0)
        self.assertLessEqual(leak, 1.0)


if __name__ == "__main__":
    unittest.main()
