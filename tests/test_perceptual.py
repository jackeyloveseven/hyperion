import unittest
import numpy as np
import torch
from hyperion.metrics.perceptual import (
    compute_dino_style_similarity,
    compute_gram_distance,
    compute_frechet_distance,
    compute_artfid,
    _matrix_sqrt,
)


class TestPerceptual(unittest.TestCase):

    def test_dino_style_similarity(self):
        cls_out = torch.tensor([1.0, 0.0, 0.0])
        cls_style = torch.tensor([1.0, 0.0, 0.0])
        sim = compute_dino_style_similarity(cls_out, cls_style)
        self.assertAlmostEqual(sim, 1.0, places=4)

        cls_orth = torch.tensor([0.0, 1.0, 0.0])
        sim_orth = compute_dino_style_similarity(cls_out, cls_orth)
        self.assertAlmostEqual(sim_orth, 0.0, places=4)

    def test_gram_distance(self):
        g1 = [torch.ones((4, 4)), torch.ones((8, 8))]
        # Same gram matrices -> distance 0
        dist_same = compute_gram_distance(g1, g1)
        self.assertAlmostEqual(dist_same, 0.0, places=4)

        g2 = [torch.zeros((4, 4)), torch.zeros((8, 8))]
        dist_diff = compute_gram_distance(g1, g2)
        self.assertGreater(dist_diff, 0.0)

    def test_matrix_sqrt(self):
        a = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float32)
        sqrt_a = _matrix_sqrt(a)
        expected = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        self.assertTrue(np.allclose(sqrt_a, expected, atol=1e-4))

    def test_frechet_and_artfid(self):
        rng = np.random.default_rng(42)
        act1 = rng.normal(0, 1, size=(20, 8)).astype(np.float32)
        # Identical activations -> FID approx 0
        fid_same = compute_frechet_distance(act1, act1)
        self.assertAlmostEqual(fid_same, 0.0, places=4)

        lpips = 0.5
        artfid = compute_artfid(lpips, fid_same)
        self.assertAlmostEqual(artfid, 1.5, places=4)


if __name__ == "__main__":
    unittest.main()
