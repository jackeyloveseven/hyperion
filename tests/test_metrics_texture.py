import unittest
import numpy as np
from hyperion.metrics.texture import compute_texture_energy, compute_orientation_coherence


class TestMetricsTexture(unittest.TestCase):

    def test_texture_energy_ordering(self):
        flat = np.ones((64, 64), dtype=np.float32) * 0.5
        textured = np.random.uniform(0, 1, size=(64, 64)).astype(np.float32)

        e_flat = compute_texture_energy(flat)
        e_textured = compute_texture_energy(textured)

        self.assertLess(e_flat, 1e-4)
        self.assertGreater(e_textured, e_flat)

    def test_orientation_coherence_bounds(self):
        img = np.random.uniform(0, 1, size=(64, 64)).astype(np.float32)
        coh = compute_orientation_coherence(img, patch_size=16)
        self.assertGreaterEqual(coh, 0.0)
        self.assertLessEqual(coh, 1.0)

    def test_orientation_coherence_on_stripes(self):
        stripes = np.zeros((64, 64), dtype=np.float32)
        stripes[::4, :] = 1.0
        stripes[1::4, :] = 1.0

        coh = compute_orientation_coherence(stripes, patch_size=16)
        self.assertGreater(coh, 0.5)


if __name__ == "__main__":
    unittest.main()
