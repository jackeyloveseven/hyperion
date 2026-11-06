import unittest
import numpy as np
from hyperion.metrics.stm import compute_style_texture_match, compute_orientation_histogram


class TestMetricsSTM(unittest.TestCase):

    def test_stm_identical_image(self):
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 1, size=(128, 128)).astype(np.float32)

        stm = compute_style_texture_match(img, img, num_bins=36)
        self.assertTrue(np.isclose(stm, 1.0, atol=1e-3))

    def test_stm_bounded_range(self):
        rng = np.random.default_rng(123)
        img1 = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)
        img2 = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)

        stm = compute_style_texture_match(img1, img2, num_bins=36)
        self.assertGreaterEqual(stm, -1.0)
        self.assertLessEqual(stm, 1.0)

    def test_orientation_histogram_normalization(self):
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)

        hist = compute_orientation_histogram(img, num_bins=18)
        self.assertEqual(len(hist), 18)
        self.assertTrue(np.isclose(hist.sum(), 1.0, atol=1e-4))

    def test_stm_even_blur_kernel(self):
        rng = np.random.default_rng(42)
        img1 = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)
        img2 = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)

        # Even kernel size (e.g. 8 and 10) must not cause shape mismatch
        stm = compute_style_texture_match(img1, img2, blur_kernel=8)
        self.assertGreaterEqual(stm, -1.0)
        self.assertLessEqual(stm, 1.0)


if __name__ == "__main__":
    unittest.main()
