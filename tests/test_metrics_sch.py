import unittest
import numpy as np
from hyperion.metrics.sch import compute_style_content_harmonic


class TestMetricsSCH(unittest.TestCase):

    def test_sch_scalar_values(self):
        self.assertTrue(np.isclose(compute_style_content_harmonic(1.0, 1.0), 1.0))
        self.assertTrue(np.isclose(compute_style_content_harmonic(0.0, 0.8), 0.0))
        self.assertTrue(np.isclose(compute_style_content_harmonic(0.5, 0.5), 0.5))

        # TPAMI values: CSD=0.712, CLIP-I=0.660 -> SCH=0.6855
        sch = compute_style_content_harmonic(0.712, 0.660)
        self.assertTrue(np.isclose(sch, 0.6855, atol=1e-3))

    def test_sch_negative_clamping(self):
        self.assertEqual(compute_style_content_harmonic(-0.5, 0.5), 0.0)
        self.assertEqual(compute_style_content_harmonic(0.5, -0.2), 0.0)

    def test_sch_array(self):
        csd = np.array([0.712, 0.557, -0.1])
        clip = np.array([0.660, 0.689, 0.5])
        sch = compute_style_content_harmonic(csd, clip)
        self.assertEqual(len(sch), 3)
        self.assertEqual(sch[2], 0.0)


if __name__ == "__main__":
    unittest.main()
