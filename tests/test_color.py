import unittest
import torch
from hyperion.core.color import feature_adain, adain_1d


class TestColor(unittest.TestCase):

    def test_feature_adain(self):
        torch.manual_seed(42)
        b, l, d = 2, 32, 16
        content = torch.randn(b, l, d) * 2.0 + 5.0
        style = torch.randn(b, l, d) * 0.5 - 2.0

        transferred = feature_adain(content, style, alpha=1.0)
        self.assertTrue(torch.allclose(transferred.mean(dim=1), style.mean(dim=1), atol=0.1))
        self.assertTrue(torch.allclose(transferred.std(dim=1), style.std(dim=1), atol=0.1))

    def test_adain_1d(self):
        torch.manual_seed(42)
        b, l, d = 1, 64, 32
        src = torch.randn(b, l, d) * 3.0 + 10.0
        tgt = torch.randn(b, l, d) * 1.0 - 5.0

        out = adain_1d(src, tgt)
        self.assertTrue(torch.allclose(out.mean(dim=1), tgt.mean(dim=1), atol=0.01))
        self.assertTrue(torch.allclose(out.std(dim=1), tgt.std(dim=1), atol=0.01))

    def test_adain_4d_latents(self):
        torch.manual_seed(42)
        b, c, h, w = 1, 16, 8, 8
        src = torch.randn(b, c, h, w) * 2.0 + 3.0
        tgt = torch.randn(b, c, h, w) * 0.5 - 1.0

        out = adain_1d(src, tgt)
        self.assertEqual(out.shape, (b, c, h, w))
        self.assertTrue(torch.allclose(out.mean(dim=(-2, -1)), tgt.mean(dim=(-2, -1)), atol=0.01))


if __name__ == "__main__":
    unittest.main()
