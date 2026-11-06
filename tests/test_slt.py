import unittest
import torch
from hyperion.core.slt import sinkhorn_latent_transport


class TestSLT(unittest.TestCase):

    def test_slt_ot_adain_shape_and_dtype(self):
        torch.manual_seed(42)
        b, l, d = 1, 64, 32
        content = torch.randn(b, l, d, dtype=torch.bfloat16)
        style_noisy = torch.randn(b, l, d, dtype=torch.bfloat16)
        style_clean = torch.randn(b, l, d, dtype=torch.bfloat16)

        out = sinkhorn_latent_transport(
            content_lat=content,
            style_lat_noisy=style_noisy,
            style_lat_clean=style_clean,
            alpha=0.10,
            n_iters=5,
            tau=0.02,
            mode="ot_adain",
        )

        self.assertEqual(out.shape, content.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())

    def test_slt_alpha_zero_identity(self):
        torch.manual_seed(42)
        b, l, d = 1, 16, 16
        content = torch.randn(b, l, d)
        style_noisy = torch.randn(b, l, d)
        style_clean = torch.randn(b, l, d)

        out = sinkhorn_latent_transport(
            content_lat=content,
            style_lat_noisy=style_noisy,
            style_lat_clean=style_clean,
            alpha=0.0,
            mode="ot_adain",
        )
        self.assertTrue(torch.allclose(out, content, atol=1e-5))

    def test_slt_blend_mode(self):
        torch.manual_seed(42)
        b, l, d = 2, 32, 16
        content = torch.randn(b, l, d)
        style_noisy = torch.randn(b, l, d)
        style_clean = torch.randn(b, l, d)

        out = sinkhorn_latent_transport(
            content_lat=content,
            style_lat_noisy=style_noisy,
            style_lat_clean=style_clean,
            alpha=0.5,
            mode="blend",
        )
        self.assertEqual(out.shape, content.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_slt_w2_and_sw_modes(self):
        torch.manual_seed(42)
        b, l, d = 1, 16, 16
        content = torch.randn(b, l, d)
        style_noisy = torch.randn(b, l, d)
        style_clean = torch.randn(b, l, d)

        out_w2 = sinkhorn_latent_transport(
            content, style_noisy, style_clean, alpha=0.1, mode="w2"
        )
        self.assertEqual(out_w2.shape, content.shape)
        self.assertFalse(torch.isnan(out_w2).any())

        out_sw = sinkhorn_latent_transport(
            content, style_noisy, style_clean, alpha=0.1, mode="sw"
        )
        self.assertEqual(out_sw.shape, content.shape)
        self.assertFalse(torch.isnan(out_sw).any())

    def test_slt_2d_input(self):
        torch.manual_seed(42)
        l, d = 16, 16
        content = torch.randn(l, d)
        style_noisy = torch.randn(l, d)
        style_clean = torch.randn(l, d)

        out = sinkhorn_latent_transport(content, style_noisy, style_clean, alpha=0.1)
        self.assertEqual(out.shape, (l, d))

    def test_compute_latent_effective_rank(self):
        from hyperion.core.slt import compute_latent_effective_rank
        torch.manual_seed(42)

        # Rank-2 latent matrix
        low_rank = torch.randn(64, 2) @ torch.randn(2, 32)
        r_low = compute_latent_effective_rank(low_rank.unsqueeze(0))
        self.assertLessEqual(r_low, 2)

        # Full-rank random latent
        high_rank = torch.randn(1, 64, 32)
        r_high = compute_latent_effective_rank(high_rank)
        self.assertGreater(r_high, 10)


if __name__ == "__main__":
    unittest.main()
