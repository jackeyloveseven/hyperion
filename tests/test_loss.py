import unittest
import torch
from hyperion.core.loss import HyperionLoss, sliced_wasserstein_distance


class TestLoss(unittest.TestCase):

    def test_hyperion_loss_backward(self):
        torch.manual_seed(42)
        b, h, l, d = 1, 2, 16, 16
        latents = torch.randn(b, l, d, requires_grad=True)
        content_anchor = latents.detach().clone()

        q_list = [torch.randn(b, h, l, d) for _ in range(2)]
        self_out_list = [torch.randn(b, h, l, d) for _ in range(2)]
        ref_k_list = [torch.randn(b, h, l, d) for _ in range(2)]
        ref_v_list = [torch.randn(b, h, l, d) for _ in range(2)]

        loss_fn = HyperionLoss(style_weight=2.0, content_weight=1.5, ot_layer_frac=0.5)
        loss = loss_fn(
            latents=latents,
            content_anchor=content_anchor,
            q_list=q_list,
            self_out_list=self_out_list,
            ref_k_list=ref_k_list,
            ref_v_list=ref_v_list,
            tau=0.1,
        )

        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

        loss.backward()
        self.assertIsNotNone(latents.grad)
        self.assertFalse(torch.isnan(latents.grad).any())

    def test_sliced_wasserstein(self):
        torch.manual_seed(42)
        x = torch.randn(2, 32, 16, requires_grad=True)
        y = torch.randn(2, 32, 16)

        dist = sliced_wasserstein_distance(x, y, n_projections=16)
        self.assertGreater(dist.item(), 0)
        dist.backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
