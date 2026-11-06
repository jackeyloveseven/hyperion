import unittest
import torch
from hyperion.core.sinkhorn import (
    sinkhorn_normalize,
    sinkhorn_attn,
    sinkhorn_attn_freq,
)


class TestSinkhorn(unittest.TestCase):

    def test_sinkhorn_normalize_doubly_stochastic(self):
        torch.manual_seed(42)
        b, n, m = 2, 16, 16
        logits = torch.randn(b, n, m)
        plan = sinkhorn_normalize(logits, n_iters=10)

        self.assertEqual(plan.shape, (b, n, m))
        self.assertTrue((plan >= 0).all(), "Transport plan elements must be non-negative")

        # Row sums should equal 1.0 (exact row-stochasticity)
        row_sums = plan.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

        # Column sums should be approximately 1.0 (approximate doubly-stochasticity)
        col_sums = plan.sum(dim=-2)
        self.assertTrue(torch.allclose(col_sums, torch.ones_like(col_sums), atol=0.2))

    def test_sinkhorn_attn_shape_and_grad(self):
        torch.manual_seed(42)
        b, h, lc, ls, d = 1, 4, 32, 32, 64
        q = torch.randn(b, h, lc, d, requires_grad=True)
        k = torch.randn(b, h, ls, d)
        v = torch.randn(b, h, ls, d)

        out = sinkhorn_attn(q, k, v, tau=0.1, n_iters=5)
        self.assertEqual(out.shape, (b, h, lc, d))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertFalse(torch.isnan(q.grad).any())

    def test_sinkhorn_attn_freq(self):
        torch.manual_seed(42)
        b, h, lc, ls, d = 2, 2, 16, 16, 32
        q = torch.randn(b, h, lc, d)
        k = torch.randn(b, h, ls, d)
        v = torch.randn(b, h, ls, d)

        out = sinkhorn_attn_freq(q, k, v, tau=0.1, n_iters=6, hi_boost=2.0)
        self.assertEqual(out.shape, (b, h, lc, d))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
