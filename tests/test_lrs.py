import unittest
import torch
from hyperion.core.lrs import apply_layout_residual_suppression


class TestLRS(unittest.TestCase):

    def test_lrs_residual_suppression(self):
        torch.manual_seed(42)
        b, h, l, d = 1, 2, 20, 16
        k_list = [torch.randn(b, h, l, d) for _ in range(4)]
        v_list = [torch.randn(b, h, l, d) for _ in range(4)]

        out_k, out_v = apply_layout_residual_suppression(
            k_list, v_list, txt_len=0, gamma_shallow=0.70, gamma_deep=0.50
        )

        self.assertEqual(len(out_k), 4)
        self.assertEqual(len(out_v), 4)

        shallow_var = (out_k[0] - out_k[0].mean(dim=2, keepdim=True)).var()
        deep_var = (out_k[-1] - out_k[-1].mean(dim=2, keepdim=True)).var()

        orig_shallow_var = (k_list[0] - k_list[0].mean(dim=2, keepdim=True)).var()
        orig_deep_var = (k_list[-1] - k_list[-1].mean(dim=2, keepdim=True)).var()

        self.assertTrue(torch.isclose(shallow_var / orig_shallow_var, torch.tensor(0.70 ** 2), atol=0.05))
        self.assertTrue(torch.isclose(deep_var / orig_deep_var, torch.tensor(0.50 ** 2), atol=0.05))

    def test_lrs_text_prefix_preservation(self):
        torch.manual_seed(42)
        b, h, l_txt, l_img, d = 1, 2, 5, 15, 16
        l_tot = l_txt + l_img
        k_list = [torch.randn(b, h, l_tot, d)]
        v_list = [torch.randn(b, h, l_tot, d)]

        out_k, out_v = apply_layout_residual_suppression(
            k_list, v_list, txt_len=l_txt, gamma_shallow=0.70, gamma_deep=0.50
        )

        self.assertTrue(torch.allclose(out_k[0][:, :, :l_txt, :], k_list[0][:, :, :l_txt, :]))
        self.assertTrue(torch.allclose(out_v[0][:, :, :l_txt, :], v_list[0][:, :, :l_txt, :]))


if __name__ == "__main__":
    unittest.main()
