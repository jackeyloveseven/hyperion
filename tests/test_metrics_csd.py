import unittest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from hyperion.metrics.csd import preprocess_for_csd, compute_csd_similarity, load_csd_model


class DummyCSDModel(nn.Module):
    def __init__(self, fixed_vector: torch.Tensor = None):
        super().__init__()
        if fixed_vector is not None:
            self.register_buffer("fixed_vector", fixed_vector)
        else:
            self.fixed_vector = None
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, x):
        # x is [B, 3, 224, 224]
        b = x.shape[0]
        c_feat = x.mean(dim=(2, 3))
        if self.fixed_vector is not None:
            s_feat = self.fixed_vector.repeat(b, 1)
        else:
            s_feat = self.conv(x).mean(dim=(2, 3))
        return c_feat, None, s_feat


class TestMetricsCSD(unittest.TestCase):

    def test_preprocess_for_csd_pil_and_tensor(self):
        # Create a test PIL image
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:50, 20:50] = [255, 128, 64]
        pil_img = Image.fromarray(arr)

        device = "cpu"
        t_from_pil = preprocess_for_csd(pil_img, device=device)
        self.assertEqual(t_from_pil.shape, (1, 3, 224, 224))
        self.assertEqual(t_from_pil.device.type, "cpu")

        # Convert to tensor [3, 100, 100] in range [0, 1]
        t_raw = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t_from_tensor = preprocess_for_csd(t_raw, device=device)
        self.assertEqual(t_from_tensor.shape, (1, 3, 224, 224))

        # Tolerances on bilinear interpolation between PIL and PyTorch
        diff = torch.abs(t_from_pil - t_from_tensor).mean().item()
        self.assertLess(diff, 0.05)

    def test_preprocess_for_csd_invalid_type(self):
        with self.assertRaises(TypeError):
            preprocess_for_csd(12345, device="cpu")

    def test_compute_csd_similarity_identical(self):
        model = DummyCSDModel().eval()
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        sim = compute_csd_similarity(model, img, img, device="cpu")
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_compute_csd_similarity_orthogonal(self):
        class OrthogonalCSDModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                # Distinguish inputs by their average brightness
                mean_val = x.mean().item()
                if mean_val > 0:
                    feat = torch.tensor([[1.0, 0.0]])
                else:
                    feat = torch.tensor([[0.0, 1.0]])
                return None, None, feat

        model = OrthogonalCSDModel().eval()
        img1 = torch.ones((1, 3, 32, 32)) * 10.0
        img2 = torch.ones((1, 3, 32, 32)) * -10.0

        sim = compute_csd_similarity(model, img1, img2, device="cpu")
        self.assertAlmostEqual(sim, 0.0, places=4)

    def test_load_csd_model(self):
        raw_model = DummyCSDModel()
        loaded = load_csd_model(raw_model, device="cpu")
        self.assertIs(loaded, raw_model)
        self.assertFalse(loaded.training)

        # Test dictionary checkpoint
        dict_ckpt = {"model": DummyCSDModel()}
        loaded_dict = load_csd_model(dict_ckpt, device="cpu")
        self.assertFalse(loaded_dict.training)

        # Test unsupported type
        with self.assertRaises(TypeError):
            load_csd_model(["unsupported"], device="cpu")


if __name__ == "__main__":
    unittest.main()
